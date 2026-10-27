"""End-to-end integration on a tiny random Qwen2-style model: banks ->
calibration -> trajectories -> mocked judge -> features -> probe -> stats,
all on CPU, plus the beta-recompute path."""

import numpy as np
import pytest

from energy_llm.banks import build_banks, load_banks
from energy_llm.calibration import calibrate_beta
from energy_llm.config import ExperimentConfig, load_config
from energy_llm.features import (
    baseline_matrix,
    build_feature_table,
    build_feature_table_at_beta,
    infer_n_layers,
    phi_matrix,
)
from energy_llm.labeling import label_samples, load_labels
from energy_llm.probe import make_locked_splits, run_base_and_full_probes
from energy_llm.stats import (
    bootstrap_auroc_ci,
    paired_bootstrap_delta_auroc,
    per_category_auroc,
    permutation_scan_threshold,
)
from energy_llm.trajectory import run_trajectories

from .conftest import make_tiny_samples
from .test_labeling import FakeJudge

N_SAMPLES = 40


@pytest.fixture(scope="module")
def e2e_run(tmp_path_factory):
    """Run the full pipeline once; individual tests assert on the artifacts."""
    from .conftest import make_tiny_model, make_tiny_tokenizer

    tmp = tmp_path_factory.mktemp("e2e")
    tokenizer = make_tiny_tokenizer()
    model = make_tiny_model(vocab_size=tokenizer.vocab_size + 4)
    samples = make_tiny_samples(N_SAMPLES)

    cfg = ExperimentConfig(name="e2e", output_dir=str(tmp / "run"))
    cfg.calibration.num_samples = 5
    cfg.generation.max_new_tokens = 4
    cfg.trajectory.cache_score_tensors = True
    cfg.stats.n_bootstrap = 100
    cfg.stats.n_permutation = 50

    # Stage 1 — banks
    banks = build_banks(model, model_id="tiny-qwen2", bank_id="gate",
                        output_path=cfg.banks_path)
    banks = load_banks(cfg.banks_path)

    # Calibration — beta* into the snapshot
    cal = calibrate_beta(cfg, model, tokenizer, banks, samples)
    yaml_path = tmp / "e2e.yaml"
    yaml_path.write_text(f"name: e2e\noutput_dir: {cfg.output_dir}\n")
    cfg = load_config(yaml_path)  # snapshot wins, carries beta*

    # Stage 2 — trajectories (two shards into one directory)
    for k in range(2):
        run_trajectories(cfg, model, tokenizer, banks, samples,
                         shard_index=k, num_shards=2)

    # Stage 3 — mocked judge
    label_samples(cfg.trajectories_dir, cfg.labels_dir, FakeJudge())
    labels = load_labels(cfg.labels_dir)

    # Stage 4 — features + probe + stats
    df = build_feature_table(cfg.trajectories_dir, labels=labels)
    df = df[df["is_hallucination"].notna()].reset_index(drop=True)
    n_layers = infer_n_layers(df)
    y = df["is_hallucination"].to_numpy(dtype=int)
    splits = make_locked_splits(
        df["sample_id"].tolist(), y, cfg.out / "split.json", seed=cfg.seed,
        force_train_ids=cal["calibration_sample_ids"],
    )
    idx = splits.index_of(df["sample_id"].tolist())
    probes = run_base_and_full_probes(
        baseline_matrix(df), phi_matrix(df, n_layers), y, idx,
        c_grid=[0.1, 1.0], inner_cv_folds=2, outer_cv_folds=2, seed=cfg.seed,
    )
    return {
        "cfg": cfg, "cal": cal, "df": df, "n_layers": n_layers, "y": y,
        "splits": splits, "idx": idx, "probes": probes, "labels": labels,
    }


class TestPipelineArtifacts:
    def test_all_samples_captured(self, e2e_run):
        cfg = e2e_run["cfg"]
        assert len(list(cfg.trajectories_dir.glob("*.json"))) == N_SAMPLES
        assert len(list(cfg.trajectories_dir.glob("*_retrieval_scores.npz"))) == N_SAMPLES

    def test_beta_star_flows_through_snapshot(self, e2e_run):
        assert e2e_run["cfg"].calibration.beta_star == pytest.approx(
            e2e_run["cal"]["beta_star"]
        )

    def test_feature_table_shape(self, e2e_run):
        df, n_layers = e2e_run["df"], e2e_run["n_layers"]
        assert n_layers == 2
        phi = phi_matrix(df, n_layers)
        assert phi.shape == (len(df), 4 * n_layers)
        assert np.isfinite(phi).all()

    def test_calibration_ids_in_train_split(self, e2e_run):
        cal_ids = set(e2e_run["cal"]["calibration_sample_ids"])
        assert cal_ids <= set(e2e_run["splits"].train_ids)


class TestStatsRun:
    def test_probes_and_bootstrap(self, e2e_run):
        probes = e2e_run["probes"]
        y_test = probes["full"]["y_test"]
        ci = bootstrap_auroc_ci(y_test, probes["full"]["test_scores"],
                                n_bootstrap=100, seed=42)
        assert 0.0 <= ci["auroc"] <= 1.0
        delta = paired_bootstrap_delta_auroc(
            y_test, probes["full"]["test_scores"], probes["base"]["test_scores"],
            n_bootstrap=100, seed=42,
        )
        assert "ci_0.950" in delta and "ci_0.975" in delta

    def test_permutation_scan(self, e2e_run):
        df, n_layers, y = e2e_run["df"], e2e_run["n_layers"], e2e_run["y"]
        out = permutation_scan_threshold(phi_matrix(df, n_layers), y,
                                         n_permutation=50, seed=42)
        assert out["observed_folded_auroc"].shape == (4 * n_layers,)
        assert 0.5 < out["scan_threshold"] <= 1.0

    def test_per_category(self, e2e_run):
        df, y = e2e_run["df"], e2e_run["y"]
        scores = df["seq_logprob"].to_numpy()
        rows = per_category_auroc(df["category"].tolist(), y, scores, min_n=5)
        assert all(0.0 <= r["auroc"] <= 1.0 for r in rows)


class TestBetaRecompute:
    def test_features_at_beta_star_match_online(self, e2e_run):
        """Recomputing the feature table from cached scores at beta* must
        reproduce the online features up to fp16 storage error."""
        cfg, df = e2e_run["cfg"], e2e_run["df"]
        beta_star = cfg.calibration.beta_star
        df_re = build_feature_table_at_beta(cfg.trajectories_dir, beta_star,
                                            labels=e2e_run["labels"])
        df_re = df_re[df_re["sample_id"].isin(df["sample_id"])].reset_index(drop=True)
        n_layers = e2e_run["n_layers"]
        a = phi_matrix(df.sort_values("sample_id").reset_index(drop=True), n_layers)
        b = phi_matrix(df_re.sort_values("sample_id").reset_index(drop=True), n_layers)
        np.testing.assert_allclose(a, b, rtol=0.05, atol=0.02)

    def test_features_change_with_beta(self, e2e_run):
        cfg = e2e_run["cfg"]
        df_lo = build_feature_table_at_beta(cfg.trajectories_dir, 1.0)
        df_hi = build_feature_table_at_beta(cfg.trajectories_dir, 50.0)
        n_layers = e2e_run["n_layers"]
        assert not np.allclose(phi_matrix(df_lo, n_layers), phi_matrix(df_hi, n_layers))
