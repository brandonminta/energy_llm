"""Resume semantics (HPC robustness): run Stage 2 on 5 samples, stop after 2
(simulated kill), rerun, and assert exactly 3 more are processed while the
first 2 artifacts are untouched byte-for-byte."""

import pytest

from energy_llm.config import ExperimentConfig
from energy_llm.io_artifacts import sha256_file
from energy_llm.trajectory import run_trajectories

from .conftest import make_tiny_samples


@pytest.fixture()
def cfg(tmp_path):
    cfg = ExperimentConfig(name="resume_test", output_dir=str(tmp_path / "run"))
    cfg.calibration.beta_star = 5.0
    cfg.generation.max_new_tokens = 4
    cfg.trajectory.cache_score_tensors = False
    return cfg


class TestResume:
    def test_kill_and_rerun_is_noop_for_finished_samples(
        self, cfg, tiny_model, tiny_tokenizer, tiny_banks
    ):
        samples = make_tiny_samples(5)

        # first run "killed" after 2 samples (max_samples_per_job simulates
        # the preemption boundary: stop cleanly between samples)
        report1 = run_trajectories(
            cfg, tiny_model, tiny_tokenizer, tiny_banks, samples,
            max_samples_per_job=2,
        )
        assert report1["n_done"] == 2 and report1["n_failed"] == 0

        done = sorted(p.name for p in cfg.trajectories_dir.glob("*.json"))
        assert len(done) == 2
        hashes_before = {
            name: sha256_file(cfg.trajectories_dir / name) for name in done
        }
        npz_hashes_before = {
            p.name: sha256_file(p) for p in cfg.trajectories_dir.glob("*.npz")
        }

        # rerun: exactly the 3 remaining are processed
        report2 = run_trajectories(cfg, tiny_model, tiny_tokenizer, tiny_banks, samples)
        assert report2["n_done"] == 3
        assert report2["n_skipped"] == 2

        # finished artifacts are untouched (the rerun was a no-op for them)
        for name, h in hashes_before.items():
            assert sha256_file(cfg.trajectories_dir / name) == h
        for name, h in npz_hashes_before.items():
            assert sha256_file(cfg.trajectories_dir / name) == h

        # third run: everything skipped
        report3 = run_trajectories(cfg, tiny_model, tiny_tokenizer, tiny_banks, samples)
        assert report3["n_done"] == 0 and report3["n_skipped"] == 5

    def test_requires_calibrated_beta(self, cfg, tiny_model, tiny_tokenizer, tiny_banks):
        cfg.calibration.beta_star = None
        with pytest.raises(RuntimeError, match="beta_star is not calibrated"):
            run_trajectories(cfg, tiny_model, tiny_tokenizer, tiny_banks,
                             make_tiny_samples(1))

    def test_failure_isolation_writes_nan_row_and_failures_jsonl(
        self, cfg, tiny_model, tiny_tokenizer, tiny_banks, monkeypatch
    ):
        import numpy as np

        import energy_llm.trajectory as traj_mod
        from energy_llm.io_artifacts import load_json, read_jsonl

        samples = make_tiny_samples(3)
        original = traj_mod.capture_sample
        calls = {"n": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("simulated OOM")
            return original(*args, **kwargs)

        monkeypatch.setattr(traj_mod, "capture_sample", flaky)
        report = run_trajectories(cfg, tiny_model, tiny_tokenizer, tiny_banks, samples)
        assert report["n_done"] == 2 and report["n_failed"] == 1

        failures = read_jsonl(cfg.trajectories_dir / "failures.jsonl")
        assert len(failures) == 1
        failed_id = failures[0]["sample_id"]
        assert "simulated OOM" in failures[0]["error"]
        assert "traceback" in failures[0]

        meta = load_json(cfg.trajectories_dir / f"{failed_id}.json")
        assert meta["failed"] is True
        with np.load(cfg.trajectories_dir / f"{failed_id}.npz") as npz:
            assert np.isnan(npz["ref_lse_term"]).all()

    def test_shard_runs_populate_one_directory(
        self, cfg, tiny_model, tiny_tokenizer, tiny_banks
    ):
        samples = make_tiny_samples(5)
        for k in range(2):
            run_trajectories(cfg, tiny_model, tiny_tokenizer, tiny_banks, samples,
                             shard_index=k, num_shards=2)
        done = sorted(p.stem for p in cfg.trajectories_dir.glob("*.json"))
        assert done == sorted(s.sample_id for s in samples)
