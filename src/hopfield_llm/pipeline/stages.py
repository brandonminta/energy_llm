"""Pipeline stages: bank extraction (Stage 1) and trajectory collection (Stage 2).

Each stage persists artifacts to disk and can be run independently:
  Stage 1: build_banks     — W_down weights → .pt artifact
  Stage 2: run_trajectory  — prefill + generation passes → .npz + .json per sample
"""

from __future__ import annotations

import dataclasses
import json
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from tqdm import tqdm

from hopfield_llm.datasets.base import DataSample
from hopfield_llm.datasets.nq import NaturalQuestionsDataset
from hopfield_llm.datasets.triviaqa import TriviaQADataset
from hopfield_llm.datasets.truthfulqa import TruthfulQADataset
from hopfield_llm.hooks.capture import capture_generation, capture_prefill
from hopfield_llm.labeling.heuristic import compute_cohen_kappa, label_sample_heuristic
from hopfield_llm.labeling.llm_judge import label_sample_with_judge
from hopfield_llm.memory.banks import extract_banks
from hopfield_llm.models.loader import HFLLM
from hopfield_llm.pipeline.config import ProbeConfig
from hopfield_llm.storage.artifacts import (
    load_json_artifact,
    load_torch_artifact,
    save_json_artifact,
    save_torch_artifact,
)
from hopfield_llm.utils.logging import get_logger

if TYPE_CHECKING:
    from hopfield_llm.labeling.llm_judge import LLMJudge

log = get_logger("pipeline.stages")


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _load_m_per_layer(meta_path: Path) -> dict[int, float]:
    """Load per-layer M values from a bank metadata JSON file."""
    m: dict[int, float] = {}
    if meta_path.exists():
        raw_meta = load_json_artifact(meta_path)
        for k, v in raw_meta.items():
            if isinstance(v, dict) and "error" not in v and "M" in v:
                m[int(k)] = float(v["M"])
        log.info("Loaded M_ℓ for %d layers from %s", len(m), meta_path)
    else:
        log.warning("banks_metadata not found at %s; M_squared_term will be 0", meta_path)
    return m


# ------------------------------------------------------------------
# Stage 1 — Bank extraction
# ------------------------------------------------------------------


def build_banks(
    model: str,
    output: str | Path,
    bank: str = "down",
    device: str | None = None,
    load_in_4bit: bool = True,
    primary_probe: ProbeConfig | None = None,
    secondary_probe: ProbeConfig | None = None,
) -> Path:
    """Extract per-layer memory banks and save as .pt artifact(s).

    Legacy mode (primary_probe=None, secondary_probe=None):
        Extracts a single bank using the ``bank`` argument and saves to
        ``output`` with a companion ``banks_metadata.json`` in the same
        directory.  Filename is whatever ``output`` specifies.

    Probe mode (at least one ProbeConfig supplied):
        For each non-None probe, extracts banks with probe.bank /
        probe.normalize and saves to:
            {output_dir}/banks_{probe.label}.pt
            {output_dir}/banks_{probe.label}_metadata.json
        where output_dir = Path(output).parent.

    Args:
        model:           Model alias or HuggingFace model ID.
        output:          Output path (.pt file).  In probe mode its parent
                         directory is used for all labeled bank files.
        bank:            Bank to use in legacy mode (default: 'down').
        device:          Target device (None = auto).
        load_in_4bit:    Use 4-bit NF4 quantization when available.
        primary_probe:   ProbeConfig for the primary (key-space) probe.
        secondary_probe: ProbeConfig for the secondary (value-space) probe.

    Returns:
        Path to the saved artifact (primary in probe mode, legacy path otherwise).
    """
    llm = HFLLM(
        model_id=model, device=device,
        load_in_4bit=load_in_4bit, output_hidden_states=False,
    )

    if primary_probe is None and secondary_probe is None:
        # ── Legacy path ─────────────────────────────────────────────
        banks_data, metadata = extract_banks(
            llm, bank=bank, normalize=False, strict=True, return_metadata=True
        )
        artifact = {
            "artifact_type": "banks",
            "model": llm.summary(),
            "bank": bank,
            "banks": banks_data,
        }
        path = save_torch_artifact(artifact, output)
        meta_path = Path(path).with_name("banks_metadata.json")
        save_json_artifact({str(k): v for k, v in metadata.items()}, meta_path)
        log.info(
            "build_banks: saved %d layers → %s, metadata → %s (model=%s, bank=%s)",
            len(banks_data), path, meta_path, llm.alias, bank,
        )
        return path

    # ── Probe-based path ─────────────────────────────────────────────
    output_dir = Path(output).parent
    first_path: Path | None = None

    for probe in [p for p in (primary_probe, secondary_probe) if p is not None]:
        banks_data, metadata = extract_banks(
            llm, bank=probe.bank, normalize=probe.normalize,
            strict=True, return_metadata=True,
        )
        artifact = {
            "artifact_type": "banks",
            "model": llm.summary(),
            "bank": probe.bank,
            "banks": banks_data,
        }
        probe_path = output_dir / f"banks_{probe.label}.pt"
        save_torch_artifact(artifact, probe_path)
        meta_path = output_dir / f"banks_{probe.label}_metadata.json"
        save_json_artifact({str(k): v for k, v in metadata.items()}, meta_path)
        log.info(
            "build_banks: saved %d layers → %s (model=%s, probe=%s, bank=%s)",
            len(banks_data), probe_path, llm.alias, probe.label, probe.bank,
        )
        if first_path is None:
            first_path = probe_path

    return first_path  # type: ignore[return-value]  # always set (≥1 probe is non-None)


# ------------------------------------------------------------------
# Stage 2 — Trajectory collection
# ------------------------------------------------------------------


def run_trajectory(
    model: str,
    dataset: str,
    banks: str | Path,
    output: str | Path,
    beta: float = 15.0,
    threshold: float = 0.1,
    energy_mode: str = "dot",
    max_samples: int | None = None,
    seed: int = 42,
    max_new_tokens: int = 50,
    diagnostic_subset: int = 20,
    h_pre_mode: str = "diagnostic",
    shard_id: int = 0,
    num_shards: int = 1,
    device: str | None = None,
    load_in_4bit: bool = True,
    primary_probe: ProbeConfig | None = None,
    secondary_probe: ProbeConfig | None = None,
    scores_subset_size: int = 200,
) -> Path:
    """Run prefill + generation for each sample; persist .npz and .json artifacts.

    Legacy mode (primary_probe=None, secondary_probe=None):
        Single probe pass.  Per-sample artifacts:
            {sample_id}.npz  — compressed metric arrays
            {sample_id}.json — metadata

    Probe mode (primary_probe is not None):
        For the primary probe, artifacts are saved as:
            {sample_id}_{primary_probe.label}.npz / .json
        For the secondary probe (if set), a teacher-forcing pass re-uses the
        already-generated text without calling model.generate() again:
            {sample_id}_{secondary_probe.label}.npz / .json
        Both JSON files contain the identical generated_text field.

    Teacher-forcing secondary pass:
        Tokenises question + generated_text and runs capture_prefill on the
        full string with the secondary probe's bank/hook settings.  The result
        is split at T_prompt (= len(tokenise(question))) to yield prefill and
        generation-equivalent metric arrays.

    h_pre saving (raw prefill query vectors [L, d_m] per sample):
        h_pre_mode='off'        — never save h_pre
        h_pre_mode='diagnostic' — save for the first ``diagnostic_subset``
                                  shard-local samples (default)
        h_pre_mode='all'        — save h_pre for every sample

    Args:
        model:            Model alias or HuggingFace model ID.
        dataset:          Dataset name: 'triviaqa', 'nq', or 'truthfulqa'.
        banks:            Path to the .pt artifact from build_banks.
                          In probe mode this is used only to derive the
                          parent directory for labeled bank files.
        output:           Directory for output artifacts.
        beta:             Hopfield inverse temperature.
        threshold:        Active-neuron count threshold.
        energy_mode:      'dot' (default) or 'cosine'.
        max_samples:      Cap on dataset size before sharding (None = all).
        seed:             RNG seed for deterministic subsampling.
        max_new_tokens:   Maximum tokens to generate per sample.
        diagnostic_subset: Samples to save h_pre for when h_pre_mode='diagnostic'.
        h_pre_mode:       h_pre saving policy: 'off', 'diagnostic', or 'all'.
        shard_id:         0-indexed shard index for this worker.
        num_shards:       Total shards; 1 = no sharding.
        device:           Target device (None = auto).
        load_in_4bit:     Use 4-bit NF4 quantization when available.
        primary_probe:    ProbeConfig for the primary capture pass.
        secondary_probe:  ProbeConfig for the teacher-forcing secondary pass.

    Returns:
        Path to the output directory.
    """
    if h_pre_mode not in {"off", "diagnostic", "all"}:
        raise ValueError(f"h_pre_mode must be 'off', 'diagnostic', or 'all'; got {h_pre_mode!r}")

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Bank loading ────────────────────────────────────────────────────────
    _probe_mode = primary_probe is not None
    _banks_dir = Path(banks).parent

    if _probe_mode:
        _prim_path = _banks_dir / f"banks_{primary_probe.label}.pt"
        _prim_art = load_torch_artifact(_prim_path)
        prim_banks: dict[int, torch.Tensor] = _prim_art["banks"]
        prim_m = _load_m_per_layer(_banks_dir / f"banks_{primary_probe.label}_metadata.json")
        model_alias = _prim_art["model"].get("alias") or model
        prim_label = primary_probe.label

        sec_banks: dict[int, torch.Tensor] | None = None
        sec_m: dict[int, float] = {}
        sec_label = ""
        if secondary_probe is not None:
            _sec_path = _banks_dir / f"banks_{secondary_probe.label}.pt"
            _sec_art = load_torch_artifact(_sec_path)
            sec_banks = _sec_art["banks"]
            sec_m = _load_m_per_layer(_banks_dir / f"banks_{secondary_probe.label}_metadata.json")
            sec_label = secondary_probe.label
    else:
        # Legacy: load single bank from the given path
        _art = load_torch_artifact(banks)
        prim_banks = _art["banks"]
        model_alias = _art["model"].get("alias") or model
        prim_m = _load_m_per_layer(Path(banks).with_name("banks_metadata.json"))
        prim_label = ""
        sec_banks = None
        sec_m = {}
        sec_label = ""

    # ── Dataset loading ─────────────────────────────────────────────────────
    dataset_name = dataset.lower()
    if dataset_name == "triviaqa":
        ds = TriviaQADataset(max_samples=max_samples, seed=seed)
    elif dataset_name == "nq":
        ds = NaturalQuestionsDataset(max_samples=max_samples, seed=seed)
    elif dataset_name == "truthfulqa":
        ds = TruthfulQADataset(max_samples=max_samples, seed=seed)
    else:
        raise ValueError(
            f"Unknown dataset: '{dataset}'. Use 'triviaqa', 'nq', or 'truthfulqa'."
        )

    samples = list(ds)
    if num_shards > 1:
        samples = samples[shard_id::num_shards]

    log.info(
        "run_trajectory: %d samples (shard %d/%d), model=%s, dataset=%s",
        len(samples), shard_id, num_shards, model_alias, dataset_name,
    )

    # ── Hook settings ───────────────────────────────────────────────────────
    if _probe_mode:
        prim_hook_target = primary_probe.hook_target
        prim_nrm_query = primary_probe.normalize_query
    else:
        prim_hook_target = "mlp_input"
        prim_nrm_query = True

    llm = HFLLM(
        model_id=model, device=device,
        load_in_4bit=load_in_4bit, output_hidden_states=False,
    )
    L = llm.n_layers

    if scores_subset_size == -1:
        log.warning(
            "scores_subset_size=-1: saving raw scores for all samples — expect large disk usage."
        )

    n_ok = n_err = 0
    bar = tqdm(
        samples,
        desc=f"trajectory[{shard_id}/{num_shards}]",
        unit="sample",
        dynamic_ncols=True,
    )

    for i, sample in enumerate(bar):
        if h_pre_mode == "off":
            save_hpre = False
        elif h_pre_mode == "diagnostic":
            save_hpre = i < diagnostic_subset
        else:  # "all"
            save_hpre = True

        save_scores_this_sample = (scores_subset_size == -1 or i < scores_subset_size)
        if i == scores_subset_size:
            log.info(
                "scores_subset_size=%d reached; raw scores will no longer be saved.",
                scores_subset_size,
            )

        try:
            # ── Primary probe: prefill ──────────────────────────────────────
            prefill_out = capture_prefill(
                llm, sample.question, prim_banks,
                beta=beta, threshold=threshold, energy_mode=energy_mode,
                return_x=save_hpre, m_per_layer=prim_m,
                hook_target=prim_hook_target, normalize_query=prim_nrm_query,
                save_scores=save_scores_this_sample,
            )
            if save_hpre:
                prefill_result, h_pre_raw = prefill_out
            else:
                prefill_result = prefill_out
                h_pre_raw = None

            # ── Primary probe: generation ───────────────────────────────────
            gen_result = capture_generation(
                llm, sample.question, prim_banks,
                beta=beta, threshold=threshold, energy_mode=energy_mode,
                max_new_tokens=max_new_tokens, m_per_layer=prim_m,
                hook_target=prim_hook_target, normalize_query=prim_nrm_query,
                save_scores=save_scores_this_sample,
            )
            gen_cfg = gen_result.generation_config
            sample = dataclasses.replace(sample, generation_config=gen_cfg)

            # ── Determine artifact filenames ────────────────────────────────
            if _probe_mode:
                npz_stem = f"{sample.id}_{prim_label}"
            else:
                npz_stem = sample.id

            # ── Save primary probe artifacts ────────────────────────────────
            np.savez_compressed(
                out_dir / f"{npz_stem}.npz",
                prefill_energy=prefill_result.energy_last,
                prefill_energy_mean=prefill_result.energy_mean,
                prefill_entropy=prefill_result.entropy[:, -1],
                prefill_norm_entropy=prefill_result.norm_entropy[:, -1],
                prefill_lse=prefill_result.lse[:, -1],
                prefill_quadratic=prefill_result.quadratic[:, -1],
                prefill_top_act=prefill_result.top_act[:, -1],
                prefill_n_active=prefill_result.n_active[:, -1],
                gen_energy=gen_result.energy,
                gen_entropy=gen_result.entropy,
                gen_norm_entropy=gen_result.norm_entropy,
                gen_lse=gen_result.lse,
                gen_quadratic=gen_result.quadratic,
                gen_top_act=gen_result.top_act,
                gen_n_active=gen_result.n_active,
                token_ids=gen_result.token_ids,
                generation_config_json=np.array(json.dumps(gen_cfg)),
            )

            _pref_scores = prefill_result.scores
            _gen_scores = gen_result.scores
            if _pref_scores is not None or _gen_scores is not None:
                _scores_kw: dict = {}
                if _pref_scores is not None:
                    _scores_kw["prefill_scores"] = _pref_scores
                if _gen_scores is not None:
                    _scores_kw["gen_scores"] = _gen_scores
                np.savez_compressed(out_dir / f"{npz_stem}_scores.npz", **_scores_kw)

            meta = {
                "id": sample.id,
                "source": sample.source,
                "question": sample.question,
                "gold_answers": sample.gold_answers,
                "generated_text": gen_result.generated_text,
                "generation_config": gen_cfg,
                "model": model_alias,
                "n_layers": L,
                "n_tokens_generated": int(len(gen_result.token_ids)),
            }
            with open(out_dir / f"{npz_stem}.json", "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2, ensure_ascii=False)

            if save_hpre and h_pre_raw is not None:
                np.savez_compressed(out_dir / f"{npz_stem}_hpre.npz", h_pre=h_pre_raw)

            # ── Secondary probe: teacher-forcing pass ───────────────────────
            if _probe_mode and secondary_probe is not None and sec_banks is not None:
                # T_prompt = number of tokens in the question alone
                _tok = llm.tokenizer(
                    sample.question, return_tensors="pt", add_special_tokens=False
                )
                T_prompt = _tok["input_ids"].shape[1]

                full_text = sample.question + gen_result.generated_text
                sec_full = capture_prefill(
                    llm, full_text, sec_banks,
                    beta=beta, threshold=threshold, energy_mode=energy_mode,
                    hook_target=secondary_probe.hook_target,
                    normalize_query=secondary_probe.normalize_query,
                    m_per_layer=sec_m,
                )

                T_total = sec_full.energy.shape[1]
                T_p = min(T_prompt, T_total)
                pref_idx = max(0, T_p - 1)
                pref_slice = slice(0, max(T_p, 1))

                np.savez_compressed(
                    out_dir / f"{sample.id}_{sec_label}.npz",
                    prefill_energy=sec_full.energy[:, pref_idx],
                    prefill_energy_mean=np.nanmean(sec_full.energy[:, pref_slice], axis=1),
                    prefill_entropy=sec_full.entropy[:, pref_idx],
                    prefill_norm_entropy=sec_full.norm_entropy[:, pref_idx],
                    prefill_lse=sec_full.lse[:, pref_idx],
                    prefill_quadratic=sec_full.quadratic[:, pref_idx],
                    prefill_top_act=sec_full.top_act[:, pref_idx],
                    prefill_n_active=sec_full.n_active[:, pref_idx],
                    gen_energy=sec_full.energy[:, T_p:],
                    gen_entropy=sec_full.entropy[:, T_p:],
                    gen_norm_entropy=sec_full.norm_entropy[:, T_p:],
                    gen_lse=sec_full.lse[:, T_p:],
                    gen_quadratic=sec_full.quadratic[:, T_p:],
                    gen_top_act=sec_full.top_act[:, T_p:],
                    gen_n_active=sec_full.n_active[:, T_p:],
                    token_ids=gen_result.token_ids,
                    generation_config_json=np.array(json.dumps(gen_cfg)),
                )
                sec_meta = {
                    "id": sample.id,
                    "source": sample.source,
                    "question": sample.question,
                    "gold_answers": sample.gold_answers,
                    "generated_text": gen_result.generated_text,
                    "generation_config": gen_cfg,
                    "model": model_alias,
                    "n_layers": L,
                    "n_tokens_generated": int(len(gen_result.token_ids)),
                }
                with open(out_dir / f"{sample.id}_{sec_label}.json", "w", encoding="utf-8") as fh:
                    json.dump(sec_meta, fh, indent=2, ensure_ascii=False)

            n_ok += 1
            bar.set_postfix(ok=n_ok, err=n_err, refresh=False)

        except Exception as exc:
            n_err += 1
            bar.set_postfix(ok=n_ok, err=n_err, refresh=False)
            tb = traceback.extract_tb(exc.__traceback__)
            origin = tb[-1] if tb else None
            loc = f"{origin.filename.split('/')[-1]}:{origin.lineno}" if origin else "?"
            log.warning("SKIP %s — %s: %s  (at %s)", sample.id, type(exc).__name__, exc, loc)

    bar.close()
    log.info("run_trajectory: finished — %d ok, %d errors → %s", n_ok, n_err, out_dir)
    return out_dir


# ------------------------------------------------------------------
# Stage 3 — Trajectory labeling
# ------------------------------------------------------------------


def label_trajectory(
    traj_dir: str | Path,
    output: str | Path,
    judge: "LLMJudge",
    heuristic_threshold: float = 0.3,
    calibration_subset_size: int = 50,
) -> Path:
    """Label each trajectory JSON with heuristic + LLM judge; save calibration report.

    For each ``{id}.json`` in ``traj_dir``:
      1. Apply SQuAD-F1 heuristic → stored in ``metadata["heuristic_is_hallucination"]``
         only; ``is_hallucination`` is NOT set by the heuristic.
      2. Apply LLM judge → sets authoritative ``is_hallucination``.
      3. Save ``{id}_labeled.json`` to ``output``.

    After labeling all samples, compute Cohen's kappa between the heuristic and
    judge labels (only for samples where the judge succeeded).  Save
    ``calibration.json`` with: ``n_paired_samples``, ``cohen_kappa``,
    ``heuristic_positive_rate``, ``judge_positive_rate``, ``agreement_rate``,
    ``warning`` (non-null when kappa < 0.5 or too few paired samples).

    Args:
        traj_dir:               Directory containing ``{id}.json`` trajectory files.
        output:                 Directory for labeled JSONs and calibration report.
        judge:                  LLMJudge instance for authoritative labeling.
        heuristic_threshold:    F1 threshold; F1 < threshold → hallucination.
        calibration_subset_size: Minimum paired samples for reliable kappa.

    Returns:
        Path to the output directory.
    """
    traj_dir = Path(traj_dir)
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(
        p for p in traj_dir.glob("*.json")
        if p.name != "calibration.json" and not p.name.endswith("_labeled.json")
    )

    heuristic_labels: list[bool] = []
    judge_labels: list[bool] = []

    for json_path in json_paths:
        raw = json.loads(json_path.read_text(encoding="utf-8"))

        sample = DataSample(
            id=raw["id"],
            source=raw.get("source", ""),
            question=raw.get("question", ""),
            gold_answers=raw.get("gold_answers", []),
            generated_text=raw.get("generated_text", ""),
            generation_config=raw.get("generation_config", {}),
            metadata=raw.get("metadata", {}),
        )

        heuristic_result = label_sample_heuristic(sample, threshold=heuristic_threshold)
        sample = dataclasses.replace(
            sample,
            metadata={
                **sample.metadata,
                "heuristic_is_hallucination": heuristic_result.is_hallucination,
            },
            correctness_score=heuristic_result.correctness_score,
        )

        sample = label_sample_with_judge(sample, judge)

        labeled: dict = {
            "id": sample.id,
            "source": sample.source,
            "question": sample.question,
            "gold_answers": sample.gold_answers,
            "generated_text": sample.generated_text,
            "generation_config": sample.generation_config,
            "metadata": sample.metadata,
            "correctness_score": sample.correctness_score,
            "is_hallucination": sample.is_hallucination,
            "label_source": sample.label_source,
            "label_version": sample.label_version,
            "judge_raw_output": sample.judge_raw_output,
            "judge_model": sample.judge_model,
            "judge_prompt_hash": sample.judge_prompt_hash,
        }
        stem = json_path.stem
        with open(out_dir / f"{stem}_labeled.json", "w", encoding="utf-8") as fh:
            json.dump(labeled, fh, indent=2, ensure_ascii=False)

        if sample.is_hallucination is not None:
            h_label = sample.metadata.get("heuristic_is_hallucination")
            if h_label is not None:
                heuristic_labels.append(bool(h_label))
                judge_labels.append(bool(sample.is_hallucination))

    n_paired = len(heuristic_labels)
    if n_paired == 0:
        kappa = 0.0
        h_pos_rate = 0.0
        j_pos_rate = 0.0
        agreement_rate = 0.0
        warning: str | None = "No paired samples available for kappa computation."
    else:
        kappa = compute_cohen_kappa(heuristic_labels, judge_labels)
        h_pos_rate = sum(heuristic_labels) / n_paired
        j_pos_rate = sum(judge_labels) / n_paired
        agreement_rate = (
            sum(int(h) == int(j) for h, j in zip(heuristic_labels, judge_labels)) / n_paired
        )
        warning = None
        if n_paired < calibration_subset_size:
            warning = (
                f"Only {n_paired} paired samples "
                f"(< {calibration_subset_size} required for reliable calibration)."
            )
        if kappa < 0.5:
            kappa_warn = f"Low inter-annotator agreement: kappa={kappa:.3f} < 0.5."
            warning = kappa_warn if warning is None else f"{warning} {kappa_warn}"

    calibration = {
        "n_paired_samples": n_paired,
        "cohen_kappa": round(kappa, 6),
        "heuristic_positive_rate": round(h_pos_rate, 6),
        "judge_positive_rate": round(j_pos_rate, 6),
        "agreement_rate": round(agreement_rate, 6),
        "warning": warning,
    }
    with open(out_dir / "calibration.json", "w", encoding="utf-8") as fh:
        json.dump(calibration, fh, indent=2)

    log.info(
        "label_trajectory: %d samples labeled, kappa=%.3f → %s",
        n_paired, kappa, out_dir / "calibration.json",
    )
    if kappa < 0.5 and n_paired > 0:
        log.warning("Low kappa=%.3f between heuristic and judge labels.", kappa)

    return out_dir
