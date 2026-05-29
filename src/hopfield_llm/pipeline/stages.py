"""Pipeline stages: bank extraction (Stage 1) and trajectory collection (Stage 2).

Each stage persists artifacts to disk and can be run independently:
  Stage 1: build_banks     — W_down weights → .pt artifact
  Stage 2: run_trajectory  — prefill + generation passes → .npz + .json per sample
"""

from __future__ import annotations

import dataclasses
import gc
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


def _topk_compress(
    scores: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compress a [..., K] scores tensor into top-k values + indices + tail-max.

    Approximates the softmax over the K axis when ``β·tail_max < β·top1`` —
    typically true for any β ≥ 1 with a well-trained model.  The tail max
    lets downstream code reconstruct ``log(Σ_{i∉topk} exp(β·s_i)) ≈ log(K-k)
    + β·tail_max``, which suffices for the lse and entropy fallback.

    Returns:
        (topk_values [..., k] float16,
         topk_indices [..., k] int32,
         tail_max [...] float16)
    """
    sc = scores.astype(np.float32)
    K = sc.shape[-1]
    k = min(int(k), K)
    # np.argpartition: O(K) per row; sort only the top-k slice afterwards.
    part = np.argpartition(sc, K - k, axis=-1)
    top_idx_unsorted = part[..., K - k:]  # [..., k]
    top_vals = np.take_along_axis(sc, top_idx_unsorted, axis=-1)
    order = np.argsort(-top_vals, axis=-1)
    top_vals = np.take_along_axis(top_vals, order, axis=-1)
    top_idx  = np.take_along_axis(top_idx_unsorted, order, axis=-1)
    if k < K:
        tail_idx = part[..., : K - k]
        tail_vals = np.take_along_axis(sc, tail_idx, axis=-1)
        tail_max = tail_vals.max(axis=-1)
    else:
        tail_max = np.full(sc.shape[:-1], -np.inf, dtype=np.float32)
    return (
        top_vals.astype(np.float16),
        top_idx.astype(np.int32),
        tail_max.astype(np.float16),
    )


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
    bank: str = "up",
    device: str | None = None,
    load_in_4bit: bool = True,
    normalize: bool = False,
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
            llm, bank=bank, normalize=normalize, strict=True, return_metadata=True
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
    normalize_query: bool = True,
    hook_target: str = "mlp_input",
    primary_probe: ProbeConfig | None = None,
    secondary_probe: ProbeConfig | None = None,
    scores_subset_size: int = 50,
    score_capture: str = "full",
    score_topk: int = 256,
    enable_logit_lens: bool = False,
    enable_null_control: bool = False,
    calibrate_beta: bool = False,
    calibration_samples: int = 20,
    beta_target: float = 0.55,
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
        Tuple of (output_directory, calibrated_beta). calibrated_beta is None
        when calibrate_beta=False or calibration was not requested.
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
        prim_hook_target = hook_target
        prim_nrm_query = normalize_query

    llm = HFLLM(
        model_id=model, device=device,
        load_in_4bit=load_in_4bit, output_hidden_states=False,
    )
    L = llm.n_layers

    # Move banks to GPU so compute_energy matmul stays on device (~10× speedup).
    prim_banks = {k: v.to(llm.device) for k, v in prim_banks.items()}
    if sec_banks is not None:
        sec_banks = {k: v.to(llm.device) for k, v in sec_banks.items()}

    if scores_subset_size == -1:
        log.warning(
            "scores_subset_size=-1: saving raw scores for all samples — expect large disk usage."
        )

    calibrated_beta: float | None = None
    if calibrate_beta:
        from hopfield_llm.metrics.calibration import calibrate_beta as _cal_beta
        _cal_questions = [s.question for s in samples[:calibration_samples]]
        beta, _cal_info = _cal_beta(
            llm, prim_banks, _cal_questions,
            target_fraction=beta_target,
            hook_target=prim_hook_target,
            normalize_query=prim_nrm_query,
        )
        calibrated_beta = beta
        save_json_artifact(_cal_info, out_dir / "calibration_beta.json")
        log.info("calibrate_beta: using β=%.2f for this run (saved calibration_beta.json)", beta)

    # Detect already-completed samples so a resumed run can skip them.
    if _probe_mode:
        _done_suffix = f"_{prim_label}.json"
        done_ids: set[str] = {
            p.name[: -len(_done_suffix)]
            for p in out_dir.glob(f"*{_done_suffix}")
        }
    else:
        done_ids = {p.stem for p in out_dir.glob("*.json")}
    n_skipped = len(done_ids)
    if n_skipped:
        log.info("run_trajectory: resuming — skipping %d already-processed samples", n_skipped)

    n_ok = n_err = 0
    bar = tqdm(
        samples,
        desc=f"trajectory[{shard_id}/{num_shards}]",
        unit="sample",
        dynamic_ncols=True,
    )

    for i, sample in enumerate(bar):
        # Resume: skip samples whose JSON artifact already exists on disk.
        if _probe_mode:
            _done_path = out_dir / f"{sample.id}_{prim_label}.json"
        else:
            _done_path = out_dir / f"{sample.id}.json"
        if _done_path.exists():
            bar.set_postfix(ok=n_ok, skip=n_skipped, err=n_err, refresh=False)
            continue

        if h_pre_mode == "off":
            save_hpre = False
        elif h_pre_mode == "diagnostic":
            save_hpre = i < diagnostic_subset
        else:  # "all"
            save_hpre = True

        if score_capture == "off" or scores_subset_size == 0:
            save_scores_this_sample = False
        elif scores_subset_size == -1:
            save_scores_this_sample = True
        else:
            save_scores_this_sample = i < scores_subset_size
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
                compute_p_ref=True,
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
                p_ref=prefill_result.p_ref,
            )
            gen_cfg = gen_result.generation_config
            sample = dataclasses.replace(sample, generation_config=gen_cfg)

            # ── Determine artifact filenames ────────────────────────────────
            if _probe_mode:
                npz_stem = f"{sample.id}_{prim_label}"
            else:
                npz_stem = sample.id

            # ── Save primary probe artifacts ────────────────────────────────
            _npz_kw = dict(
                prefill_energy=prefill_result.energy_last,
                prefill_energy_mean=prefill_result.energy_mean,
                prefill_entropy=prefill_result.entropy[:, -1],
                prefill_entropy_mean=np.nanmean(prefill_result.entropy, axis=1),
                prefill_norm_entropy=prefill_result.norm_entropy[:, -1],
                prefill_norm_entropy_mean=np.nanmean(prefill_result.norm_entropy, axis=1),
                prefill_lse=prefill_result.lse[:, -1],
                prefill_lse_mean=np.nanmean(prefill_result.lse, axis=1),
                prefill_quadratic=prefill_result.quadratic[:, -1],
                prefill_top_act=prefill_result.top_act[:, -1],
                prefill_top_act_mean=np.nanmean(prefill_result.top_act, axis=1),
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
            if gen_result.js_per_layer is not None:
                _npz_kw["gen_js"] = gen_result.js_per_layer
            if gen_result.hellinger_per_layer is not None:
                _npz_kw["gen_hellinger"] = gen_result.hellinger_per_layer
            np.savez_compressed(out_dir / f"{npz_stem}.npz", **_npz_kw)
            del _npz_kw

            _pref_scores = prefill_result.scores
            _gen_scores = gen_result.scores
            if _pref_scores is not None or _gen_scores is not None:
                _scores_kw: dict = {}
                if score_capture == "topk":
                    if _pref_scores is not None:
                        topv, topi, tail_max = _topk_compress(_pref_scores, score_topk)
                        _scores_kw["prefill_scores_topk"]      = topv
                        _scores_kw["prefill_scores_topk_idx"]  = topi
                        _scores_kw["prefill_scores_tail_max"]  = tail_max
                    if _gen_scores is not None:
                        topv, topi, tail_max = _topk_compress(_gen_scores, score_topk)
                        _scores_kw["gen_scores_topk"]      = topv
                        _scores_kw["gen_scores_topk_idx"]  = topi
                        _scores_kw["gen_scores_tail_max"]  = tail_max
                else:  # "full"
                    if _pref_scores is not None:
                        _scores_kw["prefill_scores"] = _pref_scores
                    if _gen_scores is not None:
                        _scores_kw["gen_scores"] = _gen_scores
                np.savez_compressed(out_dir / f"{npz_stem}_scores.npz", **_scores_kw)
                del _scores_kw

            meta = {
                "id": sample.id,
                "source": sample.source,
                "category": sample.category,
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

            # ── Optional: logit-lens entropy (parallel signal, no banks) ────
            if enable_logit_lens:
                from hopfield_llm.metrics.logit_lens import compute_logit_lens
                ll = compute_logit_lens(
                    llm,
                    question=sample.question,
                    generated_text=gen_result.generated_text,
                    apply_chat_template=True,
                )
                np.savez_compressed(
                    out_dir / f"{npz_stem}_logit_lens.npz",
                    prefill_logit_entropy=ll.prefill_entropy,
                    prefill_logit_top_prob=ll.prefill_top_prob,
                    gen_logit_entropy=ll.gen_entropy,
                    gen_logit_top_prob=ll.gen_top_prob,
                )

            # ── Optional: shuffled-bank null control ────────────────────────
            if enable_null_control:
                from hopfield_llm.metrics.null_control import compute_shuffled_bank_energy
                null_pref, null_gen = compute_shuffled_bank_energy(
                    llm,
                    question=sample.question,
                    generated_text=gen_result.generated_text,
                    banks=prim_banks,
                    beta=beta,
                    threshold=threshold,
                    energy_mode=energy_mode,
                    m_per_layer=prim_m,
                    hook_target=prim_hook_target,
                    normalize_query=prim_nrm_query,
                    seed=int(seed) + i,
                )
                np.savez_compressed(
                    out_dir / f"{npz_stem}_null_control.npz",
                    prefill_energy_shuffled=null_pref,    # [L]
                    gen_energy_shuffled=null_gen,         # [L, T]
                )

            # ── Secondary probe: teacher-forcing pass ───────────────────────
            if _probe_mode and secondary_probe is not None and sec_banks is not None:
                # Build the full sequence the model actually saw at generation
                # time: chat_template(question) + generated_text.  T_prompt
                # is the length of the chat-templated question alone.
                if hasattr(llm.tokenizer, "apply_chat_template"):
                    prompt_text = llm.tokenizer.apply_chat_template(
                        [{"role": "user", "content": sample.question}],
                        tokenize=False, add_generation_prompt=True,
                    )
                else:
                    prompt_text = sample.question
                _tok = llm.tokenizer(
                    prompt_text, return_tensors="pt", add_special_tokens=False
                )
                T_prompt = _tok["input_ids"].shape[1]
                full_text = prompt_text + gen_result.generated_text

                # Pass apply_chat_template=False because we already templated.
                sec_full = capture_prefill(
                    llm, full_text, sec_banks,
                    beta=beta, threshold=threshold, energy_mode=energy_mode,
                    hook_target=secondary_probe.hook_target,
                    normalize_query=secondary_probe.normalize_query,
                    m_per_layer=sec_m,
                    apply_chat_template=False,
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
                    prefill_entropy_mean=np.nanmean(sec_full.entropy[:, pref_slice], axis=1),
                    prefill_norm_entropy=sec_full.norm_entropy[:, pref_idx],
                    prefill_norm_entropy_mean=np.nanmean(sec_full.norm_entropy[:, pref_slice], axis=1),
                    prefill_lse=sec_full.lse[:, pref_idx],
                    prefill_lse_mean=np.nanmean(sec_full.lse[:, pref_slice], axis=1),
                    prefill_quadratic=sec_full.quadratic[:, pref_idx],
                    prefill_top_act=sec_full.top_act[:, pref_idx],
                    prefill_top_act_mean=np.nanmean(sec_full.top_act[:, pref_slice], axis=1),
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
                    "category": sample.category,
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
            bar.set_postfix(ok=n_ok, skip=n_skipped, err=n_err, refresh=False)

            # Free large intermediate objects and release the PyTorch CUDA
            # allocator's cached blocks so they don't accumulate across hundreds
            # of samples and eventually trigger an OOM kill.
            del prefill_result, gen_result
            if h_pre_raw is not None:
                del h_pre_raw
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Periodic GC to collect any reference cycles from HuggingFace
            # internals that Python's reference counting won't catch.
            if (n_ok + n_err) % 100 == 0:
                gc.collect()

        except Exception as exc:
            n_err += 1
            bar.set_postfix(ok=n_ok, err=n_err, refresh=False)
            tb = traceback.extract_tb(exc.__traceback__)
            origin = tb[-1] if tb else None
            loc = f"{origin.filename.split('/')[-1]}:{origin.lineno}" if origin else "?"
            log.warning("SKIP %s — %s: %s  (at %s)", sample.id, type(exc).__name__, exc, loc)

    bar.close()
    log.info("run_trajectory: finished — %d ok, %d errors → %s", n_ok, n_err, out_dir)
    return out_dir, calibrated_beta


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
