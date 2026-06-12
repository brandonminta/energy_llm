"""Stage 2 — trajectory capture (subsec:stage2, alg:per_sample).

Per sample: the question is wrapped via the tokenizer's checkpoint-default
``apply_chat_template``; the answer is greedy-decoded with a fixed
``max_new_tokens`` cap; forward pre-hooks (``register_forward_pre_hook``) on
each decoder layer's ``mlp`` module capture the probe point r_t^(l)
(eq:probe_point). Inside the hook the captured state is multiplied by the
row-normalised bank, so everything downstream operates on the retrieval
scores s (eq:retrieval_scores), never on the raw gate pre-activations g.

Capture mechanics: after greedy decoding, one hooked teacher-forced forward
pass runs over [prompt ++ generated tokens] and the hook keeps positions
``n_prompt-1 .. end``. By causal masking these states are mathematically
identical to the incremental-decoding states: position ``n_prompt-1`` is the
prefill last-token reference r^(l,ref) and positions ``n_prompt .. end`` are
the states of every newly produced token t = 1..T_g — including the final
token, which an incremental hook never sees because generation stops before
it is fed back. The prefill and generation states therefore share the
identical prompt prefix by construction (subsec:pa_probe_positions).

Hidden states are never persisted: only the [L] reference scalars, the
[L, T_g] per-token scalars, and metadata reach disk. The per-sample retrieval
score tensors are additionally cached in compressed float16 behind
``trajectory.cache_score_tensors`` (enabled for E1), so the beta-sensitivity
analysis (subsec:an_sensitivity) recomputes all features with no forward
passes.

HPC robustness: per-sample artifacts are the checkpoint unit (atomic writes,
resume-by-default), shard striding for array jobs, SIGUSR1/SIGTERM trapped
for graceful preemption, per-sample failure isolation into failures.jsonl,
and aggressive GPU/host-RAM hygiene (the per-sample artifact is flushed and
released immediately; nothing accumulates across the run).
"""

from __future__ import annotations

import gc
import logging
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from energy_llm import metrics
from energy_llm.config import ExperimentConfig, require_beta_star
from energy_llm.data import Sample
from energy_llm.io_artifacts import (
    SampleManifest,
    append_jsonl,
    save_json_atomic,
    save_npz_atomic,
)
from energy_llm.models import resolve_decoder_layers, resolve_mlp

log = logging.getLogger("energy_llm.trajectory")


# ----------------------------------------------------------------------
# Hooks
# ----------------------------------------------------------------------

class ProbePointHooks:
    """Forward pre-hooks on every decoder layer's ``mlp`` module.

    A pre-hook receives the exact input tensor of the module it is attached
    to — at this position the probe point r_t^(l), the output of the
    post-attention RMSNorm (eq:probe_point). The hook observes without
    rewriting: the computation graph is untouched (subsec:stage2).

    Inside the hook, scores ``s = r @ m_hat^T`` and the quadratic term
    ``0.5*||r||^2`` are computed and moved to CPU immediately; no GPU
    intermediate outlives the forward call.
    """

    def __init__(self, model, banks: dict[str, Any], score_dtype: str = "float32"):
        self.layers = resolve_decoder_layers(model)
        n_layers = len(self.layers)
        if banks["n_layers"] != n_layers:
            raise ValueError(
                f"banks artifact has {banks['n_layers']} layers, model has {n_layers}"
            )
        device = next(model.parameters()).device
        self._dtype = getattr(torch, score_dtype)
        self.m_hat = {
            l: banks["m_hat"][l].to(device=device, dtype=self._dtype)
            for l in range(n_layers)
        }
        self.n_layers = n_layers
        self._enabled = False
        self._position_offset = 0
        self.scores: dict[int, torch.Tensor] = {}
        self.quadratic: dict[int, torch.Tensor] = {}
        self._handles = [
            resolve_mlp(layer).register_forward_pre_hook(self._make_hook(l))
            for l, layer in enumerate(self.layers)
        ]

    def _make_hook(self, layer_idx: int):
        def hook(_module, args):
            if not self._enabled:
                return
            hidden = args[0]  # [B, T, D]
            r = hidden[0, self._position_offset:, :]  # [P, D] probe points
            s = r.to(self._dtype) @ self.m_hat[layer_idx].T  # [P, K]
            self.scores[layer_idx] = s.detach().to("cpu", torch.float32)
            r32 = r.detach().float()
            self.quadratic[layer_idx] = (
                0.5 * (r32 * r32).sum(dim=-1)
            ).to("cpu", torch.float32)
        return hook

    def capture(self, position_offset: int) -> "ProbePointHooks":
        self.scores = {}
        self.quadratic = {}
        self._position_offset = position_offset
        self._enabled = True
        return self

    def stop(self) -> None:
        self._enabled = False

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self.m_hat = {}


# ----------------------------------------------------------------------
# Prompting and generation
# ----------------------------------------------------------------------

def build_prompt_ids(tokenizer, question: str, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Wrap the question with the checkpoint-default chat template."""
    messages = [{"role": "user", "content": question}]
    try:
        enc = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
    except TypeError:  # older transformers without return_dict
        out = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        )
        input_ids = out if torch.is_tensor(out) else torch.tensor([out])
        attention_mask = torch.ones_like(input_ids)
    return input_ids.to(device), attention_mask.to(device)


@dataclass
class SampleTrajectory:
    """In-memory result of one per-sample capture before serialisation."""
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]
    score_tensors: dict[str, np.ndarray] = field(default_factory=dict)


@torch.inference_mode()
def capture_sample(
    model,
    tokenizer,
    hooks: ProbePointHooks,
    sample: Sample,
    beta_star: float,
    max_new_tokens: int = 50,
    cache_score_tensors: bool = False,
) -> SampleTrajectory:
    """Run alg:per_sample for one question; returns arrays ready to persist."""
    device = next(model.parameters()).device
    input_ids, attention_mask = build_prompt_ids(tokenizer, sample.question, device)
    n_prompt = input_ids.shape[1]

    # --- greedy decoding (hooks off); per-step logits feed the baselines ---
    hooks.stop()
    gen_out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    sequences = gen_out.sequences  # [1, n_prompt + T_g]
    gen_ids = sequences[0, n_prompt:]
    n_generated = int(gen_ids.shape[0])
    if n_generated == 0:
        raise RuntimeError("model generated zero tokens")

    # Baselines from the same decoding pass (subsec:an_contribution):
    # token log-probability of the chosen token and full-vocabulary
    # predictive entropy, per generated step.
    token_logprobs = np.empty(n_generated, dtype=np.float64)
    token_predictive_entropy = np.empty(n_generated, dtype=np.float64)
    for t in range(n_generated):
        logits = gen_out.scores[t][0].float()
        logp = torch.log_softmax(logits, dim=-1)
        token_logprobs[t] = float(logp[gen_ids[t]])
        p = torch.exp(logp)
        token_predictive_entropy[t] = float(-(p * logp).sum())

    generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    del gen_out

    # --- one hooked teacher-forced pass over prompt ++ generated tokens ---
    # Captures position n_prompt-1 (prefill reference) and positions
    # n_prompt..end (every generated token); see module docstring.
    hooks.capture(position_offset=n_prompt - 1)
    full_attention = torch.ones_like(sequences)
    model(input_ids=sequences, attention_mask=full_attention, use_cache=False)
    hooks.stop()

    scores = {l: hooks.scores[l].numpy() for l in range(hooks.n_layers)}        # [T_g+1, K_l]
    quadratic = {l: hooks.quadratic[l].numpy() for l in range(hooks.n_layers)}  # [T_g+1]
    hooks.scores = {}
    hooks.quadratic = {}
    del sequences, full_attention, input_ids, attention_mask

    # --- online metrics on the rescaled scores s (never on g) ---
    L = hooks.n_layers
    ref = {k: np.empty(L) for k in ("lse_term", "quadratic_term", "entropy", "norm_entropy")}
    gen = {k: np.empty((L, n_generated)) for k in
           ("lse_term", "quadratic_term", "entropy", "norm_entropy", "js", "hellinger_sq")}

    for l in range(L):
        s = scores[l].astype(np.float64)        # [T_g+1, K_l]
        s_ref, s_gen = s[0], s[1:]
        p_ref = metrics.retrieval_distribution(s_ref, beta_star)
        p_gen = metrics.retrieval_distribution(s_gen, beta_star)

        ref["lse_term"][l] = metrics.lse_term(s_ref, beta_star)
        ref["quadratic_term"][l] = quadratic[l][0]
        ref["entropy"][l] = metrics.entropy(p_ref)
        ref["norm_entropy"][l] = metrics.norm_entropy(p_ref)

        gen["lse_term"][l] = metrics.lse_term(s_gen, beta_star)
        gen["quadratic_term"][l] = quadratic[l][1:]
        gen["entropy"][l] = metrics.entropy(p_gen)
        gen["norm_entropy"][l] = metrics.norm_entropy(p_gen)
        gen["js"][l] = metrics.js_nats(p_gen, p_ref[None, :], axis=1)
        gen["hellinger_sq"][l] = metrics.hellinger_sq(p_gen, p_ref[None, :], axis=1)

    arrays: dict[str, np.ndarray] = {}
    for k, v in ref.items():
        arrays[f"ref_{k}"] = v
    for k, v in gen.items():
        arrays[f"gen_{k}"] = v
    arrays["token_logprobs"] = token_logprobs
    arrays["token_predictive_entropy"] = token_predictive_entropy
    arrays["beta_star"] = np.float64(beta_star)

    score_tensors: dict[str, np.ndarray] = {}
    if cache_score_tensors:
        widths = {scores[l].shape[1] for l in range(L)}
        if len(widths) != 1:
            raise ValueError(
                f"cache_score_tensors requires a uniform intermediate dim, got {widths}"
            )
        stacked = np.stack([scores[l] for l in range(L)]).astype(np.float16)  # [L, T_g+1, K]
        score_tensors["ref_retrieval_scores"] = stacked[:, 0, :]
        score_tensors["gen_retrieval_scores"] = stacked[:, 1:, :]
        score_tensors["ref_quadratic_term"] = arrays["ref_quadratic_term"]
        score_tensors["gen_quadratic_term"] = arrays["gen_quadratic_term"]
        del stacked
    del scores, quadratic

    metadata = {
        "sample_id": sample.sample_id,
        "source": sample.source,
        "category": sample.category,
        "question": sample.question,
        "gold_answers": sample.gold_answers,
        "generated_text": generated_text,
        "n_prompt_tokens": n_prompt,
        "n_tokens_generated": n_generated,
        "beta_star": beta_star,
        "bank_id": "gate",
    }
    return SampleTrajectory(arrays=arrays, metadata=metadata, score_tensors=score_tensors)


@torch.inference_mode()
def capture_prefill_reference_scores(
    model, tokenizer, hooks: ProbePointHooks, question: str
) -> np.ndarray:
    """Prefill-only capture for beta calibration: retrieval scores at the
    last prompt position of every layer. Returns [L, K] float32."""
    device = next(model.parameters()).device
    input_ids, attention_mask = build_prompt_ids(tokenizer, question, device)
    hooks.capture(position_offset=input_ids.shape[1] - 1)
    model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    hooks.stop()
    s = np.stack([hooks.scores[l][0].numpy() for l in range(hooks.n_layers)])
    hooks.scores = {}
    hooks.quadratic = {}
    return s


# ----------------------------------------------------------------------
# Stage runner: resume, sharding, preemption, failure isolation
# ----------------------------------------------------------------------

class _GracefulStop:
    """Trap SIGUSR1/SIGTERM; the run loop finishes the current sample,
    flushes, and exits in a resumable state (HPC preemption)."""

    def __init__(self) -> None:
        self.requested = False
        self._signal_name: str | None = None
        for sig in (signal.SIGUSR1, signal.SIGTERM):
            try:
                signal.signal(sig, self._handler)
            except ValueError:
                pass  # not in main thread (tests)

    def _handler(self, signum, _frame):
        self.requested = True
        self._signal_name = signal.Signals(signum).name
        log.warning("received %s — will stop after the current sample", self._signal_name)


def _memory_snapshot() -> dict[str, float]:
    snap: dict[str, float] = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                snap["rss_gb"] = round(int(line.split()[1]) / 1024**2, 3)
    except OSError:
        pass
    if torch.cuda.is_available():
        snap["gpu_alloc_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 3)
        snap["gpu_reserved_gb"] = round(torch.cuda.memory_reserved() / 1024**3, 3)
    return snap


def write_sample_artifacts(
    trajectories_dir: Path, traj: SampleTrajectory
) -> None:
    """Persist one sample. The metadata JSON is written LAST — it is the
    commit marker the manifest checks, so a kill leaves no 'complete' sample
    with missing arrays."""
    sid = traj.metadata["sample_id"]
    save_npz_atomic(trajectories_dir / f"{sid}.npz", **traj.arrays)
    if traj.score_tensors:
        save_npz_atomic(trajectories_dir / f"{sid}_retrieval_scores.npz",
                        **traj.score_tensors)
    save_json_atomic(trajectories_dir / f"{sid}.json", traj.metadata)


def nan_trajectory(sample: Sample, n_layers: int, beta_star: float,
                   error: str) -> SampleTrajectory:
    """NaN placeholder for a failed sample: median imputation in the probe
    handles it (subsec:probe); the failure itself goes to failures.jsonl."""
    nan_l = np.full(n_layers, np.nan)
    nan_lt = np.full((n_layers, 1), np.nan)
    arrays = {
        "ref_lse_term": nan_l, "ref_quadratic_term": nan_l.copy(),
        "ref_entropy": nan_l.copy(), "ref_norm_entropy": nan_l.copy(),
        "gen_lse_term": nan_lt, "gen_quadratic_term": nan_lt.copy(),
        "gen_entropy": nan_lt.copy(), "gen_norm_entropy": nan_lt.copy(),
        "gen_js": nan_lt.copy(), "gen_hellinger_sq": nan_lt.copy(),
        "token_logprobs": np.array([np.nan]),
        "token_predictive_entropy": np.array([np.nan]),
        "beta_star": np.float64(beta_star),
    }
    metadata = {
        "sample_id": sample.sample_id, "source": sample.source,
        "category": sample.category, "question": sample.question,
        "gold_answers": sample.gold_answers, "generated_text": None,
        "n_prompt_tokens": None, "n_tokens_generated": 0,
        "beta_star": beta_star, "bank_id": "gate", "failed": True,
        "error": error,
    }
    return SampleTrajectory(arrays=arrays, metadata=metadata)


def run_trajectories(
    cfg: ExperimentConfig,
    model,
    tokenizer,
    banks: dict[str, Any],
    samples: list[Sample],
    shard_index: int = 0,
    num_shards: int = 1,
    limit: int | None = None,
    max_samples_per_job: int | None = None,
) -> dict[str, Any]:
    """Stage 2 entry point. Resume-by-default: completed samples are skipped,
    so a rerun of a killed array task is a no-op for finished samples."""
    from energy_llm.data import shard as shard_fn

    beta_star = require_beta_star(cfg)
    out_dir = cfg.trajectories_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = SampleManifest(out_dir)

    work = shard_fn(samples, shard_index, num_shards)
    if limit is not None:
        work = work[:limit]
    pending_ids = set(manifest.pending([s.sample_id for s in work]))
    todo = [s for s in work if s.sample_id in pending_ids]
    log.info("shard %d/%d: %d samples, %d already complete, %d to do",
             shard_index, num_shards, len(work), len(work) - len(todo), len(todo))

    stopper = _GracefulStop()
    hooks = ProbePointHooks(model, banks, score_dtype=cfg.trajectory.score_dtype)
    n_done = n_failed = 0
    started = time.time()
    try:
        for i, sample in enumerate(todo):
            if max_samples_per_job is not None and n_done + n_failed >= max_samples_per_job:
                log.info("max_samples_per_job=%d reached — exiting resumable", max_samples_per_job)
                break
            try:
                traj = capture_sample(
                    model, tokenizer, hooks, sample, beta_star,
                    max_new_tokens=cfg.generation.max_new_tokens,
                    cache_score_tensors=cfg.trajectory.cache_score_tensors,
                )
                write_sample_artifacts(out_dir, traj)
                n_done += 1
            except Exception as exc:  # per-sample failure isolation
                import traceback
                hooks.stop()
                append_jsonl(out_dir / "failures.jsonl", {
                    "sample_id": sample.sample_id,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "time": time.time(),
                })
                write_sample_artifacts(
                    out_dir,
                    nan_trajectory(sample, hooks.n_layers, beta_star, repr(exc)),
                )
                n_failed += 1
                log.error("sample %s failed: %r", sample.sample_id, exc)
            finally:
                # release everything per sample; nothing accumulates
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if cfg.trajectory.log_memory_every and (i + 1) % cfg.trajectory.log_memory_every == 0:
                log.info("progress %d/%d  mem=%s", i + 1, len(todo), _memory_snapshot())
            if stopper.requested:
                log.info("graceful stop after %d samples — resumable", n_done + n_failed)
                break
    finally:
        hooks.remove()

    return {
        "shard_index": shard_index, "num_shards": num_shards,
        "n_done": n_done, "n_failed": n_failed,
        "n_skipped": len(work) - len(todo),
        "elapsed_s": round(time.time() - started, 1),
        "stopped_by_signal": stopper.requested,
    }
