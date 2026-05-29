"""Semantic Entropy baseline (Farquhar et al., 2024).

Samples N completions per prompt at temperature T, clusters semantically
equivalent answers via bidirectional NLI entailment, and computes the
entropy over the cluster distribution.

High semantic entropy → the model produces diverse, mutually-inconsistent
answers → likely hallucinating.

Algorithm (Farquhar 2024):
  1. Generate N completions s_1, …, s_N at T > 0.
  2. Build equivalence graph G: edge (i,j) iff
       NLI(question + s_i → s_j) = ENTAILMENT  AND
       NLI(question + s_j → s_i) = ENTAILMENT
  3. Clusters = connected components of G.
  4. Semantic entropy H = -Σ_c (n_c / N) log(n_c / N)

Usage (GPU required for both generator and NLI model):
    from hopfield_llm.evaluation.semantic_entropy import (
        SemanticEntropyScorer, run_semantic_entropy_batch,
    )
    scorer = SemanticEntropyScorer(nli_model_id="facebook/bart-large-mnli")
    run_semantic_entropy_batch(
        llm, scorer, traj_dir="exp04_results/trajectories", n_completions=10
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from hopfield_llm.utils.logging import get_logger

log = get_logger("evaluation.semantic_entropy")

_DEFAULT_NLI_MODEL = "facebook/bart-large-mnli"
_ENTAILMENT_LABEL  = "entailment"   # lowercase label used by BART-MNLI pipeline


class SemanticEntropyScorer:
    """Wraps an NLI model to compute bidirectional entailment and semantic entropy.

    Args:
        nli_model_id: HuggingFace model ID for the zero-shot NLI pipeline.
                      Tested with 'facebook/bart-large-mnli'.
        device:       PyTorch device string for the NLI model.
        entailment_threshold: Minimum NLI probability to count as entailment.
    """

    def __init__(
        self,
        nli_model_id: str = _DEFAULT_NLI_MODEL,
        device: str = "cpu",
        entailment_threshold: float = 0.5,
    ) -> None:
        from transformers import pipeline as hf_pipeline  # noqa: WPS433

        self.threshold = entailment_threshold
        log.info("Loading NLI model %s on %s ...", nli_model_id, device)
        self._pipe = hf_pipeline(
            "zero-shot-classification",
            model=nli_model_id,
            device=device,
        )
        log.info("NLI model loaded.")

    def _entails(self, premise: str, hypothesis: str) -> bool:
        """Return True if NLI(premise → hypothesis) ≥ entailment_threshold."""
        result = self._pipe(
            premise,
            candidate_labels=["entailment", "neutral", "contradiction"],
            hypothesis_template="{}",
        )
        label_scores = dict(zip(result["labels"], result["scores"]))
        return label_scores.get(_ENTAILMENT_LABEL, 0.0) >= self.threshold

    def bidirectional_entailment(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
    ) -> bool:
        """Return True iff answers are bidirectionally entailing (semantically equivalent)."""
        prefix = question.strip()
        premise_a  = f"{prefix} {answer_a.strip()}"
        premise_b  = f"{prefix} {answer_b.strip()}"
        hypo_b = answer_b.strip()
        hypo_a = answer_a.strip()
        return self._entails(premise_a, hypo_b) and self._entails(premise_b, hypo_a)

    def cluster(self, question: str, answers: list[str]) -> list[int]:
        """Assign cluster IDs to each answer using a union-find on the entailment graph.

        Returns:
            List of integer cluster IDs, one per answer (0-indexed).
        """
        N = len(answers)
        parent = list(range(N))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            parent[find(x)] = find(y)

        for i in range(N):
            for j in range(i + 1, N):
                if self.bidirectional_entailment(question, answers[i], answers[j]):
                    union(i, j)

        # Normalise cluster IDs to 0..K-1
        root_map: dict[int, int] = {}
        clusters = []
        for i in range(N):
            root = find(i)
            if root not in root_map:
                root_map[root] = len(root_map)
            clusters.append(root_map[root])
        return clusters

    def semantic_entropy(self, cluster_ids: list[int]) -> float:
        """Compute entropy H = -Σ (n_c/N) log(n_c/N) in nats."""
        N = len(cluster_ids)
        if N == 0:
            return float("nan")
        counts: dict[int, int] = {}
        for c in cluster_ids:
            counts[c] = counts.get(c, 0) + 1
        total = sum(counts.values())
        H = sum(
            -(n / total) * np.log(n / total + 1e-10)
            for n in counts.values()
        )
        return float(H)


def sample_completions(
    llm,
    question: str,
    n: int = 10,
    temperature: float = 0.5,
    max_new_tokens: int = 50,
    apply_chat_template: bool = True,
) -> list[str]:
    """Generate N stochastic completions for a question.

    Args:
        llm:                HFLLM instance.
        question:           The question / prompt.
        n:                  Number of completions to generate.
        temperature:        Sampling temperature (>0 for stochasticity).
        max_new_tokens:     Max tokens per completion.
        apply_chat_template: Apply chat template to prompt.

    Returns:
        List of decoded completion strings (generated text only, without prompt).
    """
    if apply_chat_template and hasattr(llm.tokenizer, "apply_chat_template"):
        prompt_str = llm.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_str = question

    inputs = llm.tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(llm.device)
    prompt_len = input_ids.shape[1]

    completions = []
    with torch.no_grad():
        for _ in range(n):
            out = llm.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=1.0,
                pad_token_id=llm.tokenizer.pad_token_id,
                eos_token_id=llm.tokenizer.eos_token_id,
            )
            gen_ids = out[0, prompt_len:]
            text = llm.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            completions.append(text)

    return completions


def compute_semantic_entropy(
    llm,
    scorer: SemanticEntropyScorer,
    question: str,
    n: int = 10,
    temperature: float = 0.5,
    max_new_tokens: int = 50,
    apply_chat_template: bool = True,
) -> dict:
    """Compute semantic entropy for one sample.

    Returns:
        Dict with:
          semantic_entropy   — scalar H in nats
          n_clusters         — number of equivalence classes
          n_completions      — N (may be < n if generation failed)
          completions        — list of generated strings
          cluster_ids        — list of cluster assignments
    """
    completions = sample_completions(
        llm, question, n=n, temperature=temperature,
        max_new_tokens=max_new_tokens,
        apply_chat_template=apply_chat_template,
    )
    if not completions:
        return {
            "semantic_entropy": float("nan"),
            "n_clusters":       0,
            "n_completions":    0,
            "completions":      [],
            "cluster_ids":      [],
        }

    cluster_ids = scorer.cluster(question, completions)
    H = scorer.semantic_entropy(cluster_ids)
    n_clusters = len(set(cluster_ids))

    return {
        "semantic_entropy": H,
        "n_clusters":       n_clusters,
        "n_completions":    len(completions),
        "completions":      completions,
        "cluster_ids":      cluster_ids,
    }


def run_semantic_entropy_batch(
    llm,
    scorer: SemanticEntropyScorer,
    traj_dir: Path,
    output_dir: Optional[Path] = None,
    n: int = 10,
    temperature: float = 0.5,
    max_new_tokens: int = 50,
    apply_chat_template: bool = True,
    max_samples: Optional[int] = None,
) -> list[dict]:
    """Compute semantic entropy for all trajectory samples.

    Writes {sample_id}_semmentropy.json next to each trajectory file.
    Resume-safe: already-computed files are skipped.

    Args:
        llm:                 HFLLM instance (loaded model).
        scorer:              SemanticEntropyScorer with NLI model loaded.
        traj_dir:            Trajectory directory.
        output_dir:          Output directory (defaults to traj_dir).
        n:                   Number of completions to sample per prompt.
        temperature:         Sampling temperature.
        max_new_tokens:      Max tokens per sampled completion.
        apply_chat_template: Apply chat template (must match trajectory capture).
        max_samples:         Stop after N samples.

    Returns:
        List of summary records (id, semantic_entropy, n_clusters).
    """
    traj_dir = Path(traj_dir)
    out_dir  = Path(output_dir) if output_dir else traj_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_files = sorted(
        f for f in traj_dir.glob("*.json")
        if not any(t in f.name for t in ("calibration", "_label", "_baselines",
                                          "_hallufield", "_intra", "_scores", "_hpre",
                                          "_lam", "_semmentropy"))
    )
    if max_samples:
        sample_files = sample_files[:max_samples]

    total     = len(sample_files)
    done      = 0
    skipped   = 0
    summaries = []

    log.info(
        "run_semantic_entropy_batch: %d samples, n=%d, T=%.2f",
        total, n, temperature,
    )

    for i, jf in enumerate(sample_files, 1):
        try:
            meta = json.loads(jf.read_text())
        except Exception as exc:
            log.warning("Skipping %s: %s", jf.name, exc)
            continue

        sample_id = meta.get("id", jf.stem)
        out_path  = out_dir / f"{sample_id}_semmentropy.json"

        if out_path.exists():
            skipped += 1
            continue

        question = meta.get("question", "")

        try:
            result = compute_semantic_entropy(
                llm, scorer, question,
                n=n, temperature=temperature, max_new_tokens=max_new_tokens,
                apply_chat_template=apply_chat_template,
            )
        except Exception as exc:
            log.warning("[%d/%d] FAILED %s: %s", i, total, sample_id, exc)
            continue

        record = {
            "id":               sample_id,
            "semantic_entropy": result["semantic_entropy"],
            "n_clusters":       result["n_clusters"],
            "n_completions":    result["n_completions"],
            "cluster_ids":      result["cluster_ids"],
            "completions":      result["completions"],
            "n_sampling":       n,
            "temperature":      temperature,
        }
        out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
        done += 1
        summaries.append({"id": sample_id,
                          "semantic_entropy": result["semantic_entropy"],
                          "n_clusters": result["n_clusters"]})
        log.info(
            "[%d/%d] %s  H=%.3f  clusters=%d",
            i, total, sample_id, result["semantic_entropy"], result["n_clusters"],
        )

    log.info(
        "run_semantic_entropy_batch: done=%d skipped=%d total=%d",
        done, skipped, total,
    )
    return summaries


def _resolve_alias(traj_dir: Path, model: Optional[str]) -> str:
    if model:
        return model
    first = next(iter(sorted(traj_dir.glob("*.json"))), None)
    if first is None:
        raise SystemExit(f"No trajectory JSON files in {traj_dir}")
    return json.loads(first.read_text(encoding="utf-8")).get("model", "qwen25_3b")


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m hopfield_llm.evaluation.semantic_entropy --trajectories DIR``."""
    import argparse

    from hopfield_llm.models.loader import HFLLM
    from hopfield_llm.utils.logging import setup_logging

    parser = argparse.ArgumentParser(
        description="Semantic Entropy baseline (Farquhar et al., 2024) over a trajectory dir",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--trajectories", required=True, help="Trajectory directory")
    parser.add_argument("--model", default=None, help="Generator model alias")
    parser.add_argument("--nli-model", default=_DEFAULT_NLI_MODEL, dest="nli_model")
    parser.add_argument("--n", type=int, default=10, help="Completions sampled per prompt")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=50, dest="max_new_tokens")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--nli-device", default="cpu", dest="nli_device")
    parser.add_argument("--max-samples", type=int, default=None, dest="max_samples")
    args = parser.parse_args(argv)

    setup_logging()
    traj_dir = Path(args.trajectories)
    alias = _resolve_alias(traj_dir, args.model)
    llm = HFLLM(model_id=alias, device=args.device, load_in_4bit=not args.no_4bit)
    llm.model.eval()
    scorer = SemanticEntropyScorer(nli_model_id=args.nli_model, device=args.nli_device)
    run_semantic_entropy_batch(
        llm, scorer, traj_dir, n=args.n, temperature=args.temperature,
        max_new_tokens=args.max_new_tokens, max_samples=args.max_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
