"""Run the full baseline suite over a trajectory directory.

Thin orchestrator around the module entrypoints — each baseline is also
runnable on its own via ``python -m hopfield_llm.evaluation.<name>``:

  - token-uncertainty + P(True)   →  {id}_baselines.json
  - Semantic Entropy (Farquhar)   →  {id}_semmentropy.json   [GPU + NLI model]
  - HalluField (Vu et al.)        →  {id}_hallufield.json
  - INTRA hidden states           →  {id}_intra.npz

All sidecars land next to the trajectory files and are picked up automatically
by ``hopfield-llm evaluate`` (per-baseline AUROC + nested-model test).

Usage:
    python scripts/local/run_baselines.py --trajectories outputs/<run>/trajectories --model qwen25_3b
    python scripts/local/run_baselines.py --trajectories <dir> --only baselines hallufield
"""

from __future__ import annotations

import argparse
import sys

_ALL = ["baselines", "semantic_entropy", "hallufield", "intra"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run all (or a subset of) hallucination-detection baselines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--trajectories", required=True, help="Trajectory directory")
    p.add_argument("--model", default=None, help="Model alias (inferred from metadata if omitted)")
    p.add_argument("--device", default=None)
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--only", nargs="*", choices=_ALL, default=_ALL,
                   help="Subset of baselines to run (default: all)")
    p.add_argument("--nli-device", default="cpu", help="Device for the Semantic-Entropy NLI model")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    common = ["--trajectories", args.trajectories]
    if args.model:
        common += ["--model", args.model]
    if args.device:
        common += ["--device", args.device]
    if args.no_4bit:
        common += ["--no-4bit"]
    if args.max_samples is not None:
        common += ["--max-samples", str(args.max_samples)]

    runners = {
        "baselines":        ("hopfield_llm.evaluation.baselines", common),
        "hallufield":       ("hopfield_llm.evaluation.hallufield", common),
        "intra":            ("hopfield_llm.evaluation.intra", common),
        "semantic_entropy": ("hopfield_llm.evaluation.semantic_entropy",
                             common + ["--nli-device", args.nli_device]),
    }

    rc = 0
    for name in args.only:
        module, argv = runners[name]
        print(f"\n=== {name}  (python -m {module}) ===", flush=True)
        mod = __import__(module, fromlist=["main"])
        try:
            rc |= mod.main(argv)
        except SystemExit as exc:  # argparse inside the module
            rc |= int(exc.code or 0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
