#!/usr/bin/env python3
"""Standalone interactive model test.

Loads a model via HFLLM and runs a chat loop.  Optionally instruments
each generation with Hopfield energy metrics per layer printed inline.
Completely isolated from the pipeline — no banks artifact required
(instrumentation builds a transient bank from the loaded model).

Usage
-----
    python scripts/local/chat_model.py
    python scripts/local/chat_model.py --model qwen25_3b
    python scripts/local/chat_model.py --model qwen25_3b --instrument
    python scripts/local/chat_model.py --model qwen25_3b --instrument --banks path/to/banks.pt
    python scripts/local/chat_model.py --no-4bit        # CPU / no quantization
    python scripts/local/chat_model.py --max-tokens 128

Commands during chat
--------------------
    /quit or /exit — exit the loop
    /clear         — reset conversation (prompt restarts)
    /info          — print model summary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive model test")
    p.add_argument("--model", default="qwen25_3b",
                   help="Model alias or HF model ID (default: qwen25_3b)")
    p.add_argument("--no-4bit", dest="load_in_4bit", action="store_false",
                   help="Disable 4-bit quantization (needed on CPU)")
    p.add_argument("--device", default=None,
                   help="Device override: 'cuda', 'cpu', etc.")
    p.add_argument("--max-tokens", type=int, default=200,
                   help="Max new tokens per response (default: 200)")
    p.add_argument("--instrument", action="store_true",
                   help="Print per-layer energy metrics after each response")
    p.add_argument("--banks", default=None,
                   help="Path to pre-built banks .pt artifact for instrumentation. "
                        "If omitted and --instrument is set, builds a transient bank.")
    p.add_argument("--beta", type=float, default=15.0,
                   help="Hopfield inverse temperature (default: 15.0)")
    p.add_argument("--bank-type", default="down",
                   help="Bank projection type for transient bank build (default: down)")
    return p.parse_args()


def _build_transient_banks(llm, bank_type: str) -> "dict[int, object]":
    """Extract banks from the loaded model for instrumentation."""
    from hopfield_llm.memory.banks import extract_banks
    print(f"[instrument] Building transient '{bank_type}' banks … ", end="", flush=True)
    banks = extract_banks(llm, bank=bank_type, normalize=False, strict=False)
    print(f"done ({len(banks)} layers)")
    return banks


def _load_banks_artifact(path: str) -> "dict[int, object]":
    from hopfield_llm.storage.artifacts import load_torch_artifact
    artifact = load_torch_artifact(path)
    return artifact["banks"]


def _run_with_instrumentation(
    llm,
    prompt: str,
    banks: "dict[int, object]",
    beta: float,
    max_new_tokens: int,
) -> str:
    """Run prefill + generation with energy metrics, print a compact summary."""
    from hopfield_llm.hooks.capture import capture_prefill, capture_generation
    import numpy as np

    prefill = capture_prefill(llm, prompt, banks, beta=beta)
    gen = capture_generation(llm, prompt, banks, beta=beta, max_new_tokens=max_new_tokens)

    # Compact per-layer summary: mean generation energy vs prefill energy
    gen_mean = np.nanmean(gen.energy, axis=1)   # [L]
    delta = gen_mean - prefill.energy            # [L]

    print("\n[instrument] Layer energy delta (gen_mean − prefill):")
    L = len(delta)
    cols = 8
    for row_start in range(0, L, cols):
        row = delta[row_start:row_start + cols]
        idx_str  = "  ".join(f"L{row_start + j:02d}" for j in range(len(row)))
        val_str  = "  ".join(f"{v:+.3f}" if not np.isnan(v) else "   nan" for v in row)
        print(f"  {idx_str}")
        print(f"  {val_str}")
    mean_d = float(np.nanmean(delta))
    print(f"  mean_delta={mean_d:+.4f}  tokens_generated={len(gen.token_ids)}\n")

    return gen.generated_text


def _run_plain(llm, prompt: str, max_new_tokens: int) -> str:
    return llm.generate(prompt, max_new_tokens=max_new_tokens, do_sample=False)


def main() -> None:
    args = _parse_args()

    # Add src/ to path so the package is importable without pip install -e
    repo_root = Path(__file__).resolve().parent.parent.parent
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from hopfield_llm.models.loader import HFLLM

    print(f"Loading model '{args.model}' …")
    llm = HFLLM(
        model_id=args.model,
        device=args.device,
        load_in_4bit=args.load_in_4bit,
        output_hidden_states=False,
    )
    summary = llm.summary()
    print(
        f"Ready: {summary['model_id']}  "
        f"device={summary['device']}  "
        f"quant={'4-bit NF4' if summary['load_in_4bit_effective'] else summary['dtype']}  "
        f"layers={summary['n_layers_resolved']}"
    )

    banks = None
    if args.instrument:
        if args.banks:
            print(f"[instrument] Loading banks from {args.banks}")
            banks = _load_banks_artifact(args.banks)
        else:
            banks = _build_transient_banks(llm, args.bank_type)

    mode_label = "instrumented" if args.instrument else "plain"
    print(f"\nChat mode: {mode_label} | /quit to exit | /info for model info\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"/quit", "/exit"}:
            print("Bye.")
            break
        if user_input.lower() == "/info":
            import json
            print(json.dumps(summary, indent=2))
            continue
        if user_input.lower() == "/clear":
            print("[cleared]")
            continue

        print("Model: ", end="", flush=True)
        try:
            if args.instrument and banks is not None:
                response = _run_with_instrumentation(
                    llm, user_input, banks, args.beta, args.max_tokens
                )
                # Strip the prompt echo that HF decode sometimes includes
                if response.startswith(user_input):
                    response = response[len(user_input):].lstrip()
                print(response)
            else:
                response = _run_plain(llm, user_input, args.max_tokens)
                if response.startswith(user_input):
                    response = response[len(user_input):].lstrip()
                print(response)
        except Exception as exc:
            print(f"\n[error] {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
