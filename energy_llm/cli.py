"""Single entry point: ``energy-llm {calibrate,extract,merge,status,smoke}``,
plus the verification-only ``verify`` and ``inspect`` (they never write into
production extractions).

Exit codes: 0 = finished; 75 = work remains (interrupted by SIGUSR1/SIGTERM
or failed ids still below max_attempts) — the batch script requeues on 75;
anything else = crash.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

from energy_llm.pipeline.signals import EXIT_DONE, EXIT_PENDING, GracefulStop


def _cfg_items(args):
    """Config + input items: everything the CPU-only commands need."""
    from energy_llm.config import load_config
    from energy_llm.storage import read_items

    return load_config(args.config), read_items(args.input)


def _load(args):
    """Config, items and the model bundle (bf16 enforced). Returns the model
    load time too, which the shard run records."""
    from energy_llm.core.model import load_model, prepare_bundle
    from energy_llm.manifest import set_determinism

    cfg, items = _cfg_items(args)
    set_determinism(cfg.seed)
    t0 = time.time()
    model, tok = load_model(cfg.model.id, cfg.model.revision, cfg.model.device)
    bundle = prepare_bundle(model, tok, cfg.generation.max_new_tokens, require_bf16=True)
    return cfg, items, bundle, time.time() - t0


def cmd_calibrate(args) -> int:
    from energy_llm.config import load_config
    from energy_llm.pipeline.calibrate import format_beta_stars, run_calibrate
    from energy_llm.storage import load_json

    beta_path = load_config(args.config).beta_star_path
    if beta_path.exists():
        rec = load_json(beta_path)
        print(f"beta* already calibrated: {beta_path}")
        if "variants" in rec:
            print(format_beta_stars(rec))
        return EXIT_DONE
    stopper = GracefulStop()
    cfg, items, bundle, _ = _load(args)
    rec = run_calibrate(cfg, bundle, items, args.input, stopper)
    if rec is None:
        return EXIT_PENDING
    print(format_beta_stars(rec))
    print(f"written to {cfg.beta_star_path}")
    return EXIT_DONE


def cmd_extract(args) -> int:
    from energy_llm.pipeline.extract import run_extract

    shard = args.shard_index
    if shard is None:
        env = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env is None:
            print("--shard-index is required outside a SLURM array", file=sys.stderr)
            return 2
        shard = int(env)
    if _shard_already_done(args, shard):
        print(json.dumps({"shard_index": shard, "state": "done", "skipped_model_load": True}))
        return EXIT_DONE
    stopper = GracefulStop()
    cfg, items, bundle, load_s = _load(args)
    report = run_extract(cfg, bundle, items, args.input, args.name, shard,
                         stopper=stopper, model_load_s=round(load_s, 2))
    print(json.dumps(report))
    return EXIT_DONE if report["state"] == "done" else EXIT_PENDING


def _shard_already_done(args, shard: int) -> bool:
    """Cheap pre-check so resubmitting finished shards does not load the model."""
    from energy_llm.pipeline.extract import plan_shards, shard_path, shard_state

    cfg, items = _cfg_items(args)
    shards = plan_shards(items, cfg.extraction.samples_per_shard)
    if not 0 <= shard < len(shards):
        return False  # run_extract reports the error
    sdir = shard_path(cfg.extraction_dir(args.name), shard)
    if not (sdir / "shard.json").exists():
        return False
    return shard_state(sdir, shard, [it.id for it in shards[shard]],
                       cfg.extraction.max_attempts).done


def cmd_smoke(args) -> int:
    from energy_llm.pipeline.smoke import format_summary, run_smoke

    stopper = GracefulStop()
    cfg, items, bundle, load_s = _load(args)
    s = run_smoke(cfg, bundle, items, n=args.n, stopper=stopper, model_load_s=round(load_s, 2))
    print(format_summary(s))
    return EXIT_DONE if s["n_failed"] == 0 and not s["stopped_by_signal"] else 1


def cmd_verify(args) -> int:
    from energy_llm.verification.verify import format_report, run_verify

    cfg, items, bundle, _ = _load(args)
    layers = [int(x) for x in args.reference_layers.split(",")] if args.reference_layers else None
    report = run_verify(cfg, bundle, items, n=args.n, ref_layers=layers, ref_n=args.reference_n,
                        tf_max_noise_ratio=args.tf_max_noise_ratio, fp32_check=args.fp32_reference)
    print(format_report(report))
    return EXIT_DONE if report["passed"] else 1


def cmd_inspect(args) -> int:
    from energy_llm.verification.inspection import inspect_item

    print(inspect_item(args.run, args.id))
    return EXIT_DONE


def cmd_merge(args) -> int:
    from energy_llm.pipeline.merge import run_merge

    cfg, items = _cfg_items(args)
    out = run_merge(cfg, items, args.input, args.name, allow_incomplete=args.allow_incomplete)
    print(json.dumps(out["counts"]))
    return EXIT_DONE


def cmd_status(args) -> int:
    from energy_llm.pipeline.extract import plan_shards
    from energy_llm.pipeline.merge import status_report

    cfg, items = _cfg_items(args)
    if args.num_shards:
        print(len(plan_shards(items, cfg.extraction.samples_per_shard)))
        return EXIT_DONE
    if args.extraction_dir:
        print(cfg.extraction_dir(args.name))
        return EXIT_DONE
    rep = status_report(cfg, items, args.name, args.input)
    if args.missing:
        print(",".join(str(k) for k in rep["missing_shards"]))
    elif args.json:
        print(json.dumps(rep, indent=2))
    else:
        print(f"{'shard':>6} {'ids':>6} {'done':>6} {'failed':>7} {'pending':>8}")
        for s in rep["shards"]:
            print(f"{s['index']:>6} {s['ids']:>6} {s['completed']:>6} "
                  f"{s['exhausted']:>7} {s['pending']:>8}")
        print(f"missing shards: {rep['missing_shards'] or 'none'}")
        if rep["exhausted_ids"]:
            print(f"ids out of attempts: {len(rep['exhausted_ids'])}")
        if "throughput" in rep:
            t = rep["throughput"]
            print(f"median {t['median_sec_per_item']:.2f} s/item, model load "
                  f"{t['median_model_load_s']:.0f} s -> samples_per_shard for ~2 h: "
                  f"{t['suggested_samples_per_shard']}")
    return EXIT_DONE


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="energy-llm", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, name: bool):
        sp.add_argument("--config", required=True, help="run YAML (configs/*.yaml)")
        sp.add_argument("--input", required=True, help="JSONL with {id, question}")
        if name:
            sp.add_argument("--name", required=True, help="extraction run name (e.g. truthfulqa)")

    common(sub.add_parser("calibrate", help="compute beta* on a warm-up JSONL"), name=False)
    sp = sub.add_parser("extract", help="process one shard")
    common(sp, name=True)
    sp.add_argument("--shard-index", type=int, default=None,
                    help="default: $SLURM_ARRAY_TASK_ID")
    sp = sub.add_parser("merge", help="merge finished shards into the final files")
    common(sp, name=True)
    sp.add_argument("--allow-incomplete", action="store_true")
    sp = sub.add_parser("status", help="report shard progress / missing shards")
    common(sp, name=True)
    g = sp.add_mutually_exclusive_group()
    g.add_argument("--missing", action="store_true", help="print unfinished shard indices (a,b,c)")
    g.add_argument("--num-shards", action="store_true", help="print the number of shards")
    g.add_argument("--json", action="store_true")
    g.add_argument("--extraction-dir", action="store_true", help="print the extraction directory")

    sp = sub.add_parser("smoke", help="extract + merge the first N questions in one process")
    common(sp, name=False)
    sp.add_argument("--n", type=int, default=20)

    sp = sub.add_parser("verify", help="verification: checks vs the model and dev/reference.py")
    common(sp, name=False)
    sp.add_argument("--n", type=int, default=4, help="questions to verify (default 4)")
    sp.add_argument("--reference-layers", default=None,
                    help="comma-separated layers for the float64 reference (default 0,L/2,L-1)")
    sp.add_argument("--reference-n", type=int, default=2,
                    help="questions that go through the float64 reference (default 2)")
    sp.add_argument("--tf-max-noise-ratio", type=float, default=0.1,
                    help="tf_vs_cache PASS: max p99(|trace change|) / sigma per layer (default 0.1)")
    sp.add_argument("--no-fp32-reference", dest="fp32_reference", action="store_false",
                    help="skip the float32 reference check (loads a second, float32 model)")
    sp = sub.add_parser("inspect", help="print positions and per-layer phases of one id")
    sp.add_argument("--run", required=True, help="extraction directory (with manifest.json)")
    sp.add_argument("--id", required=True)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return {"calibrate": cmd_calibrate, "extract": cmd_extract,
            "merge": cmd_merge, "status": cmd_status, "smoke": cmd_smoke,
            "verify": cmd_verify, "inspect": cmd_inspect}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
