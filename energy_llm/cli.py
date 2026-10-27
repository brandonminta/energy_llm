"""Single entry point: ``energy-llm {calibrate,extract,merge,status,smoke}``.

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

from energy_llm.signals import EXIT_DONE, EXIT_PENDING, GracefulStop


def _load(args, stopper: GracefulStop):
    from energy_llm.config import load_config
    from energy_llm.manifest import set_determinism
    from energy_llm.model import load_model, prepare_bundle
    from energy_llm.storage import read_items

    cfg = load_config(args.config)
    items = read_items(args.input)
    set_determinism(cfg.seed)
    t0 = time.time()
    model, tok = load_model(cfg.model.id, cfg.model.revision, cfg.model.device)
    bundle = prepare_bundle(model, tok, cfg.generation.max_new_tokens, require_bf16=True)
    return cfg, items, bundle, time.time() - t0


def cmd_calibrate(args) -> int:
    from energy_llm.calibrate import run_calibrate
    from energy_llm.config import load_config

    if load_config(args.config).beta_star_path.exists():
        print(f"beta* already calibrated: {load_config(args.config).beta_star_path}")
        return EXIT_DONE
    stopper = GracefulStop()
    cfg, items, bundle, _ = _load(args, stopper)
    rec = run_calibrate(cfg, bundle, items, args.input, stopper)
    if rec is None:
        return EXIT_PENDING
    print(f"beta* = {rec['beta_star']:.6g}  ({cfg.beta_star_path})")
    return EXIT_DONE


def cmd_extract(args) -> int:
    from energy_llm.extract import run_extract

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
    cfg, items, bundle, load_s = _load(args, stopper)
    report = run_extract(cfg, bundle, items, args.input, args.name, shard,
                         stopper=stopper, model_load_s=round(load_s, 2))
    print(json.dumps(report))
    return EXIT_DONE if report["state"] == "done" else EXIT_PENDING


def _shard_already_done(args, shard: int) -> bool:
    """Cheap pre-check so resubmitting finished shards does not load the model."""
    from energy_llm.extract import plan_shards, shard_path, shard_state

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
    from energy_llm.smoke import format_summary, run_smoke

    stopper = GracefulStop()
    cfg, items, bundle, load_s = _load(args, stopper)
    s = run_smoke(cfg, bundle, items, n=args.n, stopper=stopper, model_load_s=round(load_s, 2))
    print(format_summary(s))
    return EXIT_DONE if s["n_failed"] == 0 and not s["stopped_by_signal"] else 1


def _cfg_items(args):
    from energy_llm.config import load_config
    from energy_llm.storage import read_items

    return load_config(args.config), read_items(args.input)


def cmd_merge(args) -> int:
    from energy_llm.merge import run_merge

    cfg, items = _cfg_items(args)
    out = run_merge(cfg, items, args.input, args.name, allow_incomplete=args.allow_incomplete)
    print(json.dumps(out["counts"]))
    return EXIT_DONE


def cmd_status(args) -> int:
    from energy_llm.extract import plan_shards
    from energy_llm.merge import status_report

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

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return {"calibrate": cmd_calibrate, "extract": cmd_extract,
            "merge": cmd_merge, "status": cmd_status, "smoke": cmd_smoke}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
