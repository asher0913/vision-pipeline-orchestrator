"""``vpo run | report``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import experiments
from .model import Admission, Autoscale, Batching, PipelineConfig, Retry, Traffic


def _run(args) -> int:
    config = PipelineConfig(
        cpu_workers=args.cpu_workers,
        gpus=args.gpus,
        batching=Batching(args.max_batch, args.max_wait_ms),
        admission=Admission(args.admission, args.wip_limit),
        retry=Retry(args.max_attempts, transient_failure_rate=args.failure_rate, poison_rate=args.poison_rate),
        autoscale=Autoscale(enabled=args.autoscale),
        traffic=Traffic(rate_per_s=args.rate, duration_s=args.duration, seed=args.seed),
        label="cli",
    )
    print(json.dumps(experiments.simulate(config), indent=2))
    return 0


def _report(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("batching_sweep", "overload", "failures", "autoscaling"):
        if args.only and name not in args.only:
            continue
        print(f"running {name} …", flush=True)
        (out / f"{name}.json").write_text(json.dumps(getattr(experiments, name)(), indent=2) + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vpo", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="simulate one configuration")
    run.add_argument("--rate", type=float, default=300)
    run.add_argument("--duration", type=float, default=60)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--cpu-workers", type=int, default=8)
    run.add_argument("--gpus", type=int, default=1)
    run.add_argument("--max-batch", type=int, default=16)
    run.add_argument("--max-wait-ms", type=float, default=8)
    run.add_argument("--admission", choices=("unbounded", "reject", "shed_batch"), default="unbounded")
    run.add_argument("--wip-limit", type=int, default=400)
    run.add_argument("--max-attempts", type=int, default=3)
    run.add_argument("--failure-rate", type=float, default=0.0)
    run.add_argument("--poison-rate", type=float, default=0.0)
    run.add_argument("--autoscale", action="store_true")
    run.set_defaults(func=_run)
    report = sub.add_parser("report", help="run every experiment and write JSON results")
    report.add_argument("--out", default="results")
    report.add_argument("--only", nargs="*")
    report.set_defaults(func=_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
