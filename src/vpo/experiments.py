"""Summaries and the experiments reported in the README."""

from __future__ import annotations

from dataclasses import replace

from .engine import BATCH, INTERACTIVE, Simulator, Stats
from .model import Admission, Autoscale, Batching, PipelineConfig, Retry, Traffic


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q / 100 * (len(ordered) - 1)))]


def _ms(value: float | None) -> float | None:
    return None if value is None else round(1000 * value, 1)


def summarize(config: PipelineConfig, stats: Stats) -> dict:
    done_i, done_b = stats.completed[INTERACTIVE], stats.completed[BATCH]
    duration = config.traffic.duration_s
    makespan = max(stats.end_time, duration)
    return {
        "label": config.label,
        "offered_per_s": round(stats.offered / duration, 1),
        "goodput_per_s": round((len(done_i) + len(done_b)) / makespan, 1),
        "completed": len(done_i) + len(done_b),
        "rejected_interactive": stats.rejected[INTERACTIVE],
        "rejected_batch": stats.rejected[BATCH],
        "dead_lettered": stats.dead_lettered,
        "dead_letter_poison": stats.dead_letter_poison,
        "retries": stats.retries,
        "p50_ms": _ms(_pct(done_i + done_b, 50)),
        "p95_ms": _ms(_pct(done_i + done_b, 95)),
        "p99_ms": _ms(_pct(done_i + done_b, 99)),
        "interactive_p95_ms": _ms(_pct(done_i, 95)),
        "batch_p95_ms": _ms(_pct(done_b, 95)),
        "mean_gpu_batch": round(stats.gpu_items / stats.gpu_batches, 2) if stats.gpu_batches else 0.0,
        "gpu_utilization": round(stats.gpu_busy_s / (config.gpus * makespan), 3),
        "cpu_worker_seconds": round(stats.worker_seconds, 1),
        "peak_wip": stats.peak_wip,
        "drain_time_s": round(max(0.0, makespan - duration), 1),
    }


def simulate(config: PipelineConfig) -> dict:
    return summarize(config, Simulator(config).run())


def batching_sweep() -> dict:
    """GPU-bound pipeline (plenty of CPU): batch size vs throughput and latency."""
    base = PipelineConfig(cpu_workers=32, traffic=Traffic(rate_per_s=400, duration_s=60))
    under_load = [
        simulate(replace(base, batching=Batching(max_batch=b, max_wait_ms=8), label=f"max_batch={b}"))
        for b in (1, 2, 4, 8, 16, 32)
    ]
    light = replace(base, traffic=Traffic(rate_per_s=60, duration_s=60))
    wait = [
        simulate(replace(light, batching=Batching(max_batch=32, max_wait_ms=w), label=f"max_wait={w}ms"))
        for w in (0, 2, 5, 10, 20, 50)
    ]
    return {"load_400_per_s": under_load, "load_60_per_s": wait}


def overload() -> list[dict]:
    """1.5x the CPU pool's capacity for a minute, under three admission policies."""
    base = PipelineConfig(cpu_workers=8, traffic=Traffic(rate_per_s=900, duration_s=60))
    return [
        simulate(replace(base, admission=Admission(policy=p, wip_limit=400), label=p))
        for p in ("unbounded", "reject", "shed_batch")
    ]


def failures() -> list[dict]:
    """2% transient task failures and 0.2% corrupt inputs, with and without retries."""
    base = PipelineConfig(traffic=Traffic(rate_per_s=300, duration_s=60))
    rows = []
    for attempts in (1, 3):
        retry = Retry(max_attempts=attempts, transient_failure_rate=0.02, poison_rate=0.002)
        rows.append(simulate(replace(base, retry=retry, label=f"max_attempts={attempts}")))
    return rows


def autoscaling() -> dict:
    """Traffic doubles from 250/s to 500/s at t=60 s."""
    traffic = Traffic(rate_per_s=250, duration_s=180, step_at_s=60, step_factor=2.0)
    configs = [
        PipelineConfig(cpu_workers=5, traffic=traffic, label="static 5 workers"),
        PipelineConfig(cpu_workers=12, traffic=traffic, label="static 12 workers"),
        PipelineConfig(
            cpu_workers=5, traffic=traffic, autoscale=Autoscale(enabled=True), label="autoscaled, 15 s cold start"
        ),
        PipelineConfig(
            cpu_workers=5,
            traffic=traffic,
            autoscale=Autoscale(enabled=True, cold_start_s=3.0),
            label="autoscaled, 3 s warm pool",
        ),
    ]
    rows, timelines = [], {}
    for config in configs:
        stats = Simulator(config).run()
        rows.append(summarize(config, stats))
        timelines[config.label] = stats.timeline
    return {"summary": rows, "timelines": timelines}
