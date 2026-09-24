from dataclasses import replace

import pytest

from vpo.engine import BATCH, INTERACTIVE, Simulator
from vpo.experiments import simulate
from vpo.model import Admission, Autoscale, Batching, PipelineConfig, Retry, Stage, Traffic

LIGHT = PipelineConfig(traffic=Traffic(rate_per_s=100, duration_s=20))


def run(config: PipelineConfig):
    return Simulator(config).run()


def test_every_job_is_accounted_for():
    config = replace(
        LIGHT,
        traffic=Traffic(rate_per_s=700, duration_s=20),
        admission=Admission("reject", 200),
        retry=Retry(3, transient_failure_rate=0.02, poison_rate=0.01),
    )
    stats = run(config)
    completed = len(stats.completed[INTERACTIVE]) + len(stats.completed[BATCH])
    assert stats.offered == stats.admitted + sum(stats.rejected.values())
    assert stats.admitted == completed + stats.dead_lettered


def test_results_are_deterministic():
    assert simulate(LIGHT) == simulate(LIGHT)
    assert simulate(LIGHT) != simulate(replace(LIGHT, traffic=replace(LIGHT.traffic, seed=1)))


def test_batches_respect_the_size_cap():
    for cap in (1, 4, 16):
        stats = run(replace(LIGHT, traffic=Traffic(rate_per_s=400, duration_s=10), batching=Batching(cap, 8)))
        assert 1 <= stats.max_gpu_batch <= cap
    assert run(replace(LIGHT, batching=Batching(1, 8))).max_gpu_batch == 1


def test_waiting_longer_forms_bigger_batches_at_light_load():
    eager = run(replace(LIGHT, batching=Batching(32, 0)))
    patient = run(replace(LIGHT, batching=Batching(32, 30)))
    assert patient.gpu_items / patient.gpu_batches > 1.5 * (eager.gpu_items / eager.gpu_batches)


def test_batching_raises_gpu_capacity():
    heavy = replace(LIGHT, cpu_workers=32, traffic=Traffic(rate_per_s=400, duration_s=20))
    single = simulate(replace(heavy, batching=Batching(1, 8)))
    batched = simulate(replace(heavy, batching=Batching(16, 8)))
    assert single["goodput_per_s"] < 150 < 380 < batched["goodput_per_s"]


def test_wip_limit_bounds_work_in_progress():
    overloaded = replace(LIGHT, cpu_workers=4, traffic=Traffic(rate_per_s=800, duration_s=20))
    bounded = run(replace(overloaded, admission=Admission("reject", 150)))
    unbounded = run(overloaded)
    assert bounded.peak_wip <= 150 < unbounded.peak_wip


def test_shedding_protects_interactive_traffic():
    overloaded = replace(LIGHT, cpu_workers=4, traffic=Traffic(rate_per_s=800, duration_s=20))
    shed = run(replace(overloaded, admission=Admission("shed_batch", 300, soft_fraction=0.5)))
    assert shed.rejected[BATCH] > 0
    assert shed.rejected[INTERACTIVE] < 0.05 * shed.rejected[BATCH]


def test_priority_scheduling_keeps_interactive_latency_low_under_overload():
    summary = simulate(replace(LIGHT, cpu_workers=4, traffic=Traffic(rate_per_s=600, duration_s=20)))
    assert summary["interactive_p95_ms"] < 200 < summary["batch_p95_ms"]


def test_retries_recover_transient_failures_and_poison_goes_to_dead_letter():
    failing = replace(LIGHT, traffic=Traffic(rate_per_s=200, duration_s=20))
    once = run(replace(failing, retry=Retry(1, transient_failure_rate=0.02, poison_rate=0.01)))
    thrice = run(replace(failing, retry=Retry(3, transient_failure_rate=0.02, poison_rate=0.01)))
    assert thrice.dead_lettered < once.dead_lettered / 3
    assert thrice.dead_lettered - thrice.dead_letter_poison <= 2  # almost only corrupt inputs remain
    assert thrice.retries > 0


def test_join_waits_for_the_slowest_branch():
    stages = (
        Stage("load", "cpu", 1.0, jitter=0.0),
        Stage("fast", "cpu", 1.0, ("load",), jitter=0.0),
        Stage("slow", "cpu", 50.0, ("load",), jitter=0.0),
    )
    stats = run(PipelineConfig(stages=stages, cpu_workers=4, traffic=Traffic(rate_per_s=5, duration_s=10)))
    latencies = stats.completed[INTERACTIVE] + stats.completed[BATCH]
    assert min(latencies) >= 0.051 - 1e-9


def test_autoscaler_adds_workers_after_a_traffic_step_and_saves_cost():
    traffic = Traffic(rate_per_s=250, duration_s=120, step_at_s=40, step_factor=2.0)
    auto = run(PipelineConfig(cpu_workers=5, traffic=traffic, autoscale=Autoscale(enabled=True, cold_start_s=3)))
    static = run(PipelineConfig(cpu_workers=12, traffic=traffic))
    workers = [row["cpu_workers"] for row in auto.timeline]
    assert max(workers[:35]) <= 7 and max(workers[60:]) >= 9
    assert auto.worker_seconds < static.worker_seconds


def test_unknown_admission_policy():
    with pytest.raises(ValueError):
        run(replace(LIGHT, admission=Admission("teleport")))
