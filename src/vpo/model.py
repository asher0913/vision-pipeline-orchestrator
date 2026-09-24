"""Pipeline description: stages, the DAG between them, resources and policies."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Stage:
    name: str
    pool: str  # "cpu" or "gpu"
    mean_ms: float  # CPU: per item; GPU: per item inside a batch
    depends_on: tuple[str, ...] = ()
    fixed_ms: float = 0.0  # GPU: per-batch launch/transfer overhead
    jitter: float = 0.25  # lognormal sigma of service time


def default_dag() -> tuple[Stage, ...]:
    """Decode → preprocess → GPU inference → {postprocess, embedding export} → done.

    GPU cost is modelled as a per-batch overhead plus a per-image term, the
    usual shape for a CNN: batching amortises kernel launches and PCIe copies.
    """
    return (
        Stage("decode", "cpu", 6.0),
        Stage("preprocess", "cpu", 4.0, ("decode",)),
        Stage("infer", "gpu", 1.5, ("preprocess",), fixed_ms=6.0, jitter=0.1),
        Stage("postprocess", "cpu", 2.0, ("infer",)),
        Stage("export_embedding", "cpu", 1.5, ("infer",)),
    )


def validate_dag(stages: tuple[Stage, ...]) -> list[str]:
    """Return a topological order, rejecting unknown dependencies and cycles."""
    names = {s.name for s in stages}
    if len(names) != len(stages):
        raise ValueError("duplicate stage names")
    for stage in stages:
        missing = set(stage.depends_on) - names
        if missing:
            raise ValueError(f"{stage.name} depends on unknown stages {sorted(missing)}")
        if stage.pool not in {"cpu", "gpu"}:
            raise ValueError(f"{stage.name}: unknown pool {stage.pool!r}")
    order, done = [], set()
    pending = {s.name: set(s.depends_on) for s in stages}
    while pending:
        ready = sorted(n for n, deps in pending.items() if deps <= done)
        if not ready:
            raise ValueError(f"cycle among {sorted(pending)}")
        for name in ready:
            order.append(name)
            done.add(name)
            del pending[name]
    return order


@dataclass(frozen=True)
class Batching:
    max_batch: int = 16
    max_wait_ms: float = 8.0


@dataclass(frozen=True)
class Admission:
    """Backpressure by bounding work in progress (jobs admitted but not finished).

    ``unbounded``   admit everything; queues absorb overload.
    ``reject``      reject any job once ``wip_limit`` jobs are in the system.
    ``shed_batch``  reject batch-priority jobs at ``soft_fraction`` of the limit,
                    interactive jobs only at the hard limit.
    """

    policy: str = "unbounded"
    wip_limit: int = 400
    soft_fraction: float = 0.6


@dataclass(frozen=True)
class Retry:
    max_attempts: int = 3
    base_backoff_ms: float = 20.0
    transient_failure_rate: float = 0.0
    poison_rate: float = 0.0  # corrupt inputs that fail every attempt at decode


@dataclass(frozen=True)
class Autoscale:
    enabled: bool = False
    interval_s: float = 5.0
    target_utilization: float = 0.7
    min_workers: int = 2
    max_workers: int = 32
    cold_start_s: float = 15.0


@dataclass(frozen=True)
class Traffic:
    rate_per_s: float = 300.0
    duration_s: float = 120.0
    interactive_share: float = 0.3
    step_at_s: float | None = None  # multiply the rate by step_factor from this time
    step_factor: float = 1.0
    seed: int = 0


@dataclass(frozen=True)
class PipelineConfig:
    stages: tuple[Stage, ...] = field(default_factory=default_dag)
    cpu_workers: int = 8
    gpus: int = 1
    batching: Batching = field(default_factory=Batching)
    admission: Admission = field(default_factory=Admission)
    retry: Retry = field(default_factory=Retry)
    autoscale: Autoscale = field(default_factory=Autoscale)
    traffic: Traffic = field(default_factory=Traffic)
    label: str = ""
