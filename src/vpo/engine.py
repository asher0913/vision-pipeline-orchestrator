"""Discrete-event execution of a job DAG on CPU workers and batching GPUs."""

from __future__ import annotations

import heapq
import math
import random
from collections import deque
from dataclasses import dataclass, field

from .model import PipelineConfig, validate_dag

INTERACTIVE, BATCH = 0, 1


@dataclass
class Job:
    job_id: int
    arrival: float
    priority: int
    poison: bool
    waiting_on: dict[str, int]  # stage -> unfinished dependencies
    attempts: dict[str, int] = field(default_factory=dict)
    finished_stages: int = 0


@dataclass
class Stats:
    offered: int = 0
    admitted: int = 0
    rejected: dict[int, int] = field(default_factory=lambda: {INTERACTIVE: 0, BATCH: 0})
    completed: dict[int, list[float]] = field(default_factory=lambda: {INTERACTIVE: [], BATCH: []})
    dead_lettered: int = 0
    dead_letter_poison: int = 0
    retries: int = 0
    gpu_batches: int = 0
    gpu_items: int = 0
    max_gpu_batch: int = 0
    cpu_busy_s: float = 0.0
    gpu_busy_s: float = 0.0
    worker_seconds: float = 0.0
    peak_wip: int = 0
    end_time: float = 0.0
    timeline: list[dict] = field(default_factory=list)


class Simulator:
    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config
        self.order = validate_dag(config.stages)
        self.stages = {s.name: s for s in config.stages}
        self.children: dict[str, list[str]] = {s.name: [] for s in config.stages}
        for s in config.stages:
            for parent in s.depends_on:
                self.children[parent].append(s.name)
        self.roots = [s.name for s in config.stages if not s.depends_on]
        self.first_stage = self.order[0]
        self.rng = random.Random(config.traffic.seed)
        self.events: list = []
        self.seq = 0
        self.now = 0.0
        self.jobs: dict[int, Job] = {}
        self.cpu_workers = config.cpu_workers
        self.cpu_busy = 0
        self.cpu_starting = 0
        self.cpu_queue: list = []  # (priority, seq, job_id, stage)
        self.gpu_free = config.gpus
        self.gpu_queues = (deque(), deque())  # per priority: (enqueue_time, job_id, stage)
        self.gpu_timer_at: float | None = None
        self.stats = Stats()
        self._last_t = 0.0
        self._window_busy = 0.0
        self._window_started = 0.0
        self._window_admitted = 0

    # --------------------------------------------------------------- plumbing
    def _push(self, t: float, kind: str, payload=None) -> None:
        self.seq += 1
        heapq.heappush(self.events, (t, self.seq, kind, payload))

    def _advance(self, t: float) -> None:
        dt = t - self._last_t
        if dt > 0:
            self.stats.cpu_busy_s += self.cpu_busy * dt
            self._window_busy += self.cpu_busy * dt
            self.stats.gpu_busy_s += (self.cfg.gpus - self.gpu_free) * dt
            self.stats.worker_seconds += self.cpu_workers * dt
            self._last_t = t
        self.now = t

    def _service(self, mean_ms: float, jitter: float) -> float:
        return mean_ms / 1000 * math.exp(self.rng.gauss(-jitter * jitter / 2, jitter))

    @property
    def wip(self) -> int:
        return len(self.jobs)

    # ---------------------------------------------------------------- traffic
    def _schedule_arrivals(self) -> None:
        tr = self.cfg.traffic
        t, job_id = 0.0, 0
        while True:
            rate = tr.rate_per_s * (tr.step_factor if tr.step_at_s is not None and t >= tr.step_at_s else 1.0)
            t += self.rng.expovariate(rate)
            if t >= tr.duration_s:
                break
            priority = INTERACTIVE if self.rng.random() < tr.interactive_share else BATCH
            poison = self.rng.random() < self.cfg.retry.poison_rate
            self._push(t, "arrival", (job_id, priority, poison))
            job_id += 1

    def _admit(self, priority: int) -> bool:
        adm = self.cfg.admission
        if adm.policy == "unbounded":
            return True
        if adm.policy == "reject":
            return self.wip < adm.wip_limit
        if adm.policy == "shed_batch":
            limit = adm.wip_limit if priority == INTERACTIVE else adm.soft_fraction * adm.wip_limit
            return self.wip < limit
        raise ValueError(f"unknown admission policy {adm.policy!r}")

    def _on_arrival(self, payload) -> None:
        job_id, priority, poison = payload
        self.stats.offered += 1
        if not self._admit(priority):
            self.stats.rejected[priority] += 1
            return
        self.stats.admitted += 1
        self._window_admitted += 1
        waiting = {s.name: len(s.depends_on) for s in self.cfg.stages}
        job = Job(job_id, self.now, priority, poison, waiting)
        self.jobs[job_id] = job
        self.stats.peak_wip = max(self.stats.peak_wip, self.wip)
        for stage in self.roots:
            self._enqueue(job, stage)

    # -------------------------------------------------------------- queueing
    def _enqueue(self, job: Job, stage: str) -> None:
        if self.stages[stage].pool == "cpu":
            self.seq += 1
            heapq.heappush(self.cpu_queue, (job.priority, self.seq, job.job_id, stage))
            self._dispatch_cpu()
        else:
            self.gpu_queues[job.priority].append((self.now, job.job_id, stage))
            self._dispatch_gpu()

    def _dispatch_cpu(self) -> None:
        while self.cpu_busy < self.cpu_workers and self.cpu_queue:
            _, _, job_id, stage = heapq.heappop(self.cpu_queue)
            job = self.jobs.get(job_id)
            if job is None:  # a sibling branch was dead-lettered; drop the orphaned task
                continue
            spec = self.stages[stage]
            self.cpu_busy += 1
            ok = self._succeeds(job, stage)
            self._push(self.now + self._service(spec.mean_ms, spec.jitter), "cpu_done", (job_id, stage, ok))

    def _gpu_backlog(self) -> int:
        for queue in self.gpu_queues:  # discard tasks of jobs that were dead-lettered meanwhile
            while queue and queue[0][1] not in self.jobs:
                queue.popleft()
        return len(self.gpu_queues[0]) + len(self.gpu_queues[1])

    def _dispatch_gpu(self) -> None:
        batching = self.cfg.batching
        while self.gpu_free > 0 and self._gpu_backlog():
            oldest = min(q[0][0] for q in self.gpu_queues if q)
            full = self._gpu_backlog() >= batching.max_batch
            expired = self.now - oldest >= batching.max_wait_ms / 1000 - 1e-12
            if not (full or expired):
                deadline = oldest + batching.max_wait_ms / 1000
                if self.gpu_timer_at is None or deadline < self.gpu_timer_at:
                    self.gpu_timer_at = deadline
                    self._push(deadline, "gpu_timer")
                return
            batch = []
            for queue in self.gpu_queues:  # interactive first
                while queue and len(batch) < batching.max_batch:
                    _, job_id, stage = queue.popleft()
                    if job_id in self.jobs:
                        batch.append((job_id, stage))
            if not batch:
                continue
            self.gpu_free -= 1
            spec = self.stages[batch[0][1]]
            duration = self._service(spec.fixed_ms + spec.mean_ms * len(batch), spec.jitter)
            outcomes = [self._succeeds(self.jobs[j], s) for j, s in batch]
            self.stats.gpu_batches += 1
            self.stats.gpu_items += len(batch)
            self.stats.max_gpu_batch = max(self.stats.max_gpu_batch, len(batch))
            self._push(self.now + duration, "gpu_done", (batch, outcomes))

    def _succeeds(self, job: Job, stage: str) -> bool:
        if job.poison and stage == self.first_stage:
            return False
        return self.rng.random() >= self.cfg.retry.transient_failure_rate

    # ------------------------------------------------------------ completion
    def _finish_task(self, job_id: int, stage: str, ok: bool) -> None:
        job = self.jobs.get(job_id)
        if job is None:
            return
        if not ok:
            attempts = job.attempts.get(stage, 0) + 1
            job.attempts[stage] = attempts
            retry = self.cfg.retry
            if attempts < retry.max_attempts:
                self.stats.retries += 1
                backoff = retry.base_backoff_ms / 1000 * 2 ** (attempts - 1) * (0.5 + self.rng.random())
                self._push(self.now + backoff, "retry", (job_id, stage))
            else:
                self.stats.dead_lettered += 1
                self.stats.dead_letter_poison += job.poison
                del self.jobs[job_id]
            return
        job.finished_stages += 1
        if job.finished_stages == len(self.stages):
            self.stats.completed[job.priority].append(self.now - job.arrival)
            del self.jobs[job_id]
            return
        for child in self.children[stage]:
            job.waiting_on[child] -= 1
            if job.waiting_on[child] == 0:
                self._enqueue(job, child)

    # ------------------------------------------------------------- autoscale
    def _autoscale(self) -> None:
        """Throughput-driven scaling: workers needed = arrival rate × CPU seconds per job.

        Utilization alone under-reports demand during overload, because busy
        time is capped by the workers that already exist. Measuring admitted
        work instead, plus the time to drain the current backlog within one
        interval, sizes the pool for what is arriving.
        """
        auto = self.cfg.autoscale
        window = self.now - self._window_started
        admitted_rate = self._window_admitted / window if window > 0 else 0.0
        self._window_busy, self._window_admitted, self._window_started = 0.0, 0, self.now
        cpu_s_per_job = sum(s.mean_ms for s in self.cfg.stages if s.pool == "cpu") / 1000
        cpu_s_per_task = cpu_s_per_job / max(1, sum(1 for s in self.cfg.stages if s.pool == "cpu"))
        steady = admitted_rate * cpu_s_per_job
        drain = len(self.cpu_queue) * cpu_s_per_task / auto.interval_s
        desired = math.ceil(steady / auto.target_utilization + drain)
        desired = max(auto.min_workers, min(auto.max_workers, desired))
        planned = self.cpu_workers + self.cpu_starting
        if desired > planned:
            for _ in range(desired - planned):
                self.cpu_starting += 1
                self._push(self.now + auto.cold_start_s, "worker_online")
        elif desired < self.cpu_workers:
            self.cpu_workers = max(desired, auto.min_workers)  # busy workers retire when they finish
        self._push(self.now + auto.interval_s, "autoscale")

    # ------------------------------------------------------------------ loop
    def run(self) -> Stats:
        self._schedule_arrivals()
        if self.cfg.autoscale.enabled:
            self.cpu_workers = max(self.cfg.autoscale.min_workers, self.cpu_workers)
            self._push(self.cfg.autoscale.interval_s, "autoscale")
        self._push(1.0, "sample")
        horizon = self.cfg.traffic.duration_s
        while self.events:
            t, _, kind, payload = heapq.heappop(self.events)
            if kind in ("autoscale", "sample") and t > horizon and not self.jobs:
                continue
            self._advance(t)
            if kind == "arrival":
                self._on_arrival(payload)
            elif kind == "cpu_done":
                self.cpu_busy -= 1
                self._finish_task(*payload)
                self._dispatch_cpu()
            elif kind == "gpu_done":
                self.gpu_free += 1
                batch, outcomes = payload
                for (job_id, stage), ok in zip(batch, outcomes, strict=True):
                    self._finish_task(job_id, stage, ok)
                self._dispatch_gpu()
                self._dispatch_cpu()
            elif kind == "gpu_timer":
                if self.gpu_timer_at is not None and t >= self.gpu_timer_at - 1e-12:
                    self.gpu_timer_at = None
                self._dispatch_gpu()
            elif kind == "retry":
                job_id, stage = payload
                if job_id in self.jobs:
                    self._enqueue(self.jobs[job_id], stage)
            elif kind == "autoscale":
                self._autoscale()
            elif kind == "worker_online":
                self.cpu_starting -= 1
                self.cpu_workers += 1
                self._dispatch_cpu()
            elif kind == "sample":
                self.stats.timeline.append(
                    {
                        "t": round(t, 3),
                        "wip": self.wip,
                        "cpu_queue": len(self.cpu_queue),
                        "gpu_queue": self._gpu_backlog(),
                        "cpu_workers": self.cpu_workers,
                    }
                )
                if t < horizon or self.jobs:
                    self._push(t + 1.0, "sample")
        self.stats.end_time = self._last_t
        return self.stats
