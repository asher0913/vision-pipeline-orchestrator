from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import json
import random


@dataclass(frozen=True)
class Stage:
    name: str
    resource: str
    depends_on: tuple[str, ...] = ()
    duration_ms: int = 10


@dataclass
class Job:
    job_id: str
    priority: int
    stages: tuple[Stage, ...]
    completed: set[str] = field(default_factory=set)
    attempts: dict[str, int] = field(default_factory=dict)
    failed: bool = False


class PipelineOrchestrator:
    def __init__(self, cpu_slots: int = 4, gpu_slots: int = 1, max_queue: int = 100, retry_limit: int = 2) -> None:
        self.capacity = {"cpu": cpu_slots, "gpu": gpu_slots}
        self.max_queue = max_queue
        self.retry_limit = retry_limit
        self.queue: list[tuple[int, int, Job]] = []
        self.sequence = 0
        self.rejected = 0
        self.retries = 0
        self.dead_letters: list[str] = []
        self.clock_ms = 0
        self.busy_ms = {"cpu": 0, "gpu": 0}

    def submit(self, job: Job) -> bool:
        names = {stage.name for stage in job.stages}
        if any(set(stage.depends_on) - names for stage in job.stages):
            raise ValueError("stage dependency not found")
        if len(self.queue) >= self.max_queue:
            self.rejected += 1
            return False
        self.sequence += 1
        heapq.heappush(self.queue, (-job.priority, self.sequence, job))
        return True

    def ready(self, job: Job) -> list[Stage]:
        return [stage for stage in job.stages if stage.name not in job.completed and set(stage.depends_on) <= job.completed]

    def run(self, failure_plan: set[tuple[str, str, int]] | None = None) -> dict[str, object]:
        failure_plan = failure_plan or set()
        finished = []
        while self.queue:
            _, _, job = heapq.heappop(self.queue)
            if job.failed:
                continue
            ready = self.ready(job)
            if not ready:
                if len(job.completed) == len(job.stages):
                    finished.append(job.job_id)
                    continue
                raise RuntimeError("cycle or blocked DAG")
            stage = ready[0]
            attempt = job.attempts.get(stage.name, 0) + 1
            job.attempts[stage.name] = attempt
            self.clock_ms += stage.duration_ms
            self.busy_ms[stage.resource] += stage.duration_ms
            if (job.job_id, stage.name, attempt) in failure_plan:
                if attempt <= self.retry_limit:
                    self.retries += 1
                    self.sequence += 1
                    heapq.heappush(self.queue, (-job.priority, self.sequence, job))
                else:
                    job.failed = True
                    self.dead_letters.append(job.job_id)
                continue
            job.completed.add(stage.name)
            self.sequence += 1
            heapq.heappush(self.queue, (-job.priority, self.sequence, job))
        return {
            "completed": sorted(finished),
            "dead_letters": sorted(self.dead_letters),
            "retries": self.retries,
            "rejected": self.rejected,
            "makespan_ms": self.clock_ms,
            "resource_busy_ms": dict(self.busy_ms),
        }

    def recommended_workers(self, resource: str, queue_depth: int, target_jobs_per_worker: int = 8) -> int:
        return max(1, min(32, (queue_depth + target_jobs_per_worker - 1) // target_jobs_per_worker))


def stages() -> tuple[Stage, ...]:
    return (
        Stage("decode", "cpu", duration_ms=4),
        Stage("preprocess", "cpu", ("decode",), 6),
        Stage("infer", "gpu", ("preprocess",), 18),
        Stage("postprocess", "cpu", ("infer",), 5),
    )


def demo() -> dict[str, object]:
    orchestrator = PipelineOrchestrator(max_queue=40)
    for index in range(20):
        orchestrator.submit(Job(f"image-{index}", 2 if index < 3 else 1, stages()))
    report = orchestrator.run({("image-2", "infer", 1), ("image-7", "decode", 1)})
    report["recommended_gpu_workers"] = orchestrator.recommended_workers("gpu", 20)
    report["completion_rate"] = len(report["completed"]) / 20
    return report


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2, sort_keys=True))
