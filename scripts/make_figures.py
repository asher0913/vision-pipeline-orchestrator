"""Redraw docs/*.png from results/*.json (run `vpo report` first)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    return json.loads((ROOT / "results" / f"{name}.json").read_text())


def batching() -> None:
    data = load("batching_sweep")
    heavy, light = data["load_400_per_s"], data["load_60_per_s"]
    sizes = [int(r["label"].split("=")[1]) for r in heavy]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 3.6))
    left.plot(sizes, [r["goodput_per_s"] for r in heavy], marker="o", label="goodput (jobs/s)")
    left.axhline(400, color="grey", ls="--", lw=1, label="offered 400/s")
    left.set(xscale="log", xlabel="max GPU batch", ylabel="jobs/s", title="Throughput at 400 jobs/s offered")
    left.set_xticks(sizes, [str(s) for s in sizes])
    twin = left.twinx()
    twin.plot(sizes, [r["p95_ms"] / 1000 for r in heavy], marker="s", color="#d62728", label="p95 latency")
    twin.set(yscale="log", ylabel="p95 latency (s)")
    lines = left.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    left.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=8, loc="center right")
    waits = [float(r["label"].split("=")[1].rstrip("ms")) for r in light]
    right.plot(waits, [r["p50_ms"] for r in light], marker="o", label="p50 latency (ms)")
    right.plot(waits, [100 * r["gpu_utilization"] for r in light], marker="s", label="GPU busy %")
    right.plot(waits, [10 * r["mean_gpu_batch"] for r in light], marker="^", label="mean batch × 10")
    right.set(xlabel="max batching wait (ms)", title="Light load (60 jobs/s): latency vs GPU efficiency")
    right.legend(frameon=False, fontsize=8)
    for ax in (left, right):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "batching.png", dpi=150)
    plt.close(fig)


def autoscaling() -> None:
    data = load("autoscaling")["timelines"]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    for label, rows in data.items():
        t = [r["t"] for r in rows]
        top.plot(t, [r["cpu_queue"] for r in rows], label=label)
        bottom.plot(t, [r["cpu_workers"] for r in rows], label=label)
    for ax in (top, bottom):
        ax.axvline(60, color="grey", ls="--", lw=1)
        ax.grid(alpha=0.3)
    top.set_yscale("symlog", linthresh=10)
    top.set_ylim(0, 5e4)
    top.set(ylabel="CPU queue depth", title="Traffic doubles at t = 60 s (250 → 500 jobs/s)")
    bottom.set(ylabel="CPU workers", xlabel="time (s)")
    top.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "autoscaling.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    batching()
    autoscaling()
