"""Compare received Stage 1 order-1 and order-3 simulator results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dynamics_challenge import PROBLEMS, exact_series
from dynamics_challenge.physics import observable_from_counts

ROOT = Path(__file__).resolve().parent
BASELINE_FILE = ROOT / "stage1_baseline_results.json"
BASE3_FILE = ROOT / "stage1_base3_results.json"
OUTPUT_JSON = ROOT / "stage1_order1_vs_order3_comparison.json"
OUTPUT_PNG = ROOT / "stage1_order1_vs_order3_comparison.png"


def summarize(path: Path) -> dict:
    payload = json.loads(path.read_text())
    problem = PROBLEMS["stage1"]
    values = []
    errors = []
    for counts in payload["counts"]:
        value, error = observable_from_counts(problem, counts)
        values.append(value)
        errors.append(error)
    exact = exact_series(problem)
    values_array = np.asarray(values)
    return {
        "job_id": payload["job_id"],
        "model": payload.get("model"),
        "shots": payload["shots"],
        "times": list(problem.times),
        "values": values,
        "errors": errors,
        "rms_vs_exact": float(np.sqrt(np.mean((values_array - exact) ** 2))),
    }


def main() -> None:
    baseline = summarize(BASELINE_FILE)
    base3 = summarize(BASE3_FILE)
    exact = exact_series(PROBLEMS["stage1"])
    comparison = {
        "stage": "stage1",
        "exact": exact.tolist(),
        "baseline_order1": baseline,
        "base3_order3": base3,
        "rms_difference": float(
            base3["rms_vs_exact"] - baseline["rms_vs_exact"]
        ),
    }
    OUTPUT_JSON.write_text(json.dumps(comparison, indent=2) + "\n")

    times = list(PROBLEMS["stage1"].times)
    fig, axis = plt.subplots(figsize=(11, 6))
    axis.plot(times, exact, "k-", linewidth=2.3, label="exact")
    axis.errorbar(
        times, baseline["values"], yerr=baseline["errors"],
        fmt="o--", capsize=3, color="tab:blue",
        label=f"baseline order 1 (RMS {baseline['rms_vs_exact']:.4f})",
    )
    axis.errorbar(
        times, base3["values"], yerr=base3["errors"],
        fmt="s--", capsize=3, color="tab:orange",
        label=f"base-3 order 3 (RMS {base3['rms_vs_exact']:.4f})",
    )
    axis.set_title("Stage 1: order 1 baseline versus order 3")
    axis.set_xlabel("time t")
    axis.set_ylabel("magnetization")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT_JSON}")
    print(f"wrote {OUTPUT_PNG}")
    print(f"order 1 RMS: {baseline['rms_vs_exact']:.6f}")
    print(f"order 3 RMS: {base3['rms_vs_exact']:.6f}")


if __name__ == "__main__":
    main()
