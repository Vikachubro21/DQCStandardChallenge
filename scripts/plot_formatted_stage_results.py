"""Plot formatted local Stage 1/2 results against baseline and exact values."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dynamics_challenge import PROBLEMS, exact_series

ROOT = Path(__file__).resolve().parents[1]
FORMATTED_DIR = ROOT / "results"
BASELINE_DIR = ROOT / "validation" / "results" / "combined_zne_stages"
OUTPUT_DIR = ROOT / "validation"


def make_plot(stage: str) -> Path:
    formatted = json.loads((FORMATTED_DIR / f"{stage}_local_formatted.json").read_text())
    baseline = json.loads((BASELINE_DIR / f"{stage}.json").read_text())
    problem = PROBLEMS[stage]
    times = np.asarray([point["time"] for point in formatted["points"]])
    values = np.asarray([point["value"] for point in formatted["points"]])
    errors = np.asarray([point["uncertainty"] for point in formatted["points"]])
    exact = exact_series(problem)

    fig, axis = plt.subplots(figsize=(11, 6))
    axis.plot(times, exact, "k-", linewidth=2.3, label="exact")
    axis.errorbar(
        baseline["times"], baseline["baseline"], yerr=baseline["baseline_error"],
        fmt="o--", capsize=3, color="tab:gray", label="noisy baseline",
    )
    axis.errorbar(
        times, values, yerr=errors, fmt="s-", capsize=3, color="tab:blue",
        label="formatted local result",
    )
    axis.set_title(f"{stage}: formatted result vs baseline and exact")
    axis.set_xlabel("time t")
    axis.set_ylabel("observable")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    output = OUTPUT_DIR / f"{stage}_formatted_vs_baseline_exact.png"
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")
    return output


if __name__ == "__main__":
    for stage_name in ("stage1", "stage2"):
        make_plot(stage_name)
