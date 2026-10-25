"""Compare local ZNE fold sets across Stage 1 and Stage 3 3x3."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("results")
STAGES = ("stage1", "stage3_3x3")
METHODS = (
    ("baseline", "noisy baseline"),
    ("folds_1_3", "ZNE folds 1,3"),
    ("folds_1_3_5", "ZNE folds 1,3,5"),
)


def main() -> None:
    data = {
        stage: json.loads((ROOT / f"zne_{stage}_fold_comparison.json").read_text())
        for stage in STAGES
    }
    comparison = {
        stage: {
            label: data[stage][key]["rms_vs_exact"]
            for key, label in METHODS
        }
        for stage in STAGES
    }
    output = ROOT / "zne_fold_stage_comparison.json"
    output.write_text(json.dumps(comparison, indent=2) + "\n")

    x = np.arange(len(STAGES))
    width = 0.25
    fig, axis = plt.subplots(figsize=(10, 6))
    for index, (key, label) in enumerate(METHODS):
        values = [comparison[stage][label] for stage in STAGES]
        bars = axis.bar(x + (index - 1) * width, values, width, label=label)
        axis.bar_label(bars, fmt="%.4f", padding=2, fontsize=9)
    axis.set_xticks(x, ["Stage 1", "Stage 3 3×3"])
    axis.set_ylabel("RMS error vs exact")
    axis.set_title("Local ZNE fold-set comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    plot = ROOT / "zne_fold_stage_comparison.png"
    fig.savefig(plot, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")
    print(f"wrote {plot}")
    for stage in STAGES:
        print(stage, comparison[stage])


if __name__ == "__main__":
    main()
