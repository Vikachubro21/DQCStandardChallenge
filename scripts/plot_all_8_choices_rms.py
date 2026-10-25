"""Plot the Stage 1 eight-choice RMS comparison as grouped bars."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("results")
INPUT = ROOT / "stage1_all_8_choices_rms.csv"
OUTPUT = ROOT / "stage1_all_8_choices_rms.png"


def main() -> None:
    rows = list(csv.DictReader(INPUT.open()))
    methods = list(dict.fromkeys(row["method"] for row in rows))
    noise_models = list(dict.fromkeys(row["noise_model"] for row in rows))
    values = {
        model: {
            method: next(float(row["rms"]) for row in rows
                         if row["noise_model"] == model and row["method"] == method)
            for method in methods
        }
        for model in noise_models
    }
    # Descending mean RMS: the best combined method appears at the right.
    mean_rms = {
        method: np.mean([
            next(float(row["rms"]) for row in rows
                 if row["noise_model"] == model and row["method"] == method)
            for model in noise_models
        ])
        for method in methods
    }
    methods.sort(key=mean_rms.__getitem__, reverse=True)

    x = np.arange(len(noise_models))
    width = 0.8 / len(methods)
    fig, axis = plt.subplots(figsize=(13, 6))
    for index, method in enumerate(methods):
        bars = axis.bar(
            x + (index - (len(methods) - 1) / 2) * width,
            [values[model][method] for model in noise_models],
            width, label=method.replace("+", " + "),
        )
        axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    axis.set_title("Stage 1: RMS error by noise model and mitigation choice")
    axis.set_ylabel("RMS error vs exact")
    axis.set_xticks(x, noise_models)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="Noise model")
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
