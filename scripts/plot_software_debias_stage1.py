"""Export and plot the completed Stage 1 mitigation benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dynamics_challenge import PROBLEMS, exact_series


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("/tmp/opencode/software_debias_stage1_checkpoint.json"),
    )
    parser.add_argument(
        "--data-file", type=Path,
        default=Path("results/software_debias_stage1_combined.json"),
    )
    parser.add_argument(
        "--plot-file", type=Path,
        default=Path("results/software_debias_stage1_combined.png"),
    )
    args = parser.parse_args()

    checkpoint = json.loads(args.checkpoint.read_text())
    problem = PROBLEMS["stage1"]
    times = list(problem.times)
    exact = exact_series(problem).tolist()
    models = {}
    for key, result in checkpoint.items():
        model, label = key.split(":", 1)
        models.setdefault(model, {})[label] = result

    payload = {
        "problem": "stage1",
        "times": times,
        "exact": exact,
        "models": models,
    }
    args.data_file.parent.mkdir(parents=True, exist_ok=True)
    args.data_file.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def estimated_error(result) -> np.ndarray:
        """Estimate shot uncertainty from aggregate checkpoint data."""
        values = np.asarray(result["values"], dtype=float)
        effective_shots = max(1.0, result["shots_used"] * result["retained"])
        # The checkpoint does not contain per-outcome histograms.  Account for
        # the extra variance introduced by the ZNE intercept conservatively.
        amplification = 1.5 if result["label"].startswith("ZNE") else 1.0
        return amplification * np.sqrt(
            np.maximum(1.0 - values ** 2, 0.0) / effective_shots
        )

    fig, axes = plt.subplots(
        len(models), 1, figsize=(15, 5 * len(models)), sharex=True, squeeze=False
    )
    labels = ["none", "ZNE+symmetry+debias"]
    for axis, (model, results) in zip(axes[:, 0], sorted(models.items())):
        axis.plot(times, exact, "k-", linewidth=2.5, label="exact")
        for color, label in zip(("tab:gray", "tab:blue"), labels):
            if label not in results:
                continue
            result = results[label]
            values = result["values"]
            style = "--" if label == "none" else "-"
            width = 2.2 if label == "none" else 1.1
            axis.errorbar(
                times, values, yerr=estimated_error(result), linestyle=style,
                linewidth=width, color=color, marker=".", capsize=3,
                label="baseline" if label == "none" else "ZNE + symmetry + debiasing",
            )
        axis.set_title(f"Stage 1 — {model} noise")
        axis.set_ylabel("magnetization")
        axis.grid(alpha=0.25)
        axis.legend(ncol=3, fontsize=8)
    axes[-1, 0].set_xlabel("time t")
    fig.suptitle("Stage 1 combined mitigation compared with exact dynamics", y=0.995)
    fig.tight_layout()
    args.plot_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot_file, dpi=160, bbox_inches="tight")
    print(f"wrote data: {args.data_file}")
    print(f"wrote plot: {args.plot_file}")


if __name__ == "__main__":
    main()
