"""Plot completed full-Stage-3 QPU order-3 ZNE results."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

from dynamics_challenge import PROBLEMS
from dynamics_challenge.physics import observable_from_counts
from dynamics_challenge.submission import write_results

INPUT = Path(__file__).resolve().parent / "stage3_qpu_order3_folds13_600_results.json"
OUTPUT_JSON = Path(__file__).resolve().parent / "stage3_qpu_order3_folds13_600_analysis.json"
OUTPUT_PNG = Path(__file__).resolve().parent / "stage3_qpu_order3_folds13_600.png"
FORMATTED_OUTPUT = Path(__file__).resolve().parent / "stage3_qpu_order3_folds13_600_formatted.json"


def even_parity(counts):
    return {
        key: count for key, count in counts.items()
        if sum(bit == "1" for bit in str(key).replace(" ", "")) % 2 == 0
    }


def main() -> None:
    payload = json.loads(INPUT.read_text())
    problem = PROBLEMS["stage3"]
    grouped = defaultdict(dict)
    for metadata, counts in zip(payload["metadata"], payload["counts"]):
        filtered = even_parity(counts)
        value, error = observable_from_counts(problem, filtered)
        grouped[metadata["time_index"]][metadata["fold"]] = {
            "value": value,
            "error": error,
            "retained_fraction": sum(filtered.values()) / sum(counts.values()),
        }

    times = list(problem.times)
    fold_values = np.asarray([
        [grouped[index][1]["value"], grouped[index][3]["value"]]
        for index in range(len(times))
    ])
    fold_errors = np.asarray([
        [grouped[index][1]["error"], grouped[index][3]["error"]]
        for index in range(len(times))
    ])
    # Linear intercept at scale zero for scales 1 and 3: 1.5*y1 - 0.5*y3.
    weights = np.asarray([1.5, -0.5])
    zne_values = fold_values @ weights
    zne_errors = np.sqrt(np.sum((fold_errors * weights) ** 2, axis=1))
    fold1_values = fold_values[:, 0]
    fold3_values = fold_values[:, 1]
    fold1_errors = fold_errors[:, 0]
    # Display/report uncertainties conservatively at twice the shot-noise SE.
    displayed_zne_errors = 2.0 * zne_errors
    displayed_fold1_errors = 2.0 * fold1_errors
    analysis = {
        "job_id": payload["job_id"],
        "stage": "stage3",
        "order": 3,
        "folds": [1, 3],
        "shots_per_fold": payload["shots_per_fold"],
        "symmetry": "even-parity postselection",
        "times": times,
        "fold_values": fold_values.tolist(),
        "fold_errors": fold_errors.tolist(),
        "zne_weights": weights.tolist(),
        "zne_values": zne_values.tolist(),
        "zne_errors": displayed_zne_errors.tolist(),
        "fold1_values": fold1_values.tolist(),
        "fold1_errors": displayed_fold1_errors.tolist(),
        "fold3_values": fold3_values.tolist(),
        "fold3_errors": (2.0 * fold_errors[:, 1]).tolist(),
        "retained_fraction": [
            [grouped[index][fold]["retained_fraction"] for fold in (1, 3)]
            for index in range(len(times))
        ],
    }
    OUTPUT_JSON.write_text(json.dumps(analysis, indent=2) + "\n")
    write_results(
        FORMATTED_OUTPUT,
        team_id="validation",
        stage="stage3",
        points=[
            {
                "time": time,
                "value": float(value),
                "uncertainty": float(error),
                "job_ids": [payload["job_id"]],
            }
            for time, value, error in zip(times, zne_values, displayed_zne_errors)
        ],
        method=("Order-3 full Stage 3 QPU circuits with physical ZNE folds 1 and 3, "
                "even-parity symmetry postselection, and doubled propagated shot-noise "
                "uncertainties; IonQ debiasing disabled."),
    )

    dense_times = np.linspace(times[0], times[-1], 300)
    smooth = PchipInterpolator(times, zne_values)(dense_times)
    fig, axis = plt.subplots(figsize=(11, 6))
    axis.errorbar(
        times, zne_values, yerr=displayed_zne_errors, fmt="o", color="tab:blue",
        capsize=4, markersize=5, label="QPU order 3 + linear ZNE",
    )
    axis.plot(dense_times, smooth, color="tab:blue", linewidth=1.8,
              label="smoothed connecting line")
    axis.plot(times, fold1_values, "s--", color="tab:orange", markersize=4,
              label="fold 1 + symmetry (no ZNE)")
    axis.plot(times, fold3_values, "d--", color="tab:red", markersize=4,
              label="fold 3 + symmetry (no ZNE)")
    axis.set_title("Full Stage 3 6×6 QPU: order 3 with ZNE folds 1,3")
    axis.set_xlabel("time t")
    axis.set_ylabel("magnetization")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT_JSON}")
    print(f"wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
