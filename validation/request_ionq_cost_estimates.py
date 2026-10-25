"""Request free IonQ preflight cost estimates for validation circuits.

This script intentionally calls only ``dynamics_challenge.ionq.estimate_cost``.
It never calls ``backend.run`` and therefore never submits a job or consumes
shots.  IonQ's estimate endpoint is the same free endpoint used by the
notebook's ``FREE: what would the stage-1 baseline series cost?`` cell.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from qiskit_ionq import IonQProvider

from dynamics_challenge.ionq import estimate_cost
from generated_validation_circuits import CIRCUITS, CONFIGS

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).resolve().parent / "ionq_cost_estimates.json"
SHOT_COUNTS = {"stage1": (500,), "stage2": (500,), "stage3": (300, 1000)}
FOLDS = (1, 3, 5)


def main() -> None:
    load_dotenv(ROOT / ".env")
    if not os.getenv("IONQ_API_KEY"):
        raise RuntimeError("IONQ_API_KEY is not configured; no request was made")

    # Construction is authentication only.  No backend.run call appears in
    # this file; estimate_cost performs a GET request to jobs/estimate.
    provider = IonQProvider(os.environ["IONQ_API_KEY"])
    qpu = provider.get_backend("qpu.forte-1")
    estimates = []
    totals = defaultdict(float)
    for key, entries in CIRCUITS.items():
        stage, order = CONFIGS[key]
        for entry in entries:
            for total_shots in SHOT_COUNTS[stage]:
                shots_per_fold = total_shots // len(FOLDS)
                estimate = estimate_cost(qpu, entry["circuit"], shots_per_fold)
                row = {
                    "name": f"{key}_t{entry['time']:g}_fold{entry['fold']}",
                    "stage": stage,
                    "order": order,
                    "time": entry["time"],
                    "time_index": entry["time_index"],
                    "fold": entry["fold"],
                    "total_shots_per_time": total_shots,
                    "shots_per_fold_circuit": shots_per_fold,
                    "estimated_total_cost": estimate.get("estimated_total_cost"),
                    "estimated_unit": estimate.get("estimated_unit"),
                    "rate_card": estimate.get("rate_card"),
                }
                estimates.append(row)
                value = estimate.get("estimated_total_cost")
                if value is not None:
                    totals[(stage, order, total_shots)] += float(value)

    result = {
        "api_calls": "estimate_only",
        "job_submitted": False,
        "backend": qpu.name,
        "estimates": estimates,
        "totals": [
            {"stage": stage, "order": order, "total_shots_per_time": shots,
             "estimated_cost": round(value, 2)}
            for (stage, order, shots), value in sorted(totals.items())
        ],
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {OUTPUT}")
    print("IonQ estimate-only requests completed; no job submitted.")
    for row in result["totals"]:
        print(f"{row['stage']} order {row['order']} {row['total_shots_per_time']} shots: "
              f"${row['estimated_cost']:,.2f}")


if __name__ == "__main__":
    main()
