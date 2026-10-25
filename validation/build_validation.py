"""Offline validation-circuit generation and Forte cost estimation.

This file intentionally never imports qiskit_ionq, creates a provider, or
submits an estimate/job request.  It uses the Section 5 shot counts from the
standard-track notebook and the same local planning rates used previously.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dynamics_challenge import PROBLEMS  # noqa: E402
from dynamics_challenge.ionq import gate_counts  # noqa: E402
from order_mitigation_benchmark import fold_circuit, make_circuits  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent
STAGES = ("stage1", "stage2", "stage3")
ORDERS = (1, 2, 3)
FOLDS = (1, 3, 5)

# Section 5 of Standard_Track_Challenge.ipynb:
# stage 1 = 500 shots/time, stage 2 = 500 shots/time, stage 3 = 300 and 1000.
SHOT_COUNTS = {
    "stage1": (500,),
    "stage2": (500,),
    "stage3": (300, 1000),
}

# Same planning assumptions used for the earlier Forte estimates.  Actual
# IonQ estimates depend on the organization's current rate card and compiled
# gate counts, neither of which is queried here.
RATE_1Q = 0.00003
RATE_2Q = 0.00060
MINIMUM_CIRCUIT_COST = 20.0


def offline_circuit_cost(circuit, shots: int) -> float:
    oneq, twoq = gate_counts(circuit)
    gate_cost = shots * (oneq * RATE_1Q + twoq * RATE_2Q)
    return max(MINIMUM_CIRCUIT_COST, gate_cost)


def write_circuit_module(specs: list[dict]) -> Path:
    """Write a compact Python module using per-key list comprehensions."""
    path = OUTPUT_DIR / "generated_validation_circuits.py"
    source = '''"""Generated offline validation circuits.

No IonQ provider or API is used.  Import ``build_all_circuits`` to regenerate
the concrete Qiskit circuits from the recorded stage/order/fold specifications.
All circuits use ZNE folds 1, 3, 5 and no Pauli-frame debiasing.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dynamics_challenge import PROBLEMS
from order_mitigation_benchmark import fold_circuit, make_circuits

FOLDS = (1, 3, 5)
SYMMETRY_POSTPROCESSING = "even-parity postselection"
DEBIASING = False

CONFIGS = {
    "stage1_order1": ("stage1", 1),
    "stage1_order2": ("stage1", 2),
    "stage1_order3": ("stage1", 3),
    "stage2_order1": ("stage2", 1),
    "stage2_order2": ("stage2", 2),
    "stage2_order3": ("stage2", 3),
    "stage3_order1": ("stage3", 1),
    "stage3_order2": ("stage3", 2),
    "stage3_order3": ("stage3", 3),
}


def build_all_circuits():
    return {
        key: [
            {
                "time_index": time_index,
                "time": problem.times[time_index],
                "fold": fold,
                "circuit": fold_circuit(base, fold),
            }
            for time_index, base in enumerate(make_circuits(problem, order))
            for fold in FOLDS
        ]
        for key, (stage, order) in CONFIGS.items()
        for problem in (PROBLEMS[stage],)
    }


CIRCUITS = build_all_circuits()
'''
    path.write_text(source + "\n")
    return path


def build_validation() -> tuple[list[dict], dict]:
    specs = []
    costs = {}
    for stage in STAGES:
        problem = PROBLEMS[stage]
        costs[stage] = {}
        for order in ORDERS:
            base = make_circuits(problem, order)
            costs[stage][f"order{order}"] = {
                "time_points": len(base),
                "folds": list(FOLDS),
                "debiasing": False,
                "symmetry_postprocessing": "even-parity postselection",
                "shots": {},
            }
            for time_index, time in enumerate(problem.times):
                for fold in FOLDS:
                    name = f"{stage}_order{order}_t{time:g}_fold{fold}"
                    specs.append({
                        "name": name,
                        "stage": stage,
                        "order": order,
                        "time": time,
                        "time_index": time_index,
                        "fold": fold,
                    })
            for shots in SHOT_COUNTS[stage]:
                shots_per_fold = shots // len(FOLDS)
                fold_costs = []
                total_oneq = total_twoq = 0
                for fold in FOLDS:
                    for circuit in base:
                        folded = fold_circuit(circuit, fold)
                        oneq, twoq = gate_counts(folded)
                        total_oneq += oneq
                        total_twoq += twoq
                        fold_costs.append(offline_circuit_cost(folded, shots_per_fold))
                costs[stage][f"order{order}"]["shots"][str(shots)] = {
                    "shots_per_fold_circuit": shots_per_fold,
                    "circuit_count": len(base) * len(FOLDS),
                    "total_one_qubit_gates": total_oneq,
                    "total_two_qubit_gates": total_twoq,
                    "estimated_cost": round(sum(fold_costs), 2),
                }
    return specs, costs


def main() -> None:
    specs, costs = build_validation()
    circuits_file = write_circuit_module(specs)
    costs["assumptions"] = {
        "source_notebook": "notebooks/Standard_Track_Challenge.ipynb",
        "section": "5. Spending the budget wisely",
        "rate_1q": RATE_1Q,
        "rate_2q": RATE_2Q,
        "minimum_circuit_cost": MINIMUM_CIRCUIT_COST,
        "api_calls": False,
        "method": "max(minimum, shots * (1Q gates * rate_1q + 2Q gates * rate_2q))",
    }
    costs_file = OUTPUT_DIR / "validation_costs.json"
    costs_file.write_text(json.dumps(costs, indent=2) + "\n")
    print(f"wrote {circuits_file}")
    print(f"wrote {costs_file}")
    print("Offline Forte estimates; no IonQ API calls made.")
    for stage in STAGES:
        for order in ORDERS:
            entries = costs[stage][f"order{order}"]["shots"]
            formatted = ", ".join(f"{shots} shots=${value['estimated_cost']:,.2f}"
                                  for shots, value in entries.items())
            print(f"{stage} order {order}: {formatted}")


if __name__ == "__main__":
    main()
