"""Generated offline validation circuits.

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

