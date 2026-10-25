"""Compare the completed IonQ Stage 1 simulation with local Aer results."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from qiskit_aer import AerSimulator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dynamics_challenge import PROBLEMS, exact_series, forte_like_noise_model  # noqa: E402
from dynamics_challenge.physics import observable_from_counts  # noqa: E402
from order_mitigation_benchmark import extrapolation_weights  # noqa: E402
from generated_validation_circuits import CIRCUITS  # noqa: E402

INPUT = Path(__file__).resolve().parent / "stage1_order3_results.json"
LOCAL_OUTPUT = Path(__file__).resolve().parent / "stage1_order3_local_results.json"
COMPARISON_OUTPUT = Path(__file__).resolve().parent / "stage1_order3_local_vs_ionq.json"
FOLDS = (1, 3, 5)
SHOTS = 166


def merge_counts(counts: list[dict[str, int]]) -> dict[str, int]:
    merged = Counter()
    for count in counts:
        merged.update(count)
    return dict(merged)


def even_parity(counts: dict[str, int]) -> dict[str, int]:
    return {
        key: value for key, value in counts.items()
        if sum(bit == "1" for bit in key.replace(" ", "")) % 2 == 0
    }


def summarize(counts_by_circuit: list[dict[str, int]], problem) -> dict:
    values_by_fold = []
    errors_by_fold = []
    retained_by_fold = []
    for start in range(0, len(counts_by_circuit), len(FOLDS)):
        fold_counts = counts_by_circuit[start:start + len(FOLDS)]
        values = []
        errors = []
        retained = []
        for counts in fold_counts:
            filtered = even_parity(counts)
            value, error = observable_from_counts(problem, filtered)
            values.append(value)
            errors.append(error)
            retained.append(sum(filtered.values()) / sum(counts.values()))
        values_by_fold.append(values)
        errors_by_fold.append(errors)
        retained_by_fold.append(retained)
    values_by_fold = np.asarray(values_by_fold)
    errors_by_fold = np.asarray(errors_by_fold)
    retained_by_fold = np.asarray(retained_by_fold)
    weights = extrapolation_weights()
    values = values_by_fold @ weights
    errors = np.sqrt(np.sum((errors_by_fold * weights) ** 2, axis=1))
    exact = exact_series(problem)
    return {
        "fold_values": values_by_fold.tolist(),
        "fold_errors": errors_by_fold.tolist(),
        "retained_fraction": retained_by_fold.tolist(),
        "values": values.tolist(),
        "errors": errors.tolist(),
        "rms_vs_exact": float(np.sqrt(np.mean((values - exact) ** 2))),
    }


def main() -> None:
    saved = json.loads(INPUT.read_text())
    problem = PROBLEMS["stage1"]
    ionq_counts = saved["counts"]

    circuits = [entry["circuit"] for entry in CIRCUITS["stage1_order3"]]
    local_result = AerSimulator(
        noise_model=forte_like_noise_model(), seed_simulator=481516,
        max_parallel_threads=1,
    ).run(circuits, shots=SHOTS).result()
    local_counts = [local_result.get_counts(index) for index in range(len(circuits))]

    ionq_summary = summarize(ionq_counts, problem)
    local_summary = summarize(local_counts, problem)
    local_payload = {
        "backend": "local_aer_forte_like",
        "shots_per_circuit": SHOTS,
        "circuit_count": len(circuits),
        **local_summary,
        "counts": [dict(counts) for counts in local_counts],
    }
    LOCAL_OUTPUT.write_text(json.dumps(local_payload, indent=2) + "\n")

    ionq_values = np.asarray(ionq_summary["values"])
    local_values = np.asarray(local_summary["values"])
    comparison = {
        "ionq_job_id": saved["job_id"],
        "shots_per_circuit": SHOTS,
        "exact": exact_series(problem).tolist(),
        "ionq": ionq_summary,
        "local": local_summary,
        "ionq_minus_local": (ionq_values - local_values).tolist(),
        "ionq_vs_local_curve_rms": float(
            np.sqrt(np.mean((ionq_values - local_values) ** 2))
        ),
    }
    COMPARISON_OUTPUT.write_text(json.dumps(comparison, indent=2) + "\n")
    print(f"wrote {LOCAL_OUTPUT}")
    print(f"wrote {COMPARISON_OUTPUT}")
    print(f"IonQ RMS vs exact: {ionq_summary['rms_vs_exact']:.6f}")
    print(f"Local RMS vs exact: {local_summary['rms_vs_exact']:.6f}")
    print(f"IonQ/local curve RMS difference: {comparison['ionq_vs_local_curve_rms']:.6f}")


if __name__ == "__main__":
    main()
