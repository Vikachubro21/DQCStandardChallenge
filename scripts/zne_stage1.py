"""Compare zero-noise extrapolation discretizations on Stage 1.

Two noise-scaling strategies are compared:

* physically valid global circuit folding at factors 1, 3, and 5;
* simulated gate-noise scaling at factors 1, 2, 3, and 4.

The second strategy is useful for selecting an extrapolation grid, but the
even factors are not directly implementable with ordinary unitary folding.
All estimators receive the same total shot budget per time point.
"""

from __future__ import annotations

import argparse

import numpy as np
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, coherent_unitary_error
from qiskit.circuit.library import RXXGate
from qiskit.quantum_info import Operator

from dynamics_challenge import PROBLEMS, exact_series, forte_like_noise_model, series_circuits
from dynamics_challenge.physics import observable_from_counts, series_from_counts


def scaled_noise_model(scale: float, coherent: bool = False) -> NoiseModel:
    """Scale gate noise strength while leaving readout noise unchanged."""
    twoq = min(0.999, (1 - 0.9958) * scale)
    oneq = min(0.999, (1 - 0.9999) * scale)
    model = NoiseModel(basis_gates=["rx", "ry", "rz", "rxx"])
    model.add_all_qubit_quantum_error(
        depolarizing_error(twoq * 4 / 3, 2), ["rxx"]
    )
    model.add_all_qubit_quantum_error(
        depolarizing_error(oneq * 2, 1), ["rx", "ry"]
    )
    if coherent:
        model.add_all_qubit_quantum_error(
            coherent_unitary_error(Operator(RXXGate(0.01 * scale)).data), ["rxx"]
        )
    flip = 1 - 0.9959
    model.add_all_qubit_readout_error(ReadoutError([[1 - flip, flip], [flip, 1 - flip]]))
    return model


def fold_circuit(circuit, factor: int):
    if factor not in (1, 3, 5):
        raise ValueError("unitary global folding supports factors 1, 3, and 5")
    bare = circuit.remove_final_measurements(inplace=False)
    if factor == 1:
        folded = bare
    else:
        inverse = bare.inverse()
        pieces = [bare]
        for _ in range((factor - 1) // 2):
            pieces.extend((inverse, bare))
        folded = pieces[0]
        for piece in pieces[1:]:
            folded = folded.compose(piece)
    folded.measure_all()
    return folded


def run_grid(problem, circuits, scales, total_shots, coherent=False):
    shots_each = total_shots // len(scales)
    values = []
    for index, scale in enumerate(scales):
        result = AerSimulator(
            noise_model=scaled_noise_model(scale, coherent=coherent),
            seed_simulator=100 + index,
        ).run(circuits, shots=shots_each).result()
        values.append(np.array([
            observable_from_counts(problem, result.get_counts(i))[0]
            for i in range(len(circuits))
        ]))
    return np.asarray(values)


def run_fold_grid(problem, baseline, factors, total_shots, coherent=False):
    shots_each = total_shots // len(factors)
    values = []
    for index, factor in enumerate(factors):
        circuits = [fold_circuit(c, factor) for c in baseline]
        result = AerSimulator(
            noise_model=scaled_noise_model(1.0, coherent=coherent),
            seed_simulator=200 + index,
        ).run(circuits, shots=shots_each).result()
        values.append(np.array([
            observable_from_counts(problem, result.get_counts(i))[0]
            for i in range(len(circuits))
        ]))
    return np.asarray(values)


def extrapolate(values: np.ndarray, scales, degree: int) -> np.ndarray:
    degree = min(degree, len(scales) - 1)
    return np.array([
        np.polyval(np.polyfit(scales, values[:, time_index], degree), 0.0)
        for time_index in range(values.shape[1])
    ])


def report(label, estimates, exact, candidates):
    print(f"{label:<28} RMS={np.sqrt(np.mean((estimates - exact) ** 2)):.6f}")
    candidates.append((label, estimates))


def benchmark_model(problem, exact, baseline, total_shots, coherent=False):
    name = "coherent" if coherent else "depolarizing"
    print(f"\n{name} noise")
    candidates = []

    values_135 = run_fold_grid(problem, baseline, [1, 3, 5], total_shots,
                               coherent=coherent)
    values_1234 = run_grid(problem, baseline, [1, 2, 3, 4],
                           total_shots, coherent=coherent)

    report("raw scale 1", values_135[0], exact, candidates)
    report("fold 1,3,5 linear", extrapolate(values_135, [1, 3, 5], 1), exact, candidates)
    report("fold 1,3,5 quadratic", extrapolate(values_135, [1, 3, 5], 2), exact, candidates)
    report("scale 1,2,3,4 linear", extrapolate(values_1234, [1, 2, 3, 4], 1), exact, candidates)
    report("scale 1,2,3,4 quadratic", extrapolate(values_1234, [1, 2, 3, 4], 2), exact, candidates)
    report("scale 1,2,3 quadratic", extrapolate(values_1234[:3], [1, 2, 3], 2), exact, candidates)

    # Determine which method has the smallest absolute error at each time.
    errors = np.asarray([np.abs(values - exact) for _, values in candidates])
    winners = np.argmin(errors, axis=0)
    print("winner counts:")
    for index, (label, _) in enumerate(candidates):
        print(f"  {label:<26} {int(np.sum(winners == index))}/{len(exact)} time points")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=1200,
                        help="total shots per extrapolation estimator/time point")
    args = parser.parse_args()

    problem = PROBLEMS["stage1"]
    exact = exact_series(problem)
    baseline = series_circuits(problem, max_dt=0.25)
    print("Stage 1 zero-noise extrapolation")
    print(f"total shots per estimator/time point={args.shots}")
    print("physical folding uses 1,3,5; even factors use simulated noise scaling")
    benchmark_model(problem, exact, baseline, args.shots, coherent=False)
    benchmark_model(problem, exact, baseline, args.shots, coherent=True)


if __name__ == "__main__":
    main()
