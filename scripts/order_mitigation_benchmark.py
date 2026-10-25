"""Benchmark order-2/order-3 Trotterization with and without mitigation.

For each exact-reference problem this reports four requested methods:

* order 2, no mitigation;
* order 2 + ZNE + even-parity postselection + Pauli-frame debiasing;
* order 3, no mitigation;
* order 3 + the same combined mitigation.

The order-1 baseline is also reported as a reference.  No plots are created.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from dynamics_challenge import PROBLEMS, exact_series, forte_like_noise_model, series_circuits
from dynamics_challenge.physics import (
    _trotter_layer,
    observable_from_counts,
    prepare_initial_state,
    steps_for_time,
)

LOGGER = logging.getLogger("order_mitigation_benchmark")
N_VARIANTS = 25
FOLDS = (1, 3, 5)
STAGES = ("stage1", "stage2", "stage3_3x3", "stage3_4x4")


def ruth_third_order(problem, time: float, max_dt: float = 0.25) -> QuantumCircuit:
    steps = steps_for_time(time, max_dt)
    dt = time / steps
    coefficients = ((7 / 24, 2 / 3), (3 / 4, -2 / 3), (-1 / 24, 1.0))
    circuit = QuantumCircuit(problem.num_qubits, name=f"{problem.key}-ruth3-{time:g}")
    prepare_initial_state(circuit, problem)
    for _ in range(steps):
        for field_fraction, bond_fraction in coefficients:
            _trotter_layer(circuit, problem, field_fraction * dt, bonds=False)
            _trotter_layer(circuit, problem, bond_fraction * dt, fields=False)
    circuit.measure_all()
    return circuit


def fold_circuit(circuit, factor: int):
    bare = circuit.remove_final_measurements(inplace=False)
    if factor == 1:
        folded = bare
    else:
        inverse = bare.inverse()
        folded = bare
        for _ in range((factor - 1) // 2):
            folded = folded.compose(inverse).compose(bare)
    folded.measure_all()
    return folded


def apply_pauli(circuit, pauli: str, qubit: int) -> None:
    if pauli == "X":
        circuit.rx(np.pi, qubit)
    elif pauli == "Y":
        circuit.ry(np.pi, qubit)
    elif pauli == "Z":
        circuit.rz(np.pi, qubit)


def twirl_circuit(circuit, rng) -> QuantumCircuit:
    """Apply exact Pauli-frame twirls to every RXX in an existing circuit."""
    source = circuit.remove_final_measurements(inplace=False)
    output = QuantumCircuit(source.num_qubits)
    for instruction in source.data:
        operation = instruction.operation
        qubits = [source.find_bit(bit).index for bit in instruction.qubits]
        if operation.name != "rxx":
            output.append(operation, qubits)
            continue
        i, j = qubits
        p_i, p_j = rng.choice(np.array(["I", "X", "Y", "Z"]), size=2)
        flips = int(p_i in ("Y", "Z")) + int(p_j in ("Y", "Z"))
        sign = -1 if flips % 2 else 1
        if p_i != "I":
            apply_pauli(output, p_i, i)
        if p_j != "I":
            apply_pauli(output, p_j, j)
        output.rxx(sign * float(operation.params[0]), i, j)
        if p_j != "I":
            apply_pauli(output, p_j, j)
        if p_i != "I":
            apply_pauli(output, p_i, i)
    output.measure_all()
    return output


def merge_counts(counts_list) -> dict[str, int]:
    merged = Counter()
    for counts in counts_list:
        merged.update(counts)
    return dict(merged)


def even_parity_counts(counts):
    kept = {
        key: count for key, count in counts.items()
        if sum(bit == "1" for bit in str(key).replace(" ", "")) % 2 == 0
    }
    return kept, sum(kept.values()), sum(counts.values())


def extrapolation_weights() -> np.ndarray:
    return np.asarray([np.polyfit(FOLDS, np.eye(3)[i], 1)[1] for i in range(3)])


def make_circuits(problem, order: int):
    if order == 1:
        return series_circuits(problem, max_dt=0.25, order=1)
    if order == 2:
        return series_circuits(problem, max_dt=0.25, order=2)
    if order == 3:
        return [ruth_third_order(problem, time) for time in problem.times]
    raise ValueError("order must be 1, 2, or 3")


def run_raw(problem, circuits, exact, shots, seed):
    result = AerSimulator(
        noise_model=forte_like_noise_model(), seed_simulator=seed,
        max_parallel_threads=1,
    ).run(circuits, shots=shots).result()
    values, errors = [], []
    for index in range(len(circuits)):
        value, error = observable_from_counts(problem, result.get_counts(index))
        values.append(value)
        errors.append(error)
    values = np.asarray(values)
    return {
        "values": values.tolist(),
        "errors": errors,
        "rms": float(np.sqrt(np.mean((values - exact) ** 2))),
        "shots": shots,
    }


def run_combined(problem, circuits, exact, shots, seed, debias=True):
    variants_per_time = N_VARIANTS if debias else 1
    shots_per = max(1, shots // (len(FOLDS) * variants_per_time))
    weights = extrapolation_weights()
    scale_values, scale_errors, retained = [], [], []
    for fold_index, factor in enumerate(FOLDS):
        variants = []
        for time_index, circuit in enumerate(circuits):
            rng = np.random.default_rng(seed + 1009 * (time_index + 1))
            source_circuits = (
                [twirl_circuit(circuit, rng) for _ in range(N_VARIANTS)]
                if debias else [circuit]
            )
            variants.extend(fold_circuit(source, factor) for source in source_circuits)
        LOGGER.info("%s order benchmark factor %d/3 running", problem.key, fold_index + 1)
        result = AerSimulator(
            noise_model=forte_like_noise_model(), seed_simulator=seed + factor,
            max_parallel_threads=1,
        ).run(variants, shots=shots_per).result()
        values, errors, retained_scale = [], [], []
        for time_index in range(len(circuits)):
            start = time_index * variants_per_time
            merged = merge_counts([
                result.get_counts(index)
                for index in range(start, start + variants_per_time)
            ])
            filtered, kept, total = even_parity_counts(merged)
            value, error = observable_from_counts(problem, filtered)
            values.append(value)
            errors.append(error)
            retained_scale.append(kept / total)
        scale_values.append(values)
        scale_errors.append(errors)
        retained.append(retained_scale)
    scale_values = np.asarray(scale_values)
    scale_errors = np.asarray(scale_errors)
    values = weights @ scale_values
    errors = np.sqrt(np.sum((weights[:, None] * scale_errors) ** 2, axis=0))
    return {
        "values": values.tolist(),
        "errors": errors.tolist(),
        "retained": float(np.mean(retained)),
        "debiasing": debias,
        "variants_per_time": variants_per_time,
        "rms": float(np.sqrt(np.mean((values - exact) ** 2))),
        "shots": shots_per * len(FOLDS) * N_VARIANTS,
    }


def run_stage(key: str, shots: int, seed: int, debias: bool = True) -> dict:
    problem = PROBLEMS[key]
    exact = exact_series(problem)
    baseline = run_raw(problem, make_circuits(problem, 1), exact, shots, seed)
    order2_circuits = make_circuits(problem, 2)
    order3_circuits = make_circuits(problem, 3)
    result = {
        "times": list(problem.times),
        "exact": exact.tolist(),
        "baseline_order1_raw": baseline,
        "order2_raw": run_raw(problem, order2_circuits, exact, shots, seed + 20),
        "order2_combined": run_combined(
            problem, order2_circuits, exact, shots, seed + 40, debias=debias
        ),
        "order3_raw": run_raw(problem, order3_circuits, exact, shots, seed + 60),
        "order3_combined": run_combined(
            problem, order3_circuits, exact, shots, seed + 80, debias=debias
        ),
    }
    LOGGER.info("%s complete: baseline %.6f, order2 raw %.6f, order2 combined %.6f, "
                "order3 raw %.6f, order3 combined %.6f", key,
                baseline["rms"], result["order2_raw"]["rms"],
                result["order2_combined"]["rms"], result["order3_raw"]["rms"],
                result["order3_combined"]["rms"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=600)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/tmp/opencode/order_mitigation_checkpoint.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/order_mitigation_benchmark.json"))
    parser.add_argument(
        "--remove-debiasing", action="store_true",
        help="disable Pauli-frame debiasing; retain ZNE and symmetry postselection",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    checkpoint = json.loads(args.checkpoint.read_text()) if args.checkpoint.exists() else {}
    requested_debiasing = not args.remove_debiasing
    saved_config = checkpoint.get("__config__", {}).get("debiasing", True)
    if checkpoint and saved_config != requested_debiasing:
        LOGGER.info("checkpoint debiasing mode differs; starting a new mode-specific run")
        checkpoint = {}
    checkpoint["__config__"] = {"debiasing": requested_debiasing}
    pending = [key for key in STAGES if key not in checkpoint]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                run_stage, key, args.shots, 7000 + index, not args.remove_debiasing
            ): key
            for index, key in enumerate(pending)
        }
        for future in as_completed(futures):
            key = futures[future]
            checkpoint[key] = future.result()
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            args.checkpoint.write_text(json.dumps(checkpoint, indent=2))
            LOGGER.info("checkpointed %s", key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(checkpoint, indent=2) + "\n")
    print(f"wrote {args.output}; no plots generated")
    for key in STAGES:
        result = checkpoint[key]
        print(f"{key}: baseline={result['baseline_order1_raw']['rms']:.6f} "
              f"order2_raw={result['order2_raw']['rms']:.6f} "
              f"order2_combined={result['order2_combined']['rms']:.6f} "
              f"order3_raw={result['order3_raw']['rms']:.6f} "
              f"order3_combined={result['order3_combined']['rms']:.6f}")


if __name__ == "__main__":
    main()
