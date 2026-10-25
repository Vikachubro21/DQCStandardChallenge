"""Benchmark software debiasing for Stage 1.

The variants use Pauli-frame twirling around each RXX gate.  For a Pauli frame
P, the sign of the RXX angle is flipped when P anticommutes with XX, so

    P RXX(sign * theta) P = RXX(theta).

Every variant is therefore ideally equivalent to the same first-order Trotter
circuit.  The variants have the same RXX count as the baseline and differ only
by inexpensive one-qubit frame gates.  Total shots are held fixed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.circuit.library import RXXGate
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, coherent_unitary_error

from dynamics_challenge import PROBLEMS, exact_series, forte_like_noise_model, series_circuits
from dynamics_challenge.physics import (
    observable_from_counts,
    observable_from_probabilities,
    prepare_initial_state,
    steps_for_time,
)

LOGGER = logging.getLogger("software_debias_stage1")


def apply_pauli(circuit: QuantumCircuit, pauli: str, qubit: int) -> None:
    if pauli == "X":
        circuit.rx(np.pi, qubit)
    elif pauli == "Y":
        circuit.ry(np.pi, qubit)
    elif pauli == "Z":
        circuit.rz(np.pi, qubit)


def twirled_rxx(circuit: QuantumCircuit, theta: float, i: int, j: int, rng) -> None:
    p_i, p_j = rng.choice(np.array(["I", "X", "Y", "Z"]), size=2)
    anticommutations = int(p_i in ("Y", "Z")) + int(p_j in ("Y", "Z"))
    sign = -1 if anticommutations % 2 else 1
    if p_i != "I":
        apply_pauli(circuit, p_i, i)
    if p_j != "I":
        apply_pauli(circuit, p_j, j)
    circuit.rxx(sign * theta, i, j)
    if p_j != "I":
        apply_pauli(circuit, p_j, j)
    if p_i != "I":
        apply_pauli(circuit, p_i, i)


def twirled_tfim_circuit(problem, time: float, steps: int, rng) -> QuantumCircuit:
    dt = time / steps
    circuit = QuantumCircuit(problem.num_qubits, name=f"{problem.key}-twirled-{time:g}")
    prepare_initial_state(circuit, problem)
    for _ in range(steps):
        for i, j in problem.edges:
            twirled_rxx(circuit, 2 * problem.couplings["J"] * dt, i, j, rng)
        for q in range(problem.num_qubits):
            circuit.rz(2 * problem.couplings["h"] * dt, q)
    circuit.measure_all()
    return circuit


def merge_counts(counts_list) -> dict[str, int]:
    merged = Counter()
    for counts in counts_list:
        merged.update(counts)
    return dict(merged)


def even_parity_counts(counts: dict[str, int]) -> tuple[dict[str, int], int, int]:
    """Keep computational-basis outcomes with an even number of 1s."""
    kept = {
        key: count for key, count in counts.items()
        if sum(bit == "1" for bit in str(key).replace(" ", "")) % 2 == 0
    }
    total = sum(counts.values())
    retained = sum(kept.values())
    return kept, retained, total


def coherent_noise_model(overrotation: float = 0.01) -> NoiseModel:
    """Forte-like noise plus a coherent RXX over-rotation in radians."""
    model = NoiseModel(basis_gates=["rx", "ry", "rz", "rxx"])
    model.add_all_qubit_quantum_error(
        depolarizing_error((1 - 0.9958) * 4 / 3, 2), ["rxx"]
    )
    model.add_all_qubit_quantum_error(
        depolarizing_error((1 - 0.9999) * 2, 1), ["rx", "ry"]
    )
    model.add_all_qubit_quantum_error(
        coherent_unitary_error(Operator(RXXGate(overrotation)).data), ["rxx"]
    )
    flip = 1 - 0.9959
    model.add_all_qubit_readout_error(ReadoutError([[1 - flip, flip], [flip, 1 - flip]]))
    return model


def ideal_values(problem, circuits) -> np.ndarray:
    from qiskit.quantum_info import Statevector

    return np.array([
        observable_from_probabilities(
            problem,
            Statevector.from_instruction(c.remove_final_measurements(inplace=False)).probabilities(),
        )
        for c in circuits
    ])


def fold_circuit(circuit, factor: int):
    """Globally fold a circuit by an odd, physically valid factor."""
    if factor not in (1, 3, 5):
        raise ValueError("supported physical fold factors are 1, 3, and 5")
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


def extrapolate(values: np.ndarray, scales: list[int]) -> np.ndarray:
    """Use linear Richardson extrapolation to zero noise."""
    return np.array([
        np.polyval(np.polyfit(scales, values[:, time_index], 1), 0.0)
        for time_index in range(values.shape[1])
    ])


def run_combo(problem, exact, baseline, noise_model, shots, debias, symmetry,
              zne, seed):
    """Run one correction combination; safe to execute in a worker thread."""
    scales = [1, 3, 5] if zne else [1]
    variants_per_time = N_VARIANTS if debias else 1
    shots_each = max(1, shots // (len(scales) * variants_per_time))
    rng = np.random.default_rng(seed)
    values_by_scale = []
    retained_fractions = []
    label = "+".join([
        part for part, enabled in (
            ("ZNE", zne), ("symmetry", symmetry), ("debias", debias)
        ) if enabled
    ]) or "none"

    for scale_index, factor in enumerate(scales, start=1):
        LOGGER.info("[%s] scale %d/%d (factor=%d) starting", label, scale_index,
                    len(scales), factor)
        circuits = []
        for time_index, time in enumerate(problem.times):
            if debias:
                local_rng = np.random.default_rng(seed + 1009 * (time_index + 1))
                time_circuits = [
                    twirled_tfim_circuit(
                        problem, time, steps_for_time(time, 0.25), local_rng
                    )
                    for _ in range(N_VARIANTS)
                ]
            else:
                time_circuits = [baseline[time_index]]
            circuits.extend(fold_circuit(c, factor) for c in time_circuits)

        backend = AerSimulator(
            noise_model=noise_model,
            seed_simulator=seed + factor,
            max_parallel_threads=1,
        )
        result = backend.run(circuits, shots=shots_each).result()
        scale_values = []
        scale_retained = []
        for time_index in range(len(problem.times)):
            start = time_index * variants_per_time
            counts = [result.get_counts(i) for i in range(start, start + variants_per_time)]
            merged = merge_counts(counts)
            if symmetry:
                merged, retained, total = even_parity_counts(merged)
                scale_retained.append(retained / total)
            else:
                scale_retained.append(1.0)
            scale_values.append(observable_from_counts(problem, merged)[0])
        values_by_scale.append(scale_values)
        retained_fractions.append(scale_retained)
        LOGGER.info("[%s] scale %d/%d complete", label, scale_index, len(scales))

    values_by_scale = np.asarray(values_by_scale)
    estimates = extrapolate(values_by_scale, scales) if zne else values_by_scale[0]
    return {
        "label": label,
        "values": estimates.tolist(),
        "rms": float(np.sqrt(np.mean((estimates - exact) ** 2))),
        "retained": float(np.mean(retained_fractions)),
        "shots_used": shots_each * len(scales) * variants_per_time,
    }


def checkpoint_key(model_name: str, label: str) -> str:
    return f"{model_name}:{label}"


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("ignoring unreadable checkpoint %s", path)
        return {}


def save_checkpoint(path: Path, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(results, indent=2, sort_keys=True))
    temporary.replace(path)


N_VARIANTS = 25


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=600,
                        help="total shots per estimator/time point")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--workers", type=int, default=6,
                        help="parallel benchmark workers")
    parser.add_argument("--include-coherent", action="store_true",
                        help="also run the slower coherent-noise model")
    parser.add_argument(
        "--remove-debiasing", action="store_true",
        help="exclude all Pauli-frame debiasing variants; retain other combinations",
    )
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/tmp/opencode/software_debias_stage1_checkpoint.json"),
                        help="JSON checkpoint used to resume completed jobs")
    args = parser.parse_args()
    if args.shots < 3 * N_VARIANTS:
        raise ValueError("shots must be at least 3 * number of variants for combined ZNE")

    problem = PROBLEMS["stage1"]
    exact = exact_series(problem)
    baseline = series_circuits(problem, max_dt=0.25)
    combinations = [
        (False, False, False), (False, True, False), (True, False, False),
        (True, True, False), (False, False, True), (False, True, True),
        (True, False, True), (True, True, True),
    ]
    if args.remove_debiasing:
        combinations = [combination for combination in combinations if not combination[0]]
    print("Stage 1 combined software mitigation benchmark")
    print(f"shots/estimator/time={args.shots}, variants/time={N_VARIANTS}, "
          f"workers={args.workers}")
    print("ZNE uses physical global folding factors 1,3,5 and linear extrapolation.")
    print(f"ideal baseline RMS vs exact: "
          f"{np.sqrt(np.mean((ideal_values(problem, baseline) - exact) ** 2)):.6f}")

    jobs = []
    model_names = ("depolarizing", "coherent") if args.include_coherent else ("depolarizing",)
    for model_name in model_names:
        for index, combination in enumerate(combinations):
            jobs.append((model_name, combination,
                         args.seed + 10000 * int(model_name == "coherent") + index))

    def execute(job):
        model_name, (debias, symmetry, zne), seed = job
        noise = forte_like_noise_model() if model_name == "depolarizing" else coherent_noise_model()
        result = run_combo(problem, exact, baseline, noise, args.shots,
                           debias, symmetry, zne, seed)
        return model_name, result

    results_by_model = {model_name: {} for model_name in model_names}
    checkpoint = load_checkpoint(args.checkpoint)
    pending = []
    for model_name, (debias, symmetry, zne), seed in jobs:
        label = "+".join(part for part, enabled in (
            ("ZNE", zne), ("symmetry", symmetry), ("debias", debias)
        ) if enabled) or "none"
        saved = checkpoint.get(checkpoint_key(model_name, label))
        if saved is not None:
            results_by_model[model_name][label] = saved
        else:
            pending.append((model_name, (debias, symmetry, zne), seed))
    LOGGER.info("starting %d pending jobs with %d workers; loaded %d from %s",
                len(pending), args.workers, len(jobs) - len(pending), args.checkpoint)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(execute, job): job for job in pending}
        for completed, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            try:
                model_name, result = future.result()
            except Exception:
                LOGGER.exception("benchmark failed for job %s", job)
                raise
            results_by_model[model_name][result["label"]] = result
            checkpoint[checkpoint_key(model_name, result["label"])] = result
            save_checkpoint(args.checkpoint, checkpoint)
            LOGGER.info("completed %d/%d: %s / %s", completed, len(jobs),
                        model_name, result["label"])

    for model_name in model_names:
        print(f"\n{model_name} noise")
        for result in (results_by_model[model_name][
                "+".join(part for part, enabled in (
                    ("ZNE", zne), ("symmetry", symmetry), ("debias", debias)
                ) if enabled) or "none"] for debias, symmetry, zne in combinations):
            print(f"  {result['label']:<27} RMS={result['rms']:.6f} "
                  f"retained={result['retained']:.1%} shots={result['shots_used']}")


if __name__ == "__main__":
    main()
