"""Run only ZNE + parity symmetry postselection + Pauli-frame debiasing.

The benchmark covers Stage 1, Stage 2, and the exact 3x3/4x4 Stage 3
rehearsals.  The 6x6 Stage 3 problem is intentionally excluded because no
exact local reference is available and dense simulation is not practical.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from dynamics_challenge import PROBLEMS, exact_series, forte_like_noise_model, series_circuits
from dynamics_challenge.physics import observable_from_counts, prepare_initial_state, steps_for_time

LOGGER = logging.getLogger("combined_zne_stages")
N_VARIANTS = 25
FOLD_FACTORS = (1, 3, 5)


def apply_pauli(circuit: QuantumCircuit, pauli: str, qubit: int) -> None:
    if pauli == "X":
        circuit.rx(np.pi, qubit)
    elif pauli == "Y":
        circuit.ry(np.pi, qubit)
    elif pauli == "Z":
        circuit.rz(np.pi, qubit)


def twirled_rxx(circuit: QuantumCircuit, theta: float, i: int, j: int, rng) -> None:
    p_i, p_j = rng.choice(np.array(["I", "X", "Y", "Z"]), size=2)
    flips = int(p_i in ("Y", "Z")) + int(p_j in ("Y", "Z"))
    sign = -1 if flips % 2 else 1
    if p_i != "I":
        apply_pauli(circuit, p_i, i)
    if p_j != "I":
        apply_pauli(circuit, p_j, j)
    circuit.rxx(sign * theta, i, j)
    if p_j != "I":
        apply_pauli(circuit, p_j, j)
    if p_i != "I":
        apply_pauli(circuit, p_i, i)


def twirled_xx(circuit, theta: float, i: int, j: int, rng) -> None:
    twirled_rxx(circuit, theta, i, j, rng)


def twirled_yy(circuit, theta: float, i: int, j: int, rng) -> None:
    circuit.rz(-np.pi / 2, [i, j])
    twirled_rxx(circuit, theta, i, j, rng)
    circuit.rz(np.pi / 2, [i, j])


def twirled_zz(circuit, theta: float, i: int, j: int, rng) -> None:
    circuit.ry(np.pi / 2, [i, j])
    twirled_rxx(circuit, theta, i, j, rng)
    circuit.ry(-np.pi / 2, [i, j])


def twirled_circuit(problem, time: float, steps: int, rng) -> QuantumCircuit:
    dt = time / steps
    c = problem.couplings
    circuit = QuantumCircuit(problem.num_qubits, name=f"{problem.key}-debiased-{time:g}")
    prepare_initial_state(circuit, problem)
    for _ in range(steps):
        if problem.model == "tfim":
            for i, j in problem.edges:
                twirled_xx(circuit, 2 * c["J"] * dt, i, j, rng)
            for q in range(problem.num_qubits):
                circuit.rz(2 * c["h"] * dt, q)
        else:
            for i, j in problem.edges:
                twirled_xx(circuit, 2 * c["J"] * dt, i, j, rng)
                twirled_yy(circuit, 2 * c["J"] * dt, i, j, rng)
                twirled_zz(circuit, 2 * c["J"] * c["delta"] * dt, i, j, rng)
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


def merge_counts(counts_list) -> dict[str, int]:
    merged = Counter()
    for counts in counts_list:
        merged.update(counts)
    return dict(merged)


def even_parity_counts(counts: dict[str, int]) -> tuple[dict[str, int], int, int]:
    kept = {
        key: count for key, count in counts.items()
        if sum(bit == "1" for bit in str(key).replace(" ", "")) % 2 == 0
    }
    return kept, sum(kept.values()), sum(counts.values())


def intercept_weights() -> np.ndarray:
    """Weights for the linear fit at noise scale zero using 1,3,5."""
    return np.asarray([
        np.polyfit(FOLD_FACTORS, np.eye(3)[index], 1)[1]
        for index in range(3)
    ])


def run_stage(key: str, total_shots: int, seed: int, debias: bool = True) -> dict:
    problem = PROBLEMS[key]
    exact = exact_series(problem)
    baseline = series_circuits(problem, max_dt=0.25)
    baseline_backend = AerSimulator(
        noise_model=forte_like_noise_model(), seed_simulator=seed, max_parallel_threads=1
    )
    baseline_result = baseline_backend.run(baseline, shots=total_shots).result()
    baseline_values, baseline_errors = [], []
    for index in range(len(baseline)):
        value, error = observable_from_counts(problem, baseline_result.get_counts(index))
        baseline_values.append(value)
        baseline_errors.append(error)

    scale_values, scale_errors, retained = [], [], []
    variants_per_time = N_VARIANTS if debias else 1
    shots_per_circuit = total_shots // (len(FOLD_FACTORS) * variants_per_time)
    for factor_index, factor in enumerate(FOLD_FACTORS):
        circuits = []
        for time_index, time in enumerate(problem.times):
            rng = np.random.default_rng(seed + 1009 * (time_index + 1))
            source_circuits = (
                [
                    twirled_circuit(problem, time, steps_for_time(time, 0.25), rng)
                    for _ in range(N_VARIANTS)
                ]
                if debias else [
                    baseline[time_index]
                ]
            )
            circuits.extend(fold_circuit(source, factor) for source in source_circuits)
        LOGGER.info("%s factor %d/3 running", key, factor_index + 1)
        result = AerSimulator(
            noise_model=forte_like_noise_model(),
            seed_simulator=seed + factor,
            max_parallel_threads=1,
        ).run(circuits, shots=shots_per_circuit).result()
        values, errors, retained_scale = [], [], []
        for time_index in range(len(problem.times)):
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

    weights = intercept_weights()
    scale_values = np.asarray(scale_values)
    scale_errors = np.asarray(scale_errors)
    combined_values = weights @ scale_values
    combined_errors = np.sqrt(np.sum((weights[:, None] * scale_errors) ** 2, axis=0))
    result = {
        "key": key,
        "times": list(problem.times),
        "exact": exact.tolist(),
        "baseline": baseline_values,
        "baseline_error": baseline_errors,
        "combined": combined_values.tolist(),
        "combined_error": combined_errors.tolist(),
        "retained_fraction": np.mean(np.asarray(retained), axis=0).tolist(),
        "baseline_rms": float(np.sqrt(np.mean((np.asarray(baseline_values) - exact) ** 2))),
        "combined_rms": float(np.sqrt(np.mean((combined_values - exact) ** 2))),
        "shots_per_debiased_circuit": shots_per_circuit,
        "debiasing": debias,
        "variants_per_time": variants_per_time,
    }
    LOGGER.info("%s complete: baseline RMS %.6f, combined RMS %.6f", key,
                result["baseline_rms"], result["combined_rms"])
    return result


def save_checkpoint(path: Path, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(results, indent=2, sort_keys=True))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=600)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/tmp/opencode/combined_zne_stages_checkpoint.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/combined_zne_stages"))
    parser.add_argument(
        "--remove-debiasing", action="store_true",
        help="disable Pauli-frame debiasing; retain ZNE and symmetry postselection",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    keys = ("stage1", "stage2", "stage3_3x3", "stage3_4x4")
    checkpoint = json.loads(args.checkpoint.read_text()) if args.checkpoint.exists() else {}
    requested_debiasing = not args.remove_debiasing
    saved_config = checkpoint.get("__config__", {}).get("debiasing", True)
    if checkpoint and saved_config != requested_debiasing:
        LOGGER.info("checkpoint debiasing mode differs; starting a new mode-specific run")
        checkpoint = {}
    checkpoint["__config__"] = {"debiasing": requested_debiasing}
    pending = [key for key in keys if key not in checkpoint]
    LOGGER.info("starting %d pending stages with %d workers", len(pending), args.workers)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                run_stage, key, args.shots, 5000 + index, not args.remove_debiasing
            ): key
            for index, key in enumerate(pending)
        }
        for future in as_completed(futures):
            key = futures[future]
            checkpoint[key] = future.result()
            save_checkpoint(args.checkpoint, checkpoint)
            LOGGER.info("checkpointed %s", key)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key in keys:
        result = checkpoint[key]
        output = args.output_dir / f"{key}.json"
        output.write_text(json.dumps(result, indent=2) + "\n")
        fig, axis = plt.subplots(figsize=(10, 5))
        times = result["times"]
        axis.plot(times, result["exact"], "k-", linewidth=2, label="exact")
        axis.errorbar(times, result["baseline"], yerr=result["baseline_error"],
                      fmt="o--", capsize=3, label="baseline")
        axis.errorbar(times, result["combined"], yerr=result["combined_error"],
                      fmt="s-", capsize=3,
                      label="ZNE + symmetry + debiasing")
        axis.set_title(f"{key}: combined mitigation vs exact")
        axis.set_xlabel("time t")
        axis.set_ylabel("observable")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(args.output_dir / f"{key}.png", dpi=160)
        plt.close(fig)
    print(f"wrote stage data and plots to {args.output_dir}")


if __name__ == "__main__":
    main()
