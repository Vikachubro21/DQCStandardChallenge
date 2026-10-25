"""Compare linear ZNE using folds (1,3) versus (1,3,5) on a local stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit_aer import AerSimulator

from dynamics_challenge import PROBLEMS, exact_series, forte_like_noise_model
from dynamics_challenge.physics import observable_from_counts
from order_mitigation_benchmark import fold_circuit, make_circuits


def parity_filter(counts):
    return {
        key: count for key, count in counts.items()
        if sum(bit == "1" for bit in key.replace(" ", "")) % 2 == 0
    }


def linear_weights(scales):
    return np.asarray([np.polyfit(scales, np.eye(len(scales))[i], 1)[1]
                       for i in range(len(scales))])


def run_scale(problem, circuits, scale, shots, seed):
    folded = [fold_circuit(circuit, scale) for circuit in circuits]
    result = AerSimulator(
        noise_model=forte_like_noise_model(), seed_simulator=seed,
        max_parallel_threads=1,
    ).run(folded, shots=shots).result()
    values, errors, retained = [], [], []
    for index in range(len(folded)):
        counts = result.get_counts(index)
        filtered = parity_filter(counts)
        value, error = observable_from_counts(problem, filtered)
        values.append(value)
        errors.append(error)
        retained.append(sum(filtered.values()) / sum(counts.values()))
    return np.asarray(values), np.asarray(errors), np.asarray(retained)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=600)
    parser.add_argument(
        "--stage", choices=("stage1", "stage3_3x3", "stage3_4x4"),
        default="stage3_3x3",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    args = parser.parse_args()

    problem = PROBLEMS[args.stage]
    output = args.output or Path(f"results/zne_{args.stage}_fold_comparison.json")
    plot = args.plot or Path(f"results/zne_{args.stage}_fold_comparison.png")
    circuits = make_circuits(problem, 1)
    exact = exact_series(problem)
    baseline_values, baseline_errors, baseline_retained = run_scale(
        problem, circuits, 1, args.shots, 32001
    )
    results = {}
    for scales, seed in (((1, 3), 32013), ((1, 3, 5), 320135)):
        values_by_scale = []
        errors_by_scale = []
        retained_by_scale = []
        shots_each = args.shots // len(scales)
        for index, scale in enumerate(scales):
            values, errors, retained = run_scale(
                problem, circuits, scale, shots_each, seed + index
            )
            values_by_scale.append(values)
            errors_by_scale.append(errors)
            retained_by_scale.append(retained)
        values_by_scale = np.asarray(values_by_scale)
        errors_by_scale = np.asarray(errors_by_scale)
        weights = linear_weights(scales)
        estimate = weights @ values_by_scale
        estimate_error = np.sqrt(np.sum((weights[:, None] * errors_by_scale) ** 2, axis=0))
        results["folds_" + "_".join(map(str, scales))] = {
            "folds": list(scales),
            "shots_per_fold": shots_each,
            "weights": weights.tolist(),
            "fold_values": values_by_scale.tolist(),
            "fold_errors": errors_by_scale.tolist(),
            "retained_fraction": np.asarray(retained_by_scale).tolist(),
            "values": estimate.tolist(),
            "errors": estimate_error.tolist(),
            "rms_vs_exact": float(np.sqrt(np.mean((estimate - exact) ** 2))),
        }

    payload = {
        "stage": args.stage,
        "order": 1,
        "shots": args.shots,
        "debiasing": False,
        "symmetry": "even-parity postselection",
        "times": list(problem.times),
        "exact": exact.tolist(),
        "baseline": {
            "values": baseline_values.tolist(),
            "errors": baseline_errors.tolist(),
            "retained_fraction": baseline_retained.tolist(),
            "rms_vs_exact": float(np.sqrt(np.mean((baseline_values - exact) ** 2))),
        },
        **results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")

    fig, axis = plt.subplots(figsize=(11, 6))
    axis.plot(problem.times, exact, "k-", linewidth=2.3, label="exact")
    axis.errorbar(problem.times, baseline_values, yerr=baseline_errors,
                  fmt="o--", capsize=3, label=f"noisy baseline (RMS {payload['baseline']['rms_vs_exact']:.4f})")
    colors = {"folds_1_3": "tab:blue", "folds_1_3_5": "tab:green"}
    for key, result in results.items():
        axis.errorbar(problem.times, result["values"], yerr=result["errors"],
                      fmt="s-", capsize=3, color=colors[key],
                      label=f"linear ZNE folds {','.join(map(str, result['folds']))} "
                            f"(RMS {result['rms_vs_exact']:.4f})")
    axis.set_title("Stage 3 3×3: linear ZNE fold-set comparison")
    axis.set_xlabel("time t")
    axis.set_ylabel("magnetization")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")
    print(f"wrote {plot}")
    print(f"baseline RMS: {payload['baseline']['rms_vs_exact']:.6f}")
    for key, result in results.items():
        print(f"{key} RMS: {result['rms_vs_exact']:.6f}")


if __name__ == "__main__":
    main()
