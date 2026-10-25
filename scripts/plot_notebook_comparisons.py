"""Reproduce the notebook's symmetry and ZNE comparison plots locally."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from dynamics_challenge import PROBLEMS, exact_series, forte_like_noise_model, noiseless_value
from dynamics_challenge.physics import _trotter_layer, observable_from_counts, series_circuits, series_from_counts

SHOTS = 500
OUTPUT_DIR = Path("results")


def physical_bits(key, n):
    bits = str(key).replace(" ", "")
    if bits.startswith("0x"):
        bits = format(int(bits, 16), "b")
    return bits.zfill(n)[::-1]


def even_parity_counts(problem, counts):
    return {
        key: int(count) for key, count in counts.items()
        if physical_bits(key, problem.num_qubits).count("1") % 2 == 0
    }


def merged_second_order(problem, time, max_dt=0.25):
    steps = max(1, int(np.ceil(round(time / max_dt, 9))))
    dt = time / steps
    circuit = QuantumCircuit(problem.num_qubits)
    from dynamics_challenge.physics import prepare_initial_state
    prepare_initial_state(circuit, problem)
    _trotter_layer(circuit, problem, dt, field_fraction=0.5, bonds=False)
    for step in range(steps):
        _trotter_layer(circuit, problem, dt, fields=False)
        _trotter_layer(circuit, problem, dt,
                       field_fraction=0.5 if step == steps - 1 else 1.0,
                       bonds=False)
    circuit.measure_all()
    return circuit


def ruth_third_order(problem, time, max_dt=0.25):
    steps = max(1, int(np.ceil(round(time / max_dt, 9))))
    dt = time / steps
    coefficients = ((7 / 24, 2 / 3), (3 / 4, -2 / 3), (-1 / 24, 1.0))
    circuit = QuantumCircuit(problem.num_qubits)
    from dynamics_challenge.physics import prepare_initial_state
    prepare_initial_state(circuit, problem)
    for _ in range(steps):
        for field_fraction, bond_fraction in coefficients:
            _trotter_layer(circuit, problem, field_fraction * dt, bonds=False)
            _trotter_layer(circuit, problem, bond_fraction * dt, fields=False)
    circuit.measure_all()
    return circuit


def fold_rxx(circuit, scale):
    if scale < 1 or scale % 2 == 0:
        raise ValueError("scale must be a positive odd integer")
    folded = circuit.copy_empty_like()
    repetitions = (scale - 1) // 2
    for instruction in circuit.data:
        operation, qargs, cargs = instruction.operation, instruction.qubits, instruction.clbits
        if operation.name == "rxx":
            folded.append(operation, qargs, cargs)
            for _ in range(repetitions):
                folded.append(operation.inverse(), qargs, cargs)
                folded.append(operation, qargs, cargs)
        else:
            folded.append(operation, qargs, cargs)
    return folded


def run_counts(circuits, seed):
    result = AerSimulator(
        noise_model=forte_like_noise_model(), seed_simulator=seed,
    ).run(circuits, shots=SHOTS).result()
    return [result.get_counts(i) for i in range(len(circuits))]


def make_symmetry_plot():
    plot_data = {}
    p1 = PROBLEMS["stage1"]
    stage1_circuits = series_circuits(p1, max_dt=0.25)
    counts = run_counts(stage1_circuits, 7)
    filtered = [even_parity_counts(p1, item) for item in counts]
    raw, raw_errors = series_from_counts(p1, counts)
    parity, parity_errors = series_from_counts(p1, filtered)
    plot_data["Stage 1"] = (p1.times, raw, raw_errors, parity, parity_errors, exact_series(p1))

    for key, seed in (("stage2", 8), ("stage3_3x3", 9), ("stage3_4x4", 10)):
        problem = PROBLEMS[key]
        counts = run_counts(series_circuits(problem, max_dt=0.25), seed)
        filtered = [even_parity_counts(problem, item) for item in counts]
        raw, raw_errors = series_from_counts(problem, counts)
        parity, parity_errors = series_from_counts(problem, filtered)
        plot_data[key] = (problem.times, raw, raw_errors, parity, parity_errors,
                          exact_series(problem))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    for axis, (label, (times, raw, raw_errors, parity, parity_errors, exact)) in zip(
            axes.flat, plot_data.items()):
        axis.errorbar(times, raw, yerr=raw_errors, fmt="o-", color="C0",
                      label="raw / symmetry not applied")
        axis.errorbar(times, parity, yerr=parity_errors, fmt="s--", color="C3",
                      label="even-parity postselected")
        axis.plot(times, exact, ":", color="0.2", label="exact")
        axis.set_title(label)
        axis.set_xlabel("time t")
        axis.set_ylabel("observable")
        axis.legend(fontsize=8)
    fig.suptitle("Symmetry postselection comparison")
    fig.tight_layout()
    path = OUTPUT_DIR / "notebook_symmetry_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def make_zne_plot():
    problem = PROBLEMS["stage1"]
    baseline = series_circuits(problem, max_dt=0.25)
    exact = exact_series(problem)
    scales = (1, 3, 5)
    values = []
    for scale in scales:
        counts = run_counts([fold_rxx(circuit, scale) for circuit in baseline], 7 + scale)
        values.append(series_from_counts(problem, counts)[0])
    values = np.asarray(values)
    linear = np.asarray([np.polyfit(scales, values[:, i], 1)[-1]
                         for i in range(len(problem.times))])
    quadratic = np.asarray([np.polyfit(scales, values[:, i], 2)[-1]
                            for i in range(len(problem.times))])
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for scale, estimate in zip(scales, values):
        axes[0].plot(problem.times, estimate, "o-", label=f"{scale}x folded")
    axes[0].plot(problem.times, exact, "k:", label="exact")
    axes[0].set_title("Baseline at increasing noise scales")
    axes[0].set_xlabel("time t")
    axes[0].set_ylabel("m_z")
    axes[0].legend(fontsize=8)
    axes[1].plot(problem.times, values[0], "o-", label="baseline")
    axes[1].plot(problem.times, linear, "s--", label="linear ZNE")
    axes[1].plot(problem.times, quadratic, "d--", label="quadratic ZNE")
    axes[1].plot(problem.times, exact, "k:", label="exact")
    axes[1].set_title("Zero-noise extrapolation")
    axes[1].set_xlabel("time t")
    axes[1].set_ylabel("m_z")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = OUTPUT_DIR / "notebook_zne_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_eight_choice_table():
    source = json.loads((OUTPUT_DIR / "software_debias_stage1_combined.json").read_text())
    rows = []
    for noise_model, methods in sorted(source["models"].items()):
        for method, result in sorted(methods.items()):
            rows.append({"noise_model": noise_model, "method": method,
                         "rms": result["rms"]})
    csv_path = OUTPUT_DIR / "stage1_all_8_choices_rms.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("noise_model", "method", "rms"))
        writer.writeheader()
        writer.writerows(rows)
    md_path = OUTPUT_DIR / "stage1_all_8_choices_rms.md"
    lines = ["| Noise model | Method | RMS |", "|---|---|---:|"]
    lines.extend(f"| {row['noise_model']} | {row['method']} | {row['rms']:.6f} |"
                 for row in rows)
    md_path.write_text("\n".join(lines) + "\n")
    return csv_path, md_path


if __name__ == "__main__":
    print(f"wrote {make_symmetry_plot()}")
    print(f"wrote {make_zne_plot()}")
    print("wrote", *write_eight_choice_table())
