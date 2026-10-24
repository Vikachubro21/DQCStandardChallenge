"""Hamiltonians, observables, exact references and baseline Trotter circuits."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .problems import Problem

MAX_EXACT_QUBITS = 20


# --------------------------------------------------------------------------- Hamiltonian

def hamiltonian(problem: Problem):
    """Return the problem Hamiltonian as a Qiskit ``SparsePauliOp``."""
    from qiskit.quantum_info import SparsePauliOp

    n, c = problem.num_qubits, problem.couplings
    terms = []
    if problem.model == "tfim":
        terms += [("XX", [i, j], c["J"]) for i, j in problem.edges]
        terms += [("Z", [i], c["h"]) for i in range(n)]
    elif problem.model == "xxz":
        for i, j in problem.edges:
            terms += [("XX", [i, j], c["J"]), ("YY", [i, j], c["J"]), ("ZZ", [i, j], c["J"] * c["delta"])]
    else:
        raise ValueError(f"Unknown model {problem.model!r}")
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n).simplify()


# --------------------------------------------------------------------------- observables

def observable_weights(problem: Problem) -> np.ndarray:
    """Weights w_i such that the observable is (1/N) * sum_i w_i <Z_i>."""
    n = problem.num_qubits
    if problem.observable == "magnetization":
        return np.ones(n)
    if problem.observable == "staggered_magnetization":
        return np.array([(-1) ** i for i in range(n)], dtype=float)
    raise ValueError(f"Unknown observable {problem.observable!r}")


def observable_from_probabilities(problem: Problem, probabilities: np.ndarray) -> float:
    """Exact observable value from a full probability vector (index bit i = qubit i)."""
    n = problem.num_qubits
    indices = np.arange(len(probabilities))
    z = np.array([1.0 - 2.0 * ((indices >> i) & 1) for i in range(n)])   # shape (n, 2**n)
    return float(observable_weights(problem) @ (z @ probabilities) / n)


def observable_from_counts(problem: Problem, counts: Mapping[str, int]) -> tuple[float, float]:
    """Estimate the observable from measured counts. Returns ``(value, standard_error)``.

    Counts keys are Qiskit bitstrings (rightmost character = qubit 0); spaces and
    ``0x`` hex keys are accepted. The standard error is the shot-noise error only.
    """
    n = problem.num_qubits
    weights = observable_weights(problem)
    values, shots = [], []
    for key, count in counts.items():
        bits = str(key).replace(" ", "")
        if bits.startswith("0x"):
            bits = format(int(bits, 16), "b")
        bits = bits.zfill(n)
        if len(bits) != n or set(bits) - {"0", "1"}:
            raise ValueError(f"Expected {n}-bit outcomes, got {key!r}.")
        z = np.array([1.0 - 2.0 * int(bits[n - 1 - i]) for i in range(n)])
        values.append(float(weights @ z) / n)
        shots.append(int(count))
    values, shots = np.array(values), np.array(shots)
    total = shots.sum()
    if total == 0:
        raise ValueError("The counts dictionary is empty.")
    mean = float(values @ shots / total)
    variance = float(((values - mean) ** 2) @ shots / max(total - 1, 1))
    return mean, float(np.sqrt(variance / total))


# --------------------------------------------------------------------------- exact reference

def initial_statevector(problem: Problem) -> np.ndarray:
    index = sum(1 << i for i, bit in enumerate(problem.initial_bits) if bit == "1")
    psi = np.zeros(2 ** problem.num_qubits, dtype=complex)
    psi[index] = 1.0
    return psi


def exact_series(problem: Problem, times: Sequence[float] | None = None) -> np.ndarray:
    """Exact observable at each time (default: the problem's time grid)."""
    from scipy.sparse.linalg import expm_multiply

    if problem.num_qubits > MAX_EXACT_QUBITS:
        raise ValueError(
            f"{problem.key} has {problem.num_qubits} qubits; an exact reference is not available "
            "(that is the point of stage 3)."
        )
    times = np.asarray(problem.times if times is None else times, dtype=float)
    h_sparse = hamiltonian(problem).to_matrix(sparse=True)
    psi, current, out = initial_statevector(problem), 0.0, []
    for t in times:
        psi = expm_multiply(-1j * (t - current) * h_sparse, psi)
        current = t
        out.append(observable_from_probabilities(problem, np.abs(psi) ** 2))
    return np.array(out)


# --------------------------------------------------------------------------- baseline circuits

def prepare_initial_state(circuit, problem: Problem) -> None:
    for i, bit in enumerate(problem.initial_bits):
        if bit == "1":
            circuit.rx(np.pi, i)


def _xx(circuit, theta, i, j):
    circuit.rxx(theta, i, j)


def _yy(circuit, theta, i, j):
    # rz(-pi/2) maps Y to X, so YY rotation = rz(-pi/2) . rxx . rz(pi/2) on both qubits.
    circuit.rz(-np.pi / 2, [i, j])
    circuit.rxx(theta, i, j)
    circuit.rz(np.pi / 2, [i, j])


def _zz(circuit, theta, i, j):
    # ry(pi/2) maps Z to X, so ZZ rotation = ry(pi/2) . rxx . ry(-pi/2) on both qubits.
    circuit.ry(np.pi / 2, [i, j])
    circuit.rxx(theta, i, j)
    circuit.ry(-np.pi / 2, [i, j])


def _trotter_layer(circuit, problem: Problem, dt: float, field_fraction: float = 1.0, bonds: bool = True,
                   fields: bool = True) -> None:
    c = problem.couplings
    if problem.model == "tfim":
        if bonds:
            for i, j in problem.edges:
                _xx(circuit, 2 * c["J"] * dt, i, j)
        if fields:
            for q in range(problem.num_qubits):
                circuit.rz(2 * c["h"] * dt * field_fraction, q)
    else:  # xxz: XX, YY and ZZ on every bond
        for i, j in problem.edges:
            _xx(circuit, 2 * c["J"] * dt, i, j)
            _yy(circuit, 2 * c["J"] * dt, i, j)
            _zz(circuit, 2 * c["J"] * c["delta"] * dt, i, j)


def trotter_circuit(problem: Problem, time: float, steps: int, order: int = 1, measure: bool = True):
    """A plain product-formula circuit for ``exp(-i H time)`` from the initial state.

    It uses only ``rx``, ``ry``, ``rz`` and ``rxx``. It is a *baseline*: correct as
    ``steps`` grows, but nothing about it is optimized for noisy hardware.
    ``order=2`` gives the symmetric (Strang) splitting for the Ising model.
    """
    from qiskit import QuantumCircuit

    if steps < 1:
        raise ValueError("steps must be at least 1")
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    circuit = QuantumCircuit(problem.num_qubits, name=f"{problem.key}-t{time:g}-s{steps}")
    prepare_initial_state(circuit, problem)
    dt = time / steps
    for _ in range(steps):
        if order == 2 and problem.model == "tfim":
            _trotter_layer(circuit, problem, dt, field_fraction=0.5, bonds=False)
            _trotter_layer(circuit, problem, dt, fields=False)
            _trotter_layer(circuit, problem, dt, field_fraction=0.5, bonds=False)
        elif order == 2:
            _trotter_layer(circuit, problem, dt / 2)
            for i, j in reversed(problem.edges):
                c = problem.couplings
                _zz(circuit, 2 * c["J"] * c["delta"] * dt / 2, i, j)
                _yy(circuit, 2 * c["J"] * dt / 2, i, j)
                _xx(circuit, 2 * c["J"] * dt / 2, i, j)
        else:
            _trotter_layer(circuit, problem, dt)
    if measure:
        circuit.measure_all()
    return circuit


def noiseless_value(problem: Problem, circuit) -> float:
    """Observable of a circuit without noise or shot noise (up to ~20 qubits).

    Compare with ``exact_series`` to separate Trotter error from hardware error.
    """
    from qiskit.quantum_info import Statevector

    if problem.num_qubits > MAX_EXACT_QUBITS:
        raise ValueError("Too many qubits for a statevector check.")
    bare = circuit.remove_final_measurements(inplace=False)
    return observable_from_probabilities(problem, Statevector(bare).probabilities())


# --------------------------------------------------------------------------- whole time series

def steps_for_time(time: float, max_dt: float = 0.25) -> int:
    """Smallest number of Trotter steps with step size at most ``max_dt``."""
    return max(1, int(np.ceil(round(time / max_dt, 9))))


def series_circuits(problem: Problem, max_dt: float = 0.25, order: int = 1) -> list:
    """Baseline circuits for every time on the problem's grid (one circuit per time)."""
    return [trotter_circuit(problem, t, steps_for_time(t, max_dt), order) for t in problem.times]


def series_from_counts(problem: Problem, counts_list: Sequence[Mapping[str, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Turn one counts dictionary per time into ``(values, shot_noise_errors)`` arrays."""
    pairs = [observable_from_counts(problem, counts) for counts in counts_list]
    return np.array([v for v, _ in pairs]), np.array([e for _, e in pairs])


# --------------------------------------------------------------------------- local noise model

def forte_like_noise_model(two_qubit_fidelity: float = 0.9958, one_qubit_fidelity: float = 0.9999,
                           spam_fidelity: float = 0.9959):
    """A crude local Aer noise model sized from Forte 1's published median fidelities.

    Depolarizing noise on every gate plus symmetric readout error, nothing else:
    no coherent errors, crosstalk or drift. Good for rehearsing the workflow and
    for rough trade-offs (Trotter error vs. gate noise), not as a prediction of
    the real machine. Update the defaults from ``qpu.calibration()``.
    """
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    model = NoiseModel(basis_gates=["rx", "ry", "rz", "rxx"])
    # Average gate fidelity F of a d-dimensional depolarizing channel: F = 1 - lambda * (d - 1) / d.
    model.add_all_qubit_quantum_error(depolarizing_error((1 - two_qubit_fidelity) * 4 / 3, 2), ["rxx"])
    model.add_all_qubit_quantum_error(depolarizing_error((1 - one_qubit_fidelity) * 2, 1), ["rx", "ry"])
    flip = 1 - spam_fidelity
    model.add_all_qubit_readout_error(ReadoutError([[1 - flip, flip], [flip, 1 - flip]]))
    return model
