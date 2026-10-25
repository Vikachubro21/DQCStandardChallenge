"""Benchmark an adaptive variational compression of the Stage 1 dynamics.

This is a small, reproducible AVQDS-style benchmark rather than a hardware
implementation of the full McLachlan measurement loop.  The adaptive ansatz
is selected against the exact Stage 1 state so that we can isolate the circuit
compression question on a laptop.  The resulting circuits are then evaluated
with the repository's ideal observable and Forte-like Aer noise model.

The ansatz is restricted to Forte's native gates: ``GPi``, ``GPi2``, and
``ZZ``.  Single-qubit gates are deliberately allowed freely; only native ZZ
gates are treated as the scarce resource.  A gate is appended only when it
improves state fidelity beyond ``--fidelity``.  The two-qubit and total-gate
limits are safety limits for the classical search.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
from qiskit_ionq import GPI2Gate, GPIGate, ZZGate
from scipy.optimize import minimize
from scipy.sparse.linalg import expm_multiply

from dynamics_challenge import PROBLEMS, exact_series
from dynamics_challenge.physics import (
    hamiltonian,
    observable_from_counts,
    observable_from_probabilities,
)


@dataclass(frozen=True)
class Operator:
    kind: str
    qubits: tuple[int, ...]


def operator_pool(problem) -> list[tuple[Operator, ...]]:
    return [
        # Native realization of an XX-like entangler: local GPi2 basis
        # changes, one Forte ZZ, and inverse basis changes.  This block is
        # still counted as exactly one scarce two-qubit gate.
        *((Operator("gpi2", (edge[0],)), Operator("gpi2", (edge[1],)),
           Operator("zz", edge), Operator("gpi2", (edge[0],)),
           Operator("gpi2", (edge[1],))) for edge in problem.edges),
        # A single GPi2 has a fixed pi/2 rotation and cannot make the small
        # rotations needed at early times.  Two native GPi2 gates with free
        # phases synthesize an arbitrary equatorial/Z rotation while staying
        # entirely within Forte's native gateset.
        *((Operator("gpi2", (q,)), Operator("gpi2", (q,)))
          for q in range(problem.num_qubits)),
        *((Operator("gpi", (q,)), Operator("gpi", (q,)))
          for q in range(problem.num_qubits)),
    ]


def append_operator(circuit: QuantumCircuit, operator: Operator, theta: float) -> None:
    if operator.kind == "zz":
        circuit.append(ZZGate(theta), list(operator.qubits))
    elif operator.kind == "gpi2":
        circuit.append(GPI2Gate(theta), list(operator.qubits))
    else:
        circuit.append(GPIGate(theta), list(operator.qubits))


def candidate_starts(block: tuple[Operator, ...], count: int) -> list[np.ndarray]:
    """Return distinct native phase starts for multi-start local search."""
    if any(operator.kind == "zz" for operator in block):
        starts = [
            [0.25, 0.25, 0.05, 0.75, 0.75],
            [0.25, 0.25, -0.05, 0.75, 0.75],
            [0.00, 0.00, 0.05, 0.50, 0.50],
            [0.50, 0.50, 0.05, 0.00, 0.00],
        ]
    elif len(block) == 2:
        starts = [[0.0, 0.5], [0.25, 0.75], [0.0, 0.25], [0.5, 0.0]]
    else:
        starts = [[0.25]]
    return [np.asarray(start, dtype=float) for start in starts[:max(1, count)]]


def score_candidate(problem, operators, parameters, target, block, start):
    trial_ops = [*operators, *block]
    return infidelity(problem, trial_ops, np.r_[parameters, start], target)


def refine_candidate(problem, operators, parameters, target, block, start):
    trial_ops = [*operators, *block]
    if any(operator.kind == "zz" for operator in block):
        def objective(theta):
            trial = start.copy()
            trial[2] = theta[0]
            return infidelity(problem, trial_ops, np.r_[parameters, trial], target)
        result = minimize(
            objective, np.array([start[2]]), method="Nelder-Mead",
            options={"maxiter": 8, "xatol": 8e-3, "fatol": 1e-6},
        )
        refined = start.copy()
        refined[2] = result.x[0]
    else:
        result = minimize(
            lambda x: infidelity(problem, trial_ops, np.r_[parameters, x], target),
            start, method="Nelder-Mead",
            options={"maxiter": 8, "xatol": 8e-3, "fatol": 1e-6},
        )
        refined = np.asarray(result.x)
    return float(result.fun), block, refined


def ansatz_circuit(problem, operators: list[Operator], parameters: np.ndarray,
                   measure: bool = False) -> QuantumCircuit:
    circuit = QuantumCircuit(problem.num_qubits)
    for operator, theta in zip(operators, parameters):
        append_operator(circuit, operator, float(theta))
    if measure:
        circuit.measure_all()
    return circuit


def ansatz_state(problem, operators: list[Operator], parameters: np.ndarray) -> np.ndarray:
    """Fast native-state evaluation used inside the classical optimizer."""
    state = np.zeros(2 ** problem.num_qubits, dtype=complex)
    state[0] = 1.0
    indices = np.arange(len(state))
    for operator, theta in zip(operators, parameters):
        if operator.kind == "zz":
            i, j = operator.qubits
            zi = 1 - 2 * ((indices >> i) & 1)
            zj = 1 - 2 * ((indices >> j) & 1)
            state *= np.exp(-1j * np.pi * theta * zi * zj)
            continue

        phi = 2 * np.pi * theta
        if operator.kind == "gpi":
            matrix = np.array([[0, np.exp(-1j * phi)],
                               [np.exp(1j * phi), 0]], dtype=complex)
        else:
            matrix = np.array([[1, -1j * np.exp(-1j * phi)],
                               [-1j * np.exp(1j * phi), 1]], dtype=complex) / np.sqrt(2)
        q = operator.qubits[0]
        stride = 1 << q
        period = stride << 1
        updated = state.copy()
        for start in range(0, len(state), period):
            lo = slice(start, start + stride)
            hi = slice(start + stride, start + period)
            updated[lo] = matrix[0, 0] * state[lo] + matrix[0, 1] * state[hi]
            updated[hi] = matrix[1, 0] * state[lo] + matrix[1, 1] * state[hi]
        state = updated
    return state


def infidelity(problem, operators: list[Operator], parameters: np.ndarray,
               target: np.ndarray) -> float:
    overlap = np.vdot(target, ansatz_state(problem, operators, parameters))
    return float(1.0 - min(1.0, abs(overlap) ** 2))


def optimize_parameters(problem, operators: list[Operator], parameters: np.ndarray,
                        target: np.ndarray, maxiter: int = 30) -> np.ndarray:
    if not operators:
        return parameters
    result = minimize(
        lambda x: infidelity(problem, operators, x, target),
        parameters,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-9, "gtol": 1e-6},
    )
    return np.asarray(result.x, dtype=float)


def adapt_to_target(problem, target: np.ndarray, pool: list[tuple[Operator, ...]],
                    fidelity: float, max_gates: int, max_two_qubit: int,
                    initial_operators: list[Operator] | None = None,
                    initial_parameters: np.ndarray | None = None,
                    refine_every: int = 4,
                    starts_per_candidate: int = 4,
                    workers: int = 1,
                    ) -> tuple[list[Operator], np.ndarray]:
    """Incrementally adapt an ansatz, reusing the previous time point.

    Candidate blocks are optimized locally.  The expensive full-ansatz
    optimization is performed only after several blocks have been added and
    once at the end.  This is substantially faster than reoptimizing every
    parameter after every candidate, while preserving the AVQDS-style
    adaptive gate selection.
    """
    operators: list[Operator] = list(initial_operators or [])
    parameters = np.array(initial_parameters if initial_parameters is not None
                          else np.zeros(0), dtype=float)
    if len(operators) != len(parameters):
        raise ValueError("initial operators and parameters must have equal length")
    # Warm starts already contain a good approximation to the preceding time
    # point.  Avoid a full optimization before the first adaptive update.
    if operators:
        parameters = optimize_parameters(problem, operators, parameters, target, maxiter=3)
    added_blocks = 0

    while len(operators) < max_gates:
        current_error = infidelity(problem, operators, parameters, target)
        if current_error <= 1.0 - fidelity:
            break

        two_qubit_count = sum(operator.kind == "zz" for operator in operators)
        candidate_pool = [
            block for block in pool
            if (not any(operator.kind == "zz" for operator in block)
                or two_qubit_count < max_two_qubit)
        ]

        jobs = []
        # Score candidates at a deterministic native starting point.  The old
        # implementation ran a nonlinear optimizer for every candidate, which
        # made a 40-ZZ run impractical.  Only the selected block is optimized.
        # Operators may be reused.  Repeated nearest-neighbour rotations are
        # essential for representing dynamics over more than one short time
        # interval; AVQDS does not impose a one-use-per-operator restriction.
        for candidate_block in candidate_pool:
            jobs.extend(
                (candidate_block, start)
                for start in candidate_starts(candidate_block, starts_per_candidate)
            )

        def score(job):
            block, start = job
            return current_error - score_candidate(
                problem, operators, parameters, target, block, start
            )

        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            improvements = list(executor.map(score, jobs))
        scored = [
            (improvement, block, start)
            for improvement, (block, start) in zip(improvements, jobs)
        ]

        # Refine only the best candidate.  Candidate scoring is deliberately
        # cheap; full nonlinear optimization at every candidate was the main
        # reason long 40-ZZ runs became impractical.
        best = None
        refine_jobs = sorted(scored, reverse=True, key=lambda row: row[0])[:3]
        def refine(row):
            _, block, start = row
            error, refined_block, refined = refine_candidate(
                problem, operators, parameters, target, block, start
            )
            return current_error - error, refined_block, refined
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            refined_results = list(executor.map(refine, refine_jobs))
        if refined_results:
            best = max(refined_results, key=lambda row: row[0])

        if best is None or best[0] <= 1e-8:
            break
        _, candidate_block, theta = best
        operators.extend(candidate_block)
        parameters = np.r_[parameters, theta]
        added_blocks += 1
        if added_blocks % refine_every == 0:
            parameters = optimize_parameters(problem, operators, parameters, target, maxiter=8)

    parameters = optimize_parameters(problem, operators, parameters, target, maxiter=12)
    return operators, parameters


def native_noise_model(two_qubit_fidelity: float = 0.9958,
                       one_qubit_fidelity: float = 0.9999,
                       spam_fidelity: float = 0.9959) -> NoiseModel:
    """Forte-like noise for native circuits after decomposition for Aer."""
    model = NoiseModel(basis_gates=["u", "rzz"])
    model.add_all_qubit_quantum_error(
        depolarizing_error((1 - two_qubit_fidelity) * 4 / 3, 2), ["rzz"]
    )
    model.add_all_qubit_quantum_error(
        depolarizing_error((1 - one_qubit_fidelity) * 2, 1), ["u"]
    )
    flip = 1 - spam_fidelity
    model.add_all_qubit_readout_error(
        ReadoutError([[1 - flip, flip], [flip, 1 - flip]])
    )
    return model


def exact_state_series(problem) -> list[np.ndarray]:
    sparse_h = hamiltonian(problem).to_matrix(sparse=True)
    state = np.zeros(2 ** problem.num_qubits, dtype=complex)
    state[0] = 1.0
    states = []
    current = 0.0
    for time in problem.times:
        state = expm_multiply(-1j * (time - current) * sparse_h, state)
        states.append(np.asarray(state))
        current = time
    return states


def rms(values: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((values - reference) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fidelity", type=float, default=0.99,
                        help="target algorithmic state fidelity, not hardware gate fidelity")
    parser.add_argument("--max-two-qubit-final", type=int, default=40,
                        help="ZZ budget at the final time point")
    parser.add_argument("--extra-single-qubit", type=int, default=240,
                        help="additional native one-qubit gates allowed per time")
    parser.add_argument("--starts-per-candidate", type=int, default=4,
                        help="native phase starts tested for each candidate")
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel candidate/refinement workers")
    parser.add_argument("--shots", type=int, default=500)
    args = parser.parse_args()

    problem = PROBLEMS["stage1"]
    exact_values = exact_series(problem)
    exact_states = exact_state_series(problem)
    pool = operator_pool(problem)

    adaptive_circuits = []
    adaptive_values = []
    gate_rows = []
    previous_operators: list[Operator] = []
    previous_parameters = np.zeros(0, dtype=float)
    for time, target in zip(problem.times, exact_states):
        max_two_qubit = max(
            1, int(np.ceil(args.max_two_qubit_final * time / max(problem.times)))
        )
        max_gates = 5 * max_two_qubit + args.extra_single_qubit
        operators, parameters = adapt_to_target(
            problem, target, pool, args.fidelity, max_gates, max_two_qubit,
            initial_operators=previous_operators,
            initial_parameters=previous_parameters,
            starts_per_candidate=args.starts_per_candidate,
            workers=args.workers,
        )
        previous_operators, previous_parameters = operators, parameters
        circuit = ansatz_circuit(problem, operators, parameters, measure=True)
        adaptive_circuits.append(circuit)
        value = observable_from_probabilities(problem, Statevector.from_instruction(
                circuit.remove_final_measurements(inplace=False)
            ).probabilities())
        adaptive_values.append(value)
        gate_rows.append((time, len(operators), circuit.count_ops().get("zz", 0), value))
    adaptive_ideal = np.asarray(adaptive_values)

    # Aer does not know IonQ's native instruction names as noise locations, so
    # simulate a decomposed copy while retaining native gates in the benchmark
    # circuits and gate counts.  The decomposed ZZ remains one noisy RZZ event.
    noisy_adaptive = [c.decompose(reps=1) for c in adaptive_circuits]
    adaptive_result = AerSimulator(
        noise_model=native_noise_model(), seed_simulator=20
    ).run(noisy_adaptive, shots=args.shots).result()
    adaptive_noisy = np.array([
        observable_from_counts(problem, adaptive_result.get_counts(i))[0]
        for i in range(len(adaptive_circuits))
    ])

    adaptive_twoq = sum(row[2] for row in gate_rows)
    print("Stage 1 native AVQDS-style adaptive values vs exact")
    print(f"target state fidelity={args.fidelity:.6f}, final ZZ budget={args.max_two_qubit_final}, "
          f"extra 1q budget={args.extra_single_qubit}, shots={args.shots}")
    print("time   ansatz-gates   ZZ-gates   exact       ideal       ideal-error   noisy       noisy-error")
    for (time, total, twoq, ideal), noisy, exact in zip(gate_rows, adaptive_noisy, exact_values):
        print(f"{time:4.2f}   {total:12d}   {twoq:8d}   {exact: .6f}   {ideal: .6f} "
              f"{ideal - exact: .6f}   {noisy: .6f}   {noisy - exact: .6f}")
    print()
    print(f"adaptive total native ZZ gates: {adaptive_twoq}")
    print(f"ideal RMS error vs exact: {rms(adaptive_ideal, exact_values):.6f}")
    print(f"noisy RMS error vs exact: {rms(adaptive_noisy, exact_values):.6f}")


if __name__ == "__main__":
    main()
