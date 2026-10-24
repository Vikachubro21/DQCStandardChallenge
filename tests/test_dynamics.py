"""Smoke tests for the standard-track helpers."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYYGate, RZZGate
from qiskit.quantum_info import Operator

from dynamics_challenge import (
    PROBLEMS,
    check_results,
    compare_with_exact,
    exact_series,
    noiseless_value,
    observable_from_counts,
    series_circuits,
    steps_for_time,
    trotter_circuit,
)
from dynamics_challenge.ionq import gate_counts, load_job_log, record_job
from dynamics_challenge.physics import _yy, _zz


def test_problem_definitions():
    assert PROBLEMS["stage1"].num_qubits == 12 and len(PROBLEMS["stage1"].edges) == 11
    assert PROBLEMS["stage3"].num_qubits == 36 and len(PROBLEMS["stage3"].edges) == 60
    assert not PROBLEMS["stage3"].exact_reference
    with pytest.raises(ValueError, match="exact reference is not available"):
        exact_series(PROBLEMS["stage3"])


@pytest.mark.parametrize("builder, gate", [(_yy, RYYGate), (_zz, RZZGate)])
def test_basis_changed_rotations(builder, gate):
    circuit = QuantumCircuit(2)
    builder(circuit, 0.7, 0, 1)
    assert Operator(circuit).equiv(Operator(gate(0.7)))


def test_initial_values_and_trotter_convergence():
    for key in ("stage1", "stage2", "stage3_3x3"):
        problem = PROBLEMS[key]
        assert exact_series(problem, [0.0])[0] == pytest.approx(1.0)
        t = problem.times[3]
        exact = exact_series(problem, [t])[0]
        coarse = abs(noiseless_value(problem, trotter_circuit(problem, t, 2)) - exact)
        fine = abs(noiseless_value(problem, trotter_circuit(problem, t, 16)) - exact)
        assert fine < coarse and fine < 0.01


def test_circuits_use_only_rx_ry_rz_rxx():
    for key in ("stage1", "stage2", "stage3"):
        for circuit in series_circuits(PROBLEMS[key]):
            assert set(circuit.count_ops()) <= {"rx", "ry", "rz", "rxx", "measure", "barrier"}
    assert steps_for_time(0.25) == 1 and steps_for_time(3.0) == 12 and steps_for_time(1.0, 0.3) == 4
    assert gate_counts(trotter_circuit(PROBLEMS["stage1"], 1.0, 4))[1] == 44


def test_observable_from_counts_bit_order():
    problem = PROBLEMS["stage2"]                       # staggered magnetization
    neel_key = problem.initial_bits[::-1]              # Qiskit keys are qubit 0 rightmost
    value, error = observable_from_counts(problem, {neel_key: 100})
    assert value == pytest.approx(1.0) and error == pytest.approx(0.0)
    value, _ = observable_from_counts(problem, {neel_key: 50, "0" * 12: 50})
    assert value == pytest.approx(0.5)


def test_results_format_and_comparison():
    problem = PROBLEMS["stage1"]
    exact = exact_series(problem)
    data = {
        "team_id": "t",
        "stage": "stage1",
        "method": "test",
        "points": [{"time": t, "value": float(v), "uncertainty": 0.01, "job_ids": ["x"]}
                   for t, v in zip(problem.times, exact)],
    }
    check_results(data)
    report = compare_with_exact(data)
    assert report["rms_error"] == pytest.approx(0.0, abs=1e-9) and report["fraction_within_2_sigma"] == 1.0
    data["points"][0]["time"] = 0.3
    with pytest.raises(ValueError, match="time grid"):
        check_results(data)


def test_job_log(tmp_path):
    class FakeJob:
        def job_id(self):
            return "abc"

    record_job(FakeJob(), "ionq_qpu.forte-1", "x", 100, path=tmp_path / "log.json", stage="stage1")
    assert load_job_log(tmp_path / "log.json")[0]["stage"] == "stage1"
