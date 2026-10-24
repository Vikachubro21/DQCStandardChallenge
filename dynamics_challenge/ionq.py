"""IonQ helpers: Forte 1 only, debiasing always off, every job logged.

Nothing here imports ``qiskit_ionq`` at module level; the notebook creates the
backend with your team's API key and passes it in.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

IONQ_QPU = "qpu.forte-1"
IONQ_NOISE_MODEL = "forte-1"
IONQ_SIMULATOR = "ionq_simulator"
SIMULATOR_MAX_QUBITS = 29          # IonQ's cloud simulator limit, ideal or noisy
DEFAULT_JOB_LOG = "ionq_jobs.json"


def gate_counts(circuit) -> tuple[int, int]:
    """Return ``(one_qubit_gates, two_qubit_gates)``, ignoring measurements and barriers."""
    oneq = twoq = 0
    for instruction in circuit.data:
        if instruction.operation.name in ("measure", "barrier"):
            continue
        if len(instruction.qubits) == 1:
            oneq += 1
        else:
            twoq += 1
    return oneq, twoq


def estimate_cost(backend, circuit, shots: int) -> dict:
    """IonQ's estimate for one circuit with debiasing off (free; nothing runs).

    The reply includes your organization's ``rate_card``. A multi-circuit job
    costs the sum of its circuits, and IonQ bills the circuit after its own
    compilation, so treat the number as a guide.
    """
    client = backend.client
    oneq, twoq = gate_counts(circuit)
    response = client.get_with_retry(
        client.make_path("jobs", "estimate"),
        headers=client.api_headers,
        params={
            "backend": backend.name.replace("ionq_qpu", "qpu"),
            "qubits": circuit.num_qubits,
            "shots": int(shots),
            "1q_gates": oneq,
            "2q_gates": twoq,
            "error_mitigation": "false",
        },
    )
    response.raise_for_status()
    return response.json()


def run_without_debiasing(backend, circuits, shots: int, **options):
    """Submit with IonQ's built-in error mitigation (debiasing) switched off. Debiasing is not allowed."""
    from qiskit_ionq import ErrorMitigation

    if "error_mitigation" in options:
        raise ValueError("Debiasing is not allowed in this challenge; error_mitigation is always off.")
    return backend.run(circuits, shots=int(shots), error_mitigation=ErrorMitigation.NO_DEBIASING, **options)


def record_job(job, backend_name: str, label: str, shots: int, path: str | Path = DEFAULT_JOB_LOG,
               **details) -> dict:
    """Append a submitted job to a local JSON log so it survives a kernel restart."""
    entry = {
        "job_id": job.job_id(),
        "label": label,
        "backend": backend_name,
        "shots": shots,
        "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **details,
    }
    log = load_job_log(path)
    log.append(entry)
    Path(path).write_text(json.dumps(log, indent=2), encoding="utf-8")
    return entry


def load_job_log(path: str | Path = DEFAULT_JOB_LOG) -> list[dict]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []


def save_counts(path: str | Path, counts_list, **metadata) -> dict:
    """Save raw counts (one dict per circuit) plus metadata as JSON, so data outlives the kernel."""
    record = {"saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **metadata,
              "counts": [dict(c) for c in counts_list]}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


def load_counts(path: str | Path) -> dict:
    """Load a file written by ``save_counts``; ``record['counts']`` is the list of counts dicts."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
