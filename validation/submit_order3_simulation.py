"""Safely submit the order-3 validation batch to IonQ's noisy simulator.

IMPORTANT: This script is intentionally not executed by the repository
workflow.  It performs pre-submit validation before any ``backend.run`` call.

The requested Section 5 shot counts are 500 (Stage 1), 500 (Stage 2), and
300 plus 1000 (Stage 3).  IonQ's Qiskit simulator API accepts one global
``shots`` value per batch, so those four shot counts cannot be represented by
one job.  In addition, full Stage 3 is 36 qubits while the Forte simulator
currently supports at most 29 qubits.  Consequently this script refuses to
submit the requested batch until those constraints are resolved.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from qiskit_ionq import IonQProvider

from generated_validation_circuits import CIRCUITS

ROOT = Path(__file__).resolve().parents[1]
STAGE_SHOTS = {
    "stage1_order3": (500,),
    "stage2_order3": (500,),
    "stage3_order3": (300, 1000),
}
MAX_FORTE_SIMULATOR_QUBITS = 36


def batch_payload() -> tuple[list, list[dict]]:
    """Return the requested circuits and their intended shot metadata."""
    circuits = []
    metadata = []
    for key, shot_counts in STAGE_SHOTS.items():
        for shot_count in shot_counts:
            for entry in CIRCUITS[key]:
                circuits.append(entry["circuit"])
                metadata.append({
                    "configuration": key,
                    "time": entry["time"],
                    "fold": entry["fold"],
                    "requested_shots": shot_count,
                })
    return circuits, metadata


def validate_request(circuits: list, metadata: list[dict]) -> None:
    requested_shots = sorted({item["requested_shots"] for item in metadata})
    qubit_counts = sorted({circuit.num_qubits for circuit in circuits})
    problems = []
    if len(requested_shots) != 1:
        problems.append(
            "one IonQ batch has one global shots value, but the request needs "
            f"{requested_shots}"
        )
    unsupported = [count for count in qubit_counts if count > MAX_FORTE_SIMULATOR_QUBITS]
    if unsupported:
        problems.append(
            f"Forte noisy simulation supports at most {MAX_FORTE_SIMULATOR_QUBITS} "
            f"qubits, but this batch contains {unsupported}-qubit circuits"
        )
    if problems:
        raise RuntimeError("Refusing to submit:\n- " + "\n- ".join(problems))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submit", action="store_true",
        help="actually call backend.run after all validation checks pass",
    )
    args = parser.parse_args()

    circuits, metadata = batch_payload()
    print(f"prepared {len(circuits)} circuits")
    print("requested shots: 500 (Stage 1), 500 (Stage 2), "
          "300 and 1000 (Stage 3)")
    validate_request(circuits, metadata)

    if not args.submit:
        print("validation passed; use --submit to submit the single batch")
        return

    load_dotenv(ROOT / ".env")
    if not os.getenv("IONQ_API_KEY"):
        raise RuntimeError("IONQ_API_KEY is not configured")
    provider = IonQProvider(os.environ["IONQ_API_KEY"])
    simulator = provider.get_backend("ionq_simulator")
    simulator.set_options(noise_model="forte-1")
    shots = metadata[0]["requested_shots"]
    job = simulator.run(circuits, shots=shots)
    print(f"submitted job: {job.job_id()}")


if __name__ == "__main__":
    main()
