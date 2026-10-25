"""Estimate, submit, or retrieve an order-3 QPU validation run.

Configuration:

* Stage 1, Stage 2, or full 6x6 Stage 3;
* order-3 circuits;
* physical ZNE folds 1, 3 or 1, 3, and 5;
* even-parity symmetry postprocessing performed after retrieval;
* IonQ built-in debiasing disabled;
* configurable total shots per time point, split across selected folds.

No action is selected by default.  ``send`` is the only action that can
submit a paid QPU job and must be requested explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from qiskit_ionq import IonQProvider

from dynamics_challenge.ionq import estimate_cost, run_without_debiasing
from generated_validation_circuits import CIRCUITS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "qpu_order3_job.json"


def parse_folds(value: str) -> tuple[int, ...]:
    folds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if folds not in ((1, 3), (1, 3, 5)):
        raise argparse.ArgumentTypeError("folds must be exactly 1,3 or 1,3,5")
    return folds


def circuit_entries(stage: str, folds: tuple[int, ...]) -> list[dict]:
    return [
        entry for entry in CIRCUITS[f"{stage}_order3"]
        if entry["fold"] in folds
    ]


def validate_configuration(stage: str, folds: tuple[int, ...], total_shots: int,
                           entries: list[dict]) -> int:
    """Validate the exact paid-job configuration before any submission."""
    if stage != "stage3":
        raise ValueError("this paid submission must target full Stage 3")
    if folds != (1, 3):
        raise ValueError(f"expected folds (1, 3), received {folds}")
    if total_shots != 600:
        raise ValueError(f"expected 600 total shots, received {total_shots}")
    if len(entries) != 12:
        raise ValueError(f"expected 12 circuits, received {len(entries)}")
    if {entry["fold"] for entry in entries} != {1, 3}:
        raise ValueError("batch contains an unexpected ZNE fold")
    if {entry["time_index"] for entry in entries} != set(range(6)):
        raise ValueError("batch does not contain all six Stage 3 time points")
    if any(entry["circuit"].num_qubits != 36 for entry in entries):
        raise ValueError("batch contains a circuit that is not the full 36-qubit Stage 3 problem")
    if total_shots % len(folds):
        raise ValueError("total shots must divide evenly across the selected folds")
    return total_shots // len(folds)


def load_qpu():
    load_dotenv(ROOT / ".env")
    if not os.getenv("IONQ_API_KEY"):
        raise RuntimeError("IONQ_API_KEY is not configured")
    return IonQProvider(os.environ["IONQ_API_KEY"]).get_backend("qpu.forte-1")


def metadata(entries: list[dict], shots_per_fold: int) -> list[dict]:
    return [
        {
            "configuration": "order3_zne_symmetry_no_debiasing",
            "time": entry["time"],
            "time_index": entry["time_index"],
            "fold": entry["fold"],
            "shots": shots_per_fold,
        }
        for entry in entries
    ]


def estimate(stage: str, folds: tuple[int, ...], total_shots: int,
             manifest_path: Path) -> None:
    qpu = load_qpu()
    entries = circuit_entries(stage, folds)
    shots_per_fold = validate_configuration(stage, folds, total_shots, entries)
    estimates = [estimate_cost(qpu, entry["circuit"], shots_per_fold)
                 for entry in entries]
    total = sum(float(item.get("estimated_total_cost") or 0) for item in estimates)
    payload = {
        "backend": "qpu.forte-1",
        "model": "order3_zne_symmetry_no_debiasing",
        "stage": stage,
        "folds": list(folds),
        "circuits": len(entries),
        "total_shots_per_time": total_shots,
        "shots_per_fold": shots_per_fold,
        "estimated_total_cost": total,
        "unit": estimates[0].get("estimated_unit"),
        "rate_card": estimates[0].get("rate_card"),
        "estimates": estimates,
    }
    output = manifest_path.with_name(f"{stage}_qpu_order3_cost_estimate.json")
    output.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"estimated cost: ${total:,.2f}")
    print(f"wrote {output}")
    print("estimate-only request; no QPU job submitted")


def status_name(job) -> str:
    status = job.status()
    return str(getattr(status, "name", status)).upper()


def send(stage: str, folds: tuple[int, ...], total_shots: int,
         manifest_path: Path, timeout: float, poll_interval: float) -> None:
    qpu = load_qpu()
    entries = circuit_entries(stage, folds)
    shots_per_fold = validate_configuration(stage, folds, total_shots, entries)
    circuits = [entry["circuit"] for entry in entries]
    # This helper explicitly sets IonQ built-in debiasing to NO_DEBIASING.
    job = run_without_debiasing(qpu, circuits, shots_per_fold)
    manifest = {
        "job_id": job.job_id(),
        "backend": "qpu.forte-1",
        "model": "order3_zne_symmetry_no_debiasing",
        "stage": stage,
        "folds": list(folds),
        "noise_model": None,
        "circuit_count": len(circuits),
        "shots_per_fold": shots_per_fold,
        "total_shots_per_time": total_shots,
        "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metadata": metadata(entries, shots_per_fold),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    deadline = time.monotonic() + timeout
    accepted = {"INITIALIZING", "QUEUED", "RUNNING", "DONE"}
    while True:
        status = status_name(job)
        print(f"job {job.job_id()} status: {status}")
        if status in accepted:
            manifest["confirmed_status"] = status
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"job accepted; manifest written to {manifest_path}")
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("QPU job was submitted but not accepted before timeout")
        time.sleep(poll_interval)


def receive(manifest_path: Path, output_path: Path, timeout: float,
            poll_interval: float) -> None:
    manifest = json.loads(manifest_path.read_text())
    qpu = load_qpu()
    job = qpu.retrieve_job(manifest["job_id"])
    deadline = time.monotonic() + timeout
    while True:
        status = status_name(job)
        print(f"job {manifest['job_id']} status: {status}")
        if status == "DONE":
            break
        if status in {"ERROR", "FAILED", "CANCELED", "CANCELLED"}:
            raise RuntimeError(f"QPU job failed: {status}")
        if time.monotonic() >= deadline:
            raise TimeoutError("QPU job did not complete before timeout")
        time.sleep(poll_interval)
        job = qpu.retrieve_job(manifest["job_id"])
    result = job.result()
    payload = {
        **manifest,
        "received_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "counts": [dict(result.get_counts(i)) for i in range(manifest["circuit_count"])],
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=("estimate", "send", "receive"), required=True)
    parser.add_argument("--stage", choices=("stage1", "stage2", "stage3"), default="stage3")
    parser.add_argument("--folds", type=parse_folds, default=(1, 3, 5),
                        help="1,3 or 1,3,5 (default: 1,3,5)")
    parser.add_argument("--shots", type=int, default=500,
                        help="total shots per time point")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path,
                        default=Path("validation/stage3_qpu_order3_results.json"))
    parser.add_argument("--timeout", type=float, default=86400.0)
    parser.add_argument("--poll-interval", type=float, default=10.0)
    args = parser.parse_args()
    if args.action == "estimate":
        estimate(args.stage, args.folds, args.shots, args.manifest)
    elif args.action == "send":
        send(args.stage, args.folds, args.shots, args.manifest,
             args.timeout, args.poll_interval)
    else:
        receive(args.manifest, args.output, args.timeout, args.poll_interval)


if __name__ == "__main__":
    main()
