"""Run order-2 and order-3 mitigation benchmarks for stage3_3x3 only."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dynamics_challenge import PROBLEMS, exact_series
from order_mitigation_benchmark import make_circuits, run_combined, run_raw


def save(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=600)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/tmp/opencode/order_stage3_3x3_checkpoint.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/order_stage3_3x3.json"))
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument(
        "--remove-debiasing", action="store_true",
        help="disable Pauli-frame debiasing; retain ZNE and symmetry postselection",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    problem = PROBLEMS["stage3_3x3"]
    exact = exact_series(problem)
    checkpoint = json.loads(args.checkpoint.read_text()) if args.checkpoint.exists() else {}
    requested_debiasing = not args.remove_debiasing
    # Older checkpoints predate the mode marker and were produced with
    # debiasing enabled.
    saved_config = checkpoint.get("debiasing", True)
    if checkpoint and saved_config != requested_debiasing:
        logging.info("checkpoint debiasing mode differs; starting a new mode-specific run")
        checkpoint = {}
    result = {
        "times": list(problem.times),
        "exact": exact.tolist(),
        **checkpoint,
        "debiasing": requested_debiasing,
    }

    if "order2" not in result:
        circuits = make_circuits(problem, 2)
        logging.info("stage3_3x3 order 2 raw")
        result["order2"] = {
            "raw": run_raw(problem, circuits, exact, args.shots, 8102 + args.seed_offset),
            "combined": run_combined(
                problem, circuits, exact, args.shots, 8202 + args.seed_offset,
                debias=not args.remove_debiasing,
            ),
        }
        save(args.checkpoint, result)
        logging.info("checkpointed stage3_3x3 order 2")

    if "order3" not in result:
        circuits = make_circuits(problem, 3)
        logging.info("stage3_3x3 order 3 raw")
        result["order3"] = {
            "raw": run_raw(problem, circuits, exact, args.shots, 8103 + args.seed_offset),
            "combined": run_combined(
                problem, circuits, exact, args.shots, 8203 + args.seed_offset,
                debias=not args.remove_debiasing,
            ),
        }
        save(args.checkpoint, result)
        logging.info("checkpointed stage3_3x3 order 3")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for order in ("order2", "order3"):
        print(f"{order}: raw={result[order]['raw']['rms']:.6f} "
              f"combined={result[order]['combined']['rms']:.6f}")


if __name__ == "__main__":
    main()
