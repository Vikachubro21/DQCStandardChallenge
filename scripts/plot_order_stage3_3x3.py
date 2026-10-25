"""Plot Stage 3 3x3 order-1/2/3 results against the exact solution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dynamics_challenge import PROBLEMS
from order_mitigation_benchmark import make_circuits, run_raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=Path("results/order_stage3_3x3_1500.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/order_stage3_3x3_orders.png"))
    parser.add_argument("--data-output", type=Path,
                        default=Path("results/order_stage3_3x3_orders_plot.json"))
    parser.add_argument("--shots", type=int, default=1500)
    args = parser.parse_args()

    saved = json.loads(args.input.read_text())
    problem = PROBLEMS["stage3_3x3"]
    exact = saved["exact"]
    baseline = run_raw(problem, make_circuits(problem, 1), exact, args.shots, 9101)
    data = {
        "stage": "stage3_3x3",
        "shots": args.shots,
        "times": saved["times"],
        "exact": exact,
        "order1_baseline": baseline,
        "order2_raw": saved["order2"]["raw"],
        "order2_combined": saved["order2"]["combined"],
        "order3_raw": saved["order3"]["raw"],
        "order3_combined": saved["order3"]["combined"],
    }
    args.data_output.parent.mkdir(parents=True, exist_ok=True)
    args.data_output.write_text(json.dumps(data, indent=2) + "\n")

    times = data["times"]
    fig, axis = plt.subplots(figsize=(11, 6))
    axis.plot(times, data["exact"], "k-", linewidth=2.5, label="exact")
    styles = {
        "order1_baseline": ("tab:gray", "o--", "order 1 baseline"),
        "order2_raw": ("tab:orange", "^--", "order 2 raw"),
        "order2_combined": ("tab:blue", "^-", "order 2 + mitigation"),
        "order3_raw": ("tab:red", "v--", "order 3 raw"),
        "order3_combined": ("tab:green", "v-", "order 3 + mitigation"),
    }
    for key, (color, fmt, label) in styles.items():
        result = data[key]
        axis.errorbar(
            times, result["values"], yerr=result["errors"], fmt=fmt,
            color=color, capsize=3, linewidth=1.2, label=label,
        )
    axis.set_title("Stage 3 3×3: Trotter order versus exact dynamics")
    axis.set_xlabel("time t")
    axis.set_ylabel("magnetization")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote plot: {args.output}")
    print(f"wrote data: {args.data_output}")


if __name__ == "__main__":
    main()
