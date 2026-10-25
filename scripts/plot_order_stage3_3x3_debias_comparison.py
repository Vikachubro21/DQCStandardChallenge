"""Compare Stage 3 3x3 order mitigation with and without debiasing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debiasing", type=Path,
        default=Path("results/order_stage3_3x3.json"),
        help="600-shot result produced with debiasing enabled",
    )
    parser.add_argument(
        "--no-debiasing", type=Path,
        default=Path("results/order_stage3_3x3_nodebias_600.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/order_stage3_3x3_debias_comparison_600.png"),
    )
    parser.add_argument(
        "--data-output", type=Path,
        default=Path("results/order_stage3_3x3_debias_comparison_600.json"),
    )
    args = parser.parse_args()

    with_debiasing = json.loads(args.debiasing.read_text())
    without_debiasing = json.loads(args.no_debiasing.read_text())
    data = {
        "stage": "stage3_3x3",
        "shots": 600,
        "times": with_debiasing["times"],
        "exact": with_debiasing["exact"],
        "order2_with_debiasing": with_debiasing["order2"]["combined"],
        "order2_without_debiasing": without_debiasing["order2"]["combined"],
        "order3_with_debiasing": with_debiasing["order3"]["combined"],
        "order3_without_debiasing": without_debiasing["order3"]["combined"],
        "rms": {
            "order2_with_debiasing": with_debiasing["order2"]["combined"]["rms"],
            "order2_without_debiasing": without_debiasing["order2"]["combined"]["rms"],
            "order3_with_debiasing": with_debiasing["order3"]["combined"]["rms"],
            "order3_without_debiasing": without_debiasing["order3"]["combined"]["rms"],
        },
    }
    args.data_output.parent.mkdir(parents=True, exist_ok=True)
    args.data_output.write_text(json.dumps(data, indent=2) + "\n")

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    styles = (
        ("with_debiasing", "tab:blue", "o-", "with debiasing"),
        ("without_debiasing", "tab:orange", "s--", "without debiasing"),
    )
    for axis, order in zip(axes, ("order2", "order3")):
        axis.plot(data["times"], data["exact"], "k-", linewidth=2.2, label="exact")
        for suffix, color, fmt, label in styles:
            result = data[f"{order}_{suffix}"]
            axis.errorbar(
                data["times"], result["values"], yerr=result["errors"],
                fmt=fmt, color=color, capsize=3, linewidth=1.2, label=label,
            )
        axis.set_title(
            f"{order.replace('order', 'Order ')}: debiasing comparison "
            f"(RMS with={data['rms'][f'{order}_with_debiasing']:.6f}, "
            f"without={data['rms'][f'{order}_without_debiasing']:.6f})"
        )
        axis.set_ylabel("magnetization")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[-1].set_xlabel("time t")
    fig.suptitle("Stage 3 3×3: ZNE + symmetry with versus without debiasing", y=0.995)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote plot: {args.output}")
    print(f"wrote data: {args.data_output}")


if __name__ == "__main__":
    main()
