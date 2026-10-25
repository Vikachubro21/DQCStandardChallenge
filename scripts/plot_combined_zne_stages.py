"""Plot whichever completed combined-ZNE stage results are checkpointed."""

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
        "--checkpoint", type=Path,
        default=Path("/tmp/opencode/combined_zne_stages_checkpoint.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/combined_zne_stages"))
    args = parser.parse_args()

    results = json.loads(args.checkpoint.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key, result in sorted(results.items()):
        fig, axis = plt.subplots(figsize=(10, 5))
        axis.plot(result["times"], result["exact"], "k-", linewidth=2, label="exact")
        axis.errorbar(
            result["times"], result["baseline"], yerr=result["baseline_error"],
            fmt="o--", capsize=3, label="baseline",
        )
        axis.errorbar(
            result["times"], result["combined"], yerr=result["combined_error"],
            fmt="s-", capsize=3, label="ZNE + symmetry + debiasing",
        )
        axis.set_title(f"{key}: combined mitigation vs exact")
        axis.set_xlabel("time t")
        axis.set_ylabel("observable")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(args.output_dir / f"{key}.png", dpi=160)
        plt.close(fig)
        print(f"wrote {args.output_dir / f'{key}.png'}")


if __name__ == "__main__":
    main()
