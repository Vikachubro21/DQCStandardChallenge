"""Compare Stage 1 unfolded order-3 data with the earlier ZNE data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dynamics_challenge import PROBLEMS, exact_series
from dynamics_challenge.physics import observable_from_counts

ROOT = Path(__file__).resolve().parent
BASE3_FILE = ROOT / "stage1_base3_results.json"
ZNE_FILE = ROOT / "stage1_order3_local_vs_ionq.json"
OUTPUT_JSON = ROOT / "stage1_base3_vs_zne_comparison.json"
OUTPUT_PNG = ROOT / "stage1_base3_vs_zne_comparison.png"


def base3_summary() -> dict:
    payload = json.loads(BASE3_FILE.read_text())
    problem = PROBLEMS["stage1"]
    values = []
    errors = []
    for counts in payload["counts"]:
        value, error = observable_from_counts(problem, counts)
        values.append(value)
        errors.append(error)
    exact = exact_series(problem)
    values_array = np.asarray(values)
    return {
        "job_id": payload["job_id"],
        "model": "base-3 order 3, no mitigation",
        "shots": payload["shots"],
        "values": values,
        "errors": errors,
        "rms_vs_exact": float(np.sqrt(np.mean((values_array - exact) ** 2))),
    }


def main() -> None:
    base3 = base3_summary()
    saved = json.loads(ZNE_FILE.read_text())
    zne = saved["ionq"]
    exact = np.asarray(saved["exact"])
    comparison = {
        "stage": "stage1",
        "exact": exact.tolist(),
        "base3_order3": base3,
        "zne_order3": {
            "job_id": saved["ionq_job_id"],
            "model": "order 3 + ZNE + symmetry",
            "shots_per_circuit": saved["shots_per_circuit"],
            "values": zne["values"],
            "errors": zne["errors"],
            "rms_vs_exact": zne["rms_vs_exact"],
        },
        "base3_minus_zne_rms": base3["rms_vs_exact"] - zne["rms_vs_exact"],
    }
    OUTPUT_JSON.write_text(json.dumps(comparison, indent=2) + "\n")

    times = list(PROBLEMS["stage1"].times)
    fig, axis = plt.subplots(figsize=(11, 6))
    axis.plot(times, exact, "k-", linewidth=2.3, label="exact")
    axis.errorbar(
        times, base3["values"], yerr=base3["errors"], fmt="s--",
        capsize=3, color="tab:orange",
        label=f"base-3 order 3 (RMS {base3['rms_vs_exact']:.4f})",
    )
    axis.errorbar(
        times, zne["values"], yerr=zne["errors"], fmt="o-",
        capsize=3, color="tab:green",
        label=f"order 3 + ZNE + symmetry (RMS {zne['rms_vs_exact']:.4f})",
    )
    axis.set_title("Stage 1: base-3 order 3 versus order 3 with ZNE")
    axis.set_xlabel("time t")
    axis.set_ylabel("magnetization")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT_JSON}")
    print(f"wrote {OUTPUT_PNG}")
    print(f"base-3 RMS: {base3['rms_vs_exact']:.6f}")
    print(f"ZNE RMS: {zne['rms_vs_exact']:.6f}")


if __name__ == "__main__":
    main()
