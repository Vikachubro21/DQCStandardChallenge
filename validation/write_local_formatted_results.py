"""Convert completed local Stage 1/2 benchmark JSON to submission format."""

from __future__ import annotations

import json
from pathlib import Path

from dynamics_challenge.submission import write_results

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "results" / "combined_zne_stages"
OUTPUT_DIR = Path(__file__).resolve().parent


def convert(stage: str) -> Path:
    source = json.loads((SOURCE_DIR / f"{stage}.json").read_text())
    points = [
        {
            "time": time,
            "value": value,
            "uncertainty": max(error, 1e-12),
            "job_ids": [f"local-{stage}"],
        }
        for time, value, error in zip(
            source["times"], source["combined"], source["combined_error"]
        )
    ]
    output = OUTPUT_DIR / f"{stage}_local_formatted.json"
    write_results(
        output,
        team_id="local-validation",
        stage=stage,
        points=points,
        method=("Local forte-like simulation using ZNE folds 1,3,5, even-parity "
                "symmetry postselection, and Pauli-frame software debiasing; "
                "uncertainties are propagated postselected shot-noise errors."),
    )
    return output


if __name__ == "__main__":
    for stage_name in ("stage1", "stage2"):
        print(f"wrote {convert(stage_name)}")
