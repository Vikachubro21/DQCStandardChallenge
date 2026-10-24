"""The results file each team hands in, one per stage, and a checker for it.

Format (JSON)::

    {
      "team_id": "quantum_cats",
      "stage": "stage1",
      "points": [
        {"time": 0.25, "value": 0.81, "uncertainty": 0.02, "job_ids": ["..."]},
        ...
      ],
      "method": "Two or three sentences: circuits, mitigation, how uncertainty was estimated."
    }

``value`` is your best estimate of the observable at ``time``; ``uncertainty``
is your honest one-standard-deviation error bar on that estimate, including
everything you know about (shot noise, Trotter error, mitigation bias), not just
shot noise. ``job_ids`` are the IonQ jobs the value was computed from.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .problems import get_problem


def write_results(path: str | Path, team_id: str, stage: str, points: list[dict], method: str) -> dict:
    data = {"team_id": team_id, "stage": stage, "points": points, "method": method}
    check_results(data)
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def load_results(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    check_results(data)
    return data


def check_results(data: dict) -> None:
    """Raise ``ValueError`` if a results file does not follow the format."""
    for field in ("team_id", "stage", "points", "method"):
        if field not in data:
            raise ValueError(f"Missing field {field!r}.")
    problem = get_problem(data["stage"])
    times = []
    for point in data["points"]:
        for field in ("time", "value", "uncertainty", "job_ids"):
            if field not in point:
                raise ValueError(f"Point {point} is missing {field!r}.")
        if not all(math.isfinite(float(point[f])) for f in ("time", "value", "uncertainty")):
            raise ValueError(f"Point {point} has a non-finite number.")
        if float(point["uncertainty"]) <= 0:
            raise ValueError(f"Point at t={point['time']}: uncertainty must be positive.")
        if not point["job_ids"]:
            raise ValueError(f"Point at t={point['time']}: list the IonQ job IDs it came from.")
        times.append(round(float(point["time"]), 6))
    unknown = sorted(set(times) - {round(t, 6) for t in problem.times})
    if unknown:
        raise ValueError(f"Times {unknown} are not on the {problem.key} time grid {problem.times}.")
    if len(times) != len(set(times)):
        raise ValueError("Each time may appear only once.")


def compare_with_exact(data: dict) -> dict:
    """For stages with an exact reference: errors and how honest the error bars were.

    Returns the RMS error, the largest error, and the fraction of points whose
    exact value lies within two of the team's stated uncertainties (about 95%
    is what honest error bars give).
    """
    from .physics import exact_series

    problem = get_problem(data["stage"])
    if not problem.exact_reference:
        raise ValueError(f"{problem.key} has no exact reference.")
    times = [float(p["time"]) for p in data["points"]]
    exact = exact_series(problem, times)
    values = np.array([float(p["value"]) for p in data["points"]])
    sigmas = np.array([float(p["uncertainty"]) for p in data["points"]])
    errors = values - exact
    return {
        "times": times,
        "exact": exact.tolist(),
        "errors": errors.tolist(),
        "rms_error": float(np.sqrt(np.mean(errors ** 2))),
        "max_abs_error": float(np.max(np.abs(errors))),
        "fraction_within_2_sigma": float(np.mean(np.abs(errors) <= 2 * sigmas)),
    }
