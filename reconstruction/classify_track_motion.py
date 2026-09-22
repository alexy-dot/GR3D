#!/usr/bin/env python3
"""Classify tracked 3D centers relative to measured background jitter."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((float(y) - float(x)) ** 2 for x, y in zip(a, b)))


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a quantile of an empty sequence")
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def direction_consistency(centers: list[list[float]]) -> float:
    steps = [
        [float(b[i]) - float(a[i]) for i in range(3)]
        for a, b in zip(centers, centers[1:])
    ]
    net = [float(centers[-1][i]) - float(centers[0][i]) for i in range(3)]
    net_norm = math.sqrt(sum(value * value for value in net))
    if net_norm == 0:
        return 0.0
    cosines = []
    for step in steps:
        step_norm = math.sqrt(sum(value * value for value in step))
        if step_norm > 0:
            cosines.append(sum(x * y for x, y in zip(step, net)) / (step_norm * net_norm))
    return sum(cosines) / len(cosines) if cosines else 0.0


def analyze_motion(
    states: list[dict],
    background_jitter: list[float],
    thresholds: list[float] | None = None,
    min_states: int = 3,
    epsilon: float = 1e-6,
) -> dict:
    thresholds = thresholds or [2.0, 3.0, 5.0]
    valid = [state for state in states if state.get("center_xyz_median") is not None]
    valid.sort(key=lambda row: (float(row["timestamp_seconds"]), int(row["sample_index"])))
    if len(valid) < min_states:
        return {
            "motion_state": "uncertain",
            "reason": "insufficient_valid_3d_states",
            "valid_state_count": len(valid),
            "minimum_valid_states": min_states,
            "threshold_sweep": {},
        }
    if len(background_jitter) != len(valid) - 1:
        raise ValueError("background_jitter must contain one value per valid-state interval")
    if any(float(value) < 0 or not math.isfinite(float(value)) for value in background_jitter):
        raise ValueError("background_jitter values must be finite and non-negative")

    centers = [[float(value) for value in row["center_xyz_median"]] for row in valid]
    if any(len(center) != 3 or not all(math.isfinite(value) for value in center) for center in centers):
        raise ValueError("center_xyz_median must contain three finite values")
    displacements = [distance(a, b) for a, b in zip(centers, centers[1:])]
    normalized = [
        displacement / (float(jitter) + epsilon)
        for displacement, jitter in zip(displacements, background_jitter)
    ]
    consistency = direction_consistency(centers)
    median_motion = quantile(normalized, 0.5)
    p90_motion = quantile(normalized, 0.9)
    sweep = {}
    for threshold in thresholds:
        exceed_fraction = sum(value >= threshold for value in normalized) / len(normalized)
        if median_motion >= threshold and exceed_fraction >= 2 / 3 and consistency >= 0.5:
            decision = "dynamic"
        elif p90_motion < threshold / 2:
            decision = "static"
        else:
            decision = "uncertain"
        sweep[str(float(threshold))] = {
            "motion_state": decision,
            "exceed_fraction": exceed_fraction,
        }

    primary_key = "3.0" if "3.0" in sweep else str(float(thresholds[0]))
    primary = sweep[primary_key]["motion_state"]
    return {
        "motion_state": primary,
        "reason": "background_normalized_3d_motion",
        "valid_state_count": len(valid),
        "minimum_valid_states": min_states,
        "interval_displacement_model_units": displacements,
        "background_jitter_model_units": [float(value) for value in background_jitter],
        "normalized_motion": normalized,
        "median_normalized_motion": median_motion,
        "p90_normalized_motion": p90_motion,
        "direction_consistency": consistency,
        "primary_threshold": float(primary_key),
        "threshold_sweep": sweep,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON containing states and background_jitter")
    parser.add_argument("output", type=Path)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[2.0, 3.0, 5.0])
    parser.add_argument("--min-states", type=int, default=3)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = analyze_motion(
        payload["states"],
        payload["background_jitter"],
        thresholds=args.thresholds,
        min_states=args.min_states,
    )
    output = {
        "track_candidate_id": payload.get("track_candidate_id"),
        "classification": result,
        "input_provenance": payload.get("provenance", {}),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
