#!/usr/bin/env python3
"""Classify tracked 3D centers against frame-paired background evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def state_audit(states: list[dict]) -> list[dict]:
    return [
        {
            "sample_index": int(state["sample_index"]),
            "source_frame_index": (
                int(state["source_frame_index"])
                if state.get("source_frame_index") is not None
                else None
            ),
            "timestamp_seconds": float(state["timestamp_seconds"]),
            "valid_3d": state.get("center_xyz_median") is not None,
            "point_count": int(state.get("point_count", 0)),
            "quality_warnings": list(state.get("quality_warnings", [])),
        }
        for state in states
    ]


def uncertain_result(
    reason: str,
    states: list[dict],
    min_states: int,
    **details: object,
) -> dict:
    audit = state_audit(states)
    return {
        "motion_state": "uncertain",
        "reason": reason,
        "valid_state_count": sum(row["valid_3d"] for row in audit),
        "total_state_count": len(audit),
        "minimum_valid_states": min_states,
        "state_evidence": audit,
        "threshold_sweep": {},
        **details,
    }


def background_intervals_from_payload(payload: dict) -> tuple[list[dict], list[str]]:
    if "background_intervals" in payload:
        intervals = payload["background_intervals"]
        if not isinstance(intervals, list):
            return [], ["background_intervals_not_a_list"]
        return intervals, []

    values = payload.get("background_jitter")
    support = payload.get("interval_support")
    if values is None and support is None:
        return [], ["background_intervals_missing"]
    if not isinstance(values, list) or not isinstance(support, list):
        return [], ["legacy_background_evidence_incomplete"]
    if len(values) != len(support):
        return [], ["legacy_background_interval_count_mismatch"]
    intervals = []
    for value, row in zip(values, support, strict=True):
        intervals.append({
            **row,
            "status": "valid",
            "background_jitter": value,
        })
    return intervals, ["legacy_background_evidence_converted"]


def analyze_motion(
    states: list[dict],
    background_intervals: list[dict],
    thresholds: list[float] | None = None,
    min_states: int = 3,
    epsilon: float = 1e-6,
    background_evidence_warnings: list[str] | None = None,
) -> dict:
    thresholds = thresholds or [2.0, 3.0, 5.0]
    evidence_warnings = list(background_evidence_warnings or [])
    ordered = sorted(
        states,
        key=lambda row: (float(row["timestamp_seconds"]), int(row["sample_index"])),
    )
    sample_indices = [int(row["sample_index"]) for row in ordered]
    if len(sample_indices) != len(set(sample_indices)):
        raise ValueError("track states contain duplicate sample_index values")

    invalid = [row for row in ordered if row.get("center_xyz_median") is None]
    if invalid:
        return uncertain_result(
            "invalid_3d_state_evidence",
            ordered,
            min_states,
            invalid_sample_indices=[int(row["sample_index"]) for row in invalid],
            invalid_state_reasons={
                str(int(row["sample_index"])): (
                    list(row.get("quality_warnings", [])) or ["missing_3d_center"]
                )
                for row in invalid
            },
            background_evidence_warnings=evidence_warnings,
        )

    if len(ordered) < min_states:
        return uncertain_result(
            "insufficient_valid_3d_states",
            ordered,
            min_states,
            background_evidence_warnings=evidence_warnings,
        )

    centers = [[float(value) for value in row["center_xyz_median"]] for row in ordered]
    if any(
        len(center) != 3 or not all(math.isfinite(value) for value in center)
        for center in centers
    ):
        raise ValueError("center_xyz_median must contain three finite values")

    interval_by_pair: dict[tuple[int, int], dict] = {}
    for interval in background_intervals:
        try:
            pair = (int(interval["from_frame"]), int(interval["to_frame"]))
        except (KeyError, TypeError, ValueError):
            return uncertain_result(
                "invalid_background_interval_evidence",
                ordered,
                min_states,
                background_evidence_warnings=evidence_warnings
                + ["interval_frame_pair_invalid"],
            )
        if pair in interval_by_pair:
            raise ValueError(f"duplicate background interval: {pair}")
        interval_by_pair[pair] = interval

    required_pairs = list(zip(sample_indices, sample_indices[1:]))
    missing_pairs = [pair for pair in required_pairs if pair not in interval_by_pair]
    if missing_pairs:
        return uncertain_result(
            "missing_background_interval_evidence",
            ordered,
            min_states,
            required_background_intervals=[list(pair) for pair in required_pairs],
            missing_background_intervals=[list(pair) for pair in missing_pairs],
            background_evidence_warnings=evidence_warnings,
        )

    used_intervals = [interval_by_pair[pair] for pair in required_pairs]
    invalid_pairs = []
    jitter = []
    for pair, interval in zip(required_pairs, used_intervals, strict=True):
        value = interval.get("background_jitter")
        if interval.get("status", "valid") != "valid":
            invalid_pairs.append(pair)
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            invalid_pairs.append(pair)
            continue
        if numeric < 0 or not math.isfinite(numeric):
            invalid_pairs.append(pair)
            continue
        jitter.append(numeric)
    if invalid_pairs:
        return uncertain_result(
            "invalid_background_interval_evidence",
            ordered,
            min_states,
            required_background_intervals=[list(pair) for pair in required_pairs],
            invalid_background_intervals=[list(pair) for pair in invalid_pairs],
            used_background_intervals=used_intervals,
            background_evidence_warnings=evidence_warnings,
        )

    displacements = [distance(a, b) for a, b in zip(centers, centers[1:])]
    normalized = [
        displacement / (value + epsilon)
        for displacement, value in zip(displacements, jitter, strict=True)
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
        "valid_state_count": len(ordered),
        "total_state_count": len(ordered),
        "minimum_valid_states": min_states,
        "state_evidence": state_audit(ordered),
        "interval_frame_pairs": [list(pair) for pair in required_pairs],
        "interval_displacement_model_units": displacements,
        "background_jitter_model_units": jitter,
        "used_background_intervals": used_intervals,
        "unused_background_interval_count": len(interval_by_pair) - len(used_intervals),
        "background_evidence_warnings": evidence_warnings,
        "normalized_motion": normalized,
        "median_normalized_motion": median_motion,
        "p90_normalized_motion": p90_motion,
        "direction_consistency": consistency,
        "primary_threshold": float(primary_key),
        "threshold_sweep": sweep,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="track_3d_states.json")
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--background-evidence",
        type=Path,
        help="background_jitter.json; defaults to evidence embedded in input",
    )
    parser.add_argument("--thresholds", type=float, nargs="+", default=[2.0, 3.0, 5.0])
    parser.add_argument("--min-states", type=int, default=3)
    args = parser.parse_args()
    input_path = args.input.resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    evidence_path = args.background_evidence.resolve() if args.background_evidence else input_path
    evidence_payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    intervals, evidence_warnings = background_intervals_from_payload(evidence_payload)
    result = analyze_motion(
        payload["states"],
        intervals,
        thresholds=args.thresholds,
        min_states=args.min_states,
        background_evidence_warnings=evidence_warnings,
    )
    output = {
        "track_candidate_id": payload.get("track_candidate_id"),
        "classification": result,
        "input_provenance": {
            **payload.get("provenance", {}),
            "track_states_sha256": sha256(input_path),
            "background_evidence_sha256": sha256(evidence_path),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
