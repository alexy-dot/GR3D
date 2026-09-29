#!/usr/bin/env python3
"""Build answer-blind, OSI-ID-linked and time-conditioned trajectory evidence."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


ID_PATTERN = re.compile(r"\(id:\s*(\d+)\)", flags=re.IGNORECASE)
TIME_PATTERN = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b",
    flags=re.IGNORECASE,
)


def question_sha256(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def parse_question_references(question: str) -> tuple[list[int], list[float]]:
    object_ids = list(dict.fromkeys(int(value) for value in ID_PATTERN.findall(question)))
    times = list(dict.fromkeys(float(value) for value in TIME_PATTERN.findall(question)))
    return object_ids, times


def _vector(row: dict, prefix: str = "") -> np.ndarray:
    return np.asarray(
        [float(row[f"{prefix}{axis}"]) for axis in ("x", "y", "z")],
        dtype=np.float64,
    )


def _valid_3d(row: dict) -> bool:
    value = row.get("valid_3d", True)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def interpolate_track(rows: list[dict], timestamp: float, max_gap_seconds: float) -> dict:
    if max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds must be positive")
    all_rows = sorted(rows, key=lambda row: float(row["timestamp_seconds"]))
    if not all_rows:
        raise ValueError("trajectory contains no rows")
    exact_all = [
        row for row in all_rows
        if math.isclose(float(row["timestamp_seconds"]), timestamp, rel_tol=0.0, abs_tol=1e-9)
    ]
    if exact_all and not _valid_3d(exact_all[0]):
        raise ValueError("requested time has an invalid 3D state")
    ordered = [row for row in all_rows if _valid_3d(row)]
    if not ordered:
        raise ValueError("trajectory contains no valid 3D rows")
    times = np.asarray([float(row["timestamp_seconds"]) for row in ordered])
    exact = np.flatnonzero(np.isclose(times, timestamp, rtol=0.0, atol=1e-9))
    if len(exact):
        row = ordered[int(exact[0])]
        return {
            "timestamp_seconds": timestamp,
            "position_xyz": _vector(row).tolist(),
            "camera_xyz": _vector(row, "camera_").tolist(),
            "method": "exact_sample",
            "bracketing_times_seconds": [timestamp, timestamp],
        }
    right = int(np.searchsorted(times, timestamp, side="right"))
    if right == 0 or right == len(ordered):
        raise ValueError("requested time is outside trajectory coverage")
    left = right - 1
    first_time, second_time = float(times[left]), float(times[right])
    gap = second_time - first_time
    if gap > max_gap_seconds:
        raise ValueError(
            f"trajectory gap {gap:.6g}s exceeds allowed {max_gap_seconds:.6g}s"
        )
    invalid_inside = [
        row for row in all_rows
        if not _valid_3d(row)
        and first_time <= float(row["timestamp_seconds"]) <= second_time
    ]
    if invalid_inside:
        raise ValueError("trajectory interval contains an invalid 3D state")
    weight = (timestamp - first_time) / gap
    position = _vector(ordered[left]) * (1 - weight) + _vector(ordered[right]) * weight
    camera = (
        _vector(ordered[left], "camera_") * (1 - weight)
        + _vector(ordered[right], "camera_") * weight
    )
    return {
        "timestamp_seconds": timestamp,
        "position_xyz": position.tolist(),
        "camera_xyz": camera.tolist(),
        "method": "linear_interpolation",
        "bracketing_times_seconds": [first_time, second_time],
    }


def _distance(first: list[float], second: list[float]) -> float:
    return math.dist([float(value) for value in first], [float(value) for value in second])


def build_question_evidence(
    *,
    question_id: int,
    question: str,
    category: str,
    trajectories: dict[int, dict],
    max_gap_seconds: float = 2.0,
    meters_per_model_unit: float | None = None,
) -> dict:
    if meters_per_model_unit is not None:
        meters_per_model_unit = float(meters_per_model_unit)
        if not math.isfinite(meters_per_model_unit) or meters_per_model_unit <= 0:
            raise ValueError("meters_per_model_unit must be positive and finite")
    object_ids, query_times = parse_question_references(question)
    base = {
        "question_id": int(question_id),
        "question_sha256": question_sha256(question),
        "category": str(category),
        "referenced_object_ids": object_ids,
        "query_times_seconds": query_times,
        "answer_blind": True,
        "coordinate_units": "pi3_model_units",
        "metric_scale_validated": meters_per_model_unit is not None,
        "meters_per_model_unit": meters_per_model_unit,
    }
    if not object_ids:
        return {
            **base,
            "status": "not_applicable_no_object_id",
            "time_conditioned": False,
            "objects": [],
            "derived_geometry": {},
        }

    missing = [object_id for object_id in object_ids if object_id not in trajectories]
    if missing:
        return {
            **base,
            "status": "unavailable_missing_object_tracks",
            "missing_object_ids": missing,
            "time_conditioned": False,
            "objects": [],
            "derived_geometry": {},
        }

    objects = []
    failures = []
    for object_id in object_ids:
        trajectory = trajectories[object_id]
        valid_rows = [row for row in trajectory["rows"] if _valid_3d(row)]
        if not valid_rows:
            failures.append({
                "object_id": object_id,
                "time": None,
                "reason": "trajectory contains no valid 3D states",
            })
        samples = []
        for timestamp in query_times:
            try:
                samples.append(
                    interpolate_track(
                        trajectory["rows"], timestamp, max_gap_seconds=max_gap_seconds
                    )
                )
            except ValueError as error:
                failures.append({"object_id": object_id, "time": timestamp, "reason": str(error)})
        objects.append({
            "object_id": object_id,
            "external_identity": trajectory["external_identity"],
            "track_candidate_id": trajectory["track_candidate_id"],
            "motion_state": trajectory["motion_state"],
            "samples": samples,
            "trajectory_time_range_seconds": [
                min(float(row["timestamp_seconds"]) for row in valid_rows)
                if valid_rows else None,
                max(float(row["timestamp_seconds"]) for row in valid_rows)
                if valid_rows else None,
            ],
        })
    if failures:
        return {
            **base,
            "status": "unavailable_time_coverage",
            "time_conditioned": False,
            "coverage_failures": failures,
            "objects": objects,
            "derived_geometry": {},
        }

    derived: dict[str, object] = {}
    if len(objects) == 1 and len(query_times) >= 2:
        first, last = objects[0]["samples"][0], objects[0]["samples"][-1]
        displacement = _distance(first["position_xyz"], last["position_xyz"])
        duration = query_times[-1] - query_times[0]
        derived["object_displacement_model_units"] = displacement
        if duration > 0:
            derived["average_speed_model_units_per_second"] = displacement / duration
        if meters_per_model_unit is not None:
            derived["object_displacement_meters"] = displacement * meters_per_model_unit
            if duration > 0:
                derived["average_speed_meters_per_second"] = (
                    displacement * meters_per_model_unit / duration
                )
    if len(objects) == 1 and len(query_times) == 1:
        sample = objects[0]["samples"][0]
        distance = _distance(sample["position_xyz"], sample["camera_xyz"])
        derived["object_camera_distance_model_units"] = distance
        if meters_per_model_unit is not None:
            derived["object_camera_distance_meters"] = distance * meters_per_model_unit
    if len(objects) >= 2 and len(query_times) == 1:
        pairwise = []
        for index, first in enumerate(objects):
            for second in objects[index + 1 :]:
                distance = _distance(
                    first["samples"][0]["position_xyz"],
                    second["samples"][0]["position_xyz"],
                )
                row = {
                    "object_ids": [first["object_id"], second["object_id"]],
                    "distance_model_units": distance,
                }
                if meters_per_model_unit is not None:
                    row["distance_meters"] = distance * meters_per_model_unit
                pairwise.append(row)
        derived["pairwise_object_distances"] = pairwise

    return {
        **base,
        "status": "ready",
        "time_conditioned": bool(query_times),
        "objects": objects,
        "derived_geometry": derived,
    }


def render_question_evidence(
    evidence: dict,
    trajectories: dict[int, dict],
    output: Path,
) -> None:
    if evidence.get("status") != "ready":
        raise ValueError("only ready question evidence can be rendered")
    object_ids = evidence["referenced_object_ids"]
    query_times = evidence["query_times_seconds"]
    colors = plt.get_cmap("tab10")
    projections = ((0, 1, "X", "Y"), (0, 2, "X", "Z"), (1, 2, "Y", "Z"))
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for object_ordinal, object_id in enumerate(object_ids):
        rows = [row for row in trajectories[object_id]["rows"] if _valid_3d(row)]
        if query_times:
            low, high = min(query_times), max(query_times)
            segment = [
                row for row in rows
                if low - 1e-9 <= float(row["timestamp_seconds"]) <= high + 1e-9
            ]
        else:
            segment = rows
        points = np.asarray([_vector(row) for row in segment], dtype=np.float64)
        samples = next(
            row["samples"] for row in evidence["objects"] if row["object_id"] == object_id
        )
        sample_points = np.asarray(
            [row["position_xyz"] for row in samples], dtype=np.float64
        ) if samples else np.empty((0, 3), dtype=np.float64)
        color = colors(object_ordinal % 10)
        for axis, (first, second, first_name, second_name) in zip(axes, projections):
            if len(points):
                axis.plot(
                    points[:, first], points[:, second], "-o", color=color,
                    linewidth=1.5, markersize=3, label=f"OSI-{object_id}",
                )
            for sample, point in zip(samples, sample_points, strict=True):
                axis.scatter(point[first], point[second], s=45, color=color, edgecolors="black")
                axis.annotate(
                    f"{sample['timestamp_seconds']:g}s",
                    (point[first], point[second]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8,
                )
            axis.set_xlabel(f"{first_name} (Pi3 model units)")
            axis.set_ylabel(f"{second_name} (Pi3 model units)")
            axis.set_aspect("equal", adjustable="datalim")
            axis.grid(True, linewidth=0.4, alpha=0.35)
    for axis in axes:
        axis.legend(loc="best", fontsize=8)
    figure.suptitle(
        f"Question {evidence['question_id']} ID/time-linked trajectory evidence\n"
        "NON-METRIC unless an independently validated scale is declared"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
