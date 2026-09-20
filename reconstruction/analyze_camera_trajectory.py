#!/usr/bin/env python3
"""Measure a reconstructed camera path and optionally apply one metric scale anchor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-poses", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--known-path-length-m",
        type=float,
        help="Optional single ground-truth path length used only as a scale anchor.",
    )
    parser.add_argument(
        "--anchor-source",
        help="Human-readable provenance for --known-path-length-m.",
    )
    return parser.parse_args()


def trajectory_metrics(poses: np.ndarray) -> dict[str, object]:
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or poses.shape[0] < 2:
        raise ValueError(f"Expected camera poses with shape (N, 4, 4), got {poses.shape}")
    if not np.isfinite(poses).all():
        raise ValueError("Camera poses contain non-finite values")

    centers = poses[:, :3, 3].astype(np.float64)
    deltas = np.diff(centers, axis=0)
    step_lengths = np.linalg.norm(deltas, axis=1)
    path_length = float(step_lengths.sum())
    endpoint_displacement = float(np.linalg.norm(centers[-1] - centers[0]))
    if path_length <= 0:
        raise ValueError("Camera path length must be positive")

    headings = np.unwrap(np.arctan2(deltas[:, 1], deltas[:, 0]))
    return {
        "frame_count": int(poses.shape[0]),
        "camera_centers": centers.tolist(),
        "step_lengths_model_units": step_lengths.tolist(),
        "path_length_model_units": path_length,
        "endpoint_displacement_model_units": endpoint_displacement,
        "straightness_ratio": endpoint_displacement / path_length,
        "raw_xy_heading_change_degrees": float(np.degrees(headings[-1] - headings[0])),
    }


def main() -> None:
    args = parse_args()
    poses_path = args.camera_poses.expanduser().resolve()
    metrics = trajectory_metrics(np.load(poses_path))
    result: dict[str, object] = {
        "status": "complete",
        "camera_poses": str(poses_path),
        **metrics,
    }

    if args.known_path_length_m is not None:
        if args.known_path_length_m <= 0:
            raise ValueError("--known-path-length-m must be positive")
        scale = args.known_path_length_m / float(metrics["path_length_model_units"])
        result["metric_scale_anchor"] = {
            "known_path_length_m": args.known_path_length_m,
            "source": args.anchor_source,
            "meters_per_model_unit": scale,
            "scaled_step_lengths_m": (
                np.asarray(metrics["step_lengths_model_units"]) * scale
            ).tolist(),
            "warning": (
                "A single answer-derived scale anchor calibrates this run but cannot "
                "independently validate metric accuracy or scale drift."
            ),
        }

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote trajectory analysis: {output}")


if __name__ == "__main__":
    main()
