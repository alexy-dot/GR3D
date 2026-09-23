#!/usr/bin/env python3
"""Estimate adjacent-frame Pi3 background jitter by symmetric nearest neighbors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def estimate_jitter(
    points: np.ndarray,
    frames: np.ndarray,
    foreground: dict[int, set[int]],
    max_points: int,
) -> tuple[list[dict], list[int]]:
    if max_points < 1:
        raise ValueError("max_points must be positive")
    intervals = []
    frame_ids = sorted(foreground)
    for first, second in zip(frame_ids, frame_ids[1:]):
        a_idx = np.flatnonzero(frames == first)
        b_idx = np.flatnonzero(frames == second)
        a_idx = np.asarray([i for i in a_idx if i not in foreground[first]], dtype=int)
        b_idx = np.asarray([i for i in b_idx if i not in foreground[second]], dtype=int)
        if len(a_idx) > max_points:
            a_idx = a_idx[np.linspace(0, len(a_idx) - 1, max_points, dtype=int)]
        if len(b_idx) > max_points:
            b_idx = b_idx[np.linspace(0, len(b_idx) - 1, max_points, dtype=int)]
        record = {
            "from_frame": first,
            "to_frame": second,
            "from_points": len(a_idx),
            "to_points": len(b_idx),
        }
        if not len(a_idx) or not len(b_idx):
            intervals.append({
                **record,
                "status": "missing_background_support",
                "background_jitter": None,
            })
            continue
        a, b = points[a_idx], points[b_idx]
        ab = cKDTree(b).query(a, k=1, workers=-1)[0]
        ba = cKDTree(a).query(b, k=1, workers=-1)[0]
        intervals.append({
            **record,
            "status": "valid",
            "background_jitter": float(np.median(np.concatenate([ab, ba]))),
        })
    return intervals, frame_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path)
    parser.add_argument("selected_indices", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-points", type=int, default=20000)
    args = parser.parse_args()
    observations_path = args.observations.resolve()
    selected_path = args.selected_indices.resolve()
    with np.load(observations_path) as archive:
        points, frames = archive["points"], archive["frame_index"].astype(int)
    with np.load(selected_path) as archive:
        foreground = {int(name): set(archive[name].astype(int).tolist()) for name in archive.files}
    intervals, frame_ids = estimate_jitter(points, frames, foreground, args.max_points)
    payload = {
        "schema_version": 2,
        "method": "symmetric_nearest_neighbor_median",
        "background_intervals": intervals,
        "frame_ids": frame_ids,
        "max_points_per_frame": args.max_points,
        "observations_sha256": sha256(observations_path),
        "selected_indices_sha256": sha256(selected_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
