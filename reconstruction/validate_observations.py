#!/usr/bin/env python3
"""Validate Pi3 point/pixel observations before semantic voxel fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    run = args.run_directory.expanduser().resolve()
    observations = np.load(run / "point_observations.npz")
    depth = np.load(run / "depth_maps.npy")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))

    required = {"points", "colors", "frame_index", "pixel_yx", "confidence"}
    missing = required.difference(observations.files)
    if missing:
        raise ValueError(f"Missing observation arrays: {sorted(missing)}")
    count = len(observations["points"])
    for name in required:
        if len(observations[name]) != count:
            raise ValueError(f"{name} count does not match points")
    if observations["points"].shape != (count, 3):
        raise ValueError("points must have shape (N, 3)")
    if observations["colors"].shape != (count, 3):
        raise ValueError("colors must have shape (N, 3)")
    if observations["pixel_yx"].shape != (count, 2):
        raise ValueError("pixel_yx must have shape (N, 2)")
    if depth.ndim != 3:
        raise ValueError("depth maps must have shape (frames, height, width)")
    if not np.isfinite(observations["points"]).all() or not np.isfinite(depth).all():
        raise ValueError("observations contain non-finite geometry")
    frames = observations["frame_index"].astype(np.int64)
    pixels = observations["pixel_yx"].astype(np.int64)
    if frames.min() < 0 or frames.max() >= depth.shape[0]:
        raise ValueError("frame indices exceed depth-map bounds")
    if pixels[:, 0].min() < 0 or pixels[:, 0].max() >= depth.shape[1]:
        raise ValueError("pixel Y coordinates exceed depth-map bounds")
    if pixels[:, 1].min() < 0 or pixels[:, 1].max() >= depth.shape[2]:
        raise ValueError("pixel X coordinates exceed depth-map bounds")
    if count != manifest.get("point_count"):
        raise ValueError("observation count does not match manifest point_count")

    summary = {
        "status": "valid",
        "point_count": count,
        "frame_range": [int(frames.min()), int(frames.max())],
        "pixel_y_range": [int(pixels[:, 0].min()), int(pixels[:, 0].max())],
        "pixel_x_range": [int(pixels[:, 1].min()), int(pixels[:, 1].max())],
        "depth_shape": list(depth.shape),
        "depth_dtype": str(depth.dtype),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
