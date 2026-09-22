#!/usr/bin/env python3
"""Lift saved binary video masks into Pi3 world-coordinate track states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def erode(mask: np.ndarray, pixels: int) -> np.ndarray:
    result = mask.astype(bool)
    for _ in range(pixels):
        padded = np.pad(result, 1, constant_values=False)
        neighbors = [
            padded[1 + dy : 1 + dy + result.shape[0], 1 + dx : 1 + dx + result.shape[1]]
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
        ]
        result = np.logical_and.reduce(neighbors)
    return result


def lift_track(
    observations: dict[str, np.ndarray],
    entries: list[dict],
    mask_root: Path,
    image_shape: tuple[int, int],
    min_points: int = 20,
    erode_pixels: int = 1,
    minimum_confidence: float = 0.0,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    points = observations["points"]
    frames = observations["frame_index"].astype(np.int64)
    pixels = observations["pixel_yx"].astype(np.int64)
    confidence = observations["confidence"].astype(np.float32)
    states, selected = [], {}
    for entry in sorted(entries, key=lambda row: (row["sample_index"], row["timestamp_seconds"])):
        sample_index = int(entry["sample_index"])
        mask = np.asarray(Image.open(mask_root / entry["mask_path"]).convert("L")) > 0
        if mask.shape != image_shape:
            raise ValueError(f"mask shape {mask.shape} does not match Pi3 image shape {image_shape}")
        mask = erode(mask, erode_pixels)
        candidates = np.flatnonzero((frames == sample_index) & (confidence >= minimum_confidence))
        if len(candidates):
            yx = pixels[candidates]
            chosen = candidates[mask[yx[:, 0], yx[:, 1]]]
        else:
            chosen = candidates
        selected[str(sample_index)] = chosen.astype(np.int64)
        warnings = []
        if not mask.any():
            warnings.append("empty_mask_after_erosion")
        if len(chosen) < min_points:
            warnings.append("insufficient_3d_support")
            center = bbox_min = bbox_max = spread = None
            mean_confidence = None
        else:
            xyz = points[chosen].astype(np.float64)
            center = np.median(xyz, axis=0).tolist()
            bbox_min = np.quantile(xyz, 0.05, axis=0).tolist()
            bbox_max = np.quantile(xyz, 0.95, axis=0).tolist()
            spread = np.median(np.abs(xyz - np.median(xyz, axis=0)), axis=0).tolist()
            mean_confidence = float(confidence[chosen].mean())
        states.append({
            "track_candidate_id": entry["track_candidate_id"],
            "sample_index": sample_index,
            "source_frame_index": int(entry["source_frame_index"]),
            "timestamp_seconds": float(entry["timestamp_seconds"]),
            "point_count": int(len(chosen)),
            "center_xyz_median": center,
            "bbox_min_xyz": bbox_min,
            "bbox_max_xyz": bbox_max,
            "point_spread_mad": spread,
            "mean_pi3_confidence": mean_confidence,
            "quality_warnings": warnings,
        })
    return states, selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("track_manifest", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--min-points", type=int, default=20)
    parser.add_argument("--erode-pixels", type=int, default=1)
    parser.add_argument("--minimum-confidence", type=float, default=0.0)
    args = parser.parse_args()
    run = args.run_directory.resolve()
    track_path = args.track_manifest.resolve()
    track = json.loads(track_path.read_text(encoding="utf-8"))
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = track.get("pi3_manifest_sha256")
    if expected and expected != sha256(manifest_path):
        raise ValueError("track manifest belongs to a different Pi3 run manifest")
    with np.load(run / "point_observations.npz") as archive:
        observations = {name: archive[name] for name in archive.files}
    states, selected = lift_track(
        observations,
        track["frames"],
        track_path.parent,
        (int(manifest["resized_height"]), int(manifest["resized_width"])),
        args.min_points,
        args.erode_pixels,
        args.minimum_confidence,
    )
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "selected_point_indices.npz", **selected)
    payload = {
        "track_candidate_id": track["track_candidate_id"],
        "states": states,
        "parameters": vars(args) | {"run_directory": str(run), "track_manifest": str(track_path)},
        "provenance": {
            "pi3_manifest_sha256": sha256(manifest_path),
            "point_observations_sha256": sha256(run / "point_observations.npz"),
            "track_manifest_sha256": sha256(track_path),
        },
    }
    (output / "track_3d_states.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(states)} states to {output}")


if __name__ == "__main__":
    main()
