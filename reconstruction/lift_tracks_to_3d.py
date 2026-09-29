#!/usr/bin/env python3
"""Lift saved binary video masks into Pi3 world-coordinate track states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from reconstruction.tracking_contracts import (
        normalize_external_identity,
        validate_coordinate_system,
    )
except ModuleNotFoundError:
    from tracking_contracts import normalize_external_identity, validate_coordinate_system


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
    camera_centers: np.ndarray | None = None,
    external_identity: dict | None = None,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    points = observations["points"]
    frames = observations["frame_index"].astype(np.int64)
    pixels = observations["pixel_yx"].astype(np.int64)
    confidence = observations["confidence"].astype(np.float32)
    if camera_centers is not None:
        camera_centers = np.asarray(camera_centers, dtype=np.float64)
        if camera_centers.ndim != 2 or camera_centers.shape[1] != 3:
            raise ValueError("camera_centers must have shape (N, 3)")
    external_identity = normalize_external_identity(external_identity)
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
            "external_identity": external_identity,
            "sample_index": sample_index,
            "source_frame_index": int(entry["source_frame_index"]),
            "timestamp_seconds": float(entry["timestamp_seconds"]),
            "point_count": int(len(chosen)),
            "center_xyz_median": center,
            "camera_center_xyz": (
                camera_centers[sample_index].tolist()
                if camera_centers is not None and sample_index < len(camera_centers)
                else None
            ),
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
    if track.get("status") != "complete":
        raise ValueError("aligned track manifest is not complete")
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Pi3 run manifest is not complete")
    coordinate_system = validate_coordinate_system(manifest.get("coordinate_system"))
    if validate_coordinate_system(track.get("coordinate_system")) != coordinate_system:
        raise ValueError("track and Pi3 run use different coordinate systems")
    expected = track.get("pi3_manifest_sha256")
    if expected and expected != sha256(manifest_path):
        raise ValueError("track manifest belongs to a different Pi3 run manifest")
    with np.load(run / "point_observations.npz") as archive:
        observations = {name: archive[name] for name in archive.files}
    required_arrays = {"points", "colors", "frame_index", "pixel_yx", "confidence"}
    if not required_arrays.issubset(observations):
        raise ValueError("point observations archive is incomplete")
    observation_count = len(observations["points"])
    if any(
        len(observations[name]) != observation_count
        for name in ("colors", "frame_index", "pixel_yx", "confidence")
    ):
        raise ValueError("point observation arrays have inconsistent lengths")
    if observations["points"].ndim != 2 or observations["points"].shape[1] != 3:
        raise ValueError("point observations must have shape (N, 3)")
    if observations["pixel_yx"].ndim != 2 or observations["pixel_yx"].shape[1] != 2:
        raise ValueError("pixel_yx must have shape (N, 2)")
    if observation_count:
        if observations["frame_index"].min() < 0 or observations["frame_index"].max() >= int(
            manifest["frame_count"]
        ):
            raise ValueError("point observation frame index is out of bounds")
        if (
            observations["pixel_yx"][:, 0].min() < 0
            or observations["pixel_yx"][:, 0].max() >= int(manifest["resized_height"])
            or observations["pixel_yx"][:, 1].min() < 0
            or observations["pixel_yx"][:, 1].max() >= int(manifest["resized_width"])
        ):
            raise ValueError("point observation pixel coordinate is out of bounds")
    camera_poses = np.load(run / "camera_poses.npy")
    if camera_poses.ndim != 3 or camera_poses.shape[1:] != (4, 4):
        raise ValueError("camera_poses.npy must have shape (N, 4, 4)")
    if len(camera_poses) != int(manifest["frame_count"]):
        raise ValueError("camera pose count differs from Pi3 frame count")
    if not np.isfinite(camera_poses).all():
        raise ValueError("camera poses contain non-finite values")
    external_identity = normalize_external_identity(track.get("external_identity"))
    seen_samples = set()
    for entry in track.get("frames", []):
        sample_index = int(entry["sample_index"])
        if sample_index in seen_samples:
            raise ValueError("aligned track contains duplicate sample indices")
        seen_samples.add(sample_index)
        if entry.get("track_candidate_id") != track.get("track_candidate_id"):
            raise ValueError("aligned track frame belongs to another candidate")
        if normalize_external_identity(entry.get("external_identity")) != external_identity:
            raise ValueError("aligned track frame external identity mismatch")
        mask_path = track_path.parent / entry["mask_path"]
        if sha256(mask_path) != entry.get("mask_sha256"):
            raise ValueError("aligned mask hash mismatch")
    states, selected = lift_track(
        observations,
        track["frames"],
        track_path.parent,
        (int(manifest["resized_height"]), int(manifest["resized_width"])),
        args.min_points,
        args.erode_pixels,
        args.minimum_confidence,
        camera_poses[:, :3, 3],
        external_identity,
    )
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected_path = output / "selected_point_indices.npz"
    np.savez_compressed(selected_path, **selected)
    payload = {
        "track_candidate_id": track["track_candidate_id"],
        "external_identity": external_identity,
        "coordinate_system": coordinate_system,
        "states": states,
        "parameters": vars(args) | {"run_directory": str(run), "track_manifest": str(track_path)},
        "provenance": {
            "pi3_manifest_sha256": sha256(manifest_path),
            "point_observations_sha256": sha256(run / "point_observations.npz"),
            "camera_poses_sha256": sha256(run / "camera_poses.npy"),
            "track_manifest_sha256": sha256(track_path),
            "selected_point_indices_sha256": sha256(selected_path),
        },
    }
    (output / "track_3d_states.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(states)} states to {output}")


if __name__ == "__main__":
    main()
