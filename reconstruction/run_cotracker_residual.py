#!/usr/bin/env python3
"""Measure auxiliary foreground/background residual motion with CoTracker3."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

if __package__:
    from reconstruction.run_video_instance_tracking import git_state, sha256
else:
    from run_video_instance_tracking import git_state, sha256


GROUP_FOREGROUND = 0
GROUP_BACKGROUND = 1


def deterministic_points(region: np.ndarray, count: int) -> np.ndarray:
    if count < 1:
        raise ValueError("point count must be positive")
    yx = np.argwhere(region.astype(bool))
    if not len(yx):
        return np.empty((0, 2), dtype=np.float32)
    xy = yx[:, ::-1].astype(np.float64)
    if len(xy) <= count:
        return xy.astype(np.float32)
    centroid = xy.mean(axis=0)
    selected = [int(np.argmin(np.sum((xy - centroid) ** 2, axis=1)))]
    minimum_distance = np.sum((xy - xy[selected[0]]) ** 2, axis=1)
    for _ in range(1, count):
        index = int(np.argmax(minimum_distance))
        selected.append(index)
        distance = np.sum((xy - xy[index]) ** 2, axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
    return xy[selected].astype(np.float32)


def sample_queries(
    mask: np.ndarray,
    excluded_foreground: np.ndarray,
    foreground_count: int,
    background_count: int,
    erode_pixels: int,
    ring_inner_pixels: int,
    ring_outer_pixels: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if mask.shape != excluded_foreground.shape:
        raise ValueError("mask and excluded foreground shape do not match")
    if erode_pixels < 0 or ring_inner_pixels < 0:
        raise ValueError("erosion and ring radii must be non-negative")
    if ring_outer_pixels <= ring_inner_pixels:
        raise ValueError("outer ring radius must exceed inner ring radius")
    binary = mask.astype(np.uint8)
    erode_size = 2 * erode_pixels + 1
    interior = cv2.erode(binary, np.ones((erode_size, erode_size), np.uint8)) > 0
    inner_size = 2 * ring_inner_pixels + 1
    outer_size = 2 * ring_outer_pixels + 1
    inner = cv2.dilate(binary, np.ones((inner_size, inner_size), np.uint8)) > 0
    outer = cv2.dilate(binary, np.ones((outer_size, outer_size), np.uint8)) > 0
    ring = outer & ~inner & ~excluded_foreground.astype(bool)
    foreground = deterministic_points(interior, foreground_count)
    background = deterministic_points(ring, background_count)
    points = np.concatenate([foreground, background], axis=0)
    groups = np.concatenate(
        [
            np.full(len(foreground), GROUP_FOREGROUND, dtype=np.uint8),
            np.full(len(background), GROUP_BACKGROUND, dtype=np.uint8),
        ]
    )
    return points, groups, {
        "interior_pixel_count": int(interior.sum()),
        "ring_pixel_count": int(ring.sum()),
        "foreground_query_count": len(foreground),
        "background_query_count": len(background),
    }


def fit_robust_affine(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(source) != len(target) or len(source) < 6:
        raise ValueError("at least six paired background points are required")
    design = np.column_stack([source.astype(np.float64), np.ones(len(source))])
    target = target.astype(np.float64)
    active = np.arange(len(source))
    coefficients = np.zeros((3, 2), dtype=np.float64)
    for _ in range(3):
        coefficients = np.linalg.lstsq(design[active], target[active], rcond=None)[0]
        residual = np.linalg.norm(design @ coefficients - target, axis=1)
        keep = max(6, math.ceil(0.8 * len(active)))
        active = np.argsort(residual, kind="stable")[:keep]
    return coefficients


def analyze_residuals(
    tracks: np.ndarray,
    visibility: np.ndarray,
    groups: np.ndarray,
    min_foreground_points: int,
    min_background_points: int,
    epsilon: float = 1e-6,
) -> tuple[list[dict], dict]:
    if tracks.ndim != 3 or tracks.shape[2] != 2:
        raise ValueError("tracks must have shape [frames, points, 2]")
    if visibility.shape != tracks.shape[:2] or len(groups) != tracks.shape[1]:
        raise ValueError("visibility or group shape does not match tracks")
    if min_background_points < 6 or min_foreground_points < 1:
        raise ValueError("minimum supports must be at least one foreground and six background")
    rows = []
    for frame_index in range(len(tracks) - 1):
        paired = visibility[frame_index].astype(bool) & visibility[frame_index + 1].astype(bool)
        foreground = np.flatnonzero(paired & (groups == GROUP_FOREGROUND))
        background = np.flatnonzero(paired & (groups == GROUP_BACKGROUND))
        row = {
            "from_sample_index": frame_index,
            "to_sample_index": frame_index + 1,
            "foreground_visible_pairs": len(foreground),
            "background_visible_pairs": len(background),
            "valid": False,
            "quality_warnings": [],
        }
        if len(foreground) < min_foreground_points:
            row["quality_warnings"].append("insufficient_foreground_visibility")
        if len(background) < min_background_points:
            row["quality_warnings"].append("insufficient_background_visibility")
        if row["quality_warnings"]:
            rows.append(row)
            continue
        coefficients = fit_robust_affine(
            tracks[frame_index, background], tracks[frame_index + 1, background]
        )
        all_source = np.column_stack(
            [tracks[frame_index], np.ones(tracks.shape[1], dtype=np.float32)]
        )
        predicted = all_source @ coefficients
        residual = np.linalg.norm(tracks[frame_index + 1] - predicted, axis=1)
        raw_displacement = np.linalg.norm(
            tracks[frame_index + 1] - tracks[frame_index], axis=1
        )
        foreground_residual = float(np.median(residual[foreground]))
        background_residual = float(np.median(residual[background]))
        row.update(
            {
                "valid": True,
                "foreground_raw_displacement_median_pixels": float(
                    np.median(raw_displacement[foreground])
                ),
                "background_raw_displacement_median_pixels": float(
                    np.median(raw_displacement[background])
                ),
                "foreground_residual_median_pixels": foreground_residual,
                "background_fit_residual_median_pixels": background_residual,
                "normalized_foreground_residual": foreground_residual
                / (background_residual + epsilon),
                "background_affine_xy1_to_xy": coefficients.tolist(),
            }
        )
        rows.append(row)
    valid = [row for row in rows if row["valid"]]
    normalized = [row["normalized_foreground_residual"] for row in valid]
    foreground_visibility = visibility[:, groups == GROUP_FOREGROUND].mean(axis=1)
    background_visibility = visibility[:, groups == GROUP_BACKGROUND].mean(axis=1)
    summary = {
        "interval_count": len(rows),
        "valid_interval_count": len(valid),
        "foreground_visibility_fraction_min": float(foreground_visibility.min()),
        "foreground_visibility_fraction_median": float(np.median(foreground_visibility)),
        "background_visibility_fraction_min": float(background_visibility.min()),
        "background_visibility_fraction_median": float(np.median(background_visibility)),
        "normalized_foreground_residual_median": (
            float(np.median(normalized)) if normalized else None
        ),
        "normalized_foreground_residual_p90": (
            float(np.quantile(normalized, 0.9)) if normalized else None
        ),
        "warnings": (
            [] if len(valid) == len(rows) else ["some_intervals_failed_visibility_gate"]
        ),
    }
    return rows, summary


def repository_revision(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def resolve_source_revision(
    source_root: Path | None, declared_revision: str
) -> tuple[str, str | None]:
    revision = declared_revision.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("CoTracker revision must be a full 40-character Git commit hash")
    detected = repository_revision(source_root) if source_root is not None else None
    if detected is not None and detected.lower() != revision:
        raise ValueError(
            f"declared CoTracker revision {revision} does not match source checkout {detected}"
        )
    return revision, detected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("track_manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cotracker-root", type=Path)
    parser.add_argument(
        "--cotracker-revision",
        required=True,
        help="full Git commit hash for the CoTracker source, including archive installs",
    )
    parser.add_argument("--exclude-track-manifest", type=Path, action="append", default=[])
    parser.add_argument("--foreground-points", type=int, default=64)
    parser.add_argument("--background-points", type=int, default=96)
    parser.add_argument("--erode-pixels", type=int, default=3)
    parser.add_argument("--ring-inner-pixels", type=int, default=8)
    parser.add_argument("--ring-outer-pixels", type=int, default=48)
    parser.add_argument("--min-foreground-points", type=int, default=8)
    parser.add_argument("--min-background-points", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.monotonic()
    manifest_path = args.track_manifest.resolve()
    root = manifest_path.parent
    track = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = sorted(track["frames"], key=lambda row: int(row["sample_index"]))
    expected_indices = list(range(len(frames)))
    if [int(row["sample_index"]) for row in frames] != expected_indices:
        raise ValueError("track manifest must contain a dense zero-based sample sequence")
    prompt_index = int(track["prompt"]["sample_index"])
    prompt_row = frames[prompt_index]
    mask = cv2.imread(str(root / prompt_row["mask_path"]), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("cannot read prompt mask")
    excluded = mask > 0
    exclusion_hashes = []
    for path in args.exclude_track_manifest:
        resolved = path.resolve()
        other = json.loads(resolved.read_text(encoding="utf-8"))
        if other["video_sha256"] != track["video_sha256"]:
            raise ValueError("excluded track belongs to a different video")
        other_frames = {int(row["sample_index"]): row for row in other["frames"]}
        other_row = other_frames[prompt_index]
        other_mask = cv2.imread(
            str(resolved.parent / other_row["mask_path"]), cv2.IMREAD_GRAYSCALE
        )
        if other_mask is None or other_mask.shape != mask.shape:
            raise ValueError("cannot read same-size excluded prompt mask")
        excluded |= other_mask > 0
        exclusion_hashes.append({"path": str(resolved), "sha256": sha256(resolved)})
    query_xy, query_groups, sampling = sample_queries(
        mask > 0,
        excluded,
        args.foreground_points,
        args.background_points,
        args.erode_pixels,
        args.ring_inner_pixels,
        args.ring_outer_pixels,
    )
    if sampling["foreground_query_count"] < args.min_foreground_points:
        raise ValueError("not enough interior foreground points for CoTracker")
    if sampling["background_query_count"] < args.min_background_points:
        raise ValueError("not enough background-ring points for CoTracker")

    images = []
    for row in frames:
        image = cv2.imread(str(root / row["tracking_frame_path"]), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != mask.shape:
            raise ValueError(f"cannot read same-size frame {row['tracking_frame_path']}")
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    video = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)[None].float()
    query_time = np.full((len(query_xy), 1), prompt_index, dtype=np.float32)
    queries = torch.from_numpy(np.concatenate([query_time, query_xy], axis=1))[None]

    cotracker_root = args.cotracker_root or (
        Path(os.environ["COTRACKER_ROOT"]) if os.environ.get("COTRACKER_ROOT") else None
    )
    if cotracker_root is not None:
        cotracker_root = cotracker_root.resolve()
        sys.path.insert(0, str(cotracker_root))
    source_revision, detected_source_revision = resolve_source_revision(
        cotracker_root, args.cotracker_revision
    )
    from cotracker.predictor import CoTrackerPredictor

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run reconstruction/check_gpu.py first.")
    checkpoint = args.checkpoint.resolve()
    model = CoTrackerPredictor(checkpoint=str(checkpoint)).to(device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        predicted_tracks, predicted_visibility = model(
            video.to(device),
            queries=queries.to(device),
            backward_tracking=prompt_index > 0,
        )
    tracks = predicted_tracks[0].float().cpu().numpy()
    visibility = predicted_visibility[0].bool().cpu().numpy()
    intervals, summary = analyze_residuals(
        tracks,
        visibility,
        query_groups,
        args.min_foreground_points,
        args.min_background_points,
    )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / "point_tracks.npz"
    np.savez_compressed(
        archive_path,
        tracks_xy=tracks.astype(np.float32),
        visibility=visibility,
        query_xy=query_xy,
        query_group=query_groups,
    )
    revision, dirty = git_state(Path.cwd())
    payload = {
        "status": "complete",
        "track_candidate_id": track["track_candidate_id"],
        "method": "cotracker3_offline_local_affine_background_residual",
        "source_track_manifest": str(manifest_path),
        "source_track_manifest_sha256": sha256(manifest_path),
        "excluded_track_manifests": exclusion_hashes,
        "video_sha256": track["video_sha256"],
        "frames": [
            {
                "sample_index": int(row["sample_index"]),
                "source_frame_index": int(row["source_frame_index"]),
                "timestamp_seconds": float(row["timestamp_seconds"]),
                "frame_sha256": row["tracking_frame_sha256"],
            }
            for row in frames
        ],
        "model": {
            "name": "CoTracker3 offline",
            "checkpoint": str(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": sha256(checkpoint),
            "source_root": str(cotracker_root) if cotracker_root else None,
            "source_revision": source_revision,
            "detected_source_revision": detected_source_revision,
            "source_revision_git_verified": detected_source_revision is not None,
        },
        "parameters": {
            "prompt_sample_index": prompt_index,
            "foreground_points": args.foreground_points,
            "background_points": args.background_points,
            "erode_pixels": args.erode_pixels,
            "ring_inner_pixels": args.ring_inner_pixels,
            "ring_outer_pixels": args.ring_outer_pixels,
            "min_foreground_points": args.min_foreground_points,
            "min_background_points": args.min_background_points,
            "backward_tracking": prompt_index > 0,
        },
        "sampling": sampling,
        "point_tracks": {"path": archive_path.name, "sha256": sha256(archive_path)},
        "summary": summary,
        "intervals": intervals,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "elapsed_seconds": time.monotonic() - started,
            "peak_gpu_gib": (
                round(torch.cuda.max_memory_allocated(device) / (1024**3), 3)
                if device.type == "cuda"
                else None
            ),
        },
        "code_revision": revision,
        "code_dirty": dirty,
        "source_frames_modified": False,
        "limitations": [
            "auxiliary_2d_motion_evidence_not_object_identity",
            "local_affine_background_model_is_not_full_camera_geometry",
            "visibility_is_model_output_not_annotated_occlusion_ground_truth",
        ],
    }
    manifest_output = output / "cotracker_residual_manifest.json"
    manifest_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
