#!/usr/bin/env python3
"""Audit a saved video-mask track and render a deterministic contact sheet."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

try:
    from reconstruction.tracking_contracts import normalize_external_identity
except ModuleNotFoundError:
    from tracking_contracts import normalize_external_identity


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze_masks(masks: list[np.ndarray]) -> tuple[list[dict], dict]:
    if not masks:
        raise ValueError("track contains no masks")
    shape = masks[0].shape
    if any(mask.shape != shape for mask in masks):
        raise ValueError("all track masks must have the same shape")
    frame_rows = []
    previous = None
    previous_area = None
    height, width = shape
    diagonal = math.hypot(width, height)
    for index, raw_mask in enumerate(masks):
        mask = raw_mask.astype(bool)
        area = int(mask.sum())
        ys, xs = np.nonzero(mask)
        centroid = [float(xs.mean()), float(ys.mean())] if area else None
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if area else None
        if previous is None:
            iou = area_ratio = centroid_jump = None
        else:
            union = int(np.logical_or(previous, mask).sum())
            intersection = int(np.logical_and(previous, mask).sum())
            iou = intersection / union if union else 1.0
            area_ratio = max(area, previous_area) / max(min(area, previous_area), 1)
            if centroid is None or frame_rows[-1]["centroid_xy"] is None:
                centroid_jump = None
            else:
                centroid_jump = math.dist(centroid, frame_rows[-1]["centroid_xy"]) / diagonal
        frame_rows.append({
            "sample_index": index,
            "mask_area_pixels": area,
            "mask_fraction": area / mask.size,
            "bbox_xyxy": bbox,
            "centroid_xy": centroid,
            "previous_iou": iou,
            "previous_area_ratio": area_ratio,
            "previous_centroid_jump_image_diagonal": centroid_jump,
        })
        previous = mask
        previous_area = area

    fractions = [row["mask_fraction"] for row in frame_rows]
    ious = [row["previous_iou"] for row in frame_rows if row["previous_iou"] is not None]
    ratios = [row["previous_area_ratio"] for row in frame_rows if row["previous_area_ratio"] is not None]
    jumps = [row["previous_centroid_jump_image_diagonal"] for row in frame_rows if row["previous_centroid_jump_image_diagonal"] is not None]
    warnings = []
    if any(row["mask_area_pixels"] == 0 for row in frame_rows):
        warnings.append("empty_mask")
    if ratios and max(ratios) > 2.5:
        warnings.append("large_adjacent_area_change")
    if jumps and max(jumps) > 0.25:
        warnings.append("large_adjacent_centroid_jump")
    summary = {
        "frame_count": len(frame_rows),
        "mask_fraction_min": min(fractions),
        "mask_fraction_max": max(fractions),
        "consecutive_iou_min": min(ious) if ious else None,
        "consecutive_iou_median": float(np.median(ious)) if ious else None,
        "adjacent_area_ratio_max": max(ratios) if ratios else None,
        "adjacent_centroid_jump_max_image_diagonal": max(jumps) if jumps else None,
        "warnings": warnings,
    }
    return frame_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("track_manifest", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args()
    if args.sample_count < 1 or args.columns < 1:
        raise ValueError("sample-count and columns must be positive")

    manifest_path = args.track_manifest.resolve()
    root = manifest_path.parent
    track = json.loads(manifest_path.read_text(encoding="utf-8"))
    if track.get("status") != "complete":
        raise ValueError("tracking manifest is not complete")
    if track.get("source_frames_modified") is not False:
        raise ValueError("tracking run must preserve source frames")
    external_identity = normalize_external_identity(track.get("external_identity"))
    track_frames = track.get("frames", [])
    if not track_frames:
        raise ValueError("track contains no saved frames")
    sample_indices = [int(row["sample_index"]) for row in track_frames]
    source_indices = [int(row["source_frame_index"]) for row in track_frames]
    timestamps = [float(row["timestamp_seconds"]) for row in track_frames]
    if len(sample_indices) != len(set(sample_indices)):
        raise ValueError("track contains duplicate sample indices")
    if any(second <= first for first, second in zip(sample_indices, sample_indices[1:])):
        raise ValueError("track sample indices must be strictly increasing")
    if any(second <= first for first, second in zip(source_indices, source_indices[1:])):
        raise ValueError("track source frame indices must be strictly increasing")
    if any(second <= first for first, second in zip(timestamps, timestamps[1:])):
        raise ValueError("track timestamps must be strictly increasing")
    masks = []
    for row in track_frames:
        if row.get("track_candidate_id") != track.get("track_candidate_id"):
            raise ValueError("tracking frame belongs to another candidate")
        if normalize_external_identity(row.get("external_identity")) != external_identity:
            raise ValueError("tracking frame external identity mismatch")
        mask_path = root / row["mask_path"]
        frame_path = root / row["tracking_frame_path"]
        if sha256(mask_path) != row.get("mask_sha256"):
            raise ValueError("saved track mask hash mismatch")
        if sha256(frame_path) != row.get("tracking_frame_sha256"):
            raise ValueError("saved tracking frame hash mismatch")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"cannot read mask: {row['mask_path']}")
        masks.append(mask > 0)
    frame_rows, summary = analyze_masks(masks)
    for result, source in zip(frame_rows, track_frames, strict=True):
        result.update({
            "sample_index": int(source["sample_index"]),
            "source_frame_index": int(source["source_frame_index"]),
            "timestamp_seconds": float(source["timestamp_seconds"]),
            "mask_sha256": source["mask_sha256"],
            "tracking_frame_sha256": source["tracking_frame_sha256"],
        })

    selected = np.linspace(
        0,
        len(track_frames) - 1,
        min(args.sample_count, len(track_frames)),
        dtype=int,
    )
    prompt_sample_index = int(track.get("parameters", {}).get("prompt_sample_index", 0))
    prompt_positions = [
        index for index, row in enumerate(track_frames)
        if int(row["sample_index"]) == prompt_sample_index
    ]
    if not prompt_positions:
        raise ValueError("prompt sample is missing from saved track frames")
    selected = np.asarray(
        sorted(set(selected.tolist()) | {prompt_positions[0]}), dtype=int
    )
    tiles = []
    for index in selected:
        row = track_frames[int(index)]
        frame = cv2.imread(str(root / row["tracking_frame_path"]), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"cannot read tracking frame: {row['tracking_frame_path']}")
        mask = masks[int(index)]
        overlay = frame.copy()
        overlay[mask] = (0.35 * overlay[mask] + 0.65 * np.asarray([0, 0, 255])).astype(np.uint8)
        identity_label = (
            f"OSI-{external_identity['object_id']} | "
            if external_identity is not None
            else ""
        )
        label = (
            f"{identity_label}sample {row['sample_index']} | "
            f"source {row['source_frame_index']}"
        )
        cv2.putText(overlay, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(overlay)
    blank = np.zeros_like(tiles[0])
    while len(tiles) % args.columns:
        tiles.append(blank.copy())
    rows = [np.hstack(tiles[start : start + args.columns]) for start in range(0, len(tiles), args.columns)]
    contact_sheet = np.vstack(rows)

    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    image_path = output / "track_contact_sheet.jpg"
    if not cv2.imwrite(str(image_path), contact_sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError(f"failed to write {image_path}")
    payload = {
        "status": "complete",
        "track_candidate_id": track["track_candidate_id"],
        "external_identity": external_identity,
        "track_manifest_sha256": sha256(manifest_path),
        "source_frames_modified": False,
        "identity_binding_evidence": {
            "prompt": track.get("prompt"),
            "prompt_sample_index": prompt_sample_index,
            "association_is_human_auditable": external_identity is not None,
        },
        "summary": summary,
        "frames": frame_rows,
        "contact_sheet": {
            "path": image_path.name,
            "sha256": sha256(image_path),
            "selected_sample_indices": selected.tolist(),
        },
    }
    (output / "track_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
