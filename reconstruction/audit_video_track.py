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
    masks = []
    for row in track["frames"]:
        mask = cv2.imread(str(root / row["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"cannot read mask: {row['mask_path']}")
        masks.append(mask > 0)
    frame_rows, summary = analyze_masks(masks)
    for result, source in zip(frame_rows, track["frames"]):
        result.update({
            "source_frame_index": int(source["source_frame_index"]),
            "timestamp_seconds": float(source["timestamp_seconds"]),
            "mask_sha256": source["mask_sha256"],
        })

    selected = np.linspace(0, len(track["frames"]) - 1, min(args.sample_count, len(track["frames"])), dtype=int)
    tiles = []
    for index in selected:
        row = track["frames"][int(index)]
        frame = cv2.imread(str(root / row["tracking_frame_path"]), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"cannot read tracking frame: {row['tracking_frame_path']}")
        mask = masks[int(index)]
        overlay = frame.copy()
        overlay[mask] = (0.35 * overlay[mask] + 0.65 * np.asarray([0, 0, 255])).astype(np.uint8)
        label = f"sample {row['sample_index']} | source {row['source_frame_index']}"
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
        "track_manifest_sha256": sha256(manifest_path),
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
