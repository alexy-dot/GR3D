#!/usr/bin/env python3
"""Compare two saved binary video tracks on the same extracted frames."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from reconstruction.run_video_instance_tracking import git_state, sha256
else:
    from run_video_instance_tracking import git_state, sha256


def compare_masks(reference: np.ndarray, candidate: np.ndarray) -> dict:
    if reference.shape != candidate.shape:
        raise ValueError("mask shapes do not match")
    reference = reference.astype(bool)
    candidate = candidate.astype(bool)
    reference_area = int(reference.sum())
    candidate_area = int(candidate.sum())
    intersection = int(np.logical_and(reference, candidate).sum())
    union = int(np.logical_or(reference, candidate).sum())
    denominator = reference_area + candidate_area

    def centroid(mask: np.ndarray) -> np.ndarray | None:
        y, x = np.nonzero(mask)
        if len(x) == 0:
            return None
        return np.asarray([float(x.mean()), float(y.mean())])

    reference_centroid = centroid(reference)
    candidate_centroid = centroid(candidate)
    diagonal = math.hypot(reference.shape[0], reference.shape[1])
    centroid_distance = (
        float(np.linalg.norm(reference_centroid - candidate_centroid) / diagonal)
        if reference_centroid is not None and candidate_centroid is not None
        else None
    )
    return {
        "reference_area_pixels": reference_area,
        "candidate_area_pixels": candidate_area,
        "candidate_to_reference_area_ratio": (
            candidate_area / reference_area if reference_area else None
        ),
        "intersection_pixels": intersection,
        "union_pixels": union,
        "iou": intersection / union if union else 1.0,
        "dice": 2 * intersection / denominator if denominator else 1.0,
        "candidate_covered_by_reference": (
            intersection / candidate_area if candidate_area else None
        ),
        "reference_covered_by_candidate": (
            intersection / reference_area if reference_area else None
        ),
        "centroid_distance_image_diagonal": centroid_distance,
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("at least one comparison row is required")

    def values(name: str) -> list[float]:
        return [float(row[name]) for row in rows if row[name] is not None]

    ious = values("iou")
    dice = values("dice")
    ratios = values("candidate_to_reference_area_ratio")
    candidate_cover = values("candidate_covered_by_reference")
    reference_cover = values("reference_covered_by_candidate")
    centroid = values("centroid_distance_image_diagonal")
    return {
        "frame_count": len(rows),
        "exact_mask_hash_matches": sum(row["mask_hash_equal"] for row in rows),
        "iou_min": min(ious),
        "iou_median": float(np.median(ious)),
        "iou_mean": float(np.mean(ious)),
        "dice_median": float(np.median(dice)),
        "candidate_to_reference_area_ratio_median": float(np.median(ratios)),
        "candidate_covered_by_reference_min": min(candidate_cover),
        "candidate_covered_by_reference_median": float(np.median(candidate_cover)),
        "reference_covered_by_candidate_median": float(np.median(reference_cover)),
        "centroid_distance_median_image_diagonal": float(np.median(centroid)),
        "centroid_distance_max_image_diagonal": max(centroid),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_manifest", type=Path)
    parser.add_argument("candidate_manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    reference_path = args.reference_manifest.resolve()
    candidate_path = args.candidate_manifest.resolve()
    reference_root = reference_path.parent
    candidate_root = candidate_path.parent
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if reference["video_sha256"] != candidate["video_sha256"]:
        raise ValueError("track manifests refer to different videos")
    if len(reference["frames"]) != len(candidate["frames"]):
        raise ValueError("track manifests contain different frame counts")

    rows = []
    for reference_row, candidate_row in zip(reference["frames"], candidate["frames"]):
        reference_key = (
            int(reference_row["sample_index"]),
            int(reference_row["source_frame_index"]),
        )
        candidate_key = (
            int(candidate_row["sample_index"]),
            int(candidate_row["source_frame_index"]),
        )
        if reference_key != candidate_key:
            raise ValueError("track manifests contain different frame identities")
        if reference_row["tracking_frame_sha256"] != candidate_row["tracking_frame_sha256"]:
            raise ValueError("track manifests contain different extracted frame pixels")
        reference_mask = cv2.imread(
            str(reference_root / reference_row["mask_path"]), cv2.IMREAD_GRAYSCALE
        )
        candidate_mask = cv2.imread(
            str(candidate_root / candidate_row["mask_path"]), cv2.IMREAD_GRAYSCALE
        )
        if reference_mask is None or candidate_mask is None:
            raise ValueError(f"cannot read masks for sample {reference_key[0]}")
        metrics = compare_masks(reference_mask > 0, candidate_mask > 0)
        rows.append(
            {
                "sample_index": reference_key[0],
                "source_frame_index": reference_key[1],
                "timestamp_seconds": float(reference_row["timestamp_seconds"]),
                "reference_mask_sha256": reference_row["mask_sha256"],
                "candidate_mask_sha256": candidate_row["mask_sha256"],
                "mask_hash_equal": (
                    reference_row["mask_sha256"] == candidate_row["mask_sha256"]
                ),
                **metrics,
            }
        )

    revision, dirty = git_state(Path.cwd())
    payload = {
        "status": "complete",
        "reference": {
            "track_candidate_id": reference["track_candidate_id"],
            "manifest": str(reference_path),
            "manifest_sha256": sha256(reference_path),
        },
        "candidate": {
            "track_candidate_id": candidate["track_candidate_id"],
            "manifest": str(candidate_path),
            "manifest_sha256": sha256(candidate_path),
        },
        "video_sha256": reference["video_sha256"],
        "code_revision": revision,
        "code_dirty": dirty,
        "summary": summarize(rows),
        "frames": rows,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
