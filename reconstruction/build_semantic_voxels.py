#!/usr/bin/env python3
"""Fuse per-frame semantic labels into connected 3D voxel object blocks."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np


NEIGHBORS = np.asarray(
    ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)),
    dtype=np.int64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument(
        "--semantic-labels",
        required=True,
        type=Path,
        help="NPY array with shape (frames, height, width) and integer class IDs",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--voxel-size", required=True, type=float)
    parser.add_argument("--ignore-label", type=int, default=-1)
    parser.add_argument("--min-voxel-points", type=int, default=2)
    parser.add_argument("--min-voxel-views", type=int, default=1)
    parser.add_argument("--min-voxel-purity", type=float, default=0.0)
    parser.add_argument("--min-component-voxels", type=int, default=8)
    parser.add_argument(
        "--label-remap",
        type=Path,
        help=(
            "Optional JSON semantic-group remap. This is a practical extension and "
            "must be reported separately from the original ADE20K-label baseline."
        ),
    )
    return parser.parse_args()


def apply_label_remap(
    labels: np.ndarray, config: dict[str, object]
) -> tuple[np.ndarray, dict[str, str], list[dict[str, object]]]:
    """Apply declared label groups and return labels, name overrides, and audit records."""
    remapped = labels.copy()
    name_overrides: dict[str, str] = {}
    records: list[dict[str, object]] = []
    for group in config.get("groups", []):
        if not isinstance(group, dict):
            raise ValueError("label-remap groups must be JSON objects")
        name = str(group["name"])
        target = int(group["target_label"])
        sources = sorted({int(value) for value in group["source_labels"]})
        if target not in sources:
            sources.append(target)
            sources.sort()
        mask = np.isin(remapped, sources)
        changed = int(np.count_nonzero(mask & (remapped != target)))
        matched = int(np.count_nonzero(mask))
        remapped[mask] = target
        name_overrides[str(target)] = name
        records.append(
            {
                "name": name,
                "target_label": target,
                "source_labels": sources,
                "matched_observations": matched,
                "changed_observations": changed,
            }
        )
    return remapped, name_overrides, records


def majority_labels(
    inverse: np.ndarray, labels: np.ndarray, voxel_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return majority label and its vote count for every voxel."""
    order = np.lexsort((labels, inverse))
    voxel_sorted = inverse[order]
    label_sorted = labels[order]
    pair_start = np.empty(len(order), dtype=bool)
    pair_start[0] = True
    pair_start[1:] = (voxel_sorted[1:] != voxel_sorted[:-1]) | (
        label_sorted[1:] != label_sorted[:-1]
    )
    starts = np.flatnonzero(pair_start)
    counts = np.diff(np.append(starts, len(order)))
    pair_voxels = voxel_sorted[starts]
    pair_labels = label_sorted[starts]

    majority = np.full(voxel_count, -1, dtype=np.int32)
    majority_votes = np.zeros(voxel_count, dtype=np.int64)
    for voxel, label, count in zip(pair_voxels, pair_labels, counts, strict=True):
        if count > majority_votes[voxel]:
            majority[voxel] = label
            majority_votes[voxel] = count
    return majority, majority_votes


def connected_components(
    coordinates: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, int]:
    lookup = {tuple(coord): index for index, coord in enumerate(coordinates)}
    component = np.full(len(coordinates), -1, dtype=np.int32)
    component_id = 0
    for start in range(len(coordinates)):
        if component[start] >= 0:
            continue
        target_label = labels[start]
        component[start] = component_id
        queue: deque[int] = deque([start])
        while queue:
            current = queue.popleft()
            for offset in NEIGHBORS:
                neighbor = lookup.get(tuple(coordinates[current] + offset))
                if (
                    neighbor is not None
                    and component[neighbor] < 0
                    and labels[neighbor] == target_label
                ):
                    component[neighbor] = component_id
                    queue.append(neighbor)
        component_id += 1
    return component, component_id


def main() -> None:
    args = parse_args()
    if args.voxel_size <= 0:
        raise ValueError("--voxel-size must be positive")
    if args.min_voxel_points < 1 or args.min_voxel_views < 1 or args.min_component_voxels < 1:
        raise ValueError("minimum counts must be positive")
    if not 0.0 <= args.min_voxel_purity <= 1.0:
        raise ValueError("--min-voxel-purity must be in [0, 1]")

    observations_path = args.observations.expanduser().resolve()
    labels_path = args.semantic_labels.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    observations = np.load(observations_path)
    semantic = np.load(labels_path)
    label_metadata_path = labels_path.with_suffix(".json")
    label_names: dict[str, str] = {}
    if label_metadata_path.exists():
        label_metadata = json.loads(label_metadata_path.read_text(encoding="utf-8"))
        label_names = label_metadata.get("id2label", {})
    if semantic.ndim != 3 or not np.issubdtype(semantic.dtype, np.integer):
        raise ValueError("semantic labels must be an integer array shaped (F, H, W)")
    points = observations["points"].astype(np.float64)
    frames = observations["frame_index"].astype(np.int64)
    pixels = observations["pixel_yx"].astype(np.int64)
    if len(points) != len(frames) or len(points) != len(pixels):
        raise ValueError("observation arrays have inconsistent lengths")
    if frames.max() >= semantic.shape[0]:
        raise ValueError("semantic labels have fewer frames than observations")
    if pixels[:, 0].max() >= semantic.shape[1] or pixels[:, 1].max() >= semantic.shape[2]:
        raise ValueError("semantic label resolution does not match observations")

    point_labels = semantic[frames, pixels[:, 0], pixels[:, 1]].astype(np.int32)
    remap_path = None
    remap_records: list[dict[str, object]] = []
    if args.label_remap:
        remap_path = args.label_remap.expanduser().resolve()
        remap_config = json.loads(remap_path.read_text(encoding="utf-8"))
        point_labels, name_overrides, remap_records = apply_label_remap(
            point_labels, remap_config
        )
        label_names.update(name_overrides)
    keep_points = point_labels != args.ignore_label
    points = points[keep_points]
    point_labels = point_labels[keep_points]
    frames = frames[keep_points]
    if not len(points):
        raise ValueError("all observations were removed by --ignore-label")

    voxel_coordinates, inverse = np.unique(
        np.floor(points / args.voxel_size).astype(np.int64),
        axis=0,
        return_inverse=True,
    )
    point_counts = np.bincount(inverse, minlength=len(voxel_coordinates))
    voxel_frame_pairs = np.unique(np.column_stack((inverse, frames)), axis=0)
    view_counts = np.bincount(
        voxel_frame_pairs[:, 0], minlength=len(voxel_coordinates)
    )
    voxel_labels, majority_votes = majority_labels(
        inverse, point_labels, len(voxel_coordinates)
    )
    purity = majority_votes / np.maximum(point_counts, 1)
    keep_voxels = (
        (point_counts >= args.min_voxel_points)
        & (view_counts >= args.min_voxel_views)
        & (purity >= args.min_voxel_purity)
    )
    voxel_coordinates = voxel_coordinates[keep_voxels]
    voxel_labels = voxel_labels[keep_voxels]
    point_counts = point_counts[keep_voxels]
    majority_votes = majority_votes[keep_voxels]
    view_counts = view_counts[keep_voxels]
    purity = purity[keep_voxels]

    component_raw, raw_count = connected_components(voxel_coordinates, voxel_labels)
    raw_sizes = np.bincount(component_raw, minlength=raw_count)
    keep_component = raw_sizes >= args.min_component_voxels
    keep_connected_voxels = keep_component[component_raw]
    voxel_coordinates = voxel_coordinates[keep_connected_voxels]
    voxel_labels = voxel_labels[keep_connected_voxels]
    point_counts = point_counts[keep_connected_voxels]
    majority_votes = majority_votes[keep_connected_voxels]
    view_counts = view_counts[keep_connected_voxels]
    purity = purity[keep_connected_voxels]
    component_raw = component_raw[keep_connected_voxels]

    retained_raw_ids = np.flatnonzero(keep_component)
    remap = np.full(raw_count, -1, dtype=np.int32)
    remap[retained_raw_ids] = np.arange(len(retained_raw_ids), dtype=np.int32)
    components = remap[component_raw]
    centers = (voxel_coordinates.astype(np.float64) + 0.5) * args.voxel_size

    objects = []
    for component_id in range(len(retained_raw_ids)):
        selected = components == component_id
        component_centers = centers[selected]
        weights = point_counts[selected].astype(np.float64)
        semantic_label = int(voxel_labels[selected][0])
        objects.append(
            {
                "component_id": component_id,
                "semantic_label": semantic_label,
                "semantic_name": label_names.get(str(semantic_label)),
                "voxel_count": int(selected.sum()),
                "point_count": int(point_counts[selected].sum()),
                "mean_voxel_purity": float(
                    np.average(purity[selected], weights=point_counts[selected])
                ),
                "max_view_support": int(view_counts[selected].max()),
                "multiview_voxel_ratio": float(np.mean(view_counts[selected] >= 2)),
                "center_xyz": np.average(component_centers, axis=0, weights=weights).tolist(),
                "bbox_min_xyz": (component_centers.min(axis=0) - args.voxel_size / 2).tolist(),
                "bbox_max_xyz": (component_centers.max(axis=0) + args.voxel_size / 2).tolist(),
            }
        )

    np.savez_compressed(
        output_dir / "semantic_voxels.npz",
        coordinates=voxel_coordinates,
        centers=centers.astype(np.float32),
        semantic_label=voxel_labels,
        point_count=point_counts.astype(np.int32),
        majority_votes=majority_votes.astype(np.int32),
        view_count=view_counts.astype(np.int16),
        purity=purity.astype(np.float32),
        component_id=components,
    )
    weighted_purity = float(np.average(purity, weights=point_counts)) if len(purity) else 0.0
    multiview_ratio = float(np.mean(view_counts >= 2)) if len(view_counts) else 0.0
    quality_warnings = []
    if weighted_purity < 0.75:
        quality_warnings.append("low_weighted_semantic_purity")
    if multiview_ratio < 0.25:
        quality_warnings.append("low_multiview_voxel_support")
    if len(objects) > 100:
        quality_warnings.append("high_component_fragmentation")
    manifest = {
        "status": "complete",
        "observations": str(observations_path),
        "semantic_labels": str(labels_path),
        "label_remap": str(remap_path) if remap_path else None,
        "label_remap_groups": remap_records,
        "voxel_size": args.voxel_size,
        "ignore_label": args.ignore_label,
        "min_voxel_points": args.min_voxel_points,
        "min_voxel_views": args.min_voxel_views,
        "min_voxel_purity": args.min_voxel_purity,
        "min_component_voxels": args.min_component_voxels,
        "input_observations": int(len(keep_points)),
        "labeled_observations": int(keep_points.sum()),
        "retained_voxels": int(len(voxel_coordinates)),
        "retained_components": int(len(objects)),
        "quality": {
            "status": "warning" if quality_warnings else "pass",
            "weighted_semantic_purity": weighted_purity,
            "multiview_voxel_ratio": multiview_ratio,
            "median_view_support": float(np.median(view_counts)) if len(view_counts) else 0.0,
            "warnings": quality_warnings,
            "note": "These are internal consistency checks, not geometric accuracy proof.",
        },
        "outputs": ["semantic_voxels.npz", "objects.json"],
    }
    (output_dir / "objects.json").write_text(
        json.dumps({"manifest": manifest, "objects": objects}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Built {len(voxel_coordinates):,} voxels in {len(objects):,} retained components"
    )


if __name__ == "__main__":
    main()
