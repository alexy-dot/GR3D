#!/usr/bin/env python3
"""Render GR3D-style semantic voxel components without object ID annotations."""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from render_canonical_views import camera_gravity_alignment
from export_scene_instances import validate_rendered_ids


LAYOUT_CLASSES = {
    "wall", "building", "sky", "floor", "ceiling", "road", "sidewalk",
    "grass", "earth", "mountain", "water", "sea", "field", "rock",
    "sand", "path", "stairs", "runway", "river", "bridge", "fence",
    "railing", "land", "tree", "door", "windowpane",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voxels", required=True, type=Path)
    parser.add_argument("--camera-poses", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--color-by", choices=("component", "semantic"), default="component"
    )
    parser.add_argument("--style", choices=("voxels", "boxes", "both"), default="both")
    parser.add_argument("--min-component-points", type=int, default=0)
    parser.add_argument("--objects-json", type=Path)
    parser.add_argument(
        "--scene-instances", type=Path,
        help="Optional scene_instances.json; adds S### labels only to these 3D renders",
    )
    parser.add_argument("--layer", choices=("all", "layout", "objects"), default="all")
    parser.add_argument("--clip-percentile", type=float, default=1.0)
    parser.add_argument("--marker-size", type=float, default=4.0)
    return parser.parse_args()


def palette(ids: np.ndarray) -> np.ndarray:
    unique = np.unique(ids)
    lookup = {}
    golden = 0.6180339887498949
    for value in unique:
        hue = (int(value) * golden + 0.11) % 1.0
        lookup[int(value)] = colorsys.hsv_to_rgb(hue, 0.70, 0.90)
    return np.asarray([lookup[int(value)] for value in ids], dtype=np.float64)


def render(
    centers: np.ndarray,
    colors: np.ndarray,
    cameras: np.ndarray,
    axes: tuple[int, int],
    labels: tuple[str, str],
    bounds: np.ndarray,
    output: Path,
    marker_size: float,
    component_ids: np.ndarray,
    style: str,
    annotations: dict[int, tuple[str, str, np.ndarray]] | None = None,
) -> None:
    first, second = axes
    fig, axis = plt.subplots(figsize=(8, 8), dpi=180)
    if style in {"voxels", "both"}:
        axis.scatter(
            centers[:, first], centers[:, second], c=colors, marker="s",
            s=marker_size, linewidths=0,
            alpha=0.45 if style == "both" else 0.88, rasterized=True
        )
    if style in {"boxes", "both"}:
        for component_id in np.unique(component_ids):
            selected = component_ids == component_id
            lower = centers[selected][:, [first, second]].min(axis=0)
            upper = centers[selected][:, [first, second]].max(axis=0)
            color = colors[np.flatnonzero(selected)[0]]
            axis.add_patch(
                Rectangle(
                    lower,
                    *(upper - lower),
                    fill=style == "boxes",
                    facecolor=(*color, 0.14) if style == "boxes" else "none",
                    edgecolor=color,
                    linewidth=1.0,
                )
            )
    axis.plot(
        cameras[:, first], cameras[:, second], "-o", color="#ff00aa",
        linewidth=1.0, markersize=2.5, label="camera path"
    )
    axis.set_xlim(bounds[0, first], bounds[1, first])
    axis.set_ylim(bounds[0, second], bounds[1, second])
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(labels[0])
    axis.set_ylabel(labels[1])
    axis.set_title(f"No-ID semantic voxel blocks: {labels[0]}-{labels[1]}")
    axis.grid(True, color="0.85", linewidth=0.5)
    axis.legend(loc="best", fontsize=8)
    if annotations:
        for order, component_id in enumerate(sorted(annotations)):
            scene_id, _, position = annotations[component_id]
            offset = (7 + (order % 3) * 5, 7 + ((order // 3) % 3) * 5)
            axis.annotate(
                scene_id,
                xy=(position[first], position[second]),
                xytext=offset,
                textcoords="offset points",
                fontsize=6.5,
                fontweight="bold",
                color="black",
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "black", "alpha": 0.9},
                arrowprops={"arrowstyle": "-", "color": "0.25", "lw": 0.45},
            )
        legend = "\n".join(
            f"{scene_id}  {semantic_name}"
            for _, (scene_id, semantic_name, _) in sorted(
                annotations.items(), key=lambda item: item[1][0]
            )
        )
        fig.text(
            0.805, 0.5, legend, ha="left", va="center", fontsize=6.2,
            family="monospace", bbox={"boxstyle": "round", "fc": "white", "ec": "0.7"},
        )
        axis.set_title(f"3D-only scene IDs: {labels[0]}-{labels[1]}")
        fig.tight_layout(rect=(0, 0, 0.79, 1))
    else:
        fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not 0 <= args.clip_percentile < 50:
        raise ValueError("--clip-percentile must be in [0, 50)")
    voxels_path = args.voxels.expanduser().resolve()
    poses_path = args.camera_poses.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    voxels = np.load(voxels_path)
    centers = voxels["centers"].astype(np.float64)
    component_ids = voxels["component_id"].astype(np.int64)
    semantic_ids = voxels["semantic_label"].astype(np.int64)
    point_counts = voxels["point_count"].astype(np.int64)
    objects_json = (
        args.objects_json.expanduser().resolve()
        if args.objects_json
        else voxels_path.parent / "objects.json"
    )
    semantic_names: dict[int, str] = {}
    if objects_json.exists():
        payload = json.loads(objects_json.read_text(encoding="utf-8"))
        for item in payload.get("objects", []):
            if item.get("semantic_name"):
                semantic_names[int(item["semantic_label"])] = item["semantic_name"]
    scene_instances_path = (
        args.scene_instances.expanduser().resolve() if args.scene_instances else None
    )
    scene_instances: list[dict[str, object]] = []
    component_to_instance: dict[int, dict[str, object]] = {}
    if scene_instances_path:
        scene_payload = json.loads(scene_instances_path.read_text(encoding="utf-8"))
        scene_instances = scene_payload["scene_instances"]
        component_to_instance = {
            int(item["source_component_id"]): item for item in scene_instances
        }
    if args.layer != "all":
        is_layout = np.asarray(
            [semantic_names.get(int(value), "") in LAYOUT_CLASSES for value in semantic_ids]
        )
        layer_keep = is_layout if args.layer == "layout" else ~is_layout
        centers = centers[layer_keep]
        component_ids = component_ids[layer_keep]
        semantic_ids = semantic_ids[layer_keep]
        point_counts = point_counts[layer_keep]
    if args.min_component_points > 0:
        retained = []
        for component_id in np.unique(component_ids):
            selected = component_ids == component_id
            if int(point_counts[selected].sum()) >= args.min_component_points:
                retained.append(component_id)
        keep = np.isin(component_ids, retained)
        centers = centers[keep]
        component_ids = component_ids[keep]
        semantic_ids = semantic_ids[keep]
        point_counts = point_counts[keep]
    if not len(centers):
        raise ValueError("No components survived --min-component-points")
    poses = np.load(poses_path).astype(np.float64)
    origin, basis, alignment = camera_gravity_alignment(centers, poses)
    aligned_centers = (centers - origin) @ basis
    cameras = (poses[:, :3, 3] - origin) @ basis
    annotations = None
    if scene_instances_path:
        surviving_components = {int(value) for value in np.unique(component_ids)}
        table_components = set(component_to_instance)
        if surviving_components != table_components:
            missing = sorted(table_components - surviving_components)
            extra = sorted(surviving_components - table_components)
            raise ValueError(
                f"scene table/render component mismatch; missing={missing}, extra={extra}"
            )
        annotations = {}
        for component_id, item in component_to_instance.items():
            raw_center = np.asarray(item["center_xyz"], dtype=np.float64)
            aligned_center = (raw_center - origin) @ basis
            annotations[component_id] = (
                str(item["scene_instance_id"]),
                str(item.get("semantic_name") or item["semantic_label"]),
                aligned_center,
            )
        validate_rendered_ids(
            scene_instances, [value[0] for value in annotations.values()]
        )
    color_ids = component_ids if args.color_by == "component" else semantic_ids
    colors = palette(color_ids)

    low = args.clip_percentile
    bounds = np.percentile(aligned_centers, [low, 100.0 - low], axis=0)
    padding = np.maximum((bounds[1] - bounds[0]) * 0.03, 1e-9)
    bounds[0] -= padding
    bounds[1] += padding
    specs = {
        "xy": ((0, 1), ("X", "Y")),
        "xz": ((0, 2), ("X", "Z")),
        "yz": ((1, 2), ("Y", "Z")),
    }
    for name, (axes, labels) in specs.items():
        render(
            aligned_centers, colors, cameras, axes, labels, bounds,
            output_dir / f"object_blocks_{name}.png", args.marker_size,
            component_ids, args.style, annotations
        )

    manifest = {
        "status": "complete",
        "representation": (
            "3d_only_scene_id_component_candidates"
            if scene_instances_path else "no_id_semantic_voxel_blocks"
        ),
        "color_by": args.color_by,
        "style": args.style,
        "layer": args.layer,
        "layout_class_policy": sorted(LAYOUT_CLASSES),
        "retained_semantic_labels": {
            str(value): semantic_names.get(int(value))
            for value in np.unique(semantic_ids)
        },
        "min_component_points": args.min_component_points,
        "voxel_count": int(len(centers)),
        "component_count": int(len(np.unique(component_ids))),
        "semantic_class_count": int(len(np.unique(semantic_ids))),
        "alignment": alignment,
        "clip_percentile": args.clip_percentile,
        "bounds_xyz": bounds.tolist(),
        "outputs": [f"object_blocks_{name}.png" for name in specs],
    }
    if scene_instances_path:
        manifest.update(
            {
                "scene_instances_sha256": hashlib.sha256(
                    scene_instances_path.read_bytes()
                ).hexdigest(),
                "rendered_ids": sorted(
                    (value[0] for value in annotations.values()),
                    key=lambda value: int(value[1:]),
                ),
                "id_placement": "aligned component center with leader line and side legend",
                "source_frames_modified": False,
                "candidate_warning": (
                    "Rendered S### labels identify semantic component candidates within this "
                    "bounded run; they are not verified physical instances."
                ),
            }
        )
    (output_dir / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Rendered {len(centers):,} voxels from "
        f"{manifest['component_count']:,} components"
    )


if __name__ == "__main__":
    main()
