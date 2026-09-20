#!/usr/bin/env python3
"""Render deterministic raw or camera-gravity-aligned orthographic PLY views."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PLY_TYPES = {
    "char": "i1",
    "uchar": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "float": "<f4",
    "double": "<f8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-cloud", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--camera-poses", type=Path)
    parser.add_argument(
        "--alignment",
        choices=("raw", "camera-gravity"),
        default="raw",
        help="camera-gravity assumes OpenCV C2W poses: +Y camera axis points down",
    )
    parser.add_argument("--max-points", type=int, default=250_000)
    parser.add_argument("--clip-percentile", type=float, default=1.0)
    parser.add_argument("--point-size", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_binary_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        first = handle.readline().decode("ascii").strip()
        if first != "ply":
            raise ValueError(f"Not a PLY file: {path}")

        vertex_count = None
        vertex_properties: list[tuple[str, str]] = []
        in_vertex = False
        file_format = None
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("PLY header ended unexpectedly")
            text = line.decode("ascii").strip()
            parts = text.split()
            if parts[:1] == ["format"]:
                file_format = parts[1]
            elif parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
                in_vertex = True
            elif parts[:1] == ["element"]:
                in_vertex = False
            elif parts[:1] == ["property"] and in_vertex:
                if parts[1] == "list":
                    raise ValueError("List-valued vertex properties are unsupported")
                if parts[1] not in PLY_TYPES:
                    raise ValueError(f"Unsupported PLY property type: {parts[1]}")
                vertex_properties.append((parts[2], PLY_TYPES[parts[1]]))
            elif text == "end_header":
                break

        if file_format != "binary_little_endian":
            raise ValueError("Only binary_little_endian PLY files are supported")
        if vertex_count is None:
            raise ValueError("PLY has no vertex element")

        vertices = np.fromfile(
            handle, dtype=np.dtype(vertex_properties), count=vertex_count
        )

    required = {"x", "y", "z"}
    if not required.issubset(vertices.dtype.names or ()):
        raise ValueError("PLY must contain x, y, and z vertex properties")
    points = np.column_stack([vertices[name] for name in ("x", "y", "z")])
    color_names = ("red", "green", "blue")
    if set(color_names).issubset(vertices.dtype.names or ()):
        colors = np.column_stack([vertices[name] for name in color_names]) / 255.0
    else:
        colors = np.full((len(points), 3), 0.25)
    finite = np.isfinite(points).all(axis=1)
    return points[finite].astype(np.float64), colors[finite].astype(np.float64)


def choose_points(
    points: np.ndarray, colors: np.ndarray, maximum: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if maximum <= 0:
        raise ValueError("--max-points must be positive")
    if len(points) <= maximum:
        return points, colors
    indices = np.random.default_rng(seed).choice(len(points), maximum, replace=False)
    indices.sort()
    return points[indices], colors[indices]


def load_camera_poses(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    poses = np.load(path)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("Camera poses must have shape (N, 4, 4)")
    poses = poses.astype(np.float64)
    if not np.isfinite(poses).all():
        raise ValueError("Camera poses contain non-finite values")
    return poses


def normalized(vector: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        raise ValueError(f"Cannot normalize near-zero {name}")
    return vector / norm


def camera_gravity_alignment(
    points: np.ndarray, poses: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Return origin, XYZ basis columns, and diagnostics.

    Pi3 exports OpenCV-style camera-to-world matrices. Camera +Y points down, so
    -R[:, 1] is a per-frame estimate of world up. Horizontal yaw is anchored by
    the first-to-last camera displacement, then mean viewing direction, then the
    dominant horizontal scene axis as deterministic fallbacks.
    """
    if len(poses) < 2:
        raise ValueError("camera-gravity alignment requires at least two poses")
    rotations = poses[:, :3, :3]
    centers = poses[:, :3, 3]
    ups = -rotations[:, :, 1]
    ups /= np.linalg.norm(ups, axis=1, keepdims=True).clip(min=1e-9)
    z_axis = normalized(np.median(ups, axis=0), "median camera up")

    dots = np.clip(ups @ z_axis, -1.0, 1.0)
    up_errors = np.degrees(np.arccos(dots))
    median_up_error = float(np.median(up_errors))
    p90_up_error = float(np.percentile(up_errors, 90))
    max_up_error = float(np.max(up_errors))
    origin = np.median(centers, axis=0)

    def horizontal(vector: np.ndarray) -> np.ndarray:
        return vector - z_axis * float(vector @ z_axis)

    heading = horizontal(centers[-1] - centers[0])
    heading_source = "first_to_last_camera_displacement"
    scene_scale = float(np.linalg.norm(np.ptp(centers, axis=0)))
    if np.linalg.norm(heading) < max(1e-6, scene_scale * 0.05):
        heading = horizontal(np.mean(rotations[:, :, 2], axis=0))
        heading_source = "mean_camera_forward"
    if np.linalg.norm(heading) < 1e-6:
        centered = points - np.median(points, axis=0)
        projected = centered - np.outer(centered @ z_axis, z_axis)
        covariance = projected.T @ projected / max(len(projected), 1)
        values, vectors = np.linalg.eigh(covariance)
        heading = vectors[:, int(np.argmax(values))]
        mean_forward = horizontal(np.mean(rotations[:, :, 2], axis=0))
        if np.dot(heading, mean_forward) < 0:
            heading = -heading
        heading_source = "dominant_horizontal_scene_axis"

    x_axis = normalized(horizontal(heading), "horizontal heading")
    y_axis = normalized(np.cross(z_axis, x_axis), "horizontal side axis")
    x_axis = normalized(np.cross(y_axis, z_axis), "orthogonal heading axis")
    basis = np.column_stack((x_axis, y_axis, z_axis))
    diagnostics: dict[str, object] = {
        "pose_convention": "OpenCV camera-to-world; camera +Y is down",
        "origin_source": "median_camera_center",
        "origin_world": origin.tolist(),
        "basis_world_columns_xyz": basis.tolist(),
        "heading_source": heading_source,
        "camera_up_error_degrees": {
            "median": median_up_error,
            "p90": p90_up_error,
            "max": max_up_error,
        },
        "quality": (
            "warning_inconsistent_camera_up"
            if median_up_error > 10.0 or p90_up_error > 20.0
            else "pass"
        ),
    }
    return origin, basis, diagnostics


def render_view(
    points: np.ndarray,
    colors: np.ndarray,
    cameras: np.ndarray | None,
    axes: tuple[int, int],
    labels: tuple[str, str],
    bounds: np.ndarray,
    output: Path,
    point_size: float,
    coordinate_label: str,
) -> None:
    x_axis, y_axis = axes
    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    ax.scatter(
        points[:, x_axis],
        points[:, y_axis],
        c=np.clip(colors, 0.0, 1.0),
        s=point_size,
        linewidths=0,
        rasterized=True,
    )
    if cameras is not None and len(cameras):
        ax.plot(
            cameras[:, x_axis], cameras[:, y_axis], "-o", color="#ff00aa",
            linewidth=1.1, markersize=2.8, label="camera path"
        )
        ax.legend(loc="best", fontsize=8)
    ax.set_xlim(bounds[0, x_axis], bounds[1, x_axis])
    ax.set_ylim(bounds[0, y_axis], bounds[1, y_axis])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(f"{coordinate_label} {labels[0]}-{labels[1]} orthographic view")
    ax.grid(True, color="0.85", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cloud_path = args.point_cloud.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    pose_path = args.camera_poses.expanduser().resolve() if args.camera_poses else None
    if not 0.0 <= args.clip_percentile < 50.0:
        raise ValueError("--clip-percentile must be in [0, 50)")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_points, all_colors = load_binary_ply(cloud_path)
    if not len(all_points):
        raise ValueError("Point cloud contains no finite points")
    points, colors = choose_points(all_points, all_colors, args.max_points, args.seed)
    poses = load_camera_poses(pose_path)
    cameras = poses[:, :3, 3] if poses is not None else None
    alignment_details = None
    coordinate_system = "raw_model_coordinates_not_gravity_aligned"
    coordinate_label = "Raw model-coordinate"
    if args.alignment == "camera-gravity":
        if poses is None:
            raise ValueError("--alignment camera-gravity requires --camera-poses")
        origin, basis, alignment_details = camera_gravity_alignment(all_points, poses)
        all_points = (all_points - origin) @ basis
        points = (points - origin) @ basis
        cameras = (cameras - origin) @ basis
        coordinate_system = "camera_gravity_aligned_z_up"
        coordinate_label = "Camera-gravity-aligned"

    low = args.clip_percentile
    high = 100.0 - low
    bounds = np.percentile(all_points, [low, high], axis=0)
    spans = bounds[1] - bounds[0]
    padding = np.maximum(spans * 0.03, 1e-9)
    bounds[0] -= padding
    bounds[1] += padding

    view_specs = {
        "xy": ((0, 1), ("X", "Y")),
        "xz": ((0, 2), ("X", "Z")),
        "yz": ((1, 2), ("Y", "Z")),
    }
    for name, (axes, labels) in view_specs.items():
        render_view(
            points, colors, cameras, axes, labels, bounds,
            output_dir / f"view_{name}.png", args.point_size, coordinate_label
        )

    manifest = {
        "status": "complete",
        "coordinate_system": coordinate_system,
        "alignment": args.alignment,
        "alignment_details": alignment_details,
        "point_cloud": str(cloud_path),
        "point_cloud_sha256": sha256(cloud_path),
        "camera_poses": str(pose_path) if pose_path else None,
        "source_point_count": int(len(all_points)),
        "rendered_point_count": int(len(points)),
        "max_points": args.max_points,
        "seed": args.seed,
        "clip_percentile": args.clip_percentile,
        "bounds_xyz": bounds.tolist(),
        "outputs": [f"view_{name}.png" for name in view_specs],
    }
    (output_dir / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Rendered {len(points):,} of {len(all_points):,} points to {output_dir}")


if __name__ == "__main__":
    main()
