#!/usr/bin/env python3
"""Render a tracked 3D trajectory over Pi3 observations in XY/XZ/YZ views."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

try:
    from reconstruction.render_canonical_views import camera_gravity_alignment
    from reconstruction.tracking_contracts import (
        normalize_external_identity,
        validate_coordinate_system,
    )
except ModuleNotFoundError:
    from render_canonical_views import camera_gravity_alignment
    from tracking_contracts import normalize_external_identity, validate_coordinate_system


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_by_frame(path: Path, point_count: int) -> dict[int, np.ndarray]:
    with np.load(path) as archive:
        selected = {int(name): archive[name].astype(np.int64) for name in archive.files}
    for frame, indices in selected.items():
        if np.any(indices < 0) or np.any(indices >= point_count):
            raise ValueError(f"selected point index for frame {frame} is out of bounds")
    return selected


def deterministic_sample(indices: np.ndarray, limit: int) -> np.ndarray:
    if limit < 1:
        raise ValueError("sample limit must be positive")
    if len(indices) <= limit:
        return indices
    return indices[np.linspace(0, len(indices) - 1, limit, dtype=np.int64)]


def retained_static_indices(
    point_count: int, tracked_indices: np.ndarray, motion_state: str
) -> np.ndarray:
    if motion_state not in {"static", "dynamic", "uncertain"}:
        raise ValueError(f"unsupported motion state: {motion_state}")
    mask = np.ones(point_count, dtype=bool)
    if motion_state in {"dynamic", "uncertain"}:
        mask[tracked_indices] = False
    return np.flatnonzero(mask)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path)
    parser.add_argument("selected_indices", type=Path)
    parser.add_argument("track_states", type=Path)
    parser.add_argument("classification", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--entity-id")
    parser.add_argument("--camera-poses", type=Path)
    parser.add_argument("--track-audit", type=Path)
    parser.add_argument(
        "--alignment",
        choices=("raw", "camera-gravity"),
        default="raw",
    )
    parser.add_argument("--max-background-points", type=int, default=100000)
    parser.add_argument("--max-dynamic-points", type=int, default=60000)
    args = parser.parse_args()

    observations_path = args.observations.resolve()
    selected_path = args.selected_indices.resolve()
    states_path = args.track_states.resolve()
    classification_path = args.classification.resolve()
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)

    with np.load(observations_path) as archive:
        points = archive["points"].astype(np.float64)
    states_payload = json.loads(states_path.read_text(encoding="utf-8"))
    classification_payload = json.loads(classification_path.read_text(encoding="utf-8"))
    coordinate_system = validate_coordinate_system(states_payload.get("coordinate_system"))
    if validate_coordinate_system(classification_payload.get("coordinate_system")) != coordinate_system:
        raise ValueError("classification and track states use different coordinate systems")
    external_identity = normalize_external_identity(states_payload.get("external_identity"))
    if normalize_external_identity(classification_payload.get("external_identity")) != external_identity:
        raise ValueError("classification and track states use different external identities")
    state_provenance = states_payload.get("provenance", {})
    classification_provenance = classification_payload.get("input_provenance", {})
    if state_provenance.get("point_observations_sha256") != sha256(observations_path):
        raise ValueError("track states belong to different Pi3 observations")
    if state_provenance.get("selected_point_indices_sha256") != sha256(selected_path):
        raise ValueError("selected indices do not belong to these track states")
    if classification_provenance.get("track_states_sha256") != sha256(states_path):
        raise ValueError("classification belongs to different track states")
    track_audit_path = args.track_audit.resolve() if args.track_audit else None
    track_audit = None
    if external_identity is not None and external_identity["verified"]:
        if track_audit_path is None:
            raise ValueError("verified external identity requires --track-audit")
        track_audit = json.loads(track_audit_path.read_text(encoding="utf-8"))
        if track_audit.get("track_candidate_id") != states_payload["track_candidate_id"]:
            raise ValueError("track audit belongs to another candidate")
        if normalize_external_identity(track_audit.get("external_identity")) != external_identity:
            raise ValueError("track audit external identity mismatch")
        if track_audit.get("track_manifest_sha256") != state_provenance.get(
            "track_manifest_sha256"
        ):
            raise ValueError("track audit and lifted states use different track manifests")
        if track_audit.get("source_frames_modified") is not False:
            raise ValueError("track audit must preserve source frames")
        contact_sheet_path = track_audit_path.parent / track_audit["contact_sheet"]["path"]
        if sha256(contact_sheet_path) != track_audit["contact_sheet"]["sha256"]:
            raise ValueError("track audit contact-sheet hash mismatch")
    motion_state = classification_payload["classification"]["motion_state"]
    selected = selected_by_frame(selected_path, len(points))
    tracked_indices = np.unique(
        np.concatenate(list(selected.values())) if selected else np.empty(0, dtype=np.int64)
    )
    static_indices = deterministic_sample(
        retained_static_indices(len(points), tracked_indices, motion_state),
        args.max_background_points,
    )

    expected_prefix = {"static": "S", "dynamic": "D", "uncertain": "U"}[motion_state]
    entity_id = args.entity_id or f"{expected_prefix}001"
    if not entity_id.startswith(expected_prefix):
        raise ValueError(f"entity ID {entity_id} does not match motion state {motion_state}")
    all_states = sorted(
        states_payload["states"],
        key=lambda row: (float(row["timestamp_seconds"]), int(row["sample_index"])),
    )
    states = [row for row in all_states if row.get("center_xyz_median") is not None]
    if not states:
        raise ValueError("track contains no valid 3D states")
    centers = np.asarray([row["center_xyz_median"] for row in states], dtype=np.float64)
    all_camera_centers = np.asarray(
        [row.get("camera_center_xyz") for row in all_states], dtype=np.float64
    )
    if all_camera_centers.shape != (len(all_states), 3) or not np.isfinite(
        all_camera_centers
    ).all():
        raise ValueError("all track states require finite camera_center_xyz")
    valid_positions = [
        index for index, row in enumerate(all_states)
        if row.get("center_xyz_median") is not None
    ]
    camera_centers = all_camera_centers[valid_positions]
    times = np.asarray([row["timestamp_seconds"] for row in states], dtype=np.float64)

    alignment_details = None
    camera_poses_path = args.camera_poses.resolve() if args.camera_poses else None
    if args.alignment == "camera-gravity":
        if camera_poses_path is None:
            raise ValueError("camera-gravity alignment requires --camera-poses")
        if state_provenance.get("camera_poses_sha256") != sha256(camera_poses_path):
            raise ValueError("camera poses differ from the file used for 3D lifting")
        poses = np.load(camera_poses_path)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError("camera poses must have shape (N, 4, 4)")
        if len(poses) != len(all_states):
            raise ValueError("camera pose count differs from trajectory state count")
        if not np.isfinite(poses).all():
            raise ValueError("camera poses contain non-finite values")
        origin, basis, alignment_details = camera_gravity_alignment(points, poses)
        points = (points - origin) @ basis
        centers = (centers - origin) @ basis
        all_camera_centers = (all_camera_centers - origin) @ basis
        camera_centers = (camera_centers - origin) @ basis
        output_coordinate_system = {
            **coordinate_system,
            "source_coordinate_system_name": coordinate_system["name"],
            "name": f"camera_gravity_aligned_{coordinate_system['name']}",
            "alignment": "camera-gravity",
            "alignment_details": alignment_details,
        }
    else:
        output_coordinate_system = {
            **coordinate_system,
            "source_coordinate_system_name": coordinate_system["name"],
            "alignment": "raw",
            "alignment_details": None,
        }

    frame_times = {
        int(row["sample_index"]): float(row["timestamp_seconds"])
        for row in all_states
    }
    missing_times = sorted(set(selected) - set(frame_times))
    if missing_times:
        raise ValueError(f"selected indices have no matching 3D state times: {missing_times}")
    point_times = np.concatenate([
        np.full(len(indices), frame_times.get(frame, np.nan), dtype=np.float64)
        for frame, indices in selected.items()
    ]) if selected else np.empty(0, dtype=np.float64)
    all_dynamic_indices = np.concatenate(list(selected.values())) if selected else np.empty(0, dtype=np.int64)
    if len(all_dynamic_indices) > args.max_dynamic_points:
        sample_positions = np.linspace(0, len(all_dynamic_indices) - 1, args.max_dynamic_points, dtype=np.int64)
        tracked_plot_indices = all_dynamic_indices[sample_positions]
        point_times = point_times[sample_positions]
    else:
        tracked_plot_indices = all_dynamic_indices

    projections = [(0, 1, "X", "Y"), (0, 2, "X", "Z"), (1, 2, "Y", "Z")]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    color_values = (times - times.min()) / max(float(np.ptp(times)), 1e-12)
    scatter = None
    for axis, (first, second, first_name, second_name) in zip(axes, projections):
        axis.scatter(points[static_indices, first], points[static_indices, second], s=0.35, c="#b8bec7", alpha=0.24, rasterized=True)
        if len(tracked_plot_indices):
            scatter = axis.scatter(
                points[tracked_plot_indices, first],
                points[tracked_plot_indices, second],
                s=0.8,
                c=point_times,
                cmap="viridis",
                alpha=0.35,
                rasterized=True,
            )
        axis.plot(centers[:, first], centers[:, second], color="#111827", linewidth=1.5, zorder=4)
        axis.scatter(centers[:, first], centers[:, second], s=38, c=color_values, cmap="viridis", edgecolors="white", linewidths=0.6, zorder=5)
        for ordinal, (x_value, y_value) in enumerate(centers[:, [first, second]]):
            axis.annotate(str(ordinal), (x_value, y_value), xytext=(3, 3), textcoords="offset points", fontsize=7)
        axis.set_xlabel(f"{first_name} (Pi3 model units)")
        axis.set_ylabel(f"{second_name} (Pi3 model units)")
        axis.set_title(f"{first_name}{second_name}")
        axis.grid(True, linewidth=0.4, alpha=0.35)
        axis.set_aspect("equal", adjustable="datalim")
    if scatter is not None:
        figure.colorbar(scatter, ax=axes, label="Time (seconds)", shrink=0.8)
    identity_label = (
        f"OSI-{external_identity['object_id']}"
        if external_identity is not None
        else "unbound benchmark ID"
    )
    figure.suptitle(
        f"{identity_label} / {entity_id} trajectory: {motion_state} "
        f"(source {states_payload['track_candidate_id']})"
    )
    render_path = output / "dynamic_trajectory_xyz.png"
    figure.savefig(render_path, dpi=180)
    plt.close(figure)

    raw_indices = deterministic_sample(np.arange(len(points), dtype=np.int64), args.max_background_points)
    comparison, comparison_axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for column, (first, second, first_name, second_name) in enumerate(projections):
        raw_axis = comparison_axes[0, column]
        static_axis = comparison_axes[1, column]
        raw_axis.scatter(points[raw_indices, first], points[raw_indices, second], s=0.4, c="#aeb6c2", alpha=0.28, rasterized=True)
        raw_axis.scatter(points[tracked_plot_indices, first], points[tracked_plot_indices, second], s=0.8, c="#dc2626", alpha=0.38, rasterized=True)
        static_axis.scatter(points[static_indices, first], points[static_indices, second], s=0.4, c="#667085", alpha=0.3, rasterized=True)
        if motion_state == "static":
            static_axis.scatter(points[tracked_plot_indices, first], points[tracked_plot_indices, second], s=0.8, c="#16a34a", alpha=0.4, rasterized=True)
        bounds_first = np.quantile(points[:, first], [0.01, 0.99])
        bounds_second = np.quantile(points[:, second], [0.01, 0.99])
        for axis in (raw_axis, static_axis):
            axis.set_xlim(*bounds_first)
            axis.set_ylim(*bounds_second)
            axis.set_xlabel(f"{first_name} (Pi3 model units)")
            axis.set_ylabel(f"{second_name} (Pi3 model units)")
            axis.grid(True, linewidth=0.4, alpha=0.35)
            axis.set_aspect("equal", adjustable="box")
        raw_axis.set_title(f"Unfiltered {first_name}{second_name}; tracked points in red")
        suffix = "; tracked points retained" if motion_state == "static" else ""
        static_axis.set_title(f"Filtered static map {first_name}{second_name}{suffix}")
    comparison.suptitle(f"{entity_id} dynamic-point filtering audit")
    comparison_path = output / "static_filter_comparison_xyz.png"
    comparison.savefig(comparison_path, dpi=180)
    plt.close(comparison)

    table_path = output / "trajectory.csv"
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        center_by_sample = {
            int(row["sample_index"]): center
            for row, center in zip(states, centers, strict=True)
        }
        writer.writerow([
            "ordinal", "sample_index", "source_frame_index", "timestamp_seconds",
            "valid_3d", "x", "y", "z", "camera_x", "camera_y", "camera_z",
            "point_count", "quality_warnings",
        ])
        for ordinal, (row, camera_center) in enumerate(
            zip(all_states, all_camera_centers, strict=True)
        ):
            center = center_by_sample.get(int(row["sample_index"]))
            writer.writerow([
                ordinal,
                row["sample_index"],
                row["source_frame_index"],
                row["timestamp_seconds"],
                center is not None,
                *(center.tolist() if center is not None else ["", "", ""]),
                *camera_center.tolist(),
                row["point_count"],
                json.dumps(row.get("quality_warnings", []), separators=(",", ":")),
            ])

    manifest = {
        "status": "complete",
        "track_candidate_id": states_payload["track_candidate_id"],
        "external_identity": external_identity,
        "entity_id": entity_id,
        "motion_state": motion_state,
        "coordinate_system": output_coordinate_system,
        "axis_values_are_meters": False,
        "metric_scale_validated": False,
        "object_id_correspondence": bool(
            external_identity is not None and external_identity["verified"]
        ),
        "time_conditioned": True,
        "valid_state_count": len(states),
        "invalid_state_count": len(all_states) - len(states),
        "source_point_count": len(points),
        "selected_tracked_point_count": int(len(np.unique(all_dynamic_indices))),
        "excluded_from_static_point_count": int(len(tracked_indices)) if motion_state != "static" else 0,
        "parameters": {
            "max_background_points": args.max_background_points,
            "max_dynamic_points": args.max_dynamic_points,
        },
        "provenance": {
            "source_pi3_manifest_sha256": states_payload.get("provenance", {}).get(
                "pi3_manifest_sha256"
            ),
            "observations_sha256": sha256(observations_path),
            "selected_indices_sha256": sha256(selected_path),
            "track_states_sha256": sha256(states_path),
            "classification_sha256": sha256(classification_path),
            "camera_poses_sha256": (
                sha256(camera_poses_path) if camera_poses_path is not None else None
            ),
            "track_audit": str(track_audit_path) if track_audit_path else None,
            "track_audit_sha256": sha256(track_audit_path) if track_audit_path else None,
            "identity_contact_sheet": (
                str(track_audit_path.parent / track_audit["contact_sheet"]["path"])
                if track_audit is not None
                else None
            ),
            "identity_contact_sheet_sha256": (
                track_audit["contact_sheet"]["sha256"] if track_audit is not None else None
            ),
        },
        "outputs": {
            "render": {"path": render_path.name, "sha256": sha256(render_path)},
            "filter_comparison": {"path": comparison_path.name, "sha256": sha256(comparison_path)},
            "table": {"path": table_path.name, "sha256": sha256(table_path)},
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
