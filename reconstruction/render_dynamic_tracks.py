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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path)
    parser.add_argument("selected_indices", type=Path)
    parser.add_argument("track_states", type=Path)
    parser.add_argument("classification", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--entity-id")
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
    selected = selected_by_frame(selected_path, len(points))
    dynamic_indices = np.unique(
        np.concatenate(list(selected.values())) if selected else np.empty(0, dtype=np.int64)
    )
    static_mask = np.ones(len(points), dtype=bool)
    static_mask[dynamic_indices] = False
    static_indices = deterministic_sample(np.flatnonzero(static_mask), args.max_background_points)
    dynamic_indices = deterministic_sample(dynamic_indices, args.max_dynamic_points)

    states_payload = json.loads(states_path.read_text(encoding="utf-8"))
    classification_payload = json.loads(classification_path.read_text(encoding="utf-8"))
    motion_state = classification_payload["classification"]["motion_state"]
    expected_prefix = {"static": "S", "dynamic": "D", "uncertain": "U"}[motion_state]
    entity_id = args.entity_id or f"{expected_prefix}001"
    if not entity_id.startswith(expected_prefix):
        raise ValueError(f"entity ID {entity_id} does not match motion state {motion_state}")
    states = [row for row in states_payload["states"] if row.get("center_xyz_median") is not None]
    states.sort(key=lambda row: (float(row["timestamp_seconds"]), int(row["sample_index"])))
    if not states:
        raise ValueError("track contains no valid 3D states")
    centers = np.asarray([row["center_xyz_median"] for row in states], dtype=np.float64)
    times = np.asarray([row["timestamp_seconds"] for row in states], dtype=np.float64)

    frame_times = {int(row["sample_index"]): float(row["timestamp_seconds"]) for row in states}
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
        dynamic_plot_indices = all_dynamic_indices[sample_positions]
        point_times = point_times[sample_positions]
    else:
        dynamic_plot_indices = all_dynamic_indices

    projections = [(0, 1, "X", "Y"), (0, 2, "X", "Z"), (1, 2, "Y", "Z")]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    color_values = (times - times.min()) / max(float(np.ptp(times)), 1e-12)
    scatter = None
    for axis, (first, second, first_name, second_name) in zip(axes, projections):
        axis.scatter(points[static_indices, first], points[static_indices, second], s=0.35, c="#b8bec7", alpha=0.24, rasterized=True)
        if len(dynamic_plot_indices):
            scatter = axis.scatter(
                points[dynamic_plot_indices, first],
                points[dynamic_plot_indices, second],
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
    figure.suptitle(f"{entity_id} trajectory: {motion_state} (source {states_payload['track_candidate_id']})")
    render_path = output / "dynamic_trajectory_xyz.png"
    figure.savefig(render_path, dpi=180)
    plt.close(figure)

    raw_indices = deterministic_sample(np.arange(len(points), dtype=np.int64), args.max_background_points)
    comparison, comparison_axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for column, (first, second, first_name, second_name) in enumerate(projections):
        raw_axis = comparison_axes[0, column]
        static_axis = comparison_axes[1, column]
        raw_axis.scatter(points[raw_indices, first], points[raw_indices, second], s=0.4, c="#aeb6c2", alpha=0.28, rasterized=True)
        raw_axis.scatter(points[dynamic_plot_indices, first], points[dynamic_plot_indices, second], s=0.8, c="#dc2626", alpha=0.38, rasterized=True)
        static_axis.scatter(points[static_indices, first], points[static_indices, second], s=0.4, c="#667085", alpha=0.3, rasterized=True)
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
        static_axis.set_title(f"Filtered static map {first_name}{second_name}")
    comparison.suptitle(f"{entity_id} dynamic-point filtering audit")
    comparison_path = output / "static_filter_comparison_xyz.png"
    comparison.savefig(comparison_path, dpi=180)
    plt.close(comparison)

    table_path = output / "trajectory.csv"
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ordinal", "sample_index", "source_frame_index", "timestamp_seconds", "x", "y", "z", "point_count"])
        for ordinal, row in enumerate(states):
            writer.writerow([
                ordinal,
                row["sample_index"],
                row["source_frame_index"],
                row["timestamp_seconds"],
                *row["center_xyz_median"],
                row["point_count"],
            ])

    manifest = {
        "status": "complete",
        "track_candidate_id": states_payload["track_candidate_id"],
        "entity_id": entity_id,
        "motion_state": motion_state,
        "coordinate_system": "raw_pi3_model_units",
        "valid_state_count": len(states),
        "source_point_count": len(points),
        "selected_dynamic_point_count": int(len(np.unique(all_dynamic_indices))),
        "parameters": {
            "max_background_points": args.max_background_points,
            "max_dynamic_points": args.max_dynamic_points,
        },
        "provenance": {
            "observations_sha256": sha256(observations_path),
            "selected_indices_sha256": sha256(selected_path),
            "track_states_sha256": sha256(states_path),
            "classification_sha256": sha256(classification_path),
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
