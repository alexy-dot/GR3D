#!/usr/bin/env python3
"""Build a provenance-checked OSI-Bench Condition-B configuration."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

try:
    from reconstruction.input_identity import file_sha256, input_identity
except ModuleNotFoundError:
    from input_identity import file_sha256, input_identity


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")


def build_condition_b_manifest(
    subset_dir: Path,
    scene_id: str,
    frame_manifest_path: Path,
    pi3_run_dir: Path,
    canonical_views_dir: Path,
) -> dict:
    subset_dir = subset_dir.resolve()
    frame_manifest_path = frame_manifest_path.resolve()
    pi3_run_dir = pi3_run_dir.resolve()
    canonical_views_dir = canonical_views_dir.resolve()
    scene_id = str(scene_id).zfill(4)

    subset_config_path = subset_dir / "osibench_subset_config.json"
    subset_data_path = subset_dir / "data.parquet"
    video_path = subset_dir / f"{scene_id}.mp4"
    for path in (subset_config_path, subset_data_path, video_path, frame_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    frame_manifest = read_json(frame_manifest_path)
    pi3_manifest_path = pi3_run_dir / "manifest.json"
    render_manifest_path = canonical_views_dir / "render_manifest.json"
    pi3_manifest = read_json(pi3_manifest_path)
    render_manifest = read_json(render_manifest_path)

    require_equal(pi3_manifest.get("status"), "complete", "Pi3 run status")
    require_equal(render_manifest.get("status"), "complete", "render status")
    require_equal(render_manifest.get("alignment"), "camera-gravity", "render alignment")
    require_equal(
        render_manifest.get("coordinate_units"),
        "pi3_model_units",
        "render coordinate units",
    )
    require_equal(
        render_manifest.get("axis_values_are_meters"),
        False,
        "render metric-axis declaration",
    )
    require_equal(
        render_manifest.get("metric_scale_validated"),
        False,
        "render metric-scale declaration",
    )
    require_equal(
        render_manifest.get("object_id_correspondence"),
        False,
        "render object-ID correspondence declaration",
    )
    require_equal(
        render_manifest.get("time_conditioned"),
        False,
        "render time-conditioning declaration",
    )
    require_equal(
        frame_manifest.get("source_video_identity"),
        input_identity(video_path),
        "source video identity",
    )
    require_equal(
        pi3_manifest.get("frame_selection", {}).get("manifest_sha256"),
        file_sha256(frame_manifest_path),
        "Pi3 frame manifest hash",
    )
    require_equal(
        int(pi3_manifest.get("frame_count", -1)),
        len(frame_manifest.get("frames", [])),
        "Pi3 frame count",
    )
    require_equal(
        pi3_manifest.get("input_identity"),
        input_identity(frame_manifest_path.parent),
        "Pi3 input identity",
    )

    point_cloud_path = pi3_run_dir / "point_cloud.ply"
    camera_poses_path = pi3_run_dir / "camera_poses.npy"
    require_equal(
        render_manifest.get("point_cloud_sha256"),
        file_sha256(point_cloud_path),
        "render point cloud hash",
    )
    require_equal(
        Path(render_manifest.get("camera_poses", "")).resolve(),
        camera_poses_path.resolve(),
        "render camera poses path",
    )
    require_equal(
        render_manifest.get("camera_poses_sha256"),
        file_sha256(camera_poses_path),
        "render camera poses hash",
    )

    expected_outputs = ["view_xy.png", "view_xz.png", "view_yz.png"]
    require_equal(render_manifest.get("outputs"), expected_outputs, "canonical view set")
    views = []
    for name in expected_outputs:
        path = canonical_views_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        views.append({"axis": name[5:7], "path": str(path), "sha256": file_sha256(path)})

    return {
        "schema_version": 2,
        "condition": "B",
        "condition_name": "raw_video_plus_unfiltered_pi3_canonical_views_v2",
        "protocol_version": "condition-b-v2-nonmetric-explicit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": git_revision(PROJECT_ROOT),
        "answer_blind": True,
        "source_frames_modified": False,
        "original_video_preserved": True,
        "supplemental_evidence": "unfiltered Pi3 camera-gravity-aligned XY/XZ/YZ views",
        "capabilities": {
            "metric_scale": False,
            "object_id_correspondence": False,
            "time_conditioned_evidence": False,
        },
        "subset": {
            "directory": str(subset_dir),
            "data_path": str(subset_data_path),
            "data_sha256": file_sha256(subset_data_path),
            "source_config": str(subset_config_path),
            "source_config_sha256": file_sha256(subset_config_path),
        },
        "scenes": {
            scene_id: {
                "video": str(video_path),
                "video_identity": input_identity(video_path),
                "frame_manifest": str(frame_manifest_path),
                "frame_manifest_sha256": file_sha256(frame_manifest_path),
                "source_indices": frame_manifest["sampling"]["source_indices"],
                "pi3_manifest": str(pi3_manifest_path),
                "pi3_manifest_sha256": file_sha256(pi3_manifest_path),
                "point_cloud_sha256": file_sha256(point_cloud_path),
                "render_manifest": str(render_manifest_path),
                "render_manifest_sha256": file_sha256(render_manifest_path),
                "views": views,
            }
        },
    }


def build_config(source_config: dict, condition_manifest_path: Path) -> dict:
    if len(source_config.get("model", {})) != 1 or len(source_config.get("data", {})) != 1:
        raise ValueError("source config must contain exactly one model and one dataset")
    model_config = source_config["model"]
    source_dataset_name, source_dataset = next(iter(source_config["data"].items()))
    if int(source_dataset.get("nframe", -1)) != 32:
        raise ValueError("Condition B currently requires the frozen 32-frame baseline")
    condition_dataset = {
        **source_dataset,
        "class": "OSIConditionB",
        "dataset": f"{source_dataset_name}-B-v2",
        "condition_manifest": str(condition_manifest_path.resolve()),
    }
    return {
        "model": model_config,
        "data": {f"{source_dataset_name}-B-v2": condition_dataset},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--frame-manifest", required=True, type=Path)
    parser.add_argument("--pi3-run", required=True, type=Path)
    parser.add_argument("--canonical-views", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    condition_manifest = build_condition_b_manifest(
        args.subset_dir.expanduser(),
        args.scene,
        args.frame_manifest.expanduser(),
        args.pi3_run.expanduser(),
        args.canonical_views.expanduser(),
    )
    manifest_path = output_dir / "condition_b_manifest.json"
    manifest_path.write_text(
        json.dumps(condition_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    source_config = read_json(
        args.subset_dir.expanduser().resolve() / "osibench_subset_config.json"
    )
    config = build_config(source_config, manifest_path)
    config_path = output_dir / "osibench_condition_b_config.json"
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": str(manifest_path), "config": str(config_path)}, indent=2))


if __name__ == "__main__":
    main()
