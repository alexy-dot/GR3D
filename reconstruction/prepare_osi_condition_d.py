#!/usr/bin/env python3
"""Prepare provenance-checked OSI Condition-D ID/time trajectory evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    from reconstruction.input_identity import file_sha256, input_identity
    from reconstruction.osi_question_evidence import (
        build_question_evidence,
        render_question_evidence,
    )
    from reconstruction.tracking_contracts import (
        OSI_ID_NAMESPACE,
        external_identity_key,
        load_metric_scale_calibration,
        normalize_external_identity,
        validate_coordinate_system,
    )
except ModuleNotFoundError:
    from input_identity import file_sha256, input_identity
    from osi_question_evidence import build_question_evidence, render_question_evidence
    from tracking_contracts import (
        OSI_ID_NAMESPACE,
        external_identity_key,
        load_metric_scale_calibration,
        normalize_external_identity,
        validate_coordinate_system,
    )


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


def ensure_fresh_output(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)


def load_trajectory(render_directory: Path, scene_id: str) -> tuple[int, dict, dict]:
    directory = render_directory.expanduser().resolve()
    manifest_path = directory / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError(f"trajectory render is incomplete: {directory}")
    coordinate = validate_coordinate_system(manifest.get("coordinate_system"))
    if coordinate.get("alignment") != "camera-gravity":
        raise ValueError("Condition D requires camera-gravity trajectory alignment")
    if coordinate["units"] != "pi3_model_units":
        raise ValueError("Condition D currently requires original Pi3 model units")
    if manifest.get("axis_values_are_meters") is not False:
        raise ValueError("trajectory render must declare non-metric axes")
    if manifest.get("time_conditioned") is not True:
        raise ValueError("trajectory render must retain time evidence")
    identity = normalize_external_identity(manifest.get("external_identity"), required=True)
    assert identity is not None
    if identity["namespace"] != OSI_ID_NAMESPACE:
        raise ValueError("Condition D requires OSI numeric object identities")
    if identity["scene_id"] != scene_id or identity["verified"] is not True:
        raise ValueError("trajectory external identity is unverified or belongs to another scene")
    if manifest.get("object_id_correspondence") is not True:
        raise ValueError("trajectory render does not declare object-ID correspondence")
    provenance = manifest.get("provenance", {})
    audit_value = provenance.get("track_audit")
    contact_value = provenance.get("identity_contact_sheet")
    if not audit_value or not contact_value:
        raise ValueError("verified OSI identity requires a saved track audit")
    audit_path = Path(audit_value)
    contact_path = Path(contact_value)
    if file_sha256(audit_path) != provenance.get("track_audit_sha256"):
        raise ValueError("track audit hash mismatch")
    if file_sha256(contact_path) != provenance.get("identity_contact_sheet_sha256"):
        raise ValueError("identity contact-sheet hash mismatch")
    audit = read_json(audit_path)
    if audit.get("status") != "complete":
        raise ValueError("track audit is incomplete")
    if audit.get("track_candidate_id") != manifest.get("track_candidate_id"):
        raise ValueError("track audit belongs to another candidate")
    if normalize_external_identity(audit.get("external_identity")) != identity:
        raise ValueError("track audit external identity mismatch")
    if audit.get("source_frames_modified") is not False:
        raise ValueError("track audit must preserve source frames")
    audit_contact = audit_path.parent / audit["contact_sheet"]["path"]
    if audit_contact.resolve() != contact_path.resolve():
        raise ValueError("trajectory and track audit reference different contact sheets")
    if file_sha256(audit_contact) != audit["contact_sheet"]["sha256"]:
        raise ValueError("track audit contact-sheet hash mismatch")

    table = directory / manifest["outputs"]["table"]["path"]
    render = directory / manifest["outputs"]["render"]["path"]
    if file_sha256(table) != manifest["outputs"]["table"]["sha256"]:
        raise ValueError("trajectory table hash mismatch")
    if file_sha256(render) != manifest["outputs"]["render"]["sha256"]:
        raise ValueError("trajectory image hash mismatch")
    with table.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("trajectory table is empty")
    required = {
        "timestamp_seconds", "valid_3d", "x", "y", "z",
        "camera_x", "camera_y", "camera_z",
    }
    if not required.issubset(rows[0]):
        raise ValueError("trajectory table lacks position or camera columns")
    timestamps = [float(row["timestamp_seconds"]) for row in rows]
    if not all(math.isfinite(value) and value >= 0 for value in timestamps):
        raise ValueError("trajectory timestamps must be finite and non-negative")
    if any(second <= first for first, second in zip(timestamps, timestamps[1:])):
        raise ValueError("trajectory timestamps must be strictly increasing")
    valid_rows = [
        row for row in rows
        if str(row["valid_3d"]).strip().lower() in {"1", "true", "yes"}
    ]
    if not valid_rows:
        raise ValueError("trajectory table contains no valid 3D states")
    for row in rows:
        camera = [float(row[f"camera_{axis}"]) for axis in "xyz"]
        if not all(math.isfinite(value) for value in camera):
            raise ValueError("trajectory camera coordinates must be finite")
        if row in valid_rows:
            position = [float(row[axis]) for axis in "xyz"]
            if not all(math.isfinite(value) for value in position):
                raise ValueError("valid trajectory positions must be finite")
    if int(manifest.get("valid_state_count", -1)) != len(valid_rows):
        raise ValueError("trajectory valid-state count mismatch")
    if int(manifest.get("invalid_state_count", -1)) != len(rows) - len(valid_rows):
        raise ValueError("trajectory invalid-state count mismatch")
    object_id = int(identity["object_id_numeric"])
    return object_id, {
        "external_identity": identity,
        "track_candidate_id": manifest["track_candidate_id"],
        "motion_state": manifest["motion_state"],
        "rows": rows,
    }, {
        "directory": str(directory),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "render": str(render),
        "render_sha256": file_sha256(render),
        "table": str(table),
        "table_sha256": file_sha256(table),
        "external_identity": identity,
        "source_pi3_manifest_sha256": manifest.get("provenance", {}).get(
            "source_pi3_manifest_sha256"
        ),
        "camera_poses_sha256": manifest.get("provenance", {}).get(
            "camera_poses_sha256"
        ),
        "track_audit": str(audit_path),
        "track_audit_sha256": file_sha256(audit_path),
        "identity_contact_sheet": str(contact_path),
        "identity_contact_sheet_sha256": file_sha256(contact_path),
    }


def build_config(source_config: dict, condition_manifest_path: Path) -> dict:
    if len(source_config.get("model", {})) != 1 or len(source_config.get("data", {})) != 1:
        raise ValueError("source config must contain exactly one model and one dataset")
    source_name, source_dataset = next(iter(source_config["data"].items()))
    if int(source_dataset.get("nframe", -1)) != 32:
        raise ValueError("Condition D requires the frozen official 32-frame baseline")
    name = f"{source_name}-D-v1"
    return {
        "model": source_config["model"],
        "data": {
            name: {
                **source_dataset,
                "class": "OSIConditionD",
                "dataset": name,
                "condition_manifest": str(condition_manifest_path.resolve()),
            }
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--static-views", required=True, type=Path)
    parser.add_argument("--trajectory-render", required=True, type=Path, action="append")
    parser.add_argument("--scale-calibration", type=Path)
    parser.add_argument("--max-interpolation-gap", type=float, default=2.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    scene_id = str(args.scene).zfill(4)
    subset = args.subset_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    ensure_fresh_output(output)
    data_path = subset / "data.parquet"
    config_path = subset / "osibench_subset_config.json"
    video_path = subset / f"{scene_id}.mp4"
    for path in (data_path, config_path, video_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    static_dir = args.static_views.expanduser().resolve()
    static_manifest_path = static_dir / "render_manifest.json"
    static_manifest = read_json(static_manifest_path)
    if static_manifest.get("status") != "complete":
        raise ValueError("static-map render is incomplete")
    if static_manifest.get("alignment") != "camera-gravity":
        raise ValueError("Condition D requires camera-gravity static views")
    if static_manifest.get("coordinate_units") != "pi3_model_units":
        raise ValueError("Condition D static views must use Pi3 model units")
    if static_manifest.get("axis_values_are_meters") is not False:
        raise ValueError("Condition D static views must declare non-metric axes")
    static_views = []
    for name in ("view_xy.png", "view_xz.png", "view_yz.png"):
        path = static_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        static_views.append({"axis": name[5:7], "path": str(path), "sha256": file_sha256(path)})

    trajectories: dict[int, dict] = {}
    trajectory_records = []
    identity_keys = []
    for directory in args.trajectory_render:
        object_id, trajectory, record = load_trajectory(directory, scene_id)
        if object_id in trajectories:
            raise ValueError(f"duplicate OSI object trajectory: {object_id}")
        trajectories[object_id] = trajectory
        trajectory_records.append(record)
        identity_keys.append(external_identity_key(trajectory["external_identity"]))
    if len(identity_keys) != len(set(identity_keys)):
        raise ValueError("duplicate external identities")

    pi3_hashes = {
        record["source_pi3_manifest_sha256"] for record in trajectory_records
    }
    if None in pi3_hashes or len(pi3_hashes) != 1:
        raise ValueError("all trajectories must identify the same Pi3 run")
    pi3_hash = next(iter(pi3_hashes))
    camera_hashes = {record["camera_poses_sha256"] for record in trajectory_records}
    if None in camera_hashes or len(camera_hashes) != 1:
        raise ValueError("all trajectories must use the same camera poses")
    camera_hash = next(iter(camera_hashes))
    if static_manifest.get("camera_poses_sha256") != camera_hash:
        raise ValueError("static views and trajectories use different camera poses")
    static_pose_path = Path(str(static_manifest.get("camera_poses", ""))).expanduser()
    if not static_pose_path.is_absolute():
        static_pose_path = static_manifest_path.parent / static_pose_path
    static_pose_path = static_pose_path.resolve()
    if not static_pose_path.is_file():
        raise FileNotFoundError(static_pose_path)
    if file_sha256(static_pose_path) != camera_hash:
        raise ValueError("static-view camera pose file hash mismatch")
    coordinate_system = validate_coordinate_system(
        read_json(Path(trajectory_records[0]["manifest"]))["coordinate_system"]
    )
    static_map_path = Path(str(static_manifest.get("point_cloud", ""))).expanduser()
    if not static_map_path.is_absolute():
        static_map_path = static_manifest_path.parent / static_map_path
    static_map_path = static_map_path.resolve()
    static_map_manifest_path = static_map_path.parent / "manifest.json"
    if not static_map_manifest_path.is_file():
        raise FileNotFoundError(static_map_manifest_path)
    static_map_manifest = read_json(static_map_manifest_path)
    if static_map_manifest.get("status") != "complete":
        raise ValueError("static/dynamic map manifest is incomplete")
    if static_map_manifest.get("source_pi3_manifest_sha256") != pi3_hash:
        raise ValueError("static map and trajectories belong to different Pi3 runs")
    if static_map_manifest.get("raw_source_modified") is not False:
        raise ValueError("static map must preserve raw Pi3 observations")
    static_output = static_map_manifest.get("outputs", {}).get("static_map", {})
    declared_static_path = static_map_manifest_path.parent / str(
        static_output.get("path", "")
    )
    if declared_static_path.resolve() != static_map_path:
        raise ValueError("static render does not use the static-map manifest output")
    static_map_hash = file_sha256(static_map_path)
    if static_output.get("sha256") != static_map_hash:
        raise ValueError("static-map output hash mismatch")
    if int(static_output.get("point_count", -1)) != int(
        static_map_manifest.get("static_count", -2)
    ):
        raise ValueError("static-map output point count mismatch")
    static_map_coordinate = validate_coordinate_system(
        static_map_manifest.get("coordinate_system")
    )
    if static_map_coordinate["units"] != coordinate_system["units"]:
        raise ValueError("static map and trajectories use different coordinate units")
    map_tracks = {
        row["track_candidate_id"]: row for row in static_map_manifest.get("tracks", [])
    }
    for object_id, trajectory in trajectories.items():
        candidate_id = trajectory["track_candidate_id"]
        map_track = map_tracks.get(candidate_id)
        if map_track is None:
            raise ValueError(
                f"static map omits trajectory candidate {candidate_id} / OSI-{object_id}"
            )
        if normalize_external_identity(map_track.get("external_identity")) != trajectory[
            "external_identity"
        ]:
            raise ValueError("static map and trajectory external identity mismatch")
        if map_track.get("motion_state") != trajectory["motion_state"]:
            raise ValueError("static map and trajectory motion-state mismatch")
    if not any(row["motion_state"] == "dynamic" for row in trajectories.values()):
        raise ValueError("Condition D requires at least one confirmed dynamic trajectory")
    if static_map_hash != static_manifest.get("point_cloud_sha256"):
        raise ValueError("static render point-cloud hash mismatch")
    scale = None
    if args.scale_calibration:
        scale = load_metric_scale_calibration(
            args.scale_calibration,
            coordinate_system=coordinate_system,
            expected_pi3_manifest_sha256=pi3_hash,
        )
    meters_per_unit = scale["meters_per_model_unit"] if scale else None

    table = pd.read_parquet(data_path)
    scene_ids = sorted(set(table["video"].astype(str).str.zfill(4)))
    if scene_ids != [scene_id]:
        raise ValueError(
            "Condition D currently requires a single-scene subset containing only "
            f"{scene_id}; found {scene_ids}"
        )
    question_ids = table["index"].astype(int).tolist()
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("subset contains duplicate question indices")
    scene_rows = table[table["video"].astype(str).str.zfill(4) == scene_id]
    if scene_rows.empty:
        raise ValueError(f"subset contains no questions for scene {scene_id}")
    evidence_rows = []
    evidence_dir = output / "question_evidence"
    for _, row in scene_rows.iterrows():
        question_id = int(row["index"])
        evidence = build_question_evidence(
            question_id=question_id,
            question=str(row["question"]),
            category=str(row["category"]),
            trajectories=trajectories,
            max_gap_seconds=args.max_interpolation_gap,
            meters_per_model_unit=meters_per_unit,
        )
        if evidence["status"] == "ready":
            image_path = evidence_dir / f"q{question_id:06d}.png"
            render_question_evidence(evidence, trajectories, image_path)
            evidence["render"] = {
                "path": str(image_path),
                "sha256": file_sha256(image_path),
            }
        evidence_rows.append(evidence)

    evidence_path = output / "question_evidence_manifest.json"
    evidence_payload = {
        "schema_version": 1,
        "status": "complete",
        "scene_id": scene_id,
        "answer_blind": True,
        "max_interpolation_gap_seconds": args.max_interpolation_gap,
        "coordinate_system": coordinate_system,
        "metric_scale": scale,
        "questions": evidence_rows,
    }
    evidence_path.write_text(
        json.dumps(evidence_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    condition = {
        "schema_version": 1,
        "condition": "D",
        "protocol_version": "condition-d-v1-verified-id-time-contract",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": git_revision(PROJECT_ROOT),
        "answer_blind": True,
        "source_frames_modified": False,
        "original_video_preserved": True,
        "capabilities": {
            "metric_scale": scale is not None,
            "object_id_correspondence": True,
            "time_conditioned_evidence": True,
        },
        "subset": {
            "directory": str(subset),
            "data_path": str(data_path),
            "data_sha256": file_sha256(data_path),
            "source_config": str(config_path),
            "source_config_sha256": file_sha256(config_path),
        },
        "scenes": {
            scene_id: {
                "video": str(video_path),
                "video_identity": input_identity(video_path),
                "static_render_manifest": str(static_manifest_path),
                "static_render_manifest_sha256": file_sha256(static_manifest_path),
                "static_map_manifest": str(static_map_manifest_path),
                "static_map_manifest_sha256": file_sha256(static_map_manifest_path),
                "static_views": static_views,
                "trajectories": trajectory_records,
                "question_evidence": str(evidence_path),
                "question_evidence_sha256": file_sha256(evidence_path),
            }
        },
    }
    condition_path = output / "condition_d_manifest.json"
    condition_path.write_text(
        json.dumps(condition, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    config = build_config(read_json(config_path), condition_path)
    output_config = output / "osibench_condition_d_config.json"
    output_config.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": str(condition_path), "config": str(output_config)}, indent=2))


if __name__ == "__main__":
    main()
