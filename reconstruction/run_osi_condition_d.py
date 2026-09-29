#!/usr/bin/env python3
"""Run official OSI-Bench with verified OSI-ID/time-linked 3D evidence."""

from __future__ import annotations

import argparse
import csv
import copy
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

try:
    from reconstruction.input_identity import file_sha256, input_identity
    from reconstruction.osi_question_evidence import question_sha256
    from reconstruction.tracking_contracts import (
        OSI_ID_NAMESPACE,
        load_metric_scale_calibration,
        normalize_external_identity,
        validate_coordinate_system,
    )
except ModuleNotFoundError:
    from input_identity import file_sha256, input_identity
    from osi_question_evidence import question_sha256
    from tracking_contracts import (
        OSI_ID_NAMESPACE,
        load_metric_scale_calibration,
        normalize_external_identity,
        validate_coordinate_system,
    )


STATIC_EVIDENCE_TEXT = (
    "The next three images are camera-gravity-aligned orthographic views of the "
    "dynamic-filtered static Pi3 map from the same original video. Their axes use Pi3 "
    "model units, not meters. The source video remains the primary visual evidence."
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def recorded_path(value: object, *, anchor: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = anchor / path
    return path.resolve()


def is_valid_3d(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def validate_trajectory_table(path: Path, manifest: dict) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "sample_index", "source_frame_index", "timestamp_seconds", "valid_3d",
        "x", "y", "z", "camera_x", "camera_y", "camera_z",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("trajectory table lacks required time/position columns")
    timestamps = [float(row["timestamp_seconds"]) for row in rows]
    if not all(math.isfinite(value) and value >= 0 for value in timestamps):
        raise ValueError("trajectory timestamps must be finite and non-negative")
    if any(second <= first for first, second in zip(timestamps, timestamps[1:])):
        raise ValueError("trajectory timestamps must be strictly increasing")
    sample_indices = [int(row["sample_index"]) for row in rows]
    source_indices = [int(row["source_frame_index"]) for row in rows]
    if len(sample_indices) != len(set(sample_indices)):
        raise ValueError("trajectory contains duplicate sample indices")
    if any(second <= first for first, second in zip(sample_indices, sample_indices[1:])):
        raise ValueError("trajectory sample indices must be strictly increasing")
    if any(second <= first for first, second in zip(source_indices, source_indices[1:])):
        raise ValueError("trajectory source frame indices must be strictly increasing")
    valid_count = 0
    for row in rows:
        camera = [float(row[f"camera_{axis}"]) for axis in "xyz"]
        if not all(math.isfinite(value) for value in camera):
            raise ValueError("trajectory camera coordinates must be finite")
        if is_valid_3d(row["valid_3d"]):
            valid_count += 1
            position = [float(row[axis]) for axis in "xyz"]
            if not all(math.isfinite(value) for value in position):
                raise ValueError("valid trajectory positions must be finite")
    if valid_count != int(manifest.get("valid_state_count", -1)):
        raise ValueError("trajectory valid-state count mismatch")
    if len(rows) - valid_count != int(manifest.get("invalid_state_count", -1)):
        raise ValueError("trajectory invalid-state count mismatch")
    return rows


def is_answer_key(key: object) -> bool:
    lowered = str(key).lower()
    return (
        lowered in {"answer", "ground_truth", "reference_answer"}
        or lowered.endswith("_answer")
        or "ground_truth" in lowered
    )


def is_metric_value_key(key: object) -> bool:
    lowered = str(key).lower()
    return (
        lowered.endswith(("_meter", "_meters"))
        or "meters_per_" in lowered
        or "meter_per_" in lowered
    )


def forbidden_evidence_keys(value: object) -> list[str]:
    forbidden = []
    if isinstance(value, dict):
        for key, child in value.items():
            if is_answer_key(key):
                forbidden.append(str(key))
            if child is not None and is_metric_value_key(key):
                forbidden.append(str(key))
            forbidden.extend(forbidden_evidence_keys(child))
    elif isinstance(value, list):
        for child in value:
            forbidden.extend(forbidden_evidence_keys(child))
    return forbidden


def validate_condition_manifest(path: Path) -> dict:
    manifest = read_json(path)
    if manifest.get("schema_version") != 1 or manifest.get("condition") != "D":
        raise ValueError("expected a schema-v1 Condition-D manifest")
    if manifest.get("protocol_version") != "condition-d-v1-verified-id-time-contract":
        raise ValueError("unexpected Condition-D protocol version")
    if manifest.get("answer_blind") is not True:
        raise ValueError("Condition D must be answer-blind")
    if manifest.get("source_frames_modified") is not False:
        raise ValueError("Condition D must not modify source frames")
    if manifest.get("original_video_preserved") is not True:
        raise ValueError("Condition D must preserve the original video")
    capabilities = manifest.get("capabilities", {})
    if capabilities.get("object_id_correspondence") is not True:
        raise ValueError("Condition D requires verified object-ID correspondence")
    if capabilities.get("time_conditioned_evidence") is not True:
        raise ValueError("Condition D requires time-conditioned evidence")
    if len(manifest.get("scenes", {})) != 1:
        raise ValueError("Condition D v1 requires exactly one scene")

    data_path = Path(manifest["subset"]["data_path"])
    if file_sha256(data_path) != manifest["subset"]["data_sha256"]:
        raise ValueError("subset data hash mismatch")
    source_config = Path(manifest["subset"]["source_config"])
    if file_sha256(source_config) != manifest["subset"]["source_config_sha256"]:
        raise ValueError("subset source-config hash mismatch")
    for scene_id, scene in manifest.get("scenes", {}).items():
        video = Path(scene["video"])
        if input_identity(video) != scene["video_identity"]:
            raise ValueError(f"scene {scene_id} video identity mismatch")
        axes = []
        static_map_manifest = Path(scene["static_map_manifest"])
        if file_sha256(static_map_manifest) != scene["static_map_manifest_sha256"]:
            raise ValueError(f"scene {scene_id} static-map manifest hash mismatch")
        static_map = read_json(static_map_manifest)
        if static_map.get("status") != "complete" or static_map.get("raw_source_modified") is not False:
            raise ValueError(f"scene {scene_id} static map is invalid")
        static_output = static_map.get("outputs", {}).get("static_map", {})
        static_map_path = recorded_path(
            static_output.get("path", ""), anchor=static_map_manifest.parent
        )
        if file_sha256(static_map_path) != static_output.get("sha256"):
            raise ValueError(f"scene {scene_id} static-map output hash mismatch")
        if int(static_output.get("point_count", -1)) != int(
            static_map.get("static_count", -2)
        ):
            raise ValueError(f"scene {scene_id} static-map point count mismatch")
        static_render_path = Path(scene["static_render_manifest"])
        if file_sha256(static_render_path) != scene["static_render_manifest_sha256"]:
            raise ValueError(f"scene {scene_id} static-render manifest hash mismatch")
        static_render = read_json(static_render_path)
        if (
            static_render.get("status") != "complete"
            or static_render.get("alignment") != "camera-gravity"
            or static_render.get("coordinate_units") != "pi3_model_units"
            or static_render.get("axis_values_are_meters") is not False
        ):
            raise ValueError(f"scene {scene_id} static render contract mismatch")
        rendered_cloud = recorded_path(
            static_render.get("point_cloud", ""), anchor=static_render_path.parent
        )
        if rendered_cloud != static_map_path:
            raise ValueError(f"scene {scene_id} static render uses another point cloud")
        if file_sha256(rendered_cloud) != static_render.get("point_cloud_sha256"):
            raise ValueError(f"scene {scene_id} static-render cloud hash mismatch")
        static_pose_path = recorded_path(
            static_render.get("camera_poses", ""), anchor=static_render_path.parent
        )
        if file_sha256(static_pose_path) != static_render.get("camera_poses_sha256"):
            raise ValueError(f"scene {scene_id} static-render camera hash mismatch")
        for view in scene.get("static_views", []):
            view_path = Path(view["path"])
            if file_sha256(view_path) != view["sha256"]:
                raise ValueError(f"scene {scene_id} static-view hash mismatch")
            axes.append(view["axis"])
        if axes != ["xy", "xz", "yz"]:
            raise ValueError("Condition D requires ordered XY/XZ/YZ static views")
        pi3_hashes = set()
        camera_hashes = set()
        verified_tracks = {}
        verified_identity_keys = set()
        for trajectory in scene.get("trajectories", []):
            for path_key, hash_key in (
                ("manifest", "manifest_sha256"),
                ("render", "render_sha256"),
                ("table", "table_sha256"),
                ("track_audit", "track_audit_sha256"),
                ("identity_contact_sheet", "identity_contact_sheet_sha256"),
            ):
                if file_sha256(Path(trajectory[path_key])) != trajectory[hash_key]:
                    raise ValueError(f"scene {scene_id} trajectory hash mismatch")
            trajectory_manifest = read_json(Path(trajectory["manifest"]))
            coordinate = validate_coordinate_system(
                trajectory_manifest.get("coordinate_system")
            )
            identity = normalize_external_identity(
                trajectory_manifest.get("external_identity"), required=True
            )
            assert identity is not None
            if (
                trajectory_manifest.get("status") != "complete"
                or coordinate.get("alignment") != "camera-gravity"
                or coordinate.get("units") != "pi3_model_units"
                or trajectory_manifest.get("axis_values_are_meters") is not False
                or trajectory_manifest.get("object_id_correspondence") is not True
                or trajectory_manifest.get("time_conditioned") is not True
                or identity["namespace"] != OSI_ID_NAMESPACE
                or identity["scene_id"] != scene_id
                or identity["verified"] is not True
            ):
                raise ValueError(f"scene {scene_id} trajectory contract mismatch")
            candidate_id = trajectory_manifest["track_candidate_id"]
            if candidate_id in verified_tracks:
                raise ValueError(f"scene {scene_id} duplicate trajectory candidate")
            identity_key = (
                identity["namespace"], identity["scene_id"], identity["object_id_numeric"]
            )
            if identity_key in verified_identity_keys:
                raise ValueError(f"scene {scene_id} duplicate external identity")
            verified_identity_keys.add(identity_key)
            if normalize_external_identity(trajectory.get("external_identity")) != identity:
                raise ValueError(f"scene {scene_id} trajectory record identity mismatch")
            audit_path = Path(trajectory["track_audit"])
            audit = read_json(audit_path)
            if (
                audit.get("status") != "complete"
                or audit.get("track_candidate_id") != candidate_id
                or normalize_external_identity(audit.get("external_identity")) != identity
                or audit.get("source_frames_modified") is not False
            ):
                raise ValueError(f"scene {scene_id} track-audit contract mismatch")
            audit_contact = recorded_path(
                audit.get("contact_sheet", {}).get("path", ""), anchor=audit_path.parent
            )
            if audit_contact != Path(trajectory["identity_contact_sheet"]).resolve():
                raise ValueError(f"scene {scene_id} track-audit contact-sheet mismatch")
            if file_sha256(audit_contact) != audit.get("contact_sheet", {}).get("sha256"):
                raise ValueError(f"scene {scene_id} track-audit contact hash mismatch")
            table_path = Path(trajectory["table"])
            validate_trajectory_table(table_path, trajectory_manifest)
            verified_tracks[candidate_id] = {
                "track_candidate_id": candidate_id,
                "external_identity": identity,
                "motion_state": trajectory_manifest["motion_state"],
                "object_id_numeric": identity["object_id_numeric"],
            }
            provenance = trajectory_manifest.get("provenance", {})
            pi3_hashes.add(provenance.get("source_pi3_manifest_sha256"))
            camera_hashes.add(provenance.get("camera_poses_sha256"))
        if None in pi3_hashes or len(pi3_hashes) != 1:
            raise ValueError(f"scene {scene_id} trajectories use different Pi3 runs")
        if None in camera_hashes or len(camera_hashes) != 1:
            raise ValueError(f"scene {scene_id} trajectories use different camera poses")
        pi3_hash = next(iter(pi3_hashes))
        camera_hash = next(iter(camera_hashes))
        if static_map.get("source_pi3_manifest_sha256") != pi3_hash:
            raise ValueError(f"scene {scene_id} static map uses another Pi3 run")
        if static_render.get("camera_poses_sha256") != camera_hash:
            raise ValueError(f"scene {scene_id} static render uses other camera poses")
        if file_sha256(static_pose_path) != camera_hash:
            raise ValueError(f"scene {scene_id} camera-pose file differs from trajectories")
        static_tracks = {
            row["track_candidate_id"]: row for row in static_map.get("tracks", [])
        }
        if len(static_tracks) != len(static_map.get("tracks", [])):
            raise ValueError(f"scene {scene_id} static map has duplicate candidates")
        for candidate_id, trajectory in verified_tracks.items():
            map_track = static_tracks.get(candidate_id)
            if map_track is None:
                raise ValueError(f"scene {scene_id} static map omits {candidate_id}")
            if normalize_external_identity(map_track.get("external_identity")) != trajectory[
                "external_identity"
            ]:
                raise ValueError(f"scene {scene_id} static-map identity mismatch")
            if map_track.get("motion_state") != trajectory["motion_state"]:
                raise ValueError(f"scene {scene_id} static-map motion mismatch")
        if not any(
            row["motion_state"] == "dynamic" for row in verified_tracks.values()
        ):
            raise ValueError(f"scene {scene_id} has no confirmed dynamic trajectory")
        evidence_path = Path(scene["question_evidence"])
        if file_sha256(evidence_path) != scene["question_evidence_sha256"]:
            raise ValueError(f"scene {scene_id} question-evidence hash mismatch")
        evidence = read_json(evidence_path)
        if (
            evidence.get("status") != "complete"
            or evidence.get("answer_blind") is not True
            or evidence.get("scene_id") != scene_id
        ):
            raise ValueError("invalid question-evidence manifest")
        metric_declared = evidence.get("metric_scale") is not None
        if metric_declared != bool(capabilities.get("metric_scale")):
            raise ValueError("Condition-D metric capability differs from question evidence")
        evidence_coordinate = validate_coordinate_system(evidence.get("coordinate_system"))
        if evidence_coordinate.get("alignment") != "camera-gravity":
            raise ValueError("Condition-D evidence coordinate alignment mismatch")
        if metric_declared:
            scale = evidence["metric_scale"]
            calibration = load_metric_scale_calibration(
                Path(scale["calibration_path"]),
                coordinate_system=evidence_coordinate,
                expected_pi3_manifest_sha256=pi3_hash,
            )
            if calibration["calibration_sha256"] != scale["calibration_sha256"]:
                raise ValueError("Condition-D scale calibration hash mismatch")
        evidence_by_id = {}
        verified_by_object_id = {
            row["object_id_numeric"]: row for row in verified_tracks.values()
        }
        for row in evidence.get("questions", []):
            question_id = int(row["question_id"])
            if question_id in evidence_by_id:
                raise ValueError(f"duplicate question evidence: {question_id}")
            render = row.get("render")
            if render and file_sha256(Path(render["path"])) != render["sha256"]:
                raise ValueError(f"question {question_id} render hash mismatch")
            if row.get("answer_blind") is not True:
                raise ValueError(f"question {question_id} evidence is not answer-blind")
            referenced_ids = [int(value) for value in row.get("referenced_object_ids", [])]
            if len(referenced_ids) != len(set(referenced_ids)):
                raise ValueError(f"question {question_id} repeats an object ID")
            query_times = [float(value) for value in row.get("query_times_seconds", [])]
            if not all(math.isfinite(value) and value >= 0 for value in query_times):
                raise ValueError(f"question {question_id} contains an invalid time")
            if row.get("metric_scale_validated") is not metric_declared:
                raise ValueError(f"question {question_id} metric declaration mismatch")
            expected_scale = (
                float(evidence["metric_scale"]["meters_per_model_unit"])
                if metric_declared else None
            )
            if row.get("meters_per_model_unit") != expected_scale:
                raise ValueError(f"question {question_id} metric scale mismatch")
            status = row.get("status")
            if status not in {
                "ready", "not_applicable_no_object_id",
                "unavailable_missing_object_tracks", "unavailable_time_coverage",
            }:
                raise ValueError(f"question {question_id} has unsupported evidence status")
            if status == "ready":
                if render is None:
                    raise ValueError(f"question {question_id} ready evidence lacks a render")
                if row.get("time_conditioned") is not bool(query_times):
                    raise ValueError(f"question {question_id} time declaration mismatch")
                objects = row.get("objects", [])
                if sorted(int(item["object_id"]) for item in objects) != sorted(referenced_ids):
                    raise ValueError(f"question {question_id} object list mismatch")
                for item in objects:
                    object_id = int(item["object_id"])
                    track = verified_by_object_id.get(object_id)
                    if track is None:
                        raise ValueError(f"question {question_id} uses an unverified object ID")
                    if (
                        normalize_external_identity(item.get("external_identity"))
                        != track["external_identity"]
                        or item.get("track_candidate_id") != track["track_candidate_id"]
                        or item.get("motion_state") != track["motion_state"]
                    ):
                        raise ValueError(f"question {question_id} object provenance mismatch")
                    samples = item.get("samples", [])
                    if len(samples) != len(query_times):
                        raise ValueError(f"question {question_id} sample count mismatch")
                    for sample, expected_time in zip(samples, query_times, strict=True):
                        if not math.isclose(
                            float(sample["timestamp_seconds"]), expected_time,
                            rel_tol=0.0, abs_tol=1e-9,
                        ):
                            raise ValueError(f"question {question_id} sample time mismatch")
                        values = [
                            *sample["position_xyz"], *sample["camera_xyz"]
                        ]
                        if len(values) != 6 or not all(
                            math.isfinite(float(value)) for value in values
                        ):
                            raise ValueError(f"question {question_id} has invalid coordinates")
            elif render is not None:
                raise ValueError(f"question {question_id} unavailable evidence has a render")
            forbidden = forbidden_evidence_keys(row)
            answer_keys = {key for key in forbidden if is_answer_key(key)}
            if answer_keys:
                raise ValueError(
                    f"question {question_id} evidence contains answer fields"
                )
            if not metric_declared:
                metric_keys = [key for key in forbidden if key not in answer_keys]
                if metric_keys:
                    raise ValueError(
                        f"question {question_id} contains unvalidated metric values"
                    )
            evidence_by_id[question_id] = row
        scene["_question_evidence_by_id"] = evidence_by_id
    return manifest


def official_argument_value(arguments: list[str], name: str) -> str | None:
    for index, value in enumerate(arguments):
        if value == name:
            if index + 1 >= len(arguments):
                raise ValueError(f"{name} requires a value")
            return arguments[index + 1]
        prefix = name + "="
        if value.startswith(prefix):
            return value[len(prefix):]
    return None


def validate_official_config(
    official_args: list[str], condition: dict, condition_path: Path
) -> Path:
    condition_path = condition_path.expanduser().resolve()
    config_value = official_argument_value(official_args, "--config")
    if config_value is None:
        raise ValueError("Condition D requires the generated --config file")
    config_path = Path(config_value).expanduser().resolve()
    config = read_json(config_path)
    source_config = read_json(Path(condition["subset"]["source_config"]))
    if config.get("model") != source_config.get("model"):
        raise ValueError("official config model differs from the frozen source config")
    if len(config.get("data", {})) != 1:
        raise ValueError("official config must contain exactly one Condition-D dataset")
    dataset = next(iter(config["data"].values()))
    if dataset.get("class") != "OSIConditionD":
        raise ValueError("official config does not select OSIConditionD")
    if int(dataset.get("nframe", -1)) != 32 or float(dataset.get("fps", -1)) != -1:
        raise ValueError("official config must preserve the frozen 32-frame sampler")
    if Path(dataset.get("data_path", "")).expanduser().resolve() != Path(
        condition["subset"]["data_path"]
    ).parent.resolve():
        raise ValueError("official config data path differs from Condition D")
    if Path(dataset.get("condition_manifest", "")).expanduser().resolve() != condition_path:
        raise ValueError("official config condition manifest differs from launcher input")
    return config_path


def _rounded(values: list[float]) -> str:
    return "[" + ", ".join(f"{float(value):.6g}" for value in values) + "]"


def format_question_evidence(evidence: dict) -> str:
    status = evidence["status"]
    ids = evidence.get("referenced_object_ids", [])
    times = evidence.get("query_times_seconds", [])
    if status == "not_applicable_no_object_id":
        return "This question contains no OSI numeric object ID; no ID-linked trajectory is appended."
    if status == "unavailable_missing_object_tracks":
        return (
            "No verified trajectory is available for OSI object ID(s) "
            + ", ".join(str(value) for value in evidence["missing_object_ids"])
            + ". Do not infer an ID mapping from the static views."
        )
    if status == "unavailable_time_coverage":
        return (
            "Verified object identity exists, but its trajectory does not safely cover the "
            "requested time(s). Do not extrapolate beyond the recorded interval."
        )
    if status != "ready":
        raise ValueError(f"unsupported question evidence status: {status}")

    lines = [
        "The following question-specific trajectory evidence is explicitly linked to OSI "
        f"object ID(s) {', '.join(str(value) for value in ids)}."
    ]
    if times:
        lines.append(
            "It is conditioned only on requested time(s): "
            + ", ".join(f"{value:g}s" for value in times)
            + "."
        )
    for item in evidence["objects"]:
        identity = item["external_identity"]
        lines.append(
            f"OSI-{item['object_id']} identity association: "
            f"{identity['association_method']}, verified={identity['verified']}."
        )
        for sample in item["samples"]:
            lines.append(
                f"OSI-{item['object_id']} at {sample['timestamp_seconds']:g}s: "
                f"object XYZ={_rounded(sample['position_xyz'])}; "
                f"camera XYZ={_rounded(sample['camera_xyz'])}; "
                f"sampling={sample['method']}."
            )
    derived = evidence.get("derived_geometry", {})
    for key, value in derived.items():
        if isinstance(value, (int, float)):
            lines.append(f"{key}={float(value):.6g}.")
        elif key == "pairwise_object_distances":
            for row in value:
                lines.append(
                    f"pairwise IDs {row['object_ids']}: "
                    f"distance_model_units={float(row['distance_model_units']):.6g}."
                )
                if "distance_meters" in row:
                    lines.append(
                        f"pairwise IDs {row['object_ids']}: "
                        f"distance_meters={float(row['distance_meters']):.6g}."
                    )
    if evidence.get("metric_scale_validated"):
        lines.append(
            "Meter-valued derived quantities use an independently validated scale whose "
            "provenance is recorded in the condition manifest."
        )
    else:
        lines.append(
            "All coordinates and distances above are Pi3 model units, not meters. Do not "
            "convert them to metric values or treat similar numbers as meter answers."
        )
    return " ".join(lines)


def append_condition_d_evidence(
    messages: list[dict], scene: dict, question_id: int, question: str
) -> list[dict]:
    evidence = scene["_question_evidence_by_id"].get(question_id)
    if evidence is None:
        raise KeyError(f"Condition-D evidence is missing for question {question_id}")
    if evidence["question_sha256"] != question_sha256(question):
        raise ValueError(f"question {question_id} text differs from prepared evidence")
    augmented = copy.deepcopy(messages)
    augmented.append({"type": "text", "value": STATIC_EVIDENCE_TEXT})
    augmented.extend(
        {"type": "image", "value": view["path"]} for view in scene["static_views"]
    )
    augmented.append({"type": "text", "value": format_question_evidence(evidence)})
    if evidence.get("render"):
        augmented.append({"type": "image", "value": evidence["render"]["path"]})
    return augmented


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--vlmeval-root", required=True, type=Path)
    parser.add_argument("--condition-manifest", required=True, type=Path)
    return parser.parse_known_args()


def main() -> None:
    args, official_args = parse_args()
    vlmeval_root = args.vlmeval_root.expanduser().resolve()
    condition_path = args.condition_manifest.expanduser().resolve()
    if not (vlmeval_root / "run.py").is_file():
        raise FileNotFoundError(vlmeval_root / "run.py")
    condition = validate_condition_manifest(condition_path)
    validate_official_config(official_args, condition, condition_path)

    sys.path.insert(0, str(vlmeval_root))
    import vlmeval.dataset as dataset_module
    from vlmeval.dataset.OSIBench.osibench import OSIBench

    class OSIConditionD(OSIBench):
        def __init__(
            self,
            condition_manifest,
            data_path,
            dataset="OSI-Bench-Condition-D",
            nframe=32,
            fps=-1,
            download=False,
            **kwargs,
        ):
            supplied = Path(condition_manifest).expanduser().resolve()
            if supplied != condition_path:
                raise ValueError("config condition manifest differs from launcher manifest")
            if int(nframe) != 32 or float(fps) != -1:
                raise ValueError("Condition D requires the frozen official 32-frame sampler")
            super().__init__(
                data_path=data_path,
                dataset=dataset,
                nframe=nframe,
                fps=fps,
                download=download,
                **kwargs,
            )
            expected_data = Path(condition["subset"]["data_path"]).resolve()
            if Path(self.data_file).resolve() != expected_data:
                raise ValueError("dataset data path differs from Condition-D manifest")

        def build_prompt(self, line, video_llm, **kwargs):
            row = self.data.iloc[line] if isinstance(line, int) else line
            scene_id = str(row["video"]).zfill(4)
            if scene_id not in condition["scenes"]:
                raise KeyError(f"Condition-D evidence is missing for scene {scene_id}")
            messages = super().build_prompt(line, video_llm, **kwargs)
            if not video_llm or not any(item.get("type") == "video" for item in messages):
                raise ValueError("Condition D requires the original video-LLM input path")
            return append_condition_d_evidence(
                messages,
                condition["scenes"][scene_id],
                int(row["index"]),
                str(row["question"]),
            )

    dataset_module.OSIConditionD = OSIConditionD
    spec = importlib.util.spec_from_file_location(
        "official_vlmeval_run", vlmeval_root / "run.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load official VLMEvalKit run.py")
    official_run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official_run)
    os.chdir(vlmeval_root)
    sys.argv = [str(vlmeval_root / "run.py"), *official_args]
    official_run.load_env()
    official_run.main()


if __name__ == "__main__":
    main()
