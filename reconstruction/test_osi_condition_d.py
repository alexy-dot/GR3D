#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from reconstruction.input_identity import file_sha256, input_identity
from reconstruction.osi_question_evidence import question_sha256
from reconstruction.prepare_osi_condition_d import build_config, load_trajectory
from reconstruction.run_osi_condition_d import (
    append_condition_d_evidence,
    forbidden_evidence_keys,
    format_question_evidence,
    validate_condition_manifest,
    validate_official_config,
)


def ready_evidence() -> dict:
    question = "person(id: 26) between 1s and 2s"
    return {
        "question_id": 6,
        "question_sha256": question_sha256(question),
        "status": "ready",
        "referenced_object_ids": [26],
        "query_times_seconds": [1.0, 2.0],
        "metric_scale_validated": False,
        "objects": [{
            "object_id": 26,
            "external_identity": {
                "association_method": "manual_box",
                "verified": True,
            },
            "samples": [
                {
                    "timestamp_seconds": 1.0,
                    "position_xyz": [1.0, 0.0, 0.0],
                    "camera_xyz": [0.0, 0.0, 0.0],
                    "method": "exact_sample",
                },
                {
                    "timestamp_seconds": 2.0,
                    "position_xyz": [2.0, 0.0, 0.0],
                    "camera_xyz": [0.0, 0.0, 0.0],
                    "method": "exact_sample",
                },
            ],
        }],
        "derived_geometry": {"object_displacement_model_units": 1.0},
        "render": {"path": "/tmp/q6.png", "sha256": "unused"},
    }


def build_condition_fixture(root: Path) -> tuple[Path, dict]:
    identity = {
        "namespace": "osi_bench_numeric_id",
        "scene_id": "0000",
        "object_id": "26",
        "object_id_numeric": 26,
        "association_method": "manual_box",
        "verified": True,
    }
    raw_coordinate = {
        "name": "pi3_world",
        "units": "pi3_model_units",
        "axis_values_are_meters": False,
        "metric_scale_validated": False,
        "meters_per_unit": None,
        "scale_provenance": None,
    }
    aligned_coordinate = {
        **raw_coordinate,
        "name": "camera_gravity_aligned_pi3_world",
        "source_coordinate_system_name": "pi3_world",
        "alignment": "camera-gravity",
    }
    data = root / "data.parquet"
    config = root / "source_config.json"
    video = root / "0000.mp4"
    data.write_bytes(b"parquet-fixture")
    config.write_text("{}\n", encoding="utf-8")
    video.write_bytes(b"video-fixture")

    map_dir = root / "static_map"
    map_dir.mkdir()
    static_cloud = map_dir / "static_map.ply"
    static_cloud.write_bytes(b"ply-fixture")
    pi3_hash = "a" * 64
    camera_poses = root / "camera_poses.npy"
    camera_poses.write_bytes(b"camera-fixture")
    camera_hash = file_sha256(camera_poses)
    static_map_manifest = map_dir / "manifest.json"
    static_map_manifest.write_text(json.dumps({
        "status": "complete",
        "coordinate_system": raw_coordinate,
        "source_pi3_manifest_sha256": pi3_hash,
        "source_observations_sha256": "c" * 64,
        "source_count": 2,
        "static_count": 1,
        "dynamic_count": 1,
        "uncertain_count": 0,
        "raw_source_modified": False,
        "tracks": [{
            "track_candidate_id": "T026",
            "external_identity": identity,
            "motion_state": "dynamic",
            "entity_id": "D001",
        }],
        "outputs": {
            "static_map": {
                "path": static_cloud.name,
                "sha256": file_sha256(static_cloud),
                "point_count": 1,
            }
        },
    }), encoding="utf-8")

    static_render_dir = root / "static_views"
    static_render_dir.mkdir()
    static_views = []
    for axis in ("xy", "xz", "yz"):
        view = static_render_dir / f"view_{axis}.png"
        view.write_bytes(f"view-{axis}".encode())
        static_views.append({
            "axis": axis,
            "path": str(view),
            "sha256": file_sha256(view),
        })
    static_render_manifest = static_render_dir / "render_manifest.json"
    static_render_manifest.write_text(json.dumps({
        "status": "complete",
        "alignment": "camera-gravity",
        "coordinate_units": "pi3_model_units",
        "axis_values_are_meters": False,
        "point_cloud": str(static_cloud),
        "point_cloud_sha256": file_sha256(static_cloud),
        "camera_poses": str(camera_poses),
        "camera_poses_sha256": camera_hash,
    }), encoding="utf-8")

    trajectory_dir = root / "trajectory"
    trajectory_dir.mkdir()
    trajectory_render = trajectory_dir / "dynamic_trajectory_xyz.png"
    trajectory_table = trajectory_dir / "trajectory.csv"
    contact = trajectory_dir / "contact.jpg"
    audit = trajectory_dir / "track_audit.json"
    trajectory_render.write_bytes(b"trajectory-render")
    contact.write_bytes(b"contact-sheet")
    trajectory_table.write_text(
        "ordinal,sample_index,source_frame_index,timestamp_seconds,valid_3d,x,y,z,camera_x,camera_y,camera_z,point_count,quality_warnings\n"
        "0,0,0,1.0,True,1,0,0,0,0,0,20,[]\n",
        encoding="utf-8",
    )
    audit.write_text(json.dumps({
        "status": "complete",
        "track_candidate_id": "T026",
        "external_identity": identity,
        "source_frames_modified": False,
        "contact_sheet": {
            "path": contact.name,
            "sha256": file_sha256(contact),
        },
    }), encoding="utf-8")
    trajectory_manifest = trajectory_dir / "manifest.json"
    trajectory_manifest.write_text(json.dumps({
        "status": "complete",
        "track_candidate_id": "T026",
        "external_identity": identity,
        "entity_id": "D001",
        "motion_state": "dynamic",
        "coordinate_system": aligned_coordinate,
        "axis_values_are_meters": False,
        "metric_scale_validated": False,
        "object_id_correspondence": True,
        "time_conditioned": True,
        "valid_state_count": 1,
        "invalid_state_count": 0,
        "provenance": {
            "source_pi3_manifest_sha256": pi3_hash,
            "camera_poses_sha256": camera_hash,
            "track_audit": str(audit),
            "track_audit_sha256": file_sha256(audit),
            "identity_contact_sheet": str(contact),
            "identity_contact_sheet_sha256": file_sha256(contact),
        },
        "outputs": {
            "render": {
                "path": trajectory_render.name,
                "sha256": file_sha256(trajectory_render),
            },
            "table": {
                "path": trajectory_table.name,
                "sha256": file_sha256(trajectory_table),
            },
        },
    }), encoding="utf-8")

    question = "Where is person(id: 26) at 1s?"
    question_render = root / "q000006.png"
    question_render.write_bytes(b"question-render")
    evidence_path = root / "question_evidence_manifest.json"
    evidence_path.write_text(json.dumps({
        "schema_version": 1,
        "status": "complete",
        "scene_id": "0000",
        "answer_blind": True,
        "coordinate_system": aligned_coordinate,
        "metric_scale": None,
        "questions": [{
            "question_id": 6,
            "question_sha256": question_sha256(question),
            "category": "object_3d_localization",
            "status": "ready",
            "answer_blind": True,
            "referenced_object_ids": [26],
            "query_times_seconds": [1.0],
            "coordinate_units": "pi3_model_units",
            "metric_scale_validated": False,
            "meters_per_model_unit": None,
            "time_conditioned": True,
            "objects": [{
                "object_id": 26,
                "external_identity": identity,
                "track_candidate_id": "T026",
                "motion_state": "dynamic",
                "samples": [{
                    "timestamp_seconds": 1.0,
                    "position_xyz": [1.0, 0.0, 0.0],
                    "camera_xyz": [0.0, 0.0, 0.0],
                    "method": "exact_sample",
                }],
            }],
            "derived_geometry": {"object_camera_distance_model_units": 1.0},
            "render": {
                "path": str(question_render),
                "sha256": file_sha256(question_render),
            },
        }],
    }), encoding="utf-8")

    trajectory_record = {
        "manifest": str(trajectory_manifest),
        "manifest_sha256": file_sha256(trajectory_manifest),
        "render": str(trajectory_render),
        "render_sha256": file_sha256(trajectory_render),
        "table": str(trajectory_table),
        "table_sha256": file_sha256(trajectory_table),
        "track_audit": str(audit),
        "track_audit_sha256": file_sha256(audit),
        "identity_contact_sheet": str(contact),
        "identity_contact_sheet_sha256": file_sha256(contact),
        "external_identity": identity,
    }
    condition = {
        "schema_version": 1,
        "condition": "D",
        "protocol_version": "condition-d-v1-verified-id-time-contract",
        "answer_blind": True,
        "source_frames_modified": False,
        "original_video_preserved": True,
        "capabilities": {
            "metric_scale": False,
            "object_id_correspondence": True,
            "time_conditioned_evidence": True,
        },
        "subset": {
            "data_path": str(data),
            "data_sha256": file_sha256(data),
            "source_config": str(config),
            "source_config_sha256": file_sha256(config),
        },
        "scenes": {
            "0000": {
                "video": str(video),
                "video_identity": input_identity(video),
                "static_map_manifest": str(static_map_manifest),
                "static_map_manifest_sha256": file_sha256(static_map_manifest),
                "static_render_manifest": str(static_render_manifest),
                "static_render_manifest_sha256": file_sha256(static_render_manifest),
                "static_views": static_views,
                "trajectories": [trajectory_record],
                "question_evidence": str(evidence_path),
                "question_evidence_sha256": file_sha256(evidence_path),
            }
        },
    }
    condition_path = root / "condition_d_manifest.json"
    condition_path.write_text(json.dumps(condition), encoding="utf-8")
    return condition_path, condition


class OsiConditionDTests(unittest.TestCase):
    def test_nonmetric_evidence_is_explicit(self) -> None:
        text = format_question_evidence(ready_evidence())
        self.assertIn("OSI-26", text)
        self.assertIn("not meters", text)
        self.assertIn("manual_box", text)

    def test_prompt_keeps_video_and_adds_question_specific_render(self) -> None:
        question = "person(id: 26) between 1s and 2s"
        evidence = ready_evidence()
        scene = {
            "static_views": [
                {"path": "/tmp/xy.png"},
                {"path": "/tmp/xz.png"},
                {"path": "/tmp/yz.png"},
            ],
            "_question_evidence_by_id": {6: evidence},
        }
        messages = [{"type": "video", "value": "/tmp/0000.mp4"}]
        augmented = append_condition_d_evidence(messages, scene, 6, question)
        self.assertEqual(messages, [{"type": "video", "value": "/tmp/0000.mp4"}])
        self.assertEqual(sum(row["type"] == "video" for row in augmented), 1)
        self.assertEqual(sum(row["type"] == "image" for row in augmented), 4)
        self.assertEqual(augmented[-1]["value"], "/tmp/q6.png")

    def test_question_text_mismatch_is_rejected(self) -> None:
        scene = {
            "static_views": [],
            "_question_evidence_by_id": {6: ready_evidence()},
        }
        with self.assertRaisesRegex(ValueError, "differs"):
            append_condition_d_evidence([], scene, 6, "different question")

    def test_recursive_evidence_scan_finds_nested_metric_and_answer_fields(self) -> None:
        keys = forbidden_evidence_keys({
            "derived": [{"distance_meters": 2.0}],
            "metadata": {
                "answer": "forbidden",
                "gold_answer": "also forbidden",
            },
        })
        self.assertIn("distance_meters", keys)
        self.assertIn("answer", keys)
        self.assertIn("gold_answer", keys)

    def test_trajectory_loader_requires_verified_id_and_audit_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            render = root / "trajectory.png"
            table = root / "trajectory.csv"
            audit = root / "track_audit.json"
            contact = root / "contact.jpg"
            render.write_bytes(b"render")
            contact.write_bytes(b"contact")
            table.write_text(
                "ordinal,sample_index,source_frame_index,timestamp_seconds,valid_3d,x,y,z,camera_x,camera_y,camera_z,point_count,quality_warnings\n"
                "0,0,0,0.0,True,0,0,0,0,0,0,10,[]\n",
                encoding="utf-8",
            )
            identity = {
                "namespace": "osi_bench_numeric_id",
                "scene_id": "0000",
                "object_id": "26",
                "object_id_numeric": 26,
                "association_method": "manual_box",
                "verified": True,
            }
            audit.write_text(json.dumps({
                "status": "complete",
                "track_candidate_id": "T026",
                "external_identity": identity,
                "source_frames_modified": False,
                "contact_sheet": {
                    "path": contact.name,
                    "sha256": file_sha256(contact),
                },
            }), encoding="utf-8")
            manifest = {
                "status": "complete",
                "track_candidate_id": "T026",
                "external_identity": identity,
                "motion_state": "dynamic",
                "coordinate_system": {
                    "name": "camera_gravity_aligned_pi3_world",
                    "source_coordinate_system_name": "pi3_world",
                    "units": "pi3_model_units",
                    "axis_values_are_meters": False,
                    "metric_scale_validated": False,
                    "meters_per_unit": None,
                    "scale_provenance": None,
                    "alignment": "camera-gravity",
                },
                "axis_values_are_meters": False,
                "time_conditioned": True,
                "object_id_correspondence": True,
                "valid_state_count": 1,
                "invalid_state_count": 0,
                "outputs": {
                    "render": {"path": render.name, "sha256": file_sha256(render)},
                    "table": {"path": table.name, "sha256": file_sha256(table)},
                },
                "provenance": {
                    "source_pi3_manifest_sha256": "a" * 64,
                    "camera_poses_sha256": "b" * 64,
                    "track_audit": str(audit),
                    "track_audit_sha256": file_sha256(audit),
                    "identity_contact_sheet": str(contact),
                    "identity_contact_sheet_sha256": file_sha256(contact),
                },
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            object_id, trajectory, record = load_trajectory(root, "0000")
            self.assertEqual(object_id, 26)
            self.assertEqual(trajectory["track_candidate_id"], "T026")
            self.assertEqual(record["source_pi3_manifest_sha256"], "a" * 64)

    def test_runtime_revalidates_complete_condition_d_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            condition_path, _ = build_condition_fixture(Path(directory))
            result = validate_condition_manifest(condition_path)
            scene = result["scenes"]["0000"]
            self.assertIn(6, scene["_question_evidence_by_id"])

    def test_runtime_rejects_static_map_modified_after_preparation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            condition_path, condition = build_condition_fixture(root)
            static_manifest = Path(
                condition["scenes"]["0000"]["static_map_manifest"]
            )
            static_map = json.loads(static_manifest.read_text(encoding="utf-8"))
            static_path = static_manifest.parent / static_map["outputs"]["static_map"]["path"]
            static_path.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "static-map output hash"):
                validate_condition_manifest(condition_path)

    def test_runtime_rejects_question_object_candidate_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            condition_path, condition = build_condition_fixture(root)
            evidence_path = Path(
                condition["scenes"]["0000"]["question_evidence"]
            )
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence["questions"][0]["objects"][0]["track_candidate_id"] = "T999"
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
            condition["scenes"]["0000"]["question_evidence_sha256"] = file_sha256(
                evidence_path
            )
            condition_path.write_text(json.dumps(condition), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "object provenance"):
                validate_condition_manifest(condition_path)

    def test_config_keeps_official_model_and_32_frames(self) -> None:
        source = {
            "model": {"Qwen2.5-VL-3B-Instruct": {}},
            "data": {
                "OSI-Bench-Subset": {
                    "class": "OSIBench",
                    "data_path": "/data/osi",
                    "dataset": "OSI-Bench-Subset",
                    "nframe": 32,
                    "download": False,
                }
            },
        }
        config = build_config(source, Path("/tmp/condition_d_manifest.json"))
        dataset = config["data"]["OSI-Bench-Subset-D-v1"]
        self.assertEqual(config["model"], source["model"])
        self.assertEqual(dataset["class"], "OSIConditionD")
        self.assertEqual(dataset["nframe"], 32)

    def test_official_config_cannot_change_model_or_frame_sampler(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subset = root / "subset"
            subset.mkdir()
            data = subset / "data.parquet"
            data.write_bytes(b"fixture")
            source_config = subset / "source.json"
            source = {
                "model": {"Qwen2.5-VL-3B-Instruct": {}},
                "data": {"OSI": {"data_path": str(subset), "nframe": 32}},
            }
            source_config.write_text(json.dumps(source), encoding="utf-8")
            condition_path = root / "condition.json"
            condition_path.write_text("{}", encoding="utf-8")
            condition = {
                "subset": {
                    "source_config": str(source_config),
                    "data_path": str(data),
                }
            }
            config = build_config(source, condition_path)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            self.assertEqual(
                validate_official_config(
                    ["--config", str(config_path)], condition, condition_path
                ),
                config_path.resolve(),
            )
            config["data"]["OSI-D-v1"]["nframe"] = 16
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "32-frame"):
                validate_official_config(
                    ["--config", str(config_path)], condition, condition_path
                )


if __name__ == "__main__":
    unittest.main()
