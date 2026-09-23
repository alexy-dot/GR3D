#!/usr/bin/env python3
from __future__ import annotations

import json
import unittest
import tempfile
from pathlib import Path

from reconstruction.prepare_mllm_evaluation import build_requests
from reconstruction.run_dashscope_mllm_evaluation import (
    load_resume_state,
    request_payload,
)
from reconstruction.score_mllm_evaluation import normalize_mcq, parse_number, score_rows


class MllmEvaluationTests(unittest.TestCase):
    def build_package(self, root: Path) -> tuple[dict, dict, list[dict]]:
        for name in ("frame.png", "view.png", "crop.png"):
            (root / name).write_bytes(name.encode("ascii"))
        questions = [{
            "question_id": 1,
            "category": "spatial",
            "question_type": "mcq",
            "question": "Where?",
            "options": ["A. here", "B. there"],
        }]
        (root / "questions.json").write_text(json.dumps(questions), encoding="utf-8")
        scene = {
            "scene_instances": [{
                "scene_instance_id": "S001",
                "scene_id": "scene",
                "semantic_label": 12,
                "semantic_name": "person",
                "center_xyz": [1, 2, 3],
                "bbox_min_xyz": [0, 1, 2],
                "bbox_max_xyz": [2, 3, 4],
                "voxel_count": 4,
                "point_count": 20,
                "mean_voxel_purity": 0.9,
                "max_view_support": 3,
                "multiview_voxel_ratio": 0.5,
                "motion_state": "uncertain",
                "motion_evidence": "not_measured",
                "ground_truth": "must not leak",
            }]
        }
        (root / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
        crops = {"crops": [{"scene_instance_id": "S001", "crop_path": "crop.png"}]}
        (root / "crops.json").write_text(json.dumps(crops), encoding="utf-8")
        manifest = {
            "scene_id": "scene",
            "frames": [{
                "path": "frame.png",
                "sample_index": 0,
                "source_frame_index": 0,
                "timestamp_seconds": 0.0,
            }],
        }
        (root / "package_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        conditions = {
            "raw": {"frames": ["frame.png"], "views": [], "questions": "questions.json"},
            "ids": {
                "frames": ["frame.png"],
                "views": ["view.png"],
                "scene_instances": "scene.json",
                "questions": "questions.json",
            },
            "ids_crops": {
                "frames": ["frame.png"],
                "views": ["view.png"],
                "scene_instances": "scene.json",
                "representative_crop_catalog": "crops.json",
                "representative_crops": ["crop.png"],
                "questions": "questions.json",
            },
        }
        return manifest, conditions, questions

    def test_dashscope_payload_contains_images_but_no_answer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "frame.png").write_bytes(b"png")
            request = {
                "system_prompt": "answer only",
                "question": "Where?",
                "options": ["A. here", "B. there"],
                "images": [{"role": "video_frame", "path": "frame.png", "timestamp_seconds": 0.0}],
            }
            payload = request_payload(request, root, "model")
            serialized = str(payload)
            self.assertIn("data:image/png;base64", serialized)
            self.assertNotIn("ground_truth", serialized)
            self.assertEqual(payload["temperature"], 0)

    def test_scene_context_is_limited_to_id_conditions_and_answer_blind(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, conditions, questions = self.build_package(root)
            prepared = build_requests(root, conditions, questions, manifest)
            by_condition = {row["condition"]: row for row in prepared["requests"]}
            self.assertEqual(prepared["protocol_version"], 2)
            self.assertNotIn("scene_context", by_condition["raw"])
            self.assertEqual(
                by_condition["ids"]["scene_context"]["instances"][0]["scene_instance_id"],
                "S001",
            )
            self.assertNotIn("ground_truth", json.dumps(prepared))

    def test_representative_crop_has_explicit_scene_id_association(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, conditions, questions = self.build_package(root)
            prepared = build_requests(root, conditions, questions, manifest)
            request = next(row for row in prepared["requests"] if row["condition"] == "ids_crops")
            crop = next(row for row in request["images"] if row["role"] == "representative_crop")
            self.assertEqual(crop["scene_instance_id"], "S001")
            payload = request_payload(request, root, "model")
            self.assertIn("scene ID S001", json.dumps(payload))

    def test_resume_requires_exact_metadata_and_unique_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "predictions.json"
            sidecar = root / "predictions.json.run.json"
            metadata = {
                "request_manifest_sha256": "requests",
                "package_manifest_sha256": "package",
                "model": "m",
                "base_url": "https://example.invalid/v1",
                "decoding": {"temperature": 0, "max_tokens": 64},
                "protocol_version": 2,
            }
            sidecar.write_text(json.dumps(metadata), encoding="utf-8")
            output.write_text(json.dumps([{"request_id": "r1"}]), encoding="utf-8")
            rows, completed = load_resume_state(output, sidecar, metadata)
            self.assertEqual(len(rows), 1)
            self.assertEqual(completed, {"r1"})
            mismatches = {
                "request manifest": {"request_manifest_sha256": "other"},
                "package manifest": {"package_manifest_sha256": "other"},
                "model": {"model": "other"},
                "endpoint": {"base_url": "https://other.invalid/v1"},
                "decoding": {"decoding": {"temperature": 0, "max_tokens": 32}},
                "protocol": {"protocol_version": 3},
            }
            for label, change in mismatches.items():
                with self.subTest(label=label), self.assertRaisesRegex(ValueError, "metadata"):
                    load_resume_state(output, sidecar, {**metadata, **change})
            output.write_text(
                json.dumps([{"request_id": "r1"}, {"request_id": "r1"}]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_resume_state(output, sidecar, metadata)

    def test_legacy_predictions_without_sidecar_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "predictions.json"
            output.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "both predictions"):
                load_resume_state(output, root / "predictions.json.run.json", {})

    def test_parsers(self) -> None:
        self.assertEqual(normalize_mcq("Answer: b"), "B")
        self.assertEqual(parse_number("about 12.49 meters"), 12.49)

    def test_scoring_requires_complete_unique_predictions(self) -> None:
        requests = [{"request_id": "s::c::q000", "condition": "c", "question_id": 0, "category": "x", "question_type": "mcq"}]
        truth = [{"question_id": 0, "answer": "B"}]
        result = score_rows(requests, [{"request_id": "s::c::q000", "answer": "B"}], truth)
        self.assertEqual(result["summary"]["c"]["exact_accuracy"], 1.0)
        with self.assertRaisesRegex(ValueError, "missing"):
            score_rows(requests, [], truth)

    def test_numeric_error_is_reported(self) -> None:
        requests = [{"request_id": "s::c::q001", "condition": "c", "question_id": 1, "category": "distance", "question_type": "numerical"}]
        truth = [{"question_id": 1, "answer": "12.49"}]
        result = score_rows(requests, [{"request_id": "s::c::q001", "answer": "12.0 m"}], truth)
        self.assertAlmostEqual(result["rows"][0]["absolute_error"], 0.49)


if __name__ == "__main__":
    unittest.main()
