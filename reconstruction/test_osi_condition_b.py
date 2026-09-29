#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from PIL import Image

from reconstruction.input_identity import file_sha256, input_identity
from reconstruction.prepare_osi_condition_b import build_config
from reconstruction.prepare_osi_condition_b import build_condition_b_manifest
from reconstruction.run_osi_condition_b import append_condition_b_evidence


class OSIConditionBTests(unittest.TestCase):
    def test_manifest_builder_closes_video_frame_pi3_and_render_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subset = root / "subset"
            frames = root / "frames"
            pi3 = root / "pi3"
            views = root / "views"
            for path in (subset, frames, pi3, views):
                path.mkdir()

            (subset / "data.parquet").write_bytes(b"questions")
            (subset / "osibench_subset_config.json").write_text("{}", encoding="utf-8")
            video = subset / "0000.mp4"
            video.write_bytes(b"video")
            frame_path = frames / "000000_src000000.png"
            Image.new("RGB", (8, 8)).save(frame_path)
            second_frame = frames / "000001_src000001.png"
            Image.new("RGB", (8, 8), color="white").save(second_frame)
            frame_manifest = {
                "schema_version": 1,
                "source_video_identity": input_identity(video),
                "sampling": {"source_indices": [0, 1]},
                "frames": [
                    {"relative_path": frame_path.name},
                    {"relative_path": second_frame.name},
                ],
            }
            frame_manifest_path = frames / "frame_manifest.json"
            frame_manifest_path.write_text(json.dumps(frame_manifest), encoding="utf-8")

            point_cloud = pi3 / "point_cloud.ply"
            point_cloud.write_bytes(b"ply")
            camera_poses = pi3 / "camera_poses.npy"
            camera_poses.write_bytes(b"poses")
            pi3_manifest = {
                "status": "complete",
                "frame_count": 2,
                "frame_selection": {"manifest_sha256": file_sha256(frame_manifest_path)},
                "input_identity": input_identity(frames),
            }
            (pi3 / "manifest.json").write_text(json.dumps(pi3_manifest), encoding="utf-8")

            for name in ("view_xy.png", "view_xz.png", "view_yz.png"):
                Image.new("RGB", (8, 8)).save(views / name)
            render_manifest = {
                "status": "complete",
                "alignment": "camera-gravity",
                "coordinate_units": "pi3_model_units",
                "axis_values_are_meters": False,
                "metric_scale_validated": False,
                "object_id_correspondence": False,
                "time_conditioned": False,
                "point_cloud_sha256": file_sha256(point_cloud),
                "camera_poses": str(camera_poses.resolve()),
                "outputs": ["view_xy.png", "view_xz.png", "view_yz.png"],
            }
            (views / "render_manifest.json").write_text(
                json.dumps(render_manifest), encoding="utf-8"
            )

            manifest = build_condition_b_manifest(
                subset, "0000", frame_manifest_path, pi3, views
            )
            self.assertTrue(manifest["original_video_preserved"])
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(
                manifest["protocol_version"],
                "condition-b-v2-nonmetric-explicit",
            )
            self.assertEqual(
                manifest["capabilities"],
                {
                    "metric_scale": False,
                    "object_id_correspondence": False,
                    "time_conditioned_evidence": False,
                },
            )
            self.assertEqual(manifest["scenes"]["0000"]["source_indices"], [0, 1])
            self.assertEqual(
                [view["axis"] for view in manifest["scenes"]["0000"]["views"]],
                ["xy", "xz", "yz"],
            )

    def test_config_preserves_model_data_path_and_frame_count(self) -> None:
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
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "condition.json"
            config = build_config(source, manifest)
        self.assertEqual(config["model"], source["model"])
        dataset = config["data"]["OSI-Bench-Subset-B-v2"]
        self.assertEqual(dataset["class"], "OSIConditionB")
        self.assertEqual(dataset["dataset"], "OSI-Bench-Subset-B-v2")
        self.assertEqual(dataset["data_path"], "/data/osi")
        self.assertEqual(dataset["nframe"], 32)

    def test_evidence_is_appended_without_changing_original_messages(self) -> None:
        original = [
            {"type": "text", "value": "question"},
            {"type": "video", "value": "/data/0000.mp4"},
        ]
        scene = {
            "views": [
                {"axis": "xy", "path": "/views/xy.png"},
                {"axis": "xz", "path": "/views/xz.png"},
                {"axis": "yz", "path": "/views/yz.png"},
            ]
        }
        augmented = append_condition_b_evidence(original, scene)
        self.assertEqual(original, [
            {"type": "text", "value": "question"},
            {"type": "video", "value": "/data/0000.mp4"},
        ])
        self.assertEqual([item["type"] for item in augmented], ["text", "video", "text", "image", "image", "image"])
        self.assertEqual([item["value"] for item in augmented[-3:]], ["/views/xy.png", "/views/xz.png", "/views/yz.png"])
        self.assertIn("not meters", augmented[2]["value"])
        self.assertIn("do not link OSI numeric object IDs", augmented[2]["value"])


if __name__ == "__main__":
    unittest.main()
