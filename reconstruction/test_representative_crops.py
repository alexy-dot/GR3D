#!/usr/bin/env python3
"""Tests for representative crop geometry."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from reconstruction.extract_representative_crops import crop_quality_warnings, padded_bbox
except ModuleNotFoundError:
    from extract_representative_crops import crop_quality_warnings, padded_bbox


class RepresentativeCropTests(unittest.TestCase):
    def test_padded_bbox_is_clipped_to_image(self) -> None:
        pixels = np.asarray([[0, 1], [9, 8]], dtype=np.int64)
        self.assertEqual(padded_bbox(pixels, 10, 10, 0.2), (0, 0, 10, 10))

    def test_empty_pixels_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero pixels"):
            padded_bbox(np.empty((0, 2), dtype=np.int64), 10, 10, 0.2)

    def test_crop_quality_warnings(self) -> None:
        self.assertEqual(
            crop_quality_warnings((0, 0, 20, 40), 30),
            ["small_crop_dimension", "low_component_pixel_support"],
        )
        self.assertEqual(crop_quality_warnings((0, 0, 40, 40), 100), [])

    def test_image_directory_run_extracts_crop_without_file_hash_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "house"
            source.mkdir()
            Image.fromarray(np.full((4, 4, 3), 127, dtype=np.uint8)).save(
                source / "frame.png"
            )
            run = root / "run"
            run.mkdir()
            (run / "manifest.json").write_text(
                json.dumps({
                    "input": str(source),
                    "resized_width": 4,
                    "resized_height": 4,
                }),
                encoding="utf-8",
            )
            (run / "frames.txt").write_text(
                json.dumps({
                    "sequence_index": 0,
                    "source_index": 0,
                    "source_name": "frame.png",
                    "timestamp_seconds": None,
                })
                + "\n",
                encoding="utf-8",
            )
            np.savez_compressed(
                run / "point_observations.npz",
                points=np.asarray([[0.1, 0.1, 0.1]]),
                frame_index=np.asarray([0]),
                pixel_yx=np.asarray([[1, 1]]),
            )
            voxels_path = root / "voxels.npz"
            np.savez_compressed(
                voxels_path,
                coordinates=np.asarray([[0, 0, 0]]),
                component_id=np.asarray([7]),
            )
            objects_path = root / "objects.json"
            objects_path.write_text(
                json.dumps({"manifest": {"voxel_size": 1.0}}), encoding="utf-8"
            )
            instances_path = root / "scene_instances.json"
            instances_path.write_text(
                json.dumps({
                    "manifest": {"scene_id": "house"},
                    "scene_instances": [{
                        "scene_instance_id": "S001",
                        "source_component_id": 7,
                        "semantic_name": "chair",
                        "semantic_label": 1,
                    }],
                }),
                encoding="utf-8",
            )
            output = root / "crops"
            script = Path(__file__).with_name("extract_representative_crops.py")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--run-directory",
                    str(run),
                    "--voxels",
                    str(voxels_path),
                    "--objects",
                    str(objects_path),
                    "--scene-instances",
                    str(instances_path),
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(
                (output / "representative_crops.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                result["manifest"]["source_input_identity"]["input_kind"],
                "image_directory",
            )
            self.assertIsNone(result["manifest"]["source_video_sha256"])


if __name__ == "__main__":
    unittest.main()
