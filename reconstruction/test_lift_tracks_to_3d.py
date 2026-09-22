#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from reconstruction.lift_tracks_to_3d import lift_track


class LiftTracksTests(unittest.TestCase):
    def test_mask_selects_matching_frame_pixels_and_robust_center(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask = np.zeros((5, 5), dtype=np.uint8)
            mask[1:4, 1:4] = 255
            Image.fromarray(mask).save(root / "mask.png")
            observations = {
                "points": np.array([[1, 2, 3], [3, 4, 5], [99, 99, 99]], dtype=np.float32),
                "frame_index": np.array([0, 0, 1]),
                "pixel_yx": np.array([[2, 2], [2, 3], [2, 2]]),
                "confidence": np.ones(3),
            }
            entries = [{"track_candidate_id": "T001", "sample_index": 0, "source_frame_index": 12, "timestamp_seconds": 0.4, "mask_path": "mask.png"}]
            result, indices = lift_track(observations, entries, root, (5, 5), min_points=2, erode_pixels=0)
            self.assertEqual(result[0]["center_xyz_median"], [2.0, 3.0, 4.0])
            self.assertEqual(indices["0"].tolist(), [0, 1])

    def test_empty_mask_is_uncertain_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.fromarray(np.zeros((3, 3), dtype=np.uint8)).save(root / "empty.png")
            observations = {"points": np.zeros((1, 3)), "frame_index": np.array([0]), "pixel_yx": np.array([[1, 1]]), "confidence": np.ones(1)}
            entries = [{"track_candidate_id": "T001", "sample_index": 0, "source_frame_index": 0, "timestamp_seconds": 0.0, "mask_path": "empty.png"}]
            result, _ = lift_track(observations, entries, root, (3, 3), min_points=1, erode_pixels=0)
            self.assertIsNone(result[0]["center_xyz_median"])
            self.assertIn("empty_mask_after_erosion", result[0]["quality_warnings"])

    def test_wrong_mask_shape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(root / "mask.png")
            observations = {"points": np.zeros((0, 3)), "frame_index": np.array([]), "pixel_yx": np.zeros((0, 2), dtype=int), "confidence": np.array([])}
            entries = [{"track_candidate_id": "T", "sample_index": 0, "source_frame_index": 0, "timestamp_seconds": 0.0, "mask_path": "mask.png"}]
            with self.assertRaisesRegex(ValueError, "mask shape"):
                lift_track(observations, entries, root, (3, 3))


if __name__ == "__main__":
    unittest.main()
