#!/usr/bin/env python3

from __future__ import annotations

import unittest
from pathlib import Path
import subprocess
import sys

import numpy as np

from reconstruction.detect_video_candidates import (
    filtered_candidates,
    make_tracking_prompt,
    select_prompt_candidate,
)


class VideoCandidateDetectionTests(unittest.TestCase):
    def test_direct_script_entry_point_loads(self) -> None:
        script = Path(__file__).with_name("detect_video_candidates.py")
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=script.parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--checkpoint", result.stdout)

    def test_filtering_is_thresholded_class_filtered_and_deterministic(self) -> None:
        rows = filtered_candidates(
            np.asarray([[5, 5, 15, 25], [0, 0, 8, 8], [1, 1, 4, 4]]),
            np.asarray([1, 2, 1]),
            np.asarray([0.8, 0.95, 0.4]),
            ["__background__", "person", "bicycle"],
            0.7,
            {"person", "bicycle"},
        )
        self.assertEqual([row["semantic_name"] for row in rows], ["bicycle", "person"])
        self.assertEqual(rows[0]["box_xyxy"], [0.0, 0.0, 8.0, 8.0])

    def test_prompt_selection_uses_highest_score_on_exact_source_frame(self) -> None:
        candidates = [
            {"detection_id": "a", "source_frame_index": 11, "semantic_name": "person", "score": 0.8, "area_pixels": 100},
            {"detection_id": "b", "source_frame_index": 11, "semantic_name": "person", "score": 0.9, "area_pixels": 90},
            {"detection_id": "c", "source_frame_index": 12, "semantic_name": "person", "score": 0.99, "area_pixels": 200},
        ]
        self.assertEqual(select_prompt_candidate(candidates, 11, "person")["detection_id"], "b")

    def test_prompt_selection_rejects_missing_candidate(self) -> None:
        with self.assertRaisesRegex(ValueError, "no candidate"):
            select_prompt_candidate([], 11, "person")

    def test_tracking_prompt_uses_outward_integer_box_and_provenance(self) -> None:
        selected = {
            "detection_id": "F000018C001",
            "sample_index": 18,
            "source_frame_index": 111,
            "semantic_name": "person",
            "score": 0.987,
            "box_xyxy": [340.8, 130.2, 391.1, 264.01],
        }
        prompt = make_tracking_prompt(selected, "T_AUTO_001", Path("/tmp/input.mp4"))
        self.assertEqual(prompt["box_xyxy"], [340, 130, 392, 265])
        self.assertEqual(prompt["sample_index"], 18)
        self.assertEqual(prompt["source_frame_index"], 111)
        self.assertEqual(prompt["detector_candidate_id"], "F000018C001")
        self.assertEqual(prompt["detector_score"], 0.987)


if __name__ == "__main__":
    unittest.main()
