#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

from reconstruction.audit_video_track import analyze_masks


class VideoTrackAuditTests(unittest.TestCase):
    def test_identical_masks_have_perfect_iou_and_no_warnings(self) -> None:
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 2:6] = True
        rows, summary = analyze_masks([mask, mask.copy(), mask.copy()])
        self.assertEqual(rows[1]["previous_iou"], 1.0)
        self.assertEqual(summary["consecutive_iou_min"], 1.0)
        self.assertEqual(summary["warnings"], [])

    def test_large_area_change_is_reported(self) -> None:
        small = np.zeros((10, 10), dtype=bool)
        small[1:3, 1:3] = True
        large = np.zeros((10, 10), dtype=bool)
        large[1:8, 1:8] = True
        _, summary = analyze_masks([small, large])
        self.assertIn("large_adjacent_area_change", summary["warnings"])

    def test_mismatched_shapes_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "same shape"):
            analyze_masks([np.zeros((2, 2)), np.zeros((3, 3))])


if __name__ == "__main__":
    unittest.main()
