#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

from reconstruction.compare_video_tracks import compare_masks, summarize


class CompareVideoTracksTests(unittest.TestCase):
    def test_identical_masks_have_perfect_agreement(self) -> None:
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True
        result = compare_masks(mask, mask.copy())
        self.assertEqual(result["iou"], 1.0)
        self.assertEqual(result["dice"], 1.0)
        self.assertEqual(result["centroid_distance_image_diagonal"], 0.0)

    def test_partial_overlap_reports_directional_coverage(self) -> None:
        reference = np.zeros((4, 4), dtype=bool)
        candidate = np.zeros((4, 4), dtype=bool)
        reference[1:3, 1:3] = True
        candidate[1:3, 2:4] = True
        result = compare_masks(reference, candidate)
        self.assertAlmostEqual(result["iou"], 1 / 3)
        self.assertEqual(result["candidate_covered_by_reference"], 0.5)
        self.assertEqual(result["reference_covered_by_candidate"], 0.5)

    def test_summary_counts_exact_hash_matches(self) -> None:
        rows = [
            {
                **compare_masks(np.ones((2, 2)), np.ones((2, 2))),
                "mask_hash_equal": True,
            },
            {
                **compare_masks(np.ones((2, 2)), np.eye(2)),
                "mask_hash_equal": False,
            },
        ]
        result = summarize(rows)
        self.assertEqual(result["frame_count"], 2)
        self.assertEqual(result["exact_mask_hash_matches"], 1)
        self.assertAlmostEqual(result["iou_min"], 0.5)


if __name__ == "__main__":
    unittest.main()
