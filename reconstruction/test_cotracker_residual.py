#!/usr/bin/env python3

from __future__ import annotations

import unittest

import cv2
import numpy as np

from reconstruction.run_cotracker_residual import (
    GROUP_BACKGROUND,
    GROUP_FOREGROUND,
    analyze_residuals,
    sample_queries,
)


class CoTrackerResidualTests(unittest.TestCase):
    def test_query_sampling_separates_eroded_interior_and_background_ring(self) -> None:
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[15:25, 15:25] = 1
        excluded = mask.copy()
        points, groups, summary = sample_queries(mask, excluded, 8, 12, 2, 3, 10)
        foreground = points[groups == GROUP_FOREGROUND].astype(int)
        background = points[groups == GROUP_BACKGROUND].astype(int)
        eroded = cv2.erode(mask, np.ones((5, 5), np.uint8)) > 0
        self.assertTrue(all(eroded[y, x] for x, y in foreground))
        self.assertTrue(all(not excluded[y, x] for x, y in background))
        self.assertEqual(summary["foreground_query_count"], 8)
        self.assertEqual(summary["background_query_count"], 12)

    def test_dynamic_foreground_exceeds_affine_background_residual(self) -> None:
        foreground = np.asarray([[10, 10], [12, 10], [10, 12], [12, 12], [11, 11], [13, 11], [11, 13], [13, 13]], dtype=float)
        background = np.asarray([[0, 0], [20, 0], [0, 20], [20, 20], [5, 5], [15, 5], [5, 15], [15, 15], [10, 0], [0, 10], [20, 10], [10, 20], [3, 8], [17, 8], [8, 3], [8, 17]], dtype=float)
        first = np.concatenate([foreground, background])
        second = first + np.asarray([2.0, 1.0])
        second[: len(foreground)] += np.asarray([4.0, 0.0])
        noise = np.linspace(-0.05, 0.05, len(background))[:, None]
        second[len(foreground) :] += np.concatenate([noise, -noise], axis=1)
        tracks = np.stack([first, second])
        visibility = np.ones(tracks.shape[:2], dtype=bool)
        groups = np.concatenate([
            np.full(len(foreground), GROUP_FOREGROUND),
            np.full(len(background), GROUP_BACKGROUND),
        ])
        rows, summary = analyze_residuals(tracks, visibility, groups, 8, 12)
        self.assertTrue(rows[0]["valid"])
        self.assertGreater(rows[0]["normalized_foreground_residual"], 20)
        self.assertGreater(summary["normalized_foreground_residual_median"], 20)

    def test_visibility_gate_keeps_interval_invalid(self) -> None:
        tracks = np.zeros((2, 12, 2), dtype=float)
        visibility = np.ones((2, 12), dtype=bool)
        visibility[1, :4] = False
        groups = np.asarray([GROUP_FOREGROUND] * 6 + [GROUP_BACKGROUND] * 6)
        rows, summary = analyze_residuals(tracks, visibility, groups, 6, 6)
        self.assertFalse(rows[0]["valid"])
        self.assertIn("insufficient_foreground_visibility", rows[0]["quality_warnings"])
        self.assertEqual(summary["valid_interval_count"], 0)


if __name__ == "__main__":
    unittest.main()
