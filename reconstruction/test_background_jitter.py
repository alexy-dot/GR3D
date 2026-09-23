#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

from reconstruction.estimate_background_jitter import estimate_jitter


class BackgroundJitterTests(unittest.TestCase):
    def test_stationary_background_has_zero_jitter_after_foreground_exclusion(self) -> None:
        points = np.asarray([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [9.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ])
        frames = np.asarray([0, 0, 0, 1, 1, 1])
        intervals, frame_ids = estimate_jitter(
            points, frames, {0: {2}, 1: {5}}, max_points=10
        )
        self.assertEqual(frame_ids, [0, 1])
        self.assertEqual(intervals[0]["background_jitter"], 0.0)
        self.assertEqual(intervals[0]["status"], "valid")
        self.assertEqual(intervals[0]["from_frame"], 0)
        self.assertEqual(intervals[0]["to_frame"], 1)
        self.assertEqual(intervals[0]["from_points"], 2)
        self.assertEqual(intervals[0]["to_points"], 2)

    def test_empty_background_is_recorded_as_missing_evidence(self) -> None:
        points = np.zeros((2, 3))
        frames = np.asarray([0, 1])
        intervals, _ = estimate_jitter(
            points, frames, {0: {0}, 1: {1}}, max_points=10
        )
        self.assertEqual(intervals[0]["status"], "missing_background_support")
        self.assertIsNone(intervals[0]["background_jitter"])


if __name__ == "__main__":
    unittest.main()
