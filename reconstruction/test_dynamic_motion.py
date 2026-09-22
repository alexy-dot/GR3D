#!/usr/bin/env python3
"""Synthetic validation for background-normalized 3D track motion."""

from __future__ import annotations

import unittest

from reconstruction.classify_track_motion import analyze_motion


def states(centers: list[list[float]]) -> list[dict]:
    return [
        {
            "sample_index": index,
            "timestamp_seconds": float(index),
            "center_xyz_median": center,
        }
        for index, center in enumerate(centers)
    ]


class DynamicMotionTests(unittest.TestCase):
    def test_stationary_track_stays_static_relative_to_background(self) -> None:
        result = analyze_motion(
            states([[0, 0, 0], [0.01, 0, 0], [0.015, 0, 0], [0.02, 0, 0]]),
            [0.02, 0.02, 0.02],
        )
        self.assertEqual(result["motion_state"], "static")

    def test_coherent_motion_becomes_dynamic(self) -> None:
        result = analyze_motion(
            states([[0, 0, 0], [0.2, 0, 0], [0.4, 0, 0], [0.6, 0, 0]]),
            [0.02, 0.02, 0.02],
        )
        self.assertEqual(result["motion_state"], "dynamic")
        self.assertGreater(result["direction_consistency"], 0.99)
        self.assertEqual(result["threshold_sweep"]["5.0"]["motion_state"], "dynamic")

    def test_conflicting_motion_is_uncertain(self) -> None:
        result = analyze_motion(
            states([[0, 0, 0], [0.2, 0, 0], [0.0, 0, 0], [0.2, 0, 0]]),
            [0.02, 0.02, 0.02],
        )
        self.assertEqual(result["motion_state"], "uncertain")

    def test_insufficient_support_is_uncertain(self) -> None:
        result = analyze_motion(states([[0, 0, 0], [1, 0, 0]]), [0.01])
        self.assertEqual(result["motion_state"], "uncertain")
        self.assertEqual(result["reason"], "insufficient_valid_3d_states")

    def test_jitter_count_must_match_intervals(self) -> None:
        with self.assertRaisesRegex(ValueError, "one value per"):
            analyze_motion(states([[0, 0, 0], [1, 0, 0], [2, 0, 0]]), [0.1])


if __name__ == "__main__":
    unittest.main()
