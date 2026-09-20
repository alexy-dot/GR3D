#!/usr/bin/env python3
"""Lightweight tests for camera trajectory diagnostics."""

from __future__ import annotations

import unittest

import numpy as np

try:
    from reconstruction.analyze_camera_trajectory import trajectory_metrics
except ModuleNotFoundError:
    from analyze_camera_trajectory import trajectory_metrics


class CameraTrajectoryTests(unittest.TestCase):
    def test_straight_path(self) -> None:
        poses = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
        poses[:, 0, 3] = [0.0, 2.0, 5.0]
        result = trajectory_metrics(poses)
        self.assertAlmostEqual(result["path_length_model_units"], 5.0)
        self.assertAlmostEqual(result["endpoint_displacement_model_units"], 5.0)
        self.assertAlmostEqual(result["straightness_ratio"], 1.0)

    def test_turning_path(self) -> None:
        poses = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
        poses[:, :2, 3] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
        result = trajectory_metrics(poses)
        self.assertAlmostEqual(result["path_length_model_units"], 2.0)
        self.assertAlmostEqual(result["straightness_ratio"], np.sqrt(2.0) / 2.0)
        self.assertAlmostEqual(result["raw_xy_heading_change_degrees"], 90.0)


if __name__ == "__main__":
    unittest.main()
