#!/usr/bin/env python3
"""Lightweight tests for semantic voxel voting and connectivity."""

from __future__ import annotations

import unittest

import numpy as np

try:
    from reconstruction.build_semantic_voxels import connected_components, majority_labels
except ModuleNotFoundError:  # Also support direct execution from reconstruction/.
    from build_semantic_voxels import connected_components, majority_labels


class SemanticVoxelTests(unittest.TestCase):
    def test_majority_vote(self) -> None:
        inverse = np.asarray([0, 0, 0, 1, 1, 2], dtype=np.int64)
        labels = np.asarray([4, 4, 7, 3, 3, 9], dtype=np.int32)
        majority, votes = majority_labels(inverse, labels, voxel_count=3)
        np.testing.assert_array_equal(majority, [4, 3, 9])
        np.testing.assert_array_equal(votes, [2, 2, 1])

    def test_six_neighbor_same_label_components(self) -> None:
        coordinates = np.asarray(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0], [8, 0, 0]],
            dtype=np.int64,
        )
        labels = np.asarray([1, 1, 2, 2, 1], dtype=np.int32)
        components, count = connected_components(coordinates, labels)
        self.assertEqual(count, 3)
        self.assertEqual(components[0], components[1])
        self.assertEqual(components[2], components[3])
        self.assertNotEqual(components[1], components[2])
        self.assertNotEqual(components[0], components[4])


if __name__ == "__main__":
    unittest.main()
