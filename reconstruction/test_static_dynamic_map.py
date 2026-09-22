#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

from reconstruction.build_static_dynamic_map import partition_indices


class StaticDynamicMapTests(unittest.TestCase):
    def test_partition_is_disjoint_and_dynamic_wins_overlap(self) -> None:
        result = partition_indices(6, [("dynamic", np.array([1, 2])), ("uncertain", np.array([2, 3]))])
        self.assertEqual(result["static"].tolist(), [0, 4, 5])
        self.assertEqual(result["dynamic"].tolist(), [1, 2])
        self.assertEqual(result["uncertain"].tolist(), [3])

    def test_static_track_does_not_remove_points(self) -> None:
        result = partition_indices(3, [("static", np.array([1]))])
        self.assertEqual(result["static"].tolist(), [0, 1, 2])

    def test_out_of_bounds_index_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "bounds"):
            partition_indices(2, [("dynamic", np.array([2]))])


if __name__ == "__main__":
    unittest.main()
