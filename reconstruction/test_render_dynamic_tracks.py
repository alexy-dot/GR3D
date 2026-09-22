#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from reconstruction.render_dynamic_tracks import (
    deterministic_sample,
    retained_static_indices,
    selected_by_frame,
)


class RenderDynamicTracksTests(unittest.TestCase):
    def test_deterministic_sample_keeps_fixed_endpoints(self) -> None:
        values = np.arange(10)
        self.assertEqual(deterministic_sample(values, 4).tolist(), [0, 3, 6, 9])

    def test_selected_indices_are_bounds_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected.npz"
            np.savez_compressed(path, **{"0": np.asarray([0, 3])})
            with self.assertRaisesRegex(ValueError, "out of bounds"):
                selected_by_frame(path, point_count=3)

    def test_static_tracks_are_retained_and_dynamic_tracks_are_excluded(self) -> None:
        tracked = np.asarray([1, 3])
        self.assertEqual(retained_static_indices(5, tracked, "static").tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(retained_static_indices(5, tracked, "dynamic").tolist(), [0, 2, 4])
        self.assertEqual(retained_static_indices(5, tracked, "uncertain").tolist(), [0, 2, 4])


if __name__ == "__main__":
    unittest.main()
