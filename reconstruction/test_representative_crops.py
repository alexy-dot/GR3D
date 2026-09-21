#!/usr/bin/env python3
"""Tests for representative crop geometry."""

from __future__ import annotations

import unittest

import numpy as np

try:
    from reconstruction.extract_representative_crops import crop_quality_warnings, padded_bbox
except ModuleNotFoundError:
    from extract_representative_crops import crop_quality_warnings, padded_bbox


class RepresentativeCropTests(unittest.TestCase):
    def test_padded_bbox_is_clipped_to_image(self) -> None:
        pixels = np.asarray([[0, 1], [9, 8]], dtype=np.int64)
        self.assertEqual(padded_bbox(pixels, 10, 10, 0.2), (0, 0, 10, 10))

    def test_empty_pixels_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero pixels"):
            padded_bbox(np.empty((0, 2), dtype=np.int64), 10, 10, 0.2)

    def test_crop_quality_warnings(self) -> None:
        self.assertEqual(
            crop_quality_warnings((0, 0, 20, 40), 30),
            ["small_crop_dimension", "low_component_pixel_support"],
        )
        self.assertEqual(crop_quality_warnings((0, 0, 40, 40), 100), [])


if __name__ == "__main__":
    unittest.main()
