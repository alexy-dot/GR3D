#!/usr/bin/env python3
"""Tests for semantic audit coverage calculations."""

from __future__ import annotations

import unittest

import numpy as np

try:
    from reconstruction.render_semantic_audit import class_coverage
except ModuleNotFoundError:
    from render_semantic_audit import class_coverage


class SemanticAuditTests(unittest.TestCase):
    def test_coverage(self) -> None:
        labels = np.asarray([[1, 1, 2], [0, 2, 2]], dtype=np.int16)
        result = class_coverage(labels, {"one": 1, "two": 2, "missing": 3})
        self.assertEqual(result["one"]["pixel_count"], 2)
        self.assertAlmostEqual(result["one"]["pixel_ratio"], 2 / 6)
        self.assertEqual(result["two"]["pixel_count"], 3)
        self.assertTrue(result["two"]["present"])
        self.assertFalse(result["missing"]["present"])


if __name__ == "__main__":
    unittest.main()
