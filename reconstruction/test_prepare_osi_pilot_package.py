#!/usr/bin/env python3
"""Tests for OSI pilot package preparation."""

from __future__ import annotations

import unittest

try:
    from reconstruction.prepare_osi_pilot_package import strip_object_ids
except ModuleNotFoundError:
    from prepare_osi_pilot_package import strip_object_ids


class OsiPilotPackageTests(unittest.TestCase):
    def test_strip_object_ids(self) -> None:
        text = "person(id: 30) beside bicycles (id: 07)"
        self.assertEqual(strip_object_ids(text), "person beside bicycles")

    def test_preserve_non_id_numbers(self) -> None:
        text = "At 2.1 seconds, how many objects are within 38 meters?"
        self.assertEqual(strip_object_ids(text), text)


if __name__ == "__main__":
    unittest.main()
