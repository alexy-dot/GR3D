#!/usr/bin/env python3
from __future__ import annotations

import unittest
from pathlib import Path

from reconstruction.prepare_osi_official_subset import (
    build_official_config,
    git_state,
    select_scene_ids,
)


class PrepareOsiOfficialSubsetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {"video": "0000", "category": "a"},
            {"video": "0000", "category": "b"},
            {"video": "0001", "category": "b"},
            {"video": "0001", "category": "c"},
            {"video": "0002", "category": "a"},
            {"video": "0002", "category": "c"},
            {"video": "0003", "category": "a"},
        ]

    def test_selection_is_deterministic_and_covers_categories(self) -> None:
        first = select_scene_ids(self.rows, 2, 7, include_scenes=["0000"])
        second = select_scene_ids(self.rows, 2, 7, include_scenes=["0000"])
        self.assertEqual(first, second)
        self.assertEqual(first[0], "0000")
        categories = {
            row["category"] for row in self.rows if row["video"] in first
        }
        self.assertEqual(categories, {"a", "b", "c"})

    def test_invalid_selection_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "absent"):
            select_scene_ids(self.rows, 2, 7, include_scenes=["9999"])
        with self.assertRaisesRegex(ValueError, "exceeds"):
            select_scene_ids(self.rows, 5, 7)

    def test_official_config_disables_full_download(self) -> None:
        config = build_official_config(Path("/tmp/subset"), 32, "model")
        dataset = config["data"]["OSI-Bench-Subset"]
        self.assertFalse(dataset["download"])
        self.assertEqual(dataset["nframe"], 32)

    def test_git_state_is_optional_outside_repository(self) -> None:
        revision, dirty = git_state(Path("/tmp"))
        self.assertIsNone(revision)
        self.assertIsNone(dirty)


if __name__ == "__main__":
    unittest.main()
