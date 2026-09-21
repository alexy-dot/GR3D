#!/usr/bin/env python3
"""Tests for deterministic 3D-only scene IDs."""

from __future__ import annotations

import copy
import unittest

try:
    from reconstruction.export_scene_instances import (
        assign_scene_ids,
        validate_instance_table,
        validate_rendered_ids,
    )
except ModuleNotFoundError:
    from export_scene_instances import (
        assign_scene_ids,
        validate_instance_table,
        validate_rendered_ids,
    )


def candidate(component: int, name: str, center: list[float]) -> dict[str, object]:
    return {
        "component_id": component,
        "semantic_label": component + 10,
        "semantic_name": name,
        "voxel_count": 4,
        "point_count": 12,
        "mean_voxel_purity": 0.9,
        "max_view_support": 3,
        "multiview_voxel_ratio": 0.5,
        "center_xyz": center,
        "bbox_min_xyz": [value - 0.1 for value in center],
        "bbox_max_xyz": [value + 0.1 for value in center],
    }


class SceneInstanceTests(unittest.TestCase):
    def test_assignment_is_deterministic_under_input_reordering(self) -> None:
        rows = [candidate(7, "chair", [1, 0, 0]), candidate(2, "bicycle", [0, 0, 0])]
        first = assign_scene_ids(rows, "test", 1e-6)
        second = assign_scene_ids(list(reversed(copy.deepcopy(rows))), "test", 1e-6)
        self.assertEqual(first, second)
        self.assertEqual([row["scene_instance_id"] for row in first], ["S001", "S002"])

    def test_layout_components_are_excluded_by_default(self) -> None:
        rows = [candidate(1, "floor", [0, 0, 0]), candidate(2, "chair", [1, 0, 0])]
        result = assign_scene_ids(rows, "test", 1e-6)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["semantic_name"], "chair")

    def test_duplicate_ids_are_rejected(self) -> None:
        rows = assign_scene_ids([candidate(1, "chair", [0, 0, 0])], "test", 1e-6)
        rows.append(copy.deepcopy(rows[0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_instance_table(rows)

    def test_render_table_disagreement_is_rejected(self) -> None:
        rows = assign_scene_ids([candidate(1, "chair", [0, 0, 0])], "test", 1e-6)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            validate_rendered_ids(rows, ["S999"])


if __name__ == "__main__":
    unittest.main()
