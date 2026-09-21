#!/usr/bin/env python3
"""Negative tests for Phase-B provenance and experimental controls."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

try:
    from reconstruction.prepare_osi_pilot_package import ensure_fresh_output
    from reconstruction.validate_osi_pilot_package import (
        validate_declared_file_set,
        validate_scene_relationships,
    )
except ModuleNotFoundError:
    from prepare_osi_pilot_package import ensure_fresh_output
    from validate_osi_pilot_package import (
        validate_declared_file_set,
        validate_scene_relationships,
    )


def fixtures() -> tuple[dict, dict, dict, dict, dict, list, str]:
    common = {
        "color_by": "component",
        "style": "both",
        "layer": "objects",
        "layout_class_policy": ["floor"],
        "retained_semantic_labels": {"12": "person"},
        "min_component_points": 0,
        "marker_size": 4.0,
        "voxel_count": 10,
        "component_count": 1,
        "rendered_component_ids": [4],
        "semantic_class_count": 1,
        "voxels_sha256": "voxels",
        "camera_poses_sha256": "poses",
        "objects_json_sha256": "objects",
        "alignment": {"quality": "pass"},
        "clip_percentile": 1.0,
        "bounds_xyz": [[0, 0, 0], [1, 1, 1]],
    }
    no_id = {**common, "representation": "no_id_semantic_voxel_blocks"}
    id_render = {
        **common,
        "representation": "3d_only_scene_id_component_candidates",
        "scene_instances_sha256": "scene-table",
        "rendered_ids": ["S001"],
    }
    scene = {
        "manifest": {
            "scene_id": "0000",
            "source_objects_sha256": "objects",
            "source_observations_sha256": "observations",
            "source_run_manifest_sha256": "run",
            "source_video_sha256": "video",
        },
        "scene_instances": [
            {
                "scene_instance_id": "S001",
                "source_component_id": 4,
                "semantic_label": 12,
                "semantic_name": "person",
                "motion_state": "uncertain",
                "motion_evidence": "not_measured_phase_a_static_candidate",
            }
        ],
    }
    crop = {
        "manifest": {
            "scene_id": "0000",
            "scene_instances_sha256": "scene-table",
            "point_observations_sha256": "observations",
            "source_run_manifest_sha256": "run",
            "source_video_sha256": "video",
        },
        "crops": [
            {
                "scene_instance_id": "S001",
                "source_component_id": 4,
                "semantic_label": 12,
                "semantic_name": "person",
                "sample_index": 0,
                "source_frame_index": 57,
                "timestamp_seconds": 3.8,
            }
        ],
    }
    package = {
        "scene_id": "0000",
        "source_run_manifest_sha256": "run",
        "source_video_sha256": "video",
    }
    frames = [{"sample_index": 0, "source_frame_index": 57, "timestamp_seconds": 3.8}]
    return package, scene, no_id, id_render, crop, frames, "scene-table"


class PhaseBHardeningTests(unittest.TestCase):
    def validate(self, values: tuple[dict, dict, dict, dict, dict, list, str]) -> None:
        validate_scene_relationships(*values)

    def test_valid_relationships(self) -> None:
        self.validate(fixtures())

    def test_mismatched_scene_table_is_rejected(self) -> None:
        values = list(copy.deepcopy(fixtures()))
        values[3]["scene_instances_sha256"] = "other-table"
        with self.assertRaisesRegex(ValueError, "different scene table"):
            self.validate(tuple(values))

    def test_mismatched_crop_identity_is_rejected(self) -> None:
        values = list(copy.deepcopy(fixtures()))
        values[4]["crops"][0]["semantic_label"] = 99
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.validate(tuple(values))

    def test_mismatched_render_base_is_rejected(self) -> None:
        values = list(copy.deepcopy(fixtures()))
        values[3]["voxels_sha256"] = "other-voxels"
        with self.assertRaisesRegex(ValueError, "render bases differ"):
            self.validate(tuple(values))

    def test_mismatched_scene_id_is_rejected(self) -> None:
        values = list(copy.deepcopy(fixtures()))
        values[4]["manifest"]["scene_id"] = "9999"
        with self.assertRaisesRegex(ValueError, "different scene IDs"):
            self.validate(tuple(values))

    def test_mismatched_source_run_is_rejected(self) -> None:
        values = list(copy.deepcopy(fixtures()))
        values[4]["manifest"]["source_run_manifest_sha256"] = "other-run"
        with self.assertRaisesRegex(ValueError, "different reconstruction runs"):
            self.validate(tuple(values))

    def test_unmeasured_static_candidate_is_rejected(self) -> None:
        values = list(copy.deepcopy(fixtures()))
        values[1]["scene_instances"][0]["motion_state"] = "static"
        with self.assertRaisesRegex(ValueError, "falsely marked static"):
            self.validate(tuple(values))

    def test_nonempty_output_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "stale.txt").write_text("stale", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not empty"):
                ensure_fresh_output(output)

    def test_undeclared_extra_file_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unexpected.txt"):
            validate_declared_file_set(
                {"package_manifest.json", "questions.json"},
                {"package_manifest.json", "questions.json", "unexpected.txt"},
            )


if __name__ == "__main__":
    unittest.main()
