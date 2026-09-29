#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from reconstruction.osi_question_evidence import (
    build_question_evidence,
    interpolate_track,
    parse_question_references,
    render_question_evidence,
)


def trajectory(object_id: str, offset: float = 0.0) -> dict:
    return {
        "external_identity": {
            "namespace": "osi_bench_numeric_id",
            "scene_id": "0000",
            "object_id": object_id,
            "object_id_numeric": int(object_id),
            "association_method": "manual_box",
            "verified": True,
        },
        "track_candidate_id": f"T{object_id}",
        "motion_state": "dynamic",
        "rows": [
            {
                "timestamp_seconds": time,
                "x": time + offset,
                "y": 0,
                "z": 0,
                "camera_x": 0,
                "camera_y": 0,
                "camera_z": 0,
            }
            for time in (0.0, 1.0, 2.0, 3.0)
        ],
    }


class OsiQuestionEvidenceTests(unittest.TestCase):
    def test_question_parser_keeps_leading_zero_ids_semantically(self) -> None:
        ids, times = parse_question_references(
            "Where is the window(id: 03) at 2.1 seconds and 4.1s?"
        )
        self.assertEqual(ids, [3])
        self.assertEqual(times, [2.1, 4.1])

    def test_interpolation_records_bracketing_samples(self) -> None:
        result = interpolate_track(trajectory("26")["rows"], 1.5, 2.0)
        self.assertEqual(result["position_xyz"], [1.5, 0.0, 0.0])
        self.assertEqual(result["bracketing_times_seconds"], [1.0, 2.0])
        self.assertEqual(result["method"], "linear_interpolation")

    def test_out_of_range_time_is_not_extrapolated(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            interpolate_track(trajectory("26")["rows"], 4.0, 2.0)

    def test_invalid_state_cannot_be_interpolated_across(self) -> None:
        rows = trajectory("26")["rows"]
        rows[1]["valid_3d"] = "False"
        with self.assertRaisesRegex(ValueError, "invalid"):
            interpolate_track(rows, 1.5, 2.0)

    def test_displacement_is_nonmetric_without_independent_scale(self) -> None:
        evidence = build_question_evidence(
            question_id=6,
            question=(
                "What is the displacement distance of the person(id: 26) "
                "between 1s and 3s in meters?"
            ),
            category="absolute_displacement",
            trajectories={26: trajectory("26")},
        )
        self.assertEqual(evidence["status"], "ready")
        self.assertEqual(
            evidence["derived_geometry"]["object_displacement_model_units"], 2.0
        )
        self.assertNotIn("object_displacement_meters", evidence["derived_geometry"])
        self.assertFalse(evidence["metric_scale_validated"])

    def test_validated_scale_enables_separate_metric_value(self) -> None:
        evidence = build_question_evidence(
            question_id=6,
            question="person(id: 26) between 1s and 3s",
            category="absolute_displacement",
            trajectories={26: trajectory("26")},
            meters_per_model_unit=2.5,
        )
        self.assertEqual(evidence["derived_geometry"]["object_displacement_meters"], 5.0)

    def test_missing_track_and_time_coverage_are_explicit(self) -> None:
        missing = build_question_evidence(
            question_id=1,
            question="person(id: 30) at 1s",
            category="object_3d_localization",
            trajectories={26: trajectory("26")},
        )
        self.assertEqual(missing["status"], "unavailable_missing_object_tracks")
        uncovered = build_question_evidence(
            question_id=2,
            question="person(id: 26) at 9s",
            category="object_3d_localization",
            trajectories={26: trajectory("26")},
        )
        self.assertEqual(uncovered["status"], "unavailable_time_coverage")

    def test_ready_evidence_renders_a_question_specific_image(self) -> None:
        evidence = build_question_evidence(
            question_id=6,
            question="person(id: 26) between 1s and 3s",
            category="absolute_displacement",
            trajectories={26: trajectory("26")},
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evidence.png"
            render_question_evidence(evidence, {26: trajectory("26")}, output)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
