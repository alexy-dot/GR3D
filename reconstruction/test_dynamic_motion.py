#!/usr/bin/env python3
"""Synthetic validation for background-normalized 3D track motion."""

from __future__ import annotations

import tempfile
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from reconstruction.classify_track_motion import (
    analyze_motion,
    background_intervals_from_payload,
)
from reconstruction.estimate_background_jitter import estimate_jitter
from reconstruction.lift_tracks_to_3d import lift_track


def states(centers: list[list[float] | None]) -> list[dict]:
    return [
        {
            "sample_index": index,
            "timestamp_seconds": float(index),
            "center_xyz_median": center,
        }
        for index, center in enumerate(centers)
    ]


def intervals(values: list[float], frame_ids: list[int] | None = None) -> list[dict]:
    frame_ids = frame_ids or list(range(len(values) + 1))
    return [
        {
            "from_frame": first,
            "to_frame": second,
            "status": "valid",
            "background_jitter": value,
            "from_points": 10,
            "to_points": 10,
        }
        for first, second, value in zip(
            frame_ids[:-1], frame_ids[1:], values, strict=True
        )
    ]


class DynamicMotionTests(unittest.TestCase):
    def test_stationary_track_stays_static_relative_to_background(self) -> None:
        result = analyze_motion(
            states([[0, 0, 0], [0.01, 0, 0], [0.015, 0, 0], [0.02, 0, 0]]),
            intervals([0.02, 0.02, 0.02]),
        )
        self.assertEqual(result["motion_state"], "static")

    def test_coherent_motion_becomes_dynamic(self) -> None:
        result = analyze_motion(
            states([[0, 0, 0], [0.2, 0, 0], [0.4, 0, 0], [0.6, 0, 0]]),
            intervals([0.02, 0.02, 0.02]),
        )
        self.assertEqual(result["motion_state"], "dynamic")
        self.assertGreater(result["direction_consistency"], 0.99)
        self.assertEqual(result["threshold_sweep"]["5.0"]["motion_state"], "dynamic")

    def test_conflicting_motion_is_uncertain(self) -> None:
        result = analyze_motion(
            states([[0, 0, 0], [0.2, 0, 0], [0.0, 0, 0], [0.2, 0, 0]]),
            intervals([0.02, 0.02, 0.02]),
        )
        self.assertEqual(result["motion_state"], "uncertain")

    def test_insufficient_support_is_uncertain(self) -> None:
        result = analyze_motion(
            states([[0, 0, 0], [1, 0, 0]]), intervals([0.01])
        )
        self.assertEqual(result["motion_state"], "uncertain")
        self.assertEqual(result["reason"], "insufficient_valid_3d_states")

    def test_missing_interval_is_uncertain_not_a_count_exception(self) -> None:
        result = analyze_motion(
            states([[0, 0, 0], [1, 0, 0], [2, 0, 0]]),
            intervals([0.1]),
        )
        self.assertEqual(result["motion_state"], "uncertain")
        self.assertEqual(result["reason"], "missing_background_interval_evidence")
        self.assertEqual(result["missing_background_intervals"], [[1, 2]])

    def test_same_count_but_wrong_frame_pairs_is_uncertain(self) -> None:
        result = analyze_motion(
            states([[0, 0, 0], [1, 0, 0], [2, 0, 0]]),
            intervals([0.1, 0.1], [4, 5, 6]),
        )
        self.assertEqual(result["reason"], "missing_background_interval_evidence")

    def test_invalid_background_support_is_uncertain(self) -> None:
        evidence = intervals([0.1, 0.1])
        evidence[1]["status"] = "missing_background_support"
        evidence[1]["background_jitter"] = None
        result = analyze_motion(
            states([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]]), evidence
        )
        self.assertEqual(result["reason"], "invalid_background_interval_evidence")
        self.assertEqual(result["invalid_background_intervals"], [[1, 2]])

    def test_consecutive_invalid_states_preserve_warnings_and_support(self) -> None:
        rows = states([[0, 0, 0], None, None, [0.3, 0, 0]])
        rows[1].update(point_count=0, quality_warnings=["empty_mask_after_erosion"])
        rows[2].update(point_count=3, quality_warnings=["insufficient_3d_support"])
        result = analyze_motion(rows, intervals([0.01, 0.01, 0.01]))
        self.assertEqual(result["motion_state"], "uncertain")
        self.assertEqual(result["reason"], "invalid_3d_state_evidence")
        self.assertEqual(result["invalid_sample_indices"], [1, 2])
        self.assertEqual(result["state_evidence"][1]["point_count"], 0)
        self.assertIn(
            "empty_mask_after_erosion",
            result["state_evidence"][1]["quality_warnings"],
        )

    def test_valid_invalid_valid_pipeline_returns_uncertain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = []
            for index in range(3):
                mask = np.zeros((4, 4), dtype=np.uint8)
                if index != 1:
                    mask[1:3, 1:3] = 255
                Image.fromarray(mask).save(root / f"mask_{index}.png")
                entries.append({
                    "track_candidate_id": "T",
                    "sample_index": index,
                    "source_frame_index": index,
                    "timestamp_seconds": float(index),
                    "mask_path": f"mask_{index}.png",
                })
            points = []
            frames = []
            pixels = []
            for frame in range(3):
                for y, x in ((1, 1), (1, 2), (2, 1), (2, 2), (0, 0)):
                    points.append([float(x) + frame * 0.1, float(y), 0.0])
                    frames.append(frame)
                    pixels.append([y, x])
            observations = {
                "points": np.asarray(points),
                "frame_index": np.asarray(frames),
                "pixel_yx": np.asarray(pixels),
                "confidence": np.ones(len(points)),
            }
            lifted, selected = lift_track(
                observations, entries, root, (4, 4), min_points=4, erode_pixels=0
            )
            jitter, _ = estimate_jitter(
                observations["points"],
                observations["frame_index"],
                {int(key): set(value.tolist()) for key, value in selected.items()},
                max_points=20,
            )
            result = analyze_motion(lifted, jitter)
            self.assertEqual(result["motion_state"], "uncertain")
            self.assertEqual(result["reason"], "invalid_3d_state_evidence")
            self.assertEqual(result["invalid_sample_indices"], [1])
            self.assertIn(
                "empty_mask_after_erosion", result["invalid_state_reasons"]["1"]
            )

    def test_legacy_background_artifact_converts_to_explicit_pairs(self) -> None:
        converted, warnings = background_intervals_from_payload({
            "background_jitter": [0.1, 0.2],
            "interval_support": [
                {"from_frame": 0, "to_frame": 1, "from_points": 5, "to_points": 6},
                {"from_frame": 1, "to_frame": 2, "from_points": 6, "to_points": 7},
            ],
        })
        self.assertEqual(
            [(row["from_frame"], row["to_frame"]) for row in converted],
            [(0, 1), (1, 2)],
        )
        self.assertEqual(
            warnings, ["legacy_background_evidence_converted"]
        )

    def test_cli_handoff_uses_separate_saved_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            states_path = root / "track_3d_states.json"
            evidence_path = root / "background_jitter.json"
            output_path = root / "classification.json"
            states_path.write_text(
                json.dumps({
                    "track_candidate_id": "T",
                    "states": states([[0, 0, 0], [0.01, 0, 0], [0.02, 0, 0]]),
                    "provenance": {"source": "synthetic"},
                }),
                encoding="utf-8",
            )
            evidence_path.write_text(
                json.dumps({
                    "schema_version": 2,
                    "background_intervals": intervals([0.02, 0.02]),
                }),
                encoding="utf-8",
            )
            script = Path(__file__).with_name("classify_track_motion.py")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    str(states_path),
                    str(output_path),
                    "--background-evidence",
                    str(evidence_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(result["classification"]["motion_state"], "static")
            self.assertIn("track_states_sha256", result["input_provenance"])
            self.assertIn("background_evidence_sha256", result["input_provenance"])


if __name__ == "__main__":
    unittest.main()
