#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from reconstruction.tracking_contracts import (
    external_identity_key,
    load_metric_scale_calibration,
    normalize_external_identity,
    pi3_coordinate_system,
)


class TrackingContractsTests(unittest.TestCase):
    def test_osi_identity_preserves_display_id_and_normalizes_scene(self) -> None:
        identity = normalize_external_identity({
            "namespace": "osi_bench_numeric_id",
            "scene_id": "0",
            "object_id": "03",
            "association_method": "manual_box",
            "verified": True,
        })
        self.assertEqual(identity["scene_id"], "0000")
        self.assertEqual(identity["object_id"], "03")
        self.assertEqual(identity["object_id_numeric"], 3)
        self.assertEqual(
            external_identity_key(identity),
            ("osi_bench_numeric_id", "0000", 3),
        )

    def test_invalid_identity_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "association_method"):
            normalize_external_identity({
                "namespace": "osi_bench_numeric_id",
                "scene_id": "0000",
                "object_id": "26",
                "association_method": "guessed",
                "verified": True,
            })

    def test_metric_scale_must_be_independent_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "known_distance.json"
            source.write_text('{"distance_meters": 2.0}\n', encoding="utf-8")
            import hashlib

            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            calibration = root / "scale.json"
            payload = {
                "schema_version": 1,
                "status": "validated",
                "source_type": "known_distance",
                "independent_of_benchmark_answers": True,
                "pi3_manifest_sha256": "a" * 64,
                "coordinate_system_name": "pi3_world",
                "source_artifact": str(source),
                "source_artifact_sha256": digest,
                "meters_per_model_unit": 2.5,
            }
            calibration.write_text(json.dumps(payload), encoding="utf-8")
            result = load_metric_scale_calibration(
                calibration,
                coordinate_system=pi3_coordinate_system("pi3"),
                expected_pi3_manifest_sha256="a" * 64,
            )
            self.assertEqual(result["meters_per_model_unit"], 2.5)
            payload["independent_of_benchmark_answers"] = False
            calibration.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "independent"):
                load_metric_scale_calibration(
                    calibration,
                    coordinate_system=pi3_coordinate_system("pi3"),
                    expected_pi3_manifest_sha256="a" * 64,
                )


if __name__ == "__main__":
    unittest.main()
