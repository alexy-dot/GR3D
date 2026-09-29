#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from reconstruction.render_dynamic_tracks import (
    deterministic_sample,
    retained_static_indices,
    selected_by_frame,
    sha256,
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

    def test_cli_preserves_verified_external_id_time_and_units(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observations = root / "observations.npz"
            selected = root / "selected.npz"
            states = root / "states.json"
            classification = root / "classification.json"
            audit = root / "track_audit.json"
            contact = root / "contact.jpg"
            poses = root / "camera_poses.npy"
            output = root / "render"
            identity = {
                "namespace": "osi_bench_numeric_id",
                "scene_id": "0000",
                "object_id": "26",
                "object_id_numeric": 26,
                "association_method": "manual_box",
                "verified": True,
            }
            coordinate = {
                "name": "pi3_world",
                "units": "pi3_model_units",
                "axis_values_are_meters": False,
                "metric_scale_validated": False,
                "meters_per_unit": None,
                "scale_provenance": None,
            }
            np.savez_compressed(
                observations,
                points=np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float),
                colors=np.full((3, 3), 128, dtype=np.uint8),
            )
            np.savez_compressed(selected, **{"0": np.asarray([0]), "1": np.asarray([1])})
            contact.write_bytes(b"contact")
            track_manifest_hash = "b" * 64
            audit.write_text(json.dumps({
                "status": "complete",
                "track_candidate_id": "T026",
                "external_identity": identity,
                "track_manifest_sha256": track_manifest_hash,
                "source_frames_modified": False,
                "contact_sheet": {"path": contact.name, "sha256": sha256(contact)},
            }), encoding="utf-8")
            camera_poses = np.repeat(np.eye(4)[None, ...], 2, axis=0)
            camera_poses[1, 0, 3] = 1.0
            np.save(poses, camera_poses)
            state_payload = {
                "track_candidate_id": "T026",
                "external_identity": identity,
                "coordinate_system": coordinate,
                "states": [
                    {
                        "track_candidate_id": "T026", "sample_index": 0,
                        "source_frame_index": 0, "timestamp_seconds": 0.0,
                        "center_xyz_median": [0.0, 0.0, 0.0],
                        "camera_center_xyz": [0.0, 0.0, 0.0], "point_count": 1,
                        "quality_warnings": [],
                    },
                    {
                        "track_candidate_id": "T026", "sample_index": 1,
                        "source_frame_index": 1, "timestamp_seconds": 1.0,
                        "center_xyz_median": [1.0, 0.0, 0.0],
                        "camera_center_xyz": [1.0, 0.0, 0.0], "point_count": 1,
                        "quality_warnings": [],
                    },
                ],
                "provenance": {
                    "pi3_manifest_sha256": "a" * 64,
                    "point_observations_sha256": sha256(observations),
                    "selected_point_indices_sha256": sha256(selected),
                    "camera_poses_sha256": sha256(poses),
                    "track_manifest_sha256": track_manifest_hash,
                },
            }
            states.write_text(json.dumps(state_payload), encoding="utf-8")
            classification.write_text(json.dumps({
                "track_candidate_id": "T026",
                "external_identity": identity,
                "coordinate_system": coordinate,
                "classification": {"motion_state": "dynamic"},
                "input_provenance": {
                    **state_payload["provenance"],
                    "track_states_sha256": sha256(states),
                },
            }), encoding="utf-8")
            script = Path(__file__).with_name("render_dynamic_tracks.py")
            completed = subprocess.run(
                [
                    sys.executable, str(script), str(observations), str(selected),
                    str(states), str(classification), str(output),
                    "--entity-id", "D001", "--camera-poses", str(poses),
                    "--alignment", "camera-gravity", "--track-audit", str(audit),
                    "--max-background-points", "3", "--max-dynamic-points", "3",
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**__import__("os").environ, "MPLCONFIGDIR": str(root / "mpl")},
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["object_id_correspondence"])
            self.assertTrue(manifest["time_conditioned"])
            self.assertFalse(manifest["metric_scale_validated"])
            self.assertEqual(manifest["external_identity"]["object_id"], "26")
            header = (output / "trajectory.csv").read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("valid_3d", header)
            self.assertIn("camera_x", header)


if __name__ == "__main__":
    unittest.main()
