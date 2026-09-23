#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from reconstruction.align_track_to_pi3 import match_frames
from reconstruction.input_identity import input_identity


class AlignTrackToPi3Tests(unittest.TestCase):
    def test_matches_original_source_indices_not_sampling_ordinals(self) -> None:
        tracking = [
            {"sample_index": index, "source_frame_index": source}
            for index, source in enumerate((0, 3, 6, 9, 12))
        ]
        pi3 = [
            {"sequence_index": 0, "source_index": 0},
            {"sequence_index": 1, "source_index": 12},
        ]
        matched = match_frames(tracking, pi3)
        self.assertEqual([(row[0], row[1], row[2]["sample_index"]) for row in matched], [(0, 0, 0), (1, 12, 4)])

    def test_duplicate_tracking_source_index_is_rejected(self) -> None:
        tracking = [
            {"sample_index": 0, "source_frame_index": 3},
            {"sample_index": 1, "source_frame_index": 3},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            match_frames(tracking, [{"sequence_index": 0, "source_index": 3}])

    def test_image_directory_run_aligns_without_file_hash_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "house"
            source.mkdir()
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
                source / "frame.png"
            )
            run = root / "run"
            run.mkdir()
            (run / "manifest.json").write_text(
                json.dumps({
                    "input": str(source),
                    "resized_width": 4,
                    "resized_height": 4,
                }),
                encoding="utf-8",
            )
            (run / "frames.txt").write_text(
                json.dumps({
                    "sequence_index": 0,
                    "source_index": 0,
                    "source_name": "frame.png",
                })
                + "\n",
                encoding="utf-8",
            )
            track_root = root / "track"
            track_root.mkdir()
            Image.fromarray(np.full((4, 4), 255, dtype=np.uint8)).save(
                track_root / "mask.png"
            )
            track_path = track_root / "track_manifest.json"
            track_path.write_text(
                json.dumps({
                    "track_candidate_id": "T001",
                    "source_input_identity": input_identity(source),
                    "frames": [{
                        "sample_index": 0,
                        "source_frame_index": 0,
                        "mask_path": "mask.png",
                    }],
                }),
                encoding="utf-8",
            )
            output = root / "aligned"
            script = Path(__file__).with_name("align_track_to_pi3.py")
            completed = subprocess.run(
                [sys.executable, str(script), str(track_path), str(run), str(output)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(
                (output / "track_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                result["pi3_input_identity"]["input_kind"], "image_directory"
            )
            self.assertIsNone(result["pi3_video_sha256"])


if __name__ == "__main__":
    unittest.main()
