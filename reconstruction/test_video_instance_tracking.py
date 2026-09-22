#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from reconstruction.run_video_instance_tracking import extract_frames


class VideoInstanceTrackingTests(unittest.TestCase):
    def test_extract_frames_preserves_source_indices_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "input.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (12, 8)
            )
            self.assertTrue(writer.isOpened())
            for value in range(7):
                writer.write(np.full((8, 12, 3), value * 20, dtype=np.uint8))
            writer.release()

            records = extract_frames(video, root / "frames", interval=3, max_frames=None, width=6)

            self.assertEqual([row["sample_index"] for row in records], [0, 1, 2])
            self.assertEqual([row["source_frame_index"] for row in records], [0, 3, 6])
            self.assertEqual([row["timestamp_seconds"] for row in records], [0.0, 0.3, 0.6])
            self.assertTrue(all(len(row["frame_sha256"]) == 64 for row in records))
            frame = cv2.imread(str(root / "frames" / records[0]["frame_path"]))
            self.assertEqual(frame.shape[:2], (4, 6))


if __name__ == "__main__":
    unittest.main()
