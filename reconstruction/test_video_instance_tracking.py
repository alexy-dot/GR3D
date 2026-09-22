#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import torch

from reconstruction.run_video_instance_tracking import (
    extract_frames,
    normalize_prompts,
    record_masks,
)


class VideoInstanceTrackingTests(unittest.TestCase):
    def test_single_prompt_remains_supported(self) -> None:
        prompt = {
            "track_candidate_id": "T001",
            "sample_index": 2,
            "box_xyxy": [1, 2, 5, 8],
        }
        self.assertEqual(normalize_prompts(prompt)[0]["track_candidate_id"], "T001")

    def test_multi_prompts_require_unique_ids_on_same_frame(self) -> None:
        prompts = {
            "tracks": [
                {"track_candidate_id": "T001", "sample_index": 2, "box_xyxy": [1, 2, 5, 8]},
                {"track_candidate_id": "T002", "sample_index": 2, "box_xyxy": [4, 5, 9, 10]},
            ]
        }
        self.assertEqual(len(normalize_prompts(prompts)), 2)
        prompts["tracks"][1]["sample_index"] = 3
        with self.assertRaisesRegex(ValueError, "one sample_index"):
            normalize_prompts(prompts)
        prompts["tracks"][1]["sample_index"] = 2
        prompts["tracks"][1]["track_candidate_id"] = "T001"
        with self.assertRaisesRegex(ValueError, "unique"):
            normalize_prompts(prompts)

    def test_record_masks_uses_returned_object_id_order(self) -> None:
        masks = {"T001": {}, "T002": {}}
        logits = torch.tensor([[[[-1.0, 2.0]]], [[[3.0, -1.0]]]])
        record_masks(masks, 4, [2, 1], logits, {1: "T001", 2: "T002"})
        self.assertEqual(masks["T002"][4].tolist(), [[[0, 255]]])
        self.assertEqual(masks["T001"][4].tolist(), [[[255, 0]]])

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

    def test_extract_frames_preserves_absolute_indices_for_bounded_segment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "input.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (12, 8)
            )
            for value in range(10):
                writer.write(np.full((8, 12, 3), value * 20, dtype=np.uint8))
            writer.release()

            records = extract_frames(
                video,
                root / "bounded",
                interval=2,
                max_frames=None,
                width=12,
                start_frame=3,
                end_frame=8,
            )

            self.assertEqual([row["source_frame_index"] for row in records], [3, 5, 7])
            self.assertEqual([row["timestamp_seconds"] for row in records], [0.3, 0.5, 0.7])


if __name__ == "__main__":
    unittest.main()
