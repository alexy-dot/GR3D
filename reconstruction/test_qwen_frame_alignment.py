#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from reconstruction.extract_qwen_video_frames import qwen_uniform_frame_indices
from reconstruction.input_identity import file_sha256
from reconstruction.run_pi3_baseline import load_manifest_image_directory


class QwenFrameAlignmentTests(unittest.TestCase):
    def test_qwen_indices_match_torch_rounding(self) -> None:
        indices = qwen_uniform_frame_indices(406, 32)
        self.assertEqual(len(indices), 32)
        self.assertEqual(indices[:5], [0, 13, 26, 39, 52])
        self.assertEqual(indices[-5:], [353, 366, 379, 392, 405])
        self.assertEqual(indices, sorted(set(indices)))

    def test_odd_requested_count_is_rounded_to_frame_factor(self) -> None:
        self.assertEqual(len(qwen_uniform_frame_indices(100, 31)), 32)

    def test_manifest_preserves_source_indices_and_rejects_hash_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for sequence_index, source_index in enumerate((0, 13, 26)):
                name = f"{sequence_index:06d}_src{source_index:06d}.png"
                path = root / name
                Image.new("RGB", (16, 12), color=(source_index, 0, 0)).save(path)
                rows.append(
                    {
                        "sequence_index": sequence_index,
                        "source_index": source_index,
                        "timestamp_seconds": source_index / 15.0,
                        "source_name": "0000.mp4",
                        "relative_path": name,
                        "sha256": file_sha256(path),
                    }
                )
            manifest_path = root / "frame_manifest.json"
            manifest_path.write_text(
                json.dumps({"schema_version": 1, "frames": rows}), encoding="utf-8"
            )

            images, records, _ = load_manifest_image_directory(root, manifest_path)
            self.assertEqual(len(images), 3)
            self.assertEqual([record.source_index for record in records], [0, 13, 26])

            (root / rows[1]["relative_path"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_manifest_image_directory(root, manifest_path)


if __name__ == "__main__":
    unittest.main()
