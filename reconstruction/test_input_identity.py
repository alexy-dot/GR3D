#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reconstruction.input_identity import (
    file_sha256,
    image_directory_files,
    input_identity,
)


class InputIdentityTests(unittest.TestCase):
    def test_file_identity_preserves_streaming_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.mp4"
            path.write_bytes(b"video-bytes")
            identity = input_identity(path)
            self.assertEqual(identity["input_kind"], "file")
            self.assertEqual(identity["combined_sha256"], file_sha256(path))
            self.assertEqual(identity["file_count"], 1)
            self.assertEqual(identity["ordered_entries"][0]["relative_path"], "clip.mp4")

    def test_directory_identity_is_sorted_and_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            b = root / "B.jpg"
            a = root / "a.png"
            b.write_bytes(b"b")
            a.write_bytes(b"a")
            with mock.patch.object(Path, "iterdir", return_value=iter([b, a])):
                first_files = image_directory_files(root)
            with mock.patch.object(Path, "iterdir", return_value=iter([a, b])):
                second_files = image_directory_files(root)
            self.assertEqual(first_files, second_files)
            first = input_identity(root)
            second = input_identity(root)
            self.assertEqual(first, second)
            self.assertEqual(
                [row["relative_path"] for row in first["ordered_entries"]],
                ["B.jpg", "a.png"],
            )

    def test_bytes_filename_and_membership_change_directory_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "a.png"
            image.write_bytes(b"a")
            original = input_identity(root)["combined_sha256"]
            image.write_bytes(b"changed")
            changed_bytes = input_identity(root)["combined_sha256"]
            self.assertNotEqual(original, changed_bytes)
            image.rename(root / "renamed.png")
            changed_name = input_identity(root)["combined_sha256"]
            self.assertNotEqual(changed_bytes, changed_name)
            (root / "second.jpeg").write_bytes(b"second")
            changed_membership = input_identity(root)["combined_sha256"]
            self.assertNotEqual(changed_name, changed_membership)

    def test_unsupported_files_do_not_change_directory_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.png").write_bytes(b"a")
            original = input_identity(root)
            (root / "notes.txt").write_text("ignored", encoding="utf-8")
            (root / "nested").mkdir()
            (root / "nested" / "b.png").write_bytes(b"not loaded by Pi3")
            self.assertEqual(input_identity(root), original)


if __name__ == "__main__":
    unittest.main()
