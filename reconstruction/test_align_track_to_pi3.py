#!/usr/bin/env python3

from __future__ import annotations

import unittest

from reconstruction.align_track_to_pi3 import match_frames


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


if __name__ == "__main__":
    unittest.main()
