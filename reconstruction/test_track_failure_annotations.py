#!/usr/bin/env python3

from __future__ import annotations

import unittest

from reconstruction.score_track_failure_annotations import score_annotations


def track_with_ranges(ranges: list[dict], frame_count: int = 3) -> dict:
    return {
        "track_candidate_id": "T001",
        "frame_count": frame_count,
        "frame_ranges": ranges,
        "id_switch_events": [],
        "fragmentation_episodes": [],
        "occlusion_episodes": [],
    }


class TrackFailureAnnotationTests(unittest.TestCase):
    def test_scores_assessable_identity_and_non_usable_masks(self) -> None:
        payload = {
            "tracks": [
                track_with_ranges(
                    [
                        {
                            "start_sample_index": 0,
                            "end_sample_index": 1,
                            "association": "correct_target",
                            "target_visibility": "visible",
                            "mask_quality": "usable",
                        },
                        {
                            "start_sample_index": 2,
                            "end_sample_index": 2,
                            "association": "not_assessable",
                            "target_visibility": "fully_occluded",
                            "mask_quality": "empty",
                        },
                    ]
                )
            ]
        }
        result = score_annotations(payload)
        self.assertEqual(result["reviewed_frame_count"], 3)
        self.assertEqual(result["identity_assessable_frame_count"], 2)
        self.assertEqual(
            result["observed_correct_association_fraction_on_assessable_frames"], 1.0
        )
        self.assertEqual(result["non_usable_mask_frame_count"], 1)

    def test_rejects_overlapping_ranges(self) -> None:
        base = {
            "association": "correct_target",
            "target_visibility": "visible",
            "mask_quality": "usable",
        }
        track = track_with_ranges(
            [
                {**base, "start_sample_index": 0, "end_sample_index": 1},
                {**base, "start_sample_index": 1, "end_sample_index": 2},
            ]
        )
        with self.assertRaisesRegex(ValueError, "overlapping annotation"):
            score_annotations({"tracks": [track]})

    def test_rejects_incomplete_coverage(self) -> None:
        track = track_with_ranges(
            [
                {
                    "start_sample_index": 0,
                    "end_sample_index": 1,
                    "association": "correct_target",
                    "target_visibility": "visible",
                    "mask_quality": "usable",
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "does not cover samples"):
            score_annotations({"tracks": [track]})


if __name__ == "__main__":
    unittest.main()
