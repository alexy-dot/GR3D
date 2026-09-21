#!/usr/bin/env python3
from __future__ import annotations

import unittest

from reconstruction.score_mllm_evaluation import normalize_mcq, parse_number, score_rows


class MllmEvaluationTests(unittest.TestCase):
    def test_parsers(self) -> None:
        self.assertEqual(normalize_mcq("Answer: b"), "B")
        self.assertEqual(parse_number("about 12.49 meters"), 12.49)

    def test_scoring_requires_complete_unique_predictions(self) -> None:
        requests = [{"request_id": "s::c::q000", "condition": "c", "question_id": 0, "category": "x", "question_type": "mcq"}]
        truth = [{"question_id": 0, "answer": "B"}]
        result = score_rows(requests, [{"request_id": "s::c::q000", "answer": "B"}], truth)
        self.assertEqual(result["summary"]["c"]["exact_accuracy"], 1.0)
        with self.assertRaisesRegex(ValueError, "missing"):
            score_rows(requests, [], truth)

    def test_numeric_error_is_reported(self) -> None:
        requests = [{"request_id": "s::c::q001", "condition": "c", "question_id": 1, "category": "distance", "question_type": "numerical"}]
        truth = [{"question_id": 1, "answer": "12.49"}]
        result = score_rows(requests, [{"request_id": "s::c::q001", "answer": "12.0 m"}], truth)
        self.assertAlmostEqual(result["rows"][0]["absolute_error"], 0.49)


if __name__ == "__main__":
    unittest.main()
