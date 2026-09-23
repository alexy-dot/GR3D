#!/usr/bin/env python3
"""Score saved MLLM predictions without exposing answers during inference."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def normalize_mcq(value: object) -> str:
    match = re.search(r"\b([A-D])\b", str(value).upper())
    return match.group(1) if match else ""


def parse_number(value: object) -> float | None:
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", str(value).replace(",", ""))
    return float(match.group(0)) if match else None


def score_rows(requests: list[dict], predictions: list[dict], truth: list[dict]) -> dict:
    request_by_id = {row["request_id"]: row for row in requests}
    truth_by_id = {int(row["question_id"]): row["answer"] for row in truth}
    seen = set()
    rows = []
    for prediction in predictions:
        request_id = prediction["request_id"]
        if request_id in seen or request_id not in request_by_id:
            raise ValueError(f"duplicate or unknown request_id: {request_id}")
        seen.add(request_id)
        request = request_by_id[request_id]
        expected = truth_by_id[int(request["question_id"])]
        if request["question_type"] == "mcq":
            parsed = normalize_mcq(prediction.get("answer", ""))
            correct = parsed == normalize_mcq(expected)
            absolute_error = None
        else:
            parsed = parse_number(prediction.get("answer", ""))
            target = parse_number(expected)
            correct = parsed is not None and target is not None and parsed == target
            absolute_error = abs(parsed - target) if parsed is not None and target is not None else None
        rows.append({
            "request_id": request_id,
            "condition": request["condition"],
            "question_id": request["question_id"],
            "category": request["category"],
            "question_type": request["question_type"],
            "raw_answer": prediction.get("answer"),
            "parsed_answer": parsed,
            "ground_truth": expected,
            "exact_correct": correct,
            "absolute_error": absolute_error,
        })
    if seen != set(request_by_id):
        raise ValueError(f"missing {len(set(request_by_id) - seen)} predictions")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    summary = {}
    for condition, items in grouped.items():
        numerical_errors = [row["absolute_error"] for row in items if row["absolute_error"] is not None]
        summary[condition] = {
            "count": len(items),
            "exact_accuracy": sum(row["exact_correct"] for row in items) / len(items),
            "numerical_mae": sum(numerical_errors) / len(numerical_errors) if numerical_errors else None,
            "category_exact_accuracy": {
                category: sum(row["exact_correct"] for row in items if row["category"] == category)
                / sum(1 for row in items if row["category"] == category)
                for category in sorted({row["category"] for row in items})
            },
        }
    return {"status": "complete", "metric_note": "Exact accuracy and numerical MAE; not claimed as official OSI scoring.", "summary": summary, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requests", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--condition",
        action="append",
        help="Score only this condition; repeat to score a declared subset.",
    )
    args = parser.parse_args()
    request_payload = json.loads(args.requests.read_text(encoding="utf-8"))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    requests = request_payload["requests"]
    if args.condition:
        selected = set(args.condition)
        requests = [row for row in requests if row["condition"] in selected]
        missing = selected - {row["condition"] for row in requests}
        if missing:
            raise ValueError(f"unknown conditions: {sorted(missing)}")
    result = score_rows(requests, predictions, truth)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Scored {len(result['rows'])} predictions at {args.output}")


if __name__ == "__main__":
    main()
