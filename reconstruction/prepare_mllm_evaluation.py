#!/usr/bin/env python3
"""Prepare answer-blind, deterministic MLLM requests from a validated OSI package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SYSTEM_PROMPT = (
    "You are evaluating spatial reasoning from ordered video frames and optional 3D scene "
    "views. Answer only the supplied question. For multiple-choice questions return only "
    "the option letter. For numerical questions return only one number in the requested "
    "unit. Do not explain your answer."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_records(package: Path, condition: dict, frame_meta: list[dict]) -> list[dict]:
    records = []
    by_path = {item["path"]: item for item in frame_meta}
    for relative in condition["frames"]:
        meta = by_path[relative]
        records.append({
            "role": "video_frame",
            "path": relative,
            "sample_index": meta["sample_index"],
            "source_frame_index": meta["source_frame_index"],
            "timestamp_seconds": meta["timestamp_seconds"],
            "sha256": sha256(package / relative),
        })
    for relative in condition.get("views", []):
        records.append({"role": "canonical_3d_view", "path": relative, "sha256": sha256(package / relative)})
    for relative in condition.get("representative_crops", []):
        records.append({
            "role": "representative_crop",
            "scene_instance_id": Path(relative).stem,
            "path": relative,
            "sha256": sha256(package / relative),
        })
    return records


def build_requests(package: Path, conditions: dict, questions: list[dict], manifest: dict) -> dict:
    requests = []
    for condition_name, condition in conditions.items():
        condition_questions = json.loads((package / condition["questions"]).read_text(encoding="utf-8"))
        if [row["question_id"] for row in condition_questions] != [row["question_id"] for row in questions]:
            raise ValueError(f"question order differs for condition {condition_name}")
        images = image_records(package, condition, manifest["frames"])
        for question in condition_questions:
            requests.append({
                "request_id": f"{manifest['scene_id']}::{condition_name}::q{int(question['question_id']):03d}",
                "scene_id": manifest["scene_id"],
                "condition": condition_name,
                "question_id": question["question_id"],
                "category": question["category"],
                "question_type": question["question_type"],
                "system_prompt": SYSTEM_PROMPT,
                "question": question["question"],
                "options": question.get("options"),
                "images": images,
            })
    return {
        "protocol_version": 1,
        "status": "answer_blind_requests_prepared",
        "package_manifest_sha256": sha256(package / "package_manifest.json"),
        "condition_order": list(conditions),
        "question_count": len(questions),
        "request_count": len(requests),
        "decoding": {"temperature": 0, "repetitions": 1},
        "answers_included": False,
        "requests": requests,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    package = args.package.expanduser().resolve()
    manifest = json.loads((package / "package_manifest.json").read_text(encoding="utf-8"))
    conditions = json.loads((package / "conditions.json").read_text(encoding="utf-8"))
    questions = json.loads((package / "questions_no_ids.json").read_text(encoding="utf-8"))
    payload = build_requests(package, conditions, questions, manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Prepared {payload['request_count']} answer-blind requests at {args.output}")


if __name__ == "__main__":
    main()
