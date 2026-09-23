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

PROTOCOL_VERSION = 2
SCENE_CONTEXT_FIELDS = (
    "scene_instance_id",
    "scene_id",
    "semantic_label",
    "semantic_name",
    "center_xyz",
    "bbox_min_xyz",
    "bbox_max_xyz",
    "voxel_count",
    "point_count",
    "mean_voxel_purity",
    "max_view_support",
    "multiview_voxel_ratio",
    "motion_state",
    "motion_evidence",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_scene_context(package: Path, condition: dict, scene_id: str) -> dict | None:
    relative = condition.get("scene_instances")
    if not relative:
        return None
    payload = json.loads((package / relative).read_text(encoding="utf-8"))
    instances = []
    seen = set()
    for source in payload["scene_instances"]:
        row = {field: source[field] for field in SCENE_CONTEXT_FIELDS}
        instance_id = str(row["scene_instance_id"])
        if instance_id in seen:
            raise ValueError(f"duplicate scene instance ID: {instance_id}")
        if str(row["scene_id"]) != str(scene_id):
            raise ValueError(f"scene instance {instance_id} belongs to another scene")
        seen.add(instance_id)
        instances.append(row)
    return {
        "scene_id": str(scene_id),
        "representation": "answer_blind_3d_scene_table",
        "scene_instances_sha256": sha256(package / relative),
        "instances": instances,
    }


def load_crop_associations(
    package: Path, condition: dict, scene_instance_ids: set[str]
) -> dict[str, dict]:
    relative = condition.get("representative_crop_catalog")
    declared = condition.get("representative_crops", [])
    if not relative:
        if declared:
            raise ValueError("representative crops require a crop catalog")
        return {}
    payload = json.loads((package / relative).read_text(encoding="utf-8"))
    catalog_root = Path(relative).parent
    associations = {}
    for item in payload["crops"]:
        path = (catalog_root / item["crop_path"]).as_posix()
        instance_id = str(item["scene_instance_id"])
        if path in associations:
            raise ValueError(f"duplicate representative crop path: {path}")
        if instance_id not in scene_instance_ids:
            raise ValueError(f"representative crop references unknown scene ID: {instance_id}")
        associations[path] = {
            "scene_instance_id": instance_id,
            "catalog_sha256": sha256(package / relative),
        }
    if set(declared) != set(associations):
        raise ValueError("declared representative crops differ from the crop catalog")
    return associations


def image_records(
    package: Path,
    condition: dict,
    frame_meta: list[dict],
    crop_associations: dict[str, dict] | None = None,
) -> list[dict]:
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
    crop_associations = crop_associations or {}
    for relative in condition.get("representative_crops", []):
        association = crop_associations.get(relative)
        if association is None:
            raise ValueError(f"representative crop has no catalog association: {relative}")
        records.append({
            "role": "representative_crop",
            "scene_instance_id": association["scene_instance_id"],
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
        scene_context = load_scene_context(
            package, condition, str(manifest["scene_id"])
        )
        crop_associations = load_crop_associations(
            package,
            condition,
            {
                str(row["scene_instance_id"])
                for row in (scene_context or {}).get("instances", [])
            },
        )
        images = image_records(
            package, condition, manifest["frames"], crop_associations
        )
        for question in condition_questions:
            request = {
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
            }
            if scene_context is not None:
                request["scene_context"] = scene_context
            requests.append(request)
    return {
        "protocol_version": PROTOCOL_VERSION,
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
