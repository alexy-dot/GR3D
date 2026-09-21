#!/usr/bin/env python3
"""Validate an assembled OSI pilot package before evaluation or handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    args = parser.parse_args()

    root = args.package.expanduser().resolve()
    manifest = json.loads((root / "package_manifest.json").read_text(encoding="utf-8"))
    questions = json.loads((root / "questions_no_ids.json").read_text(encoding="utf-8"))
    ground_truth = json.loads((root / "ground_truth.json").read_text(encoding="utf-8"))
    conditions = json.loads((root / "conditions.json").read_text(encoding="utf-8"))

    if manifest.get("status") != "complete":
        raise ValueError("package manifest is not complete")
    if manifest.get("question_count") != len(questions) or len(questions) != len(ground_truth):
        raise ValueError("question or ground-truth count does not match the manifest")
    if manifest.get("conditions") != list(conditions):
        raise ValueError("condition manifest and conditions.json disagree")
    required = {
        "raw_frames_only",
        "raw_plus_rgb_canonical_views",
        "raw_plus_no_id_semantic_views",
        "tagged_question_control",
    }
    if not required.issubset(conditions):
        raise ValueError(f"missing required conditions: {sorted(required - set(conditions))}")
    phase_b = {
        "raw_plus_3d_only_ids",
        "raw_plus_3d_ids_and_representative_crops",
    }
    if phase_b & set(conditions) and not phase_b.issubset(conditions):
        raise ValueError("Phase-B ID and crop conditions must be present together")
    if "(id:" in json.dumps(questions, ensure_ascii=False).lower():
        raise ValueError("no-ID questions still contain an object-ID suffix")
    if any("answer" in row for row in questions):
        raise ValueError("answers leaked into the evaluation question file")
    baseline_frames = conditions["raw_frames_only"]["frames"]
    for name, condition in conditions.items():
        if condition["frames"] != baseline_frames:
            raise ValueError(f"condition {name} does not use identical sampled frames")
        if name != "tagged_question_control" and condition["questions"] != "questions_no_ids.json":
            raise ValueError(f"condition {name} does not use the shared no-ID questions")
    if phase_b.issubset(conditions):
        no_id = conditions["raw_plus_no_id_semantic_views"]
        ids = conditions["raw_plus_3d_only_ids"]
        crops = conditions["raw_plus_3d_ids_and_representative_crops"]
        if ids["frames"] != no_id["frames"] or ids["questions"] != no_id["questions"]:
            raise ValueError("no-ID and 3D-ID conditions changed frames or questions")
        if (
            crops["views"] != ids["views"]
            or crops["scene_instances"] != ids["scene_instances"]
            or crops["view_manifest"] != ids["view_manifest"]
        ):
            raise ValueError("crop condition changed the 3D-ID representation")
        scene_payload = json.loads((root / ids["scene_instances"]).read_text(encoding="utf-8"))
        render_payload = json.loads((root / ids["view_manifest"]).read_text(encoding="utf-8"))
        catalog_path = root / crops["representative_crop_catalog"]
        crop_payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        table_ids = {
            str(item["scene_instance_id"])
            for item in scene_payload["scene_instances"]
        }
        if set(render_payload.get("rendered_ids", [])) != table_ids:
            raise ValueError("3D-ID render manifest and scene table disagree")
        if render_payload.get("scene_instances_sha256") != sha256(root / ids["scene_instances"]):
            raise ValueError("3D-ID render manifest references a different scene table")
        crop_ids = [str(item["scene_instance_id"]) for item in crop_payload["crops"]]
        if len(crop_ids) != len(set(crop_ids)):
            raise ValueError("representative crop catalog contains duplicate scene IDs")
        if table_ids != set(crop_ids):
            raise ValueError("scene-instance table and representative crop IDs disagree")
        for item in crop_payload["crops"]:
            crop_path = catalog_path.parent / item["crop_path"]
            if not crop_path.is_file() or sha256(crop_path) != item["crop_sha256"]:
                raise ValueError(f"representative crop missing or modified: {item['crop_path']}")
        declared_crop_paths = set(crops["representative_crops"])
        expected_crop_paths = {
            (Path(crops["representative_crop_catalog"]).parent / item["crop_path"]).as_posix()
            for item in crop_payload["crops"]
        }
        if declared_crop_paths != expected_crop_paths:
            raise ValueError("condition crop paths disagree with the crop catalog")

    for record in manifest.get("files", []):
        relative = Path(record["path"])
        if relative.is_absolute():
            raise ValueError(f"manifest contains an absolute path: {relative}")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != record["size_bytes"]:
            raise ValueError(f"size mismatch: {relative}")
        if sha256(path) != record["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {relative}")

    print(
        f"Valid package: {len(manifest['files'])} hashed files, "
        f"{len(questions)} questions, {len(manifest['conditions'])} conditions"
    )


if __name__ == "__main__":
    main()
