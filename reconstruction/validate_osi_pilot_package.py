#!/usr/bin/env python3
"""Validate an assembled OSI pilot package before evaluation or handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from reconstruction.input_identity import comparable_identity
except ModuleNotFoundError:
    from input_identity import comparable_identity


COMMON_RENDER_KEYS = (
    "color_by",
    "style",
    "layer",
    "layout_class_policy",
    "retained_semantic_labels",
    "min_component_points",
    "marker_size",
    "voxel_count",
    "component_count",
    "rendered_component_ids",
    "semantic_class_count",
    "voxels_sha256",
    "camera_poses_sha256",
    "objects_json_sha256",
    "alignment",
    "clip_percentile",
    "bounds_xyz",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_render_pair(
    no_id_render: dict[str, object], id_render: dict[str, object]
) -> None:
    missing = [
        key for key in COMMON_RENDER_KEYS
        if key not in no_id_render or key not in id_render
    ]
    if missing:
        raise ValueError(f"render manifests lack controlled-comparison fields: {missing}")
    mismatched = [
        key for key in COMMON_RENDER_KEYS
        if no_id_render[key] != id_render[key]
    ]
    if mismatched:
        raise ValueError(f"no-ID and 3D-ID render bases differ: {mismatched}")
    if no_id_render.get("representation") != "no_id_semantic_voxel_blocks":
        raise ValueError("no-ID render manifest has the wrong representation")
    if id_render.get("representation") != "3d_only_scene_id_component_candidates":
        raise ValueError("3D-ID render manifest has the wrong representation")
    if no_id_render.get("rendered_ids"):
        raise ValueError("no-ID render manifest unexpectedly contains rendered IDs")


def validate_declared_file_set(declared: set[str], actual: set[str]) -> None:
    """Require the package to contain exactly its declared closed file set."""
    if actual != declared:
        raise ValueError(
            "package contains undeclared or missing files; "
            f"extra={sorted(actual - declared)}, missing={sorted(declared - actual)}"
        )


def validate_scene_relationships(
    package_manifest: dict[str, object],
    scene_payload: dict[str, object],
    no_id_render: dict[str, object],
    id_render: dict[str, object],
    crop_payload: dict[str, object],
    frames: list[dict[str, object]],
    packaged_scene_sha256: str,
) -> None:
    validate_render_pair(no_id_render, id_render)
    scene_manifest = scene_payload["manifest"]
    crop_manifest = crop_payload["manifest"]
    if package_manifest["scene_id"] != scene_manifest["scene_id"]:
        raise ValueError("package and scene table use different scene IDs")
    if crop_manifest.get("scene_id") != scene_manifest["scene_id"]:
        raise ValueError("crop catalog and scene table use different scene IDs")
    if id_render.get("scene_instances_sha256") != packaged_scene_sha256:
        raise ValueError("3D-ID render manifest references a different scene table")
    if crop_manifest.get("scene_instances_sha256") != packaged_scene_sha256:
        raise ValueError("crop catalog references a different scene table")
    if no_id_render["objects_json_sha256"] != scene_manifest.get("source_objects_sha256"):
        raise ValueError("render and scene table use different semantic objects")
    if scene_manifest.get("source_observations_sha256") != crop_manifest.get(
        "point_observations_sha256"
    ):
        raise ValueError("scene table and crops use different point observations")
    if package_manifest.get("source_run_manifest_sha256") != scene_manifest.get(
        "source_run_manifest_sha256"
    ) or package_manifest.get("source_run_manifest_sha256") != crop_manifest.get(
        "source_run_manifest_sha256"
    ):
        raise ValueError("package, scene table, and crops use different reconstruction runs")
    identities = [
        package_manifest.get("source_input_identity"),
        scene_manifest.get("source_input_identity"),
        crop_manifest.get("source_input_identity"),
    ]
    if any(identity is not None for identity in identities):
        if not all(identity is not None for identity in identities):
            raise ValueError(
                "cannot compare new input identity with a legacy file-hash field"
            )
        comparable = [comparable_identity(identity) for identity in identities]
        if comparable[0] != comparable[1] or comparable[0] != comparable[2]:
            raise ValueError("package, scene table, and crops use different source inputs")
    elif package_manifest.get("source_video_sha256") != scene_manifest.get(
        "source_video_sha256"
    ) or package_manifest.get("source_video_sha256") != crop_manifest.get(
        "source_video_sha256"
    ):
        raise ValueError("package, scene table, and crops use different source videos")

    table_by_id = {
        str(item["scene_instance_id"]): item
        for item in scene_payload["scene_instances"]
    }
    for item in table_by_id.values():
        if item.get("motion_state") == "static" and item.get("motion_evidence") == (
            "not_measured_phase_a_static_candidate"
        ):
            raise ValueError("unmeasured scene candidate is falsely marked static")
    crop_ids = [str(item["scene_instance_id"]) for item in crop_payload["crops"]]
    if len(crop_ids) != len(set(crop_ids)):
        raise ValueError("representative crop catalog contains duplicate scene IDs")
    if set(table_by_id) != set(crop_ids):
        raise ValueError("scene-instance table and representative crop IDs disagree")
    for crop in crop_payload["crops"]:
        table = table_by_id[str(crop["scene_instance_id"])]
        for key in ("source_component_id", "semantic_label", "semantic_name"):
            if crop.get(key) != table.get(key):
                raise ValueError(f"crop/table identity mismatch for {crop['scene_instance_id']}: {key}")
        sample_index = int(crop["sample_index"])
        if sample_index < 0 or sample_index >= len(frames):
            raise ValueError("crop sample index is outside the packaged frame list")
        frame = frames[sample_index]
        if int(crop["source_frame_index"]) != int(frame["source_frame_index"]):
            raise ValueError("crop original source-frame index does not match package")
        if crop.get("timestamp_seconds") != frame.get("timestamp_seconds"):
            raise ValueError("crop timestamp does not match package frame metadata")


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
        no_id_manifest_path = no_id.get("object_view_manifest")
        if not no_id_manifest_path:
            raise ValueError("no-ID condition lacks its object render manifest")
        no_id_render_payload = json.loads(
            (root / no_id_manifest_path).read_text(encoding="utf-8")
        )
        catalog_path = root / crops["representative_crop_catalog"]
        crop_payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        table_ids = {
            str(item["scene_instance_id"])
            for item in scene_payload["scene_instances"]
        }
        scene_table_hash = sha256(root / ids["scene_instances"])
        validate_scene_relationships(
            manifest,
            scene_payload,
            no_id_render_payload,
            render_payload,
            crop_payload,
            manifest["frames"],
            scene_table_hash,
        )
        if set(render_payload.get("rendered_ids", [])) != table_ids:
            raise ValueError("3D-ID render manifest and scene table disagree")
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

    declared = {str(Path(record["path"]).as_posix()) for record in manifest.get("files", [])}
    declared.add("package_manifest.json")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    validate_declared_file_set(declared, actual)

    print(
        f"Valid package: {len(manifest['files'])} hashed files, "
        f"{len(questions)} questions, {len(manifest['conditions'])} conditions"
    )


if __name__ == "__main__":
    main()
