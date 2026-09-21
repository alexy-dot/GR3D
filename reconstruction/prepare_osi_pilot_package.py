#!/usr/bin/env python3
"""Assemble a reproducible OSI pilot package for no-ID 3D-view evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


ID_PATTERN = re.compile(r"\s*\(id:\s*\d+\)", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--qa-json", required=True, type=Path)
    parser.add_argument("--layout-views", required=True, type=Path)
    parser.add_argument("--ade-object-views", required=True, type=Path)
    parser.add_argument("--merged-object-views", required=True, type=Path)
    parser.add_argument("--scene-id-views", type=Path)
    parser.add_argument("--scene-instances", type=Path)
    parser.add_argument("--representative-crops", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scene-id", required=True)
    return parser.parse_args()


def strip_object_ids(text: str) -> str:
    """Remove benchmark object-ID suffixes without otherwise rewriting the question."""
    return ID_PATTERN.sub("", text)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path, manifest_path: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "path": manifest_path.as_posix(),
        "size_bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def main() -> None:
    args = parse_args()
    run = args.run_directory.expanduser().resolve()
    qa_path = args.qa_json.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    qa_rows = json.loads(qa_path.read_text(encoding="utf-8"))
    if not isinstance(qa_rows, list) or not qa_rows:
        raise ValueError("--qa-json must contain a non-empty JSON list")

    files: list[dict[str, object]] = []
    frame_records = [
        json.loads(line)
        for line in (run / "frames.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    frames = []
    for index, record in enumerate(frame_records):
        relative = Path("inputs") / "frames" / f"{index:06d}.png"
        files.append(
            copy_file(run / "input_frames" / f"{index:06d}.png", output / relative, relative)
        )
        frames.append(
            {
                "sample_index": index,
                "source_frame_index": record["source_index"],
                "timestamp_seconds": record["timestamp_seconds"],
                "path": relative.as_posix(),
            }
        )

    view_sets = {
        "rgb_point_cloud": run / "canonical",
        "layout_semantic": args.layout_views.expanduser().resolve(),
        "objects_ade20k": args.ade_object_views.expanduser().resolve(),
        "objects_two_wheeler_merged": args.merged_object_views.expanduser().resolve(),
    }
    views: dict[str, list[str]] = {}
    for set_name, source_dir in view_sets.items():
        source_names = (
            [f"view_{axis}.png" for axis in ("xy", "xz", "yz")]
            if set_name == "rgb_point_cloud"
            else [f"object_blocks_{axis}.png" for axis in ("xy", "xz", "yz")]
        )
        views[set_name] = []
        for source_name, axis in zip(source_names, ("xy", "xz", "yz"), strict=True):
            relative = Path("inputs") / "views" / set_name / f"{axis}.png"
            files.append(copy_file(source_dir / source_name, output / relative, relative))
            views[set_name].append(relative.as_posix())

    phase_b_values = (
        args.scene_id_views, args.scene_instances, args.representative_crops
    )
    if any(phase_b_values) and not all(phase_b_values):
        raise ValueError(
            "--scene-id-views, --scene-instances, and --representative-crops "
            "must be provided together"
        )
    phase_b = all(phase_b_values)
    scene_instances_relative = None
    crop_catalog_relative = None
    scene_render_manifest_relative = None
    crop_paths: list[str] = []
    if phase_b:
        scene_view_dir = args.scene_id_views.expanduser().resolve()
        views["objects_3d_only_ids"] = []
        for axis in ("xy", "xz", "yz"):
            relative = Path("inputs") / "views" / "objects_3d_only_ids" / f"{axis}.png"
            files.append(
                copy_file(scene_view_dir / f"object_blocks_{axis}.png", output / relative, relative)
            )
            views["objects_3d_only_ids"].append(relative.as_posix())
        scene_render_manifest_relative = (
            Path("inputs") / "views" / "objects_3d_only_ids" / "render_manifest.json"
        )
        files.append(
            copy_file(
                scene_view_dir / "render_manifest.json",
                output / scene_render_manifest_relative,
                scene_render_manifest_relative,
            )
        )

        scene_instances_relative = Path("inputs") / "scene" / "scene_instances.json"
        files.append(
            copy_file(
                args.scene_instances.expanduser().resolve(),
                output / scene_instances_relative,
                scene_instances_relative,
            )
        )
        crop_source = args.representative_crops.expanduser().resolve()
        crop_payload = json.loads(
            (crop_source / "representative_crops.json").read_text(encoding="utf-8")
        )
        crop_catalog_relative = (
            Path("inputs") / "representative_crops" / "representative_crops.json"
        )
        files.append(
            copy_file(
                crop_source / "representative_crops.json",
                output / crop_catalog_relative,
                crop_catalog_relative,
            )
        )
        for crop in crop_payload["crops"]:
            source_relative = Path(crop["crop_path"])
            relative = Path("inputs") / "representative_crops" / source_relative
            files.append(copy_file(crop_source / source_relative, output / relative, relative))
            crop_paths.append(relative.as_posix())

    tagged_questions = []
    no_id_questions = []
    ground_truth = []
    for row in qa_rows:
        question_id = int(row["index"])
        options = row.get("options")
        tagged_questions.append(
            {
                "question_id": question_id,
                "category": row["category"],
                "question": row["question"],
                "question_type": row["question_type"],
                "options": options,
            }
        )
        no_id_questions.append(
            {
                "question_id": question_id,
                "category": row["category"],
                "question": strip_object_ids(row["question"]),
                "question_type": row["question_type"],
                "options": [strip_object_ids(option) for option in options] if options else None,
            }
        )
        ground_truth.append({"question_id": question_id, "answer": row["answer"]})

    conditions = {
        "raw_frames_only": {
            "frames": [item["path"] for item in frames],
            "views": [],
            "questions": "questions_no_ids.json",
        },
        "raw_plus_rgb_canonical_views": {
            "frames": [item["path"] for item in frames],
            "views": views["rgb_point_cloud"],
            "questions": "questions_no_ids.json",
        },
        "raw_plus_no_id_semantic_views": {
            "frames": [item["path"] for item in frames],
            "views": views["layout_semantic"] + views["objects_ade20k"],
            "questions": "questions_no_ids.json",
        },
    }
    if phase_b:
        conditions.update(
            {
                "raw_plus_3d_only_ids": {
                    "frames": [item["path"] for item in frames],
                    "views": views["layout_semantic"] + views["objects_3d_only_ids"],
                    "scene_instances": scene_instances_relative.as_posix(),
                    "view_manifest": scene_render_manifest_relative.as_posix(),
                    "questions": "questions_no_ids.json",
                },
                "raw_plus_3d_ids_and_representative_crops": {
                    "frames": [item["path"] for item in frames],
                    "views": views["layout_semantic"] + views["objects_3d_only_ids"],
                    "scene_instances": scene_instances_relative.as_posix(),
                    "view_manifest": scene_render_manifest_relative.as_posix(),
                    "representative_crop_catalog": crop_catalog_relative.as_posix(),
                    "representative_crops": crop_paths,
                    "questions": "questions_no_ids.json",
                },
            }
        )
    else:
        conditions["raw_plus_merged_views"] = {
            "frames": [item["path"] for item in frames],
            "views": views["layout_semantic"] + views["objects_two_wheeler_merged"],
            "questions": "questions_no_ids.json",
        }
    conditions["tagged_question_control"] = {
        "frames": [item["path"] for item in frames],
        "views": views["layout_semantic"] + views["objects_ade20k"],
        "questions": "questions_tagged.json",
    }

    json_outputs = {
        "questions_tagged.json": tagged_questions,
        "questions_no_ids.json": no_id_questions,
        "ground_truth.json": ground_truth,
        "conditions.json": conditions,
    }
    for name, data in json_outputs.items():
        path = output / name
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        files.append(
            {"path": name, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
        )

    run_manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "status": "complete",
        "scene_id": args.scene_id,
        "purpose": "first_version_no_id_3d_view_evaluation_package",
        "source_video_sha256": sha256(Path(run_manifest["input"])),
        "reconstruction_manifest": run_manifest,
        "frames": frames,
        "views": views,
        "question_count": len(qa_rows),
        "conditions": list(json_outputs["conditions.json"]),
        "files": sorted(files, key=lambda item: item["path"]),
        "important_limitations": [
            "Original Pi3 geometry is scale-invariant.",
            "The 28.1 m OSI answer is a calibration anchor, not independent validation.",
            "No-ID question rewriting removes textual ID suffixes only.",
            "The source benchmark frames may still contain baked-in visual number tags.",
            "Representative crops inherit any visual number tags baked into source pixels.",
            "Semantic components are not verified physical instances.",
            "This package prepares inputs; it does not contain MLLM predictions.",
        ],
    }
    manifest_path = output / "package_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(json_outputs['conditions.json'])} conditions at {output}")


if __name__ == "__main__":
    main()
