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

    json_outputs = {
        "questions_tagged.json": tagged_questions,
        "questions_no_ids.json": no_id_questions,
        "ground_truth.json": ground_truth,
        "conditions.json": {
            "raw_frames_only": {
                "frames": [item["path"] for item in frames],
                "views": [],
                "questions": "questions_no_ids.json",
            },
            "raw_plus_ade_views": {
                "frames": [item["path"] for item in frames],
                "views": views["layout_semantic"] + views["objects_ade20k"],
                "questions": "questions_no_ids.json",
            },
            "raw_plus_merged_views": {
                "frames": [item["path"] for item in frames],
                "views": views["layout_semantic"] + views["objects_two_wheeler_merged"],
                "questions": "questions_no_ids.json",
            },
            "tagged_question_control": {
                "frames": [item["path"] for item in frames],
                "views": views["layout_semantic"] + views["objects_ade20k"],
                "questions": "questions_tagged.json",
            },
        },
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
