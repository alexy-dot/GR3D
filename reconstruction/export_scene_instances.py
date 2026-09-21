#!/usr/bin/env python3
"""Export deterministic 3D-only scene IDs from semantic components."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Iterable


LAYOUT_CLASSES = {
    "wall", "building", "sky", "floor", "ceiling", "road", "sidewalk",
    "grass", "earth", "mountain", "water", "sea", "field", "rock",
    "sand", "path", "stairs", "runway", "river", "bridge", "fence",
    "railing", "land", "tree", "door", "windowpane",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(start: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=start, check=True,
            capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def deterministic_sort_key(item: dict[str, object], quantization: float) -> tuple[object, ...]:
    center = item["center_xyz"]
    quantized = tuple(round(float(value) / quantization) for value in center)
    return (
        str(item.get("semantic_name") or "").casefold(),
        int(item["semantic_label"]),
        *quantized,
        int(item["component_id"]),
    )


def assign_scene_ids(
    objects: Iterable[dict[str, object]],
    scene_id: str,
    quantization: float,
    layer: str = "objects",
) -> list[dict[str, object]]:
    if quantization <= 0:
        raise ValueError("quantization must be positive")
    candidates = list(objects)
    if layer == "objects":
        candidates = [
            item for item in candidates
            if str(item.get("semantic_name") or "") not in LAYOUT_CLASSES
        ]
    elif layer != "all":
        raise ValueError("layer must be 'objects' or 'all'")

    ordered = sorted(candidates, key=lambda item: deterministic_sort_key(item, quantization))
    instances = []
    for index, item in enumerate(ordered, start=1):
        instances.append(
            {
                "scene_instance_id": f"S{index:03d}",
                "scene_id": scene_id,
                "source_component_id": int(item["component_id"]),
                "semantic_label": int(item["semantic_label"]),
                "semantic_name": item.get("semantic_name"),
                "center_xyz": [float(value) for value in item["center_xyz"]],
                "bbox_min_xyz": [float(value) for value in item["bbox_min_xyz"]],
                "bbox_max_xyz": [float(value) for value in item["bbox_max_xyz"]],
                "voxel_count": int(item["voxel_count"]),
                "point_count": int(item["point_count"]),
                "mean_voxel_purity": float(item["mean_voxel_purity"]),
                "max_view_support": int(item["max_view_support"]),
                "multiview_voxel_ratio": float(item["multiview_voxel_ratio"]),
                "motion_state": "static",
                "motion_evidence": "not_measured_phase_a_static_candidate",
            }
        )
    validate_instance_table(instances)
    return instances


def validate_instance_table(instances: Iterable[dict[str, object]]) -> None:
    rows = list(instances)
    ids = [str(row["scene_instance_id"]) for row in rows]
    components = [int(row["source_component_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate scene_instance_id in scene table")
    if len(components) != len(set(components)):
        raise ValueError("a source component maps to more than one scene instance")
    expected = [f"S{index:03d}" for index in range(1, len(rows) + 1)]
    if ids != expected:
        raise ValueError("scene IDs must be contiguous and table-ordered")


def validate_rendered_ids(
    instances: Iterable[dict[str, object]], rendered_ids: Iterable[str]
) -> None:
    table_ids = {str(row["scene_instance_id"]) for row in instances}
    rendered = list(rendered_ids)
    if len(rendered) != len(set(rendered)):
        raise ValueError("render manifest contains duplicate IDs")
    rendered_set = set(rendered)
    if table_ids != rendered_set:
        missing = sorted(table_ids - rendered_set)
        extra = sorted(rendered_set - table_ids)
        raise ValueError(f"render/table ID mismatch; missing={missing}, extra={extra}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--layer", choices=("objects", "all"), default="objects")
    parser.add_argument("--center-quantization", type=float, default=1e-6)
    args = parser.parse_args()

    source = args.objects.expanduser().resolve()
    output = args.output.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    instances = assign_scene_ids(
        payload["objects"], args.scene_id, args.center_quantization, args.layer
    )
    result = {
        "manifest": {
            "status": "complete",
            "representation": "3d_only_static_component_candidates",
            "scene_id": args.scene_id,
            "namespace": "S###",
            "layer": args.layer,
            "candidate_count": len(instances),
            "source_objects_sha256": sha256(source),
            "source_object_count": len(payload["objects"]),
            "sort_key": "semantic_name_casefold, semantic_label, quantized_center_xyz, component_id",
            "center_quantization": args.center_quantization,
            "code_revision": git_revision(Path(__file__).resolve().parent),
            "exporter_sha256": sha256(Path(__file__).resolve()),
            "code_revision_note": (
                "The exporter file SHA-256 is authoritative if this script was copied "
                "into a worktree at a different Git revision."
            ),
            "warning": (
                "S### records are deterministic semantic-component candidates within one "
                "bounded run, not verified physical instances or cross-window identities. "
                "Static motion state is a Phase-A assumption without tracking evidence."
            ),
        },
        "scene_instances": instances,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Exported {len(instances)} deterministic scene candidates to {output}")


if __name__ == "__main__":
    main()
