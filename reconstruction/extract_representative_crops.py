#!/usr/bin/env python3
"""Extract at most one best-view source-frame crop for each 3D scene candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from reconstruction.input_identity import input_identity, legacy_video_sha256
except ModuleNotFoundError:
    from input_identity import input_identity, legacy_video_sha256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def padded_bbox(
    pixels_yx: np.ndarray, width: int, height: int, padding_fraction: float
) -> tuple[int, int, int, int]:
    if not len(pixels_yx):
        raise ValueError("cannot make a crop from zero pixels")
    y_min, x_min = pixels_yx.min(axis=0)
    y_max, x_max = pixels_yx.max(axis=0)
    pad_x = max(4, round((int(x_max) - int(x_min) + 1) * padding_fraction))
    pad_y = max(4, round((int(y_max) - int(y_min) + 1) * padding_fraction))
    left = max(0, int(x_min) - pad_x)
    upper = max(0, int(y_min) - pad_y)
    right = min(width, int(x_max) + pad_x + 1)
    lower = min(height, int(y_max) + pad_y + 1)
    return left, upper, right, lower


def crop_quality_warnings(
    box: tuple[int, int, int, int], supporting_pixels: int
) -> list[str]:
    left, upper, right, lower = box
    warnings = []
    if right - left < 32 or lower - upper < 32:
        warnings.append("small_crop_dimension")
    if supporting_pixels < 50:
        warnings.append("low_component_pixel_support")
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--voxels", required=True, type=Path)
    parser.add_argument("--objects", required=True, type=Path)
    parser.add_argument("--scene-instances", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--padding-fraction", type=float, default=0.2)
    args = parser.parse_args()
    if not 0 <= args.padding_fraction <= 2:
        raise ValueError("--padding-fraction must be in [0, 2]")

    run = args.run_directory.expanduser().resolve()
    voxels_path = args.voxels.expanduser().resolve()
    objects_path = args.objects.expanduser().resolve()
    instances_path = args.scene_instances.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    run_manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    source_input_identity = input_identity(Path(run_manifest["input"]))
    frame_records = [
        json.loads(line)
        for line in (run / "frames.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    observations = np.load(run / "point_observations.npz")
    points = observations["points"].astype(np.float64)
    frame_indices = observations["frame_index"].astype(np.int64)
    pixels = observations["pixel_yx"].astype(np.int64)
    voxels = np.load(voxels_path)
    voxel_coordinates = voxels["coordinates"].astype(np.int64)
    voxel_components = voxels["component_id"].astype(np.int64)
    objects = json.loads(objects_path.read_text(encoding="utf-8"))
    voxel_size = float(objects["manifest"]["voxel_size"])
    instances_payload = json.loads(instances_path.read_text(encoding="utf-8"))
    instances = instances_payload["scene_instances"]

    coordinate_to_component = {
        tuple(coordinate.tolist()): int(component)
        for coordinate, component in zip(voxel_coordinates, voxel_components, strict=True)
    }
    observation_components = np.asarray(
        [
            coordinate_to_component.get(tuple(coordinate.tolist()), -1)
            for coordinate in np.floor(points / voxel_size).astype(np.int64)
        ],
        dtype=np.int64,
    )

    crops = []
    for instance in instances:
        component = int(instance["source_component_id"])
        selected = observation_components == component
        if not np.any(selected):
            raise ValueError(f"no point observations found for {instance['scene_instance_id']}")
        candidate_frames, counts = np.unique(frame_indices[selected], return_counts=True)
        best_frame = int(candidate_frames[np.argmax(counts)])
        best_pixels = pixels[selected & (frame_indices == best_frame)]
        exact_frame = run / "input_frames" / f"{best_frame:06d}.png"
        if exact_frame.is_file():
            frame_path = exact_frame
            frame_reference = f"input_frames/{best_frame:06d}.png"
            coordinate_space = "saved_exact_resized_input_frame"
        else:
            source_root = Path(run_manifest["input"])
            if not source_root.is_dir():
                raise FileNotFoundError(
                    f"missing exact frame {exact_frame} and input is not an image directory"
                )
            frame_path = source_root / frame_records[best_frame]["source_name"]
            frame_reference = frame_records[best_frame]["source_name"]
            coordinate_space = "source_image_resized_in_memory_to_run_dimensions"
        with Image.open(frame_path) as image:
            image = image.convert("RGB")
            run_size = (
                int(run_manifest["resized_width"]), int(run_manifest["resized_height"])
            )
            if image.size != run_size:
                image = image.resize(run_size, Image.Resampling.BICUBIC)
            box = padded_bbox(
                best_pixels, image.width, image.height, args.padding_fraction
            )
            relative = Path("crops") / f"{instance['scene_instance_id']}.png"
            crop_path = output / relative
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            image.crop(box).save(crop_path)
        warnings = crop_quality_warnings(box, len(best_pixels))
        crops.append(
            {
                "scene_instance_id": instance["scene_instance_id"],
                "semantic_name": instance.get("semantic_name"),
                "semantic_label": int(instance["semantic_label"]),
                "source_component_id": component,
                "sample_index": best_frame,
                "source_frame_index": int(frame_records[best_frame]["source_index"]),
                "timestamp_seconds": frame_records[best_frame].get("timestamp_seconds"),
                "source_frame_reference": frame_reference,
                "source_frame_sha256": sha256(frame_path),
                "crop_coordinate_space": coordinate_space,
                "supporting_pixels": int(len(best_pixels)),
                "crop_box_ltrb": list(box),
                "crop_path": relative.as_posix(),
                "crop_sha256": sha256(crop_path),
                "crop_width": box[2] - box[0],
                "crop_height": box[3] - box[1],
                "quality_warnings": warnings,
            }
        )

    result = {
        "manifest": {
            "status": "complete",
            "representation": "one_representative_source_crop_per_3d_candidate",
            "selection": "frame with maximum retained point observations for component",
            "padding_fraction": args.padding_fraction,
            "scene_instances_sha256": sha256(instances_path),
            "scene_id": instances_payload["manifest"]["scene_id"],
            "point_observations_sha256": sha256(run / "point_observations.npz"),
            "source_run_manifest_sha256": sha256(run / "manifest.json"),
            "source_input_identity": source_input_identity,
            "source_video_sha256": legacy_video_sha256(source_input_identity),
            "source_frames_modified": False,
            "crop_count": len(crops),
            "crops_with_warnings": sum(bool(item["quality_warnings"]) for item in crops),
            "known_confound": (
                "Representative crops preserve source pixels; any source-side overlays "
                "or benchmark tags remain present and must be reported."
            ),
        },
        "crops": crops,
    }
    catalog = output / "representative_crops.json"
    catalog.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Extracted {len(crops)} representative crops to {output}")


if __name__ == "__main__":
    main()
