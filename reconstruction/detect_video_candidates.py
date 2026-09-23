#!/usr/bin/env python3
"""Detect auditable per-frame object candidates for SAM 2 initialization."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

if __package__:
    from reconstruction.input_identity import input_identity
    from reconstruction.run_video_instance_tracking import extract_frames, git_state, sha256
else:
    from input_identity import input_identity
    from run_video_instance_tracking import extract_frames, git_state, sha256


MODEL_NAME = "torchvision.fasterrcnn_resnet50_fpn_v2"
WEIGHTS_NAME = "FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1"


def filtered_candidates(
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    categories: list[str],
    score_threshold: float,
    allowed_classes: set[str] | None = None,
) -> list[dict]:
    if not (len(boxes) == len(labels) == len(scores)):
        raise ValueError("boxes, labels and scores must have equal length")
    allowed = {value.casefold() for value in allowed_classes or set()}
    rows = []
    for box, label_value, score_value in zip(boxes, labels, scores):
        label = int(label_value)
        score = float(score_value)
        if label < 0 or label >= len(categories):
            raise ValueError(f"detection label {label} is outside the category table")
        name = categories[label]
        if score < score_threshold or name == "__background__":
            continue
        if allowed and name.casefold() not in allowed:
            continue
        values = [float(value) for value in box]
        if len(values) != 4 or not all(math.isfinite(value) for value in values):
            raise ValueError("detection boxes must contain four finite values")
        x1, y1, x2, y2 = values
        if x2 <= x1 or y2 <= y1:
            continue
        rows.append(
            {
                "semantic_label": label,
                "semantic_name": name,
                "score": score,
                "box_xyxy": values,
                "area_pixels": (x2 - x1) * (y2 - y1),
            }
        )
    rows.sort(
        key=lambda row: (
            -row["score"],
            row["semantic_label"],
            *row["box_xyxy"],
        )
    )
    return rows


def select_prompt_candidate(
    candidates: list[dict], source_frame_index: int, semantic_name: str | None
) -> dict:
    matching = [
        row
        for row in candidates
        if int(row["source_frame_index"]) == source_frame_index
        and (
            semantic_name is None
            or row["semantic_name"].casefold() == semantic_name.casefold()
        )
    ]
    if not matching:
        qualifier = f" class {semantic_name!r}" if semantic_name else ""
        raise ValueError(
            f"no candidate for source frame {source_frame_index}{qualifier}"
        )
    return sorted(
        matching,
        key=lambda row: (-row["score"], -row["area_pixels"], row["detection_id"]),
    )[0]


def make_tracking_prompt(selected: dict, track_candidate_id: str, video: Path) -> dict:
    """Convert one detector result into the existing SAM 2 prompt schema."""
    return {
        "track_candidate_id": track_candidate_id,
        "sample_index": int(selected["sample_index"]),
        "box_xyxy": [
            math.floor(selected["box_xyxy"][0]),
            math.floor(selected["box_xyxy"][1]),
            math.ceil(selected["box_xyxy"][2]),
            math.ceil(selected["box_xyxy"][3]),
        ],
        "semantic_name": selected["semantic_name"],
        "initialization": "automatically_selected_highest_score_detector_box",
        "video": str(video),
        "detector_candidate_id": selected["detection_id"],
        "detector_score": float(selected["score"]),
        "source_frame_index": int(selected["source_frame_index"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=3)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--score-threshold", type=float, default=0.7)
    parser.add_argument("--classes", nargs="*", default=[])
    parser.add_argument("--prompt-source-frame", type=int)
    parser.add_argument("--prompt-class")
    parser.add_argument("--track-candidate-id", default="T_AUTO_001")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.interval < 1 or args.width < 1 or args.start_frame < 0:
        raise ValueError("interval and width must be positive and start-frame non-negative")
    if args.end_frame is not None and args.end_frame < args.start_frame:
        raise ValueError("end-frame must not precede start-frame")
    if not 0 <= args.score_threshold <= 1:
        raise ValueError("score-threshold must be between zero and one")

    video = args.video.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    frames = extract_frames(
        video,
        output / "frames",
        args.interval,
        args.max_frames,
        args.width,
        args.start_frame,
        args.end_frame,
    )

    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights,
        fasterrcnn_resnet50_fpn_v2,
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run reconstruction/check_gpu.py first.")
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    categories = list(weights.meta["categories"])
    model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval().to(device)
    transform = weights.transforms()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    candidates = []
    with torch.inference_mode():
        for frame in frames:
            frame_path = output / "frames" / frame["frame_path"]
            with Image.open(frame_path) as image:
                tensor = transform(image.convert("RGB")).to(device)
            prediction = model([tensor])[0]
            rows = filtered_candidates(
                prediction["boxes"].detach().cpu().numpy(),
                prediction["labels"].detach().cpu().numpy(),
                prediction["scores"].detach().cpu().numpy(),
                categories,
                args.score_threshold,
                set(args.classes),
            )
            for ordinal, row in enumerate(rows, start=1):
                candidates.append(
                    {
                        "detection_id": f"F{frame['sample_index']:06d}C{ordinal:03d}",
                        "sample_index": frame["sample_index"],
                        "source_frame_index": frame["source_frame_index"],
                        "timestamp_seconds": frame["timestamp_seconds"],
                        **row,
                    }
                )

    prompt_metadata = None
    if args.prompt_source_frame is not None:
        selected = select_prompt_candidate(
            candidates, args.prompt_source_frame, args.prompt_class
        )
        prompt = make_tracking_prompt(selected, args.track_candidate_id, video)
        prompt_path = output / "selected_prompt.json"
        prompt_path.write_text(json.dumps(prompt, indent=2) + "\n", encoding="utf-8")
        prompt_metadata = {
            "path": prompt_path.name,
            "sha256": sha256(prompt_path),
            "source_frame_index": selected["source_frame_index"],
            "detection_id": selected["detection_id"],
        }

    revision, dirty = git_state(Path.cwd())
    manifest = {
        "status": "complete",
        "video": str(video),
        "video_sha256": sha256(video),
        "source_input_identity": input_identity(video),
        "detector": {
            "model": MODEL_NAME,
            "weights": WEIGHTS_NAME,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "torchvision": __import__("torchvision").__version__,
        },
        "parameters": {
            "interval": args.interval,
            "start_frame": args.start_frame,
            "end_frame": args.end_frame,
            "max_frames": args.max_frames,
            "width": args.width,
            "score_threshold": args.score_threshold,
            "classes": args.classes,
            "prompt_source_frame": args.prompt_source_frame,
            "prompt_class": args.prompt_class,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "elapsed_seconds": time.monotonic() - started,
            "peak_gpu_gib": (
                round(torch.cuda.max_memory_allocated(device) / (1024**3), 3)
                if device.type == "cuda"
                else None
            ),
        },
        "code_revision": revision,
        "code_dirty": dirty,
        "source_frames_modified": False,
        "source_images_annotated": False,
        "frames": [
            {
                **frame,
                "frame_path": str(Path("frames") / frame["frame_path"]),
            }
            for frame in frames
        ],
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selected_prompt": prompt_metadata,
    }
    manifest_path = output / "detection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "frame_count": len(frames),
                "candidate_count": len(candidates),
                "selected_prompt": prompt_metadata,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
