#!/usr/bin/env python3
"""Propagate one manually confirmed object box through a video with SAM 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(directory: Path) -> tuple[str | None, bool | None]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "diff", "--quiet", "--ignore-space-at-eol", "HEAD", "--"],
            cwd=directory,
            check=False,
        )
        if diff.returncode not in (0, 1):
            raise subprocess.CalledProcessError(diff.returncode, diff.args)
        dirty = diff.returncode == 1
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def extract_frames(
    video: Path,
    directory: Path,
    interval: int,
    max_frames: int | None,
    width: int,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> list[dict]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        raise ValueError("video FPS is unavailable")
    directory.mkdir(parents=True, exist_ok=True)
    records, source_index = [], 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if end_frame is not None and source_index > end_frame:
            break
        if source_index >= start_frame and (source_index - start_frame) % interval == 0:
            if width and frame.shape[1] != width:
                height = round(frame.shape[0] * width / frame.shape[1])
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            sample = len(records)
            path = directory / f"{sample:06d}.jpg"
            if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"failed to write {path}")
            records.append({"sample_index": sample, "source_frame_index": source_index, "timestamp_seconds": source_index / fps, "frame_path": str(path.name), "frame_sha256": sha256(path)})
            if max_frames is not None and len(records) >= max_frames:
                break
        source_index += 1
    capture.release()
    if not records:
        raise ValueError("no frames were extracted")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("prompt", type=Path, help="JSON with track_candidate_id, sample_index, box_xyxy, semantic_name")
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sam2-revision", required=True)
    parser.add_argument("--model-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--interval", type=int, default=3)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()
    if args.interval < 1 or args.width < 1 or args.start_frame < 0:
        raise ValueError("interval and width must be positive and start-frame non-negative")
    if args.end_frame is not None and args.end_frame < args.start_frame:
        raise ValueError("end-frame must not precede start-frame")
    video, output = args.video.resolve(), args.output.resolve()
    started = time.monotonic()
    prompt = json.loads(args.prompt.read_text(encoding="utf-8"))
    frames = extract_frames(
        video,
        output / "frames",
        args.interval,
        args.max_frames,
        args.width,
        args.start_frame,
        args.end_frame,
    )
    if not 0 <= int(prompt["sample_index"]) < len(frames):
        raise ValueError("prompt sample_index is outside extracted frames")

    from sam2.build_sam import build_sam2_video_predictor

    if not torch.cuda.is_available():
        raise RuntimeError("SAM 2 video tracking requires CUDA in this workflow")
    torch.cuda.reset_peak_memory_stats()
    predictor = build_sam2_video_predictor(args.model_config, str(args.checkpoint.resolve()))
    state = predictor.init_state(video_path=str(output / "frames"))
    object_id = 1
    box = np.asarray(prompt["box_xyxy"], dtype=np.float32)
    masks_dir = output / "masks" / prompt["track_candidate_id"]
    masks_dir.mkdir(parents=True, exist_ok=True)
    frame_masks = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        frame_index, object_ids, logits = predictor.add_new_points_or_box(inference_state=state, frame_idx=int(prompt["sample_index"]), obj_id=object_id, box=box)
        frame_masks[int(frame_index)] = (logits[0] > 0).cpu().numpy().astype(np.uint8) * 255
        for frame_index, object_ids, logits in predictor.propagate_in_video(state):
            frame_masks[int(frame_index)] = (logits[0] > 0).cpu().numpy().astype(np.uint8) * 255
        if int(prompt["sample_index"]) > 0:
            for frame_index, object_ids, logits in predictor.propagate_in_video(state, reverse=True):
                frame_masks[int(frame_index)] = (logits[0] > 0).cpu().numpy().astype(np.uint8) * 255
    entries = []
    for record in frames:
        index = record["sample_index"]
        if index not in frame_masks:
            continue
        mask = np.squeeze(frame_masks[index])
        mask_path = masks_dir / f"{index:06d}.png"
        cv2.imwrite(str(mask_path), mask)
        mask_pixels = int(np.count_nonzero(mask))
        entries.append({
            "track_candidate_id": prompt["track_candidate_id"],
            "sample_index": index,
            "source_frame_index": record["source_frame_index"],
            "timestamp_seconds": record["timestamp_seconds"],
            "semantic_name": prompt.get("semantic_name"),
            "tracking_frame_path": str((output / "frames" / record["frame_path"]).relative_to(output)),
            "tracking_frame_sha256": record["frame_sha256"],
            "mask_path": str(mask_path.relative_to(output)),
            "mask_sha256": sha256(mask_path),
            "mask_area_pixels": mask_pixels,
            "mask_fraction": mask_pixels / mask.size,
            "detector_confidence": None,
            "tracker_confidence": None,
            "visible": bool(mask.any()),
            "occluded": False,
        })
    revision, dirty = git_state(Path.cwd())
    manifest = {
        "status": "complete",
        "track_candidate_id": prompt["track_candidate_id"],
        "video": str(video),
        "video_sha256": sha256(video),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint.resolve()),
        "sam2_revision": args.sam2_revision,
        "model_config": args.model_config,
        "prompt": prompt,
        "prompt_sha256": sha256(args.prompt.resolve()),
        "parameters": {
            "interval": args.interval,
            "start_frame": args.start_frame,
            "end_frame": args.end_frame,
            "max_frames": args.max_frames,
            "width": args.width,
            "bidirectional_from_prompt": int(prompt["sample_index"]) > 0,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(),
            "elapsed_seconds": time.monotonic() - started,
            "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / (1024**3), 3),
        },
        "code_revision": revision,
        "code_dirty": dirty,
        "source_frames_modified": False,
        "frames": entries,
    }
    (output / "track_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(entries)} masks to {output}")


if __name__ == "__main__":
    main()
