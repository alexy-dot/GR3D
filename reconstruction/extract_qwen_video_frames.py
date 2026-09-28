#!/usr/bin/env python3
"""Extract the exact uniform frames requested by Qwen's video input path."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image

try:
    from reconstruction.input_identity import file_sha256, input_identity
except ModuleNotFoundError:
    from input_identity import file_sha256, input_identity


FRAME_FACTOR = 2


def round_by_factor(value: int, factor: int) -> int:
    return round(value / factor) * factor


def qwen_uniform_frame_indices(total_frames: int, requested_frames: int) -> list[int]:
    if total_frames < FRAME_FACTOR:
        raise ValueError("video must contain at least two frames")
    nframes = round_by_factor(requested_frames, FRAME_FACTOR)
    if not FRAME_FACTOR <= nframes <= total_frames:
        raise ValueError(
            f"rounded frame count must be in [{FRAME_FACTOR}, {total_frames}], got {nframes}"
        )
    return torch.linspace(0, total_frames - 1, nframes).round().long().tolist()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def qwen_utils_provenance() -> dict:
    try:
        spec = importlib.util.find_spec("qwen_vl_utils.vision_process")
    except ModuleNotFoundError:
        spec = None
    source = Path(spec.origin).resolve() if spec and spec.origin else None
    return {
        "package_version": package_version("qwen-vl-utils"),
        "vision_process_path": str(source) if source else None,
        "vision_process_sha256": file_sha256(source) if source and source.is_file() else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--nframes", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    if video_path.suffix.lower() != ".mp4" or not video_path.is_file():
        raise ValueError("--video must be an existing MP4 file")
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_images = sorted(output_dir.glob("*.png"))
    if existing_images or (output_dir / "frame_manifest.json").exists():
        raise FileExistsError(
            "output already contains extracted frames or a manifest; use a new directory"
        )

    try:
        import decord  # noqa: F401
        from qwen_vl_utils.vision_process import _read_video_decord
    except ImportError as error:
        raise RuntimeError(
            "install qwen-vl-utils and decord to match the official Qwen video backend"
        ) from error

    decoded = _read_video_decord(
        {"video": str(video_path), "nframes": args.nframes}
    )
    if not isinstance(decoded, tuple) or len(decoded) != 3:
        raise RuntimeError(
            "installed qwen-vl-utils does not expose frame-index metadata; "
            "use the same package version as the official baseline environment"
        )
    video_tensor, video_metadata, sample_fps = decoded
    indices = [int(index) for index in video_metadata["frames_indices"]]
    total_frames = int(video_metadata["total_num_frames"])
    average_fps = float(video_metadata["fps"])
    expected_indices = qwen_uniform_frame_indices(total_frames, args.nframes)
    if indices != expected_indices:
        raise ValueError("installed Qwen sampler differs from the fixed-nframes contract")
    batch = video_tensor.permute(0, 2, 3, 1).cpu().numpy()

    frames = []
    for sequence_index, (source_index, pixels) in enumerate(zip(indices, batch)):
        relative_path = f"{sequence_index:06d}_src{source_index:06d}.png"
        destination = output_dir / relative_path
        Image.fromarray(pixels).save(destination)
        frames.append(
            {
                "sequence_index": sequence_index,
                "source_index": source_index,
                "timestamp_seconds": source_index / average_fps,
                "source_name": video_path.name,
                "relative_path": relative_path,
                "size_bytes": destination.stat().st_size,
                "sha256": file_sha256(destination),
            }
        )

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_video": str(video_path),
        "source_video_identity": input_identity(video_path),
        "video": {
            "total_frames": total_frames,
            "average_fps": average_fps,
        },
        "sampling": {
            "implementation": "qwen_vl_utils_uniform_nframes_compatible_v1",
            "backend": "decord",
            "reader": "qwen_vl_utils.vision_process._read_video_decord",
            "requested_nframes": args.nframes,
            "effective_nframes": len(indices),
            "sample_fps": float(sample_fps),
            "frame_factor": FRAME_FACTOR,
            "formula": "torch.linspace(0, total_frames - 1, nframes).round().long()",
            "source_indices": indices,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "decord": package_version("decord"),
            "qwen_vl_utils": qwen_utils_provenance(),
        },
        "frames": frames,
    }
    manifest_path = output_dir / "frame_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "video": str(video_path),
                "total_frames": total_frames,
                "average_fps": average_fps,
                "extracted_frames": len(frames),
                "source_indices": indices,
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
