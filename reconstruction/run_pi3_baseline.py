#!/usr/bin/env python3
"""Run a bounded Pi3/Pi3X reconstruction and save reproducibility artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PI3_ROOT = PROJECT_ROOT / "third_party" / "Pi3"
if str(PI3_ROOT) not in sys.path:
    sys.path.insert(0, str(PI3_ROOT))

@dataclass(frozen=True)
class FrameRecord:
    sequence_index: int
    source_index: int
    timestamp_seconds: float | None
    source_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct a bounded image/video clip with Pi3 or Pi3X."
    )
    parser.add_argument("--input", required=True, help="Image directory or MP4 video.")
    parser.add_argument("--output", required=True, help="Directory for this run.")
    parser.add_argument("--model", choices=("pi3", "pi3x"), default="pi3")
    parser.add_argument("--checkpoint", default=None, help="Optional local model checkpoint.")
    parser.add_argument("--interval", type=int, default=30, help="Keep every Nth frame.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=8,
        help="Hard cap after sampling. Use 8 first on an 8 GiB GPU.",
    )
    parser.add_argument(
        "--pixel-limit",
        type=int,
        default=100_000,
        help="Maximum resized pixels per frame; dimensions remain multiples of 14.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.10)
    parser.add_argument("--edge-rtol", type=float, default=0.03)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16"), default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--save-observations",
        action="store_true",
        help=(
            "Save filtered point-to-frame/pixel correspondence and dense depth maps "
            "for later 2D-semantic-to-3D voxel fusion."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only sample/resize frames and write metadata; do not load the model.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.interval < 1:
        raise ValueError("--interval must be at least 1")
    if args.start_frame < 0:
        raise ValueError("--start-frame must be non-negative")
    if args.max_frames < 2:
        raise ValueError("--max-frames must be at least 2 for multi-view reconstruction")
    if args.pixel_limit < 14 * 14:
        raise ValueError("--pixel-limit is too small")
    if not 0.0 < args.confidence_threshold < 1.0:
        raise ValueError("--confidence-threshold must be in (0, 1)")


def target_size(width: int, height: int, pixel_limit: int) -> tuple[int, int]:
    scale = math.sqrt(pixel_limit / (width * height))
    target_w = width * scale
    target_h = height * scale
    patches_w = max(1, round(target_w / 14))
    patches_h = max(1, round(target_h / 14))

    while patches_w * patches_h * 14 * 14 > pixel_limit:
        if patches_w / patches_h > target_w / target_h:
            patches_w -= 1
        else:
            patches_h -= 1
        patches_w = max(1, patches_w)
        patches_h = max(1, patches_h)

    return patches_w * 14, patches_h * 14


def pil_to_tensor(image: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    resized = image.resize(size, Image.Resampling.LANCZOS)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def load_image_directory(
    path: Path,
    interval: int,
    start_frame: int,
    max_frames: int,
) -> tuple[list[Image.Image], list[FrameRecord]]:
    filenames = sorted(
        file for file in path.iterdir() if file.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    selected = filenames[start_frame::interval][:max_frames]
    images = [Image.open(file).convert("RGB") for file in selected]
    records = [
        FrameRecord(i, filenames.index(file), None, file.name) for i, file in enumerate(selected)
    ]
    return images, records


def load_video(
    path: Path,
    interval: int,
    start_frame: int,
    max_frames: int,
) -> tuple[list[Image.Image], list[FrameRecord]]:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "Video input requires opencv-python. Install third_party/Pi3/requirements.txt "
            "or provide an extracted image directory."
        ) from error

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"Cannot open video: {path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else None
    images: list[Image.Image] = []
    records: list[FrameRecord] = []
    source_index = 0

    try:
        while len(images) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if source_index >= start_frame and (source_index - start_frame) % interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(Image.fromarray(rgb))
                records.append(
                    FrameRecord(
                        sequence_index=len(images) - 1,
                        source_index=source_index,
                        timestamp_seconds=(source_index / fps) if fps else None,
                        source_name=path.name,
                    )
                )
            source_index += 1
    finally:
        capture.release()

    return images, records


def load_frames(args: argparse.Namespace) -> tuple[torch.Tensor, list[FrameRecord], tuple[int, int]]:
    input_path = Path(args.input).expanduser().resolve()
    if input_path.is_dir():
        images, records = load_image_directory(
            input_path, args.interval, args.start_frame, args.max_frames
        )
    elif input_path.suffix.lower() == ".mp4":
        images, records = load_video(input_path, args.interval, args.start_frame, args.max_frames)
    else:
        raise ValueError("--input must be an image directory or an MP4 file")

    if len(images) < 2:
        raise RuntimeError(f"Only {len(images)} frame(s) were loaded; at least 2 are required")

    width, height = images[0].size
    size = target_size(width, height, args.pixel_limit)
    tensors = [pil_to_tensor(image, size) for image in images]
    return torch.stack(tensors), records, size


def choose_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU/PyTorch build does not support bfloat16")
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_checkpoint(path: str) -> dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    if args.model == "pi3":
        from pi3.models.pi3 import Pi3

        if args.checkpoint:
            model = Pi3()
            model.load_state_dict(load_checkpoint(args.checkpoint), strict=True)
        else:
            model = Pi3.from_pretrained("yyfz233/Pi3")
    else:
        from pi3.models.pi3x import Pi3X

        if args.checkpoint:
            model = Pi3X(use_multimodal=False)
            model.load_state_dict(load_checkpoint(args.checkpoint), strict=False)
        else:
            model = Pi3X.from_pretrained("yyfz233/Pi3X")
            model.disable_multimodal(free_cuda_cache=False)

    return model.eval().to(device)


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def save_trajectory_plot(camera_poses: np.ndarray, path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    centers = camera_poses[:, :3, 3]
    fig = plt.figure(figsize=(7, 6))
    axis = fig.add_subplot(111, projection="3d")
    axis.plot(centers[:, 0], centers[:, 1], centers[:, 2], "o-", linewidth=1.5)
    axis.scatter(centers[0, 0], centers[0, 1], centers[0, 2], c="green", label="start")
    axis.scatter(centers[-1, 0], centers[-1, 1], centers[-1, 2], c="red", label="end")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images_cpu, frame_records, resized_size = load_frames(args)
    (output_dir / "frames.txt").write_text(
        "\n".join(json.dumps(asdict(record), ensure_ascii=False) for record in frame_records) + "\n",
        encoding="utf-8",
    )
    if args.save_observations:
        frame_dir = output_dir / "input_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        for index, tensor in enumerate(images_cpu):
            array = (
                tensor.permute(1, 2, 0).clamp(0, 1).mul(255).round()
                .to(torch.uint8).numpy()
            )
            Image.fromarray(array).save(frame_dir / f"{index:06d}.png")

    base_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frames_prepared" if args.dry_run else "running",
        "arguments": vars(args),
        "input": str(Path(args.input).expanduser().resolve()),
        "frame_count": len(frame_records),
        "resized_width": resized_size[0],
        "resized_height": resized_size[1],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "pi3_revision": git_revision(PI3_ROOT),
        "checkpoint_sha256": (
            file_sha256(Path(args.checkpoint).expanduser().resolve())
            if args.checkpoint and not args.dry_run
            else None
        ),
    }
    write_json(output_dir / "manifest.json", base_manifest)

    print(
        f"Prepared {len(frame_records)} frames at {resized_size[0]}x{resized_size[1]} "
        f"({resized_size[0] * resized_size[1]:,} pixels/frame)."
    )
    if args.dry_run:
        print(f"Dry run complete: {output_dir}")
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run reconstruction/check_gpu.py first.")
    dtype = choose_dtype(args.precision)

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        from pi3.utils.basic import write_ply
        from pi3.utils.geometry import depth_normal_edge, recover_intrinsic_from_rays_d

        model = load_model(args, device)
        images = images_cpu.unsqueeze(0).to(device, non_blocking=True)
        del images_cpu

        with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=dtype):
            result = model(images)

        probabilities = torch.sigmoid(result["conf"][..., 0])
        mask = probabilities > args.confidence_threshold
        edge = depth_normal_edge(result["local_points"], rtol=args.edge_rtol, mask=mask)
        mask = torch.logical_and(mask, ~edge)[0]
        if mask.sum().item() == 0:
            raise RuntimeError("No points survived filtering; lower --confidence-threshold")

        point_cloud = result["points"][0][mask]
        colors = images[0].permute(0, 2, 3, 1)[mask]
        write_ply(point_cloud.float().cpu(), colors.float().cpu(), str(output_dir / "point_cloud.ply"))

        observation_files: list[str] = []
        if args.save_observations:
            frame_y_x = torch.nonzero(mask, as_tuple=False).cpu().numpy()
            observation_points = point_cloud.float().cpu().numpy()
            observation_colors = (
                colors.float().clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
            )
            observation_confidence = (
                probabilities[0][mask].float().cpu().numpy().astype(np.float16)
            )
            np.savez_compressed(
                output_dir / "point_observations.npz",
                points=observation_points,
                colors=observation_colors,
                frame_index=frame_y_x[:, 0].astype(np.uint16),
                pixel_yx=frame_y_x[:, 1:3].astype(np.uint16),
                confidence=observation_confidence,
            )
            depth_maps = result["local_points"][0, ..., 2].float().cpu().numpy()
            np.save(output_dir / "depth_maps.npy", depth_maps.astype(np.float16))
            observation_files = [
                "input_frames/",
                "point_observations.npz",
                "depth_maps.npy",
            ]

        poses = result["camera_poses"][0].float().cpu().numpy()
        np.save(output_dir / "camera_poses.npy", poses)
        write_json(output_dir / "camera_poses.json", poses.tolist())

        rays = torch.nn.functional.normalize(result["local_points"].float(), dim=-1)
        intrinsics = recover_intrinsic_from_rays_d(
            rays, force_center_principal_point=True
        )[0].cpu().numpy()
        np.save(output_dir / "intrinsics.npy", intrinsics)

        probability_cpu = probabilities.float().cpu().numpy()
        confidence_stats = {
            "threshold": args.confidence_threshold,
            "edge_rtol": args.edge_rtol,
            "min": float(probability_cpu.min()),
            "mean": float(probability_cpu.mean()),
            "median": float(np.median(probability_cpu)),
            "max": float(probability_cpu.max()),
            "total_pixels": int(probability_cpu.size),
            "kept_points": int(mask.sum().item()),
            "kept_ratio": float(mask.sum().item() / probability_cpu.size),
        }
        write_json(output_dir / "confidence_stats.json", confidence_stats)

        elapsed = time.perf_counter() - started
        manifest = {
            **base_manifest,
            "status": "complete",
            "dtype": str(dtype),
            "elapsed_seconds": elapsed,
            "gpu": torch.cuda.get_device_name(device),
            "gpu_total_gib": round(
                torch.cuda.get_device_properties(device).total_memory / 1024**3, 2
            ),
            "peak_gpu_gib": round(torch.cuda.max_memory_allocated(device) / 1024**3, 2),
            "point_count": confidence_stats["kept_points"],
            "observation_files": observation_files,
            "trajectory_plot_written": save_trajectory_plot(
                poses, output_dir / "trajectory.png"
            ),
        }
        write_json(output_dir / "manifest.json", manifest)
        print(f"Reconstruction complete: {output_dir}")
        print(
            f"Kept {manifest['point_count']:,} points; peak allocated GPU memory "
            f"{manifest['peak_gpu_gib']:.2f} GiB."
        )
    except torch.OutOfMemoryError as error:
        failure = {
            **base_manifest,
            "status": "cuda_oom",
            "error": str(error),
            "recommendation": (
                "Retry with --max-frames 6 and/or --pixel-limit 70000. "
                "Close other GPU applications first."
            ),
        }
        write_json(output_dir / "manifest.json", failure)
        raise SystemExit(failure["recommendation"]) from error
    except Exception as error:
        failure = {
            **base_manifest,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_gib": round(
                torch.cuda.max_memory_allocated(device) / 1024**3, 2
            ),
        }
        write_json(output_dir / "manifest.json", failure)
        raise


if __name__ == "__main__":
    main()
