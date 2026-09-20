#!/usr/bin/env python3
"""Render target-class semantic overlays and write per-frame coverage statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PALETTE = [
    (230, 25, 75),
    (60, 180, 75),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--semantic-labels", required=True, type=Path)
    parser.add_argument("--semantic-metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--target-classes",
        default="bicycle,minibike,person",
        help="Comma-separated ADE20K class names to audit.",
    )
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--columns", type=int, default=4)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def class_coverage(
    label_map: np.ndarray, target_ids: dict[str, int]
) -> dict[str, dict[str, float | int | bool]]:
    total = int(label_map.size)
    result: dict[str, dict[str, float | int | bool]] = {}
    for name, label_id in target_ids.items():
        pixels = int(np.count_nonzero(label_map == label_id))
        result[name] = {
            "label_id": label_id,
            "pixel_count": pixels,
            "pixel_ratio": pixels / total,
            "present": pixels > 0,
        }
    return result


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1")
    if args.columns < 1:
        raise ValueError("--columns must be positive")

    run = args.run_directory.expanduser().resolve()
    labels_path = args.semantic_labels.expanduser().resolve()
    metadata_path = args.semantic_metadata.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    labels = np.load(labels_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    id2label = {int(key): value for key, value in metadata["id2label"].items()}
    label2id = {value: key for key, value in id2label.items()}
    target_names = [name.strip() for name in args.target_classes.split(",") if name.strip()]
    missing = [name for name in target_names if name not in label2id]
    if missing:
        raise ValueError(f"Unknown semantic classes: {missing}")
    target_ids = {name: label2id[name] for name in target_names}

    records = [
        json.loads(line)
        for line in (run / "frames.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if labels.ndim != 3 or labels.shape[0] != len(records):
        raise ValueError(
            f"Semantic labels shape {labels.shape} does not match {len(records)} frames"
        )

    colors = {name: PALETTE[index % len(PALETTE)] for index, name in enumerate(target_names)}
    font = ImageFont.load_default()
    rendered: list[Image.Image] = []
    frame_stats = []
    for index, record in enumerate(records):
        frame_path = run / "input_frames" / f"{index:06d}.png"
        image = Image.open(frame_path).convert("RGB")
        if image.size != (labels.shape[2], labels.shape[1]):
            image = image.resize((labels.shape[2], labels.shape[1]), Image.Resampling.LANCZOS)
        base = np.asarray(image, dtype=np.float32)
        overlay = base.copy()
        frame_labels = labels[index]
        for name, label_id in target_ids.items():
            mask = frame_labels == label_id
            if mask.any():
                color = np.asarray(colors[name], dtype=np.float32)
                overlay[mask] = (1.0 - args.alpha) * base[mask] + args.alpha * color

        canvas = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(canvas)
        frame_number = record.get(
            "source_index",
            record.get("source_frame_index", record.get("frame_index", index)),
        )
        draw.rectangle((0, 0, canvas.width, 18), fill=(0, 0, 0))
        draw.text((4, 4), f"sample {index} | source frame {frame_number}", fill="white", font=font)
        x = 4
        y = canvas.height - 16
        draw.rectangle((0, y - 2, canvas.width, canvas.height), fill=(0, 0, 0))
        coverage = class_coverage(frame_labels, target_ids)
        for name in target_names:
            text = f"{name}:{coverage[name]['pixel_count']}"
            draw.rectangle((x, y, x + 8, y + 8), fill=colors[name])
            draw.text((x + 11, y - 2), text, fill="white", font=font)
            x += 12 + max(48, len(text) * 6)
        canvas.save(output / f"overlay_{index:06d}.png")
        rendered.append(canvas)
        frame_stats.append(
            {
                "sample_index": index,
                "source_frame_index": frame_number,
                "timestamp_seconds": record.get("timestamp_seconds"),
                "source_name": record.get("source_name"),
                "coverage": coverage,
            }
        )

    columns = min(args.columns, len(rendered))
    rows = math.ceil(len(rendered) / columns)
    width, height = rendered[0].size
    sheet = Image.new("RGB", (columns * width, rows * height), "white")
    for index, image in enumerate(rendered):
        sheet.paste(image, ((index % columns) * width, (index // columns) * height))
    sheet.save(output / "contact_sheet.png")

    summary = {}
    for name in target_names:
        present_frames = sum(item["coverage"][name]["present"] for item in frame_stats)
        summary[name] = {
            "label_id": target_ids[name],
            "frames_present": int(present_frames),
            "frame_count": len(frame_stats),
            "total_pixels": int(
                sum(item["coverage"][name]["pixel_count"] for item in frame_stats)
            ),
        }
    audit = {
        "status": "complete",
        "representation": "evaluation_only_semantic_overlay_without_object_ids",
        "run_directory": str(run),
        "semantic_labels": str(labels_path),
        "semantic_labels_sha256": sha256(labels_path),
        "target_classes": target_names,
        "colors_rgb": colors,
        "alpha": args.alpha,
        "summary": summary,
        "frames": frame_stats,
        "outputs": ["contact_sheet.png"]
        + [f"overlay_{index:06d}.png" for index in range(len(rendered))],
        "warning": (
            "Pixel presence measures semantic-model coverage only. It does not establish "
            "instance identity or 2D-to-3D correspondence."
        ),
    }
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote semantic audit for {len(rendered)} frames: {output}")


if __name__ == "__main__":
    main()
