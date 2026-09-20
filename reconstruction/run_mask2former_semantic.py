#!/usr/bin/env python3
"""Run local Mask2Former ADE20K semantic segmentation on Pi3 input frames."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    run = args.run_directory.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output
        else run / "semantic_labels.npy"
    )
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    width = int(manifest["resized_width"])
    height = int(manifest["resized_height"])
    records = [
        json.loads(line)
        for line in (run / "frames.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    saved_frame_dir = run / "input_frames"
    if saved_frame_dir.is_dir():
        frame_paths = [saved_frame_dir / f"{index:06d}.png" for index in range(len(records))]
    else:
        source = Path(manifest["input"])
        if not source.is_dir():
            raise ValueError(
                "Video runs require saved input_frames; re-run with --save-observations"
            )
        frame_paths = [source / record["source_name"] for record in records]

    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_path, local_files_only=True
    ).to(args.device)
    model.eval()
    label_maps = []
    for index, frame_path in enumerate(frame_paths):
        image = Image.open(frame_path).convert("RGB").resize(
            (width, height), Image.Resampling.LANCZOS
        )
        inputs = processor(images=image, return_tensors="pt")
        inputs = {name: value.to(args.device) for name, value in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
        semantic = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(height, width)]
        )[0]
        label_maps.append(semantic.to(torch.int16).cpu().numpy())
        print(f"Segmented frame {index + 1}/{len(frame_paths)}: {frame_path.name}")

    labels = np.stack(label_maps)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, labels)
    id2label = {str(key): value for key, value in model.config.id2label.items()}
    metadata = {
        "status": "complete",
        "model": str(model_path),
        "model_sha256": sha256(model_path / "model.safetensors"),
        "task": "ADE20K semantic segmentation",
        "frame_count": len(frame_paths),
        "label_shape": list(labels.shape),
        "label_dtype": str(labels.dtype),
        "unique_labels": [int(value) for value in np.unique(labels)],
        "id2label": id2label,
        "output": str(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote semantic labels: {output}")


if __name__ == "__main__":
    main()
