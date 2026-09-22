#!/usr/bin/env python3
"""Align dense tracking masks to Pi3 sample indices and resolution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def match_frames(track_frames: list[dict], pi3_frames: list[dict]) -> list[tuple[int, int, dict]]:
    by_source: dict[int, dict] = {}
    for row in track_frames:
        source_index = int(row["source_frame_index"])
        if source_index in by_source:
            raise ValueError(f"duplicate tracking source frame index: {source_index}")
        by_source[source_index] = row
    matched = []
    for ordinal, pi3_frame in enumerate(pi3_frames):
        sample_index = int(pi3_frame.get("sequence_index", ordinal))
        source_index = int(pi3_frame.get("source_frame_index", pi3_frame["source_index"]))
        if source_index in by_source:
            matched.append((sample_index, source_index, by_source[source_index]))
    return matched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("track_manifest", type=Path)
    parser.add_argument("pi3_run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    source_path, run = args.track_manifest.resolve(), args.pi3_run.resolve()
    track = json.loads(source_path.read_text(encoding="utf-8"))
    run_manifest_path = run / "manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    frame_rows = [json.loads(line) for line in (run / "frames.txt").read_text(encoding="utf-8").splitlines() if line]
    pi3_input = Path(run_manifest["input"])
    provenance_warnings = []
    if pi3_input.exists():
        pi3_video_sha256 = sha256(pi3_input)
        if track.get("video_sha256") != pi3_video_sha256:
            raise ValueError("tracking and Pi3 runs use different source videos")
    else:
        pi3_video_sha256 = None
        provenance_warnings.append("pi3_input_video_unavailable_for_hash_validation")
    output = args.output.resolve()
    masks_dir = output / "masks" / track["track_candidate_id"]
    masks_dir.mkdir(parents=True, exist_ok=True)
    aligned = []
    size = (int(run_manifest["resized_width"]), int(run_manifest["resized_height"]))
    matches = match_frames(track["frames"], frame_rows)
    if len(matches) != len(frame_rows) and not args.allow_partial:
        raise ValueError(
            f"only {len(matches)} of {len(frame_rows)} Pi3 frames have tracking masks"
        )
    for sample_index, source_index, source in matches:
        source_mask = source_path.parent / source["mask_path"]
        with Image.open(source_mask) as image:
            mask = image.convert("L").resize(size, Image.Resampling.NEAREST)
        target = masks_dir / f"{sample_index:06d}.png"
        mask.save(target)
        aligned.append({
            **source,
            "tracking_sample_index": source["sample_index"],
            "sample_index": sample_index,
            "source_frame_index": source_index,
            "mask_path": str(target.relative_to(output)),
            "source_mask_sha256": sha256(source_mask),
            "mask_sha256": sha256(target),
        })
    result = {
        "status": "complete",
        "track_candidate_id": track["track_candidate_id"],
        "pi3_manifest_sha256": sha256(run_manifest_path),
        "source_track_manifest_sha256": sha256(source_path),
        "source_video_sha256": track.get("video_sha256"),
        "pi3_video_sha256": pi3_video_sha256,
        "provenance_warnings": provenance_warnings,
        "target_size_wh": list(size),
        "matched_frame_count": len(aligned),
        "pi3_frame_count": len(frame_rows),
        "frames": aligned,
    }
    (output / "track_manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Aligned {len(aligned)} of {len(frame_rows)} Pi3 frames")


if __name__ == "__main__":
    main()
