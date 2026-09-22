#!/usr/bin/env python3
"""Partition Pi3 observations into static, dynamic, and uncertain artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def partition_indices(count: int, tracks: list[tuple[str, np.ndarray]]) -> dict[str, np.ndarray]:
    dynamic, uncertain = set(), set()
    for state, indices in tracks:
        values = {int(value) for value in np.asarray(indices).reshape(-1)}
        if any(value < 0 or value >= count for value in values):
            raise ValueError("track point index exceeds observation bounds")
        if state == "dynamic":
            dynamic.update(values)
        elif state == "uncertain":
            uncertain.update(values)
        elif state != "static":
            raise ValueError(f"unsupported motion state: {state}")
    uncertain.difference_update(dynamic)
    excluded = dynamic | uncertain
    static = sorted(set(range(count)) - excluded)
    return {
        "static": np.asarray(static, dtype=np.int64),
        "dynamic": np.asarray(sorted(dynamic), dtype=np.int64),
        "uncertain": np.asarray(sorted(uncertain), dtype=np.int64),
    }


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    vertex = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for index, name in enumerate(("x", "y", "z")):
        vertex[name] = points[:, index]
    for index, name in enumerate(("red", "green", "blue")):
        vertex[name] = colors[:, index]
    header = ("ply\nformat binary_little_endian 1.0\n" f"element vertex {len(vertex)}\n" "property float x\nproperty float y\nproperty float z\n" "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        vertex.tofile(stream)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("track_bundle", type=Path, help="JSON list of classifications and selected-index NPZ files")
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    run = args.run_directory.resolve()
    bundle_path = args.track_bundle.resolve()
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    source_path = run / "point_observations.npz"
    with np.load(source_path) as archive:
        observations = {name: archive[name] for name in archive.files}
    tracks = []
    records = []
    for item in bundle["tracks"]:
        classification_path = (bundle_path.parent / item["classification_path"]).resolve()
        indices_path = (bundle_path.parent / item["selected_indices_path"]).resolve()
        classification = json.loads(classification_path.read_text(encoding="utf-8"))["classification"]
        with np.load(indices_path) as archive:
            indices = np.concatenate([archive[name] for name in archive.files]) if archive.files else np.empty(0, dtype=np.int64)
        tracks.append((classification["motion_state"], indices))
        records.append({"track_candidate_id": item["track_candidate_id"], "motion_state": classification["motion_state"], "classification_sha256": sha256(classification_path), "selected_indices_sha256": sha256(indices_path)})
    groups = partition_indices(len(observations["points"]), tracks)
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_ply(output / "static_map.ply", observations["points"][groups["static"]], observations["colors"][groups["static"]])
    for name in ("dynamic", "uncertain"):
        np.savez_compressed(output / f"{name}_observations.npz", indices=groups[name], points=observations["points"][groups[name]], colors=observations["colors"][groups[name]])
    manifest = {"status": "complete", "source_observations_sha256": sha256(source_path), "source_count": len(observations["points"]), "static_count": len(groups["static"]), "dynamic_count": len(groups["dynamic"]), "uncertain_count": len(groups["uncertain"]), "tracks": records, "raw_source_modified": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
