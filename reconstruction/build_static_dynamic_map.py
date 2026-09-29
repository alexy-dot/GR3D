#!/usr/bin/env python3
"""Partition Pi3 observations into static, dynamic, and uncertain artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from reconstruction.tracking_contracts import (
        external_identity_key,
        normalize_external_identity,
        validate_coordinate_system,
    )
except ModuleNotFoundError:
    from tracking_contracts import (
        external_identity_key,
        normalize_external_identity,
        validate_coordinate_system,
    )


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


def assign_entity_ids(records: list[dict]) -> None:
    prefixes = {"static": "S", "dynamic": "D", "uncertain": "U"}
    for state, prefix in prefixes.items():
        matching = sorted(
            (row for row in records if row["motion_state"] == state),
            key=lambda row: row["track_candidate_id"],
        )
        for ordinal, row in enumerate(matching, start=1):
            row["entity_id"] = f"{prefix}{ordinal:03d}"


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
    run_manifest_path = run / "manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("status") != "complete":
        raise ValueError("Pi3 run manifest is not complete")
    coordinate_system = validate_coordinate_system(run_manifest.get("coordinate_system"))
    with np.load(source_path) as archive:
        observations = {name: archive[name] for name in archive.files}
    required_arrays = {"points", "colors", "frame_index", "pixel_yx", "confidence"}
    if not required_arrays.issubset(observations):
        raise ValueError("point observations archive is incomplete")
    point_count = len(observations["points"])
    if observations["points"].ndim != 2 or observations["points"].shape[1] != 3:
        raise ValueError("point observations must have shape (N, 3)")
    if observations["colors"].shape != (point_count, 3):
        raise ValueError("point colors must have shape (N, 3)")
    if any(
        len(observations[name]) != point_count
        for name in ("frame_index", "pixel_yx", "confidence")
    ):
        raise ValueError("point observation arrays have inconsistent lengths")
    tracks = []
    records = []
    candidate_ids = [str(item["track_candidate_id"]) for item in bundle.get("tracks", [])]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("track bundle contains duplicate candidate IDs")
    for item in bundle["tracks"]:
        classification_path = (bundle_path.parent / item["classification_path"]).resolve()
        indices_path = (bundle_path.parent / item["selected_indices_path"]).resolve()
        classification_payload = json.loads(classification_path.read_text(encoding="utf-8"))
        classification = classification_payload["classification"]
        if classification_payload.get("track_candidate_id") != item["track_candidate_id"]:
            raise ValueError("track bundle candidate ID differs from classification")
        if validate_coordinate_system(classification_payload.get("coordinate_system")) != coordinate_system:
            raise ValueError("track classification coordinate system differs from Pi3 run")
        external_identity = normalize_external_identity(
            classification_payload.get("external_identity")
        )
        input_provenance = classification_payload.get("input_provenance", {})
        if input_provenance.get("pi3_manifest_sha256") != sha256(run_manifest_path):
            raise ValueError("classification belongs to a different Pi3 run")
        if input_provenance.get("point_observations_sha256") != sha256(source_path):
            raise ValueError("classification belongs to different Pi3 observations")
        if input_provenance.get("selected_point_indices_sha256") != sha256(indices_path):
            raise ValueError("selected indices do not belong to this classification")
        with np.load(indices_path) as archive:
            indices = np.concatenate([archive[name] for name in archive.files]) if archive.files else np.empty(0, dtype=np.int64)
        tracks.append((classification["motion_state"], indices))
        records.append({
            "track_candidate_id": item["track_candidate_id"],
            "external_identity": external_identity,
            "motion_state": classification["motion_state"],
            "classification_sha256": sha256(classification_path),
            "selected_indices_sha256": sha256(indices_path),
        })
    external_keys = [
        external_identity_key(row["external_identity"])
        for row in records
        if row["external_identity"] is not None
    ]
    if len(external_keys) != len(set(external_keys)):
        raise ValueError("track bundle contains duplicate external identities")
    assign_entity_ids(records)
    groups = partition_indices(point_count, tracks)
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    static_path = output / "static_map.ply"
    write_ply(static_path, observations["points"][groups["static"]], observations["colors"][groups["static"]])
    output_records = {
        "static_map": {
            "path": static_path.name,
            "sha256": sha256(static_path),
            "point_count": int(len(groups["static"])),
        }
    }
    for name in ("dynamic", "uncertain"):
        path = output / f"{name}_observations.npz"
        np.savez_compressed(path, indices=groups[name], points=observations["points"][groups[name]], colors=observations["colors"][groups[name]])
        output_records[f"{name}_observations"] = {
            "path": path.name,
            "sha256": sha256(path),
            "point_count": int(len(groups[name])),
        }
    manifest = {
        "status": "complete",
        "coordinate_system": coordinate_system,
        "source_pi3_manifest_sha256": sha256(run_manifest_path),
        "source_observations_sha256": sha256(source_path),
        "source_count": point_count,
        "static_count": len(groups["static"]),
        "dynamic_count": len(groups["dynamic"]),
        "uncertain_count": len(groups["uncertain"]),
        "tracks": records,
        "raw_source_modified": False,
        "outputs": output_records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
