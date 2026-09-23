#!/usr/bin/env python3
"""Validate and summarize manual within-clip track failure annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

if __package__:
    from reconstruction.run_video_instance_tracking import git_state
else:
    from run_video_instance_tracking import git_state


ASSOCIATIONS = {"correct_target", "wrong_target", "not_assessable"}
VISIBILITIES = {"visible", "partially_occluded", "fully_occluded", "not_visible"}
MASK_QUALITIES = {"usable", "partial", "fragmented", "empty"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expand_frame_ranges(track: dict) -> list[dict]:
    frame_count = int(track["frame_count"])
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    labels: list[dict | None] = [None] * frame_count
    for segment in track["frame_ranges"]:
        start = int(segment["start_sample_index"])
        end = int(segment["end_sample_index"])
        if start < 0 or end < start or end >= frame_count:
            raise ValueError("annotation range is outside the declared frame count")
        if segment["association"] not in ASSOCIATIONS:
            raise ValueError(f"unsupported association label: {segment['association']}")
        if segment["target_visibility"] not in VISIBILITIES:
            raise ValueError(f"unsupported visibility label: {segment['target_visibility']}")
        if segment["mask_quality"] not in MASK_QUALITIES:
            raise ValueError(f"unsupported mask-quality label: {segment['mask_quality']}")
        for sample_index in range(start, end + 1):
            if labels[sample_index] is not None:
                raise ValueError(f"overlapping annotation at sample {sample_index}")
            labels[sample_index] = {
                "sample_index": sample_index,
                "association": segment["association"],
                "target_visibility": segment["target_visibility"],
                "mask_quality": segment["mask_quality"],
            }
    missing = [index for index, label in enumerate(labels) if label is None]
    if missing:
        raise ValueError(f"annotation does not cover samples: {missing}")
    return [label for label in labels if label is not None]


def validate_episodes(track: dict, key: str) -> list[dict]:
    frame_count = int(track["frame_count"])
    episodes = track.get(key, [])
    previous_end = -1
    for episode in episodes:
        start = int(episode["start_sample_index"])
        end = int(episode["end_sample_index"])
        if start < 0 or end < start or end >= frame_count:
            raise ValueError(f"{key} interval is outside the declared frame count")
        if start <= previous_end:
            raise ValueError(f"{key} intervals must be sorted and non-overlapping")
        previous_end = end
    return episodes


def score_track(track: dict) -> dict:
    labels = expand_frame_ranges(track)
    association_counts = Counter(row["association"] for row in labels)
    visibility_counts = Counter(row["target_visibility"] for row in labels)
    quality_counts = Counter(row["mask_quality"] for row in labels)
    id_switch_events = track.get("id_switch_events", [])
    for event in id_switch_events:
        sample_index = int(event["sample_index"])
        if sample_index < 1 or sample_index >= len(labels):
            raise ValueError("ID-switch event must reference a non-initial sample")
    fragmentation = validate_episodes(track, "fragmentation_episodes")
    occlusion = validate_episodes(track, "occlusion_episodes")
    assessable = association_counts["correct_target"] + association_counts["wrong_target"]
    return {
        "track_candidate_id": track["track_candidate_id"],
        "frame_count": len(labels),
        "identity_assessable_frame_count": assessable,
        "correct_target_frame_count": association_counts["correct_target"],
        "wrong_target_frame_count": association_counts["wrong_target"],
        "observed_correct_association_fraction_on_assessable_frames": (
            association_counts["correct_target"] / assessable if assessable else None
        ),
        "id_switch_event_count": len(id_switch_events),
        "fragmentation_episode_count": len(fragmentation),
        "occlusion_episode_count": len(occlusion),
        "mask_failure_frame_count": quality_counts["fragmented"] + quality_counts["empty"],
        "mask_failure_frame_fraction": (
            quality_counts["fragmented"] + quality_counts["empty"]
        )
        / len(labels),
        "association_counts": dict(sorted(association_counts.items())),
        "visibility_counts": dict(sorted(visibility_counts.items())),
        "mask_quality_counts": dict(sorted(quality_counts.items())),
    }


def score_annotations(payload: dict) -> dict:
    tracks = [score_track(track) for track in payload["tracks"]]
    total_frames = sum(track["frame_count"] for track in tracks)
    assessable = sum(track["identity_assessable_frame_count"] for track in tracks)
    correct = sum(track["correct_target_frame_count"] for track in tracks)
    failures = sum(track["mask_failure_frame_count"] for track in tracks)
    return {
        "track_count": len(tracks),
        "reviewed_frame_count": total_frames,
        "identity_assessable_frame_count": assessable,
        "correct_target_frame_count": correct,
        "observed_correct_association_fraction_on_assessable_frames": (
            correct / assessable if assessable else None
        ),
        "id_switch_event_count": sum(track["id_switch_event_count"] for track in tracks),
        "fragmentation_episode_count": sum(
            track["fragmentation_episode_count"] for track in tracks
        ),
        "occlusion_episode_count": sum(track["occlusion_episode_count"] for track in tracks),
        "mask_failure_frame_count": failures,
        "mask_failure_frame_fraction": failures / total_frames,
        "tracks": tracks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    annotation_path = args.annotations.resolve()
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    revision, dirty = git_state(Path.cwd())
    result = {
        "status": "complete",
        "annotation_path": str(annotation_path),
        "annotation_sha256": sha256(annotation_path),
        "annotation_scope": payload["annotation_scope"],
        "summary": score_annotations(payload),
        "code_revision": revision,
        "code_dirty": dirty,
        "limitations": payload["limitations"],
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
