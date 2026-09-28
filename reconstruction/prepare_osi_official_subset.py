#!/usr/bin/env python3
"""Create a deterministic, official-evaluator-compatible OSI-Bench subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_METADATA_URL = (
    "https://huggingface.co/datasets/HarmlessSR07/OSI-Bench/resolve/main/"
    "data.parquet"
)
DEFAULT_ARCHIVE_URLS = tuple(
    "https://huggingface.co/datasets/HarmlessSR07/OSI-Bench/resolve/main/"
    f"videos_part{part}.zip"
    for part in range(1, 5)
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(directory: Path) -> tuple[str | None, bool | None]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=directory,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def select_scene_ids(
    rows: list[dict],
    scene_count: int,
    seed: int,
    include_scenes: list[str] | None = None,
) -> list[str]:
    if scene_count < 1:
        raise ValueError("scene_count must be positive")
    rows_by_scene: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_scene[str(row["video"])].append(row)
    available = set(rows_by_scene)
    includes = list(dict.fromkeys(str(scene) for scene in (include_scenes or [])))
    missing = set(includes) - available
    if missing:
        raise ValueError(f"included scenes are absent from metadata: {sorted(missing)}")
    if len(includes) > scene_count:
        raise ValueError("scene_count is smaller than the required included scenes")
    if scene_count > len(available):
        raise ValueError("scene_count exceeds the number of available scenes")

    rng = random.Random(seed)
    candidates = sorted(available - set(includes))
    rng.shuffle(candidates)
    tie_rank = {scene: rank for rank, scene in enumerate(candidates)}
    selected = list(includes)
    category_counts: Counter[str] = Counter()
    all_categories = {str(row["category"]) for row in rows}

    def add_scene(scene: str) -> None:
        category_counts.update(str(row["category"]) for row in rows_by_scene[scene])

    for scene in selected:
        add_scene(scene)

    while len(selected) < scene_count:
        uncovered = all_categories - set(category_counts)

        def score(scene: str) -> tuple[float, float, int, int]:
            scene_categories = {
                str(row["category"]) for row in rows_by_scene[scene]
            }
            coverage_gain = len(scene_categories & uncovered)
            balance_gain = sum(
                1.0 / (1.0 + category_counts[category])
                for category in scene_categories
            )
            return (
                float(coverage_gain),
                balance_gain,
                len(scene_categories),
                -tie_rank[scene],
            )

        chosen = max((scene for scene in candidates if scene not in selected), key=score)
        selected.append(chosen)
        add_scene(chosen)
    return selected


def download_file(url: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "GR3D-OSI-subset/1"})
    with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    temporary.replace(destination)


def download_selected_videos(
    scene_ids: list[str], output_dir: Path, archive_urls: tuple[str, ...]
) -> dict[str, dict]:
    try:
        from remotezip import RemoteZip
    except ImportError as exc:
        raise RuntimeError(
            "video download requires remotezip: pip install remotezip"
        ) from exc

    wanted = {f"{scene}.mp4" for scene in scene_ids}
    records: dict[str, dict] = {}
    for archive_url in archive_urls:
        unresolved = wanted - set(records)
        if not unresolved:
            break
        with RemoteZip(archive_url) as archive:
            available = unresolved & set(archive.namelist())
            for member in sorted(available):
                info = archive.getinfo(member)
                destination = output_dir / Path(member).name
                if destination.exists() and destination.stat().st_size == info.file_size:
                    status = "reused"
                else:
                    temporary = destination.with_suffix(destination.suffix + ".part")
                    with archive.open(member) as source, temporary.open("wb") as target:
                        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                    temporary.replace(destination)
                    status = "downloaded"
                records[member] = {
                    "archive_url": archive_url,
                    "size_bytes": destination.stat().st_size,
                    "sha256": sha256(destination),
                    "status": status,
                }
    missing = wanted - set(records)
    if missing:
        raise FileNotFoundError(f"selected videos were not found in archives: {sorted(missing)}")
    return records


def build_official_config(output_dir: Path, nframe: int, model: str) -> dict:
    return {
        "model": {model: {}},
        "data": {
            "OSI-Bench-Subset": {
                "class": "OSIBench",
                "data_path": str(output_dir.resolve()),
                "dataset": "OSI-Bench-Subset",
                "nframe": nframe,
                "download": False,
            }
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--metadata", default=DEFAULT_METADATA_URL)
    parser.add_argument("--scene-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--include-scene", action="append", default=[])
    parser.add_argument("--download-videos", action="store_true")
    parser.add_argument("--nframe", type=int, default=32)
    parser.add_argument("--model", default="Qwen2.5-VL-3B-Instruct")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("install pandas and pyarrow before preparing the subset") from exc

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "source_data.parquet"
    if str(args.metadata).startswith(("http://", "https://")):
        download_file(str(args.metadata), metadata_path)
    else:
        source = Path(args.metadata).resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        if source != metadata_path:
            shutil.copy2(source, metadata_path)

    frame = pd.read_parquet(metadata_path)
    if "video_id" in frame.columns and "video" not in frame.columns:
        frame = frame.rename(columns={"video_id": "video"})
    required = {"video", "category"}
    if not required.issubset(frame.columns):
        raise ValueError(f"metadata is missing columns: {sorted(required - set(frame.columns))}")
    frame["video"] = frame["video"].astype(str).str.zfill(4)
    rows = frame.to_dict("records")
    selected = select_scene_ids(
        rows, args.scene_count, args.seed, include_scenes=args.include_scene
    )
    subset = frame[frame["video"].isin(selected)].copy()
    subset = subset.sort_values("index") if "index" in subset.columns else subset
    subset_path = output_dir / "data.parquet"
    subset.to_parquet(subset_path, index=False)

    video_records = {}
    if args.download_videos:
        video_records = download_selected_videos(
            selected, output_dir, DEFAULT_ARCHIVE_URLS
        )

    config = build_official_config(output_dir, args.nframe, args.model)
    config_path = output_dir / "osibench_subset_config.json"
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    category_counts = {
        str(key): int(value)
        for key, value in subset["category"].value_counts().sort_index().items()
    }
    scene_question_counts = {
        str(key): int(value)
        for key, value in subset["video"].value_counts().sort_index().items()
    }
    revision, dirty = git_state(Path.cwd())
    manifest = {
        "schema_version": 1,
        "selection_strategy": "complete_scene_greedy_category_coverage_then_balance_v1",
        "code_revision": revision,
        "code_dirty": dirty,
        "script_sha256": sha256(Path(__file__).resolve()),
        "source_metadata": str(metadata_path),
        "source_metadata_sha256": sha256(metadata_path),
        "subset_metadata_sha256": sha256(subset_path),
        "seed": args.seed,
        "scene_count": len(selected),
        "question_count": len(subset),
        "selected_scenes_in_selection_order": selected,
        "included_scenes": args.include_scene,
        "category_question_counts": category_counts,
        "scene_question_counts": scene_question_counts,
        "videos_downloaded": args.download_videos,
        "archive_urls": list(DEFAULT_ARCHIVE_URLS),
        "video_records": video_records,
        "official_config": str(config_path),
        "official_config_sha256": sha256(config_path),
        "model": args.model,
        "nframe": args.nframe,
    }
    manifest_path = output_dir / "subset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "scene_count": len(selected),
        "question_count": len(subset),
        "categories": category_counts,
        "config": str(config_path),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
