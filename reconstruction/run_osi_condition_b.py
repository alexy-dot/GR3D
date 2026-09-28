#!/usr/bin/env python3
"""Run official OSI-Bench with original video plus Condition-B 3D views."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path

try:
    from reconstruction.input_identity import file_sha256, input_identity
except ModuleNotFoundError:
    from input_identity import file_sha256, input_identity


EVIDENCE_TEXT = (
    "The following three images are camera-gravity-aligned orthographic XY, XZ, and YZ "
    "views of an unfiltered Pi3 reconstruction from the same original video. Axes and "
    "the camera path are shown. Use them only as additional spatial evidence."
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_condition_manifest(path: Path) -> dict:
    manifest = read_json(path)
    if manifest.get("schema_version") != 1 or manifest.get("condition") != "B":
        raise ValueError("expected a schema-v1 Condition-B manifest")
    if not manifest.get("answer_blind") or not manifest.get("original_video_preserved"):
        raise ValueError("Condition-B manifest must preserve original video and be answer-blind")
    data_path = Path(manifest["subset"]["data_path"])
    if file_sha256(data_path) != manifest["subset"]["data_sha256"]:
        raise ValueError("subset data hash mismatch")
    for scene_id, scene in manifest.get("scenes", {}).items():
        video = Path(scene["video"])
        if input_identity(video) != scene["video_identity"]:
            raise ValueError(f"scene {scene_id} video identity mismatch")
        axes = []
        for view in scene.get("views", []):
            view_path = Path(view["path"])
            if file_sha256(view_path) != view["sha256"]:
                raise ValueError(f"scene {scene_id} view hash mismatch: {view_path}")
            axes.append(view["axis"])
        if axes != ["xy", "xz", "yz"]:
            raise ValueError(f"scene {scene_id} must provide ordered XY/XZ/YZ views")
    return manifest


def append_condition_b_evidence(messages: list[dict], scene: dict) -> list[dict]:
    augmented = copy.deepcopy(messages)
    augmented.append({"type": "text", "value": EVIDENCE_TEXT})
    augmented.extend({"type": "image", "value": view["path"]} for view in scene["views"])
    return augmented


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--vlmeval-root", required=True, type=Path)
    parser.add_argument("--condition-manifest", required=True, type=Path)
    return parser.parse_known_args()


def main() -> None:
    args, official_args = parse_args()
    vlmeval_root = args.vlmeval_root.expanduser().resolve()
    condition_path = args.condition_manifest.expanduser().resolve()
    if not (vlmeval_root / "run.py").is_file():
        raise FileNotFoundError(vlmeval_root / "run.py")
    condition = validate_condition_manifest(condition_path)

    sys.path.insert(0, str(vlmeval_root))
    import vlmeval.dataset as dataset_module
    from vlmeval.dataset.OSIBench.osibench import OSIBench

    class OSIConditionB(OSIBench):
        def __init__(
            self,
            condition_manifest,
            data_path,
            dataset="OSI-Bench-Condition-B",
            nframe=32,
            fps=-1,
            download=False,
            **kwargs,
        ):
            supplied = Path(condition_manifest).expanduser().resolve()
            if supplied != condition_path:
                raise ValueError("config condition manifest differs from launcher manifest")
            super().__init__(
                data_path=data_path,
                dataset=dataset,
                nframe=nframe,
                fps=fps,
                download=download,
                **kwargs,
            )
            expected_data = Path(condition["subset"]["data_path"]).resolve()
            if Path(self.data_file).resolve() != expected_data:
                raise ValueError("dataset data path differs from Condition-B manifest")

        def build_prompt(self, line, video_llm, **kwargs):
            row = self.data.iloc[line] if isinstance(line, int) else line
            scene_id = str(row["video"]).zfill(4)
            if scene_id not in condition["scenes"]:
                raise KeyError(f"Condition-B evidence is missing for scene {scene_id}")
            messages = super().build_prompt(line, video_llm, **kwargs)
            if not video_llm or not any(item.get("type") == "video" for item in messages):
                raise ValueError("Condition B requires the original video-LLM input path")
            return append_condition_b_evidence(messages, condition["scenes"][scene_id])

    dataset_module.OSIConditionB = OSIConditionB

    spec = importlib.util.spec_from_file_location("official_vlmeval_run", vlmeval_root / "run.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load official VLMEvalKit run.py")
    official_run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official_run)

    os.chdir(vlmeval_root)
    sys.argv = [str(vlmeval_root / "run.py"), *official_args]
    official_run.load_env()
    official_run.main()


if __name__ == "__main__":
    main()
