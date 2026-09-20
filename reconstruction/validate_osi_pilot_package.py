#!/usr/bin/env python3
"""Validate an assembled OSI pilot package before evaluation or handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    args = parser.parse_args()

    root = args.package.expanduser().resolve()
    manifest = json.loads((root / "package_manifest.json").read_text(encoding="utf-8"))
    questions = json.loads((root / "questions_no_ids.json").read_text(encoding="utf-8"))
    ground_truth = json.loads((root / "ground_truth.json").read_text(encoding="utf-8"))

    if manifest.get("status") != "complete":
        raise ValueError("package manifest is not complete")
    if manifest.get("question_count") != len(questions) or len(questions) != len(ground_truth):
        raise ValueError("question or ground-truth count does not match the manifest")
    if len(manifest.get("conditions", [])) != 4:
        raise ValueError("the first-version package must contain four conditions")
    if "(id:" in json.dumps(questions, ensure_ascii=False).lower():
        raise ValueError("no-ID questions still contain an object-ID suffix")
    if any("answer" in row for row in questions):
        raise ValueError("answers leaked into the evaluation question file")

    for record in manifest.get("files", []):
        relative = Path(record["path"])
        if relative.is_absolute():
            raise ValueError(f"manifest contains an absolute path: {relative}")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != record["size_bytes"]:
            raise ValueError(f"size mismatch: {relative}")
        if sha256(path) != record["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {relative}")

    print(
        f"Valid package: {len(manifest['files'])} hashed files, "
        f"{len(questions)} questions, {len(manifest['conditions'])} conditions"
    )


if __name__ == "__main__":
    main()
