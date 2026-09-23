#!/usr/bin/env python3
"""Deterministic identities for Pi3 video files and image directories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


IDENTITY_SCHEMA_VERSION = 1
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg"})


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_directory_files(path: Path) -> list[Path]:
    return sorted(
        (
            entry
            for entry in path.iterdir()
            if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda entry: entry.relative_to(path).as_posix(),
    )


def input_identity(path: Path) -> dict:
    source = path.expanduser().resolve()
    if source.is_file():
        digest = file_sha256(source)
        entries = [{
            "relative_path": source.name,
            "size_bytes": source.stat().st_size,
            "sha256": digest,
        }]
        kind = "file"
        combined = digest
    elif source.is_dir():
        entries = [
            {
                "relative_path": entry.relative_to(source).as_posix(),
                "size_bytes": entry.stat().st_size,
                "sha256": file_sha256(entry),
            }
            for entry in image_directory_files(source)
        ]
        kind = "image_directory"
        digest_payload = {
            "schema_version": IDENTITY_SCHEMA_VERSION,
            "input_kind": kind,
            "ordered_entries": entries,
        }
        combined = hashlib.sha256(
            json.dumps(
                digest_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
    else:
        raise FileNotFoundError(source)
    return {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "algorithm": "sha256",
        "input_kind": kind,
        "combined_sha256": combined,
        "file_count": len(entries),
        "ordered_entries": entries,
    }


def legacy_video_sha256(identity: dict) -> str | None:
    return (
        str(identity["combined_sha256"])
        if identity.get("input_kind") == "file"
        else None
    )


def comparable_identity(identity: dict) -> dict:
    """Return the stable fields that must agree across provenance manifests."""
    return {
        "schema_version": identity.get("schema_version"),
        "algorithm": identity.get("algorithm"),
        "input_kind": identity.get("input_kind"),
        "combined_sha256": identity.get("combined_sha256"),
        "file_count": identity.get("file_count"),
        "ordered_entries": identity.get("ordered_entries"),
    }
