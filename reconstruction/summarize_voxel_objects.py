#!/usr/bin/env python3
"""Print the largest semantic voxel components for inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("objects", type=Path)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    payload = json.loads(args.objects.read_text(encoding="utf-8"))
    objects = sorted(payload["objects"], key=lambda item: item["point_count"], reverse=True)
    print("component\tclass\tvoxels\tpoints\tbbox_size_xyz")
    for item in objects[: args.limit]:
        extent = [
            maximum - minimum
            for minimum, maximum in zip(
                item["bbox_min_xyz"], item["bbox_max_xyz"], strict=True
            )
        ]
        name = item.get("semantic_name") or str(item["semantic_label"])
        print(
            f"{item['component_id']}\t{name}\t{item['voxel_count']}\t"
            f"{item['point_count']}\t" + ",".join(f"{value:.3f}" for value in extent)
        )


if __name__ == "__main__":
    main()
