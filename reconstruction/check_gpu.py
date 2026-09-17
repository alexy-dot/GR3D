#!/usr/bin/env python3
"""Check whether the current environment can run the reconstruction baseline."""

from __future__ import annotations

import json
import platform

import torch


def main() -> None:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "gpu": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "vram_gib": round(props.total_memory / 1024**3, 2),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )

    print(json.dumps(info, indent=2, ensure_ascii=False))

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable. Install a CUDA-enabled PyTorch build and run this "
            "inside WSL2/Linux or a supported Windows environment."
        )


if __name__ == "__main__":
    main()
