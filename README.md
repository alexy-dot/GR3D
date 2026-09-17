# GR3D Geometry Reconstruction Reproduction

This repository contains a focused reproduction of the geometry-construction stage used in the GR3D research workflow.

Current target:

```text
RGB video
-> sampled frames
-> point maps and camera poses
-> overlapping-window alignment
-> persistent point cloud and camera trajectory
```

The first milestone compares original Pi3 with Pi3X on the same short monocular video sample. Long-video alignment and dynamic-region filtering are later extensions, not claims of exact GR3D reproduction.

## Start here

1. Read `PROJECT_CONTEXT.md` for the current state and exact next steps.
2. Follow `reconstruction/README.md` for Windows/WSL2 setup.
3. Clone the pinned Pi3 and VGGT-Long dependencies separately under `third_party/`.
4. Run `python reconstruction/check_gpu.py` before inference.

Model weights, datasets, source videos, point clouds, and generated outputs are intentionally excluded from Git.

