# GR3D Geometry Reconstruction Baseline

This directory contains the first reproducible geometry-only stage for the GR3D study.
It produces a point cloud and camera trajectory without object annotation or MLLM inference.

## Recommended machine roles

- RTX 4060 Laptop 8 GiB: short Pi3/Pi3X clips, initially 8 frames at about 100K pixels/frame.
- Cloud RTX 3090/4090 24 GiB: longer windows, Pi3XVO and VGGT-Long experiments.

The laptop GPU is suitable for correctness checks, not for the final unbounded-video claim.

## WSL2 setup on the gaming laptop

Run these commands inside Ubuntu 22.04 on WSL2:

```bash
sudo apt update
sudo apt install -y git ffmpeg python3-venv

python3 -m venv ~/venvs/pi3
source ~/venvs/pi3/bin/activate
python -m pip install --upgrade pip

pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r third_party/Pi3/requirements.txt
pip install matplotlib
```

The CUDA version shown by `nvidia-smi` is the maximum version supported by the driver.
It does not have to equal the CUDA runtime bundled with the PyTorch wheel.

## Check the environment

```bash
python reconstruction/check_gpu.py
```

Expected key fields for this laptop:

```text
cuda_available: true
gpu: NVIDIA GeForce RTX 4060 Laptop GPU
vram_gib: about 8
```

## First run: prepare frames only

This verifies paths and sampling without downloading weights:

```bash
python reconstruction/run_pi3_baseline.py \
  --input third_party/Pi3/examples/skating.mp4 \
  --output outputs/pi3_dry_run \
  --model pi3 \
  --interval 12 \
  --max-frames 8 \
  --pixel-limit 100000 \
  --dry-run
```

## First GPU reconstruction

```bash
python reconstruction/run_pi3_baseline.py \
  --input third_party/Pi3/examples/skating.mp4 \
  --output outputs/pi3_skating_8f \
  --model pi3 \
  --interval 12 \
  --max-frames 8 \
  --pixel-limit 100000 \
  --save-observations
```

Then run the practical Pi3X comparison on exactly the same frames:

```bash
python reconstruction/run_pi3_baseline.py \
  --input third_party/Pi3/examples/skating.mp4 \
  --output outputs/pi3x_skating_8f \
  --model pi3x \
  --interval 12 \
  --max-frames 8 \
  --pixel-limit 100000
```

Each run writes:

- `point_cloud.ply`
- `camera_poses.npy` and `camera_poses.json`
- `intrinsics.npy`
- `confidence_stats.json`
- `frames.txt`
- `manifest.json`
- `trajectory.png` when matplotlib is installed
- `point_observations.npz` and `depth_maps.npy` when `--save-observations` is set;
  these preserve the filtered point-to-frame/pixel mapping required to fuse 2D
  semantic labels into a 3D voxel grid

Validate this interface before semantic fusion:

```bash
python reconstruction/validate_observations.py outputs/pi3_house_8f_observations
```

After a segmentation model has written integer label maps with shape
`(frames, height, width)`, fuse them into GR3D-style semantic voxel components:

```bash
python reconstruction/run_mask2former_semantic.py \
  --run-directory outputs/pi3_house_8f_observations \
  --model ~/models/mask2former-swin-small-ade-semantic
```

```bash
python reconstruction/build_semantic_voxels.py \
  --observations outputs/pi3_house_8f_observations/point_observations.npz \
  --semantic-labels outputs/pi3_house_8f_observations/semantic_labels.npy \
  --output outputs/pi3_house_8f_observations/semantic_voxels \
  --voxel-size 0.03 \
  --min-voxel-points 2 \
  --min-voxel-views 1 \
  --min-voxel-purity 0.0 \
  --min-component-voxels 8
```

Voxel size is explicit because original Pi3 scale is not physically validated.
Do not compare object dimensions between scenes until metric scale is established.
The output manifest reports weighted semantic purity, the fraction of voxels
supported by multiple views, fragmentation warnings, and the exact thresholds.
These internal checks detect obvious failures but do not replace LiDAR/ground-truth
geometry evaluation.

Render the retained components without projecting persistent IDs back onto images:

```bash
python reconstruction/render_semantic_voxels.py \
  --voxels outputs/pi3_house_8f_observations/semantic_voxels/semantic_voxels.npz \
  --camera-poses outputs/pi3_house_8f_observations/camera_poses.npy \
  --output outputs/pi3_house_8f_observations/object_block_views \
  --color-by component \
  --style both
```

For MLLM-facing images, render large structural classes separately from discrete
objects so layout boxes do not obscure furniture, vehicles, or people:

```bash
python reconstruction/render_semantic_voxels.py ... --layer layout
python reconstruction/render_semantic_voxels.py ... --layer objects
```

The exact class policy and retained semantic labels are written to each render
manifest. This is a presentation split, not a claim of perfect thing/stuff
instance segmentation.

## Deterministic coordinate views

Render fixed orthographic projections after reconstruction:

```bash
python reconstruction/render_canonical_views.py \
  --point-cloud outputs/pi3_skating_8f/point_cloud.ply \
  --camera-poses outputs/pi3_skating_8f/camera_poses.npy \
  --output outputs/pi3_skating_8f/canonical_views \
  --alignment camera-gravity
```

This writes `view_xy.png`, `view_xz.png`, `view_yz.png`, and a
`render_manifest.json` containing the input hash and rendering parameters. The
views are deterministic for a fixed seed. `camera-gravity` uses the consensus
camera-up direction to make Z vertical, then uses camera motion/view direction
to make horizontal orientation deterministic. The manifest records the origin,
rotation basis, pose convention, heading source, and camera-up disagreement.
This provides scene-independent upright views, but it does not recover geographic
north and should not be described as a semantic room-axis alignment.

## Camera trajectory diagnostics

Measure path length, endpoint displacement, straightness, and raw-coordinate heading change:

```bash
python reconstruction/analyze_camera_trajectory.py \
  --camera-poses outputs/run/camera_poses.npy \
  --output outputs/run/trajectory_analysis.json
```

For a scale-invariant Pi3 run, one known distance may be recorded as an explicit scale anchor:

```bash
python reconstruction/analyze_camera_trajectory.py \
  --camera-poses outputs/run/camera_poses.npy \
  --output outputs/run/trajectory_analysis.json \
  --known-path-length-m 28.1 \
  --anchor-source "OSI-Bench question index 9"
```

This calibrates that run only. It is not an independent metric-accuracy or scale-drift measurement.

## Increasing the workload

Change one variable at a time:

1. Keep `pixel-limit=100000`, increase `max-frames` from 8 to 10, then 12.
2. Only after 12 frames works, try `pixel-limit=150000`.
3. Record peak memory from `manifest.json` after every run.

For CUDA out-of-memory errors, first close browsers/games and retry with:

```text
--max-frames 6 --pixel-limit 70000
```

Do not increase Windows virtual memory to treat GPU out-of-memory errors; system memory
does not replace VRAM for this workload.
