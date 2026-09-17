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
  --pixel-limit 100000
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
