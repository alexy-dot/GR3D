# Project Context

Last updated: 2026-09-20

## Current objective

Reproduce and evaluate the 3D scene-construction stage related to GR3D, then extend it toward persistent mapping from real, potentially very long monocular video.

Current pipeline target:

```text
RGB video
-> bounded frame/keyframe selection
-> per-window point maps and camera poses
-> overlap alignment
-> persistent global point cloud and camera trajectory
```

Out of scope for the current milestone:

- object IDs or boxes redrawn onto source images;
- GR3D visual re-annotation;
- geometry-to-text conversion;
- downstream MLLM inference or training;
- full dynamic 4D reconstruction.

The first dynamic-scene target is a stable static environmental map with moving regions filtered, not explicit moving-object trajectories.

## Research background

The reading path was `OSI-Bench -> Omni-View -> GR3D`. This led to the current concern: indoor/static multiview methods do not directly resolve persistent reconstruction for real outdoor video containing camera motion, long duration, scale ambiguity, loop drift, and moving objects.

The previous Omni-View/OSI-Bench evaluation work is background context only and is not stored in this focused repository.

## Decisions and reasons

1. **Separate reproduction from extension.** Original Pi3 is used to reproduce the geometry interface closest to GR3D. Pi3X/Pi3XVO/VGGT-Long results must be labeled as practical extensions.
2. **Do not assume physical metric scale.** Original Pi3 describes scale-invariant geometry. Pi3X provides approximate metric behavior, which still requires validation against known distances or trajectories.
3. **Do not write long-video alignment from scratch first.** Official Pi3 includes `Pi3XVO` with overlapping windows and Sim(3) alignment. VGGT-Long supports a Pi3 backend, disk-backed chunks, loop detection, and global optimization.
4. **Build a static map before dynamic 4D modeling.** Moving people/vehicles can create duplicate points and corrupt alignment. First measure the effect of masking them.
5. **Do not use codec motion vectors as the reconstruction core.** They may later help keyframe or motion-region selection but do not provide the required 3D geometry.
6. **Use Git handoff instead of Codex Remote.** Windows Codex Remote repeatedly fails while phone-to-Mac Remote works. Do not continue Remote debugging unless explicitly requested.

## Implemented state

### Research plan

- `research/experiments/2026-09-17-GR3D三维场景重建复现计划.md`
- Defines Phase A0/A1 short-video reproduction, Phase B window fusion, Phase C loop closure, and Phase D dynamic-object masking.

### Reconstruction entry point

- `reconstruction/check_gpu.py`
  - Reports PyTorch, CUDA, GPU, VRAM, compute capability, and BF16 support.
- `reconstruction/run_pi3_baseline.py`
  - Accepts an MP4 or image directory.
  - Bounds work with `--start-frame`, `--interval`, `--max-frames`, and `--pixel-limit`.
  - Runs original Pi3 or Pi3X.
  - Exports filtered point cloud, C2W camera poses, recovered intrinsics, confidence statistics, frame manifest, run manifest, and optional trajectory plot.
  - Writes an actionable manifest on CUDA OOM.
- `reconstruction/README.md`
  - Contains WSL2 setup and exact initial commands for the gaming laptop.

### Local verification completed on Mac

- Python syntax compilation passed for both reconstruction scripts.
- A frame-preparation `--dry-run` passed at approximately 100K pixels per frame.
- Full model inference has not run on Mac because CUDA and MPS are unavailable in the current PyTorch environment.

### Windows/WSL verification and first CUDA baseline

- Ubuntu 22.04 under WSL2 is initialized with Python 3.10.12 and Git 2.34.1.
- CUDA PyTorch check passed with PyTorch 2.5.1+cu121, CUDA available, compute capability 8.9, 8.0 GiB VRAM, and BF16 support.
- The 8-frame dry-run passed on `third_party/Pi3/examples/skating.mp4` at 434x224 (97,216 pixels/frame), using Pi3 revision `9fa3ddb3f8d53041f8b2738df404f62223bbaa7b`.
- Direct Hugging Face access from WSL failed because both IPv4 and IPv6 connections were unreachable. The official Pi3 `model.safetensors` was downloaded through Windows and copied into WSL.
- Local Pi3 checkpoint SHA-256: `33580e4702ac671558aedeab1148fd08118f7ce45bdbeb99f3e3cf340062875d` (3.6 GiB).
- First full original-Pi3 CUDA run succeeded with 8 frames and 100K pixel limit. It retained 343,214 points and used 5.52 GiB peak allocated GPU memory.
- The CUDA-compiled RoPE2D extension was unavailable, so Pi3 used its slower PyTorch fallback; this affected speed, not run completion.

## Compute environments

### Mac

- Apple Silicon MacBook Air, macOS 14.8.3.
- Current PyTorch: 2.10.0.
- CUDA unavailable; MPS unavailable in this build.
- Use for code, documentation, frame preparation, result inspection, and lightweight post-processing.

### Windows gaming laptop

- NVIDIA GeForce RTX 4060 Laptop GPU.
- 8,188 MiB VRAM.
- Driver 596.08; `nvidia-smi` reports CUDA capability up to 13.2.
- GPU power limit shown as 135W in the Windows `nvidia-smi` check on 2026-09-17; the earlier 45W observation was incorrect or came from a different power state.
- WSL2 is active after the Windows restart, and Ubuntu 22.04 is registered and initialized as a version-2 distribution.
- Linux user `dell` (UID 1000) is initialized and belongs to the `sudo` group.
- Ubuntu reports Python 3.10.12 and Git 2.34.1.
- WSL GPU passthrough is verified: `nvidia-smi` sees the NVIDIA GeForce RTX 4060 Laptop GPU, 8,188 MiB VRAM, Windows driver 596.08, and CUDA capability up to 13.2.
- Firmware virtualization is enabled and the CPU reports VM monitor mode extensions and second-level address translation support.
- No `python` executable is available in the current Windows PowerShell environment.
- Intended first workload: 8 frames at about 100K pixels/frame; then 10 and 12 frames only after successful recorded runs.
- Preferred runtime: WSL2 Ubuntu 22.04 with CUDA-enabled PyTorch.

### Cloud rental

- Reserve for long-video Pi3XVO/VGGT-Long experiments.
- Initial target: Linux with RTX 3090/4090 24GB and persistent disk.
- Do not rent until the same short input succeeds on the gaming laptop and produces inspectable artifacts.

## External dependencies and pinned revisions

These directories are intentionally ignored by the main repository and must be cloned separately:

```bash
git clone https://github.com/yyfz/Pi3.git third_party/Pi3
git clone https://github.com/DengKaiCQ/VGGT-Long.git third_party/VGGT-Long
```

Revisions inspected on 2026-09-17:

- Pi3: `9fa3ddb3f8d53041f8b2738df404f62223bbaa7b`
- VGGT-Long: `c160869d1d99c96bb227f414afb3bc68c29c9a76`

After cloning on another machine, check out these revisions before reproducing the current state.

## Known unresolved issues

1. The WSL virtual environment and CUDA-enabled PyTorch are installed and verified. Native Windows Python is absent, but it is not required for the selected WSL workflow.
2. The original Pi3 weight is available locally, but Pi3X weights have not yet been downloaded on Windows.
3. The first full CUDA run and its artifacts were inspected and recorded. The remaining quality question is whether Pi3X reduces the observed grid/ray artifacts on the same frames.
4. The current baseline has not yet been compared against ground-truth trajectory or known metric distances.
5. `Pi3XVO` still retains the selected image tensor and merged dense points in memory; it is a medium-sequence validation step, not the final unbounded map store.
6. VGGT-Long's Pi3 path and loop closure have been inspected but not executed in this project.
7. Dynamic-region masking method and dataset are not selected yet; this decision waits for the static baseline output.
8. Keep this repository isolated from earlier Omni-View workspaces. Git must exclude weights, PDFs, outputs, scratch files, and independent third-party clones.

## Exact next steps

1. Download the official Pi3X checkpoint through Windows because WSL Hugging Face access is unavailable.
2. Run Pi3X on exactly the same 8 frames and 100K pixel limit.
3. Compare Pi3 and Pi3X confidence, trajectory, point count, visual artifacts, runtime, and peak memory before changing frame count.
4. Only after the fixed-input comparison, decide whether the laptop can support a 10/12-frame window.

## Handoff status

- Main repository remote: `https://github.com/alexy-dot/GR3D.git` (`origin`).
- Repository visibility: public, explicitly confirmed by the user before the first push.
- Main branch: maintained as a focused GR3D reconstruction workspace.
- Codex Remote troubleshooting: intentionally paused.
