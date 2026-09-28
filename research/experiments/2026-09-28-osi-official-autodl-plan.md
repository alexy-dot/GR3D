# OSI-Bench official evaluation plan on AutoDL

## Objective

Measure whether the project representation improves official OSI-Bench scores while
keeping model, question subset, original visual evidence, decoding, and scorer fixed.
The benchmark score answers whether the representation helps; map-quality audits explain
which geometric or tracking component caused the change.

## Fixed comparison

1. `A`: official raw-video input.
2. `B`: the same original evidence plus an unfiltered Pi3 map.
3. `C`: the same original evidence plus the dynamic-filtered static map.
4. `D`: the same original evidence plus the static map, `D###` trajectories, and the
   answer-blind scene table.

The official 32-frame baseline is first reproduced unchanged. The A--D method ablation
then uses the same source frames in every condition; it must not compare an 8-frame
enhanced condition directly against a 32-frame raw condition and attribute the difference
only to 3D information.

## Compute stages

### Stage 0: one-scene smoke test

- Scene: `0000`, retained because the complete local pipeline and prior diagnostic result
  are already auditable.
- Purpose: verify official dataset loading, Qwen2.5-VL-3B inference, result serialization,
  and official scoring. It is not a research result.

### Stage 1: fixed cross-scene pilot

- Select 12 complete scenes with seed `20260928`, always including scene `0000`.
- Keep every question belonging to each selected scene. Sampling complete scenes avoids
  reconstructing a different video for nearly every randomly sampled question.
- Require coverage of all released categories and record the selected scene IDs,
  question counts, source metadata hash, and downloaded-video hashes.
- The frozen selection from official metadata SHA-256
  `ca5556d6f39e15e04236a321f36a756b8479065677bdc782303c285507c336cb` is:
  `0000, 1546, 1452, 1328, 1973, 0592, 0684, 1729, 0755, 1373, 1906, 0626`.
  It contains 141 questions and covers all nine categories in the released parquet.

### Stage 2: confirmation subset

- Expand to approximately 30 complete scenes only after Stage 1 is reproducible.
- Freeze all selection and model settings before comparing A--D.

### Stage 3: full benchmark

- Run all released questions only after the official baseline, adapter, and subset results
  are stable. The official repository requires at least 160 GB for its default full
  download; reserve additional space for model weights, frames, maps, and run artifacts.

## AutoDL instance

- Preferred first instance: RTX 3090 or RTX 4090 with 24 GB VRAM.
- Python 3.10 and a CUDA PyTorch image compatible with the pinned project dependencies.
- Pilot data disk: at least 100 GB. Full evaluation: at least 300 GB usable space.
- Do not place datasets, weights, or outputs in Git.

## Setup

```bash
git clone https://github.com/alexy-dot/GR3D.git
cd GR3D
git switch codex/osi-official-eval-prep

git clone https://github.com/mingrui-wu/OSI-Bench.git third_party/OSI-Bench
git -C third_party/OSI-Bench checkout e391fab13eecf6e0b03607ff84f38f7ed5a3ec87

conda create -n osi-eval python=3.10 -y
conda activate osi-eval
pip install -e third_party/OSI-Bench/VLMEvalKit
pip install remotezip pandas pyarrow
```

Prepare the one-scene smoke-test package without downloading the four complete ZIP files:

```bash
python reconstruction/prepare_osi_official_subset.py \
  /root/autodl-tmp/osi_subset_smoke \
  --scene-count 1 \
  --include-scene 0000 \
  --seed 20260928 \
  --download-videos
```

Run the generated official configuration:

```bash
cd third_party/OSI-Bench/VLMEvalKit
python run.py \
  --config /root/autodl-tmp/osi_subset_smoke/osibench_subset_config.json \
  --work-dir /root/autodl-tmp/osi_official_smoke_outputs
```

After the smoke run succeeds, prepare the frozen 12-scene pilot by changing only
`--scene-count 12` and the output directory. Do not change the seed, model, frame count,
or scorer between conditions.

## Required evidence per run

- GR3D and official OSI-Bench Git revisions;
- AutoDL GPU, CUDA, PyTorch, Python, and available-disk report;
- subset manifest and metadata/video hashes;
- exact model and checkpoint revision;
- frame count, decoding settings, condition, elapsed time, peak VRAM, and failures;
- raw predictions and official category-level score output;
- no benchmark answer used during scene selection, reconstruction, or prompt generation.

## Stop conditions

- Stop before method comparison if the official raw baseline cannot be reproduced on the
  smoke scene.
- Stop and fix provenance if the same request can resume under mismatched subset, model,
  protocol, or checkpoint metadata.
- Do not scale past 12 scenes if any condition uses different questions, source frames,
  model settings, or scoring code.

## Status on 2026-09-28

The official raw-video path is reproduced. On the frozen 12-scene, 141-question pilot,
Qwen2.5-VL-3B-Instruct scored `0.240426` (`24.0426/100`), close to the paper's full-set
`24.2` and therefore sufficient as an implementation sanity check. The run took 1,340
seconds and peaked at 15,516 MiB VRAM on an RTX 4090.

Condition-B code is implemented but GPU execution is pending. It now:

- extracts the exact Qwen fixed-32 source indices with the Decord backend and saves a
  hash-closed frame manifest;
- makes Pi3 consume that explicit frame manifest without resampling;
- validates the chain from original video to extracted frames, Pi3 run, point cloud, and
  camera-gravity canonical renders;
- launches the pinned official evaluator without editing its scorer, preserving the
  original video and adding only the three unfiltered Pi3 views.

The next experiment is only scene `0000`. First try all 32 frames at 100,000 pixels per
frame. If Pi3 raises CUDA OOM, retain the same 32 frame indices and reduce only
`--pixel-limit` to 70,000. Do not process the other 11 scenes until the ten-question A/B
comparison completes and its prompt/output artifacts have been inspected.
