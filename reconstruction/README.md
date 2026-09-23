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

By default the runner imports Pi3 from `third_party/Pi3` under this repository.
When the independent clone lives elsewhere, point both imports and revision
recording at the actual clone before running:

```bash
export PI3_ROOT=/absolute/path/to/Pi3
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

## Semantic coverage audit

Render evaluation-only target-class overlays and per-frame coverage statistics without object IDs:

```bash
python reconstruction/render_semantic_audit.py \
  --run-directory outputs/run \
  --semantic-labels outputs/run/semantics.npy \
  --semantic-metadata outputs/run/semantics.json \
  --output outputs/run/semantic_audit \
  --target-classes bicycle,minibike,person
```

The resulting presence statistics measure semantic-model coverage only. They do not establish instance identity or 2D-to-3D correspondence.

For a practical extension, closely related unstable labels may be merged before voxel voting with `--label-remap`. Keep this result separate from the unmodified ADE20K/GR3D comparison baseline:

```bash
python reconstruction/build_semantic_voxels.py \
  --observations outputs/run/point_observations.npz \
  --semantic-labels outputs/run/semantics.npy \
  --output outputs/run/voxels_two_wheeler \
  --voxel-size 0.03 \
  --min-component-voxels 10 \
  --label-remap research/configs/osi-two-wheeler-remap.json
```

## Prepare and validate a no-ID OSI pilot package

After the reconstruction and semantic view sets are complete, assemble the six
Phase-B evaluation conditions with answers stored separately:

```bash
python reconstruction/prepare_osi_pilot_package.py \
  --run-directory outputs/osi_0000_pi3_8f_exact \
  --qa-json tmp/osi0000_qa.json \
  --layout-views outputs/osi_0000_pi3_8f_exact/voxel_views_layout \
  --ade-object-views outputs/osi_0000_pi3_8f_exact/voxel_views_objects_c10 \
  --merged-object-views outputs/osi_0000_pi3_8f_exact/voxel_views_objects_c10_tw \
  --scene-id-views outputs/osi_0000_pi3_8f_exact/scene_instances_c10/views \
  --scene-instances outputs/osi_0000_pi3_8f_exact/scene_instances_c10/scene_instances.json \
  --representative-crops outputs/osi_0000_pi3_8f_exact/scene_instances_c10/representative_crops \
  --output outputs/osi_0000_first_version_package \
  --scene-id 0000

python reconstruction/validate_osi_pilot_package.py \
  outputs/osi_0000_first_version_package
```

The validator rejects missing or modified files, absolute manifest paths, object-ID
suffixes in no-ID questions, answer leakage, and unexpected question/condition counts.
Generated packages stay under `outputs/` and must not be committed.

## Add deterministic IDs only to the 3D representation

Export object-layer semantic components as deterministic single-run `S###` candidates:

```bash
python reconstruction/export_scene_instances.py \
  --objects outputs/run/semantic_voxels/objects.json \
  --output outputs/run/scene_instances/scene_instances.json \
  --scene-id scene_name

python reconstruction/render_semantic_voxels.py \
  --voxels outputs/run/semantic_voxels/semantic_voxels.npz \
  --camera-poses outputs/run/camera_poses.npy \
  --objects-json outputs/run/semantic_voxels/objects.json \
  --scene-instances outputs/run/scene_instances/scene_instances.json \
  --output outputs/run/scene_instances/views \
  --layer objects --color-by component --style both
```

These IDs identify deterministic semantic-component candidates within one bounded run.
They are not verified physical instances and do not persist across reconstruction windows.
Original frames are neither copied with ID overlays nor modified.

Optionally extract one appearance crop per candidate without adding an ID to the pixels:

```bash
python reconstruction/extract_representative_crops.py \
  --run-directory outputs/run \
  --voxels outputs/run/semantic_voxels/semantic_voxels.npz \
  --objects outputs/run/semantic_voxels/objects.json \
  --scene-instances outputs/run/scene_instances/scene_instances.json \
  --output outputs/run/scene_instances/representative_crops
```

The crop catalog records source/crop hashes, support, bounding boxes, and quality warnings.
Any visual tags already baked into source pixels remain a declared confound.

## Detect video candidates without source-image annotation

The optional Phase-D1 front end uses torchvision Faster R-CNN ResNet-50 FPN v2
with a local COCO V1 checkpoint. It writes structured boxes and scores without
drawing IDs or boxes onto source images. Semantic category proposes candidates;
it does not determine their motion state.

```bash
python reconstruction/detect_video_candidates.py \
  input.mp4 outputs/candidate_detection \
  --checkpoint /path/to/fasterrcnn_resnet50_fpn_v2_coco.pth \
  --interval 3 --start-frame 57 --end-frame 147 --width 640 \
  --score-threshold 0.7 --classes person bicycle car motorcycle \
  --prompt-source-frame 111 --prompt-class person \
  --track-candidate-id T_AUTO_001
```

`selected_prompt.json` is compatible with `run_video_instance_tracking.py` when
the detector and tracker use the same frame bounds, interval and width. The
detection manifest records the video, checkpoint, frames, boxes, scores,
parameters, runtime and hashes. Candidate IDs are frame-local audit identifiers,
not persistent object identities.

Compare an automatically prompted track against a fixed manual-box control:

```bash
python reconstruction/compare_video_tracks.py \
  outputs/manual_track/track_manifest.json \
  outputs/automatic_track/track_manifest.json \
  outputs/automatic_track/manual_comparison.json
```

The comparison requires identical video and extracted-frame hashes and records
per-frame overlap, directional coverage, area ratio and normalized centroid shift.

To propagate several boxes initialized on the same sampled frame, pass a prompt
JSON containing a `tracks` list. The tracker shares one SAM 2 video state and writes
`multi_track_manifest.json` plus one backward-compatible
`track_manifest_<track_candidate_id>.json` per candidate. Each per-track manifest
can be passed directly to `align_track_to_pi3.py`.

## Run the manual dynamic-object pilot

Keep SAM 2 tracking and Pi3 reconstruction sequential on an 8 GiB GPU. The tracking
stream may be denser than the Pi3 stream, but both manifests must retain original source
frame indices and timestamps. Start from one manually confirmed box:

```bash
python reconstruction/run_video_instance_tracking.py \
  third_party/Pi3/examples/skating.mp4 \
  research/configs/dynamic-pilot-skating-manual-prompt.json \
  outputs/dynamic_skating_pilot/sam2_tiny \
  --checkpoint /path/to/sam2.1_hiera_tiny.pt \
  --sam2-revision 2b90b9f5ceec907a1c18123530e92e794ad901a4 \
  --model-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --interval 3 --width 640

python reconstruction/align_track_to_pi3.py \
  outputs/dynamic_skating_pilot/sam2_tiny/track_manifest.json \
  outputs/dynamic_skating_pilot/pi3_8f_observations \
  outputs/dynamic_skating_pilot/aligned_track

python reconstruction/lift_tracks_to_3d.py \
  outputs/dynamic_skating_pilot/pi3_8f_observations \
  outputs/dynamic_skating_pilot/aligned_track/track_manifest.json \
  outputs/dynamic_skating_pilot/lifted_track \
  --min-points 20 --erode-pixels 1

python reconstruction/estimate_background_jitter.py \
  outputs/dynamic_skating_pilot/pi3_8f_observations/point_observations.npz \
  outputs/dynamic_skating_pilot/lifted_track/selected_point_indices.npz \
  outputs/dynamic_skating_pilot/background_jitter.json
```

The background artifact stores explicit `from_frame`/`to_frame` intervals. Missing
background support is saved as invalid evidence rather than raising. Pass the two saved
artifacts directly to the classifier; no combined JSON or manual editing is required:

```bash
python reconstruction/classify_track_motion.py \
  outputs/dynamic_skating_pilot/lifted_track/track_3d_states.json \
  outputs/dynamic_skating_pilot/motion_classification.json \
  --background-evidence outputs/dynamic_skating_pilot/background_jitter.json \
  --thresholds 2 3 5
```

The classifier pairs every object displacement with the exact same background frame
interval. Invalid 3D states, occlusion gaps, missing intervals, or invalid background
support produce an auditable `uncertain` result with preserved warnings and point counts.
Assign `D###` only after the motion decision. `build_static_dynamic_map.py` writes a
derived static PLY and separate dynamic/uncertain observations without modifying the raw
Pi3 run.

Render the decision from identical views:

```bash
python reconstruction/render_dynamic_tracks.py \
  outputs/dynamic_skating_pilot/pi3_8f_observations/point_observations.npz \
  outputs/dynamic_skating_pilot/lifted_track/selected_point_indices.npz \
  outputs/dynamic_skating_pilot/lifted_track/track_3d_states.json \
  outputs/dynamic_skating_pilot/motion_classification.json \
  outputs/dynamic_skating_pilot/trajectory_render \
  --entity-id D001
```

The resulting ID is valid only within the bounded clip. Background-normalized Pi3 motion
is not a physical metric trajectory or proof of cross-window identity.

Optionally add CoTracker3 point evidence as a separate diagnostic. Sample points from
the eroded target-mask interior and a surrounding background ring, then explain local
camera/background motion with a robust affine fit:

```bash
COTRACKER_ROOT=/path/to/co-tracker \
python reconstruction/run_cotracker_residual.py \
  outputs/automatic_track/track_manifest.json \
  outputs/automatic_track/cotracker_residual \
  --checkpoint /path/to/scaled_offline.pth \
  --cotracker-revision 82e02e8029753ad4ef13cf06be7f4fc5facdda4d
```

The output records point visibility, invalid intervals, raw and residual displacement,
checkpoint size/hash, and the explicitly declared source revision. A local Git checkout
is checked against that declaration when available; archive installs remain reproducible
without a `.git` directory. It is auxiliary 2D motion evidence, not an
object-identity source or a replacement for Pi3-linked 3D motion classification.

Validate and summarize a manually reviewed track-failure annotation file:

```bash
python reconstruction/score_track_failure_annotations.py \
  research/annotations/2026-09-23-osi0000-track-failure-labels.json \
  outputs/track_failure_scores.json
```

The scorer requires complete, non-overlapping frame-range coverage and reports observed
association, ID-switch, fragmentation, occlusion and mask-failure counts. These labels
remain a declared review subset rather than benchmark ground truth.

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

Prepare the fixed answer-blind request manifest after validating the package:

```bash
python reconstruction/prepare_mllm_evaluation.py \
  outputs/osi_0000_phase_b_package_hardened \
  outputs/osi_0000_mllm_eval/requests.json
```

This creates 60 protocol-v2 requests for the current six-condition, ten-question pilot.
Only the two declared 3D-ID conditions receive the answer-blind scene table; representative
crops carry an explicit `S###` association. Raw and no-ID conditions receive no scene-table
context. The request generator records ordered image paths and hashes but never reads
`ground_truth.json`.

The DashScope runner writes predictions separately from a required `.run.json` sidecar.
The sidecar binds a resumable run to the request and package hashes, model, endpoint,
decoding settings, and protocol version. A metadata mismatch or duplicate request ID stops
the run instead of mixing predictions. Use a new output path for protocol v2; protocol-v1
predictions remain an image-only ID-render result.

After a declared model adapter has saved one `{"request_id": ..., "answer": ...}` prediction
per request, score it without claiming the diagnostic numerical metrics are the official
OSI evaluator:

```bash
python reconstruction/score_mllm_evaluation.py \
  outputs/osi_0000_mllm_eval/requests.json \
  outputs/osi_0000_mllm_eval/predictions.json \
  outputs/osi_0000_phase_b_package_hardened/ground_truth.json \
  outputs/osi_0000_mllm_eval/scores.json
```

For a deliberate partial rerun, repeat `--condition` once per included condition. The
predictions file must still contain exactly one result for every selected request and no
others.
