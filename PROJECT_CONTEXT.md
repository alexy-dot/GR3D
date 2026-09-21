# Project Context

Last updated: 2026-09-21 (3D-only scene IDs approved as the next representation pilot)

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

The intended downstream research extension is deliberately different from full GR3D: do not continuously re-annotate object IDs in effectively unbounded real video. Instead, render the persistent but potentially coarse 3D map into canonical coordinate views (initially XY, XZ, and YZ projections with visible axes/grid), provide those renders together with a small task-relevant subset of original frames, and measure whether an MLLM can reason spatially and associate coarse 3D regions with real-image objects without explicit persistent IDs.

This extension is a hypothesis to test, not an assumed replacement for GR3D's ID-linked representation.

## Research background

The reading path was `OSI-Bench -> Omni-View -> GR3D`. This led to the current concern: indoor/static multiview methods do not directly resolve persistent reconstruction for real outdoor video containing camera motion, long duration, scale ambiguity, loop drift, and moving objects.

The previous Omni-View/OSI-Bench evaluation work is background context only and is not stored in this focused repository.

## Decisions and reasons

1. **Separate reproduction from extension.** Original Pi3 is used to reproduce the geometry interface closest to GR3D. Pi3X/Pi3XVO/VGGT-Long results must be labeled as practical extensions.
   After the controlled skating and house comparisons, original Pi3 is fixed as the default reconstruction baseline; Pi3X is not a drop-in replacement for evaluation runs.
2. **Do not assume physical metric scale.** Original Pi3 describes scale-invariant geometry. Pi3X provides approximate metric behavior, which still requires validation against known distances or trajectories.
3. **Do not write long-video alignment from scratch first.** Official Pi3 includes `Pi3XVO` with overlapping windows and Sim(3) alignment. VGGT-Long supports a Pi3 backend, disk-backed chunks, loop detection, and global optimization.
4. **Build a static map before dynamic 4D modeling.** Moving people/vehicles can create duplicate points and corrupt alignment. First measure the effect of masking them.
5. **Do not use codec motion vectors as the reconstruction core.** They may later help keyframe or motion-region selection but do not provide the required 3D geometry.
6. **Use Git handoff instead of Codex Remote.** Windows Codex Remote repeatedly fails while phone-to-Mac Remote works. Do not continue Remote debugging unless explicitly requested.
7. **Do not maintain persistent object IDs over unbounded video.** The expected annotation and data-association burden is incompatible with the target of long, real-world streams.
8. **Evaluate canonical 3D renders as the alternative interface.** The proposed downstream input is a small set of original frames plus axis-aligned XY/XZ/YZ renders of the reconstructed scene, optionally including color, camera frusta, coordinates, or depth. This must be labeled as an extension rather than exact GR3D.
9. **Treat object correspondence as an empirical question.** A model may infer that a coarse 3D cluster corresponds to an object in a source image from appearance and geometry, but this is not guaranteed without IDs. The evaluation must separately measure spatial reasoning and 2D-to-3D correspondence.
10. **Add IDs in the 3D representation without redrawing them on every source frame.** Static scene instances should receive deterministic `S###` IDs in the 3D views and object table. Original frames remain unchanged. This is a middle condition between no-ID views and full GR3D image-to-geometry ID projection.
11. **Represent dynamics separately from the persistent static map.** Do not fuse confirmed moving-object points into the static map. Use `D###` for time-indexed dynamic tracks and `U###` for components whose motion state is not yet supported by enough evidence. Semantic category alone must not decide whether an object is moving.

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
  - Records the local checkpoint SHA-256 and writes a structured failed manifest for non-OOM exceptions.
  - Optionally preserves filtered point-to-frame/pixel correspondence and dense depth maps for GR3D-style semantic back-projection and voxel fusion.
- `reconstruction/render_canonical_views.py`
  - Exports deterministic RGB XY/XZ/YZ orthographic projections with coordinates, grid, and optional camera path.
  - Records the point-cloud SHA-256, sampling seed, percentile bounds, and rendered/source point counts.
  - Supports raw coordinates or scene-independent `camera-gravity` alignment using robust consensus camera-up and deterministic horizontal heading.
  - Records the full alignment basis, origin, heading source, pose convention, and camera-up consistency; inconsistent orientation raises a manifest warning.
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
- Failure-manifest handling was validated with an intentionally empty checkpoint: the run recorded `status=failed`, the exception type/message, checkpoint hash, elapsed time, and peak GPU allocation.
- The official Pi3X checkpoint was downloaded through Windows and copied into WSL. SHA-256: `69972d6e1c4492cb4d737a84fe940e357087d81c52f5c9b7c160b49c1f41669a` (5,440,325,620 bytes).
- A fixed-input Pi3X comparison completed on the same 8 frames and pixel limit. It retained 730,395 points, used 6.14 GiB peak allocated GPU memory, and completed in 27.19 seconds.
- Pi3X retained 93.91% of candidate points versus Pi3's 44.13%, but visual orthographic inspection still showed regular ray/grid structure and dynamic-person artifacts. More retained points are therefore not treated as proof of better reconstruction.
- Deterministic XY/XZ/YZ export was validated on both Pi3 and Pi3X outputs at 250,000 rendered points. The large scale/orientation difference between their raw coordinate systems confirms that axis alignment and normalized presentation are required before downstream comparison.
- A second controlled comparison completed on Pi3's eight-view static `house` example (420x238). Original Pi3 retained 498,273 points in 51.04 seconds at 5.51 GiB; Pi3X retained 598,415 points in 27.30 seconds at 6.14 GiB.
- On the static indoor sample, original Pi3 produced the more compact and visually recognizable raw-coordinate layout. Pi3X showed substantially larger scene stretching and camera-path jumps despite being faster and retaining more points. Original Pi3 remains the faithful default baseline.
- Camera-gravity alignment was validated on both the static house and dynamic skating outputs. On the house sample the aligned side views make the floor approximately horizontal and walls vertical; median camera-up disagreement is 2.36 degrees. The largest single-view disagreement is 35.67 degrees and is retained as a diagnostic rather than hidden.
- The original-Pi3 house run was repeated with semantic-fusion observations enabled. Validation passed for 498,273 point observations spanning frames 0-7 and the complete 420x238 pixel domain; dense depth maps have shape `(8, 238, 420)`. The compressed observation file is 6.9 MiB and depth maps are 1.6 MiB.
- The missing GR3D semantic-object stage is now operational on the house sample. Official Mask2Former Swin-small ADE20K labels were fused into Pi3 points using voxel majority voting and same-label six-neighbor connectivity.
- At voxel size 0.03 model units, 2 points/voxel, and 8 voxels/component, the result contains 13,123 voxels and 123 components. Raising only the component threshold to 30 leaves 12,076 voxels and 48 components while retaining table, chair, door, and window components.
- No-ID component-colored XY/XZ/YZ renders are generated by `reconstruction/render_semantic_voxels.py`. They expose meaningful coarse structure but also visible fragmentation, so they are an experimental input rather than assumed superior representation.
- A formal staged quality policy is recorded in `docs/QUALITY_GATES.md`. The 30-voxel house condition has 0.9278 weighted semantic purity and 0.3750 multiview voxel ratio. Requiring two views per voxel reduced the map to 3,175 voxels/17 components and removed table/chair components, so single-view evidence is retained but explicitly marked lower confidence.
- Combined voxel-plus-box rendering is implemented. The house diagnostic showed that large structural boxes obscure small objects, so downstream inputs must separate a layout layer (floor/wall/road/building) from an object layer (furniture/vehicle/person etc.) rather than overlaying every box in one image.
- The layout/object split is now implemented and verified on the house pilot. At voxel size 0.03 and 30 voxels/component, the layout view contains 10,757 voxels from 34 components and the object view contains 1,319 voxels from 14 components. This is a curated presentation policy, not a formal ADE20K thing/stuff taxonomy.
- Official OSI-Bench metadata was inspected without downloading the full video collection. `data.parquet` contains 8,766 rows over 997 videos. The videos are stored in four approximately 20--21 GB ZIP files, but their remote central directories support extracting individual MP4s with HTTP Range requests.
- Two official OSI videos were run end to end with exact sampled-frame preservation. Scene 0963 is a 6.93-second indoor shopping corridor and is retained only as an integration test. Scene 0000 is a 27.07-second outdoor service-road sequence and is the first outdoor pilot.
- Outdoor scene 0000 used frames 0, 57, 114, 171, 228, 285, 342, and 399 at 420x238. Original Pi3 retained 522,620 points in 20.21 seconds at 5.51 GiB peak allocation. Point/pixel observation validation passed.
- Camera-gravity alignment on OSI 0000 has camera-up disagreement median 0.25 degrees, P90 1.68 degrees, and maximum 4.71 degrees, passing the internal orientation gate. This has not yet been compared with OSI IMU/GPS and is not evidence of ground-truth heading or scale.
- On OSI 0000, a one-variable voxel-size sweep at fixed 2 points/voxel and 30 voxels/component produced: size 0.02 = 28,555 voxels/44 components; 0.03 = 20,294/28; 0.05 = 8,635/18. Weighted voting purity remained 0.947--0.958 and median view support was 2, but these are internal consistency diagnostics only.
- At size 0.03 the OSI 0000 layout layer retains 20,150 voxels from 24 components, whereas the object layer retains only 144 voxels from 4 components. The current 30-voxel component cutoff is therefore too destructive for small outdoor objects and must be tuned separately from the house presentation setting.
- A fixed-size threshold sweep on OSI 0000 changed only `min_component_voxels`: threshold 5 produced 21,334 voxels/141 components and a fragmentation warning; 10 produced 20,817/62 without the warning; 20 produced 20,500/37; 30 produced 20,294/28. Threshold 10 retains 3 bicycle, 4 minibike, 1 person, and 3 signboard components, while threshold 30 retains only 2 minibike components among those targets.
- The threshold-10 object-only render contains 332 voxels from 17 components and is the best initial presentation tradeoff for this one outdoor scene. It is not a cross-scene default.
- A reusable trajectory audit is implemented in `reconstruction/analyze_camera_trajectory.py`. On OSI 0000, Pi3 path length is 1.877205 model units, endpoint displacement is 1.876864, straightness is 0.999818, and raw-XY heading change is 1.09 degrees. This agrees with the benchmark's `straight` trajectory answer.
- Using OSI question index 9's 28.1 m trajectory length as a single scale anchor gives 14.9691 meters per Pi3 model unit. This calibrates scene 0000 only and is not independent metric validation or a scale-drift measurement.
- `reconstruction/render_semantic_audit.py` now generates evaluation-only target-class overlays, a contact sheet, and per-frame pixel coverage without object IDs. On OSI 0000, bicycle is present in 6/8 sampled frames (9,323 pixels), minibike in 5/8 (12,182), and person in 8/8 (10,842).
- Visual inspection of the overlay audit suggests that corresponding parked two-wheeler regions alternate between ADE20K `bicycle` and `minibike` across views. No stable instance tracker is present, so this is evidence of cross-view semantic instability, not proof that the algorithm identified the same physical object.
- An optional, explicitly non-GR3D `--label-remap` extension is implemented in `build_semantic_voxels.py`. With only bicycle/minibike merged into `two_wheeler` at fixed voxel size 0.03 and threshold 10, total components drop from 62 to 60, two-wheeler components from 7 to 5, object-layer components from 17 to 15, and weighted purity rises from 0.9463 to 0.9488. The merge changes 5,552 of 13,653 matched observations.
- The first OSI evaluation package is now generated from scene 0000 by `reconstruction/prepare_osi_pilot_package.py` and independently checked by `reconstruction/validate_osi_pilot_package.py`. It contains 8 exact sampled frames, 12 canonical/semantic views, 10 no-ID questions, separated ground truth, and four declared comparison conditions. Validation passed for 24 hashed input/metadata files; the complete ignored package is 3.7 MiB.
- The four first-version conditions are raw frames only, raw plus unmodified-ADE layout/object views, raw plus the optional two-wheeler-merged views, and a tagged-question diagnostic control. Answers are not present in either question file. Source frames may still contain benchmark number tags baked into the pixels, which is recorded as a limitation rather than silently treated as no-ID imagery.
- `docs/FIRST_VERSION_HANDOFF.md` is the current senior-facing handoff. The first version is ready as a reproducible input/preprocessing package; downstream MLLM predictions have deliberately not been claimed.

### Paper-to-code coverage

- `docs/GR3D_REPRODUCTION_STATUS.md` maps every major GR3D paper stage to the current implementation.
- The completed work covers bounded frame selection, neural geometry reconstruction, camera parameters, and global point-cloud export.
- Implemented paper-adjacent stages now include camera-derived gravity alignment, Mask2Former ADE20K semantics, point/pixel label back-projection, voxel majority voting, same-label connectivity, axis-aligned component boxes, and separate no-ID layout/object renders.
- Paper stages intentionally not implemented include persistent image ID projection, textual indexed geometry, and downstream MLLM inference. Metric sensor alignment, occlusion filtering, and rigorous geometric primitive/instance validation are also still incomplete.
- The planned downstream extension will not implement persistent ID overlays by default. After geometry is stable, it will render canonical XY/XZ/YZ scene views and compare no-ID inputs against raw-frame and small-scale ID-linked controls.

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
2. Both original Pi3 and Pi3X weights are available locally and fingerprinted; they must remain outside Git.
3. Fixed-input comparisons are complete on both dynamic skating and static indoor house samples. Pi3X is denser and faster, but original Pi3 currently gives the more recognizable static indoor layout and remains the baseline.
4. The current baseline has not yet been compared against ground-truth trajectory or known metric distances.
5. `Pi3XVO` still retains the selected image tensor and merged dense points in memory; it is a medium-sequence validation step, not the final unbounded map store.
6. VGGT-Long's Pi3 path and loop closure have been inspected but not executed in this project.
7. Dynamic-region masking method and dataset are not selected yet; this decision waits for the static baseline output.
8. Keep this repository isolated from earlier Omni-View workspaces. Git must exclude weights, PDFs, outputs, scratch files, and independent third-party clones.
9. The static house views show that furniture can remain visually recognizable in the original-Pi3 raw projection, but it is still unknown whether an MLLM can reliably match those regions to objects in an original frame.
10. GR3D's ID ablation does not directly answer the proposed no-ID-render question: that ablation removes the explicit link between textual object geometry and image regions, whereas the proposed method supplies geometry visually and omits the indexed geometry text. A dedicated controlled evaluation is required.
11. The first OSI outdoor pilot proves pipeline integration only. The paper describes synchronized stereo, 32-beam LiDAR, and IMU/GPS and says additional raw multimodal videos will be released, but the current official GitHub/Hugging Face release contains only MP4 and QA metadata. It cannot currently provide sensor-grounded ATE, metric scale, or heading validation.
12. ADE20K semantic voting on OSI 0000 produces coherent large layout classes, but the current 30-voxel cutoff removes most small-object components. Internal purity must not be interpreted as correct object segmentation.

## Exact next steps

The executable task specification is `research/experiments/2026-09-21-3d-only-id-dynamic-map-todo.md`.

1. Implement deterministic static scene IDs (`S001`, `S002`, ...) from the existing `objects.json` components and render them only in 3D views. Do not modify the original input frames.
2. Export a machine-readable scene-instance table containing ID, semantic label, center, bounding box, confidence evidence, and the source component. Validate one-to-one consistency between rendered IDs and table entries.
3. Build controlled package conditions for raw frames, no-ID 3D views, 3D-only IDs, and 3D-only IDs plus one representative crop per instance. Keep the current full/tagged condition only as a diagnostic control.
4. Validate the static-ID implementation on both the house pilot and OSI scene 0000 before introducing tracking. Do not describe connected semantic components as verified physical instances.
5. After the static-ID ablation is prepared, run one fixed MLLM/evaluation protocol and report per-category accuracy. Keep `ground_truth.json` outside the model prompt.
6. Only then start a separate dynamic pilot: track object masks across frames, transform their 3D centers into the common world frame, classify them as static/dynamic/uncertain from camera-compensated motion evidence, and store dynamic objects as trajectories rather than fusing them into the static cloud.
7. Long-video overlapping-window association remains a later phase. Stable IDs across windows require explicit matching and must not be inferred from per-run component numbers.

## Handoff status

- Main repository remote: `https://github.com/alexy-dot/GR3D.git` (`origin`).
- Repository visibility: public, explicitly confirmed by the user before the first push.
- Main branch: maintained as a focused GR3D reconstruction workspace.
- Codex Remote troubleshooting: intentionally paused.
