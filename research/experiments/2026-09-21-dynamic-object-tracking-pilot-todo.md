# Dynamic-object tracking and hybrid 3D map TODO

- Date: 2026-09-21
- Status: Phase D0 manual-track pilot and one single-candidate Phase-D1 detector pilot complete on 2026-09-22; multi-candidate automation and full Phase D2 comparison remain open
- Priority gate: fix GitHub Issue #1 and complete the fixed-protocol static-ID MLLM ablation before claiming or prioritizing dynamic-scene results
- Initial scope: one short monocular clip with one unambiguously moving person or vehicle

## Research target

The current pipeline assigns `S###` IDs after semantic voting and 3D connected-component extraction. Those IDs name deterministic 3D component candidates, but they do not prove that observations in different frames belong to the same physical object.

The dynamic pilot must establish identity before static fusion:

```text
dense low-resolution video frames
-> per-frame object discovery and instance masks
-> cross-frame 2D track identity
-> lift tracked masks into Pi3 world coordinates
-> camera-compensated 3D motion evidence
-> static S### / dynamic D### / uncertain U###
```

The target representation is:

```text
StaticMap = persistent background geometry + supported S### candidates
DynamicTracks = D### + ordered {time, center_xyz, bbox_xyz, confidence}
Uncertain = U### candidates retained for audit and excluded from confident static fusion
```

Do not begin with dense 4D reconstruction. The first question is whether tracked masks plus Pi3 world-frame evidence can separate static-map content from moving-object trajectories.

## Important conceptual distinctions

The implementation must keep these four levels separate:

1. **Semantic class:** what category a pixel belongs to, such as `person` or `bicycle`.
2. **2D tracked instance:** which image region is believed to be the same candidate across adjacent frames.
3. **3D physical-instance candidate:** whether the tracked observations occupy a consistent world-space object.
4. **Persistent dynamic track:** how one physical candidate changes position through time and, later, across reconstruction windows.

Semantic equality alone is not identity. Spatial connectedness alone is not identity. An `S###` or `D###` ID may be assigned only after the declared evidence and validation step for its level.

## Proposed architecture

### Dual-rate frame processing

Use two frame streams rather than forcing one sampler to satisfy both tracking and reconstruction:

```text
dense or moderately sampled low-resolution frames
-> 2D instance tracking and motion continuity

sparse geometry-valid frames
-> Pi3 reconstruction and world-frame point observations
```

The dense stream preserves temporal continuity for masks and point tracks. The sparse stream remains bounded for the RTX 4060 and supplies 3D geometry. Track states are lifted into 3D only at frames shared with the Pi3 reconstruction run.

Do not infer a dynamic trajectory from the current eight widely separated frames alone if the object cannot be tracked reliably between them.

### Stage A: object discovery and video masks

Primary engineering candidate:

```text
Grounding DINO or an instance/panoptic detector
-> initial object boxes or masks
-> SAM 2 video mask propagation
-> stable within-clip track candidates
```

The first pilot may use one manually confirmed initial mask to validate the geometry and motion pipeline. Automation is a separate variable and should follow only after the manually initialized track succeeds.

The current ADE20K semantic Mask2Former output may be used as class evidence, but it must not be treated as an instance tracker. If a panoptic Mask2Former checkpoint is tested, record it as a separate front end and do not silently replace the current semantic baseline.

Save masks as lossless binary PNGs or compressed RLE plus a manifest. Do not draw IDs onto original source frames.

Minimum per-frame 2D track record:

```json
{
  "track_candidate_id": "T001",
  "sample_index": 0,
  "source_frame_index": 0,
  "timestamp_seconds": 0.0,
  "semantic_name": "person",
  "mask_path": "masks/T001/000000.png",
  "detector_confidence": 0.0,
  "tracker_confidence": 0.0,
  "visible": true,
  "occluded": false
}
```

The exact confidence fields may be nullable when a selected model does not expose a calibrated score. Do not invent confidence values.

### Stage B: optional point-track motion evidence

Use CoTracker3 or another declared point tracker as an auxiliary motion verifier, not as the sole object-identity source.

Track points from both:

- the interior of the candidate mask;
- a surrounding background ring that excludes other foreground masks.

This permits a residual-motion comparison:

```text
foreground point motion
- motion explained by the camera/background
= candidate object residual motion
```

Record point visibility and uncertainty. Reject tracks that are dominated by occlusion, boundary points, or background leakage.

### Stage C: lift tracked masks into Pi3 3D

For every frame shared by the 2D tracker and Pi3 observations:

1. load `point_observations.npz`;
2. select retained Pi3 points whose source pixels fall inside the tracked mask;
3. remove low-confidence points and optionally erode the mask before point selection to reduce boundary contamination;
4. compute robust world-frame statistics rather than trusting a single mean;
5. save a time-indexed state without fusing it into the static map.

Minimum 3D state:

```json
{
  "track_candidate_id": "T001",
  "sample_index": 0,
  "source_frame_index": 0,
  "timestamp_seconds": 0.0,
  "point_count": 0,
  "center_xyz_median": [0.0, 0.0, 0.0],
  "bbox_min_xyz": [0.0, 0.0, 0.0],
  "bbox_max_xyz": [0.0, 0.0, 0.0],
  "point_spread_mad": [0.0, 0.0, 0.0],
  "mean_pi3_confidence": 0.0,
  "quality_warnings": []
}
```

Use coordinate-wise medians or another declared robust center estimator. Use quantile bounds or an explicitly justified outlier filter for the 3D box. Preserve raw selected-point indices or a reproducible selection manifest for audit.

### Stage D: estimate camera-compensated motion

Pi3 observations are already expressed in the reconstruction's shared world coordinate system, but dynamic-object geometry may still be distorted. Therefore, classify motion relative to measured background jitter rather than using raw image displacement or a fixed meter threshold.

For each valid consecutive pair of object states:

```text
object_displacement_t = ||center(t+1) - center(t)||
background_jitter_t = robust displacement/noise statistic from static background support
normalized_motion_t = object_displacement_t / (background_jitter_t + epsilon)
```

Also record:

- displacement direction consistency;
- track duration and number of valid 3D states;
- object-point support and spread stability;
- mask IoU or tracker continuity;
- foreground-versus-background CoTracker residual when available;
- occlusion and truncation flags.

Do not assume Pi3 model units are meters. If a scale anchor is used, report both model-unit and calibrated values and do not reuse an answer-derived anchor as independent validation.

### Stage E: classify static, dynamic, or uncertain

Initial diagnostic policy:

- `static`: sufficient repeated-view support and residual motion consistently near the background-jitter distribution;
- `dynamic`: coherent displacement exceeds background jitter across multiple valid intervals;
- `uncertain`: insufficient support, conflicting direction, high point spread, tracker loss, occlusion, or scores between the static and dynamic regions.

Do not encode a universal hard threshold as a research conclusion. Implement configurable threshold sweeps over normalized motion, initially including `2`, `3`, and `5`, and report sensitivity. Require at least three valid 3D states for a confident dynamic decision in the first pilot; otherwise retain `U###`.

Semantic category must not determine motion. A parked vehicle may be static during the observed interval, and a normally static object may move.

### Stage F: build the hybrid map

After classification:

- retain supported `S###` observations in the static map;
- exclude confirmed `D###` observations from static fusion;
- store `D###` as ordered time-indexed states and render its trajectory separately;
- exclude `U###` from confident static fusion by default, while preserving it in audit outputs;
- never delete raw Pi3 outputs or source observations.

The first dynamic output should be a static-map point cloud plus a separate dynamic-trajectory JSON/render, not a dense deforming 4D mesh.

## Within-clip and cross-window identity

### Within one short clip

SAM 2 mask propagation or the selected video tracker provides the initial temporal identity. Validate it visually and, where possible, against a small manually checked mask/track subset.

### Across occlusion or later reconstruction windows

This is a later phase. Candidate matching should combine:

- semantic compatibility;
- appearance embedding similarity, for example DINOv2 or CLIP;
- predicted world-space position and motion;
- 3D box/point overlap when meaningful;
- temporal gap;
- mask or point-track continuity.

Use gated assignment followed by a declared matcher such as Hungarian assignment. Do not associate objects solely because their semantic labels match, and do not reuse per-window component numbers as persistent IDs.

## Proposed implementation files

Keep the dynamic branch modular:

```text
reconstruction/run_video_instance_tracking.py
reconstruction/lift_tracks_to_3d.py
reconstruction/classify_track_motion.py
reconstruction/build_static_dynamic_map.py
reconstruction/render_dynamic_tracks.py
research/configs/dynamic-pilot-*.json
```

Suggested responsibilities:

- `run_video_instance_tracking.py`: detector/prompt initialization, mask propagation, frame-index manifest and raw track outputs;
- `lift_tracks_to_3d.py`: mask/Pi3 intersection and robust per-time 3D states;
- `classify_track_motion.py`: background-jitter estimation, threshold sweep and `S/D/U` decision evidence;
- `build_static_dynamic_map.py`: static-map filtering without modifying raw reconstruction artifacts;
- `render_dynamic_tracks.py`: static map, camera path and time-colored `D###` trajectories for audit.

Do not combine all stages into one script. Each stage must accept saved artifacts so failures can be inspected without rerunning GPU models.

## Controlled experiment conditions

For the same video, Pi3 run and sampled geometry frames, compare:

1. `pi3_unfiltered`: current point cloud with all retained observations;
2. `semantic_category_masking`: remove categories commonly assumed dynamic, recorded as a weak baseline only;
3. `tracked_mask_filtering`: use cross-frame instance masks but no 3D motion decision;
4. `tracked_3d_motion_filtering`: use camera-compensated `S/D/U` decisions;
5. optional dynamic-specific method such as MonST3R, clearly labeled as a separate baseline rather than a drop-in Pi3 reproduction.

This comparison distinguishes the value of semantic priors, temporal identity and measured 3D motion.

## Measurements

### Tracking and identity

- track duration and visible-frame coverage;
- ID switches and fragmentation on a small manually checked subset;
- mask stability and failure after occlusion;
- correspondence between tracked 2D masks and selected Pi3 observations.

### Motion classification

- foreground displacement in model units;
- background jitter distribution;
- normalized motion score and threshold sensitivity;
- direction consistency;
- static/dynamic/uncertain decision with evidence;
- false removal of parked vehicles or other static objects.

### Static-map quality

- removed-point count and ratio;
- reduction of duplicate/ghost geometry around the moving target;
- effect on camera trajectory and static background completeness;
- visual comparison from identical canonical viewpoints;
- retained static-scene coverage.

### Downstream value

- dynamic-question and spatial-question accuracy;
- whether a separate `D###` trajectory improves answers over unfiltered Pi3 geometry;
- whether excluding dynamic points improves static-layout questions;
- GPU time, peak VRAM, storage, latency and frame count for every stage.

## Tests and validation gates

Required unit or synthetic tests:

- sampled-frame index and original-source-frame index remain distinct and correct;
- masks cannot be paired with a different video/run manifest;
- empty masks and insufficient 3D support produce `U###`, not crashes or false static labels;
- stationary synthetic tracks remain static after simulated camera motion;
- coherently moving synthetic tracks become dynamic after camera compensation;
- high-noise or conflicting tracks remain uncertain;
- dynamic point indices are absent from the filtered static map and present in the dynamic artifact;
- all source frames and raw Pi3 artifacts remain byte-for-byte unchanged;
- repeated CPU post-processing with identical inputs is deterministic;
- no benchmark answer or ground-truth motion label enters model prompts or selection logic.

Required artifact provenance:

- source video/frame hashes;
- tracking model and checkpoint identity;
- text prompts or initialization masks;
- Pi3 run-manifest and point-observation hashes;
- frame-index mapping;
- all thresholds and mask morphology parameters;
- device, runtime and code revision;
- warnings and failure state.

## RTX 4060 8GB execution policy

Run models sequentially and persist intermediate artifacts:

1. run video tracking at reduced resolution with a small/tiny SAM 2 configuration where available;
2. release the tracking model and GPU memory;
3. reuse the existing bounded Pi3 reconstruction or run Pi3 separately;
4. perform mask-to-3D lifting, motion classification and rendering on CPU where practical;
5. reduce tracking resolution or temporal batch size before renting cloud hardware;
6. use cloud GPU only for a validated need such as a heavier dynamic-3D baseline.

Do not load Pi3, SAM 2 and CoTracker simultaneously on the 8GB GPU.

## Stronger baseline escalation

If tracked masks are stable but Pi3 dynamic-object point maps are too distorted for usable centers or trajectories, evaluate a dynamic-scene method such as MonST3R on the same short clip. Record its exact code/checkpoint revision, compute cost, camera trajectory, dynamic/static separation and output compatibility.

Do not switch reconstruction backends merely because a qualitative render looks better. Compare the same input, trajectory evidence, dynamic-object artifacts and resource use.

Primary references to inspect before implementation:

- SAM 2: https://arxiv.org/abs/2408.00714
- Grounded SAM 2: https://github.com/IDEA-Research/Grounded-SAM-2
- CoTracker3: https://github.com/facebookresearch/co-tracker
- MonST3R: https://github.com/Junyi42/monst3r
- DynaSLAM static-map/dynamic-removal precedent: https://arxiv.org/abs/1806.05620

## Ordered execution checklist

### Gate 0: repair current Phase-B package

- [x] Close GitHub Issue #1 after the hardening commit is pushed.
- [x] Remove unsupported `motion_state=static` from unmeasured Phase-A candidates.
- [x] Revalidate the six-condition package and documentation (51 hashed files; 10 questions; 6 conditions).

### Gate 1: finish the static representation result

- [x] Prepare a deterministic answer-blind 60-request protocol for six conditions and ten questions.
- [x] Implement strict prediction completeness checks plus exact/category/numerical-error diagnostics.
- [x] Run one fixed MLLM protocol on the prepared static conditions.
- [x] Determine whether no-ID semantics, 3D-only IDs or representative crops provide measurable value on the first pilot scene.
- [x] Record correspondence-specific failures that motivate dynamic tracking.

### Phase D0: one manually initialized dynamic track

- [x] Select one short clip with one obvious moving object and relatively stable camera reconstruction.
- [x] Save dense low-resolution tracking frames with exact source indices and timestamps.
- [x] Manually confirm one initial object mask.
- [x] Propagate the mask through the clip.
- [x] Lift the track at Pi3 frames and render its world-space center trajectory.
- [x] Compare object displacement with background jitter.
- [x] Produce a declared `D###` or `U###` result with evidence.

### Phase D0 control

- [x] Run one manually initialized parked-object negative control and verify that measured-static points are retained; recorded as OSI 0000 `S001` with explicit occlusion fragmentation.
- [x] Run one second genuinely dynamic clip before treating the motion policy as reliable; recorded as OSI 0000 walking-person candidate `T003` with a run-scoped `D001` result.

### Phase D1: automated object discovery

- [x] Add Grounding DINO or a declared instance/panoptic front end; validated torchvision Faster R-CNN COCO V1 on the OSI 0000 walking-person interval without modifying source pixels.
- [ ] Track multiple candidates with SAM 2.
- [ ] Add CoTracker foreground/background residual evidence.
- [ ] Measure ID switches, fragmentation and occlusion failures.

### Phase D2: static-map filtering

- [x] Remove only confirmed dynamic observations from a derived static map.
- [x] Preserve raw Pi3 output unchanged.
- [ ] Compare unfiltered, semantic-only, tracked-mask and tracked-3D-motion conditions.

### Phase D3: later long-video persistence

- [ ] Add cross-window track association only after the short pilot is validated.
- [ ] Combine appearance, 3D position, motion prediction and temporal gating.
- [ ] Evaluate loop return, long occlusion and ID re-entry separately.

## Completion criterion for the first dynamic pilot

The first pilot is complete only when one moving candidate has:

- an auditable within-clip 2D track;
- at least three valid Pi3-linked 3D states or an explicit insufficient-evidence result;
- background-normalized motion evidence;
- a justified `D###` or `U###` decision;
- raw and filtered map artifacts;
- a time-indexed trajectory render/table;
- recorded compute use and failure modes;
- no unsupported claim of physical metric accuracy, full 4D reconstruction or long-term persistent identity.

