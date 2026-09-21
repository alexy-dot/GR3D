# First-version handoff: no-ID 3D scene views on OSI-Bench

Date: 2026-09-20

## What is ready

The first research package is now reproducible on official OSI-Bench scene `0000`:

```text
official outdoor MP4
-> 8 exact sampled frames spanning the sequence
-> original Pi3 geometry and camera poses
-> camera-gravity-aligned RGB point-cloud views
-> Mask2Former ADE20K labels
-> 3D semantic voxel components
-> separate layout and object XY/XZ/YZ views
-> no-ID evaluation package with QA/answers separated
```

Original Pi3 is retained as the GR3D-comparable geometry baseline. The optional bicycle/minibike-to-`two_wheeler` remap is a practical extension and is always reported separately.

## Current evidence

- CUDA baseline: 522,620 retained points, 5.51 GiB peak allocated GPU memory.
- Orientation consistency: camera-up disagreement median 0.25 degrees, P90 1.68 degrees, maximum 4.71 degrees.
- Trajectory shape: straightness 0.999818 and raw-XY heading change 1.09 degrees, consistent with the benchmark's `straight` answer.
- Answer-derived scale anchor: 14.9691 meters per Pi3 model unit from the benchmark's 28.1 m path-length answer. This calibrates the scene but is not independent metric validation.
- At voxel size 0.03 and 10 voxels/component, unmodified ADE20K gives 62 components and a 17-component object layer.
- The same two-wheelers alternate between `bicycle` and `minibike` across frames. Merging only those labels reduces their 3D components from 7 to 5 and the full component count from 62 to 60.

## Prepared comparison

`reconstruction/prepare_osi_pilot_package.py` now builds six auditable conditions:

1. `raw_frames_only`: sampled frames and questions with textual ID suffixes removed.
2. `raw_plus_rgb_canonical_views`: frames plus RGB point-cloud XY/XZ/YZ views.
3. `raw_plus_no_id_semantic_views`: frames plus layout/object views without scene IDs.
4. `raw_plus_3d_only_ids`: unchanged frames plus 3D `S###` views and scene table.
5. `raw_plus_3d_ids_and_representative_crops`: the same 3D-ID representation plus one source-pixel crop per candidate.
6. `tagged_question_control`: original tagged questions plus the original-ADE views.

Answers are stored separately in `ground_truth.json`. Every copied input is fingerprinted in `package_manifest.json`.

The hardened scene 0000 Phase-B package validator passed with 51 hashed input/metadata files, 10 questions, and all six conditions. It verifies equal no-ID/ID geometry and render settings, shared scene/run/video provenance, crop-to-component identity, original frame indices and timestamps, and a closed declared file set. The package itself remains under ignored `outputs/`; the repository stores the reproducible builders, validator, experiment record, and exact configuration rather than generated media.

## What this first version does not prove

- The public OSI release currently lacks raw LiDAR, IMU/GPS, calibration, timestamp, and reference-pose files, so sensor-grounded ATE, scale drift, and heading error cannot yet be measured.
- The 28.1 m answer is used as a scale anchor; it cannot also serve as an independent scale test.
- Semantic components are not verified object instances.
- Removing textual `(id: NN)` suffixes does not remove any number tags already baked into benchmark video pixels.
- MLLM inference and accuracy comparison have not yet run. The package is the verified input artifact for that next experiment.

## Recommendation for the next meeting

Present the first version as a feasibility result, not a final method claim:

> We can reconstruct an official OSI outdoor sequence into stable, axis-aligned 3D views and produce no-ID semantic object blocks without manually re-annotating every frame. The first pilot exposes semantic label flicker and non-metric-scale limitations, but both are recorded and partially controlled. The next decisive experiment is whether adding these 3D views improves OSI question accuracy over raw frames alone.

The proposed method should be judged by the planned downstream ablation, not by point-cloud appearance alone.

## 2026-09-21 representation update

The next approved pilot is now implemented: deterministic `S###` labels exist only in the aligned 3D object views and a machine-readable scene table. Source frames remain unchanged. House and OSI 0000 both pass deterministic export and table/render consistency checks.

The OSI package now contains six controlled conditions, including no-ID semantics, 3D-only IDs, and 3D-only IDs plus one representative crop per candidate. This prepares the static-ID ablation but does not supply an MLLM accuracy result. Some OSI crops contain benchmark number tags baked into the source pixels, so the crop condition is explicitly confounded rather than claimed as tag-free.
