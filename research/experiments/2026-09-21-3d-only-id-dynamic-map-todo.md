# 3D-only scene IDs and static/dynamic map TODO

- Date: 2026-09-21
- Status: Phase A and Phase-B input preparation implemented; MLLM evaluation and dynamics not started
- Immediate target: add object IDs only to the reconstructed 3D representation while leaving every original input frame unchanged

## Research question

Can a spatial-reasoning MLLM use IDs that exist only in a unified 3D scene representation, avoiding GR3D-style repeated ID projection onto every source frame?

The first experiment must separate two questions:

1. Do 3D-only IDs improve reasoning over no-ID 3D views?
2. Can the model associate an object in an unchanged source frame with the correct 3D ID when several objects look similar?

## Representation contract

Use three namespaces:

- `S###`: persistent static scene instance.
- `D###`: dynamic object track with time-indexed states.
- `U###`: uncertain component whose motion state or instance identity is not sufficiently supported.

For the first implementation, create only `S###` candidates within one bounded reconstruction run. Existing semantic `component_id` values are component identifiers, not verified physical-instance identities and not persistent IDs across reruns or windows.

Each scene-instance entry must contain:

```text
scene_id
source_component_id
semantic_label and semantic_name
center_xyz
bbox_min_xyz and bbox_max_xyz
voxel_count and point_count
mean_voxel_purity
max_view_support and multiview_voxel_ratio
motion_state: static | dynamic | uncertain
```

## Phase A: deterministic 3D-only static IDs

1. Add a dedicated scene-instance export step that reads `objects.json` and writes `scene_instances.json`.
2. Assign IDs deterministically for identical inputs. Use a documented sort key such as semantic name followed by quantized `center_xyz`; do not rely only on traversal order.
3. Render IDs at component centers or box centers in the semantic XY/XZ/YZ views.
4. Keep original source frames byte-for-byte unchanged.
5. Avoid unreadable label overlap. The render may use leader lines or a side legend, but every visible ID must map to exactly one scene-table record.
6. Add tests for deterministic assignment, duplicate IDs, missing table entries, and render/table disagreement.

Acceptance criteria:

- repeated execution on identical inputs produces the same IDs and table order;
- all rendered IDs occur exactly once in `scene_instances.json`;
- no source image is modified or copied with an added ID overlay;
- the output manifest records source hashes, object-table hash, render parameters, and code revision;
- documentation calls these semantic component candidates, not ground-truth instances.

## Phase B: controlled MLLM input conditions

Prepare conditions that change one representation variable at a time:

1. `raw_frames_only`;
2. `raw_plus_rgb_canonical_views`;
3. `raw_plus_no_id_semantic_views`;
4. `raw_plus_3d_only_ids`;
5. `raw_plus_3d_ids_and_representative_crops`;
6. optional small-scale full image-side ID condition as a diagnostic upper/control condition.

The representative-crop condition does not redraw every frame. It selects at most one declared best-view crop for each 3D ID and places it in an object catalog. This condition tests whether appearance evidence is required for 2D-to-3D correspondence.

Package validation must check:

- answers never enter model prompts;
- question, frame, view, object-table, and crop hashes are recorded;
- all conditions use the same questions and underlying sampled frames;
- no-ID and 3D-ID conditions differ only in the declared 3D ID/table information;
- baked-in OSI visual tags are either removed in a separate controlled condition or explicitly reported as a confound.

Report category-level accuracy and correspondence-specific failures, not only the overall score.

## Phase C: dynamic-object pilot

Do not begin with dense 4D reconstruction. Use a hybrid representation:

```text
StaticMap = persistent geometry + S### instances
DynamicTracks = D### + {time, center_xyz, bbox_xyz, confidence}
Uncertain = U### components retained for audit but not fused as permanent objects
```

Minimum dynamic pilot:

1. Select one short clip with an unambiguously moving person or vehicle.
2. Obtain a temporally associated 2D mask/track for each candidate object.
3. Use Pi3 point/pixel observations and camera poses to estimate the candidate's 3D center in the common world frame at every visible time.
4. Measure residual 3D motion after camera compensation and normalize it by a declared scene scale.
5. Mark a candidate static only with sufficient repeated-view support and low residual motion; mark coherent displacement as dynamic; otherwise retain `U###`.
6. Exclude confirmed dynamic points from the persistent static cloud and store their time-indexed states separately.

Do not classify motion from semantic category alone. A parked vehicle can be static during the observation interval, while normally static furniture may move.

The detailed executable design, artifact schemas, model candidates, validation gates, RTX 4060 execution policy, controlled comparisons, and ordered checklist are maintained in:

- `research/experiments/2026-09-21-dynamic-object-tracking-pilot-todo.md`

That document does not change the current priority gate: repair GitHub Issue #1 and finish the fixed-protocol static-ID MLLM ablation before treating the dynamic pilot as the active experimental result.

## Phase D: long-video persistence

After the bounded static and dynamic pilots work, extend to overlapping reconstruction windows.

Required association evidence across windows:

- common world-frame geometry and overlap alignment;
- semantic compatibility;
- 3D center distance and box overlap;
- appearance evidence from representative crops or embeddings;
- explicit unmatched/new/retired states.

Per-run `component_id` values must never be treated as cross-window persistent IDs. Record ID switches, false merges, fragmentation, and unmatched components separately.

## Current boundary

This TODO does not claim:

- correct physical instance segmentation;
- verified metric scale;
- complete dynamic reconstruction;
- stable IDs across long videos;
- improved MLLM reasoning before the controlled evaluation runs.

## Implementation checkpoint

The deterministic single-run static-candidate representation and controlled input packaging are implemented and recorded in `2026-09-21-3d-only-id-static-pilot.md`. Dynamic tracking remains intentionally gated on the fixed-protocol MLLM ablation described above.

## First executable step on Windows

Use the existing house semantic-voxel output and OSI scene 0000 output. Implement deterministic `scene_instances.json` export plus ID rendering and tests before downloading more data or running another reconstruction model.
