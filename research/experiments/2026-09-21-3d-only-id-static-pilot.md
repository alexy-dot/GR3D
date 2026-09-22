# 3D-only static candidate ID pilot

- Date: 2026-09-21
- Status: Phase A and hardened Phase-B verified; first single-scene MLLM evaluation complete
- Scope: deterministic `S###` candidate IDs inside one bounded reconstruction run; no source-frame ID overlays and no cross-window persistence claim

## Implementation

`reconstruction/export_scene_instances.py` reads an existing semantic `objects.json`, excludes layout surfaces by the repository's declared presentation policy, and sorts remaining components by semantic name, semantic label, quantized 3D center, and source component ID. It exports contiguous `S001`, `S002`, ... records with geometry, semantic evidence, voxel/point counts, purity, view support, source hash, sort parameters, and code revision.

`reconstruction/render_semantic_voxels.py --scene-instances ...` adds the IDs only to the canonical 3D semantic views. Each view uses component-center labels with leader lines plus a side legend. Its manifest stores the scene-table hash, complete rendered-ID list, alignment, rendering parameters, and `source_frames_modified=false`.

`reconstruction/extract_representative_crops.py` selects at most one best-view source crop per `S###` using the frame with the most retained point observations for that component. It preserves the source pixels, records source and crop hashes, and flags small or weakly supported crops.

## Verified runs

| Scene | Component setting | 3D candidate IDs | Crop warnings | Determinism |
|---|---|---:|---:|---|
| Pi3 house | voxel 0.03, 30 voxels/component, object layer | 14 | 4/14 | repeated export SHA-256 identical |
| OSI 0000 | voxel 0.03, 10 voxels/component, object layer | 17 | 4/17 | repeated export SHA-256 identical |

Earlier deterministic tables were byte-identical under repeat runs. After provenance hardening and removal of the unsupported static assumption, the regenerated OSI 0000 scene table SHA-256 is `f0098fe3637411e3f583f32733f8818fe17a07f305e800cc51a6d243b433210c`. The table records the exact exporter file hash so copied-script runs remain auditable even when the surrounding WSL checkout is older.

Visual inspection confirmed that ID labels occur in the new 3D views and not on generated copies of source frames. The OSI object view remains sparse and some labels are close together, but the side legend remains readable and preserves the one-to-one mapping.

## Controlled OSI package

The ignored hardened package `outputs/osi_0000_phase_b_package_hardened` contains 51 hashed files, 10 questions, and six conditions. Its package-manifest SHA-256 is `900e5281721496aa085d65db338fd67f290cc9781ab598422260c57a300698d3`:

1. raw frames only;
2. raw frames plus RGB canonical views;
3. raw frames plus no-ID semantic views;
4. raw frames plus 3D-only `S###` views and scene table;
5. the same 3D-only ID representation plus one representative crop per candidate;
6. tagged-question diagnostic control.

Validation enforces identical sampled frames and questions across experimental conditions, answer separation, no-ID/ID render equality except annotations and the scene table, reconstruction/video/voxel/object/camera provenance, scene-table/crop component and semantic identity, correct sampled versus original frame indices and timestamps, crop hashes, and an exact closed set of declared files. Package creation refuses a non-empty output directory so stale artifacts cannot be silently mixed into a run.

## Limitations and decision

- `S###` denotes a semantic connected-component candidate, not a verified physical instance.
- Unmeasured candidates use `motion_state=uncertain` with `motion_evidence=not_measured_phase_a_static_candidate`; no static/dynamic claim is made without measurements.
- Four crops in each pilot trigger small-dimension or low-support warnings.
- OSI number tags baked into source pixels remain visible in some crops and are a correspondence confound. This crop condition must not be described as tag-free.
- The first fixed-protocol Qwen3-VL result is recorded in `2026-09-22-dashscope-mllm-run.md`. On OSI 0000, 3D-only IDs matched raw-only exact accuracy (0.10) rather than improving it; no-ID semantic views reached 0.20. Treat this as a single-scene diagnostic, not a general conclusion.
