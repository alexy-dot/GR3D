# 3D-only static candidate ID pilot

- Date: 2026-09-21
- Status: Phase A and Phase-B input preparation verified
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

With the final exporter fingerprint embedded, the house scene table SHA-256 is `3cb42a23c23074859eac18071a549187f896ce260e140fef0041511da45659fc`; OSI 0000 is `84a385c80c68f974151ce65c4cce74bd64510a008f71e3599f40e00a938ba2cb`. Pre-finalization repeat runs were byte-identical; the final tables additionally record the exact exporter file hash so copied-script runs remain auditable even when the surrounding WSL checkout is older.

Visual inspection confirmed that ID labels occur in the new 3D views and not on generated copies of source frames. The OSI object view remains sparse and some labels are close together, but the side legend remains readable and preserves the one-to-one mapping.

## Controlled OSI package

The ignored `outputs/osi_0000_phase_b_package` is 4.2 MiB and contains 47 hashed files, 10 questions, and six conditions. Its final package-manifest SHA-256 is `c2811654909a1a9552cc95cf1a1b586b7017ae60dea112356dba8bbba1bd0c4c`:

1. raw frames only;
2. raw frames plus RGB canonical views;
3. raw frames plus no-ID semantic views;
4. raw frames plus 3D-only `S###` views and scene table;
5. the same 3D-only ID representation plus one representative crop per candidate;
6. tagged-question diagnostic control.

Validation enforces identical sampled frames and questions across experimental conditions, answer separation, scene-table/crop one-to-one correspondence, crop hashes, declared file hashes, and no absolute package paths.

## Limitations and decision

- `S###` denotes a semantic connected-component candidate, not a verified physical instance.
- `motion_state=static` is a Phase-A assumption and is explicitly paired with `motion_evidence=not_measured_phase_a_static_candidate`.
- Four crops in each pilot trigger small-dimension or low-support warnings.
- OSI number tags baked into source pixels remain visible in some crops and are a correspondence confound. This crop condition must not be described as tag-free.
- No MLLM accuracy result exists yet. Do not start dynamic `D###/U###` work until the fixed-protocol static-ID ablation is run, as required by the approved plan.
