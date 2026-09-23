# Frame-paired motion-evidence hardening

- Date: 2026-09-23
- Scope: GitHub Issue #2, CPU post-processing only
- Input model/runtime artifacts: existing saved Pi3/SAM outputs; no model inference rerun

## Problem and policy

The previous classifier filtered invalid 3D states and then paired the remaining object
displacements with a positional jitter array. A valid-invalid-valid sequence could therefore
crash on a count mismatch or, if edited manually, pair evidence from different intervals.

Background evidence now uses explicit `from_frame`/`to_frame` records. The classifier uses
only the record whose frame pair exactly matches each object displacement. Any invalid 3D
state makes the bounded track uncertain; missing or invalid background evidence also returns
uncertain. These paths preserve state warnings, point counts and the required/missing frame
pairs. Duplicate interval records remain a malformed-input error rather than ambiguous
evidence.

The CLI accepts `track_3d_states.json` and `background_jitter.json` as separate saved inputs.
Legacy artifacts containing both `background_jitter` and `interval_support` are converted
to explicit records only when their counts agree.

## Validation

The synthetic suite covers:

- fully valid static and coherent dynamic tracks with unchanged decisions;
- valid -> invalid -> valid through lift, background estimation and classification;
- empty mask after erosion and insufficient 3D support;
- several consecutive invalid states with warnings and support counts retained;
- missing interval evidence, wrong frame pairs and missing background support;
- conversion of the previous saved-evidence schema;
- direct CLI handoff between separate state and background files.

The complete reconstruction suite passed 84 tests. Syntax compilation passed for the two
changed runtime scripts.

## Existing-control replay

The OSI 0000 parked-bicycle negative control was replayed from saved artifacts:

- states SHA-256: `de097ca7769a11d06519c60648a68c601e970401728175befcd8b0a1d6c50549`
- legacy background artifact SHA-256: `0b575974c3b43456c9ef7eb0ab147381c099a47342ed4795a2b38e714ee065a8`
- revised classification SHA-256: `b23b5dda7e17d8d62f5f9d778559fbe38be98a1724041cf1390384854b277dad`

The legacy artifact converted to exact intervals `(1,2)` and `(2,3)`. The decision remained
`static` at thresholds 2, 3 and 5, with normalized motion 0.4522 and 0.4659. Thus the repair
did not change this fully supported control.

## Limits

No Pi3, SAM 2, CoTracker or GPU inference was rerun. The replay has no invalid Pi3-linked
state; invalid-state behavior is established by synthetic end-to-end tests. This repair
prevents unsupported classifications and interval mispairing, but does not improve tracking,
recover missing geometry or validate robustness beyond the existing bounded pilots.
