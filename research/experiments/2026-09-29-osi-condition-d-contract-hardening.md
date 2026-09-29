# OSI Condition-D identity, time, scale, and provenance hardening

## Purpose

Close the protocol gaps exposed by the scene-0000 Condition-B audit before spending more
GPU time. The corrected target is:

```text
unchanged official video
+ same-run dynamic-filtered static map
+ verified OSI-ID-linked 3D trajectories
+ question-time-conditioned evidence
+ optional independently validated metric scale
```

No new OSI score is reported here. This is a code and protocol hardening change.

## Gaps found

1. Project-local `T###` and `D###` identifiers were not an OSI numeric identity.
2. The exact Pi3 frame directory did not force alignment back to the tracking source video.
3. Lifted states did not include the corresponding camera center.
4. Trajectory CSV files omitted invalid 3D states, permitting accidental interpolation
   across an occlusion or support failure.
5. Classifications, selected point indices, static maps, and trajectory renders could be
   paired manually without a complete same-run hash chain.
6. Global trajectory renders were not question-time conditioned.
7. Original Pi3 units could still be confused with meters unless every downstream stage
   carried an explicit coordinate contract.
8. The official evaluator had no Condition-D adapter with answer-blind ID/time evidence.
9. Static-map and camera-pose declarations did not independently bind the actual output
   files at every consumer.
10. A manually edited official config could change the model or 32-frame sampler after
    the Condition-D package had been prepared.

## Implemented contract

- `external_identity` is separate from local candidate and motion-state IDs. It records
  namespace, scene, preserved display ID, canonical numeric ID, association method, and
  verification state.
- A verified OSI identity requires a track audit/contact sheet containing the initialized
  target. The source video frames are not modified.
- Exact-frame Pi3 alignment compares the tracking-video identity with the source-video
  identity recorded by the frame manifest. The Pi3 frame-manifest timestamp is the
  aligned trajectory time source; the tracking timestamp is retained only for audit.
- Pi3 runs declare model coordinates as non-metric. Lifted states contain object and
  camera centers at the exact sampled frame and source-video timestamp.
- Invalid states remain in `trajectory.csv`. Interpolation is rejected outside coverage,
  across invalid states, or across a gap larger than the configured threshold.
- Observation, selected-index, lifted-state, classification, audit, camera-pose,
  static-map, trajectory-table, question-text, and generated-image hashes are checked.
- The Condition-D package requires the static map to contain the same classified tracks
  and at least one confirmed dynamic trajectory. V1 accepts exactly one frozen scene so
  missing evidence cannot be hidden inside a larger subset.
- Question parsing uses only question text, category, IDs, and times. It does not read or
  serialize benchmark answers.
- Meter-valued fields are absent by default. Optional calibration must be positive,
  hash-bound to the Pi3 run and source artifact, use an allowed external source type, and
  declare independence from benchmark answers.
- The runtime independently revalidates the static-map PLY, camera poses, track audit,
  trajectory CSV, question evidence, generated images, source config, model, data path,
  and frozen 32-frame sampler before importing the official runner.

## Validation

- Python syntax compilation passes for all reconstruction scripts.
- 52 focused tests pass, including a subprocess trajectory-render integration test and
  complete synthetic Condition-D manifest tamper checks.
- The full discovery run reaches 109 tests; 104 pass and the same five pre-existing
  modules fail to import because the local Mac Python lacks `cv2`. No new test failure
  appears.
- CUDA, SAM 2, and full Pi3 inference were not rerun on the Mac.

## Remaining empirical work

1. Rerun the same official 32-frame Pi3 scene-0000 input with `--save-observations`.
2. Manually bind and audit OSI IDs `26` and `30`; automatic OCR/tag association remains
   future scalability work.
3. Generate the same-run filtered static map and Condition-D package.
4. Inspect every ID contact sheet and question-specific render before official scoring.
5. Report meter-valued results only if an independent scale source becomes available.
