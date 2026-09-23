# OSI 0000 bounded track-failure annotation

- Date: 2026-09-23
- Status: complete for two visually assessable tracks
- Scope: single-reviewer within-clip audit of identity switches, fragmentation and occlusion; not independent tracking ground truth

## Fixed subset and protocol

The subset contains the two tracks whose targets remain visually distinguishable at the available 640-pixel tracking resolution:

- `T_AUTO_001`: central black-clothed walking person, 31 frames, source frames 57--147.
- `T002`: parked bicycle, 39 frames, source frames 57--171.

The three additional automatic person candidates were excluded before scoring because one is absent for long spans and two are extremely small. Counting them as zero-switch tracks would create unsupported precision.

Every frame was reviewed in a deterministic full-frame contact sheet with the saved SAM 2 mask rendered in red. The annotation stores compact, non-overlapping frame ranges for target visibility, association and mask quality. `score_track_failure_annotations.py` expands all ranges, rejects gaps or overlaps, validates event intervals and computes the summary. No persistent IDs were added to source images, and source pixels were not modified.

Labels are audit judgments from one reviewer, not independent ground truth or pixel-accurate mask annotations.

## Provenance

- Code revision: `a672cbd2060353d3ab63642feb55641f260315c9`, `code_dirty=false`
- Complete reconstruction tests: 74/74
- Annotation file: `research/annotations/2026-09-23-osi0000-track-failure-labels.json`
- Annotation SHA-256: `62c9174599da2df00271a88a66a9e4178908bb7157285921f9d2c41e68217e6d`
- Score output: `/home/dell/project/GR3D/outputs/2026-09-23-track-failure-scores.json`
- Score SHA-256: `ada9908673fd5a624747c535bb2fde51f1b2fc3010dda6d7f2e5e4dd93076132`
- `T_AUTO_001` track manifest SHA-256: `58cf680f319fe093764380ab6feb17408b88a0505559b3d072f7a078e2966ffe`
- `T002` track manifest SHA-256: `f87cecda6875102493aa6a9f963e69c639854709b28a0d3b870f543ab1932b5f`
- `T_AUTO_001` all-frame review sheet SHA-256: `105a5cd3f57cc352fb47009f2e323968ccf57a5261c47d5bb52f58a2c0fdc567`
- `T002` all-frame review sheet SHA-256: `c70f8270617c06068bcf907593b2e63712c4044985f032acb71758f35a8ed73a`

## Results

| Measurement | `T_AUTO_001` person | `T002` parked bicycle | Combined |
| --- | ---: | ---: | ---: |
| Reviewed frames | 31 | 39 | 70 |
| Identity-assessable frames | 31 | 38 | 69 |
| Frames observed on the correct target | 31 | 38 | 69 |
| Wrong-target frames | 0 | 0 | 0 |
| Observed ID-switch events | 0 | 0 | 0 |
| Occlusion episodes | 0 | 1 | 1 |
| Fragmentation episodes | 0 | 1 | 1 |
| Empty or fragmented mask frames | 0 | 2 | 2 |

The bicycle occlusion spans samples 29--32 (source frames 144--153) as a walking person crosses in front. Sample 30 is fully occluded and correctly left empty; sample 31 contains a small fragment of the same bicycle; samples 32 onward recover the original target. The mask never transfers to the occluding person. The bicycle's strict empty/fragmented-frame rate is 2/39 (5.13%); the combined rate is 2/70 (2.86%). Two additional bicycle frames are labeled partial rather than failed.

The moving-person track remains on the same visually identifiable person in all 31 frames, with no annotated occlusion or fragmentation episode.

## Conclusion boundary

This bounded audit supports zero observed identity switches for two assessable tracks and measures one recoverable bicycle fragmentation event. It does not estimate a general ID-switch rate: the sample contains only one scene, one reviewer and two selected tracks, and the three hard distant candidates are explicitly excluded rather than scored. Cross-scene, crowded-overlap and long-occlusion evaluation remain open.

The Phase-D1 checklist item is complete at pilot scope. The next controlled variable is the Phase-D2 comparison between unfiltered Pi3, semantic-only removal, tracked-mask removal and tracked-3D-motion removal.
