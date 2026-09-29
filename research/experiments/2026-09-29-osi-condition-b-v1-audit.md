# OSI scene 0000 Condition-B v1 audit

## Purpose

Audit the first official OSI-Bench comparison between raw video (`A`) and the same video
plus unfiltered Pi3 camera-gravity-aligned XY/XZ/YZ views (`B-v1`). This record separates
the observed score from what the representation actually supports.

## Fixed inputs

- Scene: `0000`
- Questions: 10
- Model: `Qwen2.5-VL-3B-Instruct`
- Video sampling: the same official 32 source-frame identities in A and B
- Geometry: original Pi3, 32 frames, approximately 100K pixels per frame
- Scoring: unchanged official OSI-Bench evaluator

## Result

| Condition | Official score |
| --- | ---: |
| A: raw video | 0.18 |
| B-v1: raw video plus unfiltered Pi3 views | 0.27 |

The apparent gain is `+0.09`, but all of it comes from one row. For person ID 26
displacement between 2.1 and 4.1 seconds, A predicted `10`, B-v1 predicted `2`, and the
answer was `2.18`; the official score changed from `0` to `0.9`. The other nine row scores
were unchanged.

Three additional predictions changed without improving their score: window-ID 03 camera
distance (`2.1 -> 2.5`, answer `12.49`), motorcycle counting at 23 seconds (`1 -> 0`,
answer `9`), and person-ID 30 displacement (`10 -> 1.2`, answer `6.07`).

## Visual audit

The three global orthographic views are valid renderer outputs but weak task evidence:

- geometry is fragmented into multiple disconnected or overlapping point-cloud regions;
- the YZ projection heavily overlaps structures and does not expose object instances;
- no OSI numeric ID is associated with a 3D region or trajectory;
- the views are global rather than conditioned on the question timestamp;
- numeric axes show arbitrary Pi3 model units, but B-v1 did not state that they were not
  meters.

The last point is a protocol confound for OSI's meter-valued questions. The improved
prediction `2` may reflect accidental use of a visible model-unit interval rather than
valid metric reconstruction. B-v1 therefore does not establish a stable 3D benefit.

## Decision

1. Preserve A and B-v1 predictions, images, manifests, and hashes unchanged.
2. Do not scale B-v1 to the frozen 12 scenes.
3. Introduce Condition-B protocol v2 with visibly non-metric axes, explicit prompt text,
   a schema-v2 capability declaration, and a distinct dataset/output name.
4. Keep ID correspondence, timestamp selection, static-map filtering, and dynamic
   trajectories out of B. Evaluate them as separate C/D variables.
5. Treat metric scale as an independent requirement. Never derive scale from benchmark
   answers.

## Protocol-v2 follow-up

Condition-B v2 completed on 2026-09-29 after adding explicit non-metric labels to every
render, telling the model not to treat coordinates or point-cloud distances as meters,
and declaring that the views contain no OSI-ID correspondence or question-time slice.
The official score remained `0.27`.

The person-ID 26 displacement prediction remained `2` for answer `2.18`, retaining score
`0.9`. The person-ID 30 displacement prediction changed from B-v1 `1.2` to B-v2 `2`, but
remained incorrect for answer `6.07`. Other reported numerical predictions and category
scores were unchanged.

This follow-up rejects the narrow explanation that the entire B-v1 improvement disappears
when the model is warned that Pi3 axes are not meters. It does not establish correct
metric reasoning: the gain still comes from only one question, Pi3 remains non-metric,
and the representation still lacks an explicit mapping from OSI IDs to reconstructed
objects. The appropriate claim is a repeatable single-scene, single-question improvement
under the corrected protocol, with unresolved causal mechanism.

## Interpretation boundary

This experiment proves that supplemental Pi3 views can alter model predictions. It does
not prove that the model used correct geometry, recovered metric displacement, associated
the requested OSI IDs, or improved generally across scenes or categories.
