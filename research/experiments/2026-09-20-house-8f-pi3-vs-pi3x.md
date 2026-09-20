# Static indoor comparison: house, 8 frames

Date: 2026-09-20

## Purpose

Evaluate original Pi3 and Pi3X on a static indoor scene with recognizable furniture. This is more relevant to downstream object correspondence than the dynamic skating sample.

## Controlled input

- Input: `third_party/Pi3/examples/house`, eight provided PNG views
- Sampling: start 0, interval 1, maximum 8 frames
- Prepared resolution: 420 x 238 (99,960 pixels per frame)
- Confidence threshold: 0.1; edge relative tolerance: 0.03
- Pi3 source revision: `9fa3ddb3f8d53041f8b2738df404f62223bbaa7b`
- Device: NVIDIA GeForce RTX 4060 Laptop GPU, 8.0 GiB
- PyTorch: 2.5.1+cu121; dtype: bfloat16

## Results

| Measurement | Original Pi3 | Pi3X |
| --- | ---: | ---: |
| Checkpoint SHA-256 | `33580e4702ac671558aedeab1148fd08118f7ce45bdbeb99f3e3cf340062875d` | `69972d6e1c4492cb4d737a84fe940e357087d81c52f5c9b7c160b49c1f41669a` |
| Runtime | 51.04 s | 27.30 s |
| Peak allocated GPU memory | 5.51 GiB | 6.14 GiB |
| Retained points | 498,273 | 598,415 |
| Retained ratio | 62.31% | 74.83% |
| Mean confidence | 0.4872 | 0.4086 |
| Median confidence | 0.5226 | 0.4465 |

Both runs completed and produced finite reconstruction outputs. Deterministic raw-coordinate XY/XZ/YZ views were generated from 250,000 sampled points per cloud.

## Visual inspection

- Original Pi3 produces a compact scene in which the long table, surrounding chairs, wall, doorway, and adjacent room structure remain recognizable in the raw XY projection.
- Pi3X retains more points and is substantially faster, but its raw projection spans a much larger range and shows stronger scene stretching and large camera-path jumps.
- The raw axes and scales are model-dependent. This comparison therefore supports qualitative geometry diagnosis but not direct physical-distance comparison.

## Decision

Keep original Pi3 as the faithful GR3D reconstruction baseline. Do not replace it with Pi3X based on runtime or point count. Implement a recorded axis-alignment/presentation stage next, then evaluate recognizable object correspondence on aligned views.

## Semantic-fusion interface follow-up

The original-Pi3 condition was repeated with point-to-frame/pixel observation export enabled. Validation passed:

- 498,273 observations, matching the manifest point count;
- frame indices 0 through 7;
- pixel Y range 0 through 237 and X range 0 through 419;
- dense depth shape `(8, 238, 420)`, float16, all finite;
- `point_observations.npz`: 6.9 MiB;
- `depth_maps.npy`: 1.6 MiB.

This establishes the exact correspondence needed to sample per-frame Mask2Former labels and perform GR3D-style voxel majority voting.
