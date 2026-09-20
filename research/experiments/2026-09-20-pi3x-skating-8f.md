# Pi3X fixed-input comparison: skating, 8 frames

Date: 2026-09-20

## Purpose

Change only the reconstruction model from original Pi3 to Pi3X on the first validated CUDA input. This is an engineering comparison, not an exact GR3D baseline.

## Input and environment

- Input: `third_party/Pi3/examples/skating.mp4`
- Sampling: start frame 0, interval 12, maximum 8 frames
- Prepared resolution: 434 x 224 (97,216 pixels per frame)
- Pi3 source revision: `9fa3ddb3f8d53041f8b2738df404f62223bbaa7b`
- Checkpoint SHA-256: `69972d6e1c4492cb4d737a84fe940e357087d81c52f5c9b7c160b49c1f41669a`
- Device: NVIDIA GeForce RTX 4060 Laptop GPU, 8.0 GiB
- PyTorch: 2.5.1+cu121; dtype: bfloat16
- Confidence threshold: 0.1; edge relative tolerance: 0.03

## Result

- Status: complete
- Runtime: 27.19 seconds
- Peak allocated GPU memory: 6.14 GiB
- Retained points: 730,395 of 777,728 (93.91%)
- Confidence: min 0.00348, mean 0.27544, median 0.28957, max 0.48956
- Camera-pose step magnitudes: 0.1301, 0.0598, 0.1267, 0.0876, 0.1316, 0.1317, 0.2133 (model coordinate units)

The final trajectory increment is larger than the preceding increments, but absolute values must not be compared directly with original Pi3 because coordinate scale is not shared or physically validated.

## Visual inspection against original Pi3

- Pi3X retains about 2.13 times as many points and produces a denser ice surface.
- Both outputs retain regular sampling/ray patterns.
- Both outputs contain artifacts from the moving skater; original Pi3 shows multiple repeated silhouettes especially clearly.
- The skating scene is useful for exposing dynamic-object failure, but it is not a strong test of recognizable static objects or downstream 2D-to-3D correspondence.

## Interpretation

Pi3X was faster on this one run but required 0.62 GiB more peak GPU allocation. Its larger point count is not sufficient evidence of better geometry. The next comparison needs a mostly static indoor scene with recognizable objects and deterministic canonical views.
