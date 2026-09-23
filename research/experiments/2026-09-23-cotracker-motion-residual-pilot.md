# CoTracker foreground/background residual pilot

- Date: 2026-09-23
- Status: complete for one moving-person track and one parked-bicycle control
- Scope: auxiliary 2D point-motion evidence for Phase D1; this does not replace the existing Pi3-linked 3D motion decision

## Question and fixed comparison

Test whether CoTracker3 points sampled inside an existing SAM 2 mask have larger motion residuals than nearby background-ring points after a robust local affine background fit.

Only the target track changed between the two runs. Both used 64 eroded-mask foreground queries, 96 background-ring queries, three-pixel erosion, ring radii 8--48 pixels, minimum supports of 8 foreground and 16 background points, and the same model/checkpoint/code.

- Moving target: automatic multi-candidate `T_AUTO_001`, source frames 57--147 every three frames, prompted at source frame 111.
- Stationary control: manually initialized parked-bicycle `T002`, source frames 57--171 every three frames, prompted at source frame 171.
- The other three same-frame person tracks were excluded from the moving target's background ring.
- Semantic class did not determine either motion state.

## Environment and provenance

- Device: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GiB
- Python: 3.10.12
- PyTorch: 2.5.1+cu121
- Repository revision: `780a25a0a9ee75c3a294794b4481cc32892af8f6`, `code_dirty=false`
- CoTracker source revision: `82e02e8029753ad4ef13cf06be7f4fc5facdda4d`
- Source archive: `/tmp/co-tracker-82e02e8029753ad4ef13cf06be7f4fc5facdda4d.tar.gz`
- Source archive size: 33,507,704 bytes
- Source archive SHA-256: `45ccb696ddd27b89caffefe6b9c2f5c36f61357fd8419ae02da7ba7b8cc999d1`
- CoTracker model repository revision reported by the download endpoint: `bf55ea50d4390e1820a267f131cd6587240fb2c5`
- Checkpoint: `scaled_offline.pth`, 101,890,938 bytes
- Checkpoint SHA-256: `2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834`
- Moving track manifest SHA-256: `58cf680f319fe093764380ab6feb17408b88a0505559b3d072f7a078e2966ffe`
- Parked-bicycle track manifest SHA-256: `f87cecda6875102493aa6a9f963e69c639854709b28a0d3b870f543ab1932b5f`

The source came from a verified archive rather than a Git checkout. The run manifest therefore records the explicit source revision and `source_revision_git_verified=false`; the archive size and hash above preserve the missing provenance layer. The first attempted run exposed and then fixed a parent-repository revision leak: `git rev-parse` had climbed out of the archive directory and reported the containing GR3D checkout. The final implementation queries Git only when the declared source root itself contains `.git`.

The official checkpoint host was unreachable from both WSL and Windows. The same Hugging Face repository object was downloaded through `hf-mirror.com`; its byte count and SHA-256 match the `X-Linked-Size` and `X-Linked-Etag` returned for the official model object.

## Results

| Measurement | Moving person | Parked bicycle |
| --- | ---: | ---: |
| Frames / adjacent intervals | 31 / 30 | 39 / 38 |
| Valid intervals | 26 | 36 |
| Foreground visibility min / median | 0.1094 / 0.7500 | 0.0000 / 0.7656 |
| Background visibility min / median | 0.6458 / 0.9688 | 0.2500 / 0.4271 |
| Median foreground residual (pixels) | 2.2370 | 0.4782 |
| Median background-fit residual (pixels) | 0.6472 | 0.2536 |
| Normalized residual Q25 / median / Q75 | 1.9473 / 2.7778 / 4.1861 | 1.0694 / 1.9046 / 2.6604 |
| Normalized residual P90 | 6.6717 | 3.1840 |
| Valid intervals above 2 / 3 / 5 | 73.08% / 46.15% / 23.08% | 47.22% / 19.44% / 0.00% |
| Runtime | 5.4497 s | 4.3202 s |
| Peak allocated VRAM | 2.759 GiB | 3.430 GiB |

The moving run's four invalid intervals are samples 1--5, corresponding to the distant-person portion near source frames 60--72. The parked-bicycle run's two invalid intervals are samples 29--31, around source frames 144--150, coinciding with the known SAM-mask disappearance and pedestrian occlusion.

Raw artifacts remain outside Git:

| Artifact | SHA-256 |
| --- | --- |
| moving residual manifest | `06e3d4f88c628174594c67185af48f1fb7b6fe213951acf270b1934e73e9657d` |
| moving point tracks | `56d243abebc0c6a9406f8ee7ba3b6d40ed3db07ddd361833d1684ff5853f9008` |
| parked residual manifest | `277cd4de29eb06a20e04054a35137723778b04b2a2270a142b90c4162c265061` |
| parked point tracks | `1e289135ef19ee69e0e1877d2bf19afcaed9f6fdb891063f90d64b2783ce4351` |

## Interpretation and limits

The moving-person distribution is higher than the parked control on this pair, so the residual is useful auxiliary evidence. It is not cleanly separable: nearly half of the parked control's valid intervals exceed 2, while fewer than one quarter of the moving intervals exceed 5. A universal per-interval threshold is therefore unsupported.

The stationary control has much lower background-point visibility, and both tracks inherit SAM initialization, occlusion and mask-boundary errors. The local affine model is a 2D approximation rather than recovered camera geometry. These runs contain no annotated point tracks or motion ground truth. CoTracker remains a diagnostic alongside the Pi3 world-frame classifier, which still labels the moving person dynamic and the parked bicycle static at its declared thresholds.

## Decision

Keep CoTracker as saved auxiliary evidence and mark the Phase-D1 implementation item complete. Do not feed its score into the current `S/D/U` decision until more labeled controls establish calibration. The next independent work remains annotated tracking-failure measurement and the four-condition Phase-D2 static-map comparison.
