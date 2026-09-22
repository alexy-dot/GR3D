# Skating manual dynamic-track pilot

- Date: 2026-09-22
- Status: Phase D0 complete for one manually initialized within-clip track
- Result ID: `D001` (source tracking candidate `T001`)
- Scope: dynamic/static separation in Pi3 model coordinates, not metric 4D reconstruction

## Input and fixed models

- Video: Pi3 `examples/skating.mp4`
- Video SHA-256: `fe64b74f942806e413515fdb79f7825a19ed4e77affb61652dcb554d7b54e05d`
- Manual initialization: sample 0, box `[292, 64, 405, 336]` on the 640-pixel-wide first frame
- Prompt SHA-256: `220527c718fbb1d2c7ad2c1c8c5a4cb1e978d168efb2c86964c3ef43efae7261`
- SAM 2 source revision: `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- SAM 2 model: `configs/sam2.1/sam2.1_hiera_t.yaml`
- SAM 2 checkpoint SHA-256: `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`
- Original Pi3 revision: `9fa3ddb3f8d53041f8b2738df404f62223bbaa7b`
- Pi3 checkpoint SHA-256: `33580e4702ac671558aedeab1148fd08118f7ce45bdbeb99f3e3cf340062875d`
- Pi3 runner's last repository change: `a896e0ef8c9d948562a8f44a5f5687a43e46d370`
- Final SAM run code revision: `e5f7f986f4f7347cd272c1375062a3ba2927deff` with `code_dirty=false`
- Final map/ID/render code revision: `89e5d51`

The Pi3 run was created earlier in the same session before the run manifest stored a main-repository revision. Its raw manifest, model revision, checkpoint hash, exact saved input frames and observation hash are preserved below. Do not infer a stronger code-revision claim from the `last repository change` line.

## Parameters and device

Tracking stream:

- source interval: 3 frames
- extracted frames: 34, covering source indices 0--99
- resized width: 640 pixels
- mask format: lossless binary PNG
- device: NVIDIA GeForce RTX 4060 Laptop GPU
- PyTorch: 2.5.1+cu121
- final SAM elapsed time: 9.1828 seconds
- final SAM peak allocated VRAM: 1.042 GiB

Geometry stream:

- source interval: 12 frames
- frames: 8, source indices 0--84
- resized shape: 434 x 224
- pixel limit: 100,000 per frame
- Pi3 confidence threshold: 0.1
- Pi3 elapsed time: 24.9784 seconds
- Pi3 peak allocated VRAM: 5.52 GiB
- retained observations: 343,214

Post-processing:

- mask erosion: 1 pixel at Pi3 resolution
- minimum 3D support: 20 points
- background jitter: symmetric nearest-neighbor median after excluding selected target points
- maximum background support: 20,000 points per frame
- minimum valid states: 3
- normalized-motion thresholds: 2, 3, 5

## Tracking and 3D evidence

SAM 2 produced a visible target mask in all 34 tracking frames. Mask area ranged from 4.1135% to 6.5055% of the resized image. Visual inspection of the mask contact sheet showed one continuous skater silhouette and no obvious switch to another person. The original run plus two clean reruns produced identical ordered per-frame mask SHA-256 lists.

All eight Pi3 frames matched an exact source frame and yielded a valid robust 3D center. Per-state support ranged from 2,996 to 4,976 retained points. The center path length is 1.229425 Pi3 model units and endpoint displacement is 1.001954 model units. These values are not meters.

The seven object displacements are:

```text
0.179514, 0.157103, 0.122939, 0.207404, 0.166647, 0.171071, 0.224749
```

The paired background-jitter estimates are:

```text
0.036991, 0.011419, 0.007809, 0.019264, 0.008947, 0.010124, 0.022489
```

Normalized motion is `[4.8528, 13.7566, 15.7419, 10.7659, 18.6230, 16.8963, 9.9933]`, with median 13.7566 and direction consistency 0.8235. The classification is `dynamic` at thresholds 2, 3 and 5; threshold 5 is exceeded on 6/7 intervals. The candidate is therefore assigned `D001` for this bounded clip.

## Derived map

- source observations: 343,214
- retained static observations: 310,029
- separate dynamic observations: 33,185
- uncertain observations: 0
- removed dynamic ratio: 9.6689%
- raw source modified: false

The final identical-view comparison shows the eight time-separated skater point clusters in red in the unfiltered map and their removal from the derived static map. The trajectory render shows a continuous time ordering in XY, XZ and YZ. This is evidence for the declared within-clip result, not proof of long-term identity or globally correct Pi3 geometry.

## Artifact hashes

Raw outputs remain outside Git under `outputs/dynamic_skating_pilot/` in the WSL project workspace.

| Artifact | SHA-256 |
| --- | --- |
| final SAM track manifest | `1b42e05d934d688f017c960f017243f7d48597d854198b6a05e5c37fd4175e86` |
| Pi3 run manifest | `a549230a468163a2de0505faa2554e0b65aa9064a452b5ea9953202881fffc87` |
| Pi3 observations | `ee11320c4b51dc83ac8e39d56f9209838311a9ad7722d4cf9526d201ae2b1bc0` |
| aligned track manifest | `177ad9bee44f467b545fc54f0a8c5c6385557e119341ff9f3cf1ad4734671264` |
| lifted 3D states | `7b97a9206650bc18a429e528d57378b759db88e11e16fd1f85d498fef2fcb7c7` |
| selected point indices | `5bdff351208a0b81eae489e45cc3219786c1b463daa88f12bdc6dc23b7971ab8` |
| background jitter | `4c5187dd9eba5a9d51c512df60d4f9a25544cdd26700f3036ddaa7ebe1791376` |
| motion classification | `65d69bba13d10d31f398438fe33432bffcacc621ffd035b024a570d74efebc97` |
| final hybrid-map manifest | `491508004876662434d76bd290eed6cd5da3b1423cc77c3ba54aa4e9f537f7ad` |
| filtered static PLY | `0b8bd5823c6216c9a1ea227fd039e230422ac38fcdc8f832734c3074fb90c67f` |
| dynamic observations | `3f2370cad352dd5f24760ecf63de2694094c6893c6120ca02a8d9740ee13ae11` |
| final render manifest | `a4c1dc82c66c38a8bc6cfe501e311695d97b8756e56981a592e6c03e305488c7` |

## Validation and limitations

- Nineteen focused dynamic-pipeline tests pass, including frame-index separation, mask/point bounds, insufficient evidence, stationary/moving/noisy synthetic motion, disjoint map partitioning and deterministic sampling.
- SAM 2 emitted a warning because its optional CUDA extension `_C` was unavailable. Hole-filling post-processing was skipped. The tracker completed, masks were visually audited, and reruns were deterministic, but this remains a declared implementation difference.
- The target was initialized manually; automated discovery was not evaluated.
- No annotated masks or identity ground truth were available, so visual continuity is not a measured ID-switch rate.
- Background nearest-neighbor jitter is an internal Pi3 consistency statistic, not a sensor-grounded camera-motion estimate.
- This single obvious moving-person clip is not a false-positive test. A stationary-person or parked-vehicle control is required before generalizing the policy.
- `D001` is valid only within this run. No cross-window or long-video association is claimed.
