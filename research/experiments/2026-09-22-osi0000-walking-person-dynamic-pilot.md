# OSI 0000 walking-person dynamic pilot

- Date: 2026-09-22
- Status: second manually initialized dynamic clip complete
- Result ID: run-scoped `D001` (source tracking candidate `T003`)
- Scope: within-clip dynamic/static separation in Pi3 model coordinates, not cross-run identity or metric 4D reconstruction

The `D001` identifier is assigned independently inside this OSI run. It is not the same persistent identity as the skating pilot's run-scoped `D001`.

## Input and fixed models

- Video: OSI scene 0000, local path `tmp/osi_pilot_0000.mp4`
- Video SHA-256: `cd4e3edb7dc8d6f96a9a169319fb24511f6f18cc264965ef27b2573a8be19536`
- Target: one black-clothed pedestrian walking toward the camera over source frames 57--147
- Manual initialization: source frame 111, tracking sample 18, box `[340, 130, 392, 265]` at 640-pixel width
- Prompt SHA-256: `c7db597e1ea55b1de67d841ce6ef3ee90a50a328613adbf076d85d61ea9fea38`
- SAM 2 source revision: `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- SAM 2 model: `configs/sam2.1/sam2.1_hiera_t.yaml`
- SAM 2 checkpoint SHA-256: `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`
- Original Pi3 revision: `9fa3ddb3f8d53041f8b2738df404f62223bbaa7b`
- Pi3 checkpoint SHA-256: `33580e4702ac671558aedeab1148fd08118f7ce45bdbeb99f3e3cf340062875d`
- SAM run repository revision: `c2a3507efc1e5dc473e3c70afbb629f403ef4ca1`, with `code_dirty=false`
- Final Pi3 runner repository revision: `2b2bda8`

## Parameters and compute

Tracking stream:

- source interval: 3 frames
- bounded source range: 57--147 inclusive
- extracted frames: 31
- resized width: 640 pixels
- propagation: bidirectional from source frame 111
- mask format: lossless binary PNG
- device: NVIDIA GeForce RTX 4060 Laptop GPU
- PyTorch: 2.5.1+cu121
- first elapsed time: 14.2276 seconds
- repeat elapsed time: 11.6412 seconds
- peak allocated VRAM: 0.996 GiB in both runs

Geometry stream:

- source frames: 57, 75, 93, 111, 129 and 147
- resized shape: 420 x 238
- pixel limit: 100,000 per frame
- Pi3 confidence threshold: 0.1
- final v2 elapsed time: 35.5354 seconds
- final v2 peak allocated VRAM: 5.47 GiB
- retained observations: 407,846
- dtype: BF16

Post-processing:

- mask erosion: 1 pixel at Pi3 resolution
- minimum 3D support: 20 points
- background jitter: symmetric nearest-neighbor median after excluding selected target points
- maximum background support: 20,000 points per frame
- minimum valid states: 3
- normalized-motion thresholds: 2, 3 and 5

The CPU alignment, lifting, audit, classification, map-building and rendering commands did not record exact elapsed time or peak memory. File modification timestamps are not substituted for runtime measurements.

## Tracking audit

SAM 2 emitted a non-empty mask in all 31 tracking frames. The deterministic contact sheet was visually inspected and follows the same pedestrian from the far field to the near field without an obvious switch to another person or a static object.

- mask fraction range: 0.0021484--0.0632595
- minimum consecutive mask IoU: 0.3751783
- median consecutive mask IoU: 0.5923282
- maximum adjacent area ratio: 1.3556414
- maximum adjacent centroid jump: 0.0265736 image diagonals
- audit warnings: none

An independent rerun produced identical ordered hashes for all 31 extracted frames and all 31 masks. SAM 2 again warned that its optional `_C` extension was unavailable, so hole-filling post-processing was skipped. No annotated identity ground truth exists; visual continuity and deterministic masks are not a measured ID-switch rate.

## Pi3 provenance correction

The first formal Pi3 attempt failed before inference with `ModuleNotFoundError: No module named 'pi3'` because the independent WSL clone was not on the import path. The structured failure manifest is preserved.

A subsequent successful run loaded the correct fixed clone through `PYTHONPATH`, but exposed a runner provenance defect: the manifest inspected only `third_party/Pi3` under the Windows-mounted workspace and recorded `pi3_revision=null`. The raw run remains preserved but is superseded.

`run_pi3_baseline.py` now accepts `PI3_ROOT` and uses that same resolved path for imports and revision recording. Two new tests cover the default and external-clone paths. Final v2 records the exact Pi3 revision `9fa3ddb3f8d53041f8b2738df404f62223bbaa7b`. Its point-cloud and observation hashes are byte-identical to the superseded successful run, showing that the correction changed provenance rather than geometry.

## 3D evidence and classification

All six Pi3 frames have valid 3D states. Per-frame support is 52, 28, 151, 510, 1,169 and 5,262 observations. The first two frames are retained but explicitly low-support because the distant person occupies few pixels.

The five center displacements in Pi3 model units are:

```text
0.333454, 0.131080, 0.073093, 0.119089, 0.098493
```

Paired background-jitter estimates are:

```text
0.014733, 0.016689, 0.015768, 0.014801, 0.013342
```

Normalized motion is `[22.6321, 7.8539, 4.6352, 8.0453, 7.3817]`, with median 7.8539, P90 16.7974 and direction consistency 0.6014. The candidate is `dynamic` at thresholds 2, 3 and 5. All intervals exceed thresholds 2 and 3; four of five exceed threshold 5.

## Derived map

- source observations: 407,846
- retained static observations: 400,674
- separate dynamic observations: 7,172
- uncertain observations: 0
- removed dynamic ratio: 1.7585%
- raw source modified: false

The identical-view comparison shows the tracked observations in red before filtering and their absence from the derived static map. The trajectory audit shows six time-ordered centers. This supports the declared within-clip dynamic result; it does not establish correct physical scale, complete human geometry or persistent identity.

## Final artifact hashes

Raw outputs remain outside Git under `outputs/dynamic_osi0000_person_pilot/` and `outputs/osi_0000_person_pi3_6f_final_v2/` in the WSL workspace. The SAM manifests contain individual hashes for every saved frame and mask.

| Artifact | SHA-256 |
| --- | --- |
| final SAM track manifest | `8bebf1b7a05154bdf30f512eb3a6048015fb1c7c6a8925e12e942bbda51a8305` |
| repeat SAM track manifest | `67963dfede368b367e7417012574b4f0b96ea50522bb468906c31507fffb43d7` |
| track audit JSON | `41135600270dbfbcd3e6fa7ed15b39d10da7e96b603c0a53c87c6064a7a8271d` |
| audit contact sheet | `84c2a4a00acb0454f544d7d7b636462c87f3aa44cfd0a6ae8368c5b34c17d3ad` |
| initial failed Pi3 manifest | `14dc2a330ef0a1462e4fb2fb2d17b2a5b2b7260ff6fcd44576e370026fd4d417` |
| final v2 Pi3 manifest | `a651a95bad755fed94b53ab6965684a301806fdb3d944056bdd1521feab3d5d5` |
| final v2 Pi3 observations | `a924d87ab7e696d0e5ae3407253f9864babb45a17cd136f653d57032ce195008` |
| aligned track manifest v2 | `095081af8ab23dc5dd0a3c5da7e6e655bf47ad8ad1a24021573f5274b89dd57a` |
| selected point indices v2 | `4fa9ff359ac96308d597792bb2c478947e25b22880b7c633b4ba77ed506b862b` |
| lifted 3D states v2 | `5b7828e97fca499fb12e199c9fecb53b40532f65e4754ad876a7aca7e9080947` |
| background jitter v2 | `ba3abb3e521518948e61b66fda4f00843bb7930485de440698ce82cf045e1d65` |
| motion classification v2 | `ca1e189bec276b0b6be5dd005faf3a608279e306c6192f465866dd463850a021` |
| hybrid-map manifest v2 | `20203f5166fafba86e496e701d96e504bff628adb26670675bbb046341e05d4e` |
| retained static PLY v2 | `f3b701cecad16cd2df3b7de8a3f19ffe5159e86d088f49eed5721bd7b04c49de` |
| dynamic observations v2 | `560ff797a4d64c0bebadbc6f7214a7201c6c751eec8ea11a3977813cec524afb` |
| trajectory/render manifest v2 | `b2c4f2e196322392584cace13818ebdbc53f8c357b8c20c4718fe43f92173f9a` |
| trajectory render v2 | `c8d224f5ecaaddb28f8c0bdc490cad4827cbcfba03b630df357eed0d71898aa4` |
| filter comparison v2 | `00519b4037b82e78af4aeb151ead6ee2acc7be72d70ed0bca2e2ea82e5fdc771` |
| trajectory table v2 | `69bb48220cd9e00c6e401ce6971d4c60e16fd2753f0b21cbcb83dec2babd733f` |

## Validation and limitations

- The complete reconstruction suite passes 54 tests after the external Pi3-root fix.
- The source video, masks, model outputs, point clouds and local environments remain outside Git.
- The target was initialized manually; automated discovery was not evaluated.
- Early 3D states have only 52 and 28 supporting observations, so their large first displacement may include more reconstruction noise than later states. The remaining four intervals still provide repeated dynamic evidence.
- The result covers the same OSI scene as the stationary control and is not a cross-scene robustness result.
- No annotated masks, identity labels, metric trajectory, cross-window association or long-term re-identification are available.
- This completes the required second dynamic clip, but it does not complete Phase D1 automation or the Phase D2 four-condition comparison.
