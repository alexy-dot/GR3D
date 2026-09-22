# OSI 0000 parked-bicycle stationary control

- Date: 2026-09-22
- Status: stationary-object negative control complete for one manually initialized within-clip track
- Result ID: `S001` (source tracking candidate `T002`)
- Scope: false-removal control for the Pi3/SAM 2 hybrid static-map policy, not long-term identity or metric motion validation

## Input and fixed models

- Video: OSI scene 0000, local path `tmp/osi_pilot_0000.mp4`
- Video SHA-256: `cd4e3edb7dc8d6f96a9a169319fb24511f6f18cc264965ef27b2573a8be19536`
- Target: parked bicycle over source frames 57--171
- Manual initialization: source frame 171, sample 38, box `[435, 145, 513, 235]` at 640-pixel width
- Prompt SHA-256: `1287527ee988f4405de5c6429a10ecb57e84eef3e47ba643d8b9910e937cb063`
- SAM 2 source revision: `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- SAM 2 model: `configs/sam2.1/sam2.1_hiera_t.yaml`
- SAM 2 checkpoint SHA-256: `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`
- Original Pi3 revision: `9fa3ddb3f8d53041f8b2738df404f62223bbaa7b`
- Pi3 checkpoint SHA-256: `33580e4702ac671558aedeab1148fd08118f7ce45bdbeb99f3e3cf340062875d`
- Final SAM/audit code revision: `96d0a9a778135b7a1a87e3edf65e934e4a2a4be2`, with `code_dirty=false`
- Final static-track rendering revision: `489c77d`

## Parameters and compute

Tracking stream:

- source interval: 3 frames
- bounded source range: 57--171 inclusive
- extracted frames: 39
- resized width: 640 pixels
- propagation: bidirectional from the prompt on the final sampled frame
- mask format: lossless binary PNG
- device: NVIDIA GeForce RTX 4060 Laptop GPU
- PyTorch: 2.5.1+cu121
- elapsed time: 10.9755 seconds
- peak allocated VRAM: 1.096 GiB

Geometry stream:

- Pi3 frames used by this control: source frames 57, 114 and 171
- full source Pi3 run: 8 frames at source interval 57, resized to 420 x 238
- pixel limit: 100,000 per frame
- Pi3 confidence threshold: 0.1
- full Pi3 elapsed time: 20.2059 seconds
- full Pi3 peak allocated VRAM: 5.51 GiB
- retained source observations: 522,620

Post-processing:

- mask erosion: 1 pixel at Pi3 resolution
- minimum 3D support: 20 points
- background jitter: symmetric nearest-neighbor median after excluding selected target points
- maximum background support: 20,000 points per frame
- minimum valid states: 3
- normalized-motion thresholds: 2, 3 and 5

The CPU alignment, lifting, audit, classification, map-building and rendering commands did not record exact elapsed time or peak memory. File modification timestamps are not substituted for runtime measurements.

## Tracking audit

SAM 2 emitted 39 masks over the bounded interval. The target was prompted at the last frame and propagated backward. The deterministic contact sheet was visually inspected.

- mask fraction range: 0.0--0.0135243
- median consecutive mask IoU: 0.6614765
- minimum consecutive mask IoU: 0.0
- maximum adjacent centroid jump: 0.0197501 image diagonals
- recorded warnings: `empty_mask`, `large_adjacent_area_change`
- empty mask: sample 30, source frame 147
- large area changes occur around source frames 147, 150 and 153 during a pedestrian occlusion

This is not a continuous error-free identity track. The mask disappears for one sampled frame and is fragmented around the occlusion. No annotated identity ground truth exists, so an ID-switch rate cannot be claimed. All three Pi3-aligned frames are outside the empty-mask event and have valid 3D support.

## 3D evidence and classification

The three Pi3-linked states contain 94, 339 and 912 selected observations, for 1,345 selected tracked observations in total. Their consecutive center displacements are `0.0115532` and `0.0120901` Pi3 model units. Paired background-jitter estimates are `0.0255469` and `0.0259484`.

Normalized motion is `[0.4522174, 0.4659126]`, with median `0.4590650`, P90 `0.4645430` and direction consistency `0.4340919`. The track remains `static` at thresholds 2, 3 and 5, with zero intervals exceeding any threshold. Candidate `T002` is therefore assigned `S001` for this bounded run.

The low normalized motion supports the intended negative-control behavior: camera motion did not cause this parked target to be classified as moving. These values remain in Pi3 model coordinates and are not physical meters.

## Derived map

- source observations: 522,620
- retained static observations: 522,620
- dynamic observations: 0
- uncertain observations: 0
- selected tracked observations retained: 1,345
- excluded tracked observations: 0
- raw source modified: false

The identical-view retention audit shows the selected points in red before classification and in green after the static-map policy retains them. This control measures false removal for one parked target only. It does not establish general dynamic-scene robustness.

## Final artifact hashes

Raw outputs remain outside Git under `outputs/dynamic_osi0000_stationary_pilot/` in the WSL project workspace. The SAM manifest contains the individual hashes for all 39 saved frames and masks.

| Artifact | SHA-256 |
| --- | --- |
| final SAM track manifest | `f87cecda6875102493aa6a9f963e69c639854709b28a0d3b870f543ab1932b5f` |
| track audit JSON | `8a035f5fdef28353380384ef298fc9a5ff0b1b25d24d537c5d575b4e159e7d4c` |
| audit contact sheet | `7aa99fc754b3bbc4375923bc8f1d4c4a0e00943efe57574c9702c90910e8da66` |
| Pi3 run manifest | `4984797fdf59df6085b88a2f461c2ebfab530ea69d4382374d179ff38ebfff29` |
| Pi3 observations | `d598a4c035764eea2c66f28be57d0a719f0457eff136565a23d48a2634f56396` |
| aligned track manifest | `8ad2a16923372642d4ce6ec726b895b3683fc1dd7268d6f3611cf2d77d04eea7` |
| lifted 3D states | `de097ca7769a11d06519c60648a68c601e970401728175befcd8b0a1d6c50549` |
| selected point indices | `c12670bfdb73091825c50426f4030688a22bdfac44fbb7331d4d2f407082da68` |
| background jitter | `0b575974c3b43456c9ef7eb0ab147381c099a47342ed4795a2b38e714ee065a8` |
| motion input | `feed6b2efe9f537ea23c1f7d780f16a8344f4a933713acc9b3a64454df1eae17` |
| motion classification | `b441340821a6ea7b8da4c30da44749e0ca39ac333b4955a09868c8212f34d63f` |
| track bundle | `839bb71e997b701ed9755a3a0f438cde3fd458f878a7a36d0e5c620767889800` |
| hybrid-map manifest | `747aa77513585e12526ded826c663c3eafb5cdb1c06bce881a9a4976c46c30bf` |
| retained static PLY | `5d4346ecf49203bc9d7ddaa696c16983a72310836c7fb83ea8611f47b5fd8eaa` |
| empty dynamic observations | `7a76375fce1fabe427d9c7951d62b69f1cd1f585a61cbf052b07fe940efa67a0` |
| empty uncertain observations | `7a76375fce1fabe427d9c7951d62b69f1cd1f585a61cbf052b07fe940efa67a0` |
| trajectory/render manifest | `72692f88fc95862a5cd920e936ba8fec9bc7c6d126d045267945ca291c117685` |
| trajectory render | `8ca7e4a3f317c295e2014b911f9def025465b4d52575002d2a8574200486b8c6` |
| retention comparison | `83d641d14100939e797e35fb9e038abd2a25b7db456030b64ba99b78ea1a84b7` |
| trajectory table | `6529f618808f123090cc0907cc424e155bee37ea2db74b7bfabe7620db94d2af` |

## Validation and limitations

- The full reconstruction test suite passes 52 tests after the stationary-control implementation.
- The source video, Pi3 observations and generated run outputs remain outside Git.
- The target was initialized manually; automated discovery was not evaluated.
- Occlusion fragmentation is present and explicitly retained in the audit rather than hidden.
- The three geometry states are sufficient for this declared classification but too sparse for long-term identity claims.
- This result supports one false-removal negative control. A second genuinely dynamic clip is still required before treating the policy as reliable.
- No metric scale, sensor-grounded motion, cross-window persistence or full 4D reconstruction is claimed.
