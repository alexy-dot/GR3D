# OSI-Bench Pi3 integration pilots

Date: 2026-09-20

## Purpose

Validate the current no-ID reconstruction path on official OSI-Bench videos before any full benchmark download:

```text
official MP4 -> exact sampled frames -> original Pi3 -> camera-gravity views
-> Mask2Former ADE20K -> semantic voxels -> separate layout/object renders
```

This is an integration and representation pilot. It does not establish metric accuracy, trajectory accuracy, or downstream MLLM performance.

## Dataset access

- Hugging Face dataset: `HarmlessSR07/OSI-Bench`.
- Official metadata: `data.parquet`, 381,866 bytes, 8,766 rows, 997 unique videos.
- The four video archives are approximately 20.6--21.5 GB each. Their remote ZIP directories support HTTP Range requests, so individual MP4 files were extracted without downloading an entire archive.
- Datasets and videos remain ignored and outside Git.

## Common configuration

- Pi3 revision: `9fa3ddb3f8d53041f8b2738df404f62223bbaa7b`.
- Pi3 checkpoint SHA-256: `33580e4702ac671558aedeab1148fd08118f7ce45bdbeb99f3e3cf340062875d`.
- Mask2Former model: `facebook/mask2former-swin-small-ade-semantic`.
- Mask2Former checkpoint SHA-256: `2533310a1e902580a79e575814f1fa04078a99e2321b9bda7afd4f22bba49189`.
- Device: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GiB.
- Frames: 8, uniformly spaced across each sequence.
- Pixel limit: 100,000; actual resized frames: 420x238.
- Pi3 precision: BF16.
- Exact resized input frames, point/pixel observations, depth maps, manifests, checkpoint hashes, camera poses, and raw point clouds were preserved in the ignored run directories.

## Pilot 0963: integration-only indoor scene

- Source SHA-256: `d34f68f6aa650a399104bfe476fdb1f15856ea272139c9b6410a9f67bfda7751`.
- Source: 1920x1080, 15 FPS, 104 frames, 6.93 seconds.
- Sampling interval: 14 frames.
- Visual inspection: indoor shopping/arcade corridor, so this sequence is not valid evidence for outdoor robustness.
- Pi3: 618,164 retained points, 18.51 seconds, 5.51 GiB peak allocation.
- Camera-up disagreement: median 0.09 degrees, P90 0.22 degrees, maximum 0.23 degrees.

Voxel-size sweep, holding the other settings fixed at 2 points/voxel, 1 view/voxel, 0 minimum purity, and 30 voxels/component:

| Voxel size | Voxels | Components | Weighted purity | Multiview ratio | Median views |
|---:|---:|---:|---:|---:|---:|
| 0.02 | 23,590 | 39 | 0.9321 | 0.7717 | 3 |
| 0.03 | 12,049 | 24 | 0.9300 | 0.7889 | 3 |
| 0.05 | 4,753 | 14 | 0.9304 | 0.8275 | 3 |

This scene exposed a workflow bug: the older WSL copy of `run_pi3_baseline.py` saved observations but not exact sampled PNGs. The semantic stage correctly refused video processing. The current script was synchronized, the run was repeated into a new directory, and all eight exact frames were verified before segmentation.

## Pilot 0000: outdoor scene

- Source SHA-256: `cd4e3edb7dc8d6f96a9a169319fb24511f6f18cc264965ef27b2573a8be19536`.
- Source: 1920x1080, 15 FPS, 406 frames, 27.07 seconds.
- Scene content: outdoor service road with buildings, trees, bicycles, motorcycles, sidewalk, and drainage features.
- Sampling interval: 57 frames, covering frames 0 through 399.
- Pi3: 522,620 retained points, 20.21 seconds, 5.51 GiB peak allocation.
- Camera-up disagreement: median 0.25 degrees, P90 1.68 degrees, maximum 4.71 degrees; the internal orientation gate passes.

Voxel-size sweep with the same fixed settings:

| Voxel size | Voxels | Components | Weighted purity | Multiview ratio | Median views |
|---:|---:|---:|---:|---:|---:|
| 0.02 | 28,555 | 44 | 0.9583 | 0.6309 | 2 |
| 0.03 | 20,294 | 28 | 0.9538 | 0.6106 | 2 |
| 0.05 | 8,635 | 18 | 0.9473 | 0.6756 | 2 |

At voxel size 0.03 the large components correspond to road, buildings, trees, sidewalk, wall, fence, and two minibike components. After the curated layout/object split, the layout view retains 20,150 voxels from 24 components, while the object view retains only 144 voxels from 4 components.

### Fixed-scale component-threshold sweep

Voxel size was then held at 0.03 while only the minimum component size changed:

| Minimum voxels/component | Retained voxels | Components | Purity | Multiview ratio | Target-category result |
|---:|---:|---:|---:|---:|---|
| 5 | 21,334 | 141 | 0.9418 | 0.6016 | 4 bicycle, 9 minibike, 8 person, but fragmentation warning |
| 10 | 20,817 | 62 | 0.9463 | 0.6085 | 3 bicycle, 4 minibike, 1 person, 3 signboard; no fragmentation warning |
| 20 | 20,500 | 37 | 0.9510 | 0.6094 | 3 bicycle, 2 minibike; person and signboard removed |
| 30 | 20,294 | 28 | 0.9538 | 0.6106 | 2 minibike; bicycle/person/signboard removed |

The 10-voxel condition gives the best initial presentation tradeoff on this scene. Its object layer contains 332 voxels from 17 components across person, plant, base, signboard, awning, minibike, and bicycle labels. It is retained as a scene-specific pilot setting, not a universal default.

### 2D semantic coverage and two-wheeler label stability

An evaluation-only overlay audit was run on the eight exact Pi3 input frames. It does not draw object IDs and is not used as the downstream representation.

- `bicycle`: present in 6/8 frames, 9,323 total labeled pixels;
- `minibike`: present in 5/8 frames, 12,182 pixels;
- `person`: present in 8/8 frames, 10,842 pixels.

Visual inspection shows that the same parked two-wheeler regions switch between `bicycle` and `minibike` across frames. The target is therefore covered in 2D, but the ADE20K class boundary is temporally unstable and can split related 3D components. Pixel coverage alone does not establish instance identity or correspondence.

A practical extension merged only labels 116 (`minibike`) and 127 (`bicycle`) into `two_wheeler`, while holding the geometry, voxel size 0.03, and 10-voxel component threshold fixed. Compared with the unmodified ADE20K condition:

| Condition | Voxels | All components | Two-wheeler components | Object-layer components | Weighted purity |
|---|---:|---:|---:|---:|---:|
| Original ADE20K | 20,817 | 62 | 7 (3 bicycle + 4 minibike) | 17 | 0.9463 |
| Two-wheeler merge | 20,836 | 60 | 5 | 15 | 0.9488 |

The merge changed 5,552 of 13,653 matched point observations. It modestly reduced fragmentation and retained 351 object-layer voxels rather than 332. This is promising but remains a labeled practical extension, not an exact GR3D reproduction and not proof that five components equal five physical instances.

### Trajectory shape and answer-derived scale anchor

The eight Pi3 camera centers produce:

- path length: 1.877205 model units;
- endpoint displacement: 1.876864 model units;
- straightness ratio: 0.999818;
- raw-XY heading change: 1.09 degrees.

The near-one straightness ratio and small heading change agree with OSI-Bench question index 8, whose answer is `straight`. Question index 9 gives a 28.1 m full-trajectory length. Using that answer as a single scale anchor yields 14.9691 meters per Pi3 model unit.

This is calibration, not independent validation: the same 28.1 m answer defines the scale, so it cannot measure absolute-scale error or scale drift. The calculation and warning are preserved in `trajectory_analysis.json` by `reconstruction/analyze_camera_trajectory.py`.

### Sensor-ground-truth availability

The paper states that the acquisition rig used synchronized stereo RGB, 32-beam LiDAR, and IMU/GPS, and that the raw streams were calibrated during benchmark construction. The current official GitHub repository and Hugging Face release expose benchmark MP4 files and QA metadata, but no raw LiDAR, IMU/GPS, calibration, timestamp, sparse-depth, or reference-pose files were found. The paper says that additional raw multimodal videos will be released. Until that happens, camera-gravity consistency and answer-derived scale anchors must remain explicitly weaker than sensor-grounded evaluation.

## Interpretation and decision

- The official-OSI-video-to-no-ID-voxel pipeline now runs end to end without downloading the full dataset.
- The camera-gravity alignment is internally consistent on the outdoor pilot, but it has not been compared with OSI IMU/GPS gravity or heading and must not be called ground-truth orientation.
- The reconstructed camera path is almost perfectly straight and agrees with the benchmark's qualitative trajectory label. The benchmark's 28.1 m trajectory answer can anchor scene scale, but cannot independently validate it.
- High voxel label purity is only self-consistency after semantic voting. It does not prove correct geometry or correct object segmentation.
- The 30-voxel component threshold is too destructive for small outdoor objects: most of the usable representation is layout, while only four object-layer components survive at size 0.03. Lowering only this threshold to 10 recovers 17 object-layer components without triggering the current fragmentation warning; lowering it to 5 produces 141 total components and does trigger the warning.
- Do not choose a universal voxel size from these model-coordinate sweeps. Original Pi3 is scale-invariant. A metric outdoor threshold requires sensor calibration or scale alignment.
- Next controlled change: evaluate whether the scene-specific threshold-10 object layer preserves correspondence to the tagged source objects, then compare orientation and scale against available OSI sensor data if it can be obtained separately.
