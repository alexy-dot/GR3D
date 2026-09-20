# GR3D Paper Reproduction Status

Paper: *Boosting MLLM Spatial Reasoning with Geometrically Referenced 3D Scene Representations*, arXiv:2603.08592v2.

This document separates the paper's complete pipeline from the current geometry-only milestone. A successful PLY export is not a complete GR3D reproduction.

## Paper pipeline and current coverage

| Paper stage | Paper implementation detail | Current status |
| --- | --- | --- |
| Frame selection | Apply reconstruction to subsampled video frames; paper evaluation uses 16 frames for GPT models and 32 for InternVL2 | Partially implemented. Bounded deterministic sampling works; the first validated baseline uses 8 frames for the 8GB laptop |
| Neural reconstruction | Recover dense depth/point geometry, camera intrinsics, and camera extrinsics with π3; VGGT is an ablation | Implemented for bounded original-Pi3/Pi3X runs. The fixed-input 8-frame comparison is complete |
| Global point cloud | Unproject pixels into a global coordinate system | Implemented through Pi3 global point maps and PLY export |
| Axis alignment | Estimate up from floor points and horizontal room-boundary directions, then rotate the cloud upright | Cross-scene camera-gravity alignment is implemented and validated; exact paper floor-semantic/room-boundary refinement is not implemented |
| 2D semantics | Run Mask2Former trained on ADE20K on every input image | Implemented and validated with the official Swin-small ADE20K checkpoint on the 8-view house sample |
| 2D-to-3D fusion | Back-project semantic labels and aggregate them in a voxel grid by majority vote | Implemented through exact point/frame/pixel correspondence and voxel majority voting |
| Object instances | Group spatially connected voxels with the same label | Implemented with six-neighbor connectivity and configurable minimum voxel/point thresholds; known fragmentation/merging limitations remain |
| Geometric attributes | Compute centers and side lengths of 3D bounding boxes; fit vertical cylinders to suitable round objects | Component centers and axis-aligned bounding boxes are implemented; cylinder fitting is not implemented |
| Text conversion | Convert object geometry into compact text references indexed by numeric IDs | Explicitly out of the current milestone |
| Image association | Project object centers into images and place matching numeric IDs | Explicitly out of the current milestone |
| Occlusion filtering | Reject an annotation when projected object-center depth is behind the reconstructed depth map at that pixel | Explicitly out of the current milestone |
| MLLM evaluation | Provide annotated images plus geometric text to the MLLM and evaluate spatial questions | Explicitly out of the current milestone |

## Completed evidence

- WSL2 CUDA environment verified on an RTX 4060 Laptop GPU with 8GB VRAM.
- Original Pi3 is pinned to `9fa3ddb3f8d53041f8b2738df404f62223bbaa7b`.
- Official Pi3 checkpoint fingerprint is recorded.
- The official `skating.mp4` completed with 8 sampled frames at 434x224, using 5.52 GiB peak allocated GPU memory.
- Exported poses and intrinsics are finite and structurally valid.
- The output contains 343,214 retained colored points.
- Inspection found a broadly coherent static surface, repeated dynamic-person geometry, and regular grid/ray artifacts.

## Completed fixed-input comparison

The experiment changed only the reconstruction model:

- fixed input video;
- fixed sampled frames (`start=0`, `interval=12`, `max_frames=8`);
- fixed pixel limit (100,000);
- fixed confidence and edge thresholds;
- original Pi3 -> Pi3X.

| Measurement | Original Pi3 | Pi3X |
| --- | ---: | ---: |
| Runtime | 35.26 s | 27.19 s |
| Peak allocated GPU memory | 5.52 GiB | 6.14 GiB |
| Retained points | 343,214 | 730,395 |
| Retained ratio | 44.13% | 93.91% |
| Mean confidence | 0.1124 | 0.2754 |

Orthographic visual inspection found that both outputs contain regular sampling/ray structure and geometry from the moving skater. Original Pi3 shows repeated person silhouettes prominently; Pi3X produces a much denser surface but still contains dynamic and boundary artifacts. Because the models' confidence calibration and coordinate scales differ, higher confidence, more points, and absolute trajectory increments are not direct quality measurements. Pi3X remains an engineering extension and must not be reported as the paper's original π3 baseline.

The first deterministic XY/XZ/YZ exports also show that original Pi3 and Pi3X use substantially different raw coordinate ranges and orientations. These diagnostic renders are reproducible, but they are not yet comparable floor-plan views. A recorded axis-alignment transform is required before downstream spatial evaluation.

An additional eight-view static indoor `house` comparison preserves recognizable tables, chairs, walls, and openings in original Pi3's raw projection. Pi3X is faster and denser but shows much larger stretching and camera-path jumps on this sample. Original Pi3 therefore remains the reproduction baseline; Pi3X is retained only as a measured extension.

The implemented cross-scene alignment assumes Pi3's OpenCV camera-to-world convention and estimates up from the robust consensus of per-camera up vectors. It uses camera displacement/view direction to fix horizontal yaw and records the complete transform plus disagreement statistics. On the house example, aligned side views make floor-like surfaces horizontal and walls vertical. This is suitable for indoor/outdoor presentation but remains different from the paper's semantic floor and room-boundary alignment.

## Important interpretation boundary

The paper states that its selected π3 reconstruction predicts metric scale, while the pinned original Pi3 implementation describes its original outputs as scale-invariant and Pi3X as approximately metric. Until a known-distance or ground-truth trajectory evaluation is performed, this project must not label either output as verified physical metric geometry.

## Later exact-reproduction work

After the geometry comparison is stable, an explicit scope decision is required before implementing the object-level GR3D stages. The smallest faithful next block would be:

1. choose an indoor benchmark sample matching the paper's setting;
2. run Mask2Former ADE20K segmentation on the exact reconstruction frames;
3. back-project labels using the recovered depth and camera parameters;
4. voxelize and assign majority semantic labels;
5. extract connected object components;
6. fit bounding boxes and selected vertical cylinders;
7. measure attribute accuracy before adding text or MLLM evaluation.

## Planned no-ID long-video extension

The practical research target differs from full GR3D. Maintaining stable object IDs and repeatedly re-annotating frames is considered unrealistic for effectively unbounded real-world video. The alternative interface to evaluate is:

```text
long video
-> persistent static 3D map
-> canonical XY/XZ/YZ renders with axes and coordinates
-> small task-relevant set of original frames + canonical renders
-> MLLM spatial reasoning without persistent object IDs
```

The GR3D ID ablation is important but does not settle this hypothesis. In the paper, IDs connect object marks in images to indexed textual geometric attributes. Removing IDs breaks that cross-modal lookup. The proposed extension instead provides geometry as images and does not depend on an indexed text list, so it is a different representation with different failure modes.

The minimum downstream evaluation should compare:

1. original sampled frames only;
2. canonical 3D renders only;
3. original frames plus canonical 3D renders, without IDs;
4. a small-scene GR3D-style ID-linked condition as an upper/control condition when feasible.

Evaluate at least two capabilities separately:

- **spatial reasoning:** relative direction, distance ordering, route/layout, and coarse size;
- **cross-view correspondence:** which region/object in an original image matches a selected coarse cluster or coordinate region in the 3D render.

Useful render variants should be introduced one variable at a time: RGB point colors, depth shading, occupancy density, coordinate grid, camera frusta, and task-local crops. A visually attractive render is not evidence that correspondence works.
