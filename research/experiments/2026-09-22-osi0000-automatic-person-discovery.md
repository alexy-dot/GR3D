# OSI 0000 automatic person discovery

- Date: 2026-09-22
- Status: first single-candidate Phase-D1 discovery and automatic-prompt track complete
- Scope: Phase-D1 frame-local candidate discovery for the existing T003 walking-person interval

## Fixed input and protocol

- Video: OSI scene 0000, local path `tmp/osi_pilot_0000.mp4`
- Video SHA-256: `cd4e3edb7dc8d6f96a9a169319fb24511f6f18cc264965ef27b2573a8be19536`
- Source range: frames 57--147 inclusive
- Sampling interval: 3 source frames
- Expected extracted frames: 31
- Resized width: 640 pixels
- Detector: torchvision Faster R-CNN ResNet-50 FPN v2
- Weights: `FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1`
- Checkpoint size: 175,221,657 bytes
- Checkpoint SHA-256: `dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf`
- Score threshold: 0.7
- Allowed class: `person`
- Automatic prompt frame: source frame 111
- Automatic candidate policy: highest-score `person` on the exact prompt frame
- Manual comparison box: `[340, 130, 392, 265]`
- Device: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GiB
- PyTorch: 2.5.1+cu121

Semantic class is used only to propose detector candidates. It does not determine whether the candidate is static or dynamic. Detector identifiers are frame-local and do not establish identity across frames.

## Preserved startup failure

The first formal command at repository revision `9276d37` failed before frame extraction, model loading or detector inference with:

```text
ModuleNotFoundError: No module named 'reconstruction'
```

The unit tests imported the detector as a package, whereas the documented command launched `reconstruction/detect_video_candidates.py` directly. The script now selects the package import when imported and the sibling import when run as a file. A subprocess-level direct-entry test covers the documented launch path. No output directory or model artifact was created by the failed attempt.

## Detector result

The clean detector run used repository revision `efaa86f49612f493097dc547624110df1800f18e` with `code_dirty=false`, torchvision `0.20.1+cu121`, and the pinned checkpoint above.

- extracted frames: 31
- thresholded person candidates: 121
- frames with at least one candidate: 31/31
- per-frame candidate count: minimum 1, median 4, maximum 6
- elapsed time: 19.0025 seconds
- peak allocated VRAM: 0.640 GiB
- selected detector candidate: `F000018C001`
- selected detector score: 0.9990609
- raw detector box: `[338.1989, 132.8676, 367.6010, 221.7011]`
- outward-rounded SAM 2 prompt: `[338, 132, 368, 222]`
- IoU with the manual prompt box: 0.3500

Visual inspection of source frame 111 confirms that both boxes cover the same central black-clothed pedestrian. The lower box IoU is primarily due to the automatic box being substantially tighter; it is not evidence by itself that the candidate identity is wrong.

## Automatic-prompt SAM 2 result

SAM 2.1 Hiera Tiny was run independently from the automatic prompt with the same pinned revision and checkpoint as T003. The clean run also records repository revision `efaa86f49612f493097dc547624110df1800f18e` and `code_dirty=false`.

- non-empty masks: 31/31
- elapsed time: 14.6988 seconds
- peak allocated VRAM: 0.996 GiB
- mask fraction: 0.0019748--0.0623915
- minimum consecutive IoU: 0.3242468
- median consecutive IoU: 0.5756229
- maximum adjacent area ratio: 1.3591317
- maximum centroid jump: 0.0265664 image diagonals
- audit warnings: none

The optional SAM 2 `_C` extension was unavailable, as in the manual run, so hole-filling post-processing was skipped. The contact sheet was visually inspected and follows the same person from far field to near field without an obvious switch. This is not an annotated ID-switch measurement.

## Comparison with manual T003

`compare_video_tracks.py` validated identical source-video and extracted-frame hashes before comparing all masks. The comparison was generated at clean repository revision `c5156717e7e0e6bc079beab80a8c8a1ee6d8a0e9`.

- exact mask-hash matches: 0/31
- per-frame IoU: minimum 0.8896797, median 0.9549237, mean 0.9492060
- median Dice: 0.9769422
- median automatic/manual area ratio: 0.9632224
- minimum fraction of the automatic mask covered by the manual mask: 0.9559055
- median fraction of the automatic mask covered by the manual mask: 1.0
- median fraction of the manual mask covered by the automatic mask: 0.9624877
- centroid offset: median 0.0003540 and maximum 0.0011902 image diagonals

The masks are highly overlapping but not byte-identical. The evidence supports that the tighter automatic prompt selected the same within-clip target for this example; it does not establish general automatic discovery robustness.

## Pi3-linked motion check

The automatic masks were aligned to all six frames of the existing final-v2 Pi3 run. With the same one-pixel erosion, 20-point minimum support, 20,000-point background cap and thresholds 2/3/5, all six states were valid.

- support points: `[51, 29, 146, 486, 1141, 5232]`
- normalized motion: `[22.4426, 7.8333, 4.5840, 8.0631, 7.2194]`
- median normalized motion: 7.8333
- direction consistency: 0.6008
- threshold 2: `dynamic`, exceed fraction 1.0
- threshold 3: `dynamic`, exceed fraction 1.0
- threshold 5: `dynamic`, exceed fraction 0.8

This reproduces the manual T003 motion decision with slightly different masks and selected point support. The values remain Pi3 model-coordinate diagnostics rather than physical metric motion.

## Artifact hashes

Raw outputs remain outside Git under `outputs/2026-09-22-osi0000-person-detection/` and `outputs/2026-09-22-osi0000-person-auto-sam2/`.

| Artifact | SHA-256 |
| --- | --- |
| detection manifest | `981faacde7ae5031437be239d848877fd9fa2d61e7ee2a876d18d03972ce25f2` |
| selected prompt | `6bec9db90f1afa03786a431f3debdda2f45d0e3e8e5aee46c9c309d6661ab9ac` |
| automatic SAM 2 track manifest | `25f97d659cdfa04c7c26b76841be13f00847ebc6f3ba88caea58858d60c568ff` |
| automatic track audit | `80fef52ebb7b7fe80695476cb86b43a8eaa7a587f23772edaa255b726bace47b` |
| audit contact sheet | `9f87e39ec72920b6c5a2838ec6649256a0f3f995f05b8d49744d1385c2057c5f` |
| manual/automatic mask comparison | `9108845849f97fe9b015f4b3ae16516a0ebd3e1c9002640fc723326c7356a11c` |
| Pi3-aligned track manifest | `17ac90acc3670a135551534f0dd852e661157b1e66389432f1558eee4e2a7a06` |
| lifted 3D states | `405d276ecc74fd635807c45ac800996b133ca7962540851279484453e0ca3dab` |
| selected point indices | `eca79c219fffa64504cc484537e2044c3a108c4f46145f0528d4d89338c5b4a4` |
| background jitter | `4f6835cdb7a1fafa583f1c0b0e895cfcd2c9afb2091beda98bfdce908137a6c8` |
| motion input | `47c5036d62cf8b2b7d68edcc3b06d0a01c1057e8bcb9d84551f3bf321f73c6c2` |
| motion classification | `c2fcea85c85fe5e0f8b77602330f0a23f691c812426c3bd6b254d4d2c25d1205` |

## Limitations and next boundary

- The test covers one detector class, one selected prompt frame, one target and one scene.
- Highest detector score is not a general data-association policy and does not track multiple candidates.
- The OSI source video contains baked-in numbered markers, which remain a visual confound.
- No annotated detection box, segmentation mask or identity ground truth is available.
- This completes the declared instance-detector front end for one pilot; multi-candidate SAM 2, CoTracker residual evidence, annotated failure measurement and cross-window association remain open.
