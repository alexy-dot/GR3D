# OSI 0000 multi-person SAM 2 pilot

- Date: 2026-09-23
- Status: first same-frame four-candidate SAM 2 run complete
- Scope: bounded within-clip multi-candidate propagation and Pi3 support gating, not persistent identity or annotated tracking evaluation

## Input and implementation

The input is the same OSI 0000 interval used by the manual T003 and automatic single-candidate pilots: source frames 57--147 inclusive, sampled every three frames at width 640. All four `person` detections above score 0.7 on source frame 111 were initialized together.

- prompt config: `research/configs/dynamic-pilot-osi0000-auto-persons-source111.json`
- prompt config SHA-256: `7b69def32948ce3a6f3e2d9dcf7717e7372576ed39703400d1e05a571a217798`
- source detection manifest SHA-256: `981faacde7ae5031437be239d848877fd9fa2d61e7ee2a876d18d03972ce25f2`
- SAM 2 revision: `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- SAM 2 checkpoint SHA-256: `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`
- repository revision: `11dd7eeb726b89873f262d4c57805ca5db256669`
- code dirty: false
- complete reconstruction tests: 65/65

`run_video_instance_tracking.py` now accepts either the original single-prompt schema or a same-frame `tracks` list. Multi-candidate mode shares one SAM 2 video state, assigns run-local numeric object IDs, and exports both an aggregate manifest and one existing-pipeline-compatible manifest per track candidate. Source pixels remain unchanged.

## Compute

- candidates: 4
- extracted frames: 31
- saved masks: 124/124
- elapsed time: 23.4621 seconds
- peak allocated VRAM: 1.067 GiB
- device: NVIDIA GeForce RTX 4060 Laptop GPU
- PyTorch: 2.5.1+cu121

The optional SAM 2 `_C` extension remained unavailable, so hole-filling post-processing was skipped as in prior runs.

## Per-track audit and Pi3 support

| Candidate | Detector score | Non-empty masks | Mask pixels min--max | Consecutive IoU min / median | Audit warning | Six-frame Pi3 support | Valid 3D states | Motion result |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |
| `T_AUTO_001` | 0.9991 | 31/31 | 455--14,375 | 0.3242 / 0.5756 | none | 51, 29, 146, 486, 1,141, 5,232 | 6 | dynamic, inherited from the byte-identical single-candidate masks |
| `T_AUTO_002` | 0.9504 | 14/31 | 0--977 | 0.0000 / 1.0000 | empty mask; large adjacent area change | 0, 0, 0, 2, 23, 214 | 2 | uncertain: insufficient valid 3D states |
| `T_AUTO_003` | 0.9255 | 31/31 | 46--184 | 0.0000 / 0.3043 | none from coarse thresholds | 0, 0, 0, 0, 0, 0 | 0 | uncertain: insufficient valid 3D states |
| `T_AUTO_004` | 0.8547 | 31/31 | 72--115 | 0.0000 / 0.3596 | none from coarse thresholds | 0, 3, 0, 2, 1, 0 | 0 | uncertain: insufficient valid 3D states |

`T_AUTO_002` is empty on source frames 57--105 and becomes non-empty near the prompt. This may reflect the target entering visibility rather than a tracker defect, but the 147x adjacent area change and only two valid Pi3 states prevent a motion claim. `T_AUTO_003` and `T_AUTO_004` are extremely small; their zero minimum adjacent IoU indicates fragmentation risk even though the existing coarse warning thresholds do not fire.

All four contact sheets were visually inspected. `T_AUTO_001` follows the known central walking person. The other masks remain in small distant-person regions, but their size and lack of identity annotation prevent a measured ID-switch conclusion.

## Single-versus-multi control

The multi-candidate `T_AUTO_001` track was compared against the earlier independent single-candidate run after validating the source video and every extracted-frame hash. All 31 mask hashes match, and every per-frame IoU is 1.0. On this clip, adding the other three objects did not perturb the main candidate.

This exact match is a useful implementation control, not proof that multi-object competition is harmless in crowded or overlapping scenes.

## Artifact hashes

Raw outputs remain outside Git under `outputs/2026-09-22-osi0000-person-multi-sam2/`.

| Artifact | SHA-256 |
| --- | --- |
| aggregate multi-track manifest | `ac84bb032b3f1e7f3b136858e4fb03922805826f3fcb1a6199a85ca54b8c6eda` |
| single-versus-multi T_AUTO_001 comparison | `ea461b98cd26ca3267cb102484274d98dd633a5abd75c825d8f3495ca8435eb8` |
| T_AUTO_001 track manifest | `58cf680f319fe093764380ab6feb17408b88a0505559b3d072f7a078e2966ffe` |
| T_AUTO_001 audit | `38978879a4ee4ade7ea085184bc50c05ebeeade657365e3c0e1179ebbeff6c6a` |
| T_AUTO_001 3D states | `f697cdb9671f92562120fae47506448b6a46f39ce38be8501d8f746afb959f7c` |
| T_AUTO_002 track manifest | `67c8548d8456ecc67c850bf4a4c59163604861eb4c6bd461271bb49a2c560c6f` |
| T_AUTO_002 audit | `1cdf69cc399a98ba8d067509f351ab11f100df9922a882f31a737d925a8f84aa` |
| T_AUTO_002 3D states | `bc508577b95a494fc7ef2706e5b8b73a0f2438911f8459795884e9af21f541ab` |
| T_AUTO_002 classification | `4cb6cb645c3b2223613f90a0af0d6734030ed20ed1c30ce88486c49a58bb4f09` |
| T_AUTO_003 track manifest | `9e4c4b9733e815808a7e474477c2ddf36ffa7c5050b0efc405448caae8b58466` |
| T_AUTO_003 audit | `e5d42c0d128b067cd3af2263619a781d5f792f34c4d3f8ca89494f1ae73e0f58` |
| T_AUTO_003 3D states | `0d2841737efa89e0cb43d2f309044c756013090468339bb502139fcc26a5750b` |
| T_AUTO_003 classification | `5846f0a9c60ee0830f6b5e014bf1519b2dec66f3d1bb41b83c1687a44a8673fd` |
| T_AUTO_004 track manifest | `f1c41bec95a85b76d10c3ccad2f8181b9d0271f4c83fe8120fd5fb03dec06ac3` |
| T_AUTO_004 audit | `ee33699d433a74e5d295b55a4343c42e81f91bb33bdb7592c4a306cc8c469c23` |
| T_AUTO_004 3D states | `57c7da84c83ba53121a06a934bbdcd0585ae3d81280ddd4a22d01548d23625b3` |
| T_AUTO_004 classification | `74086b6fed17198ee74f6fd1b522585a69fdd700a3b83ef3e07ff6c65a442f62` |

## Limitations and next boundary

- Candidate numbers are run-local and do not persist across windows or reruns.
- Three of four detections are too small or insufficiently visible for supported 3D motion estimates.
- The source contains baked-in OSI number markers.
- No annotated boxes, masks, occlusion states or physical identities are available, so ID switches and fragmentation are not measured against ground truth.
- The next independent variable is CoTracker foreground/background residual evidence. It must not silently replace the detector-plus-SAM baseline.
