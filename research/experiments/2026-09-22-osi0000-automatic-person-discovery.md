# OSI 0000 automatic person discovery

- Date: 2026-09-22
- Status: in progress; command-line startup defect fixed before detector inference
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

## Pending measurements

- automatic box, score and IoU with the manual T003 box;
- candidate counts across all 31 frames;
- detector elapsed time and peak allocated VRAM;
- manifest, selected-prompt and frame hashes;
- independent SAM 2 propagation from the automatic prompt;
- mask agreement and tracking audit versus manual T003.
