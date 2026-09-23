# Deterministic input-identity hardening

- Date: 2026-09-23
- Scope: GitHub Issue #3, provenance for MP4 files and Pi3 image directories
- Model/runtime artifacts: existing saved inputs and CPU post-processing only

## Problem and policy

Several downstream provenance checks treated `run_manifest["input"]` as a file and
raised `IsADirectoryError` for Pi3 House/image-directory runs. Input identity schema
version 1 now covers both supported forms through one shared helper.

- File identity uses the existing streaming SHA-256 as `combined_sha256`, preserving
  historical MP4 behavior, and records one ordered entry plus `input_kind=file`.
- Image-directory identity includes only sorted top-level `.png`, `.jpg`, and `.jpeg`
  files, matching the Pi3 loader. Every entry records its normalized relative path,
  byte size and SHA-256; the versioned ordered-entry payload produces one combined
  digest.
- Unsupported and nested files are excluded because the loader excludes them.
- Structured identities are compared across new artifacts. A directory digest is not
  silently compared with a legacy bare file-hash field. Package validation rejects
  mixed new/legacy provenance, while alignment records an explicit warning when an
  old track hash cannot be compared with an image directory.
- Legacy `video_sha256` fields remain populated for files and are `null` for image
  directories.

The shared helper is used by Pi3 run manifests, detector/tracker manifests, track-to-Pi3
alignment, scene-instance export, representative-crop extraction, OSI package assembly
and package validation.

## Validation

The complete reconstruction suite passed 92 tests. The focused tests cover repeated
digest stability, filesystem iteration order, byte/name/membership changes, ignored
unsupported and nested files, unchanged file SHA-256 behavior, explicit new/legacy
compatibility, and House alignment/crop subprocess paths.

A real House dry run used eight top-level images at 420x238:

- input kind: `image_directory`
- file count: 8
- combined SHA-256: `5cf8d152e669c422e1277af9f887e051b582efc7ef978ffa800fc4a7cea46819`
- manifest SHA-256: `7d900485c9a67fcaf7b8c41b188ceb840e66f7fda640564332c395d66c48ace7`

The existing House representative-crop path was replayed successfully. It produced 14
crops; every crop SHA-256 matched the prior output. The replay catalog SHA-256 is
`a5f4356c3e71d1021341d778aab9631b23c21fb31b2397bc4066d3c6a42e4376`.

For the existing OSI 0000 MP4, the new file identity digest is
`cd4e3edb7dc8d6f96a9a169319fb24511f6f18cc264965ef27b2573a8be19536`,
exactly equal to the historical `source_video_sha256` recorded by the package and
tracking artifacts.

## Limits

No Pi3, SAM 2, CoTracker or MLLM inference was rerun. The House check validates identity
generation and the existing crop path; it does not re-establish reconstruction quality.
Directory identity deliberately covers only the current non-recursive Pi3 loader contract.
Changing the loader's accepted extensions or recursion policy requires a schema/version
review. Old directory runs without structured identity remain explicitly non-comparable to
new directory identities unless regenerated from their available source directory.
