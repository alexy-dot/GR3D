# Project Agent Instructions

## Source of truth

1. Treat the repository contents and recent Git history as authoritative.
2. Read `PROJECT_CONTEXT.md` before continuing unfinished research or implementation.
3. If `PROJECT_CONTEXT.md` conflicts with the code, inspect the code and record the correction in the context file.
4. Do not use old chat history as the primary project state when the repository contains a newer handoff.

## Active research boundary

- The active implementation target is the geometry-only part inspired by GR3D:
  `RGB video -> sampled frames -> point maps and camera poses -> persistent 3D map`.
- Do not add object-ID overlays, source-image re-annotation, geometry-to-text conversion, or MLLM inference unless the research scope is explicitly changed.
- The intended downstream extension is now explicit: test whether canonical axis-aligned renders of the reconstructed 3D scene (XY/XZ/YZ, with visible coordinates) plus a small number of original frames can support spatial reasoning without persistent object-ID annotation. Keep that evaluation separate from exact GR3D reproduction.
- Keep exact GR3D comparability separate from practical extensions:
  original Pi3 is the comparison baseline; Pi3X/Pi3XVO/VGGT-Long are long-video or metric-oriented extensions.
- Claims about metric scale, long-term consistency, or dynamic-scene robustness require measurements. A visually plausible PLY file is not sufficient evidence.
- Do not claim that an unlabeled 3D render establishes object correspondence. Correspondence between a rendered 3D region and an object in an original image must be evaluated explicitly.

## Experiment discipline

- Every meaningful run must save the exact input frames, model/checkpoint identity, code revision, parameters, device, outputs, and observed failure.
- Preserve raw outputs. Do not replace evidence with a verbal summary.
- Change one important variable at a time for initial comparisons.
- For long-video reconstruction, inspect window-boundary alignment, camera trajectory, drift, resource growth, and dynamic-object artifacts separately.
- Record experiments under `research/experiments/` using the existing template and date convention.

## Repository hygiene

- Never commit model weights, datasets, videos, point clouds, run outputs, credentials, or local virtual environments.
- External research implementations under `third_party/Pi3` and `third_party/VGGT-Long` are independent clones. Record their exact commit hashes in `PROJECT_CONTEXT.md`; do not vendor their `.git` directories into this repository.
- Keep `AGENTS.md` durable. Put temporary status, machine-specific details, blockers, and next actions in `PROJECT_CONTEXT.md`.
- Before a device/session handoff, update `PROJECT_CONTEXT.md`, run the relevant lightweight checks, commit the coherent state, and push when a remote is configured.

## Validation expectations

- On machines without CUDA, run syntax checks and `--dry-run` only.
- On an NVIDIA machine, run `reconstruction/check_gpu.py` before model inference.
- Start the RTX 4060 Laptop 8GB baseline at 8 frames and about 100K pixels per frame. Reduce to 6 frames/70K pixels on OOM; increase only one parameter after a successful recorded run.
- Report tests that could not be run and why.

## Cross-device continuation

Follow `docs/CODEX_HANDOFF.md`. Do not spend project time debugging Codex Remote connectivity unless the user explicitly changes that priority.

## Continuous Git synchronization

- GitHub is the shared project memory for the Windows and Mac workspaces.
- After every meaningful verified milestone, update `PROJECT_CONTEXT.md` and the relevant experiment record, then commit and push the coherent documentation/code state promptly.
- Do not wait until a device switch to record important results, blockers, decisions, dependency revisions, or exact next steps.
- Before every push, confirm that weights, datasets, videos, generated point clouds, run outputs, credentials, and machine-local environments remain untracked.
- A clean working tree on one computer does not prove another clone is synchronized; compare the local branch with `origin/main` before continuing work.
