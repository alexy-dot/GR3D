# Codex Cross-Device Handoff

## What is synchronized

- Git tracks source code, configuration, durable instructions, experiment plans, and current project state.
- `AGENTS.md` contains stable project-wide rules.
- `PROJECT_CONTEXT.md` contains the current effective state and exact next actions.
- Model weights, datasets, papers, videos, point clouds, caches, and experiment outputs are not synchronized through Git.

## Before switching computers

Ask Codex:

> Prepare a handoff for another Codex session. Inspect the repository and update PROJECT_CONTEXT.md with the current objective, completed work, decisions and reasons, failed approaches worth remembering, unresolved issues, validation performed, and exact next steps. Keep AGENTS.md limited to durable project instructions. Check that no weights, datasets, credentials, or generated outputs are staged. Then commit and push the coherent changes.

The session should then run the relevant lightweight checks, inspect the staged file list, commit, and push.

## On the other computer

```bash
git pull --ff-only
```

Then ask the new Codex session:

> Read AGENTS.md and PROJECT_CONTEXT.md, inspect the repository and recent Git history, and continue the unfinished work. Treat the repository as the source of truth. If the handoff conflicts with the code, explain the conflict and update PROJECT_CONTEXT.md before proceeding.

## First-time machine bootstrap

The external reconstruction repositories are not stored in the main Git repository:

```bash
git clone https://github.com/yyfz/Pi3.git third_party/Pi3
git -C third_party/Pi3 checkout 9fa3ddb3f8d53041f8b2738df404f62223bbaa7b

git clone https://github.com/DengKaiCQ/VGGT-Long.git third_party/VGGT-Long
git -C third_party/VGGT-Long checkout c160869d1d99c96bb227f414afb3bc68c29c9a76
```

Follow `reconstruction/README.md` for the Python/GPU environment. Weights are downloaded independently on each compute machine.

## Handoff quality check

A handoff is incomplete if it lacks any of:

- current objective and scope boundary;
- code/config files changed;
- tests actually performed and tests not performed;
- external dependency revisions;
- unresolved blockers;
- one exact next executable step.

Do not paste full chat transcripts into the context file. Preserve only conclusions that remain valid and failed paths that prevent repeated work.
