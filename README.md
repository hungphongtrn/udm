# udm

Hypothesis branches (`hyp/<slug>`) on top of a vendored fork of Kev in [`kev/`](kev/). Fork provenance and change log:
[`kev/UPSTREAM.md`](kev/UPSTREAM.md).

- Code is written and reviewed on the dev box and runs only on the training machine; see [`CLAUDE.md`](CLAUDE.md) for
  the branch and machine rules. Work from the repo root: `./kev.sh <command>` runs a Kev command inside `kev/`
  (kev-relative paths; `*.py` runs with `kev/.venv/bin/python`). Setup and full run reference:
  [`kev/scripts/pref/README.md`](kev/scripts/pref/README.md).
- Questions by status (`todo/`, `in-progress/`, `done/`) and the 3-GPU plan: [`docs/hypotheses/README.md`](docs/hypotheses/README.md).
- Training-data budget and the controlled eval set: [`docs/data-eval.md`](docs/data-eval.md).
- Past SCRM (set-conditioned reward model) results: [`docs/studies/`](docs/studies/README.md). Their code (`src/`,
  `scripts/`, `configs/`, `tests/`, `docs/TRAINING.md` etc.) was removed on this branch but lives in git history on
  `claude/set-conditioned-reward-model-hvsb9s` (the branch this one was cut from); read it with `git show origin/claude/set-conditioned-reward-model-hvsb9s:<path>`.
