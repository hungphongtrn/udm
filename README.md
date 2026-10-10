# udm

Hypothesis branches (`hyp/<slug>`) on top of a vendored fork of Kev in [`kev/`](kev/). Fork provenance and change log:
[`kev/UPSTREAM.md`](kev/UPSTREAM.md).

- Code is written and reviewed on the dev box and runs only on the training machine; see [`CLAUDE.md`](CLAUDE.md) for
  the branch and machine rules. Run commands are in each hypothesis doc.
- Current hypotheses: [`docs/hypotheses/`](docs/hypotheses/).
- Past SCRM (set-conditioned reward model) results: [`docs/studies/`](docs/studies/README.md). Their code (`src/`,
  `scripts/`, `configs/`, `tests/`, `docs/TRAINING.md` etc.) was removed on this branch but lives in git history on
  `main`; read it with `git show main:<path>`.
