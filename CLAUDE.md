@AGENTS.md

## One branch per hypothesis

Branches are how we track hypotheses. Each hypothesis gets its own branch, named `hyp/<short-slug>`
(e.g. `hyp/preference-reward-model`), cut from the branch whose results it builds on.

- **State it before writing code.** The first commit on the branch adds `docs/hypotheses/<slug>.md`. It lists the
  claim being tested, the arms (ID, what changes, what stays fixed), the metrics and the decision rule. Record these
  before any run.
- **One hypothesis per branch.** Unrelated changes (fixes, infra, other ideas) go on their own branch. A side
  question that comes up mid-study becomes a new `hyp/` branch.
- **Results stay on the branch.** Results go into the hypothesis doc as they come in (run IDs, seeds, numbers,
  deviations from the plan), plus `docs/studies/` per AGENTS.md. The branch ends with a verdict: supported, refuted
  or inconclusive, and why.
- **Merging.** A concluded branch is merged back to its parent with its verdict, whatever the outcome; negative
  results are kept. An abandoned branch gets a final commit that says why it stopped.
- **Machines.** Code is written and reviewed on the dev box and runs only on the training machine. Do not run training,
  feature extraction or downloads on the dev box. Each `hyp/` branch's doc lists the exact commands to run there.

## Working as an orchestrator

The main session plans, reviews and integrates. It delegates the rest to subagents to keep cost down, and still owns
getting the work finished.

- **Code: Sonnet at medium effort** (`model: "sonnet"`, `effort: "medium"`). Writing or editing code, tests, scripts
  and docs. Give each agent a self-contained brief: the goal, the files involved, the interfaces to keep, and the
  machine rule (syntax checks only on the dev box).
- **Scouting: Haiku at low effort** (`model: "haiku"`, `effort: "low"`). Read-only lookups: finding files, call sites,
  flags and schemas. Ask for a short answer, not file dumps.
- **The orchestrator itself:** splits the work, runs independent agents in parallel (agents that touch the same files
  run one after another), reviews each result against the brief, fixes small gaps directly, and reports.
- Delegate even when a task looks small, unless writing the brief would cost more than doing the work (a one-line
  fix, for example).
