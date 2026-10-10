# Upstream

Vendored from https://github.com/jaredpalmer/kev at commit `fc4a17e194bca35a48c57211392ec49ea54a94ac`. This in-repo copy is
our fork.

## Vendored subset

`kev/` package, `pyproject.toml`, `uv.lock`, `LICENSE`, `.python-version`, `.gitattributes`, `README.md`, `AGENTS.md`,
`evals/v7/decision-v7`, `evals/v4/transfer-v4`, `scripts/calibrate_checkpoint.py`, `experiments/q35-4b.json`.
`README.md` and `AGENTS.md` are replaced by one-line pointers to upstream.

Removed from the vendored `kev/` package as unused by the study's closure (train, benchmark, checkpoint, predictors and what they
import): `anchors`, `autoresearch`, `calibrate`, `compare`, `evaluate`, `jev`, `mirror`, `plot`, `publish`, `serve`, `study_v3`,
`transfer_v9`. In `predictors.py`, `JevPredictor`, `JevRefused`, `oversize` and `PRICE_PER_MILLION` were removed too (they drove the
non-vendored `playground/scripts/jev-evaluate.mjs`). `experiment.py`'s `EVALUATOR_FILES` no longer names `evaluate.py` / `study_v3.py`.
Comments that mention `kev.serve` etc. are upstream history. `pyproject.toml` (incl. the `serve` extra) and `uv.lock` are untouched.

Not vendored: `runs/`, other evals, playground, space, `modal_app`, docs. Upstream tests were removed (they need
unvendored files) and replaced by `tests/test_pref.py`.

This is its own uv project (torch 2.8 locked), separate from udm's (torch 2.10).

## Changes from upstream

Defaults keep upstream behaviour bit-identical (`--head pointer`, `--input_format kev`, `--loss ce`).

- `kev/model.py`: `ScalarHead` (one `Linear(d, 1)` reward per option) and `make_head`/`HEAD_KINDS`; `INPUT_FORMATS = ("kev", "chat")`,
  `chat_template_parts` / `chat_template_digest` / `chat_splits` / `chat_prompt` (chat input layout built from the tokenizer's own
  chat template, no added tokens); `encode(chat=True)` adds `enc["chat"]`; `forward_chat_batch`;
  `_kev_format_only` guard on the pointer/Kev-only probability paths.
- `kev/checkpoint.py`: `head_kind`, `input_format`, `chat_template_sha256` in checkpoint meta and `COMPAT_FIELDS`; the MLX backend
  rejects non-pointer/non-Kev checkpoints. `kev/predictors.py`: the long-row check also counts `enc["chat_row_max"]`.
- `kev/pref.py` (new): Bradley-Terry pair loss from hard or soft labels, per-option BCE, BSR / L2 reward regularizers.
- `kev/train.py`: flags `--head`, `--input_format {kev,chat}`, `--loss {ce,pref}`, `--reg`, `--reg_w`, `--bce_w`, `--bce_types`;
  `objective` and `batch_loss` route to the pref loss; pref stats (`pair_acc`, reward mean/std/absmax) in logs.
- `tests/test_pref.py` (new): CPU tests of the pref losses, chat layout and checkpoint round-trip.
- `scripts/pref/` (new): setup, arm definitions, run/eval/wave scripts and reward report for the preference study
  (see `scripts/pref/README.md`).

## Reproduce the diff

```
git clone https://github.com/jaredpalmer/kev /tmp/kev-upstream
git -C /tmp/kev-upstream checkout fc4a17e194bca35a48c57211392ec49ea54a94ac
for p in kev pyproject.toml uv.lock LICENSE .python-version .gitattributes README.md AGENTS.md \
         evals/v7/decision-v7 evals/v4/transfer-v4 scripts/calibrate_checkpoint.py experiments/q35-4b.json; do
  diff -ru /tmp/kev-upstream/$p kev/$p      # from the udm repo root
done
```
