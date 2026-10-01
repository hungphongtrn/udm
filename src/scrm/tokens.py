"""Special tokens used to delimit state / candidate set / candidates."""
from __future__ import annotations

STATE_START = "<|state_start|>"
STATE_END = "<|state_end|>"
SET_START = "<|candidate_set_start|>"
CAND_START = "<|candidate_start|>"
CAND_END = "<|candidate_end|>"
SET_END = "<|candidate_set_end|>"
SPECIAL_TOKENS = [STATE_START, STATE_END, SET_START, CAND_START, CAND_END, SET_END]


def prepare_tokenizer(tok):
    """Add the SCRM special tokens (idempotent). Returns (tokenizer, ids dict, n_base_vocab).

    For Qwen3 the embedding matrix has more rows (151936) than the tokenizer has tokens
    (151669), so the six new ids fit in existing (unused) rows and no resize is needed.
    """
    n_base = len(tok)
    missing = [t for t in SPECIAL_TOKENS if tok.convert_tokens_to_ids(t) in (None, tok.unk_token_id) or t not in tok.get_vocab()]
    if missing:
        tok.add_special_tokens({"additional_special_tokens": list(tok.additional_special_tokens) + missing}
                               if getattr(tok, "additional_special_tokens", None) else
                               {"additional_special_tokens": missing})
    ids = {t: tok.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS}
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else SPECIAL_TOKENS[-1]
    tok.padding_side = "right"
    return tok, ids, n_base
