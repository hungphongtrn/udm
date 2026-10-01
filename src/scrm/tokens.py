"""Tokenizer preparation. SCRM adds NO new tokens: inputs are plain text in the pretrained chat format."""
from __future__ import annotations


def prepare_tokenizer(tok):
    """Ensure a pad token and right padding (candidate read-out positions are absolute indices)."""
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok
