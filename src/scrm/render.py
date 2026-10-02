"""Row parsing, text rendering, tokenisation, candidate subsampling, sequence assembly (over-long sets are dropped)."""
from __future__ import annotations

import json
import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ----------------------------------------------------------------------------- text rendering


def _loads(s: Any) -> Any:
    if isinstance(s, (bytes, bytearray)):
        s = s.decode("utf-8", "replace")
    if isinstance(s, str):
        try:
            return json.loads(s)
        except Exception:
            return s
    return s


def _dump(v: Any) -> str:
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False, separators=(", ", ": "))


def render_instruction(obj: Any) -> str:
    """JSON string -> as is. Object {type, instructions, criteria?} -> instructions + criteria lines."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        parts = []
        ins = obj.get("instructions", obj.get("instruction"))
        if ins is not None:
            parts.append(_dump(ins).strip())
        crit = obj.get("criteria")
        if crit:
            lines = []
            if isinstance(crit, dict):
                lines = [f"- {k}: {_dump(v)}" for k, v in crit.items()]
            elif isinstance(crit, (list, tuple)):
                lines = [f"- {_dump(v)}" for v in crit]
            else:
                lines = [f"- {_dump(crit)}"]
            parts.append("Criteria:\n" + "\n".join(lines))
        if not parts:
            parts.append(_dump(obj))
        return "\n".join(parts)
    return _dump(obj)


def render_state(obj: Any) -> str:
    """JSON string -> as is. Anything else -> compact readable JSON."""
    if obj is None:
        return ""
    return obj if isinstance(obj, str) else _dump(obj)


@dataclass
class Example:
    decision_set_id: str
    source_id: str
    family: str
    label_kind: str
    source_split: str
    instruction: str
    state: str
    choice_ids: list[str]
    texts: list[str]
    tiers: list[int]

    @classmethod
    def from_raw(cls, instruction, state, candidates: list[str]) -> "Example":
        """Unlabelled example for inference (tiers all 0)."""
        return cls("", "", "", "", "", render_instruction(instruction), render_state(state),
                   [str(i) for i in range(len(candidates))], [c if isinstance(c, str) else _dump(c) for c in candidates],
                   [0] * len(candidates))


def parse_row(row: dict, use_candidate_rows: bool = True) -> Example | None:
    """Contract row -> Example, or None if untrainable (<2 candidates / no pair)."""
    try:
        options = _loads(row["options_json"])
        tier_list = _loads(row["tier_json"])
        if not isinstance(options, dict) or not isinstance(tier_list, list):
            return None
        tier_of = {}
        for k, grp in enumerate(tier_list):
            for cid in grp:
                tier_of[cid] = k
        order: list[str] = []
        cr = row.get("candidate_rows") if use_candidate_rows else None
        if cr:
            order = [c["choice_id"] for c in sorted(cr, key=lambda c: c["candidate_order"])]
            order = [c for c in order if c in options]
            seen = set(order)
            order += [c for c in options if c not in seen]
        else:
            order = list(options.keys())
        order = [c for c in order if c in tier_of]
        if len(order) < 2 or len({tier_of[c] for c in order}) < 2:
            return None
        return Example(
            decision_set_id=str(row.get("decision_set_id") or ""), source_id=str(row.get("source_id") or ""),
            family=str(row.get("family") or ""), label_kind=str(row.get("label_kind") or ""),
            source_split=str(row.get("source_split") or ""),
            instruction=render_instruction(_loads(row.get("instruction_json"))),
            state=render_state(_loads(row.get("state_json"))),
            choice_ids=order, texts=[_dump(options[c]) for c in order], tiers=[tier_of[c] for c in order])
    except Exception:
        return None


# ----------------------------------------------------------------------------- tokenisation / assembly

# max_len: tokens per sequence (prompt + graded option). The state is never truncated: a set whose longest sequence
# exceeds max_len is dropped (assemble -> None).
DEFAULT_RENDER = {"max_len": 8192, "max_candidates": 64,
                  # wrap the sequence as one user turn of the tokenizer's chat template and end right after
                  # the assistant header ("<|im_start|>assistant\n"; any <think> the template appends is cut)
                  "chat_template": True, "system_prompt": None,
                  # Plain-text layout, no new tokens. ONE SEQUENCE PER GRADED OPTION: the user turn shows the state,
                  # the instruction, ALL shown options and then asks to grade one of them; the sequence stops at the
                  # assistant header and option k's reward is read at its last token:
                  #   <|im_start|>user\nState: {state}\n\nInstruction: {instruction}\n\nOptions:\nOption 1: {c1}\n...\n
                  #   \nGrade this choice: Option k: {ck}<|im_end|>\n<|im_start|>assistant\n
                  "state_label": "State: ", "instruction_label": "Instruction: ",
                  "options_header": "Options:\n", "option_label": "Option {k}: ",
                  "grade_prompt": "\nGrade this choice: ",
                  # max options graded per set (one sequence each; the prompt still shows all kept options).
                  # None = grade every shown option. Subsampling keeps tier-0 + one per other tier.
                  "max_graded": None, "eval_max_graded": None}

_SENTINEL = "@@SCRM_BODY@@"


def chat_wrap_text(tokenizer, system_prompt: str | None = None) -> tuple[str, str]:
    """(prefix, suffix) strings around the user content, from the tokenizer's chat template with
    add_generation_prompt=True, truncated right after the last assistant header."""
    msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + \
        [{"role": "user", "content": _SENTINEL}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    pre, post = text.split(_SENTINEL)
    head = post.rfind("assistant")
    if head < 0:
        raise ValueError(f"chat template has no assistant header: {post!r}")
    nl = post.find("\n", head)
    post = post[: (nl + 1) if nl >= 0 else head + len("assistant")]
    return pre, post


@dataclass
class TokExample:
    ex: Example
    instr_ids: list[int]
    state_ids: list[int]
    cand_ids: list[list[int]]
    tiers: np.ndarray


@dataclass
class Item:
    """One decision set = one sequence per graded option: a shared prefix (state, instruction, all options,
    "Grade this choice: ") + a per-option suffix ("Option k: {text}" + end of user turn + assistant header)."""
    prefix: np.ndarray         # shared token ids
    suffixes: list             # list[np.ndarray], one per graded slot
    tiers: np.ndarray          # tier per graded slot
    order: np.ndarray          # tokex candidate index for each graded slot (display order)
    choice_ids: list[str]
    family: str
    source_id: str
    decision_set_id: str
    tok: TokExample = field(repr=False, default=None)
    kept: list = field(default_factory=list)     # shown candidates (tokex indices, sorted)
    graded: list = field(default_factory=list)   # graded candidates (tokex indices, sorted)

    @property
    def seqs(self) -> list:
        """Full per-option sequences (prefix + suffix)."""
        return [np.concatenate([self.prefix, x]) for x in self.suffixes]

    @property
    def n_tokens(self) -> int:
        """Packed token cost: every graded option is a full sequence prefix + suffix."""
        return len(self.suffixes) * len(self.prefix) + sum(len(x) for x in self.suffixes)


def subsample_indices(tiers: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    """Pick <= max_n candidates keeping ALL tier-0 items (as far as budget allows) and at least one
    from every other tier; remaining slots filled randomly. Returned sorted (canonical order)."""
    n = len(tiers)
    if n <= max_n:
        return np.arange(n)
    levels = sorted(set(tiers.tolist()))
    top = np.where(tiers == levels[0])[0]
    n_other_levels = len(levels) - 1
    keep: list[int] = []
    # reserve room for one representative of other tiers (at most max_n - 1 of them keep a pair possible)
    reserve = min(n_other_levels, max(1, max_n // 2)) if n_other_levels else 0
    top_cap = max(1, max_n - max(reserve, 1))
    top_keep = rng.permutation(top)[:top_cap] if len(top) > top_cap else top
    keep.extend(top_keep.tolist())
    reps = []
    for lv in levels[1:]:
        reps.append(int(rng.choice(np.where(tiers == lv)[0])))
    rng.shuffle(reps)
    for r in reps:
        if len(keep) < max_n:
            keep.append(r)
    chosen = set(keep)
    rest = np.array([i for i in range(n) if i not in chosen])
    room = max_n - len(keep)
    if room > 0 and len(rest):
        keep.extend(rng.permutation(rest)[:room].tolist())
    return np.array(sorted(keep))


class Renderer:
    def __init__(self, tokenizer, rcfg: dict | None = None):
        self.tok = tokenizer
        self.cfg = {**DEFAULT_RENDER, **(rcfg or {})}
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.nl2 = tokenizer("\n\n", add_special_tokens=False)["input_ids"]
        nl = tokenizer("\n", add_special_tokens=False)["input_ids"]
        self.option_end_id = nl[-1]          # reward read-out position: newline ending each option line
        self.header = tokenizer(self.cfg["options_header"], add_special_tokens=False)["input_ids"]
        self._labels: dict[int, list[int]] = {}
        self.chat_prefix, self.chat_suffix = [], []
        if self.cfg.get("chat_template"):
            if not getattr(tokenizer, "chat_template", None):
                raise ValueError("render.chat_template=true but the tokenizer has no chat template "
                                 "(set data.render.chat_template=false to disable)")
            pre, post = chat_wrap_text(tokenizer, self.cfg.get("system_prompt"))
            self.chat_prefix = tokenizer(pre, add_special_tokens=False)["input_ids"]
            self.chat_suffix = tokenizer(post, add_special_tokens=False)["input_ids"]
        self.grade = tokenizer(self.cfg["grade_prompt"], add_special_tokens=False)["input_ids"]
        self.state_label = tokenizer(self.cfg["state_label"], add_special_tokens=False)["input_ids"]
        self.instr_label = tokenizer(self.cfg["instruction_label"], add_special_tokens=False)["input_ids"]

    def label(self, k: int) -> list[int]:
        if k not in self._labels:
            self._labels[k] = self.tok(self.cfg["option_label"].format(k=k), add_special_tokens=False)["input_ids"]
        return self._labels[k]

    # --- tokenisation
    def tokenize(self, ex: Example) -> TokExample:
        c = self.cfg
        # Huge states only need to be known to overflow: tokenise a head first and the full text only if that head
        # still fits. A head of > max_len tokens proves the set will be dropped (its ids are then incomplete).
        head = 16 * c["max_len"]
        texts = [ex.instruction, ex.state[:head]] + list(ex.texts)
        ids = self.tok(texts, add_special_tokens=False)["input_ids"]
        if len(ex.state) > head and len(ids[1]) <= c["max_len"]:
            ids[1] = self.tok(ex.state, add_special_tokens=False)["input_ids"]
        instr = ids[0] + (self.nl2 if ex.instruction else [])
        cands = [x or [self.nl2[0] if self.nl2 else 0] for x in ids[2:]]
        return TokExample(ex, instr, ids[1], cands, np.asarray(ex.tiers))

    # --- assembly
    def assemble(self, t: TokExample, rng: np.random.Generator | None, shuffle: bool, kept=None,
                 relax: bool = False, graded=None) -> Item | None:
        """Build an Item. `rng`=None & shuffle=False -> canonical order, deterministic subsampling.
        relax=True (inference): never drop the example for lack of trainable pairs.
        Returns None when the longest sequence (prompt + graded option) exceeds max_len: nothing is truncated."""
        c = self.cfg
        sub_rng = rng if rng is not None else np.random.default_rng(
            zlib.crc32(t.ex.decision_set_id.encode()) if t.ex.decision_set_id else 0)
        n = len(t.cand_ids)
        tiers = t.tiers
        if len(t.state_ids) > c["max_len"]:
            return None
        if kept is None:
            idxs = subsample_indices(tiers, c["max_candidates"], sub_rng).tolist() if n > c["max_candidates"] \
                else list(range(n))
        else:
            idxs = list(kept)
        cand = t.cand_ids
        if len(idxs) < 1 or (not relax and len(set(tiers[idxs].tolist())) < 2):
            return None
        kept_final = sorted(idxs)
        if graded is None:
            mg = c.get("max_graded")
            if mg and len(kept_final) > mg:
                sel = subsample_indices(tiers[kept_final], mg, sub_rng)
                graded = sorted(kept_final[j] for j in sel)
            else:
                graded = list(kept_final)
        graded = list(graded)
        if not relax and len(set(tiers[graded].tolist())) < 2:
            return None
        display = list(kept_final)
        if shuffle:
            (rng or sub_rng).shuffle(display)
        state = t.state_ids
        prompt = list(self.chat_prefix)
        if state:
            prompt += self.state_label + state + self.nl2
        if t.instr_ids:
            prompt += self.instr_label + t.instr_ids
        prompt += self.header
        gset = set(graded)
        order = []
        for k, i in enumerate(display, 1):
            prompt += self.label(k) + cand[i] + [self.option_end_id]
        prompt += self.grade
        sufs = []
        for k, i in enumerate(display, 1):
            if i in gset:
                order.append(i)
                sufs.append(np.asarray(self.label(k) + cand[i] + self.chat_suffix, dtype=np.int64))
        # order-independent bound (longest option label), so reshuffle() of a kept Item never overflows
        lab = max(len(self.label(k)) for k in range(1, len(display) + 1))
        if len(prompt) + lab + max(len(cand[i]) for i in graded) + len(self.chat_suffix) > c["max_len"]:
            return None
        order_a = np.asarray(order)
        it = Item(np.asarray(prompt, dtype=np.int64), sufs, tiers[order_a].astype(np.int64), order_a, [t.ex.choice_ids[i] for i in order], t.ex.family,
                  t.ex.source_id, t.ex.decision_set_id, tok=t)
        it.kept = kept_final
        it.graded = sorted(graded)
        return it

    def reshuffle(self, item: Item, rng: np.random.Generator) -> Item:
        return self.assemble(item.tok, rng, True, kept=item.kept, graded=item.graded)
