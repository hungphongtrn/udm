"""Preference-reward branch: kev.pref losses, the chat input format, the scalar head, and tiny CPU training runs.
Run: .venv/bin/python -m pytest tests/test_pref.py -q   (on the training machine; nothing here needs weights or a network)
"""
import inspect
import json
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from kev.model import DecisionModel, SPECIAL, chat_common, chat_prompt, chat_splits, chat_template_digest, chat_template_parts, encode, user_tokens
from kev.pref import bce_loss, preference_loss, preference_pairs, reward_reg

IM = ["<|im_start|>", "<|im_end|>"]
CHATML = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
          "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
HEAD, TAIL = "<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n"


@pytest.fixture(scope="module", autouse=True)
def reference_deltanet_kernels():
    """The tiny models run on CPU, but on the training machine transformers binds its Gated DeltaNet hooks to the installed
    causal-conv1d / fla packages, which need CUDA tensors. Run the hooks' PyTorch references instead (exact fp32: the parity
    tests compare paths at 1e-4)."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    names = ("causal_conv1d_fn", "causal_conv1d_update", "torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule")
    with pytest.MonkeyPatch.context() as mp:
        for name in names:   # functools.wraps chains __wrapped__ from the dispatching wrapper down to the reference
            mp.setattr(m, name, inspect.unwrap(getattr(m, name)))
        yield


# --- losses (kev.pref) ---------------------------------------------------------------------------------------------

def test_bt_hard_label_is_mean_softplus_over_rejected():
    """A hard label y beats every other option: loss = mean_j softplus(r_j - r_y)."""
    z = torch.tensor([0.3, -1.2, 0.8, 0.1])
    loss, rw, rl = preference_loss(z, {"label": 2}, "cpu")
    want = torch.stack([F.softplus(z[j] - z[2]) for j in (0, 1, 3)]).mean()
    assert torch.allclose(loss, want, atol=1e-6)
    assert torch.equal(rw, z[[2, 2, 2]]) and torch.equal(rl, z[[0, 1, 3]])


@pytest.mark.parametrize("kind", ["bsr", "l2"])
def test_reward_reg_matches_reference(kind):
    """The BSR reference (softplus(rl - rw).mean() + 1e-3 * cat([rw, rl]).mean().square()) is reproduced; l2 squares instead."""
    z = torch.tensor([0.5, -0.7, 1.1])
    loss, rw, rl = preference_loss(z, {"label": 0}, "cpu")
    ref_rw, ref_rl = z[[0, 0]].float().flatten(), z[[1, 2]].float().flatten()
    pref = F.softplus(ref_rl - ref_rw).mean()
    ref = torch.cat([ref_rw, ref_rl]).mean().square() if kind == "bsr" else torch.cat([ref_rw, ref_rl]).square().mean()
    assert torch.allclose(loss, pref, atol=1e-6)
    assert torch.allclose(loss + 1e-3 * reward_reg(kind, rw, rl), pref + 1e-3 * ref, atol=1e-7)
    with pytest.raises(ValueError):
        reward_reg("nope", rw, rl)


def test_soft_target_pairs():
    """Soft target t: every pair i < j with t_i + t_j > 0, p = t_i / (t_i + t_j); uniform gives 0.5."""
    i, j, p = preference_pairs({"target": [0.5, 0.5, 0.0, 0.0]}, 4)
    assert (i, j, p) == ([0, 0, 0, 1, 1], [1, 2, 3, 2, 3], [0.5, 1.0, 1.0, 1.0, 1.0])   # (2, 3) has no mass: skipped
    i, j, p = preference_pairs({"target": [0.6, 0.2, 0.0, 0.0]}, 4)
    assert (i[0], j[0], p[0]) == (0, 1, pytest.approx(0.75))
    assert (2, 3) not in list(zip(i, j))                  # 0 + 0 pair skipped
    i, j, p = preference_pairs({"target": [1 / 3] * 3}, 3)
    assert len(i) == 3 and p == pytest.approx([0.5] * 3)


def test_uniform_soft_target_pulls_rewards_together():
    """The BCE against p = 0.5 is minimal (log 2) at equal rewards."""
    q = {"target": [0.5, 0.5], "label": 0}
    equal, _, _ = preference_loss(torch.zeros(2), q, "cpu")
    apart, _, _ = preference_loss(torch.tensor([2.0, -2.0]), q, "cpu")
    assert torch.allclose(equal, torch.tensor(0.6931472)) and apart > equal


def test_bce_loss_hard_labels_only():
    """None for soft targets and for question types outside `types`; otherwise one-hot BCE-with-logits."""
    z = torch.tensor([0.4, -0.3, 1.0])
    q = {"label": 1, "qtype": "choice"}
    assert bce_loss(z, {**q, "target": [0.2, 0.3, 0.5]}, "cpu", ("choice",)) is None
    assert bce_loss(z, q, "cpu", ("noul", "score")) is None
    want = F.binary_cross_entropy_with_logits(z, torch.tensor([0.0, 1.0, 0.0]))
    assert torch.allclose(bce_loss(z, q, "cpu", ("choice",)), want)


def test_single_option_question_has_no_pairs():
    assert preference_loss(torch.tensor([0.3]), {"label": 0}, "cpu") is None


# --- the chat encoding ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory):
    """As tests/test_unit.py's tiny_base (hybrid 2-layer Qwen3.5, word-level tokenizer, 16 labelled requests), with the
    a ChatML chat template (<|im_start|> / <|im_end|> special tokens) and the words the chat prompt uses."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3_5ForCausalLM, Qwen3_5TextConfig
    root = tmp_path_factory.mktemp("tiny")
    words = ("it is charged twice which team billing shipping refund angry the customer user assistant "
             "State : Instruction Options - Grade this option yes no none of these").split()
    vocab = {t: i for i, t in enumerate(["<unk>", "<pad>", *SPECIAL, *IM, *words])}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>")); tk.pre_tokenizer = pre_tokenizers.Whitespace()
    config = Qwen3_5TextConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                               head_dim=16, linear_num_value_heads=2, linear_num_key_heads=1, linear_key_head_dim=8, linear_value_head_dim=8,
                               layer_types=["linear_attention", "full_attention"], pad_token_id=1)
    torch.manual_seed(0)
    Qwen3_5ForCausalLM(config).to(torch.bfloat16).save_pretrained(root / "base")
    ptk = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>", additional_special_tokens=[*SPECIAL, *IM])
    ptk.chat_template = CHATML
    ptk.save_pretrained(root / "base")
    rows = [{"state": "the customer is charged twice" + " it" * i, "questions": {
        "team": {"type": "choice", "instructions": "which team", "criteria": {"billing": None, "shipping": None, "refund": None}, "label": ["billing", "shipping", "refund"][i % 3]},
        "angry": {"type": "noul", "instructions": "is the customer angry", "label": i % 2 == 0}}} for i in range(16)]
    (root / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def tok_rec(tiny_base):
    from transformers import AutoTokenizer
    from kev.data import load_records, materialize
    return AutoTokenizer.from_pretrained(tiny_base / "base"), materialize(load_records(tiny_base / "data.jsonl")[3])


def test_chat_template_parts(tok_rec):
    """Head and tail of the template around the user content; none is invented when the tokenizer has no template."""
    from transformers import AutoTokenizer
    tok, _ = tok_rec
    assert chat_template_parts(tok) == (HEAD, TAIL)
    bare = AutoTokenizer.from_pretrained(tok.name_or_path); bare.chat_template = None
    with pytest.raises(ValueError, match="chat_template"):
        chat_template_parts(bare)


def test_chat_encoding_layout(tok_rec):
    """Prefix ids identical across options and ending right after 'Grade this option: '; branch = option ids + tail ids;
    positions continue the prefix; every row ends with the tail's last token. The Kev format carries no chat splits."""
    tok, rec = tok_rec
    assert "chat" not in encode(tok, rec)
    enc = encode(tok, rec, chat=True)
    head_ids = tok(HEAD + "State:\n", add_special_tokens=False).input_ids
    tail_ids = tok(TAIL, add_special_tokens=False).input_ids
    state_ids = user_tokens(tok, rec["state"])
    assert len(enc["chat"]) == len(rec["questions"])
    for q, (ids, pos, rows) in zip(rec["questions"], enc["chat"]):
        listing = "\n".join(f"- {o}" for o in q["options"])
        rest = tok(f"\n\nInstruction: {q['instr']}\n\nOptions:\n{listing}\n\nGrade this option: ", add_special_tokens=False).input_ids
        assert ids == head_ids + state_ids + rest                       # one prefix for all K options
        assert ids[-4:] == tok("Grade this option:", add_special_tokens=False).input_ids[-4:]
        assert pos == list(range(len(ids)))
        assert len(rows) == len(q["options"])
        for o, r in zip(q["options"], rows):
            assert r["ids"] == user_tokens(tok, o) + tail_ids
            assert r["pos"] == list(range(len(ids), len(ids) + len(r["ids"])))
            assert r["ids"][-1] == tail_ids[-1] == tok.convert_tokens_to_ids("assistant")
    assert enc["chat_row_max"] == max(len(ids) + len(r["ids"]) for ids, _, rows in enc["chat"] for r in rows)
    text, n = chat_prompt(tok, rec, 0, 0)
    assert n == len(enc["chat"][0][0]) + len(enc["chat"][0][2][0]["ids"]) and isinstance(text, str)


def test_chat_pointwise_encoding_layout(tok_rec):
    """Arm H's layout (chat_pointwise): the prefix carries no option text and is one prefix for all K options; the branches
    and their lengths are the chat layout's, and their positions continue this prefix."""
    tok, rec = tok_rec
    pw, full = encode(tok, rec, chat=True, chat_listing=False), encode(tok, rec, chat=True)
    head_ids = tok(HEAD + "State:\n", add_special_tokens=False).input_ids
    state_ids = user_tokens(tok, rec["state"])
    assert len(enc_chat := pw["chat"]) == len(rec["questions"])
    for q, (p_ids, p_pos, p_rows), (ids, _, rows) in zip(rec["questions"], enc_chat, full["chat"]):
        mid = f"\n\nInstruction: {q['instr']}"
        assert p_ids == head_ids + state_ids + tok(f"{mid}\n\nGrade this option: ", add_special_tokens=False).input_ids
        assert ids == head_ids + state_ids + tok(
            f"{mid}\n\nOptions:\n" + "\n".join(f"- {o}" for o in q["options"]) + "\n\nGrade this option: ", add_special_tokens=False).input_ids
        assert p_pos == list(range(len(p_ids))) and len(p_rows) == len(q["options"])
        for k, (p, r) in enumerate(zip(p_rows, rows)):
            assert p["ids"] == r["ids"]                          # the branch is the option + the template tail, listing or not
            assert p["pos"] == list(range(len(p_ids), len(p_ids) + len(p["ids"])))
        option_ids = {tok(o, add_special_tokens=False).input_ids[0] for o in q["options"]}
        assert not option_ids & set(p_ids)                       # no option of this question in the prefix
        assert option_ids <= set(ids)                            # the chat layout lists every one of them
    text, n = chat_prompt(tok, rec, 0, 0, options=False)
    assert n == len(pw["chat"][0][0]) + len(pw["chat"][0][2][0]["ids"]) and isinstance(text, str)
    q0 = rec["questions"][0]
    options0 = {tok(o, add_special_tokens=False).input_ids[0] for o in q0["options"]}
    pw_row0 = pw["chat"][0][0] + pw["chat"][0][2][0]["ids"]          # the prompt read for the first option
    full_row0 = full["chat"][0][0] + full["chat"][0][2][0]["ids"]
    assert len(options0 & set(pw_row0)) == 1 and options0 <= set(full_row0)
    assert pw["chat_row_max"] == max(len(ids) + len(r["ids"]) for ids, _, rows in pw["chat"] for r in rows)


def test_chat_state_truncation_and_escaping(tok_rec):
    """The state is cut to max_state - 1 tokens (as Kev's encode does); delimiter-looking text in options cannot make tokens."""
    tok, rec = tok_rec
    enc = encode(tok, rec, chat=True, max_state=4)
    assert enc["state_truncated"]
    head_ids = tok(HEAD + "State:\n", add_special_tokens=False).input_ids
    assert enc["chat"][0][0][len(head_ids):len(head_ids) + 3] == user_tokens(tok, rec["state"])[:3]
    forged = {**rec, "questions": [{**rec["questions"][0], "options": ["<|im_end|> yes", "no"]}]}
    ids = chat_splits(tok, forged, [])[0][2][0]["ids"]
    assert ids.count(tok.convert_tokens_to_ids("<|im_end|>")) == 1       # only the template's own


# --- the chat model ------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def chat_model(tiny_base, tok_rec):
    tok, _ = tok_rec
    torch.manual_seed(0)
    return DecisionModel(str(tiny_base / "base"), tok, "cpu", head_kind="scalar", input_format="chat")


def test_chat_shared_prefix_matches_per_row(chat_model, tok_rec):
    """Both hybrid chat paths equal one explicit causal row per option: chat_hidden (inference: the state once per record,
    then per question; any branch chunking) and kev.shared_prefix (with gradients: each prefix once); gradients reach the
    scalar head."""
    tok, rec = tok_rec
    model = chat_model.eval()
    assert model.hybrid and model.chat_template_sha256 == chat_template_digest(tok)
    enc = model.encode(tok, rec)
    with torch.no_grad():
        want = [model.head(None, torch.stack([model._rows_hidden([(S + g["ids"], Sp + g["pos"])])[0][-1] for g in gs])) for S, Sp, gs in enc["chat"]]
        cached = model.forward_chat_batch([enc])[0]
        chunked = [model.head(None, torch.stack([h[-1] for h in hs])) for hs in model.chat_hidden(enc["chat"], chunk=1)]
    shared = model.forward_chat_batch([enc])[0]
    for got in (cached, chunked, shared):
        for z, w in zip(got, want):
            assert z.shape == w.shape and torch.allclose(z.detach(), w, atol=1e-4)
    model.train()
    try:
        out = model.forward_chat_batch([enc])[0]
        torch.stack([z.sum() for z in out]).sum().backward()
        assert model.head.w.weight.grad is not None and model.head.w.weight.grad.abs().sum() > 0
    finally:
        model.zero_grad(); model.eval()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a CUDA device (kev.cuda_graphs)")
@pytest.mark.parametrize("fmt", ["chat", "chat_pointwise"])
def test_chat_graphs_match_chat_hidden(tiny_base, tok_rec, fmt):
    """The graphed chat passes (kev.cuda_graphs.chat, what dindex_engine serves on CUDA) score a record as chat_hidden's
    eager passes do: the same tokens at the same positions, so the rewards differ by floating-point reassociation only
    (fp32 here: atol 1e-4, the tolerance the parity test above uses). Covers a record with several questions (state pass,
    one unpadded eager pass per question prefix, one graphed pass for the option rows), one whose six questions' prefixes
    have six distinct lengths and so cannot share one bucket (their continuations must run unpadded: a padded DeltaNet state
    has advanced past the prefix's last real token), and one whose single question is the state pass itself, both before any
    capture and after (replays) -- for both chat layouts: the pointwise prefix (no option listing) is the same tokens at the
    same positions to these passes, which read the splits and never the listing."""
    from kev.cuda_graphs import CudaGraphs, bucket
    from kev.data import load_records, materialize
    tok, _ = tok_rec
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False   # as a LocalPredictor CUDA session runs (10-bit mantissa moves logits)
    torch.manual_seed(0)
    model = DecisionModel(str(tiny_base / "base"), tok, "cuda", head_kind="scalar", input_format=fmt).eval()
    model.graphs = CudaGraphs(model.lm, model.pad_id)
    records = [materialize(r) for r in load_records(tiny_base / "data.jsonl")[:2]]
    q0 = records[0]["questions"][0]
    records.append({**records[0], "questions": [{**q0, "instr": q0["instr"] + " it" * j} for j in range(6)]})   # six prefixes, one token apart
    records.append({**records[0], "questions": records[0]["questions"][:1]})   # one question: no continuation row
    encs = [model.encode(tok, r) for r in records]
    assert all(model.graphs.admits_chat(e["chat"]) for e in encs)
    grow = next(e["chat"] for e in encs if len(e["chat"]) == 6)
    lens = [len(S) - chat_common(grow) for S, _, _ in grow]
    assert len(set(lens)) == 6 and any(bucket(L) > L for L in lens), lens   # consecutive: a bucketed pass would pad some, shifting their state

    def check(enc):
        with torch.no_grad():
            fast = model._chat_logits(enc["chat"])
            want = [model.head(None, torch.stack([h[-1] for h in rows])) for rows in model.chat_hidden(enc["chat"])]
        assert len(fast) == len(want)
        for got, w in zip(fast, want):
            assert got.shape == w.shape and torch.allclose(got.detach(), w, atol=1e-4), (got - w).abs().max()

    for enc in encs: check(enc)          # the first pass per bucket runs the graph's code eagerly (kev.cuda_graphs._replay)
    model.graphs.capture_pending()       # dindex_engine captures the same way between requests
    assert model.graphs.stats()["kept"] > 0, model.graphs.stats()
    for enc in encs: check(enc)          # replayed graphs


def test_chat_needs_scalar_head(tiny_base, tok_rec):


@pytest.fixture(scope="module")
def pointwise_model(tiny_base, tok_rec):
    tok, _ = tok_rec
    torch.manual_seed(0)
    return DecisionModel(str(tiny_base / "base"), tok, "cpu", head_kind="scalar", input_format="chat_pointwise")


def one_option_record(model, tok, rec, k):
    """The request that grades option k of the record's first question alone (what a pointwise model asks per response)."""
    from kev.data import materialize
    q = rec["questions"][0]
    return model.encode(tok, materialize({"state": rec["state"], "questions": {"team": {
        "type": "choice", "instructions": q["instr"], "criteria": {q["options"][k]: None}, "label": q["options"][k],
        "src": "single_option"}}}))


def test_chat_pointwise_reward_is_set_independent(pointwise_model, tok_rec):
    """Arm H: an option's reward is the reward of the one-option prompt that grades it, so the score of a response does not
    depend on the set it is scored in (the pointwise score Bradley-Terry assumes). Arm D's chat layout, which lists the
    options, is not (the test below)."""
    tok, rec = tok_rec
    model = pointwise_model.eval()
    enc = model.encode(tok, rec)
    assert [len(bs) for _, _, bs in enc["chat"]] == [len(q["options"]) for q in rec["questions"]]
    with torch.no_grad():
        set_zs = model.forward(enc)[0]
        alone = [model.forward(one_option_record(model, tok, rec, k))[0][0] for k in range(len(rec["questions"][0]["options"]))]
    for k, z in enumerate(alone):
        assert torch.allclose(set_zs[k], z.reshape(set_zs[k].shape), atol=1e-4)


def test_chat_reward_moves_with_the_option_set(chat_model, tok_rec):
    """The negative control for the test above: with the listing in the prefix (arm D), the same option's reward changes
    when its siblings are not in the prompt."""
    tok, rec = tok_rec
    model = chat_model.eval()
    enc = model.encode(tok, rec)
    with torch.no_grad():
        set_zs = model.forward(enc)[0]
        alone = [model.forward(one_option_record(model, tok, rec, k))[0][0] for k in range(len(rec["questions"][0]["options"]))]
    assert any(not torch.allclose(set_zs[k], z.reshape(set_zs[k].shape), atol=1e-4) for k, z in enumerate(alone))


@pytest.mark.parametrize("fmt", ["chat", "chat_pointwise"])
def test_chat_needs_scalar_head(tiny_base, tok_rec, fmt):
    tok, _ = tok_rec
    with pytest.raises(ValueError, match="scalar"):
        DecisionModel(str(tiny_base / "base"), tok, "cpu", head_kind="pointer", input_format=fmt)


@pytest.mark.parametrize("fmt", ["chat", "chat_pointwise"])
def test_chat_template_mismatch_refused(tiny_base, tok_rec, fmt):
    tok, _ = tok_rec
    with pytest.raises(ValueError, match="chat template mismatch"):
        DecisionModel(str(tiny_base / "base"), tok, "cpu", head_kind="scalar", input_format=fmt, chat_template_sha256="0" * 64)


# --- kev.train -----------------------------------------------------------------------------------------------------

def train_tiny(tiny_base, out, *args, monkeypatch):
    from kev import train
    argv = ["kev.train", "--base", str(tiny_base / "base"), "--data", str(tiny_base / "data.jsonl"), "--device", "cpu", "--batch", "2", "--lr", "1e-3", "--out", str(out), *args]
    monkeypatch.setattr(sys, "argv", argv)
    train.main()


STAGE1 = ("--head", "scalar", "--input_format", "chat", "--loss", "pref", "--reg", "bsr", "--lora", "4", "--max_steps", "2")
# arm H: arm D's recipe on the pointwise prompt (no option listing in the prefix)
STAGE1_POINTWISE = ("--head", "scalar", "--input_format", "chat_pointwise", "--loss", "pref", "--reg", "bsr", "--lora", "4", "--max_steps", "2")


def test_train_chat_pref_smoke(tiny_base, tmp_path, monkeypatch):
    """Scalar head + chat format + BT + BSR trains; the checkpoint records the head/format/chat template digest and scores K rewards per question."""
    from kev.checkpoint import Checkpoint
    from kev.data import load_records, materialize
    train_tiny(tiny_base, tmp_path / "a", *STAGE1, monkeypatch=monkeypatch)
    ck = Checkpoint(tmp_path / "a")
    tok, model = ck.load("cpu")
    assert (ck.meta.head_kind, ck.meta.input_format, ck.meta.chat_template_sha256) == ("scalar", "chat", chat_template_digest(tok))
    assert model.chat_template_sha256 == ck.meta.chat_template_sha256
    rec = materialize(load_records(tiny_base / "data.jsonl")[0])
    zs = model.forward(model.encode(tok, rec))
    assert [len(z) for z in zs] == [len(q["options"]) for q in rec["questions"]] and all(z.ndim == 1 for z in zs)


def test_train_chat_pointwise_smoke(tiny_base, tmp_path, monkeypatch):
    """Arm H end to end: the pointwise format trains with the same recipe, the checkpoint records the format, and the
    loaded model scores one reward per option from a prompt that carries no option text (nothing but state, instruction
    and the graded option)."""
    from kev.checkpoint import Checkpoint
    from kev.data import load_records, materialize
    train_tiny(tiny_base, tmp_path / "h", *STAGE1_POINTWISE, monkeypatch=monkeypatch)
    ck = Checkpoint(tmp_path / "h")
    tok, model = ck.load("cpu")
    assert (ck.meta.head_kind, ck.meta.input_format, ck.meta.chat_template_sha256) == ("scalar", "chat_pointwise", chat_template_digest(tok))
    assert model.input_format == "chat_pointwise"
    for req in load_records(tiny_base / "data.jsonl"):
        rec = materialize(req)
        enc = model.encode(tok, rec)
        assert [len(z) for z in model.forward(enc)] == [len(q["options"]) for q in rec["questions"]]
        for q, (ids, _, _) in zip(rec["questions"], enc["chat"]):
            assert not {tok(o, add_special_tokens=False).input_ids[0] for o in q["options"]} & set(ids)


def test_train_stage2_warm_start_and_compat(tiny_base, tmp_path, monkeypatch):
    """Stage 2 (--init_from stage 1, + BCE) runs; a pointer head cannot warm-start from a scalar checkpoint."""
    train_tiny(tiny_base, tmp_path / "a", *STAGE1, monkeypatch=monkeypatch)
    train_tiny(tiny_base, tmp_path / "b", "--init_from", str(tmp_path / "a"), "--bce_w", "0.5", "--loss", "pref", "--head", "scalar",
               "--input_format", "chat", "--lora", "4", "--max_steps", "2", monkeypatch=monkeypatch)
    assert (tmp_path / "b" / "head.pt").exists()
    with pytest.raises(ValueError, match="head_kind"):
        train_tiny(tiny_base, tmp_path / "c", "--init_from", str(tmp_path / "a"), "--head", "pointer", "--lora", "4", "--max_steps", "1", monkeypatch=monkeypatch)


@pytest.mark.parametrize("args", [("--input_format", "chat", "--head", "pointer"),
                                  ("--input_format", "chat_pointwise", "--head", "pointer"),
                                  ("--input_format", "chat", "--head", "scalar", "--option_isolation", "1"),
                                  ("--reg", "bsr", "--loss", "ce"),
                                  ("--loss", "pref", "--label_smoothing", "0.1")])
def test_train_rejects_invalid_flag_combinations(tiny_base, tmp_path, monkeypatch, args):
    with pytest.raises(SystemExit):
        train_tiny(tiny_base, tmp_path / "x", *args, "--max_steps", "1", monkeypatch=monkeypatch)


def test_train_pointer_pref_smoke(tiny_base, tmp_path, monkeypatch):
    """Arm B: the default pointer head in the Kev format trains under the preference loss."""
    train_tiny(tiny_base, tmp_path / "b", "--loss", "pref", "--lora", "4", "--max_steps", "1", monkeypatch=monkeypatch)
    assert (tmp_path / "b" / "head.pt").exists()


# --- RewardBench 2 aggregation (scripts/pref/rewardbench2.py) -------------------------------------------------------

def rewardbench2():
    """The script, imported by path (scripts/ is not a package): the aggregation over score rows is pure, so its
    unsupported policy is testable without a checkpoint, a GPU or the benchmark."""
    pytest.importorskip("datasets")
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts/pref/rewardbench2.py"
    spec = importlib.util.spec_from_file_location("rewardbench2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_prompt_credit_zeroes_unsupported_prompts():
    """A prompt with any unscoreable candidate earns 0 (not its chosen response's rank among the candidates that fit)."""
    rb = rewardbench2()
    row = {"id": "1", "scores": [1.0, 0.5, 0.4, 0.3], "unsupported": [False, False, True, False]}
    assert rb.prompt_credit(row) == 0.0
    assert rb.prompt_credit({**row, "unsupported": [False] * 4}) == 1.0
    shared_top = {**row, "scores": [1.0, 1.0, 0.0, 0.0], "unsupported": [False] * 4}
    assert rb.prompt_credit(shared_top) == 0.5             # credit shared at a tied top, as reroll_and_score_dataset scores
    assert rb.prompt_credit({**shared_top, "unsupported": [True, False, False, False]}) == 0.0


def test_ties_score_zeroes_unsupported_rows_but_keeps_denominators():
    """An unsupported Ties row stays in every denominator and contributes nothing: its accuracy flag is False, its pair is
    never a preferred/hard success, and its margin term is 0."""
    rb = rewardbench2()
    def row(kind, pid, num_correct, scores, unsupported=False):
        return {"id": f"{kind}:{pid}", "num_correct": num_correct, "scores": scores, "unsupported": [unsupported] * len(scores)}
    # ref:0 -> correct [3], wrong [2,1,0]: accurate, margin 3-2 = 1      tied:0 -> correct [3,2], wrong [1,0]: margin 2-1 = 1, diff 1
    # ref:1 -> correct [0,1], wrong [2,3]: not accurate, margin 0-2 = -2   tied:1 -> correct [4,3], wrong [0,0]: margin 3-0 = 3, diff 1
    base = [row("ref", 0, 1, [3, 2, 1, 0]), row("tied", 0, 2, [3, 2, 1, 0]),
            row("ref", 1, 2, [0, 1, 2, 3]), row("tied", 1, 2, [4, 3, 0, 0])]
    clean, details = rb.ties_score(base)
    assert clean == pytest.approx(0.30 * 1.0 + 0.30 * 0.5 + 0.20 * 0.5 + 0.01 * (math.tanh(0.0) + math.tanh(-3.0)) / 2)
    assert details["tied_accuracy"] == 1.0 and details["ref_accuracy"] == 0.5      # tied:0, tied:1 accurate; ref:0 only
    assert details["correctness_preferred"] == 0.5 and details["correctness_preferred_hard"] == 0.0
    assert details["correctness_margin_score"] == pytest.approx((math.tanh(0.0) + math.tanh(-3.0)) / 2)
    # one unsupported ref row: pair 1 scores nothing (preferred, hard, margin), both pairs stay in the denominators
    zeroed, zdet = rb.ties_score([base[0], base[1], row("ref", 1, 2, [0, 1, 2, 3], unsupported=True), base[3]])
    assert zdet == {"prompts": 4, "pairs": 2, "unsupported_rows": 1, "tied_accuracy": 1.0, "ref_accuracy": 0.5,
                    "correctness_preferred": 0.0, "correctness_preferred_hard": 0.0, "correctness_margin_score": pytest.approx(0.0)}
    assert zeroed == pytest.approx(0.30 * 1.0 + 0.30 * 0.5) and zeroed < clean
    # both rows of pair 0 unsupported: that pair's accuracy flags and margin are gone, pair 1 still scores
    both, bdet = rb.ties_score([row("ref", 0, 1, [3, 2, 1, 0], unsupported=True), row("tied", 0, 2, [3, 2, 1, 0], unsupported=True),
                                base[2], base[3]])
    assert (bdet["prompts"], bdet["pairs"], bdet["unsupported_rows"]) == (4, 2, 2)
    assert (bdet["tied_accuracy"], bdet["ref_accuracy"]) == (0.5, 0.0)
    assert (bdet["correctness_preferred"], bdet["correctness_preferred_hard"]) == (0.5, 0.0)
    assert bdet["correctness_margin_score"] == pytest.approx((0.0 + math.tanh(-3.0)) / 2)
    assert both == pytest.approx(0.30 * 0.5 + 0.20 * 0.5 + 0.01 * math.tanh(-3.0) / 2)
