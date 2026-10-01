"""SCRM model: Qwen3.5 text backbone (+LoRA) -> hidden state at the end of each option line -> set transformer -> scalar reward."""
from __future__ import annotations

import contextlib
import json
import os
from typing import Any

import torch
import torch.nn as nn

from .config import DEFAULTS, deep_update
from .tokens import prepare_tokenizer


class SetEncoder(nn.Module):
    """Bidirectional pre-LN transformer over candidates; no positional embeddings (permutation equivariant)."""

    def __init__(self, d_in, d_set=768, layers=2, heads=8, ffn_mult=2, dropout=0.1, head_hidden=None, input_norm=True):
        super().__init__()
        self.in_norm = nn.LayerNorm(d_in) if input_norm else nn.Identity()
        self.proj = nn.Linear(d_in, d_set)
        self.layers = layers
        if layers > 0:
            layer = nn.TransformerEncoderLayer(d_set, heads, dim_feedforward=ffn_mult * d_set, dropout=dropout,
                                               activation="gelu", batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, layers, norm=None, enable_nested_tensor=False)
        else:
            self.encoder = None
        hh = head_hidden or d_set // 2
        self.head = nn.Sequential(nn.LayerNorm(d_set), nn.Linear(d_set, hh), nn.SiLU(), nn.Linear(hh, 1))

    def forward(self, E: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """E [B,N,d], mask [B,N] bool (True = real) -> rewards [B,N] fp32, 0 at masked slots."""
        dev_type = E.device.type
        with torch.autocast(device_type=dev_type, enabled=False):
            x = self.proj(self.in_norm(E.float()))
            if self.encoder is not None:
                kpm = ~mask
                kpm = kpm.clone()
                kpm[:, 0] = kpm[:, 0] & ~kpm.all(dim=1)   # fully padded rows: keep slot 0 visible (avoid NaN)
                x = self.encoder(x, src_key_padding_mask=kpm)
            r = self.head(x).squeeze(-1)
            r = torch.nan_to_num(r).masked_fill(~mask, 0.0)
        return r


class SCRM(nn.Module):
    def __init__(self, backbone: nn.Module, hidden: int, cfg: dict, backbone_trainable: bool = True):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg
        self.backbone_trainable = backbone_trainable
        self.set_encoder = SetEncoder(hidden, cfg["d_set"], cfg["set_layers"], cfg["set_heads"], cfg["set_ffn_mult"],
                                      cfg["set_dropout"], cfg.get("head_hidden"), cfg.get("input_norm", True))
        self.tokenizer = None
        self.render_cfg: dict | None = None

    def encode(self, input_ids, attention_mask):
        ctx = contextlib.nullcontext() if self.backbone_trainable else torch.no_grad()
        with ctx:
            return self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def forward(self, input_ids, attention_mask, read_pos, seq_b, seq_slot, candidate_mask, max_tokens=None):
        """input_ids [S, L]: one row per graded option (shared prompt listing all options + that option as the
        assistant response). The end-of-turn hidden state of each row is that option's embedding; the set encoder
        then scores the options of each set jointly. Returns rewards [B, N] (0 at padded slots).
        max_tokens: optional cap on padded tokens per backbone call (rows are processed in chunks)."""
        S, L = input_ids.shape
        step = S if not max_tokens else max(1, int(max_tokens) // max(L, 1))
        outs = []
        for a in range(0, S, step):
            h = self.encode(input_ids[a:a + step], attention_mask[a:a + step])
            outs.append(h[torch.arange(h.size(0), device=h.device), read_pos[a:a + step]])
        e = torch.cat(outs, 0)
        B, N = candidate_mask.shape
        E = e.new_zeros(B, N, e.size(-1))
        E = E.index_put((seq_b, seq_slot), e)
        return self.set_encoder(E, candidate_mask)

    def score(self, b: dict, max_tokens=None):
        """Forward on a collated batch dict."""
        return self(b["input_ids"], b["attention_mask"], b["read_pos"], b["seq_b"], b["seq_slot"],
                    b["candidate_mask"], max_tokens=max_tokens)

    # ----- parameter groups / persistence -----
    def head_state_dict(self) -> dict:
        return {f"set_encoder.{k}": v.detach().cpu() for k, v in self.set_encoder.state_dict().items()}

    def load_head_state_dict(self, sd: dict):
        own = {k[len("set_encoder."):]: v for k, v in sd.items() if k.startswith("set_encoder.")}
        self.set_encoder.load_state_dict(own)

    def lora_parameters(self):
        return [p for n, p in self.backbone.named_parameters() if p.requires_grad]

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        if hasattr(self.backbone, "save_pretrained") and self.cfg["lora"]["enabled"] and not self.cfg["freeze_backbone"]:
            self.backbone.save_pretrained(os.path.join(path, "adapter"))
        torch.save(self.head_state_dict(), os.path.join(path, "scrm_head.pt"))
        with open(os.path.join(path, "scrm_config.json"), "w") as f:
            json.dump({"model": self.cfg, "render": self.render_cfg}, f, indent=2)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(os.path.join(path, "tokenizer"))

    # ----- inference helpers -----
    @torch.no_grad()
    def rank(self, instruction, state, candidates: list[str], device=None, max_tokens: int | None = 16384) -> list[dict]:
        """Score candidates (given order; no shuffling). Returns [{index, reward}] sorted by reward desc;
        `index` is the position in the input list."""
        from .render import Renderer, Example
        was_training = self.training
        self.eval()
        dev = device or next(self.set_encoder.parameters()).device
        rr = Renderer(self.tokenizer, {**(self.render_cfg or {}), "max_graded": None})   # grade every candidate
        ex = Example.from_raw(instruction, state, candidates)
        item = rr.assemble(rr.tokenize(ex), rng=None, shuffle=False, relax=True)
        from .collator import collate
        from .collator import to_device
        b = to_device(collate([item], pad_id=rr.pad_id), dev)
        amp = dev.type == "cuda"
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            r = self.score(b, max_tokens=max_tokens)[0]
        out = [{"index": int(item.order[k]), "reward": float(r[k])} for k in range(len(item.order))]
        out.sort(key=lambda d: -d["reward"])
        self.train(was_training)
        return out

    @staticmethod
    def pairwise_probability(r_i, r_j, tau: float = 1.0):
        """P(i preferred over j) = sigmoid((r_i - r_j)/tau)."""
        return torch.sigmoid((torch.as_tensor(r_i, dtype=torch.float32) - torch.as_tensor(r_j, dtype=torch.float32)) / tau)


def pairwise_probability(r_i, r_j, tau: float = 1.0):
    return SCRM.pairwise_probability(r_i, r_j, tau)


def _resolve_attn(impl: str, device: torch.device) -> str:
    if impl != "auto":
        return impl
    if device.type == "cuda":
        try:
            import flash_attn  # noqa: F401
            return "flash_attention_2"
        except Exception:
            pass
    return "sdpa"


def _apply_liger(mcfg: dict, device: torch.device) -> bool:
    """Liger kernels through the HF integration (`liger_kernel.transformers`, the same entry point HF Trainer's
    `use_liger_kernel` uses). Patches the HF modeling classes BEFORE the model is built (RMSNorm incl. q/k norms,
    SwiGLU MLP). No LM head is used, so fused-linear-cross-entropy is off. Triton kernels are CUDA-only."""
    if not mcfg.get("liger_kernel") or device.type != "cuda":
        return False
    from transformers import AutoConfig
    from liger_kernel.transformers import monkey_patch as lk
    model_type = AutoConfig.from_pretrained(mcfg["name_or_path"]).model_type
    fn = lk.MODEL_TYPE_TO_APPLY_LIGER_FN.get(model_type)
    if fn is None:
        print(f"[scrm] liger: no kernels for model_type={model_type}; skipped")
        return False
    import inspect
    kw = dict(rope=False, cross_entropy=False, fused_linear_cross_entropy=False, rms_norm=True, swiglu=True)
    fn(**{k: v for k, v in kw.items() if k in inspect.signature(fn).parameters})
    print(f"[scrm] liger kernels applied ({fn.__name__}: rms_norm, swiglu)")
    return True


def _text_only(m: nn.Module) -> nn.Module:
    """Qwen3.5 checkpoints are vision-language (Qwen3_5Model = visual + language_model). Keep the text decoder only."""
    lm = getattr(m, "language_model", None)
    if lm is None:
        return m
    if hasattr(m, "visual"):
        del m.visual
    return lm


def _load_backbone(mcfg: dict, device: torch.device, tokenizer_len: int):
    import transformers
    from transformers import AutoModel

    dtype = torch.bfloat16 if (mcfg["dtype"] == "bfloat16" and device.type == "cuda") or \
        (mcfg["dtype"] == "bfloat16" and device.type != "cuda" and mcfg["name_or_path"] != "tiny") else torch.float32
    if mcfg["name_or_path"] == "tiny":
        from .tiny import make_tiny_backbone
        return make_tiny_backbone(mcfg["tiny"], 512, dtype=torch.float32).to(device)
    _apply_liger(mcfg, device)
    kw: dict[str, Any] = {"attn_implementation": _resolve_attn(mcfg["attn_implementation"], device)}
    kw["dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"] = dtype
    if mcfg.get("quantize_4bit"):
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
        kw["device_map"] = {"": device.index or 0}
        return _text_only(AutoModel.from_pretrained(mcfg["name_or_path"], **kw))
    m = _text_only(AutoModel.from_pretrained(mcfg["name_or_path"], **kw))
    return m.to(device)


def build_scrm(mcfg: dict, device: torch.device | str = "cpu", tokenizer=None, adapter_dir: str | None = None,
               seed: int = 0) -> tuple["SCRM", Any]:
    """Build tokenizer + backbone (+LoRA) + set block + head. `adapter_dir` loads a trained LoRA adapter."""
    from transformers import AutoTokenizer
    full = deep_update(json.loads(json.dumps(DEFAULTS["model"])), mcfg)
    mcfg = full
    device = torch.device(device)
    if tokenizer is None:
        if mcfg["name_or_path"] == "tiny":
            from .tiny import make_tiny_tokenizer
            tokenizer = make_tiny_tokenizer()
        else:
            tokenizer = AutoTokenizer.from_pretrained(mcfg["name_or_path"])
    tokenizer = prepare_tokenizer(tokenizer)
    backbone = _load_backbone(mcfg, device, len(tokenizer))
    hidden = backbone.config.hidden_size
    freeze = mcfg["freeze_backbone"]
    lora = mcfg["lora"]

    for p in backbone.parameters():
        p.requires_grad_(False)
    if mcfg["gradient_checkpointing"] and not freeze:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        backbone.config.use_cache = False
    backbone_trainable = False
    if lora["enabled"] and not freeze:
        from peft import LoraConfig, get_peft_model, PeftModel
        if adapter_dir:
            backbone = PeftModel.from_pretrained(backbone, adapter_dir, is_trainable=True)
        else:
            lc = LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                            target_modules=lora["target_modules"], bias="none",
                            use_rslora=lora.get("use_rslora", True))
            backbone = get_peft_model(backbone, lc)
        backbone_trainable = True
    model = SCRM(backbone, hidden, mcfg, backbone_trainable=backbone_trainable)
    model.set_encoder.to(device)
    model.tokenizer = tokenizer
    return model, tokenizer


def load_scrm(ckpt_dir: str, device: str | torch.device = "cpu", merged_ok: bool = True) -> "SCRM":
    """Load a checkpoint written by SCRM.save_pretrained (LoRA adapter + head + tokenizer + config)."""
    from transformers import AutoTokenizer
    with open(os.path.join(ckpt_dir, "scrm_config.json")) as f:
        saved = json.load(f)
    mcfg = saved["model"]
    tok = AutoTokenizer.from_pretrained(os.path.join(ckpt_dir, "tokenizer"))
    adapter = os.path.join(ckpt_dir, "adapter")
    mcfg = dict(mcfg)
    if os.path.isdir(os.path.join(ckpt_dir, "backbone_merged")):
        mcfg["name_or_path"] = os.path.join(ckpt_dir, "backbone_merged")
        mcfg["lora"] = dict(mcfg["lora"], enabled=False)
        adapter = None
    model, tok = build_scrm(mcfg, device, tokenizer=tok, adapter_dir=adapter if adapter and os.path.isdir(adapter) else None)
    sd = torch.load(os.path.join(ckpt_dir, "scrm_head.pt"), map_location="cpu")
    model.load_head_state_dict(sd)
    model.render_cfg = saved["render"]
    model.eval()
    return model
