"""Frozen-backbone readout experiments on cached per-option features (see scrm.lossgrid for the loss grid).

Cache layout (<cache_dir>/<split>/): shard_XXXXX.safetensors with `feat_L{layer}` [M,d] bf16 (one row per
(set, variant, option), canonical option order inside a variant block) + shard_XXXXX.parquet with one row per
(set, variant) (decision_set_id, variant, perm_json, n_options, row_offset, tiers_json, probabilities_json,
abs_label_json, label_kind, family, source_id, split, truncated).

Heads: linear | mlp | set (SetEncoder). Readouts: ce, brier, bt, sigmoid (scrm.losses). Training length, data order and
init depend only on (seed, config), never on which loss terms are active, so arms are directly comparable."""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .feat_metrics import compute_metrics
from .losses import (Readouts, bt_loss, brier_loss, center_loss, perm_consistency_loss, sigmoid_loss,
                     softmax_ce_loss)
from .model import SetEncoder
from .wandb_utils import NullRun

TERMS = ("ce", "brier", "bt", "sigmoid")

FEAT_DEFAULTS: dict = {
    "seed": 0,
    "cache_dir": "outputs/feat_cache",
    "data": {"layer": 24, "train_split": "train", "val_split": "validation", "test_splits": ["test"],
             "max_options": None, "drop_truncated": False,
             "augment_variants": False},          # train on a random presentation variant per set (else variant 0 only)
    "model": {"head": "linear", "hidden": 256, "dropout": 0.1,       # mlp
              "d_set": 256, "set_layers": 2, "set_heads": 4, "set_ffn_mult": 2, "set_dropout": 0.1},
    "train": {"steps": 2000, "batch_size": 64, "eval_batch_size": 128, "lr": 1e-3, "weight_decay": 0.01,
              "warmup_steps": 50, "grad_clip": 1.0, "device": "auto", "readout_lr_mult": 10.0, "log_every": 100},
    "loss": {
        "terms": ["ce"], "weights": {"ce": 1.0, "brier": 1.0, "bt": 1.0, "sigmoid": 1.0}, "weight": 1.0,
        "w_center": 0.01, "w_perm": 0.0,
        "ce_temperature": 1.0, "brier_temperature": 1.0, "bt_tau": 1.0, "alpha_init": 1.0, "max_alpha": 100.0,
        "bias_init": None,                 # None -> -log(avg set size - 1) (SigLIP-style prior), measured on train
        "per_family_bias": False, "learnable_scales": True,
    },
    "wandb": {"enabled": False, "project": None, "entity": None, "run_name": None, "tags": [], "mode": None},
}


# --------------------------------------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------------------------------------

def _floats(js, n):
    """JSON list (entries may be null) / null -> float32 [n] with NaN for missing."""
    if js is None or (isinstance(js, float) and math.isnan(js)):
        return np.full(n, np.nan, np.float32)
    v = json.loads(js) if isinstance(js, str) else js
    if v is None:
        return np.full(n, np.nan, np.float32)
    return np.array([np.nan if x is None else x for x in v], np.float32)


@dataclass
class SetRec:
    id: str
    n: int
    rows: dict                    # variant -> global row offset into FeatSplit.feats
    tiers: np.ndarray             # [n] canonical order, lower = better
    probs: np.ndarray | None      # [n] or None
    abs: np.ndarray               # [n] NaN = excluded from the sigmoid loss
    source: str
    kind: str
    family: str


@dataclass
class FeatSplit:
    feats: torch.Tensor           # [M,d] bf16
    sets: list = field(default_factory=list)
    d: int = 0

    def __len__(self):
        return len(self.sets)


def load_split(cache_dir: str, split: str, layer, max_options: int | None = None, drop_truncated: bool = False) -> FeatSplit:
    from safetensors import safe_open
    key = f"feat_L{layer}"
    with open(os.path.join(cache_dir, "manifest.json")) as f:
        manifest = json.load(f)
    state = manifest.get("splits", {}).get(split)
    if not state or not state.get("complete"):
        raise ValueError(f"Feature split {split!r} is incomplete; resume extraction before training heads")
    # The manifest, not a directory glob, defines committed data. Ignore any uncommitted crash leftovers.
    pq = [os.path.join(cache_dir, split, rec["name"] + ".parquet")
          for rec in sorted(state["shards"], key=lambda s: s["name"])
          if rec.get("files", rec.get("n_records", 0) > 0)]
    if not pq:
        raise FileNotFoundError(f"no shards under {os.path.join(cache_dir, split)}")
    chunks, base, rows, meta = [], 0, {}, {}
    for p in pq:
        df = pd.read_parquet(p)
        with safe_open(p[:-len(".parquet")] + ".safetensors", "pt") as f:
            if key not in f.keys():
                raise KeyError(f"{key} not in {p} (have {[k for k in f.keys() if k.startswith('feat_')]})")
            chunks.append(f.get_tensor(key))
        for r in df.itertuples(index=False):
            rows.setdefault(r.decision_set_id, {})[int(r.variant)] = base + int(r.row_offset)
            if int(r.variant) == 0:
                meta[r.decision_set_id] = r
        base += chunks[-1].shape[0]
    sets = []
    for sid, r in meta.items():
        n = int(r.n_options)
        if (max_options and n > max_options) or (drop_truncated and bool(r.truncated)):
            continue
        probs = _floats(r.probabilities_json, n)
        sets.append(SetRec(sid, n, rows[sid], np.array(json.loads(r.tiers_json), np.int64),
                           None if np.isnan(probs).all() else np.nan_to_num(probs), _floats(r.abs_label_json, n),
                           str(r.source_id), str(r.label_kind), str(r.family)))
    feats = torch.cat(chunks) if len(chunks) > 1 else chunks[0]
    return FeatSplit(feats, sets, feats.shape[1])


def make_batch(split: FeatSplit, idxs, variants=0, fam_index: dict | None = None, drop_best: bool = False) -> dict:
    """Pad sets idxs to [B,Nmax]. variants: int or per-set list (falls back to 0 when a set lacks it).
    drop_best: remove the (single) tier-0 option from the set (mask False, tier -1)."""
    recs = [split.sets[i] for i in idxs]
    B, N = len(recs), max(r.n for r in recs)
    x = torch.zeros(B, N, split.d)
    tiers = torch.full((B, N), -1, dtype=torch.long)
    probs = torch.zeros(B, N)
    has_probs = torch.zeros(B, dtype=torch.bool)
    ab = torch.zeros(B, N)
    abm = torch.zeros(B, N, dtype=torch.bool)
    fam = torch.zeros(B, dtype=torch.long)
    for b, r in enumerate(recs):
        v = variants if isinstance(variants, int) else variants[b]
        off = r.rows.get(v, r.rows[0])
        x[b, :r.n] = split.feats[off:off + r.n].float()
        tiers[b, :r.n] = torch.from_numpy(r.tiers)
        if r.probs is not None:
            probs[b, :r.n] = torch.from_numpy(r.probs)
            has_probs[b] = True
        a = torch.from_numpy(r.abs)
        abm[b, :r.n] = ~torch.isnan(a)
        ab[b, :r.n] = torch.nan_to_num(a)
        if fam_index is not None:
            fam[b] = fam_index.get(r.family, len(fam_index))
    mask = tiers >= 0
    if drop_best:
        top = tiers.masked_fill(~mask, 10**9).argmin(1)
        ar = torch.arange(B)
        mask[ar, top] = False
        tiers[ar, top] = -1
        x[ar, top] = 0
        abm &= mask
    return {"x": x, "mask": mask, "tiers": tiers, "probs": probs, "has_probs": has_probs, "abs": ab, "abs_mask": abm,
            "fam": fam}


def _to(b: dict, device) -> dict:
    return {k: v.to(device) for k, v in b.items()}


# --------------------------------------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------------------------------------

class FeatModel(nn.Module):
    """Standardise features (train mean/std) -> head -> reward [B,N] (0 at padding)."""

    def __init__(self, d: int, mcfg: dict, mu: torch.Tensor | None = None, sd: torch.Tensor | None = None):
        super().__init__()
        self.kind = mcfg["head"]
        self.register_buffer("mu", torch.zeros(d) if mu is None else mu.clone())
        self.register_buffer("sd", torch.ones(d) if sd is None else sd.clone())
        if self.kind == "linear":
            self.net = nn.Linear(d, 1)
        elif self.kind == "mlp":
            self.net = nn.Sequential(nn.Linear(d, mcfg["hidden"]), nn.GELU(), nn.Dropout(mcfg["dropout"]),
                                     nn.Linear(mcfg["hidden"], 1))
        elif self.kind == "set":
            self.net = SetEncoder(d, mcfg["d_set"], mcfg["set_layers"], mcfg["set_heads"], mcfg["set_ffn_mult"],
                                  mcfg["set_dropout"])
        else:
            raise ValueError(f"unknown head {self.kind!r}")

    def forward(self, x, mask):
        x = (x - self.mu) / self.sd
        if self.kind == "set":
            return self.net(x, mask)
        return self.net(x).squeeze(-1).masked_fill(~mask, 0.0)


# --------------------------------------------------------------------------------------------------------------------
# loss
# --------------------------------------------------------------------------------------------------------------------

def _mean_valid(l, valid):
    return (l * valid).sum() / valid.sum().clamp(min=1)


def feat_loss(r, b: dict, readouts: Readouts, terms, weights: dict, lcfg: dict, r_perm=None):
    """Weighted sum of the selected terms (+ small center loss, + optional perm consistency). Each term is averaged over
    the sets where it is defined. Returns (total scalar, parts {name: scalar tensor})."""
    tiers = b["tiers"]
    _, n_pairs = bt_loss(r, tiers)
    rank_valid = ((n_pairs > 0) | (b["has_probs"] & (tiers >= 0).any(1))).float()
    parts = {}
    if "ce" in terms:
        parts["ce"] = _mean_valid(softmax_ce_loss(r, tiers, readouts.T_ce(), b["probs"], b["has_probs"]), rank_valid)
    if "brier" in terms:
        parts["brier"] = _mean_valid(brier_loss(r, tiers, readouts.T_brier(), b["probs"], b["has_probs"]), rank_valid)
    if "bt" in terms:
        l, n = bt_loss(r, tiers, tau=readouts.tau_bt())
        parts["bt"] = _mean_valid(l, (n > 0).float())
    if "sigmoid" in terms:
        bias = readouts.bias_for(b["fam"])
        l, n = sigmoid_loss(r, b["abs"], b["abs_mask"], readouts.alpha(), bias)
        parts["sigmoid"] = _mean_valid(l, (n > 0).float())
    total = sum(weights[t] * parts[t] for t in parts)
    if lcfg.get("w_center", 0.0):
        parts["center"] = _mean_valid(center_loss(r, tiers), (tiers >= 0).any(1).float())
        total = total + lcfg["w_center"] * parts["center"]
    if lcfg.get("w_perm", 0.0) and r_perm is not None:
        parts["perm"] = _mean_valid(perm_consistency_loss(r, r_perm, tiers >= 0), (tiers >= 0).any(1).float())
        total = total + lcfg["w_perm"] * parts["perm"]
    return total, parts


# --------------------------------------------------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, readouts, split: FeatSplit, terms, tcfg: dict, device, fam_index=None, stability: bool = True) -> dict:
    """Metrics (feat_metrics.compute_metrics) on variant 0 + abstention + order stability vs the other variants."""
    model.eval()
    S, bs = len(split), tcfg["eval_batch_size"]
    Nmax = max(r.n for r in split.sets)
    T = float(readouts.T_ce() if "ce" in terms else readouts.T_brier() if "brier" in terms else 1.0)
    sig = "sigmoid" in terms

    def run(idxs, variants=0, drop_best=False):
        out = np.zeros((len(idxs), Nmax))
        bias = np.zeros(len(idxs))
        for i in range(0, len(idxs), bs):
            b = _to(make_batch(split, idxs[i:i + bs], variants if isinstance(variants, int) else variants[i:i + bs],
                               fam_index, drop_best), device)
            r = model(b["x"], b["mask"]).float().cpu().numpy()
            out[i:i + len(r), :r.shape[1]] = r
            bias[i:i + len(r)] = readouts.bias_for(b["fam"]).detach().cpu().numpy() * np.ones(len(r))
        return out, bias

    all_idx = list(range(S))
    scores, bias = run(all_idx)
    full = make_batch_meta(split, Nmax)
    after = after_mask = None
    elig = [i for i in all_idx if (split.sets[i].tiers == split.sets[i].tiers.min()).sum() == 1 and split.sets[i].n >= 2]
    if elig:
        a, _ = run(elig, drop_best=True)
        after = np.zeros((S, Nmax))
        after_mask = np.zeros((S, Nmax), bool)
        after[elig] = a
        for i in elig:
            m = full["tiers"][i] >= 0
            m[int(np.argmin(np.where(m, full["tiers"][i], 10**9)))] = False
            after_mask[i] = m
    alt = []
    if stability:
        maxv = max((max(r.rows) for r in split.sets), default=0)
        for v in range(1, maxv + 1):
            idx = [i for i in all_idx if v in split.sets[i].rows]
            if idx:
                alt.append((np.array(idx), run(idx, v)[0]))
    return compute_metrics(
        scores, full["tiers"], full["probs"], full["has_probs"], full["abs"], full["abs_mask"], full["sources"],
        full["kinds"], temperature=T, alpha=float(readouts.alpha()) if sig else None, bias=bias if sig else None,
        after_scores=after, after_mask=after_mask, alt_scores=alt)


def make_batch_meta(split: FeatSplit, Nmax: int) -> dict:
    S = len(split)
    tiers = np.full((S, Nmax), -1, np.int64)
    probs = np.zeros((S, Nmax))
    ab = np.full((S, Nmax), np.nan)
    has = np.zeros(S, bool)
    for i, r in enumerate(split.sets):
        tiers[i, :r.n] = r.tiers
        ab[i, :r.n] = r.abs
        if r.probs is not None:
            probs[i, :r.n] = r.probs
            has[i] = True
    return {"tiers": tiers, "probs": probs, "has_probs": has, "abs": np.nan_to_num(ab), "abs_mask": ~np.isnan(ab),
            "sources": [r.source for r in split.sets], "kinds": [r.kind for r in split.sets]}


# --------------------------------------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------------------------------------

_SPLITS: dict = {}


def get_split(cfg: dict, name: str) -> FeatSplit:
    d = cfg["data"]
    k = (os.path.abspath(cfg["cache_dir"]), name, str(d["layer"]), d["max_options"], d["drop_truncated"])
    if k not in _SPLITS:
        _SPLITS[k] = load_split(cfg["cache_dir"], name, d["layer"], d["max_options"], d["drop_truncated"])
    return _SPLITS[k]


def _device(tcfg):
    return torch.device("cuda" if tcfg["device"] == "auto" and torch.cuda.is_available() else
                        "cpu" if tcfg["device"] == "auto" else tcfg["device"])


def term_weights(lcfg: dict, terms) -> dict:
    return {t: float(lcfg["weights"][t]) * float(lcfg.get("weight", 1.0)) for t in terms}


def train_run(cfg: dict, wb=None, log=print):
    """Train one head with cfg["loss"]["terms"]; returns (model, readouts, info). Deterministic given (seed, cfg)."""
    wb = wb or NullRun()
    tcfg, lcfg, dcfg = cfg["train"], cfg["loss"], cfg["data"]
    terms = [t for t in TERMS if t in lcfg["terms"]]
    assert terms, "need at least one loss term"
    weights = term_weights(lcfg, terms)
    seed, device = int(cfg["seed"]), _device(tcfg)
    split = get_split(cfg, dcfg["train_split"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # standardisation stats / readout priors from the train variant-0 rows
    rows = torch.cat([split.feats[r.rows[0]:r.rows[0] + r.n] for r in split.sets]).float()
    mu, sd = rows.mean(0), rows.std(0).clamp(min=1e-4)
    avg_n = float(np.mean([r.n for r in split.sets]))
    fam_index = None
    if lcfg["per_family_bias"]:
        fam_index = {f: i for i, f in enumerate(sorted({r.family for r in split.sets}))}
    model = FeatModel(split.d, cfg["model"], mu, sd).to(device)
    readouts = Readouts(avg_n, len(fam_index) + 1 if fam_index else 1, lcfg["ce_temperature"], lcfg["brier_temperature"],
                        lcfg["bt_tau"], lcfg["alpha_init"], lcfg["bias_init"], lcfg["max_alpha"],
                        lcfg["learnable_scales"]).to(device)
    ro = [p for p in readouts.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": list(model.parameters()), "lr": tcfg["lr"], "weight_decay": tcfg["weight_decay"]},
         {"params": ro, "lr": tcfg["lr"] * tcfg["readout_lr_mult"], "weight_decay": 0.0}])
    base = [g["lr"] for g in opt.param_groups]
    steps, bsz, warm = int(tcfg["steps"]), int(tcfg["batch_size"]), int(tcfg["warmup_steps"])
    S = len(split)
    queue = np.zeros(0, dtype=np.int64)
    t0, ema = time.time(), None
    model.train()
    for step in range(steps):
        while len(queue) < bsz:                          # epoch boundary: reshuffle (arm-independent rng use)
            queue = np.concatenate([queue, rng.permutation(S)])
        idxs, queue = queue[:bsz].tolist(), queue[bsz:]
        if dcfg["augment_variants"]:
            vs = [int(rng.choice(sorted(split.sets[i].rows))) for i in idxs]
        else:
            vs = 0
        f = min(1.0, (step + 1) / max(warm, 1)) * (0.5 * (1 + math.cos(math.pi * step / max(steps, 1))))
        for g, b0 in zip(opt.param_groups, base):
            g["lr"] = b0 * f
        b = _to(make_batch(split, idxs, vs, fam_index), device)
        r = model(b["x"], b["mask"])
        rp = None
        if lcfg["w_perm"]:
            rp = model(_to(make_batch(split, idxs, 1, fam_index), device)["x"], b["mask"])
        loss, parts = feat_loss(r, b, readouts, terms, weights, lcfg, rp)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if tcfg["grad_clip"]:
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + ro, tcfg["grad_clip"])
        opt.step()
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if (step + 1) % tcfg["log_every"] == 0 or step + 1 == steps:
            log(f"step {step + 1}/{steps} loss {loss.item():.4f} ema {ema:.4f} ({time.time() - t0:.0f}s)")
            wb.log({"train/loss": loss.item(), **{f"train/{k}": v.item() for k, v in parts.items()}}, step + 1)
    return model, readouts, {"train_loss_ema": ema, "fam_index": fam_index, "terms": terms, "weights": weights}


def run_experiment(cfg: dict, eval_splits: list, log=print) -> dict:
    """Train once, evaluate on the named splits. Returns a JSON-able result dict."""
    from .wandb_utils import init_wandb
    wb = init_wandb(cfg["wandb"], cfg) if cfg["wandb"].get("enabled") else NullRun()
    t0 = time.time()
    model, readouts, info = train_run(cfg, wb, log)
    device = _device(cfg["train"])
    metrics = {}
    for name in eval_splits:
        metrics[name] = evaluate(model, readouts, get_split(cfg, name), info["terms"], cfg["train"], device,
                                 info["fam_index"])
        wb.log({f"eval/{name}/{k}": v for k, v in metrics[name]["all"].items()}, cfg["train"]["steps"])
    wb.finish()
    return {"terms": info["terms"], "weights": info["weights"], "lr": cfg["train"]["lr"], "seed": cfg["seed"],
            "steps": cfg["train"]["steps"], "head": cfg["model"]["head"], "layer": cfg["data"]["layer"],
            "train_loss_ema": info["train_loss_ema"], "readouts": readouts.values(), "metrics": metrics,
            "seconds": time.time() - t0}
