"""Thin, failure-tolerant W&B wrapper. Env: WANDB_PROJECT, WANDB_ENTITY, WANDB_RUN_NAME, WANDB_TAGS, WANDB_MODE
(online|offline|disabled). If wandb is missing, disabled, or init fails, logging becomes a no-op."""
from __future__ import annotations

import os


class NullRun:
    enabled = False

    def log(self, *a, **k): pass
    def log_hist(self, *a, **k): pass
    def log_group_table(self, *a, **k): pass
    def finish(self): pass


class WandbRun:
    enabled = True

    def __init__(self, run, wandb):
        self.run, self.wandb = run, wandb

    def log(self, data: dict, step: int):
        try:
            self.run.log(data, step=step)
        except Exception as e:  # never crash training because of logging
            print(f"[wandb] log failed: {e}")

    def log_hist(self, name: str, values, step: int):
        try:
            self.run.log({name: self.wandb.Histogram(values)}, step=step)
        except Exception as e:
            print(f"[wandb] hist failed: {e}")

    def log_group_table(self, name: str, metrics: dict, step: int):
        try:
            cols = ["group", "n", "pair_acc", "pair_acc_micro", "top1", "mrr", "ndcg", "kendall_tau", "loss"]
            t = self.wandb.Table(columns=cols)
            for g, d in sorted(metrics.items()):
                t.add_data(g, *[d.get(c, float("nan")) for c in cols[1:]])
            self.run.log({name: t}, step=step)
        except Exception as e:
            print(f"[wandb] table failed: {e}")

    def finish(self):
        try:
            self.run.finish()
        except Exception:
            pass


def init_wandb(wcfg: dict, config: dict, out_dir: str | None = None, resume_id: str | None = None):
    mode = os.environ.get("WANDB_MODE") or wcfg.get("mode")
    if not wcfg.get("enabled", True) or mode == "disabled":
        return NullRun()
    try:
        import wandb
    except Exception:
        print("[wandb] not installed; logging disabled")
        return NullRun()
    tags = wcfg.get("tags") or []
    if os.environ.get("WANDB_TAGS"):
        tags = tags + os.environ["WANDB_TAGS"].split(",")
    try:
        run = wandb.init(project=os.environ.get("WANDB_PROJECT") or wcfg.get("project") or "scrm",
                         entity=os.environ.get("WANDB_ENTITY") or wcfg.get("entity"),
                         name=os.environ.get("WANDB_RUN_NAME") or wcfg.get("run_name"),
                         tags=tags, config=config, mode=mode, dir=out_dir, id=resume_id,
                         resume="allow" if resume_id else None)
        return WandbRun(run, wandb)
    except Exception as e:
        print(f"[wandb] init failed ({e}); logging disabled")
        return NullRun()
