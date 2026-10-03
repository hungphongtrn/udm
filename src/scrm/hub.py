"""Hugging Face Hub checkpoint backup and `hf://` checkpoint paths.

Uploads run in the background (one at a time) so training is not blocked; `wait()` must be called before a local
checkpoint directory that may still be uploading is replaced or deleted (train.save_ckpt does this).
"""
from __future__ import annotations

import os

HF_PREFIX = "hf://"


def resolve_ckpt(path: str) -> str:
    """`hf://org/repo[@revision]/sub/dir` -> local snapshot dir of that subfolder; local paths pass through."""
    if not path.startswith(HF_PREFIX):
        return path
    parts = path[len(HF_PREFIX):].strip("/").split("/")
    if len(parts) < 2:
        raise ValueError(f"expected hf://org/repo[@revision]/subdir, got {path!r}")
    name, rev = (parts[1].split("@", 1) + [None])[:2]
    repo, sub = f"{parts[0]}/{name}", "/".join(parts[2:])
    from huggingface_hub import snapshot_download
    local = snapshot_download(repo, revision=rev, allow_patterns=[f"{sub}/*"] if sub else None)
    out = os.path.join(local, sub)
    if not os.path.isfile(os.path.join(out, "scrm_config.json")):
        raise FileNotFoundError(f"{path}: no SCRM checkpoint (scrm_config.json) at {sub or 'repo root'}")
    return out


class HubSync:
    def __init__(self, hcfg: dict, out_dir: str):
        self.repo_id = hcfg.get("repo_id")
        self.enabled = bool(self.repo_id)
        self.future = None
        if not self.enabled:
            return
        from huggingface_hub import HfApi
        self.api = HfApi()
        self.run = hcfg.get("run") or os.path.basename(os.path.normpath(out_dir))
        self.ignore = [] if hcfg.get("include_optimizer") else ["trainer_state.pt"]
        self.api.create_repo(self.repo_id, private=bool(hcfg.get("private", True)), exist_ok=True)
        print(f"[hub] backing up checkpoints to https://huggingface.co/{self.repo_id}/tree/main/{self.run}", flush=True)

    def wait(self):
        """Block until the pending upload finishes; a failed upload is reported, not raised (training goes on)."""
        if self.future is None:
            return
        f, self.future = self.future, None
        try:
            f.result()
        except Exception as e:  # noqa: BLE001 - network / auth errors must not kill a long run
            print(f"[hub] upload failed: {type(e).__name__}: {e}", flush=True)

    def push(self, local_dir: str, slot: str, message: str):
        """Upload local_dir to {run}/{slot} in the background (replacing that folder's files)."""
        if not self.enabled:
            return
        self.wait()
        self.future = self.api.upload_folder(
            repo_id=self.repo_id, folder_path=local_dir, path_in_repo=f"{self.run}/{slot}",
            commit_message=f"{self.run}/{slot}: {message}", ignore_patterns=self.ignore,
            delete_patterns=["*"], run_as_future=True)
