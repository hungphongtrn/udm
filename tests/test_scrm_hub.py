import concurrent.futures
import fnmatch
import os
import shutil

import pytest
import torch

import huggingface_hub
from scrm.config import load_config
from scrm.hub import resolve_ckpt
from scrm.model import load_scrm
from scrm.synth import write_synth
from scrm.train import train


class FakeHub:
    """Stands in for the Hub: upload_folder copies into a local 'remote' dir; snapshot_download serves it."""

    def __init__(self, root):
        self.root, self.commits, self.downloads = root, [], []

    def api(self):
        hub = self

        class Api:
            def create_repo(self, repo_id, private=True, exist_ok=False):
                os.makedirs(os.path.join(hub.root, repo_id), exist_ok=True)

            def upload_folder(self, repo_id, folder_path, path_in_repo, commit_message, ignore_patterns,
                              delete_patterns, run_as_future):
                dst = os.path.join(hub.root, repo_id, path_in_repo)
                if delete_patterns == ["*"]:
                    shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(folder_path, dst, dirs_exist_ok=True,
                                ignore=lambda d, names: [n for n in names if any(fnmatch.fnmatch(n, p) for p in ignore_patterns)])
                hub.commits.append((path_in_repo, commit_message))
                f = concurrent.futures.Future(); f.set_result(None)
                return f
        return Api()

    def snapshot_download(self, repo_id, revision=None, allow_patterns=None):
        self.downloads.append((repo_id, revision, allow_patterns))
        return os.path.join(self.root, repo_id)


@pytest.fixture()
def fake_hub(tmp_path, monkeypatch):
    h = FakeHub(str(tmp_path / "remote"))
    monkeypatch.setattr(huggingface_hub, "HfApi", h.api)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", h.snapshot_download)
    return h


def test_training_backs_up_weights_to_hub_and_loads_back(tmp_path, fake_hub):
    synth = write_synth(str(tmp_path / "synth"))
    out = tmp_path / "myrun"
    cfg = load_config("configs/debug_tiny.yaml", [f"data.local_dir={synth}", f"output_dir={out}", "train.max_steps=4",
                                                  "train.eval_every=2", "train.save_every=2", "hub.repo_id=me/scrm"])
    train(cfg)
    remote = os.path.join(fake_hub.root, "me/scrm/myrun")
    for slot in ("best", "last"):
        files = set(os.listdir(os.path.join(remote, slot)))
        assert {"adapter", "scrm_head.pt", "scrm_config.json", "tokenizer"} <= files
        assert "trainer_state.pt" not in files                      # weights only by default
    assert [c[0] for c in fake_hub.commits].count("myrun/last") == 2   # step 2 and step 4 (final)

    m_hub = load_scrm("hf://me/scrm@abc123/myrun/best")
    m_loc = load_scrm(str(out / "best"))
    assert fake_hub.downloads[-1] == ("me/scrm", "abc123", ["myrun/best/*"])
    for k, v in m_loc.head_state_dict().items():
        assert torch.equal(v, m_hub.head_state_dict()[k])


def test_resolve_ckpt_rejects_missing_checkpoint(fake_hub):
    os.makedirs(os.path.join(fake_hub.root, "me/scrm/run/best"))
    with pytest.raises(FileNotFoundError):
        resolve_ckpt("hf://me/scrm/run/best")
    assert resolve_ckpt("outputs/x/best") == "outputs/x/best"
