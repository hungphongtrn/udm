"""Hugging Face / remote parquet access helpers (download with retry, remote range-read file)."""
from __future__ import annotations

import io
import os
import time
from typing import Optional

import pyarrow.parquet as pq


def resolve_url(repo_id: str, revision: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{path}"


def retry(fn, tries: int = 8, base: float = 5.0, what: str = ""):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # network / 429 / transient
            last = e
            msg = str(e)
            if "404" in msg and "429" not in msg:
                raise
            time.sleep(min(base * (2 ** i), 120))
    raise RuntimeError(f"giving up after {tries} tries ({what}): {last}")


def download(repo_id: str, revision: str, path: str, cache_dir: Optional[str]) -> str:
    """Download ONE file with huggingface_hub (hf_xet; HF_XET_HIGH_PERFORMANCE) and return its
    local path. Resumable and lock-protected, so concurrent workers can request the same file."""
    from huggingface_hub import hf_hub_download

    return retry(lambda: hf_hub_download(repo_id=repo_id, repo_type="dataset", revision=revision, filename=path,
                                         cache_dir=cache_dir), what=f"download {path}")


def local_if_cached(repo_id: str, revision: str, path: str, cache_dir: Optional[str]) -> Optional[str]:
    from huggingface_hub import try_to_load_from_cache

    p = try_to_load_from_cache(repo_id=repo_id, filename=path, cache_dir=cache_dir, revision=revision, repo_type="dataset")
    return p if isinstance(p, str) else None


class HttpRangeFile(io.RawIOBase):
    """Read-only seekable file over HTTPS range requests (used for streaming remote parquet footers /
    row groups without downloading whole files). The CDN redirect target is cached and re-resolved on error."""

    def __init__(self, url: str, block: int = 8 << 20):
        import httpx

        self._httpx = httpx
        self.url = url
        self._client = httpx.Client(follow_redirects=True, timeout=120.0)
        self._pos = 0
        self._block = block
        self._buf = (0, b"")
        r = retry(lambda: self._get(0, 0), what="size probe")
        cr = r.headers.get("content-range", "")
        self.size = int(cr.split("/")[-1]) if "/" in cr else int(r.headers.get("content-length", 0))

    def _get(self, start: int, end: int):
        r = self._client.get(self.url, headers={"Range": f"bytes={start}-{end}"})
        if r.status_code not in (200, 206):
            raise IOError(f"HTTP {r.status_code} for {self.url}")
        return r

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, off, whence=0):
        self._pos = {0: off, 1: self._pos + off, 2: self.size + off}[whence]
        return self._pos

    def readinto(self, b):
        n = len(b)
        if n == 0 or self._pos >= self.size:
            return 0
        start = self._pos
        end = min(self.size, start + n) - 1
        bs, bd = self._buf
        if not (bs <= start and end < bs + len(bd)):
            fetch_end = min(self.size - 1, max(end, start + self._block - 1))
            r = retry(lambda: self._get(start, fetch_end), what=f"range {start}-{fetch_end}")
            self._buf = (start, r.content)
            bs, bd = self._buf
        chunk = bd[start - bs: end - bs + 1]
        b[: len(chunk)] = chunk
        self._pos += len(chunk)
        return len(chunk)

    def close(self):
        try:
            self._client.close()
        finally:
            super().close()


def open_parquet(repo_id: str, revision: str, path: str, cache_dir: Optional[str], stream: bool = False,
                 local_root: Optional[str] = None) -> pq.ParquetFile:
    if local_root:
        return pq.ParquetFile(os.path.join(local_root, path))
    if stream:
        loc = local_if_cached(repo_id, revision, path, cache_dir)
        if loc:
            return pq.ParquetFile(loc)
        f = io.BufferedReader(HttpRangeFile(resolve_url(repo_id, revision, path)), buffer_size=1 << 20)
        return pq.ParquetFile(f)
    return pq.ParquetFile(download(repo_id, revision, path, cache_dir))
