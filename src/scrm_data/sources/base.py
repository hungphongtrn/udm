"""Shared pieces for the source converters."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple, Optional

from ..tiers import DropRow


class ConvResult(NamedTuple):
    split: Optional[str]      # output split or None when dropped
    row: Optional[dict]       # 26-column dict or None when dropped
    key: int                  # content key (uint64) or 0
    reason: Optional[str]     # drop reason when dropped


def dropped(reason: str) -> ConvResult:
    return ConvResult(None, None, 0, reason)


def safe(fn: Callable[..., ConvResult]) -> Callable[..., ConvResult]:
    """Wrap a converter: DropRow and encoding problems become counted drops."""

    def wrapper(*a, **kw) -> ConvResult:
        try:
            return fn(*a, **kw)
        except DropRow as e:
            return dropped(e.reason)
        except UnicodeError:
            return dropped("invalid_unicode")
        except ValueError as e:  # NaN/Infinity in canonical JSON
            if "Out of range float" in str(e):
                return dropped("non_finite_number")
            raise

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


@dataclass
class Unit:
    """One parallel work item: a source file (or row-group range of a file)."""
    source: str                    # source slug
    unit_id: str                   # unique, filesystem-safe
    repo_id: str
    revision: str
    path: str                      # path inside the repo
    split: Optional[str]           # raw/output split for the file when fixed
    config: str = "default"
    row_groups: Optional[tuple] = None   # (start, end) row-group range, or None for the whole file
    extra: dict = field(default_factory=dict)
    stage: str = "train"           # "eval" (validation/test) is processed before "train"
