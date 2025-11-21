from __future__ import annotations

from typing import Iterable, Iterator, Optional, TypeVar

try:  # pragma: no cover - optional dependency behaviour
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is unavailable
    _tqdm = None

T = TypeVar("T")


def _infer_total(iterable: Iterable[object]) -> Optional[int]:
    try:
        return len(iterable)  # type: ignore[arg-type]
    except TypeError:
        return None


def iter_with_progress(
    iterable: Iterable[T],
    *,
    description: str = "",
    total: Optional[int] = None,
    min_display: int = 1,
    leave: bool = False,
) -> Iterator[T]:
    """
    Wrap an iterable with a tqdm progress iterator when tqdm is available.

    Args:
        iterable: Items to iterate over.
        description: Optional label for the progress bar.
        total: Explicit total length. If omitted we attempt to infer via len().
        min_display: Minimum length required before showing a progress bar.
        leave: Whether tqdm should leave the progress bar on completion.

    Returns:
        Iterator over the original items, optionally showing progress.
    """

    if _tqdm is None:
        for item in iterable:
            yield item
        return

    computed_total = total if total is not None else _infer_total(iterable)
    disable = computed_total is not None and computed_total < min_display

    progress_iter = _tqdm(
        iterable,
        total=computed_total,
        desc=description,
        leave=leave,
        disable=disable,
    )
    try:
        for item in progress_iter:
            yield item
    finally:
        progress_iter.close()

