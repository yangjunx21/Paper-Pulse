from typing import Dict, Type

from .arxiv_fetcher import ArxivFetcher
from .base import PaperFetcher
from .hf_daily_fetcher import HuggingFaceDailyFetcher
from .neurips_fetcher import NeuripsFetcher

FETCHER_REGISTRY: Dict[str, Type[PaperFetcher]] = {
    ArxivFetcher.source_name: ArxivFetcher,
    HuggingFaceDailyFetcher.source_name: HuggingFaceDailyFetcher,
    NeuripsFetcher.source_name: NeuripsFetcher,
}

__all__ = [
    "PaperFetcher",
    "ArxivFetcher",
    "HuggingFaceDailyFetcher",
    "NeuripsFetcher",
    "FETCHER_REGISTRY",
    "get_fetcher",
]


def get_fetcher(source: str) -> PaperFetcher:
    try:
        fetcher_cls = FETCHER_REGISTRY[source]
    except KeyError as exc:
        raise ValueError(f"Unsupported source '{source}'. Available: {', '.join(FETCHER_REGISTRY)}") from exc
    return fetcher_cls()

