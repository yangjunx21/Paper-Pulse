from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import List, Optional

from ..models import RawPaper


class PaperFetcher(ABC):
    """Abstract base class for all paper fetchers."""

    source_name: str

    @abstractmethod
    def fetch(
        self,
        *,
        target_date: Optional[date],
        start_date: Optional[date],
        end_date: Optional[date],
        max_results: Optional[int] = None,
    ) -> List[RawPaper]:
        """Fetch papers given date filters."""
        raise NotImplementedError

