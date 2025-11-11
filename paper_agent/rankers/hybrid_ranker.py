from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, List

from ..models import ClassifiedPaper, RankedPaper


class HybridRanker:
    def __init__(self, recency_half_life_hours: float = 72.0) -> None:
        self.recency_half_life_hours = recency_half_life_hours

    def rank(self, classified_papers: Iterable[ClassifiedPaper]) -> List[RankedPaper]:
        now = datetime.now(timezone.utc)
        scored: List[tuple[ClassifiedPaper, float]] = []
        for paper in classified_papers:
            age = (now - paper.paper.published.astimezone(timezone.utc)).total_seconds() / 3600
            recency_weight = 0.5 ** (age / self.recency_half_life_hours) if age >= 0 else 1.0
            score = paper.relevance_score * 0.85 + recency_weight * 0.15
            scored.append((paper, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        ranked: List[RankedPaper] = []
        for idx, (paper, score) in enumerate(scored, start=1):
            ranked.append(
                RankedPaper(
                    paper=paper.paper,
                    score=score,
                    rank=idx,
                    is_relevant=paper.is_relevant,
                    relevance_score=paper.relevance_score,
                    main_topic=paper.main_topic,
                    reasoning=paper.reasoning,
                    summary=paper.summary,
                )
            )
        return ranked

