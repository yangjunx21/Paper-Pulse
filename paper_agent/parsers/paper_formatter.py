from __future__ import annotations

from typing import Iterable, List

from ..models import RawPaper


def format_papers_for_llm(papers: Iterable[RawPaper]) -> str:
    """Return a concise string representation for LLM prompting."""
    formatted: List[str] = []
    for paper in papers:
        authors = ", ".join(paper.authors[:5])
        formatted.append(
            f"- id: {paper.id}\n"
            f"  title: {paper.title}\n"
            f"  authors: {authors}\n"
            f"  published: {paper.published.isoformat()}\n"
            f"  summary: {paper.summary.strip()[:1000]}\n"
            f"  link: {paper.link}"
        )
    return "\n\n".join(formatted)

