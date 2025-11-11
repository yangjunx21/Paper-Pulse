from __future__ import annotations

import logging
import urllib.parse
from datetime import date, datetime, timedelta
from typing import List, Optional, Set

import feedparser
from dateutil import parser as date_parser

from ..models import RawPaper
from .base import PaperFetcher

LOGGER = logging.getLogger(__name__)
ARXIV_RSS_API = "http://export.arxiv.org/api/query"


class ArxivFetcher(PaperFetcher):
    source_name = "arxiv"

    def __init__(self, page_size: int = 200) -> None:
        self.page_size = page_size

    def fetch(
        self,
        *,
        target_date: Optional[date],
        start_date: Optional[date],
        end_date: Optional[date],
        max_results: Optional[int] = None,
    ) -> List[RawPaper]:
        if start_date and end_date:
            return self.fetch_by_date_range(start_date, end_date, max_results)
        if target_date:
            return self.fetch_by_date(target_date, max_results)
        raise ValueError("Either target_date or start_date/end_date must be provided for arXiv fetches.")

    def fetch_by_date(self, target_date: date, max_results: Optional[int] = None) -> List[RawPaper]:
        start = target_date.strftime("%Y%m%d0000")
        end = target_date.strftime("%Y%m%d2359")
        query = f"submittedDate:[{start} TO {end}]"
        LOGGER.info("Fetching arXiv papers for date %s using query '%s'", target_date.isoformat(), query)
        papers = self._fetch_with_query(search_query=query, max_results=max_results)
        LOGGER.info("Fetched %d arXiv papers for date %s", len(papers), target_date.isoformat())
        return papers

    def fetch_by_date_range(
        self, start_date: date, end_date: date, max_results: Optional[int] = None
    ) -> List[RawPaper]:
        if end_date < start_date:
            raise ValueError("end_date must be on or after start_date.")
        LOGGER.info(
            "Fetching arXiv papers for date range %s - %s",
            start_date.isoformat(),
            end_date.isoformat(),
        )
        papers: List[RawPaper] = []
        for offset, current in enumerate(_daterange(start_date, end_date), start=1):
            LOGGER.debug("Fetching arXiv papers chunk %d for %s", offset, current.isoformat())
            chunk = self.fetch_by_date(current, None)
            papers.extend(chunk)
            if max_results is not None and len(papers) >= max_results:
                LOGGER.info(
                    "Reached max_results=%d while fetching arXiv range %s-%s.",
                    max_results,
                    start_date.isoformat(),
                    end_date.isoformat(),
                )
                return papers[:max_results]
        LOGGER.info(
            "Fetched %d arXiv papers for range %s - %s",
            len(papers),
            start_date.isoformat(),
            end_date.isoformat(),
        )
        return papers

    def _fetch_with_query(self, search_query: str, max_results: Optional[int]) -> List[RawPaper]:
        papers: List[RawPaper] = []
        seen_ids: Set[str] = set()
        start_index = 0
        while True:
            remaining = None if max_results is None else max_results - len(papers)
            if remaining is not None and remaining <= 0:
                break
            batch_size = self.page_size if remaining is None else min(self.page_size, remaining)
            LOGGER.debug(
                "Requesting arXiv batch: query='%s', start=%d, max_results=%s",
                search_query,
                start_index,
                batch_size,
            )
            params = {
                "search_query": search_query,
                "start": start_index,
                "max_results": batch_size,
                "sortBy": "submittedDate",
                "sortOrder": "ascending",
            }
            url = f"{ARXIV_RSS_API}?{urllib.parse.urlencode(params)}"
            feed = feedparser.parse(url)
            entries = getattr(feed, "entries", [])
            if not entries:
                LOGGER.debug("No arXiv entries returned for query '%s' (start=%d).", search_query, start_index)
                break
            for entry in entries:
                try:
                    parsed = self._parse_entry(entry)
                except Exception as exc:  # pylint: disable=broad-except
                    LOGGER.warning("Failed to parse arXiv entry for query %s: %s", search_query, exc)
                    continue
                if parsed.id in seen_ids:
                    LOGGER.debug("Skipping duplicate arXiv paper id=%s", parsed.id)
                    continue
                seen_ids.add(parsed.id)
                papers.append(parsed)
            start_index += len(entries)
            if len(entries) < batch_size:
                LOGGER.debug(
                    "Received final batch for query '%s': %d entries (requested %d).",
                    search_query,
                    len(entries),
                    batch_size,
                )
                break
        return papers

    @staticmethod
    def _parse_entry(entry: feedparser.FeedParserDict) -> RawPaper:
        authors = [author.name for author in entry.get("authors", [])]
        categories = []
        for tag in entry.get("tags", []):
            term = getattr(tag, "term", None)
            if not term and isinstance(tag, dict):
                term = tag.get("term")
            if term:
                categories.append(str(term))
        published = ArxivFetcher._parse_datetime(entry.get("published", entry.get("updated")))
        return RawPaper(
            id=entry["id"],
            title=entry["title"],
            summary=entry.get("summary", ""),
            authors=authors,
            link=entry["link"],
            published=published,
            source=ArxivFetcher.source_name,
            categories=categories,
        )

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime:
        if not value:
            return datetime.utcnow()
        return date_parser.parse(value)


def _daterange(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)
