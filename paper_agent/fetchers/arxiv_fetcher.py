from __future__ import annotations

import logging
import time
import urllib.parse
from datetime import date, datetime, timedelta
from typing import List, Optional, Sequence, Set

import feedparser
import requests
from dateutil import parser as date_parser
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..models import RawPaper
from ..progress import iter_with_progress
from .base import PaperFetcher

LOGGER = logging.getLogger(__name__)
ARXIV_RSS_API = "http://export.arxiv.org/api/query"
REQUEST_TIMEOUT_SECONDS = 20
REQUEST_MIN_INTERVAL_SECONDS = 10.0
DEFAULT_MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 1.5
RETRY_STATUS_FORCELIST = (429, 500, 502, 503, 504)
DEFAULT_HEADERS = {
    "User-Agent": "PaperPulseBot/1.1 (+https://github.com/yangjunxiao/paper-pulse; mailto:paper-pulse@yangjunxiao.dev)",
    "Accept": "application/atom+xml; charset=utf-8",
    "Accept-Encoding": "gzip, deflate",
}

class ArxivFetcher(PaperFetcher):
    source_name = "arxiv"
    DEFAULT_CATEGORIES: tuple[str, ...] = ("cs.CL", "cs.LG", "cs.AI")

    def __init__(
        self,
        page_size: int = 200,
        session: Optional[requests.Session] = None,
        categories: Optional[Sequence[str]] = None,
    ) -> None:
        self.page_size = page_size
        self.categories: tuple[str, ...] = self._normalize_categories(categories)
        self._category_clause: Optional[str] = self._build_category_clause(self.categories)
        self.session = session or self._build_session()
        self._last_request_ts: float = 0.0

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
        date_clause = f"submittedDate:[{start} TO {end}]"
        if self._category_clause:
            query = f"({date_clause}) AND ({self._category_clause})"
        else:
            query = date_clause
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
        total_days = (end_date - start_date).days + 1
        day_iterator = iter_with_progress(
            _daterange(start_date, end_date),
            description="Fetching arXiv range",
            total=total_days,
        )
        for offset, current in enumerate(day_iterator, start=1):
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
            feed = self._request_feed(params)
            if feed is None:
                LOGGER.warning(
                    "Stopping arXiv fetch due to repeated request failures. query='%s', start=%d",
                    search_query,
                    start_index,
                )
                break
            entries = getattr(feed, "entries", [])
            if not entries:
                LOGGER.debug("No arXiv entries returned for query '%s' (start=%d).", search_query, start_index)
                break
            for entry in iter_with_progress(
                entries,
                description="Parsing arXiv batch",
                total=len(entries),
            ):
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
        affiliations = []
        for author in entry.get("authors", []):
            affiliation = None
            if isinstance(author, dict):
                affiliation = author.get("affiliation") or author.get("arxiv_affiliation")
            else:
                affiliation = getattr(author, "affiliation", None) or getattr(author, "arxiv_affiliation", None)
            if affiliation:
                text = str(affiliation).strip()
                if text and text not in affiliations:
                    affiliations.append(text)
        entry_affiliation = entry.get("arxiv_affiliation")
        if entry_affiliation:
            text = str(entry_affiliation).strip()
            if text and text not in affiliations:
                affiliations.append(text)
        categories = []
        for tag in entry.get("tags", []):
            term = getattr(tag, "term", None)
            if not term and isinstance(tag, dict):
                term = tag.get("term")
            if term:
                categories.append(str(term))
        published = ArxivFetcher._parse_datetime(entry.get("published", entry.get("updated")))
        pdf_url = ArxivFetcher._extract_pdf_url(entry)
        return RawPaper(
            id=entry["id"],
            title=entry["title"],
            summary=entry.get("summary", ""),
            authors=authors,
            link=entry["link"],
            pdf_url=pdf_url,
            published=published,
            source=ArxivFetcher.source_name,
            categories=categories,
            affiliations=affiliations,
        )

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime:
        if not value:
            return datetime.utcnow()
        return date_parser.parse(value)

    def _request_feed(self, params: dict) -> Optional[feedparser.FeedParserDict]:
        self._respect_rate_limit()
        try:
            response = self.session.get(
                ARXIV_RSS_API,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            self._last_request_ts = time.monotonic()
            response.raise_for_status()
        except requests.RequestException as exc:
            LOGGER.warning("Failed to fetch arXiv feed: %s", exc)
            self._last_request_ts = time.monotonic()
            return None

        feed = feedparser.parse(response.text)

        status = getattr(feed, "status", response.status_code)
        if status != 200:
            LOGGER.warning("Unexpected arXiv response status: %s", status)
        if getattr(feed, "bozo", False):
            bozo_exception = getattr(feed, "bozo_exception", None)
            LOGGER.debug("arXiv feed parsing issue: %s", bozo_exception)
        return feed

    def _respect_rate_limit(self) -> None:
        if self._last_request_ts <= 0:
            return
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < REQUEST_MIN_INTERVAL_SECONDS:
            sleep_for = REQUEST_MIN_INTERVAL_SECONDS - elapsed
            time.sleep(max(0.0, sleep_for))

    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=DEFAULT_MAX_RETRIES,
            read=DEFAULT_MAX_RETRIES,
            connect=DEFAULT_MAX_RETRIES,
            backoff_factor=RETRY_BACKOFF_FACTOR,
            status_forcelist=RETRY_STATUS_FORCELIST,
            allowed_methods=frozenset({"GET"}),
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        for key, value in DEFAULT_HEADERS.items():
            session.headers.setdefault(key, value)
        return session

    @staticmethod
    def _normalize_categories(categories: Optional[Sequence[str]]) -> tuple[str, ...]:
        if not categories:
            return ArxivFetcher.DEFAULT_CATEGORIES
        normalized = [str(category).strip() for category in categories if str(category).strip()]
        if not normalized:
            return ArxivFetcher.DEFAULT_CATEGORIES
        unique = tuple(dict.fromkeys(normalized))
        return unique

    @staticmethod
    def _build_category_clause(categories: Optional[Sequence[str]]) -> Optional[str]:
        if not categories:
            return None
        clauses = [f"cat:{category}" for category in categories if category]
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return " OR ".join(clauses)

    @staticmethod
    def _extract_pdf_url(entry: feedparser.FeedParserDict) -> Optional[str]:
        links = entry.get("links", []) or []
        for link_info in links:
            href = getattr(link_info, "href", None)
            link_type = getattr(link_info, "type", None)
            title = getattr(link_info, "title", None)
            if isinstance(link_info, dict):
                href = href or link_info.get("href")
                link_type = link_type or link_info.get("type")
                title = title or link_info.get("title")
            if not href:
                continue
            if isinstance(link_type, str) and "pdf" in link_type.lower():
                return href
            if isinstance(title, str) and "pdf" in title.lower():
                return href
        fallback = entry.get("link") or entry.get("id")
        if isinstance(fallback, str) and "/abs/" in fallback:
            pdf_url = fallback.replace("/abs/", "/pdf/")
            if not pdf_url.endswith(".pdf"):
                pdf_url = f"{pdf_url}.pdf"
            return pdf_url
        return None


def _daterange(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)
