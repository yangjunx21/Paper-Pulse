from __future__ import annotations

import logging
import sys
import time
from datetime import date, datetime
from typing import List, Optional, Sequence

import requests
from dateutil import parser as date_parser
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is unavailable
    tqdm = None

from ..models import RawPaper
from .base import PaperFetcher

LOGGER = logging.getLogger(__name__)
DATA_URL = "https://neurips.cc/static/virtual/data/neurips-2025-orals-posters.json"
DEFAULT_HEADERS = {
    "User-Agent": "paper-agent/1.0 (+https://github.com/yangjunxiao)",
    "Accept": "application/json,text/plain,*/*",
}
REQUEST_TIMEOUT_SECONDS = 15
MAX_FETCH_ATTEMPTS = 5
FETCH_BACKOFF_SECONDS = 2.0


class NeuripsFetcher(PaperFetcher):
    source_name = "neurips_2025"

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        *,
        show_progress: bool = True,
    ) -> None:
        self.show_progress = show_progress
        self.session = session or self._build_session()
        for key, value in DEFAULT_HEADERS.items():
            self.session.headers.setdefault(key, value)
        self._cache: Optional[List[RawPaper]] = None

    def fetch(
        self,
        *,
        target_date: Optional[date],
        start_date: Optional[date],
        end_date: Optional[date],
        max_results: Optional[int] = None,
    ) -> List[RawPaper]:
        papers = list(self._load())
        if start_date or end_date or target_date:
            papers = [
                paper
                for paper in papers
                if _matches_date(paper.published.date(), target_date, start_date, end_date)
            ]
        if max_results is not None:
            return papers[:max_results]
        return papers

    def _load(self) -> List[RawPaper]:
        if self._cache is not None:
            return self._cache
        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_FETCH_ATTEMPTS + 1):
            try:
                response = self.session.get(DATA_URL, timeout=REQUEST_TIMEOUT_SECONDS)
                response.raise_for_status()
                payload = response.json()
                break
            except requests.RequestException as exc:
                last_error = exc
                LOGGER.warning(
                    "Attempt %s/%s failed to fetch NeurIPS 2025 proceedings data: %s",
                    attempt,
                    MAX_FETCH_ATTEMPTS,
                    exc,
                )
                if attempt == MAX_FETCH_ATTEMPTS:
                    self._cache = []
                    return self._cache
                time.sleep(FETCH_BACKOFF_SECONDS)
        else:
            LOGGER.warning("Failed to fetch NeurIPS 2025 proceedings data: %s", last_error)
            self._cache = []
            return self._cache

        items = self._extract_items(payload)
        papers: List[RawPaper] = []
        seen_ids: set[str] = set()
        progress_bar = self._create_progress_bar(items)
        for item in items:
            try:
                if not isinstance(item, dict):
                    LOGGER.debug(
                        "Skipping NeurIPS entry with unexpected type: %s",
                        type(item).__name__,
                    )
                    continue
                paper = self._convert_item(item)
                if paper.id in seen_ids:
                    LOGGER.debug("Skipping duplicate NeurIPS entry %s", paper.id)
                    continue
                seen_ids.add(paper.id)
                papers.append(paper)
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.debug("Skipping NeurIPS entry due to parse error: %s", exc, exc_info=exc)
                continue
            finally:
                if progress_bar is not None:
                    progress_bar.update()
        if progress_bar is not None:
            progress_bar.close()
        self._cache = papers
        LOGGER.info("Loaded %d NeurIPS 2025 papers from proceedings dataset.", len(papers))
        return papers

    def _convert_item(self, item: dict) -> RawPaper:
        identifier = item.get("uid") or item.get("id")
        if not identifier:
            raise ValueError("Missing identifier in NeurIPS dataset entry.")
        paper_id = f"{self.source_name}-{identifier}"
        title = str(item.get("name") or "").strip() or f"NeurIPS 2025 Paper {identifier}"
        summary = str(item.get("abstract") or "").strip()
        authors: List[str] = []
        affiliations: List[str] = []
        for author in item.get("authors", []):
            if not isinstance(author, dict):
                continue
            name = str(author.get("fullname") or "").strip()
            if name:
                authors.append(name)
            affiliation = str(
                author.get("institution") or author.get("affiliation") or ""
            ).strip()
            if affiliation and affiliation not in affiliations:
                affiliations.append(affiliation)
        link = _extract_link(item)
        published_str = item.get("starttime") or item.get("endtime")
        published = self._parse_datetime(published_str)
        categories: List[str] = []
        topic = item.get("topic")
        if isinstance(topic, str) and topic.strip():
            categories = [part.strip() for part in topic.split("->") if part.strip()]
        return RawPaper(
            id=paper_id,
            title=title,
            summary=summary,
            authors=authors,
            link=link,
            published=published,
            source=self.source_name,
            categories=categories,
            affiliations=affiliations,
        )

    @staticmethod
    def _parse_datetime(value: Optional[str]) -> datetime:
        if not value:
            return datetime(2025, 12, 1)
        try:
            return date_parser.parse(value)
        except (ValueError, TypeError):
            return datetime(2025, 12, 1)

    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=3,
            read=3,
            connect=3,
            backoff_factor=1.2,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _create_progress_bar(self, items: Sequence[dict]):
        if not items or not self.show_progress or tqdm is None:
            return None
        disable = False
        stderr = getattr(sys, "stderr", None)
        if stderr is not None:
            isatty = getattr(stderr, "isatty", None)
            if callable(isatty):
                disable = not isatty()
        return tqdm(
            total=len(items),
            desc="Processing NeurIPS 2025 papers",
            unit="paper",
            leave=False,
            disable=disable,
        )

    @staticmethod
    def _extract_items(payload: dict) -> List[dict]:
        if isinstance(payload, list):
            return payload
        if not isinstance(payload, dict):
            LOGGER.warning("Unexpected NeurIPS proceedings payload type: %s", type(payload).__name__)
            return []
        candidates = ["results", "papers", "items", "data"]
        for key in candidates:
            value = payload.get(key)
            if not value:
                continue
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                return list(value.values())
        LOGGER.warning("Unexpected NeurIPS proceedings payload structure: %s", type(payload).__name__)
        return []


def _extract_link(item: dict) -> str:
    link = item.get("paper_url")
    if isinstance(link, str) and link.startswith("http"):
        return link
    virtual_url = item.get("virtualsite_url")
    if isinstance(virtual_url, str) and virtual_url.startswith("/"):
        return f"https://neurips.cc{virtual_url}"
    return "https://neurips.cc"


def _matches_date(
    published_date: date,
    target_date: Optional[date],
    start_date: Optional[date],
    end_date: Optional[date],
) -> bool:
    if start_date and end_date:
        return start_date <= published_date <= end_date
    if target_date:
        return published_date == target_date
    return True

