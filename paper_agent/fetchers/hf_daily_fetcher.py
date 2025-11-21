from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta
from typing import Iterable, List, Optional

import requests
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

from ..models import RawPaper
from ..progress import iter_with_progress
from .base import PaperFetcher

LOGGER = logging.getLogger(__name__)
CANONICAL_BASE_URL = "https://huggingface.co"
DEFAULT_BASE_URLS: Iterable[str] = (
    "https://huggingface.co",
    "https://hf-mirror.com",
)
DEFAULT_HEADERS = {
    "User-Agent": "paper-agent/1.0 (+https://github.com/yangjunxiao)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


class HuggingFaceDailyFetcher(PaperFetcher):
    source_name = "huggingface_daily"

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        base_urls: Optional[Iterable[str]] = None,
    ) -> None:
        self.session = session or requests.Session()
        for key, value in DEFAULT_HEADERS.items():
            self.session.headers.setdefault(key, value)
        self.base_urls = list(base_urls or DEFAULT_BASE_URLS)

    def fetch(
        self,
        *,
        target_date: Optional[date],
        start_date: Optional[date],
        end_date: Optional[date],
        max_results: Optional[int] = None,
    ) -> List[RawPaper]:
        results: List[RawPaper] = []
        if start_date and end_date:
            day_count = (end_date - start_date).days + 1
            day_iterator = (
                start_date + timedelta(days=offset) for offset in range(day_count)
            )
            for current in iter_with_progress(
                day_iterator,
                description="Fetching Hugging Face range",
                total=day_count,
            ):
                results.extend(self._fetch_single_day(current, max_results))
                if max_results is not None and len(results) >= max_results:
                    return results[:max_results]
            return results
        if target_date:
            return self._fetch_single_day(target_date, max_results)
        raise ValueError("Either target_date or start_date/end_date must be provided for HuggingFace Daily.")

    def _fetch_single_day(self, target_date: date, max_results: Optional[int]) -> List[RawPaper]:
        errors: List[str] = []
        for base_url in self.base_urls:
            url = f"{base_url}/papers/date/{target_date.isoformat()}"
            response = self._get_with_retries(url, target_date)
            if response is None:
                errors.append(f"{base_url}: request failed")
                continue
            papers = self._parse_daily_page(response.text, target_date)
            if papers:
                if max_results is not None:
                    return papers[:max_results]
                return papers
            errors.append(f"{base_url}: no daily data found")
        LOGGER.warning(
            "Unable to retrieve Hugging Face daily papers for %s (attempts=%s).",
            target_date.isoformat(),
            "; ".join(errors) or "none",
        )
        return []

    def _parse_daily_page(self, html: str, target_date: date) -> List[RawPaper]:
        soup = BeautifulSoup(html, "html.parser")
        data_props_raw: Optional[str] = None
        for node in soup.select("[data-props]"):
            content = node.get("data-props")
            if content and '"dailyPapers"' in content:
                data_props_raw = content
                break
        if not data_props_raw:
            LOGGER.debug("No dailyPapers data found on Hugging Face page for %s", target_date.isoformat())
            return []

        try:
            payload = json.loads(data_props_raw)
        except json.JSONDecodeError as exc:
            LOGGER.warning("Failed to parse Hugging Face data for %s: %s", target_date.isoformat(), exc)
            return []

        entries = payload.get("dailyPapers", [])
        papers: List[RawPaper] = []
        for item in iter_with_progress(
            entries,
            description="Parsing Hugging Face entries",
            total=len(entries),
        ):
            paper_info = item.get("paper") or {}
            paper_id = str(paper_info.get("id") or "").strip()
            if not paper_id:
                continue
            title = (item.get("title") or paper_info.get("title") or "").strip()
            summary = (item.get("summary") or paper_info.get("summary") or "").strip()
            author_entries = paper_info.get("authors", [])
            authors = []
            affiliations: List[str] = []
            for author in author_entries:
                if not isinstance(author, dict):
                    continue
                name = str(author.get("name") or "").strip()
                if name:
                    authors.append(name)
                affiliation = str(
                    author.get("affiliation") or author.get("institution") or ""
                ).strip()
                if affiliation and affiliation not in affiliations:
                    affiliations.append(affiliation)
            published_str = paper_info.get("publishedAt") or item.get("publishedAt")
            published = self._parse_datetime(published_str)
            keywords = paper_info.get("ai_keywords") or []
            categories = [str(keyword).strip() for keyword in keywords if str(keyword).strip()]
            link = f"{CANONICAL_BASE_URL}/papers/{paper_id}"
            papers.append(
                RawPaper(
                    id=paper_id,
                    title=title or f"Hugging Face Daily Paper {paper_id}",
                    summary=summary,
                    authors=authors,
                    link=link,
                    published=published,
                    source=self.source_name,
                    categories=categories,
                    affiliations=affiliations,
                )
            )
        return papers

    def _get_with_retries(
        self,
        url: str,
        target_date: date,
        *,
        retries: int = 3,
        backoff: float = 1.5,
        timeout: int = 30,
    ) -> Optional[requests.Response]:
        for attempt in range(1, retries + 1):
            try:
                response = self.session.get(url, timeout=timeout)
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                if attempt >= retries:
                    LOGGER.warning(
                        "Failed to fetch Hugging Face daily papers for %s from %s: %s",
                        target_date.isoformat(),
                        url,
                        exc,
                    )
                    return None
                sleep_seconds = backoff ** (attempt - 1)
                LOGGER.debug(
                    "Retrying Hugging Face daily fetch for %s (attempt %d/%d) in %.1fs due to %s",
                    target_date.isoformat(),
                    attempt + 1,
                    retries,
                    sleep_seconds,
                    exc,
                )
                time.sleep(sleep_seconds)

    @staticmethod
    def _parse_datetime(value: Optional[str]) -> datetime:
        if not value:
            return datetime.utcnow()
        try:
            return date_parser.parse(value)
        except (ValueError, TypeError):
            return datetime.utcnow()

