from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

from ..models import RawPaper
from .base import PaperFetcher

LOGGER = logging.getLogger(__name__)
BASE_URL = "https://huggingface.co"
DAILY_URL_TEMPLATE = BASE_URL + "/papers/date/{iso_date}"

DEFAULT_HEADERS = {
    "User-Agent": "paper-agent/1.0 (+https://github.com/yangjunxiao)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


class HuggingFaceDailyFetcher(PaperFetcher):
    source_name = "huggingface_daily"

    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self.session = session or requests.Session()
        for key, value in DEFAULT_HEADERS.items():
            self.session.headers.setdefault(key, value)

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
            current = start_date
            while current <= end_date:
                results.extend(self._fetch_single_day(current, max_results))
                if max_results is not None and len(results) >= max_results:
                    return results[:max_results]
                current += timedelta(days=1)
            return results
        if target_date:
            return self._fetch_single_day(target_date, max_results)
        raise ValueError("Either target_date or start_date/end_date must be provided for HuggingFace Daily.")

    def _fetch_single_day(self, target_date: date, max_results: Optional[int]) -> List[RawPaper]:
        url = DAILY_URL_TEMPLATE.format(iso_date=target_date.isoformat())
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            LOGGER.warning("Failed to fetch Hugging Face daily papers for %s: %s", target_date.isoformat(), exc)
            return []

        soup = BeautifulSoup(response.text, "html.parser")
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
        for item in entries:
            paper_info = item.get("paper") or {}
            paper_id = str(paper_info.get("id") or "").strip()
            if not paper_id:
                continue
            title = (item.get("title") or paper_info.get("title") or "").strip()
            summary = (item.get("summary") or paper_info.get("summary") or "").strip()
            author_entries = paper_info.get("authors", [])
            authors = [
                str(author.get("name")).strip()
                for author in author_entries
                if isinstance(author, dict) and author.get("name")
            ]
            published_str = paper_info.get("publishedAt") or item.get("publishedAt")
            published = self._parse_datetime(published_str)
            keywords = paper_info.get("ai_keywords") or []
            categories = [str(keyword).strip() for keyword in keywords if str(keyword).strip()]
            link = f"{BASE_URL}/papers/{paper_id}"
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
                )
            )
        if max_results is not None:
            return papers[:max_results]
        return papers

    @staticmethod
    def _parse_datetime(value: Optional[str]) -> datetime:
        if not value:
            return datetime.utcnow()
        try:
            return date_parser.parse(value)
        except (ValueError, TypeError):
            return datetime.utcnow()

