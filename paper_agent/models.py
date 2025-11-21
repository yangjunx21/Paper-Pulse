from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl, model_validator


def _default_target_date() -> date:
    return datetime.now(timezone.utc).date()


AVAILABLE_SOURCES: tuple[str, ...] = ("arxiv", "huggingface_daily", "neurips_2025")
DEFAULT_SOURCES: tuple[str, ...] = ("arxiv",)


class TopicRequest(BaseModel):
    topic: str
    max_results: int = 10


class RawPaper(BaseModel):
    id: str
    title: str
    summary: str
    authors: List[str]
    link: HttpUrl
    pdf_url: Optional[HttpUrl] = None
    full_text: Optional[str] = None
    published: datetime
    source: str
    categories: List[str] = Field(default_factory=list)
    affiliations: List[str] = Field(default_factory=list)


class ClassifiedPaper(BaseModel):
    paper: RawPaper
    is_relevant: bool
    relevance_score: float
    main_topic: Optional[str] = None
    reasoning: Optional[str] = None
    summary: Optional[str] = None


class RankedPaper(BaseModel):
    paper: RawPaper
    score: float
    rank: int
    is_relevant: bool
    relevance_score: float
    main_topic: Optional[str] = None
    reasoning: Optional[str] = None
    summary: Optional[str] = None


class PipelineSettings(BaseModel):
    topics: List[str] = Field(default_factory=lambda: ["LLM Safety"])
    sources: List[str] = Field(default_factory=lambda: list(DEFAULT_SOURCES))
    keywords: Optional[List[str]] = None
    required_keywords: Optional[List[str]] = None
    keywords_file: Optional[str] = None
    target_date: Optional[date] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    max_results_per_topic: int = 80
    send_email: bool = False
    receiver_email: Optional[str] = None
    relevance_threshold: float = 0.8
    fallback_report_limit: int = 10
    llm_max_workers: int = 4
    summary_language: str = "English"
    enable_pdf_analysis: bool = False

    @model_validator(mode="after")
    def _validate_dates(self) -> "PipelineSettings":
        has_start = self.start_date is not None
        has_end = self.end_date is not None
        if has_start or has_end:
            if not has_start or not has_end:
                raise ValueError("Both start_date and end_date must be provided when specifying a date range.")
            if self.start_date > self.end_date:
                raise ValueError("start_date must be on or before end_date.")
            if self.target_date is not None:
                raise ValueError("target_date cannot be combined with start_date/end_date.")
        else:
            if self.target_date is None:
                self.target_date = _default_target_date()
        if not 0.0 <= self.relevance_threshold <= 1.0:
            raise ValueError("relevance_threshold must be between 0.0 and 1.0.")
        if self.fallback_report_limit <= 0:
            raise ValueError("fallback_report_limit must be a positive integer.")
        if self.llm_max_workers <= 0:
            raise ValueError("llm_max_workers must be a positive integer.")

        if self.keywords:
            normalized_keywords: List[str] = []
            seen_keywords: set[str] = set()
            for keyword in self.keywords:
                text = str(keyword).strip()
                if not text:
                    continue
                lowered = text.lower()
                if lowered in seen_keywords:
                    continue
                seen_keywords.add(lowered)
                normalized_keywords.append(text)
            self.keywords = normalized_keywords or None

        if self.required_keywords:
            normalized_required: List[str] = []
            seen_required: set[str] = set()
            for keyword in self.required_keywords:
                text = str(keyword).strip()
                if not text:
                    continue
                lowered = text.lower()
                if lowered in seen_required:
                    continue
                seen_required.add(lowered)
                normalized_required.append(text)
            self.required_keywords = normalized_required or None

        if self.keywords_file:
            file_path = str(self.keywords_file).strip()
            self.keywords_file = file_path or None

        summary_language = str(self.summary_language or "").strip()
        self.summary_language = summary_language or "English"

        normalized_sources: List[str] = []
        seen = set()
        for source in self.sources or []:
            normalized = str(source).strip().lower()
            if not normalized:
                continue
            if normalized not in AVAILABLE_SOURCES:
                raise ValueError(
                    f"Unsupported source '{source}'. Supported values: {', '.join(AVAILABLE_SOURCES)}."
                )
            if normalized in seen:
                continue
            seen.add(normalized)
            normalized_sources.append(normalized)
        if not normalized_sources:
            raise ValueError("At least one valid source must be specified.")
        self.sources = normalized_sources
        return self


class PipelineResult(BaseModel):
    settings: PipelineSettings
    papers: List[RankedPaper]
    email_subject: str
    email_body: str
    external_outputs: Dict[str, Any] = Field(default_factory=dict)

