from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import ValidationError

from .models import ClassifiedPaper, PipelineSettings, RawPaper

CACHE_VERSION = 1
DEFAULT_CACHE_DIR = Path(os.getenv("PAPER_AGENT_CACHE_DIR", Path.home() / ".cache" / "paper_agent"))


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _normalize_topics(topics: Sequence[str] | None) -> List[str]:
    if not topics:
        return []
    normalized = []
    for topic in topics:
        text = str(topic).strip()
        if not text:
            continue
        normalized.append(text)
    return sorted(normalized)


def _normalize_keywords(keywords: Sequence[str] | None) -> List[str]:
    if not keywords:
        return []
    normalized: List[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        text = str(keyword).strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(text)
    return sorted(normalized, key=lambda item: item.lower())


def _hash_dict(payload: Dict[str, object]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _paper_digest(paper: RawPaper) -> str:
    payload = paper.model_dump(mode="json")
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _context_hash(
    *,
    focus: str | None,
    keywords: Sequence[str] | None,
    required_keywords: Sequence[str] | None,
    system_prompt: str | None,
) -> str:
    payload = {
        "focus": focus or "",
        "keywords": _normalize_keywords(keywords),
        "required": _normalize_keywords(required_keywords),
        "system_prompt": hashlib.sha256((system_prompt or "").encode("utf-8")).hexdigest(),
    }
    return _hash_dict(payload)


def _date_to_iso(value: date | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)


@dataclass(frozen=True)
class CachePaths:
    root: Path
    raw: Path
    filters: Path
    classification: Path
    summaries: Path

    @classmethod
    def resolve(cls, base_dir: Path | None = None) -> "CachePaths":
        base = (base_dir or DEFAULT_CACHE_DIR).expanduser().resolve()
        raw = base / "raw"
        filters = base / "filters"
        classification = base / "classification"
        summaries = base / "summaries"
        for path in (base, raw, filters, classification, summaries):
            path.mkdir(parents=True, exist_ok=True)
        return cls(root=base, raw=raw, filters=filters, classification=classification, summaries=summaries)


@dataclass
class Layer1FilterOutcomeSnapshot:
    retained_ids: List[str]
    removed_by_category: List[str]
    removed_by_keywords: List[str]
    removed_by_required: List[str]
    raw_key: str
    raw_fingerprint: str
    metadata: Dict[str, object]


class CacheManager:
    """
    Lightweight filesystem cache for paper runs.

    The cache is organised by namespaces:
    - raw/: deduplicated raw paper payloads for a given (source, date range)
    - filters/: retained and excluded paper identifiers for Layer 1 filtering
    - classification/: per-paper LLM classification outputs
    - summaries/: per-paper abstractive summaries
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self.paths = CachePaths.resolve(base_dir)

    # ------------------------------------------------------------------
    # Raw paper caching
    # ------------------------------------------------------------------
    def _raw_key(self, settings: PipelineSettings) -> str:
        payload: Dict[str, object] = {
            "target_date": _date_to_iso(settings.target_date),
            "start_date": _date_to_iso(settings.start_date),
            "end_date": _date_to_iso(settings.end_date),
            "sources": sorted(settings.sources or []),
        }
        return _hash_dict(payload)

    def _raw_path(self, key: str) -> Path:
        return self.paths.raw / f"{key}.json"

    def raw_key(self, settings: PipelineSettings) -> str:
        return self._raw_key(settings)

    def load_raw_papers(self, settings: PipelineSettings) -> Optional[List[RawPaper]]:
        key = self._raw_key(settings)
        return self.load_raw_papers_by_key(key)

    def load_raw_papers_by_key(self, key: str) -> Optional[List[RawPaper]]:
        path = self._raw_path(key)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if data.get("version") != CACHE_VERSION:
            return None
        papers_data = data.get("papers", [])
        papers: List[RawPaper] = []
        for item in papers_data:
            try:
                paper = RawPaper.model_validate(item)
            except ValidationError:
                continue
            papers.append(paper)
        return papers

    def store_raw_papers(self, settings: PipelineSettings, papers: Iterable[RawPaper]) -> str:
        paper_list = list(papers)
        key = self._raw_key(settings)
        path = self._raw_path(key)
        payload = {
            "version": CACHE_VERSION,
            "created_at": _now_iso(),
            "target_date": _date_to_iso(settings.target_date),
            "start_date": _date_to_iso(settings.start_date),
            "end_date": _date_to_iso(settings.end_date),
            "sources": sorted(settings.sources or []),
            "paper_count": len(paper_list),
            "papers": [paper.model_dump(mode="json") for paper in paper_list],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return key

    # ------------------------------------------------------------------
    # Layer 1 filtering cache
    # ------------------------------------------------------------------
    def _filter_key(
        self,
        *,
        raw_key: str,
        topics: Sequence[str] | None,
        keywords: Sequence[str] | None,
        required_keywords: Sequence[str] | None,
    ) -> str:
        payload = {
            "raw_key": raw_key,
            "topics": _normalize_topics(topics),
            "keywords": _normalize_keywords(keywords),
            "required": _normalize_keywords(required_keywords),
        }
        return _hash_dict(payload)

    def _filter_path(self, key: str) -> Path:
        return self.paths.filters / f"{key}.json"

    def load_layer1_candidates(
        self,
        settings: PipelineSettings,
        raw_papers: Sequence[RawPaper],
        *,
        keywords: Sequence[str] | None,
        required_keywords: Sequence[str] | None,
    ) -> Optional[List[RawPaper]]:
        raw_key = self._raw_key(settings)
        raw_map = {paper.id: paper for paper in raw_papers}
        key = self._filter_key(
            raw_key=raw_key,
            topics=settings.topics,
            keywords=keywords,
            required_keywords=required_keywords,
        )
        path = self._filter_path(key)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if data.get("version") != CACHE_VERSION:
            return None
        retained_ids = data.get("retained_ids", [])
        candidates: List[RawPaper] = []
        for paper_id in retained_ids:
            paper = raw_map.get(paper_id)
            if paper is None:
                return None
            candidates.append(paper)
        return candidates

    def store_layer1_result(
        self,
        settings: PipelineSettings,
        *,
        raw_key: str,
        raw_papers: Sequence[RawPaper],
        retained: Sequence[RawPaper],
        removed_by_category: Sequence[RawPaper],
        removed_by_keywords: Sequence[RawPaper],
        removed_by_required: Sequence[RawPaper],
        keywords: Sequence[str] | None,
        required_keywords: Sequence[str] | None,
    ) -> Layer1FilterOutcomeSnapshot:
        raw_fingerprint = self._fingerprint_papers(raw_papers)
        key = self._filter_key(
            raw_key=raw_key,
            topics=settings.topics,
            keywords=keywords,
            required_keywords=required_keywords,
        )
        path = self._filter_path(key)
        snapshot = Layer1FilterOutcomeSnapshot(
            retained_ids=[paper.id for paper in retained],
            removed_by_category=[paper.id for paper in removed_by_category],
            removed_by_keywords=[paper.id for paper in removed_by_keywords],
            removed_by_required=[paper.id for paper in removed_by_required],
            raw_key=raw_key,
            raw_fingerprint=raw_fingerprint,
            metadata={
                "version": CACHE_VERSION,
                "created_at": _now_iso(),
                "target_date": _date_to_iso(settings.target_date),
                "start_date": _date_to_iso(settings.start_date),
                "end_date": _date_to_iso(settings.end_date),
                "topics": _normalize_topics(settings.topics),
                "keywords": _normalize_keywords(keywords),
                "required_keywords": _normalize_keywords(required_keywords),
            },
        )
        payload = {
            **snapshot.metadata,
            "raw_key": raw_key,
            "raw_fingerprint": raw_fingerprint,
            "retained_ids": snapshot.retained_ids,
            "removed_by_category": snapshot.removed_by_category,
            "removed_by_keywords": snapshot.removed_by_keywords,
            "removed_by_required": snapshot.removed_by_required,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return snapshot

    def load_filter_records(
        self,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> List[Layer1FilterOutcomeSnapshot]:
        records: List[Layer1FilterOutcomeSnapshot] = []
        for path in self.paths.filters.glob("*.json"):
            try:
                data = json.loads(path.read_text("utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if data.get("version") != CACHE_VERSION:
                continue
            record_start = _parse_date(data.get("start_date") or data.get("target_date"))
            record_end = _parse_date(data.get("end_date") or data.get("target_date"))
            if start_date and record_end and record_end < start_date:
                continue
            if end_date and record_start and record_start > end_date:
                continue
            snapshot = Layer1FilterOutcomeSnapshot(
                retained_ids=list(map(str, data.get("retained_ids") or [])),
                removed_by_category=list(map(str, data.get("removed_by_category") or [])),
                removed_by_keywords=list(map(str, data.get("removed_by_keywords") or [])),
                removed_by_required=list(map(str, data.get("removed_by_required") or [])),
                raw_key=str(data.get("raw_key") or ""),
                raw_fingerprint=str(data.get("raw_fingerprint") or ""),
                metadata={
                    "version": data.get("version"),
                    "created_at": data.get("created_at"),
                    "target_date": data.get("target_date"),
                    "start_date": data.get("start_date"),
                    "end_date": data.get("end_date"),
                    "topics": data.get("topics") or [],
                    "keywords": data.get("keywords") or [],
                    "required_keywords": data.get("required_keywords") or [],
                },
            )
            records.append(snapshot)
        return records

    def get_keyword_filtered_papers(
        self,
        *,
        start_date: date,
        end_date: date,
    ) -> List[RawPaper]:
        records = self.load_filter_records(start_date=start_date, end_date=end_date)
        aggregated: Dict[str, RawPaper] = {}
        for record in records:
            if not record.removed_by_keywords:
                continue
            raw_papers = self.load_raw_papers_by_key(record.raw_key) or []
            raw_map = {paper.id: paper for paper in raw_papers}
            for paper_id in record.removed_by_keywords:
                paper = raw_map.get(paper_id)
                if paper:
                    aggregated[paper.id] = paper
        return list(aggregated.values())

    # ------------------------------------------------------------------
    # Classification cache
    # ------------------------------------------------------------------
    def _classification_path(self, paper_id: str) -> Path:
        digest = hashlib.sha256(paper_id.encode("utf-8")).hexdigest()
        folder = self.paths.classification / digest[:2]
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{digest}.json"

    def load_classification(
        self,
        paper: RawPaper,
        *,
        focus: str | None,
        keywords: Sequence[str] | None,
        required_keywords: Sequence[str] | None,
        system_prompt: str | None,
        allow_cross_context: bool = True,
    ) -> Tuple[Optional[ClassifiedPaper], bool]:
        path = self._classification_path(paper.id)
        if not path.is_file():
            return None, False
        try:
            payload = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return None, False
        if payload.get("version") != CACHE_VERSION:
            return None, False
        entries = payload.get("entries") or []
        if not isinstance(entries, list):
            return None, False
        requested_hash = _context_hash(
            focus=focus,
            keywords=keywords,
            required_keywords=required_keywords,
            system_prompt=system_prompt,
        )
        paper_digest = _paper_digest(paper)
        fallback: Optional[Tuple[ClassifiedPaper, bool]] = None
        for entry in entries:
            if entry.get("paper_digest") != paper_digest:
                continue
            classification_data = entry.get("classification")
            if not isinstance(classification_data, dict):
                continue
            cached_focus_hash = entry.get("context_hash")
            result = ClassifiedPaper(
                paper=paper,
                is_relevant=bool(classification_data.get("is_relevant")),
                relevance_score=float(classification_data.get("relevance_score", 0.0)),
                reasoning=classification_data.get("reasoning"),
                main_topic=classification_data.get("main_topic"),
            )
            if cached_focus_hash == requested_hash:
                return result, True
            if allow_cross_context and fallback is None:
                fallback = (result, False)
        if fallback:
            return fallback
        return None, False

    def store_classification(
        self,
        classified: ClassifiedPaper,
        *,
        focus: str | None,
        keywords: Sequence[str] | None,
        required_keywords: Sequence[str] | None,
        system_prompt: str | None,
    ) -> None:
        path = self._classification_path(classified.paper.id)
        try:
            payload = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if payload.get("version") != CACHE_VERSION:
            payload = {"version": CACHE_VERSION, "entries": []}
        entries: List[dict] = payload.setdefault("entries", [])
        context_hash = _context_hash(
            focus=focus,
            keywords=keywords,
            required_keywords=required_keywords,
            system_prompt=system_prompt,
        )
        paper_digest = _paper_digest(classified.paper)
        entry_payload = {
            "context_hash": context_hash,
            "paper_digest": paper_digest,
            "created_at": _now_iso(),
            "classification": {
                "is_relevant": classified.is_relevant,
                "relevance_score": classified.relevance_score,
                "reasoning": classified.reasoning,
                "main_topic": classified.main_topic,
            },
        }
        entries = [entry for entry in entries if entry.get("context_hash") != context_hash or entry.get("paper_digest") != paper_digest]
        entries.append(entry_payload)
        payload["entries"] = entries
        path.write_text(json.dumps(payload), encoding="utf-8")

    # ------------------------------------------------------------------
    # Summary cache
    # ------------------------------------------------------------------
    def _summary_path(self, paper_id: str) -> Path:
        digest = hashlib.sha256(paper_id.encode("utf-8")).hexdigest()
        folder = self.paths.summaries / digest[:2]
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{digest}.json"

    def load_summary(self, paper: RawPaper) -> Optional[str]:
        path = self._summary_path(paper.id)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if payload.get("version") != CACHE_VERSION:
            return None
        if payload.get("paper_digest") != _paper_digest(paper):
            return None
        summary = payload.get("summary")
        if isinstance(summary, str) and summary.strip():
            return summary
        return None

    def store_summary(self, paper: RawPaper, summary: str) -> None:
        if not summary.strip():
            return
        path = self._summary_path(paper.id)
        payload = {
            "version": CACHE_VERSION,
            "paper_digest": _paper_digest(paper),
            "summary": summary,
            "created_at": _now_iso(),
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _fingerprint_papers(self, papers: Sequence[RawPaper]) -> str:
        identifiers = sorted({paper.id for paper in papers})
        payload = {"ids": identifiers, "count": len(identifiers)}
        return _hash_dict(payload)

