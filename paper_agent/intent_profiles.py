from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

LOGGER = logging.getLogger(__name__)

DEFAULT_PROFILE_NAME = "default"
_PROFILE_SUBDIR = "intent_profiles"


class IntentProfileError(Exception):
    """Base exception for intent profile failures."""


class IntentProfileNotFoundError(IntentProfileError):
    """Raised when a requested profile does not exist."""


def _default_store_root() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "config" / _PROFILE_SUBDIR


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(value: str) -> str:
    text = value.strip().lower() or DEFAULT_PROFILE_NAME
    return re.sub(r"[^a-z0-9._-]+", "-", text)


def _normalize_strings(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    normalized: List[str] = []
    for raw in items:
        text = str(raw).strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(text)
    return normalized


@dataclass
class IntentProfile:
    name: str
    description: str
    topics: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    required_keywords: List[str] = field(default_factory=list)
    notes: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    def normalize(self) -> "IntentProfile":
        self.topics = _normalize_strings(self.topics)
        self.keywords = _normalize_strings(self.keywords)
        self.required_keywords = _normalize_strings(self.required_keywords)
        if self.notes:
            self.notes = self.notes.strip()
        self.description = self.description.strip()
        return self

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "topics": self.topics,
            "keywords": self.keywords,
            "required_keywords": self.required_keywords,
            "notes": self.notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "IntentProfile":
        return cls(
            name=str(payload.get("name") or DEFAULT_PROFILE_NAME),
            description=str(payload.get("description") or "").strip(),
            topics=_normalize_strings(payload.get("topics") or []),
            keywords=_normalize_strings(payload.get("keywords") or []),
            required_keywords=_normalize_strings(payload.get("required_keywords") or []),
            notes=(payload.get("notes") or None),
            created_at=payload.get("created_at"),
            updated_at=payload.get("updated_at"),
        )


class IntentProfileStore:
    def __init__(self, root_dir: str | Path | None = None) -> None:
        if root_dir:
            self._root = Path(root_dir).expanduser().resolve()
        else:
            self._root = _default_store_root()

    @property
    def root(self) -> Path:
        self._root.mkdir(parents=True, exist_ok=True)
        return self._root

    def list_profiles(self) -> List[str]:
        if not self.root.exists():
            return []
        return sorted(path.stem for path in self.root.glob("*.json"))

    def exists(self, name: str) -> bool:
        return self.path_for(name).is_file()

    def path_for(self, name: str) -> Path:
        slug = _slugify(name)
        return self.root / f"{slug}.json"

    def save(self, profile: IntentProfile) -> Path:
        normalized = profile.normalize()
        now = _now_iso()
        if not normalized.created_at:
            normalized.created_at = now
        normalized.updated_at = now
        payload = normalized.to_dict()
        path = self.path_for(profile.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("Saved intent profile '%s' to %s", profile.name, path)
        return path

    def load(self, name: str) -> IntentProfile:
        path = self.path_for(name)
        if not path.is_file():
            raise IntentProfileNotFoundError(f"Intent profile '{name}' not found at {path}")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        profile = IntentProfile.from_dict(payload)
        profile.name = name
        return profile.normalize()


def apply_profile_defaults(settings_kwargs: dict, profile: IntentProfile) -> dict:
    """Inject profile defaults into pipeline settings kwargs."""
    if "topics" not in settings_kwargs and profile.topics:
        settings_kwargs["topics"] = list(profile.topics)
    if "keywords" not in settings_kwargs and profile.keywords:
        settings_kwargs["keywords"] = list(profile.keywords)
    if "required_keywords" not in settings_kwargs and profile.required_keywords:
        settings_kwargs["required_keywords"] = list(profile.required_keywords)
    return settings_kwargs


