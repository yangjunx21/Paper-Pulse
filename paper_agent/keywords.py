from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Sequence

import yaml

LOGGER = logging.getLogger(__name__)

ENV_KEYWORDS_FILE = "PAPER_AGENT_KEYWORDS_FILE"

@dataclass(frozen=True)
class KeywordConfig:
    keywords: List[str]
    required_keywords: List[str]

DEFAULT_LAYER1_KEYWORDS: tuple[str, ...] = (
    "safety",
    "safe",
    "security",
    "robustness",
    "trust",
    "trustworthy",
    "alignment",
    "value alignment",
    "ethic",
    "ethical",
    "bias",
    "fairness",
    "debias",
    "moral",
    "jailbreak",
    "red team",
    "red teaming",
    "adversarial attack",
    "adversarial defense",
    "defense",
    "poisoning",
    "privacy",
    "confidentiality",
    "hallucination",
    "factuality",
    "veracity",
    "interpretability",
    "transparency",
    "mechanistic interpretability",
    "watermark",
    "detection",
    "provenance",
)

DEFAULT_REQUIRED_KEYWORDS: tuple[str, ...] = ()

def resolve_keywords(
    explicit_keywords: Sequence[str] | None = None,
    explicit_required_keywords: Sequence[str] | None = None,
    *,
    keyword_file: str | os.PathLike[str] | None = None,
) -> KeywordConfig:
    """
    Resolve the keyword list used for Layer 1 filtering and LLM prompts.

    Precedence:
    1. Explicit keywords supplied via PipelineSettings (highest priority).
    2. Keywords loaded from a YAML file (settings or env variable).
    3. Built-in defaults.
    """
    keywords: List[str] | None = None
    required_keywords: List[str] | None = None

    if explicit_keywords:
        keywords = _normalize_keywords(explicit_keywords)
        if keywords:
            LOGGER.debug("Using %d explicitly provided keywords.", len(keywords))
        else:
            LOGGER.warning("Explicit keywords were provided but empty after normalisation; falling back.")

    if explicit_required_keywords:
        required_keywords = _normalize_keywords(explicit_required_keywords)
        if required_keywords:
            LOGGER.debug("Using %d explicitly provided required keywords.", len(required_keywords))
        else:
            LOGGER.warning(
                "Explicit required keywords were provided but empty after normalisation; falling back."
            )

    path = keyword_file or os.getenv(ENV_KEYWORDS_FILE)
    if path:
        resolved_path = _normalize_path(path)
        file_config = _load_keywords_from_file(resolved_path)
        LOGGER.info(
            "Loaded %d keywords and %d required keywords from configuration file: %s",
            len(file_config.keywords),
            len(file_config.required_keywords),
            resolved_path,
        )
        if keywords is None:
            keywords = file_config.keywords
        if required_keywords is None:
            required_keywords = file_config.required_keywords

    if keywords is None:
        keywords = list(DEFAULT_LAYER1_KEYWORDS)
        LOGGER.debug("Falling back to %d built-in default keywords.", len(keywords))
    if required_keywords is None:
        required_keywords = list(DEFAULT_REQUIRED_KEYWORDS)
        if required_keywords:
            LOGGER.debug("Using %d built-in default required keywords.", len(required_keywords))
    return KeywordConfig(keywords=keywords, required_keywords=required_keywords)


@lru_cache(maxsize=8)
def _load_keywords_from_file(path: Path) -> KeywordConfig:
    if not path.is_file():
        raise FileNotFoundError(f"Keyword configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        try:
            data = yaml.safe_load(fh)  # type: ignore[no-untyped-call]
        except yaml.YAMLError as exc:  # type: ignore[attr-defined]
            raise ValueError(f"Failed to parse keyword configuration: {path}") from exc
    keywords, required_keywords = _extract_keywords_config(data)
    if not keywords:
        raise ValueError(f"Keyword configuration {path} did not contain any keywords.")
    return KeywordConfig(keywords=keywords, required_keywords=required_keywords)


def _extract_keywords_config(data: object) -> tuple[List[str], List[str]]:
    if isinstance(data, dict):
        keywords = _extract_keywords_section(
            data,
            direct_keys=("keywords",),
            group_keys=("keyword_groups", "groups", "categories"),
        )
        required_keywords = _extract_keywords_section(
            data,
            direct_keys=("required_keywords", "required", "must_include", "must_have"),
            group_keys=("required_keyword_groups", "required_groups", "must_include_groups"),
        )
        return keywords, required_keywords
    keywords = _extract_keywords_section(data, direct_keys=(), group_keys=())
    return keywords, []


def _extract_keywords_section(
    data: object,
    *,
    direct_keys: Sequence[str],
    group_keys: Sequence[str],
) -> List[str]:
    if data is None:
        return []
    if isinstance(data, list):
        return _normalize_keywords(_flatten_iterable(data))
    if isinstance(data, dict):
        collected: List[str] = []
        for key in direct_keys:
            if key in data and data[key] is not None:
                collected.extend(_flatten_iterable(data[key]))
        for key in group_keys:
            groups = data.get(key)
            if not groups:
                continue
            for group in _ensure_iterable(groups):
                if isinstance(group, dict):
                    group_items = (
                        group.get("keywords")
                        or group.get("items")
                        or group.get("values")
                        or group.get("terms")
                    )
                    if group_items:
                        collected.extend(_flatten_iterable(group_items))
                        continue
                collected.extend(_flatten_iterable(group))
        return _normalize_keywords(collected)
    if isinstance(data, (str, int, float)):
        return _normalize_keywords([str(data)])
    raise ValueError("Unsupported keyword configuration structure. Expected list or dict.")


def _normalize_keywords(items: Iterable[str]) -> List[str]:
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


def _flatten_iterable(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (str, int, float)):
        return [str(value)]
    if isinstance(value, dict):
        # When encountering dicts directly, gather their values recursively
        collected: List[str] = []
        for item in value.values():
            collected.extend(_flatten_iterable(item))
        return collected
    try:
        iterator = iter(value)  # type: ignore[arg-type]
    except TypeError:
        return [str(value)]
    collected: List[str] = []
    for item in iterator:
        collected.extend(_flatten_iterable(item))
    return collected


def _ensure_iterable(value: object) -> List[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalize_path(path: str | os.PathLike[str]) -> Path:
    return Path(path).expanduser().resolve()


