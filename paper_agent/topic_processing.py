from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

from .llm import LLMClient, LLMMessage
from .llm.prompts import TOPIC_EXPANSION_SYSTEM_PROMPT, TOPIC_EXPANSION_USER_PROMPT_TEMPLATE
from .llm.utils import prepare_llm_json_content

LOGGER = logging.getLogger(__name__)


@dataclass
class TopicDefinition:
    basic_topic: str
    subtopics: List[str] = field(default_factory=list)

    def to_prompt_dict(self) -> dict[str, object]:
        return {"basic_topic": self.basic_topic, "subtopics": self.subtopics}


@dataclass
class SubtopicSynonyms:
    name: str
    synonyms: List[str] = field(default_factory=list)


@dataclass
class TopicSynonyms:
    basic_topic: str
    base_synonyms: List[str] = field(default_factory=list)
    subtopic_synonyms: List[SubtopicSynonyms] = field(default_factory=list)

    def all_base_synonyms(self) -> List[str]:
        return list(dict.fromkeys([*self.base_synonyms, self.basic_topic]))

    def all_subtopic_synonyms(self) -> List[str]:
        collected: List[str] = []
        for group in self.subtopic_synonyms:
            collected.extend(group.synonyms)
            collected.append(group.name)
        return list(dict.fromkeys(collected))


class TopicHierarchy:
    def __init__(self, topics: Sequence[TopicDefinition]) -> None:
        normalized = {topic.basic_topic.lower(): topic for topic in topics}
        self._topics = normalized

    @classmethod
    def from_file(cls, path: str | Path) -> "TopicHierarchy":
        real_path = Path(path)
        if not real_path.exists():
            raise FileNotFoundError(f"Topic hierarchy file not found: {real_path}")
        with real_path.open("r", encoding="utf-8") as fh:
            try:
                raw = json.load(fh)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Topic hierarchy file is not valid JSON: {real_path}") from exc
        topics: List[TopicDefinition] = []
        if isinstance(raw, dict):
            for basic_topic, subtopics in raw.items():
                topics.append(
                    TopicDefinition(
                        basic_topic=str(basic_topic).strip(),
                        subtopics=_coerce_subtopics(subtopics),
                    )
                )
        elif isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict) or "basic" not in item:
                    raise ValueError("List-based topic hierarchy entries must include a 'basic' key.")
                basic_topic = str(item["basic"]).strip()
                subtopics = _coerce_subtopics(item.get("subtopics", []))
                topics.append(TopicDefinition(basic_topic=basic_topic, subtopics=subtopics))
        else:
            raise ValueError("Topic hierarchy must be a JSON object or list.")
        return cls(topics)

    def select_topics(self, requested_topics: Iterable[str]) -> List[TopicDefinition]:
        selections: List[TopicDefinition] = []
        for name in requested_topics:
            key = name.strip().lower()
            if not key:
                continue
            topic = self._topics.get(key)
            if topic:
                selections.append(topic)
            else:
                LOGGER.warning("Requested topic '%s' not found in hierarchy; using without subtopics.", name)
                selections.append(TopicDefinition(basic_topic=name.strip(), subtopics=[]))
        return selections


def expand_topic_synonyms(llm: LLMClient, topics: Iterable[TopicDefinition]) -> List[TopicSynonyms]:
    topic_list = list(topics)
    if not topic_list:
        return []
    payload_topics = [topic.to_prompt_dict() for topic in topic_list]
    payload = TOPIC_EXPANSION_USER_PROMPT_TEMPLATE.format(
        topics=json.dumps(payload_topics, ensure_ascii=False, indent=2)
    )
    messages = [
        LLMMessage(role="system", content=TOPIC_EXPANSION_SYSTEM_PROMPT),
        LLMMessage(role="user", content=payload),
    ]
    response = llm.chat_completion(messages)
    parsed_content = prepare_llm_json_content(response.content, context="topic expansion")
    try:
        data = json.loads(parsed_content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse LLM topic expansion response: {response.content}") from exc
    results: List[TopicSynonyms] = []
    for item in data.get("topics", []):
        basic_topic = str(item.get("basic_topic") or item.get("name") or "").strip()
        if not basic_topic:
            LOGGER.warning("Skipping topic in LLM response without 'basic_topic'.")
            continue
        base_synonyms = _coerce_strings(item.get("synonyms", []))
        subtopic_items = item.get("subtopics", [])
        subtopic_synonyms: List[SubtopicSynonyms] = []
        if isinstance(subtopic_items, list):
            for subtopic in subtopic_items:
                if not isinstance(subtopic, dict):
                    continue
                sub_name = str(subtopic.get("name") or subtopic.get("subtopic") or "").strip()
                if not sub_name:
                    continue
                sub_synonyms = _coerce_strings(subtopic.get("synonyms", []))
                subtopic_synonyms.append(SubtopicSynonyms(name=sub_name, synonyms=sub_synonyms))
        results.append(
            TopicSynonyms(
                basic_topic=basic_topic,
                base_synonyms=_ensure_keyword_in_list(basic_topic, base_synonyms),
                subtopic_synonyms=subtopic_synonyms,
            )
        )
    # Preserve order matching the requested topics
    lookup = {item.basic_topic.lower(): item for item in results}
    ordered: List[TopicSynonyms] = []
    for topic in topic_list:
        result = lookup.get(topic.basic_topic.lower())
        if result:
            ordered.append(result)
        else:
            LOGGER.warning(
                "LLM topic expansion did not return synonyms for '%s'; falling back to literal keyword.",
                topic.basic_topic,
            )
            ordered.append(
                TopicSynonyms(
                    basic_topic=topic.basic_topic,
                    base_synonyms=[topic.basic_topic],
                    subtopic_synonyms=[
                        SubtopicSynonyms(name=name, synonyms=[name]) for name in topic.subtopics
                    ],
                )
            )
    return ordered


def _coerce_subtopics(value: object) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return []


def _coerce_strings(value: object) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        # support comma separated synonyms
        return [part.strip() for part in text.split(",") if part.strip()]
    return []


def _ensure_keyword_in_list(keyword: str, values: List[str]) -> List[str]:
    normalized = [value.strip() for value in values if value.strip()]
    if keyword not in normalized:
        normalized.insert(0, keyword)
    seen = set()
    deduped: List[str] = []
    for value in normalized:
        lower = value.lower()
        if lower in seen:
            continue
        seen.add(lower)
        deduped.append(value)
    return deduped

