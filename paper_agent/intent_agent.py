from __future__ import annotations

import json
import logging
from typing import Optional

from .intent_profiles import IntentProfile
from .llm import LLMClient, LLMMessage
from .llm.prompts import INTENT_SYSTEM_PROMPT, INTENT_USER_PROMPT_TEMPLATE
from .llm.utils import prepare_llm_json_content

LOGGER = logging.getLogger(__name__)


class IntentAgentError(Exception):
    """Raised when the intent agent cannot interpret LLM output."""


class IntentAgent:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()

    def draft_profile(
        self,
        *,
        profile_name: str,
        description: str,
        seed_profile: IntentProfile | None = None,
    ) -> IntentProfile:
        """Generate an initial profile from a free-form description."""
        return self._run_llm(
            profile_name=profile_name,
            description=description,
            existing=seed_profile,
            feedback=None,
        )

    def refine_profile(self, profile: IntentProfile, feedback: str) -> IntentProfile:
        """Apply user feedback on top of an existing profile."""
        return self._run_llm(
            profile_name=profile.name,
            description=profile.description,
            existing=profile,
            feedback=feedback,
        )

    def _run_llm(
        self,
        *,
        profile_name: str,
        description: str,
        existing: IntentProfile | None,
        feedback: Optional[str],
    ) -> IntentProfile:
        payload = INTENT_USER_PROMPT_TEMPLATE.format(
            description=description.strip(),
            existing_profile=self._format_existing(existing),
            feedback=(feedback or "").strip() or "None",
        )
        messages = [
            LLMMessage(role="system", content=INTENT_SYSTEM_PROMPT),
            LLMMessage(role="user", content=payload),
        ]
        LOGGER.debug("Submitting intent interpretation request (profile=%s).", profile_name)
        response = self._llm.chat_completion(messages, temperature=0.0)
        raw_content = prepare_llm_json_content(response.content or "", context="intent interpretation")
        try:
            data = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise IntentAgentError(f"Failed to parse intent profile JSON: {raw_content}") from exc
        profile = self._build_profile_from_payload(
            profile_name=profile_name,
            fallback_description=description,
            payload=data,
            existing=existing,
        )
        LOGGER.info(
            "Intent profile '%s' parsed successfully (topics=%d, keywords=%d, required=%d).",
            profile.name,
            len(profile.topics),
            len(profile.keywords),
            len(profile.required_keywords),
        )
        return profile

    @staticmethod
    def _format_existing(existing: IntentProfile | None) -> str:
        if not existing:
            return "None"
        return json.dumps(existing.to_dict(), ensure_ascii=False, indent=2)

    @staticmethod
    def _build_profile_from_payload(
        *,
        profile_name: str,
        fallback_description: str,
        payload: dict,
        existing: IntentProfile | None,
    ) -> IntentProfile:
        description = str(payload.get("description") or fallback_description).strip()
        topics = _coerce_string_list(payload.get("topics"))
        if not topics and existing:
            topics = list(existing.topics)
        keywords = _coerce_string_list(payload.get("keywords"))
        if not keywords and existing:
            keywords = list(existing.keywords)
        required = _coerce_string_list(payload.get("required_keywords"))
        if not required and existing:
            required = list(existing.required_keywords)
        notes = payload.get("notes") or (existing.notes if existing else None)
        profile = IntentProfile(
            name=profile_name,
            description=description,
            topics=list(topics),
            keywords=list(keywords),
            required_keywords=list(required),
            notes=str(notes).strip() if isinstance(notes, str) else notes,
            created_at=existing.created_at if existing else None,
            updated_at=existing.updated_at if existing else None,
        )
        return profile.normalize()


def _coerce_string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    return []


