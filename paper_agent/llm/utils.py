from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)


def prepare_llm_json_content(raw_content: str, *, context: str) -> str:
    """Strip markdown fences and surrounding whitespace from an LLM JSON response."""
    content = raw_content.strip()
    if not content:
        raise ValueError(f"LLM {context} response was empty.")
    if content.startswith("```"):
        LOGGER.debug("Detected fenced code block in LLM %s response; stripping fences.", context)
        lines = content.splitlines()
        lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines.pop()
        content = "\n".join(lines).strip()
    if not content:
        raise ValueError(f"LLM {context} response only contained a code fence.")
    return content

