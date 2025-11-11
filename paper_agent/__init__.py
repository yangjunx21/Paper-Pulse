"""
paper_agent: Minimal LLM-powered paper classification and recommendation pipeline.
"""

from importlib import import_module
from typing import Any

__all__ = ["generate_recommendations", "PipelineSettings"]


def __getattr__(name: str) -> Any:  # pragma: no cover - simple lazy loader
    if name in __all__:
        module = import_module(".pipeline", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
