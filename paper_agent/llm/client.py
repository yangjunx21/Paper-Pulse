from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence

from openai import OpenAI

from ..config import LLMConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class LLMMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMOutput:
    content: str
    usage_prompt_tokens: Optional[int] = None
    usage_completion_tokens: Optional[int] = None
    usage_total_tokens: Optional[int] = None


@dataclass
class LLMChatRequest:
    messages: Sequence[LLMMessage]
    temperature: float = 0.0
    metadata: Optional[dict[str, Any]] = None


@dataclass
class LLMChatResult:
    request: LLMChatRequest
    output: Optional[LLMOutput] = None
    error: Optional[Exception] = None


class LLMClient:
    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self._config = config or LLMConfig()
        self._client = OpenAI(
            api_key=self._config.api_key,
            base_url=self._config.api_base,
            timeout=self._config.timeout,
        )

    def chat_completion(
        self,
        messages: Iterable[LLMMessage],
        temperature: float = 0.2,
        max_attempts: int = 5,
    ) -> LLMOutput:
        last_error: Optional[Exception] = None
        chat_messages: List[dict[str, str]] = [message.to_dict() for message in messages]

        for attempt in range(1, max_attempts + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._config.model,
                    messages=chat_messages,
                    temperature=temperature,
                )
                choice = response.choices[0]
                usage = response.usage
                return LLMOutput(
                    content=choice.message.content or "",
                    usage_prompt_tokens=getattr(usage, "prompt_tokens", None),
                    usage_completion_tokens=getattr(usage, "completion_tokens", None),
                    usage_total_tokens=getattr(usage, "total_tokens", None),
                )
            except Exception as exc:  # pylint: disable=broad-except
                last_error = exc
                # LOGGER.warning(
                #     "LLM chat completion attempt %s/%s failed: %s",
                #     attempt,
                #     max_attempts,
                #     exc,
                #     exc_info=True,
                # )
                if attempt == max_attempts:
                    raise
                continue

        # Should never reach here because we either return or raise above
        raise RuntimeError("LLM chat completion failed") from last_error

    def chat_completion_batch(
        self,
        requests: Sequence[LLMChatRequest],
        *,
        max_workers: int = 4,
        allow_errors: bool = True,
    ) -> List[LLMChatResult]:
        if not requests:
            return []
        worker_count = max(1, int(max_workers or 1))
        results: List[Optional[LLMChatResult]] = [None] * len(requests)

        def _submit(request_index: int, request: LLMChatRequest) -> LLMChatResult:
            output = self.chat_completion(request.messages, temperature=request.temperature)
            return LLMChatResult(request=request, output=output)

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_index = {
                executor.submit(_submit, idx, request): idx for idx, request in enumerate(requests)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                request = requests[idx]
                try:
                    result = future.result()
                except Exception as exc:  # pylint: disable=broad-except
                    if not allow_errors:
                        raise
                    # LOGGER.warning(
                    #     "LLM chat request failed (metadata=%s): %s",
                    #     request.metadata,
                    #     exc,
                    #     exc_info=True,
                    # )
                    results[idx] = LLMChatResult(request=request, error=exc)
                else:
                    results[idx] = result

        # replace any missing (should not happen) with fallback error entries
        final_results: List[LLMChatResult] = []
        for idx, result in enumerate(results):
            if result is None:
                request = requests[idx]
                error = RuntimeError("LLM request did not produce a result.")
                final_results.append(LLMChatResult(request=request, error=error))
            else:
                final_results.append(result)
        return final_results

