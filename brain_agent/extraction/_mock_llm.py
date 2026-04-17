"""Test-only LLMProvider fake that records calls and returns canned responses.

DO NOT use this class in production code.
"""
from __future__ import annotations

from collections import deque
from typing import Any

from brain_agent.providers.base import LLMProvider, LLMResponse


class RecordingLLMProvider(LLMProvider):
    """FIFO fake LLM provider for extraction tests."""

    def __init__(
        self,
        default_model: str = "mock-default",
        responses: list[LLMResponse] | list[str] | None = None,
    ):
        self._default_model = default_model
        self._responses: deque[LLMResponse] = deque()
        self.calls: list[dict[str, Any]] = []
        if responses:
            for response in responses:
                if isinstance(response, str):
                    self.enqueue_content(response)
                else:
                    self._responses.append(response)

    def enqueue_content(self, content: str) -> None:
        self._responses.append(LLMResponse(content=content))

    def enqueue_response(self, response: LLMResponse) -> None:
        self._responses.append(response)

    def get_default_model(self) -> str:
        return self._default_model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        assert self._responses, "RecordingLLMProvider response queue exhausted"
        return self._responses.popleft()
