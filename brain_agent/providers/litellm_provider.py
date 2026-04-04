"""LiteLLM-based universal LLM provider. Supports OpenAI, Anthropic, local models, etc."""
from __future__ import annotations

import logging
import sys
import os
from typing import Any

from brain_agent.providers.base import LLMProvider, LLMResponse, ToolCallRequest

logger = logging.getLogger(__name__)

# Workaround: Windows Long Path issue prevents litellm from installing to
# the default site-packages. If litellm was installed to C:/pylibs, add it.
_LITELLM_ALT_PATH = "C:/pylibs"
if os.path.isdir(_LITELLM_ALT_PATH) and _LITELLM_ALT_PATH not in sys.path:
    sys.path.insert(0, _LITELLM_ALT_PATH)


class LiteLLMProvider(LLMProvider):
    def __init__(self, model: str = "openai/gpt-4o-mini", api_key: str | None = None):
        self._model = model
        self._api_key = api_key

    def get_default_model(self) -> str:
        return self._model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        import litellm

        kwargs: dict[str, Any] = {
            "model": model or self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if tools:
            kwargs["tools"] = tools

        try:
            response = await litellm.acompletion(**kwargs)
            choice = response.choices[0]
            content = choice.message.content
            tool_calls = []
            if choice.message.tool_calls:
                import json

                for tc in choice.message.tool_calls:
                    args = tc.function.arguments
                    if isinstance(args, str):
                        args = json.loads(args)
                    tool_calls.append(
                        ToolCallRequest(
                            id=tc.id, name=tc.function.name, arguments=args
                        )
                    )
            usage = {}
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=choice.finish_reason or "stop",
                usage=usage,
            )
        except Exception as e:
            logger.error(
                "LiteLLM call failed: %s (model=%s, msg_count=%d, max_tokens=%d)",
                e, kwargs.get("model", "?"), len(messages), max_tokens,
            )
            return LLMResponse(
                content=None,
                finish_reason="error",
                usage={"error": str(e)},
            )
