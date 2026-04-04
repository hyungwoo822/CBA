"""LLM-level token counting middleware."""
from __future__ import annotations

import logging

from brain_agent.middleware.base import Middleware, MiddlewareContext
from brain_agent.middleware.registry import register_middleware

logger = logging.getLogger(__name__)


class TokenCounterMiddleware(Middleware):
    """Logs token usage from every LLM call."""

    async def __call__(self, context, next_fn):
        context = await next_fn(context)

        usage = context.get("usage", {})
        if usage:
            logger.info(
                "[LLM] Tokens — prompt: %s, completion: %s, total: %s",
                usage.get("prompt_tokens", "?"),
                usage.get("completion_tokens", "?"),
                usage.get("total_tokens", "?"),
            )
        return context


register_middleware("token_counter", TokenCounterMiddleware)
