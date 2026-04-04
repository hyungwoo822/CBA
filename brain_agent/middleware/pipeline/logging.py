"""Pipeline-level logging middleware — logs input/output of process_request."""
from __future__ import annotations

import logging
import time

from brain_agent.middleware.base import Middleware, MiddlewareContext
from brain_agent.middleware.registry import register_middleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(Middleware):
    """Logs pipeline input and output with elapsed time."""

    async def __call__(self, context, next_fn):
        user_input = context.get("user_input", "")
        logger.info("[Pipeline] Input: %.120s", user_input)
        t0 = time.perf_counter()

        context = await next_fn(context)

        elapsed = time.perf_counter() - t0
        logger.info("[Pipeline] Output: %.120s (%.2fs)", context.get("result", ""), elapsed)
        return context


register_middleware("logging", LoggingMiddleware)
