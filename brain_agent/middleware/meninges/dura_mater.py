"""Dura Mater — outermost meningeal layer protecting the brain.

Monitors all pipeline activity like the dura mater envelops and
shields the central nervous system.  Logs input/output timing so
downstream diagnostics can detect pathological latency.

Reference: Weller 2005, "Microscopic morphology and histology of
the human meninges."
"""
from __future__ import annotations

import logging
import time

from brain_agent.middleware.base import Middleware, MiddlewareContext
from brain_agent.middleware.registry import register_middleware

logger = logging.getLogger(__name__)


class DuraMater(Middleware):
    """Protective monitoring of the entire pipeline cycle.

    Wraps ``process_request`` at the outermost layer, recording
    wall-clock latency and truncated input/output for observability.
    """

    async def __call__(self, context, next_fn):
        user_input = context.get("user_input", "")
        logger.info("[DuraMater] Incoming signal: %.120s", user_input)
        t0 = time.perf_counter()

        context = await next_fn(context)

        elapsed = time.perf_counter() - t0
        logger.info(
            "[DuraMater] Cycle complete: %.120s (%.2fs)",
            context.get("result", ""),
            elapsed,
        )
        return context


register_middleware("dura_mater", DuraMater)
