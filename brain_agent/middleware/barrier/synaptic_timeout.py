"""Synaptic Timeout — temporal gating at the neuromuscular junction.

Real synapses have a refractory period and will fail to transmit if
the presynaptic signal takes too long.  This middleware enforces a
maximum execution duration for tool calls, preventing runaway
processes from starving the rest of the neural pipeline.

Reference: Katz 1966, "Nerve, Muscle, and Synapse."
"""
from __future__ import annotations

import asyncio
import logging

from brain_agent.middleware.base import Middleware, MiddlewareContext
from brain_agent.middleware.registry import register_middleware

logger = logging.getLogger(__name__)

# Default synaptic window (seconds) — overridable per-context
DEFAULT_TIMEOUT_SEC = 60


class SynapticTimeout(Middleware):
    """Enforces temporal limits on tool execution.

    If the downstream chain does not complete within the synaptic
    window, the call is cancelled and an error result is returned —
    analogous to synaptic transmission failure due to prolonged
    refractory period.
    """

    def __init__(self, timeout_sec: float = DEFAULT_TIMEOUT_SEC):
        self._timeout = timeout_sec

    async def __call__(self, context, next_fn):
        tool_name = context.get("tool_name", "unknown")
        timeout = context.get("timeout_override", self._timeout)

        try:
            context = await asyncio.wait_for(
                next_fn(context), timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[SynapticTimeout] Transmission failure: %s exceeded %.1fs window",
                tool_name, timeout,
            )
            context["result"] = (
                f"Error: tool '{tool_name}' exceeded synaptic window "
                f"({timeout}s) — transmission terminated"
            )

        return context


register_middleware("synaptic_timeout", SynapticTimeout)
