"""Blood-Brain Barrier (BBB) — selective permeability for tool access.

The BBB permits essential nutrients while blocking pathogens and
neurotoxins.  This middleware filters tool invocations, permitting
safe tools and blocking those on the exclusion list.

Reference: Abbott et al. 2010, "Structure and function of the
blood-brain barrier."
"""
from __future__ import annotations

import logging

from brain_agent.middleware.base import Middleware, MiddlewareContext
from brain_agent.middleware.registry import register_middleware

logger = logging.getLogger(__name__)

# Tools that must never cross the barrier
_NEUROTOXINS: set[str] = set()


def block_tool(name: str) -> None:
    """Add a tool to the neurotoxin list (permanently blocked)."""
    _NEUROTOXINS.add(name)


def unblock_tool(name: str) -> None:
    """Remove a tool from the neurotoxin list."""
    _NEUROTOXINS.discard(name)


class BloodBrainBarrier(Middleware):
    """Selective permeability gate for tool execution.

    Checks the tool name against ``_NEUROTOXINS``.  If blocked, the
    barrier rejects the call and returns an error result without
    invoking the downstream chain.
    """

    async def __call__(self, context, next_fn):
        tool_name = context.get("tool_name", "")

        if tool_name in _NEUROTOXINS:
            logger.warning(
                "[BBB] Neurotoxin blocked: %s — tool is on exclusion list",
                tool_name,
            )
            context["result"] = f"Error: tool '{tool_name}' blocked by Blood-Brain Barrier"
            return context

        logger.debug("[BBB] Permitted passage: %s", tool_name)
        return await next_fn(context)


register_middleware("blood_brain_barrier", BloodBrainBarrier)
