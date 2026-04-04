"""Tool-level permission checking middleware."""
from __future__ import annotations

import logging

from brain_agent.middleware.base import Middleware, MiddlewareContext
from brain_agent.middleware.registry import register_middleware

logger = logging.getLogger(__name__)

# Tools that are always blocked
_BLOCKED_TOOLS: set[str] = set()


class PermissionMiddleware(Middleware):
    """Blocks execution of disallowed tools."""

    async def __call__(self, context, next_fn):
        tool_name = context.get("tool_name", "")
        if tool_name in _BLOCKED_TOOLS:
            logger.warning("[Tool] Blocked execution of: %s", tool_name)
            context["result"] = f"Error: tool '{tool_name}' is not permitted"
            return context

        logger.debug("[Tool] Permitted: %s", tool_name)
        return await next_fn(context)


register_middleware("permission", PermissionMiddleware)
