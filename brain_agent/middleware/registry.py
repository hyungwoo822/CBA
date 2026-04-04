"""Middleware registry — resolves config names to middleware instances."""
from __future__ import annotations

import logging
from typing import Type

from brain_agent.middleware.base import Middleware, MiddlewareChain

logger = logging.getLogger(__name__)

# Global mapping: name → middleware class
_MIDDLEWARE_CLASSES: dict[str, Type[Middleware]] = {}


def register_middleware(name: str, cls: Type[Middleware]) -> None:
    _MIDDLEWARE_CLASSES[name] = cls


def _auto_register() -> None:
    """Import built-in middlewares so they self-register."""
    from brain_agent.middleware.pipeline.logging import LoggingMiddleware  # noqa: F401
    from brain_agent.middleware.llm.token_counter import TokenCounterMiddleware  # noqa: F401
    from brain_agent.middleware.tool.permission import PermissionMiddleware  # noqa: F401


class MiddlewareRegistry:
    """Builds MiddlewareChain instances from config name lists."""

    def __init__(self) -> None:
        _auto_register()

    def build_chain(self, names: list[str]) -> MiddlewareChain:
        middlewares: list[Middleware] = []
        for name in names:
            cls = _MIDDLEWARE_CLASSES.get(name)
            if cls is None:
                logger.warning("Unknown middleware: %s — skipped", name)
                continue
            middlewares.append(cls())
        return MiddlewareChain(middlewares)
