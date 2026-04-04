"""Middleware ABC and chain executor (onion model)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable


@dataclass
class MiddlewareContext:
    """Shared state that flows through a middleware chain."""
    data: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


class Middleware(ABC):
    """Base middleware. Follows the onion (chain-of-responsibility) pattern.

    Each middleware receives a context and a ``next_fn`` coroutine that
    invokes the next layer (or the core function at the centre).
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    async def __call__(
        self,
        context: MiddlewareContext,
        next_fn: Callable[[MiddlewareContext], Awaitable[MiddlewareContext]],
    ) -> MiddlewareContext:
        ...


class MiddlewareChain:
    """Composes a list of middlewares into a single callable chain."""

    def __init__(self, middlewares: list[Middleware] | None = None):
        self._middlewares: list[Middleware] = middlewares or []

    def add(self, middleware: Middleware) -> None:
        self._middlewares.append(middleware)

    async def execute(
        self,
        context: MiddlewareContext,
        core_fn: Callable[[MiddlewareContext], Awaitable[MiddlewareContext]],
    ) -> MiddlewareContext:
        """Run the chain: mw[0] → mw[1] → … → core_fn → … → mw[1] → mw[0]."""

        async def _build_next(
            index: int,
        ) -> Callable[[MiddlewareContext], Awaitable[MiddlewareContext]]:
            if index >= len(self._middlewares):
                return core_fn

            mw = self._middlewares[index]

            async def _next(ctx: MiddlewareContext) -> MiddlewareContext:
                inner = await _build_next(index + 1)
                return await mw(ctx, inner)

            return _next

        runner = await _build_next(0)
        return await runner(context)
