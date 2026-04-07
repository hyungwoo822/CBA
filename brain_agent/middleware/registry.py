"""Middleware registry — resolves config names to middleware instances."""
from __future__ import annotations

import logging
from typing import Type

from brain_agent.middleware.base import Middleware, MiddlewareChain

logger = logging.getLogger(__name__)

# Global mapping: name -> middleware class
_MIDDLEWARE_CLASSES: dict[str, Type[Middleware]] = {}


def register_middleware(name: str, cls: Type[Middleware]) -> None:
    _MIDDLEWARE_CLASSES[name] = cls


def _auto_register() -> None:
    """Import built-in middlewares so they self-register.

    Anatomical mapping:
      meninges  — pipeline-level protective membranes
      myelin    — LLM-level signal insulation
      barrier   — tool-level selective permeability
    """
    # Meninges (pipeline-level)
    from brain_agent.middleware.meninges.dura_mater import DuraMater  # noqa: F401
    from brain_agent.middleware.meninges.arachnoid_tracer import ArachnoidTracer  # noqa: F401
    # Myelin (LLM-level)
    from brain_agent.middleware.myelin.sheath import MyelinSheath  # noqa: F401
    # Barrier (tool-level)
    from brain_agent.middleware.barrier.blood_brain_barrier import BloodBrainBarrier  # noqa: F401
    from brain_agent.middleware.barrier.synaptic_timeout import SynapticTimeout  # noqa: F401
    from brain_agent.middleware.barrier.microglial_defense import MicroglialDefense  # noqa: F401


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
