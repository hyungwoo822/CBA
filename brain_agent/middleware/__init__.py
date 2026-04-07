"""Neural sheath system — 3-layer protective architecture.

Anatomical layers:
  meninges  — pipeline-level (DuraMater, ArachnoidTracer)
  myelin    — LLM-level (MyelinSheath)
  barrier   — tool-level (BloodBrainBarrier, SynapticTimeout, MicroglialDefense)
"""
from brain_agent.middleware.base import (
    Middleware,
    MiddlewareContext,
    MiddlewareChain,
)
from brain_agent.middleware.registry import MiddlewareRegistry

__all__ = ["Middleware", "MiddlewareContext", "MiddlewareChain", "MiddlewareRegistry"]
