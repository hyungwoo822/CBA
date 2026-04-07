"""Arachnoid Tracer — cerebrospinal-fluid-like audit trail.

The arachnoid mater sits between dura and pia; cerebrospinal fluid
circulates in its subarachnoid space, carrying metabolic waste and
diagnostic markers.  This middleware emulates that role by recording
a structured audit trace of every pipeline cycle.

Reference: Sakka et al. 2011, "Anatomy and physiology of
cerebrospinal fluid."
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from brain_agent.middleware.base import Middleware, MiddlewareContext
from brain_agent.middleware.registry import register_middleware

logger = logging.getLogger(__name__)


class ArachnoidTracer(Middleware):
    """Cerebrospinal-fluid audit — circulates diagnostic data through
    the pipeline cycle.

    Attaches a unique ``trace_id`` and records ``started_at`` /
    ``finished_at`` timestamps plus a status flag.  Downstream
    components (e.g. the dashboard) can read ``context["audit"]``
    to render the full trace.
    """

    async def __call__(self, context, next_fn):
        trace_id = str(uuid.uuid4())[:8]
        audit: dict[str, Any] = {
            "trace_id": trace_id,
            "started_at": time.time(),
            "user_input": str(context.get("user_input", ""))[:200],
            "status": "in_progress",
        }
        context["audit"] = audit
        logger.debug("[ArachnoidTracer] trace=%s started", trace_id)

        try:
            context = await next_fn(context)
            audit["status"] = "completed"
        except Exception:
            audit["status"] = "error"
            raise
        finally:
            audit["finished_at"] = time.time()
            audit["elapsed_ms"] = round(
                (audit["finished_at"] - audit["started_at"]) * 1000, 1,
            )
            logger.info(
                "[ArachnoidTracer] trace=%s %s (%.1fms)",
                trace_id, audit["status"], audit["elapsed_ms"],
            )

        return context


register_middleware("arachnoid_tracer", ArachnoidTracer)
