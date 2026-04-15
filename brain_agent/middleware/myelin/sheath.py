"""Myelin Sheath — insulates neural signal transmissions.

Myelin wraps axons to increase conduction velocity and prevent
signal degradation.  In CBA this middleware wraps every LLM call,
monitoring metabolic cost (token usage) and signal fidelity.

Reference: Nave & Werner 2014, "Myelination of the nervous system:
mechanisms and functions."
"""
from __future__ import annotations

import logging

from brain_agent.middleware.base import Middleware, MiddlewareContext
from brain_agent.middleware.registry import register_middleware

logger = logging.getLogger(__name__)


class MyelinSheath(Middleware):
    """Insulates LLM transmissions, tracking metabolic cost.

    Wraps every ``LLMProvider.chat()`` call and logs token
    consumption — the neural metabolic equivalent of glucose uptake
    during high-frequency axonal firing.

    When a ``trace_parent`` is present in the context, creates a
    LangSmith child run of type ``llm`` for cost tracking.
    """

    def __init__(self):
        self._total_prompt = 0
        self._total_completion = 0
        self._call_count = 0

    async def __call__(self, context, next_fn):
        # ── Trace: create LLM child run if tracing is active ──
        trace_parent = context.get("trace_parent")
        trace_region = context.get("trace_region", "unknown")
        llm_run = None
        model_name = context.get("model") or ""
        if trace_parent:
            try:
                llm_run = trace_parent.create_child(
                    name="llm.chat",
                    run_type="llm",
                    inputs={
                        "messages": context.get("messages", []),
                        "model": model_name,
                    },
                    extra={
                        "region": trace_region,
                        "metadata": {
                            "ls_model_name": model_name,
                            "ls_provider": model_name.split("/")[0] if "/" in model_name else "",
                        },
                    },
                )
            except Exception as e:
                logger.warning("Failed to create LLM trace run: %s", e)

        context = await next_fn(context)

        usage = context.get("usage", {})

        # ── Token accounting (existing logic) ──
        if usage and "error" not in usage:
            prompt = usage.get("prompt_tokens", 0)
            completion = usage.get("completion_tokens", 0)
            total = usage.get("total_tokens", 0)

            self._total_prompt += prompt
            self._total_completion += completion
            self._call_count += 1

            logger.info(
                "[MyelinSheath] Transmission #%d — prompt: %s, completion: %s, "
                "total: %s (cumulative: %s)",
                self._call_count, prompt, completion, total,
                self._total_prompt + self._total_completion,
            )

        # ── Trace: finalize LLM child run ──
        if llm_run:
            try:
                response = context.get("response")
                llm_run.end(outputs={
                    "content": response.content if response else None,
                    "usage_metadata": {
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    },
                })
                llm_run.post()
            except Exception as e:
                logger.warning("Failed to end LLM trace run: %s", e)

        return context


register_middleware("myelin_sheath", MyelinSheath)
