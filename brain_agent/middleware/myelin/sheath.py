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
    """

    def __init__(self):
        self._total_prompt = 0
        self._total_completion = 0
        self._call_count = 0

    async def __call__(self, context, next_fn):
        context = await next_fn(context)

        usage = context.get("usage", {})
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
        return context


register_middleware("myelin_sheath", MyelinSheath)
