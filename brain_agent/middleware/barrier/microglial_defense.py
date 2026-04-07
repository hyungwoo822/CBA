"""Microglial Defense — the brain's resident immune system.

Microglia are the primary immune cells of the central nervous system.
They survey the parenchyma for pathogens, damaged cells, and foreign
material, mounting an inflammatory response when threats are detected.

This middleware inspects tool inputs for potentially dangerous patterns
(command injection, path traversal, etc.) and blocks them before they
reach the effector system.

Reference: Nimmerjahn et al. 2005, "Resting microglial cells are
highly dynamic surveillants of brain parenchyma in vivo."
"""
from __future__ import annotations

import logging
import re

from brain_agent.middleware.base import Middleware, MiddlewareContext
from brain_agent.middleware.registry import register_middleware

logger = logging.getLogger(__name__)

# Pathogen signatures — regex patterns that indicate hostile input
_PATHOGEN_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("command_injection", re.compile(r"[;&|`$]\s*(?:rm|del|format|mkfs|dd)\b", re.IGNORECASE)),
    ("path_traversal", re.compile(r"\.\.[/\\]")),
    ("env_exfiltration", re.compile(r"\$\{?\w*(?:KEY|SECRET|TOKEN|PASS)\w*\}?", re.IGNORECASE)),
]


class MicroglialDefense(Middleware):
    """Surveys tool inputs for pathogenic patterns.

    Scans all string values in the tool parameters for known attack
    signatures.  If a pathogen is detected, the inflammatory response
    blocks execution and logs the threat class.
    """

    async def __call__(self, context, next_fn):
        tool_name = context.get("tool_name", "")
        params = context.get("params", {})

        # Survey all string parameter values
        threat = self._survey(params)
        if threat:
            logger.warning(
                "[MicroglialDefense] Inflammatory response — "
                "pathogen '%s' detected in tool '%s' params",
                threat, tool_name,
            )
            context["result"] = (
                f"Error: Microglial defense blocked '{tool_name}' — "
                f"pathogenic pattern detected: {threat}"
            )
            return context

        return await next_fn(context)

    def _survey(self, params: dict) -> str | None:
        """Scan parameter values for pathogen signatures."""
        for value in self._extract_strings(params):
            for name, pattern in _PATHOGEN_PATTERNS:
                if pattern.search(value):
                    return name
        return None

    def _extract_strings(self, obj, depth: int = 0) -> list[str]:
        """Recursively extract string values from nested structures."""
        if depth > 5:
            return []
        strings: list[str] = []
        if isinstance(obj, str):
            strings.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                strings.extend(self._extract_strings(v, depth + 1))
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                strings.extend(self._extract_strings(v, depth + 1))
        return strings


register_middleware("microglial_defense", MicroglialDefense)
