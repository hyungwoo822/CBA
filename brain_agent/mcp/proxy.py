"""MCPToolProxy — wraps an MCP remote tool as a local Tool for ToolRegistry.

In the autonomic nervous system, visceral signals travel via the vagus
nerve between internal organs and the brainstem.  MCP tools are the
digital equivalent — remote capabilities mediated by a standardised
protocol.

Each proxy carries a ``risk_level`` that the Basal Ganglia can use
for go/no-go gating, just as the amygdala modulates autonomic output
based on threat assessment.
"""
from __future__ import annotations

import re
from typing import Any

from brain_agent.tools.base import Tool

# Patterns that indicate higher-risk (write/mutate) operations
_WRITE_PATTERNS = re.compile(
    r"(?:send|write|create|delete|remove|update|post|put|patch|push|publish)",
    re.IGNORECASE,
)
# Patterns that indicate lower-risk (read-only) operations
_READ_PATTERNS = re.compile(
    r"(?:read|get|list|search|fetch|query|poll|status|check|describe)",
    re.IGNORECASE,
)


class MCPToolProxy(Tool):
    """Adapts an MCP server tool to the brain-agent Tool interface.

    This makes MCP tools indistinguishable from built-in tools
    when accessed through ToolRegistry, while carrying autonomic
    metadata (risk_level) for neural gating.
    """

    def __init__(
        self,
        tool_name: str,
        tool_description: str,
        tool_parameters: dict[str, Any],
        call_fn,
    ):
        self._name = tool_name
        self._description = tool_description
        self._parameters = tool_parameters
        self._call_fn = call_fn  # MCPClient.call_tool bound to this tool's name
        self._risk_level = self._assess_risk(tool_name, tool_description)

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    @property
    def risk_level(self) -> float:
        """Autonomic risk assessment (0.0 = safe read, 1.0 = dangerous mutation).

        Used by Basal Ganglia nogo_score: higher risk requires higher
        confidence (go_score) to proceed, exactly like the amygdala
        raising sympathetic tone before a potentially harmful action.
        """
        return self._risk_level

    async def execute(self, **kwargs: Any) -> str:
        result = await self._call_fn(self._name, kwargs)
        return str(result)

    def to_schema(self) -> dict[str, Any]:
        """Extended schema with risk metadata for neural gating."""
        schema = super().to_schema()
        schema["function"]["metadata"] = {
            "risk_level": self._risk_level,
            "source": "mcp",
        }
        return schema

    @staticmethod
    def _assess_risk(name: str, description: str) -> float:
        """Classify tool risk based on name/description semantics.

        This is analogous to the amygdala's rapid threat assessment
        (LeDoux 1996) — a fast, heuristic evaluation that biases
        the system toward caution for unknown or write-oriented tools.
        """
        combined = f"{name} {description}"

        if _WRITE_PATTERNS.search(combined):
            return 0.7  # Write/mutate = elevated risk
        if _READ_PATTERNS.search(combined):
            return 0.2  # Read-only = low risk
        return 0.5  # Unknown = moderate caution
