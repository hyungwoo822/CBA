"""MCPToolProxy — wraps an MCP remote tool as a local Tool for ToolRegistry."""
from __future__ import annotations

from typing import Any

from brain_agent.tools.base import Tool


class MCPToolProxy(Tool):
    """Adapts an MCP server tool to the brain-agent Tool interface.

    This makes MCP tools indistinguishable from built-in tools
    when accessed through ToolRegistry.
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

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    async def execute(self, **kwargs: Any) -> str:
        result = await self._call_fn(self._name, kwargs)
        return str(result)
