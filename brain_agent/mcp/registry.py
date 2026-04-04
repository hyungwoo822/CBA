"""MCPRegistry — manages multiple MCP servers and bridges to ToolRegistry."""
from __future__ import annotations

import logging
from typing import Any

from brain_agent.config.schema import MCPServerConfig
from brain_agent.mcp.client import MCPClient
from brain_agent.mcp.proxy import MCPToolProxy
from brain_agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class MCPRegistry:
    """Lifecycle manager for all configured MCP servers."""

    def __init__(self):
        self._clients: dict[str, MCPClient] = {}

    async def initialize(self, servers: dict[str, MCPServerConfig]) -> None:
        """Connect to all configured MCP servers."""
        for name, config in servers.items():
            client = MCPClient(name, config)
            try:
                await client.connect()
                self._clients[name] = client
            except Exception as e:
                logger.error("Failed to connect MCP server '%s': %s", name, e)

    def bridge_to_tool_registry(self, tool_registry: ToolRegistry) -> None:
        """Register all discovered MCP tools into the ToolRegistry."""
        for name, client in self._clients.items():
            for tool_schema in client.tools:
                tool_name = tool_schema["name"]
                # Prefix with server name to avoid collisions
                prefixed_name = f"{name}:{tool_name}"
                proxy = MCPToolProxy(
                    tool_name=prefixed_name,
                    tool_description=tool_schema.get("description", ""),
                    tool_parameters=tool_schema.get("inputSchema", {}),
                    call_fn=client.call_tool,
                )
                tool_registry.register(proxy)
                logger.info("Registered MCP tool: %s", prefixed_name)

    @property
    def clients(self) -> dict[str, MCPClient]:
        return self._clients

    async def shutdown(self) -> None:
        """Disconnect all MCP servers."""
        for name, client in self._clients.items():
            try:
                await client.disconnect()
            except Exception as e:
                logger.error("Error disconnecting MCP server '%s': %s", name, e)
        self._clients.clear()
