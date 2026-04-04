"""MCPClient — manages a single MCP server session."""
from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from typing import Any

from brain_agent.config.schema import MCPServerConfig
from brain_agent.mcp.transport import resolve_transport

logger = logging.getLogger(__name__)


class MCPClient:
    """Connects to one MCP server and exposes its tools."""

    def __init__(self, name: str, config: MCPServerConfig):
        self.name = name
        self.config = config
        self._session: Any | None = None
        self._exit_stack = AsyncExitStack()
        self._tools: list[dict[str, Any]] = []

    async def connect(self) -> None:
        """Establish connection and discover tools."""
        transport_info = await resolve_transport(self.config)
        transport_type = transport_info[0]

        try:
            if transport_type == "stdio":
                await self._connect_stdio(transport_info)
            else:
                await self._connect_http(transport_info)

            # Discover tools
            result = await self._session.list_tools()
            self._tools = [
                {
                    "name": tool.name,
                    "description": getattr(tool, "description", ""),
                    "inputSchema": getattr(tool, "inputSchema", {}),
                }
                for tool in result.tools
            ]
            logger.info("MCP [%s] connected — %d tools discovered", self.name, len(self._tools))
        except Exception as e:
            logger.error("MCP [%s] connection failed: %s", self.name, e)
            raise

    async def _connect_stdio(self, transport_info: tuple) -> None:
        from mcp.client.stdio import stdio_client
        from mcp import ClientSession

        _, params = transport_info
        read_stream, write_stream = await self._exit_stack.enter_async_context(
            stdio_client(params)
        )
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()

    async def _connect_http(self, transport_info: tuple) -> None:
        from mcp.client.sse import sse_client
        from mcp import ClientSession

        _, url, headers = transport_info
        read_stream, write_stream = await self._exit_stack.enter_async_context(
            sse_client(url=url, headers=headers)
        )
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()

    @property
    def tools(self) -> list[dict[str, Any]]:
        return self._tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool on the remote MCP server."""
        if not self._session:
            raise RuntimeError(f"MCP [{self.name}] not connected")
        result = await self._session.call_tool(name, arguments)
        # Extract text content
        if hasattr(result, "content") and result.content:
            parts = []
            for block in result.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
            return "\n".join(parts) if parts else str(result)
        return str(result)

    async def disconnect(self) -> None:
        """Close the session and transport."""
        await self._exit_stack.aclose()
        self._session = None
        self._tools = []
        logger.info("MCP [%s] disconnected", self.name)
