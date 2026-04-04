"""MCP transport resolver — stdio / SSE / Streamable HTTP."""
from __future__ import annotations

import logging
from typing import Any

from brain_agent.config.schema import MCPServerConfig

logger = logging.getLogger(__name__)


async def resolve_transport(config: MCPServerConfig) -> Any:
    """Create the appropriate MCP transport based on config.

    - ``command`` present → stdio (spawn local process)
    - ``url`` present → SSE or Streamable HTTP
    """
    if config.command:
        return await _create_stdio_transport(config)
    if config.url:
        return await _create_http_transport(config)
    raise ValueError("MCPServerConfig must have either 'command' (stdio) or 'url' (http)")


async def _create_stdio_transport(config: MCPServerConfig) -> Any:
    from mcp.client.stdio import StdioServerParameters, stdio_client  # noqa: F811

    params = StdioServerParameters(
        command=config.command,
        args=config.args,
        env=config.env if config.env else None,
        cwd=config.cwd,
    )
    return ("stdio", params)


async def _create_http_transport(config: MCPServerConfig) -> Any:
    transport_type = config.transport or "sse"

    if transport_type == "sse":
        from mcp.client.sse import sse_client  # noqa: F811
        return ("sse", config.url, config.headers)
    elif transport_type == "streamable-http":
        # Streamable HTTP — uses the same SSE client with different config
        from mcp.client.sse import sse_client  # noqa: F811
        return ("streamable-http", config.url, config.headers)
    else:
        raise ValueError(f"Unknown MCP transport type: {transport_type}")
