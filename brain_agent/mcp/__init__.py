"""MCP (Model Context Protocol) client — connects to local/remote MCP servers."""
from brain_agent.mcp.client import MCPClient
from brain_agent.mcp.registry import MCPRegistry
from brain_agent.mcp.proxy import MCPToolProxy

__all__ = ["MCPClient", "MCPRegistry", "MCPToolProxy"]
