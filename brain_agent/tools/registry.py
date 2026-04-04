"""Tool registry for dynamic tool management."""
from __future__ import annotations

import logging
from typing import Any

from brain_agent.tools.base import Tool

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    def get_definitions(self) -> list[dict[str, Any]]:
        return [t.to_schema() for t in self._tools.values()]

    async def execute(self, name: str, params: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if not tool:
            return f"Error: tool '{name}' not found"
        try:
            return await tool.execute(**params)
        except Exception as e:
            return f"Error executing {name}: {e}"

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    # --- Config-based loading ---

    def load_builtins(self, enabled: list[str]) -> None:
        """Load built-in tools by name from config."""
        from brain_agent.tools.builtin import BUILTIN_TOOLS

        for name in enabled:
            cls = BUILTIN_TOOLS.get(name)
            if cls is None:
                logger.warning("Unknown built-in tool: %s — skipped", name)
                continue
            self.register(cls())
            logger.info("Loaded built-in tool: %s", name)
