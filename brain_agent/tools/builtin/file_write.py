"""File Write — motor output to the local filesystem.

Like the motor cortex sends commands to muscles to produce physical
action, this effector writes content to local files — the agent's
primary means of producing persistent artifacts.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from brain_agent.tools.base import Tool

logger = logging.getLogger(__name__)


class FileWriteTool(Tool):
    """Write or append content to a local file."""

    @property
    def name(self) -> str:
        return "file_write"

    @property
    def description(self) -> str:
        return (
            "Write content to a file. Creates the file (and parent "
            "directories) if they do not exist. Use mode='append' to "
            "add to an existing file."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write",
                },
                "content": {
                    "type": "string",
                    "description": "The text content to write",
                },
                "mode": {
                    "type": "string",
                    "enum": ["write", "append"],
                    "description": "Write mode — 'write' overwrites, 'append' adds to end",
                    "default": "write",
                },
            },
            "required": ["path", "content"],
        }

    async def execute(
        self,
        path: str = "",
        content: str = "",
        mode: str = "write",
        **kwargs: Any,
    ) -> str:
        logger.info("[FileWrite] %s → %s (%d chars)", mode, path, len(content))
        path = os.path.expanduser(path)

        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            file_mode = "a" if mode == "append" else "w"
            with open(path, file_mode, encoding="utf-8") as f:
                f.write(content)
            return f"OK: {'appended to' if mode == 'append' else 'wrote'} {path} ({len(content)} chars)"
        except Exception as e:
            return f"Error: {e}"
