"""File Read — sensory input from the local filesystem.

Like the retina transduces light into neural signals, this effector
reads local files and converts them into text for the brain to process.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from brain_agent.tools.base import Tool

logger = logging.getLogger(__name__)

# Maximum file size to read (bytes) — prevents memory exhaustion
_MAX_READ_BYTES = 512_000  # 500 KB


class FileReadTool(Tool):
    """Read the contents of a local file."""

    @property
    def name(self) -> str:
        return "file_read"

    @property
    def description(self) -> str:
        return (
            "Read the contents of a file from the local filesystem. "
            "Returns the text content. For large files, only the first "
            "500 KB is returned."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read",
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (0-based, default 0)",
                    "default": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to return (default: all)",
                    "default": 0,
                },
            },
            "required": ["path"],
        }

    async def execute(
        self, path: str = "", offset: int = 0, limit: int = 0, **kwargs: Any,
    ) -> str:
        logger.info("[FileRead] Reading: %s", path)
        path = os.path.expanduser(path)

        if not os.path.exists(path):
            return f"Error: file not found — {path}"
        if os.path.isdir(path):
            return f"Error: '{path}' is a directory, not a file"
        if os.path.getsize(path) > _MAX_READ_BYTES:
            return (
                f"Error: file exceeds {_MAX_READ_BYTES // 1000} KB limit. "
                f"Use offset/limit parameters to read in chunks."
            )

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            if offset:
                lines = lines[offset:]
            if limit:
                lines = lines[:limit]

            return "".join(lines) or "(empty file)"
        except Exception as e:
            return f"Error: {e}"
