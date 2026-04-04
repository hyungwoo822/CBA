"""Shell command execution tool."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from brain_agent.tools.base import Tool

logger = logging.getLogger(__name__)


class ShellTool(Tool):
    """Execute a shell command and return its output."""

    @property
    def name(self) -> str:
        return "shell"

    @property
    def description(self) -> str:
        return "Execute a shell command and return stdout/stderr."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 30)",
                    "default": 30,
                },
            },
            "required": ["command"],
        }

    async def execute(self, command: str = "", timeout: int = 30, **kwargs: Any) -> str:
        logger.info("[ShellTool] Executing: %s", command)
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
            output = stdout.decode(errors="replace")
            if stderr:
                output += "\n[stderr]\n" + stderr.decode(errors="replace")
            return output.strip() or "(no output)"
        except asyncio.TimeoutError:
            return f"Error: command timed out after {timeout}s"
        except Exception as e:
            return f"Error: {e}"
