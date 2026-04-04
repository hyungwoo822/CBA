"""Built-in tools for brain-agent."""
from brain_agent.tools.builtin.shell import ShellTool

BUILTIN_TOOLS: dict[str, type] = {
    "shell": ShellTool,
}

__all__ = ["BUILTIN_TOOLS", "ShellTool"]
