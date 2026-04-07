"""Built-in effectors for brain-agent.

Somatic effectors — directly controlled by the brain's motor system.
These are the agent's core action primitives, analogous to voluntary
muscles innervated by the somatic nervous system.
"""
from brain_agent.tools.builtin.shell import ShellTool
from brain_agent.tools.builtin.web_fetch import WebFetchTool
from brain_agent.tools.builtin.web_search import WebSearchTool
from brain_agent.tools.builtin.file_read import FileReadTool
from brain_agent.tools.builtin.file_write import FileWriteTool

BUILTIN_TOOLS: dict[str, type] = {
    "shell": ShellTool,
    "web_fetch": WebFetchTool,
    "web_search": WebSearchTool,
    "file_read": FileReadTool,
    "file_write": FileWriteTool,
}

__all__ = [
    "BUILTIN_TOOLS",
    "ShellTool",
    "WebFetchTool",
    "WebSearchTool",
    "FileReadTool",
    "FileWriteTool",
]
