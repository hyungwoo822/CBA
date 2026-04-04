import pytest
from brain_agent.tools.base import Tool
from brain_agent.tools.registry import ToolRegistry


class MockTool(Tool):
    @property
    def name(self):
        return "mock_tool"

    @property
    def description(self):
        return "A test tool"

    @property
    def parameters(self):
        return {"type": "object", "properties": {"x": {"type": "string"}}}

    async def execute(self, **kwargs) -> str:
        return f"executed with {kwargs}"


def test_register_and_get():
    reg = ToolRegistry()
    reg.register(MockTool())
    assert reg.has("mock_tool")
    assert reg.get("mock_tool") is not None


def test_unregister():
    reg = ToolRegistry()
    reg.register(MockTool())
    reg.unregister("mock_tool")
    assert not reg.has("mock_tool")


def test_get_definitions():
    reg = ToolRegistry()
    reg.register(MockTool())
    defs = reg.get_definitions()
    assert len(defs) == 1
    assert defs[0]["function"]["name"] == "mock_tool"


async def test_execute():
    reg = ToolRegistry()
    reg.register(MockTool())
    result = await reg.execute("mock_tool", {"x": "hello"})
    assert "executed" in result


async def test_execute_missing_tool():
    reg = ToolRegistry()
    result = await reg.execute("nonexistent", {})
    assert "Error" in result


def test_tool_schema():
    t = MockTool()
    schema = t.to_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "mock_tool"
