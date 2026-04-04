from brain_agent.providers.base import LLMProvider, LLMResponse, ToolCallRequest


def test_llm_response_no_tool_calls():
    r = LLMResponse(content="hello")
    assert not r.has_tool_calls
    assert r.content == "hello"


def test_llm_response_with_tool_calls():
    r = LLMResponse(content=None, tool_calls=[
        ToolCallRequest(id="1", name="read_file", arguments={"path": "a.py"})
    ])
    assert r.has_tool_calls
    assert r.tool_calls[0].name == "read_file"


def test_tool_call_request():
    tc = ToolCallRequest(id="abc", name="test", arguments={"x": 1})
    assert tc.id == "abc"
    assert tc.arguments["x"] == 1


async def test_litellm_provider_instantiation():
    from brain_agent.providers.litellm_provider import LiteLLMProvider
    p = LiteLLMProvider(model="openai/gpt-4o-mini")
    assert p.get_default_model() == "openai/gpt-4o-mini"
