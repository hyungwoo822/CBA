"""Tests for ChannelAdapter base class event emission."""
import pytest
from unittest.mock import AsyncMock, patch

from brain_agent.channels.base import ChannelAdapter


class StubAdapter(ChannelAdapter):
    """Concrete adapter for testing."""
    name = "stub"

    async def start(self):
        pass

    async def stop(self):
        pass

    async def send_to_chat(self, text, chat_id=None):
        pass


@pytest.mark.asyncio
async def test_handle_message_emits_channel_events():
    agent = AsyncMock()
    agent.process = AsyncMock(return_value=AsyncMock(response="reply"))

    adapter = StubAdapter(agent)

    with patch("brain_agent.channels.base.event_bus") as mock_bus:
        mock_bus.emit = AsyncMock()
        result = await adapter.handle_message(text="hello")

    assert result == "reply"
    calls = mock_bus.emit.call_args_list
    # First call: inbound
    assert calls[0].args[0] == "channel_message"
    assert calls[0].args[1]["direction"] == "inbound"
    assert calls[0].args[1]["channel"] == "stub"
    assert calls[0].args[1]["text"] == "hello"
    # Second call: outbound
    assert calls[1].args[0] == "channel_message"
    assert calls[1].args[1]["direction"] == "outbound"
    assert calls[1].args[1]["text"] == "reply"
