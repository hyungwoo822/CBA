"""Tests for ChannelManager."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from brain_agent.channels.manager import ChannelManager


class FakeAdapter:
    """Minimal adapter stub for testing."""
    def __init__(self, name: str):
        self.name = name
        self.started = False
        self.stopped = False
        self.sent_messages: list[tuple[str, str | int | None]] = []

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True

    async def send_to_chat(self, text: str, chat_id=None):
        self.sent_messages.append((text, chat_id))


@pytest.fixture
def mgr():
    return ChannelManager()


def test_register_and_status(mgr):
    adapter = FakeAdapter("telegram")
    mgr.register(adapter)
    status = mgr.status()
    assert len(status) == 1
    assert status[0]["name"] == "telegram"
    assert status[0]["connected"] is True
    assert status[0]["broadcast"] is True


def test_set_broadcast(mgr):
    mgr.register(FakeAdapter("telegram"))
    mgr.set_broadcast("telegram", False)
    status = mgr.status()
    assert status[0]["broadcast"] is False


def test_set_last_chat_id(mgr):
    mgr.register(FakeAdapter("telegram"))
    mgr.set_last_chat_id("telegram", 12345)
    status = mgr.status()
    assert status[0]["last_chat_id"] == 12345


@pytest.mark.asyncio
async def test_broadcast_response_sends_to_enabled(mgr):
    tg = FakeAdapter("telegram")
    dc = FakeAdapter("discord")
    mgr.register(tg)
    mgr.register(dc)
    mgr.set_last_chat_id("telegram", 111)
    mgr.set_last_chat_id("discord", 222)
    mgr.set_broadcast("discord", False)

    await mgr.broadcast_response("hello")

    assert len(tg.sent_messages) == 1
    assert tg.sent_messages[0] == ("hello", 111)
    assert len(dc.sent_messages) == 0


@pytest.mark.asyncio
async def test_broadcast_response_excludes_channel(mgr):
    tg = FakeAdapter("telegram")
    dc = FakeAdapter("discord")
    mgr.register(tg)
    mgr.register(dc)
    mgr.set_last_chat_id("telegram", 111)
    mgr.set_last_chat_id("discord", 222)

    await mgr.broadcast_response("hello", exclude="telegram")

    assert len(tg.sent_messages) == 0
    assert len(dc.sent_messages) == 1


@pytest.mark.asyncio
async def test_broadcast_skips_no_chat_id(mgr):
    tg = FakeAdapter("telegram")
    mgr.register(tg)
    # No chat_id set

    await mgr.broadcast_response("hello")

    assert len(tg.sent_messages) == 0


@pytest.mark.asyncio
async def test_stop_all(mgr):
    tg = FakeAdapter("telegram")
    dc = FakeAdapter("discord")
    mgr.register(tg)
    mgr.register(dc)

    await mgr.stop_all()

    assert tg.stopped
    assert dc.stopped
