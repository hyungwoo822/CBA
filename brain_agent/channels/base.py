"""Base channel adapter for brain_agent.

All messaging channels (Telegram, Discord, etc.) inherit from ChannelAdapter
and share the same BrainAgent instance. The adapter handles platform-specific
message receiving/sending while the agent does all the thinking.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain_agent.agent import BrainAgent

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependency at module load time.
_event_bus = None


def _get_event_bus():
    global _event_bus
    if _event_bus is None:
        try:
            from brain_agent.dashboard.server import event_bus as _bus
            _event_bus = _bus
        except Exception:
            _event_bus = None
    return _event_bus


class _EventBusProxy:
    """Proxy so tests can patch `base.event_bus.emit`."""
    async def emit(self, event_type, payload):
        bus = _get_event_bus()
        if bus:
            await bus.emit(event_type, payload)

event_bus = _EventBusProxy()


class ChannelAdapter(ABC):
    """Abstract base for all channel adapters."""

    name: str = "base"

    def __init__(self, agent: BrainAgent):
        self.agent = agent
        self._channel_mgr = None  # Set by ChannelManager on registration

    @abstractmethod
    async def start(self) -> None:
        """Start listening for messages on this channel."""

    @abstractmethod
    async def stop(self) -> None:
        """Gracefully shut down the channel."""

    @abstractmethod
    async def send_to_chat(self, text: str, chat_id: int | str | None = None) -> None:
        """Send a message to a specific chat (for broadcast from dashboard)."""

    async def handle_message(
        self,
        text: str = "",
        image: bytes | None = None,
        audio: bytes | None = None,
    ) -> str:
        """Route a message through the brain pipeline and return the response."""
        await event_bus.emit("channel_message", {
            "channel": self.name,
            "direction": "inbound",
            "text": text,
            "has_image": image is not None,
            "has_audio": audio is not None,
        })

        try:
            result = await self.agent.process(text, image=image, audio=audio)
            response = result.response
        except Exception as e:
            logger.error("[%s] Pipeline error: %s", self.name, e)
            response = "처리 중 오류가 발생했습니다."

        await event_bus.emit("channel_message", {
            "channel": self.name,
            "direction": "outbound",
            "text": response,
        })

        return response


def split_message(text: str, limit: int) -> list[str]:
    """Split a long message into chunks respecting the character limit."""
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks
