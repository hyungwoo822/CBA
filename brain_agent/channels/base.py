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


class ChannelAdapter(ABC):
    """Abstract base for all channel adapters."""

    name: str = "base"

    def __init__(self, agent: BrainAgent):
        self.agent = agent

    @abstractmethod
    async def start(self) -> None:
        """Start listening for messages on this channel."""

    @abstractmethod
    async def stop(self) -> None:
        """Gracefully shut down the channel."""

    async def handle_message(
        self,
        text: str = "",
        image: bytes | None = None,
        audio: bytes | None = None,
    ) -> str:
        """Route a message through the brain pipeline and return the response."""
        try:
            result = await self.agent.process(text, image=image, audio=audio)
            return result.response
        except Exception as e:
            logger.error("[%s] Pipeline error: %s", self.name, e)
            return "처리 중 오류가 발생했습니다."


def split_message(text: str, limit: int) -> list[str]:
    """Split a long message into chunks respecting the character limit."""
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        # Try to split at newline
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks
