"""Channel adapter registry with broadcast control."""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain_agent.agent import BrainAgent
    from brain_agent.channels.base import ChannelAdapter

logger = logging.getLogger(__name__)

# Mapping: env var -> (module path, class name)
_CHANNEL_REGISTRY: list[tuple[str, str, str]] = [
    ("TELEGRAM_BOT_TOKEN", "brain_agent.channels.telegram_adapter", "TelegramAdapter"),
    ("DISCORD_BOT_TOKEN", "brain_agent.channels.discord_adapter", "DiscordAdapter"),
]


class ChannelManager:
    """Registry of active channel adapters with broadcast control."""

    def __init__(self):
        self._adapters: dict[str, ChannelAdapter] = {}
        self._broadcast: dict[str, bool] = {}
        self._last_chat_ids: dict[str, int | str] = {}

    def register(self, adapter: ChannelAdapter, broadcast: bool = True) -> None:
        """Register an already-created adapter."""
        self._adapters[adapter.name] = adapter
        self._broadcast[adapter.name] = broadcast

    async def start_all(self, agent: BrainAgent) -> None:
        """Discover configured channels from env vars, create and start them."""
        import importlib

        for env_var, module_path, class_name in _CHANNEL_REGISTRY:
            token = os.environ.get(env_var, "")
            if not token:
                print(f"[channels] {env_var} not set, skipping {class_name}")
                continue
            try:
                mod = importlib.import_module(module_path)
                cls = getattr(mod, class_name)
                adapter = cls(agent, token=token)
                adapter._channel_mgr = self
                await adapter.start()
                self.register(adapter)
                print(f"[channels] {adapter.name} started")
            except Exception as e:
                print(f"[channels] Failed to start {class_name}: {e}")

    async def stop_all(self) -> None:
        """Stop all active adapters."""
        for name, adapter in self._adapters.items():
            try:
                await adapter.stop()
                logger.info("[channels] %s stopped", name)
            except Exception as e:
                logger.error("[channels] Error stopping %s: %s", name, e)

    def set_broadcast(self, channel: str, enabled: bool) -> None:
        """Toggle broadcast for a channel."""
        if channel in self._adapters:
            self._broadcast[channel] = enabled

    def set_last_chat_id(self, channel: str, chat_id: int | str) -> None:
        """Track the last active chat_id for a channel."""
        if channel in self._adapters:
            self._last_chat_ids[channel] = chat_id

    async def broadcast_response(self, text: str, exclude: str = "") -> None:
        """Send response to all broadcast-enabled channels."""
        for name, adapter in self._adapters.items():
            if name == exclude:
                continue
            if not self._broadcast.get(name, False):
                continue
            chat_id = self._last_chat_ids.get(name)
            if chat_id is None:
                continue
            try:
                await adapter.send_to_chat(text, chat_id=chat_id)
            except Exception as e:
                logger.error("[channels] Broadcast to %s failed: %s", name, e)

    def status(self) -> list[dict]:
        """Return channel list with name, connected, broadcast, last_chat_id."""
        return [
            {
                "name": name,
                "connected": True,
                "broadcast": self._broadcast.get(name, True),
                "last_chat_id": self._last_chat_ids.get(name),
            }
            for name in self._adapters
        ]
