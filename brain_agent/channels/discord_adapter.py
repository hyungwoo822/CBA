"""Discord channel adapter using discord.py (async).

Requires: pip install discord.py>=2.3

Set DISCORD_BOT_TOKEN in .env or pass token directly.
The bot responds to DMs and @mentions in guild channels.
"""
from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

from brain_agent.channels.base import ChannelAdapter, split_message

if TYPE_CHECKING:
    from brain_agent.agent import BrainAgent

logger = logging.getLogger(__name__)

DISCORD_MAX_MESSAGE_LENGTH = 2000


class DiscordAdapter(ChannelAdapter):
    """Discord bot adapter — routes messages through BrainAgent."""

    name = "discord"

    def __init__(self, agent: BrainAgent, token: str):
        super().__init__(agent)
        self._token = token
        self._client = None

    async def start(self) -> None:
        try:
            import discord
        except ImportError:
            raise RuntimeError(
                "discord.py is required: pip install 'discord.py>=2.3'"
            )

        intents = discord.Intents.default()
        intents.message_content = True

        self._client = discord.Client(intents=intents)
        adapter = self  # closure reference

        @self._client.event
        async def on_ready():
            logger.info("[discord] Bot ready as %s", self._client.user)

        @self._client.event
        async def on_message(message: discord.Message):
            # Ignore own messages
            if message.author == self._client.user:
                return

            # Respond to DMs or @mentions
            is_dm = message.guild is None
            is_mentioned = self._client.user in message.mentions if message.guild else False

            if not is_dm and not is_mentioned:
                return

            # Strip the mention from text
            text = message.content
            if is_mentioned and self._client.user:
                text = text.replace(f"<@{self._client.user.id}>", "").strip()

            # Handle attachments
            image_bytes = None
            audio_bytes = None
            for att in message.attachments:
                data = await att.read()
                if att.content_type and att.content_type.startswith("image/"):
                    image_bytes = data
                elif att.content_type and att.content_type.startswith("audio/"):
                    audio_bytes = data

            if not text and not image_bytes and not audio_bytes:
                return

            async with message.channel.typing():
                response = await adapter.handle_message(
                    text=text,
                    image=image_bytes,
                    audio=audio_bytes,
                )

            for chunk in split_message(response, DISCORD_MAX_MESSAGE_LENGTH):
                await message.channel.send(chunk)

        # Run the bot (non-blocking start)
        logger.info("[discord] Bot starting")
        await self._client.login(self._token)
        # start() is blocking, so we use a task
        import asyncio
        self._task = asyncio.create_task(self._client.connect())

    async def stop(self) -> None:
        if self._client and not self._client.is_closed():
            logger.info("[discord] Shutting down bot")
            await self._client.close()
