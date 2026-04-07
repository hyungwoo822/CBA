"""Telegram channel adapter using python-telegram-bot (async).

Requires: pip install python-telegram-bot>=21.0

Set TELEGRAM_BOT_TOKEN in .env or pass token directly.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from brain_agent.channels.base import ChannelAdapter, split_message

if TYPE_CHECKING:
    from brain_agent.agent import BrainAgent

logger = logging.getLogger(__name__)

TELEGRAM_MAX_MESSAGE_LENGTH = 4096


class TelegramAdapter(ChannelAdapter):
    """Telegram bot adapter — routes messages through BrainAgent."""

    name = "telegram"

    def __init__(self, agent: BrainAgent, token: str):
        super().__init__(agent)
        self._token = token
        self._app = None

    async def start(self) -> None:
        try:
            from telegram.ext import (
                Application,
                CommandHandler,
                MessageHandler,
                filters,
            )
        except ImportError:
            raise RuntimeError(
                "python-telegram-bot is required: pip install 'python-telegram-bot>=21.0'"
            )

        self._app = (
            Application.builder()
            .token(self._token)
            .build()
        )

        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_text)
        )
        self._app.add_handler(
            MessageHandler(filters.PHOTO, self._on_photo)
        )
        self._app.add_handler(
            MessageHandler(filters.VOICE | filters.AUDIO, self._on_audio)
        )

        logger.info("[telegram] Bot starting — polling for updates")
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)

    async def stop(self) -> None:
        if self._app:
            logger.info("[telegram] Shutting down bot")
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

    async def send_to_chat(self, text: str, chat_id: int | str | None = None) -> None:
        """Send a message to a Telegram chat (used by broadcast)."""
        if not self._app or not chat_id:
            return
        for chunk in split_message(text, TELEGRAM_MAX_MESSAGE_LENGTH):
            await self._app.bot.send_message(chat_id=chat_id, text=chunk)

    # ── Handlers ─────────────────────────────────────────────

    def _track_chat(self, update) -> None:
        """Report last active chat_id to ChannelManager."""
        if self._channel_mgr and update.message:
            self._channel_mgr.set_last_chat_id("telegram", update.message.chat_id)

    async def _cmd_start(self, update, context) -> None:
        self._track_chat(update)
        await update.message.reply_text(
            "안녕하세요! 메시지를 보내주세요."
        )

    async def _on_text(self, update, context) -> None:
        text = update.message.text or ""
        if not text.strip():
            return
        self._track_chat(update)
        response = await self.handle_message(text=text)
        await self._send_reply(update, response)

    async def _on_photo(self, update, context) -> None:
        self._track_chat(update)
        caption = update.message.caption or ""
        photo = update.message.photo[-1]  # highest resolution
        file = await context.bot.get_file(photo.file_id)
        image_bytes = await file.download_as_bytearray()
        response = await self.handle_message(text=caption, image=bytes(image_bytes))
        await self._send_reply(update, response)

    async def _on_audio(self, update, context) -> None:
        self._track_chat(update)
        voice = update.message.voice or update.message.audio
        if not voice:
            return
        file = await context.bot.get_file(voice.file_id)
        audio_bytes = await file.download_as_bytearray()
        caption = update.message.caption or ""
        response = await self.handle_message(text=caption, audio=bytes(audio_bytes))
        await self._send_reply(update, response)

    async def _send_reply(self, update, text: str) -> None:
        for chunk in split_message(text, TELEGRAM_MAX_MESSAGE_LENGTH):
            await update.message.reply_text(chunk)
