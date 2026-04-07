"""Web Fetch — visual cortex extension for reading external documents.

Like the visual cortex processes incoming light, this effector
retrieves textual information from a URL for the brain to process.
"""
from __future__ import annotations

import logging
from typing import Any

from brain_agent.tools.base import Tool

logger = logging.getLogger(__name__)


class WebFetchTool(Tool):
    """Fetch the text content of a URL."""

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return (
            "Fetch the text content of a given URL. Returns the page body "
            "as plain text (HTML tags stripped). Useful for reading articles, "
            "documentation, or API responses."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch",
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum characters to return (default 8000)",
                    "default": 8000,
                },
            },
            "required": ["url"],
        }

    async def execute(
        self, url: str = "", max_length: int = 8000, **kwargs: Any,
    ) -> str:
        logger.info("[WebFetch] Fetching: %s", url)
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        return f"Error: HTTP {resp.status}"
                    html = await resp.text()

            # Strip HTML tags for plain text extraction
            text = self._strip_html(html)
            if len(text) > max_length:
                text = text[:max_length] + "\n...(truncated)"
            return text or "(empty page)"
        except ImportError:
            return "Error: aiohttp is required for web_fetch (pip install aiohttp)"
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def _strip_html(html: str) -> str:
        """Lightweight HTML tag removal."""
        import re

        # Remove script/style blocks
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
        # Remove tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text
