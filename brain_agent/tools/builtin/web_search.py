"""Web Search — hippocampal retrieval extended to external knowledge.

Like the hippocampus searches internal memory stores, this effector
queries external search engines to retrieve knowledge that the brain
has never encountered before.
"""
from __future__ import annotations

import logging
from typing import Any

from brain_agent.tools.base import Tool

logger = logging.getLogger(__name__)


class WebSearchTool(Tool):
    """Search the web and return summarised results."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web using DuckDuckGo and return the top results. "
            "Returns title, URL, and snippet for each result."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    async def execute(
        self, query: str = "", max_results: int = 5, **kwargs: Any,
    ) -> str:
        logger.info("[WebSearch] Query: %s", query)
        try:
            return await self._search_duckduckgo(query, max_results)
        except ImportError:
            return await self._search_fallback(query, max_results)
        except Exception as e:
            return f"Error: {e}"

    async def _search_duckduckgo(self, query: str, max_results: int) -> str:
        """Primary: use duckduckgo_search library."""
        from duckduckgo_search import AsyncDDGS

        results = []
        async with AsyncDDGS() as ddgs:
            async for r in ddgs.text(query, max_results=max_results):
                results.append(r)

        if not results:
            return "No results found."

        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            href = r.get("href", r.get("link", ""))
            body = r.get("body", r.get("snippet", ""))
            lines.append(f"{i}. {title}\n   {href}\n   {body}")
        return "\n\n".join(lines)

    async def _search_fallback(self, query: str, max_results: int) -> str:
        """Fallback: use aiohttp to query DuckDuckGo HTML."""
        try:
            import aiohttp
            import re
        except ImportError:
            return (
                "Error: web_search requires either 'duckduckgo_search' "
                "or 'aiohttp' package"
            )

        url = "https://html.duckduckgo.com/html/"
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                data={"q": query},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                html = await resp.text()

        # Extract result snippets from the HTML
        titles = re.findall(r'class="result__a"[^>]*>(.*?)</a>', html)
        snippets = re.findall(r'class="result__snippet">(.*?)</', html, re.DOTALL)
        links = re.findall(r'class="result__url"[^>]*href="([^"]*)"', html)

        if not titles:
            return "No results found."

        lines = []
        for i, (t, s, l) in enumerate(
            zip(titles[:max_results], snippets[:max_results], links[:max_results]), 1,
        ):
            t = re.sub(r"<[^>]+>", "", t).strip()
            s = re.sub(r"<[^>]+>", "", s).strip()
            lines.append(f"{i}. {t}\n   {l}\n   {s}")
        return "\n\n".join(lines) or "No results found."
