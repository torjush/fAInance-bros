"""Global News Agent - Fetches and analyzes global financial news for macro context."""

import asyncio
import json
import logging
from typing import Any

import aiohttp
import anthropic

from config import Config, PROMPTS
from data.sources import GoogleNewsRSS
from utils import strip_code_blocks

logger = logging.getLogger(__name__)

# Queries for broad global financial context
GLOBAL_NEWS_QUERIES = [
    "global financial markets economy",
    "central bank interest rates inflation",
    "oil energy commodity prices geopolitics",
]


class GlobalNewsAgent:
    """
    Agent that fetches global financial news and extracts macro themes.

    Provides broader market context for the stock analysis — e.g. central bank
    decisions, commodity moves, or geopolitical events that may affect the
    company being analyzed.
    """

    def __init__(self, config: Config):
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    async def fetch(self) -> dict[str, Any]:
        """
        Fetch global financial news and extract key themes.

        Returns:
            Dictionary with news_items, key_themes, macro_events,
            market_sentiment, and summary.
        """
        logger.info("Fetching global financial news")

        async with aiohttp.ClientSession() as session:
            tasks = [
                GoogleNewsRSS.fetch_news(query, session=session, hl="en", gl="US")
                for query in GLOBAL_NEWS_QUERIES
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        all_news: list[dict] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Global news fetch failed: {result}")
                continue
            all_news.extend(result)

        # Deduplicate by URL
        seen: set[str] = set()
        unique_news: list[dict] = []
        for item in all_news:
            url = item.get("url", "")
            if url not in seen:
                seen.add(url)
                unique_news.append(item)

        if not unique_news:
            logger.warning("No global news fetched")
            return {
                "news_items": [],
                "key_themes": [],
                "macro_events": [],
                "market_sentiment": "neutral",
                "summary": "No global news available.",
            }

        logger.info(f"Fetched {len(unique_news)} unique global news items")
        return await self._extract_themes(unique_news[:25])

    async def _extract_themes(self, news_items: list[dict]) -> dict[str, Any]:
        """Use Claude Haiku to extract key global themes from headlines."""
        headlines = "\n".join(
            f"- {item.get('title', '')} ({item.get('source', 'Unknown')})"
            for item in news_items
        )

        prompt = PROMPTS["extract_global_news"].format(headlines=headlines)

        try:
            response = self.client.messages.create(
                model=self.config.extraction_model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text.strip()
            content = strip_code_blocks(content)
            extracted = json.loads(content)
        except Exception as e:
            logger.error(f"Failed to extract global news themes: {e}")
            extracted = {
                "market_sentiment": "neutral",
                "key_themes": [],
                "macro_events": [],
                "summary": "Unable to extract global news themes.",
            }

        return {
            "news_items": news_items,
            **extracted,
        }
