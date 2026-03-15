"""Targeted News Agent - Fetches and analyzes sector/geography-specific news."""

import asyncio
import json
import logging
from typing import Any

import aiohttp
import anthropic

from config import Config, PROMPTS
from data.sources import GoogleNewsRSS
from data.storage import Storage
from utils import strip_code_blocks

logger = logging.getLogger(__name__)


class TargetedNewsAgent:
    """
    Agent that fetches news targeted to the company's specific sectors and geographies.

    Uses the search_queries from CompanyProfileAgent to fetch Google News RSS
    in parallel, then deduplicates and extracts sector/geo themes via Haiku.

    Raw articles are cached in news_cache; extracted themes are saved as an insight.
    """

    def __init__(self, storage: Storage, config: Config):
        self.storage = storage
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    async def fetch(
        self,
        ticker: str,
        company_profile: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Fetch and analyze targeted sector/geography news.

        Args:
            ticker: Stock ticker symbol
            company_profile: Output from CompanyProfileAgent

        Returns:
            {sector_themes, geo_themes, summary, raw_articles}
        """
        search_queries = company_profile.get("search_queries", [])

        if not search_queries:
            logger.warning(f"[Targeted News] No search queries for {ticker}, skipping")
            return self._empty_result()

        logger.info(
            f"[Targeted News] Fetching news for {ticker} "
            f"with {len(search_queries)} queries: {search_queries}"
        )

        # Fetch all queries in parallel
        async with aiohttp.ClientSession() as session:
            tasks = [
                GoogleNewsRSS.fetch_news(query, session=session, hl="en", gl="US")
                for query in search_queries
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge and deduplicate by URL, tracking which query found each article
        seen_urls: set[str] = set()
        all_articles: list[dict] = []
        for query, result in zip(search_queries, results):
            if isinstance(result, Exception):
                logger.warning(f"[Targeted News] Query '{query}' failed: {result}")
                continue
            for item in result:
                url = item.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_articles.append({**item, "query_tag": query})

        if not all_articles:
            logger.warning(f"[Targeted News] No articles fetched for {ticker}")
            return self._empty_result()

        logger.info(f"[Targeted News] Fetched {len(all_articles)} unique articles for {ticker}")

        # Cache raw articles (skip already-cached URLs)
        for item in all_articles:
            url = item.get("url", "")
            if url and not self.storage.news_exists(url):
                self.storage.save_news_item(
                    ticker=ticker,
                    url=url,
                    title=item.get("title", ""),
                    published=item.get("published"),
                    source=item.get("source", "Unknown"),
                    extracted_data={"query_tag": item.get("query_tag", "")},
                )

        # Extract themes via Haiku
        extracted = await self._extract_themes(all_articles[:30])

        # Persist themes as insight
        self.storage.save_insight(
            ticker=ticker,
            insight_type="targeted_news",
            content=extracted,
            summary=extracted.get("summary", ""),
        )

        return {**extracted, "raw_articles": all_articles}

    async def _extract_themes(self, articles: list[dict]) -> dict[str, Any]:
        """Use Claude Haiku to extract sector/geo themes from headlines."""
        headlines = "\n".join(
            f"[{item.get('query_tag', '')}] {item.get('title', '')} ({item.get('source', 'Unknown')})"
            for item in articles
        )

        prompt = PROMPTS["extract_targeted_news"].format(headlines=headlines)

        try:
            response = self.client.messages.create(
                model=self.config.extraction_model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text.strip()
            content = strip_code_blocks(content)
            return json.loads(content)

        except Exception as e:
            logger.error(f"[Targeted News] Theme extraction failed: {e}")
            return {
                "sector_themes": [],
                "geo_themes": [],
                "summary": "Unable to extract targeted news themes.",
            }

    def _empty_result(self) -> dict[str, Any]:
        return {
            "sector_themes": [],
            "geo_themes": [],
            "summary": "No targeted news available.",
            "raw_articles": [],
        }
