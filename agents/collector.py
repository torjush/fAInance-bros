"""Data Collection Agent - Fetches and extracts data from external sources."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

import anthropic

from config import Config, PROMPTS
from data.sources import fetch_all_data, YFinanceSource
from data.storage import Storage
from utils import strip_code_blocks

logger = logging.getLogger(__name__)


class CollectorAgent:
    """
    Agent responsible for collecting data from external sources.

    This agent:
    - Fetches stock prices via yfinance
    - Fetches news via Google News RSS
    - Fetches regulatory filings via Oslo Børs Newsweb
    - Uses Claude Haiku to extract structured data from news/filings
    - Caches all data to avoid re-fetching
    """

    def __init__(self, storage: Storage, config: Config):
        self.storage = storage
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    async def collect(
        self,
        ticker: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Collect all relevant data for a ticker.

        Args:
            ticker: Stock ticker symbol
            context: Context from the ContextAgent

        Returns:
            Dictionary containing all collected and processed data
        """
        logger.info(f"Starting data collection for {ticker}")

        # Determine if we need incremental or full fetch
        start_date = None
        if not context.get("is_new_ticker"):
            last_analyzed = context.get("last_analyzed")
            if last_analyzed:
                start_date = last_analyzed.split("T")[0]  # Get just the date part
                logger.info(f"Incremental fetch from {start_date}")

        # Get company name for news search
        company_name = None
        if context.get("company"):
            company_name = context["company"].get("name")

        # Fetch all data in parallel
        raw_data = await fetch_all_data(
            ticker=ticker,
            company_name=company_name,
            start_date=start_date,
        )

        # Update company info in database
        stock_info = raw_data.get("stock_info", {})
        self.storage.upsert_company(
            ticker=ticker,
            name=stock_info.get("name"),
            sector=stock_info.get("sector"),
        )

        # Save price data
        prices = raw_data.get("prices", [])
        if prices:
            self.storage.save_prices(ticker, prices)
            logger.info(f"Saved {len(prices)} price records")

        # Process and cache news (with LLM extraction for new items)
        news_items = raw_data.get("news", [])
        processed_news = await self._process_news(ticker, news_items)

        # Process and cache filings
        filings = raw_data.get("filings", [])
        processed_filings = await self._process_filings(ticker, filings)

        # Compile results
        result = {
            "ticker": ticker,
            "stock_info": stock_info,
            "prices": prices,
            "news": processed_news,
            "filings": processed_filings,
            "macro": raw_data.get("macro"),
            "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
            "is_incremental": start_date is not None,
        }

        logger.info(
            f"Data collection complete for {ticker}: "
            f"{len(prices)} prices, {len(processed_news)} news, {len(processed_filings)} filings"
        )

        return result

    async def _process_news(
        self,
        ticker: str,
        news_items: list[dict],
    ) -> list[dict]:
        """Process news items, extracting structured data with LLM."""
        processed = []

        for item in news_items:
            url = item.get("url", "")

            # Check if already cached
            if self.storage.news_exists(url):
                logger.debug(f"News already cached: {url}")
                continue

            # Extract structured data using Haiku
            extracted = await self._extract_news_data(ticker, item)

            # Save to cache
            self.storage.save_news_item(
                ticker=ticker,
                url=url,
                title=item.get("title", ""),
                published=item.get("published"),
                source=item.get("source", "Unknown"),
                extracted_data=extracted,
            )

            processed.append({
                **item,
                "extracted": extracted,
            })

        # Also include recently cached news
        cached = self.storage.get_cached_news(ticker)
        for item in cached:
            if item not in processed:
                processed.append(item)

        return processed[:30]  # Limit to most recent 30

    async def _process_filings(
        self,
        ticker: str,
        filings: list[dict],
    ) -> list[dict]:
        """Process filings, extracting structured data with LLM."""
        processed = []

        for filing in filings:
            url = filing.get("url", "")

            # Check if already cached
            if self.storage.filing_exists(url):
                logger.debug(f"Filing already cached: {url}")
                continue

            # Extract structured data using Haiku
            extracted = await self._extract_filing_data(ticker, filing)

            # Save to cache
            self.storage.save_filing(
                ticker=ticker,
                url=url,
                title=filing.get("title", ""),
                published=filing.get("published"),
                filing_type=filing.get("filing_type"),
                extracted_data=extracted,
            )

            processed.append({
                **filing,
                "extracted": extracted,
            })

        # Also include recently cached filings
        cached = self.storage.get_cached_filings(ticker)
        for item in cached:
            if item not in processed:
                processed.append(item)

        return processed[:20]  # Limit to most recent 20

    async def _extract_news_data(
        self,
        ticker: str,
        news_item: dict,
    ) -> dict | None:
        """Use Claude Haiku to extract structured data from news."""
        try:
            # Build article text from available data
            article_text = f"""
Title: {news_item.get('title', 'No title')}
Source: {news_item.get('source', 'Unknown')}
Published: {news_item.get('published', 'Unknown')}
"""

            prompt = PROMPTS["extract_news"].format(
                ticker=ticker,
                article_text=article_text,
            )

            response = self.client.messages.create(
                model=self.config.extraction_model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse JSON response (strip code blocks if present)
            content = response.content[0].text.strip()
            content = strip_code_blocks(content)
            return json.loads(content)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse news extraction JSON: {e}")
            logger.warning(f"Response (first 300 chars): {content[:300]}")
            return None
        except Exception as e:
            logger.error(f"Error extracting news data: {e}")
            return None

    async def _extract_filing_data(
        self,
        ticker: str,
        filing: dict,
    ) -> dict | None:
        """Use Claude Haiku to extract structured data from filing."""
        try:
            filing_text = f"""
Title: {filing.get('title', 'No title')}
Type: {filing.get('filing_type', 'Unknown')}
Published: {filing.get('published', 'Unknown')}
Description: {filing.get('description', 'No description available')}
"""

            prompt = PROMPTS["extract_filing"].format(
                ticker=ticker,
                filing_text=filing_text,
            )

            response = self.client.messages.create(
                model=self.config.extraction_model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.content[0].text.strip()
            content = strip_code_blocks(content)
            return json.loads(content)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse filing extraction JSON: {e}")
            logger.warning(f"Response (first 300 chars): {content[:300]}")
            return None
        except Exception as e:
            logger.error(f"Error extracting filing data: {e}")
            return None

    def calculate_price_stats(self, prices: list[dict]) -> dict:
        """Calculate price statistics for analysis."""
        if not prices:
            return {}

        closes = [p["close"] for p in prices if p.get("close")]
        volumes = [p["volume"] for p in prices if p.get("volume")]

        if not closes:
            return {}

        # Calculate basic stats
        current = closes[0]  # Most recent
        high = max(closes)
        low = min(closes)
        avg = sum(closes) / len(closes)

        # Calculate returns
        if len(closes) >= 2:
            daily_return = (closes[0] - closes[1]) / closes[1] * 100
        else:
            daily_return = 0

        if len(closes) >= 5:
            weekly_return = (closes[0] - closes[4]) / closes[4] * 100
        else:
            weekly_return = 0

        if len(closes) >= 22:
            monthly_return = (closes[0] - closes[21]) / closes[21] * 100
        else:
            monthly_return = 0

        # Volatility (standard deviation of daily returns)
        if len(closes) >= 2:
            returns = [(closes[i] - closes[i+1]) / closes[i+1] for i in range(len(closes)-1)]
            volatility = (sum((r - sum(returns)/len(returns))**2 for r in returns) / len(returns)) ** 0.5
        else:
            volatility = 0

        return {
            "current_price": round(current, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "average": round(avg, 2),
            "daily_return_pct": round(daily_return, 2),
            "weekly_return_pct": round(weekly_return, 2),
            "monthly_return_pct": round(monthly_return, 2),
            "volatility": round(volatility * 100, 2),
            "avg_volume": round(sum(volumes) / len(volumes)) if volumes else 0,
            "data_points": len(closes),
        }
