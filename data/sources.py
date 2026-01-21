"""Data source integrations for stock data, news, and filings."""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import aiohttp
import yfinance as yf

logger = logging.getLogger(__name__)


class YFinanceSource:
    """Yahoo Finance data source for stock prices."""

    @staticmethod
    def get_stock_info(ticker: str) -> dict:
        """Get basic stock information."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                "ticker": ticker,
                "name": info.get("longName") or info.get("shortName", ticker),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "currency": info.get("currency", "NOK"),
                "exchange": info.get("exchange", "OSL"),
            }
        except Exception as e:
            logger.error(f"Error fetching stock info for {ticker}: {e}")
            return {"ticker": ticker, "name": ticker, "sector": "Unknown"}

    @staticmethod
    def get_price_history(
        ticker: str,
        start_date: str = None,
        end_date: str = None,
        period: str = "1y",
    ) -> list[dict]:
        """
        Fetch historical price data.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            period: Period to fetch if dates not provided (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
        """
        try:
            stock = yf.Ticker(ticker)

            if start_date and end_date:
                df = stock.history(start=start_date, end=end_date)
            else:
                df = stock.history(period=period)

            if df.empty:
                logger.warning(f"No price data found for {ticker}")
                return []

            prices = []
            for date, row in df.iterrows():
                prices.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "open": round(row["Open"], 2) if row["Open"] else None,
                    "high": round(row["High"], 2) if row["High"] else None,
                    "low": round(row["Low"], 2) if row["Low"] else None,
                    "close": round(row["Close"], 2) if row["Close"] else None,
                    "volume": int(row["Volume"]) if row["Volume"] else None,
                })

            logger.info(f"Fetched {len(prices)} price records for {ticker}")
            return prices

        except Exception as e:
            logger.error(f"Error fetching price history for {ticker}: {e}")
            return []


class GoogleNewsRSS:
    """Google News RSS feed for company news."""

    BASE_URL = "https://news.google.com/rss/search"

    @staticmethod
    async def fetch_news(
        query: str,
        max_results: int = 20,
        session: aiohttp.ClientSession = None,
    ) -> list[dict]:
        """
        Fetch news from Google News RSS.

        Args:
            query: Search query (company name or ticker)
            max_results: Maximum number of results to return
            session: Optional aiohttp session for connection reuse
        """
        # Build URL with query
        encoded_query = quote_plus(query)
        url = f"{GoogleNewsRSS.BASE_URL}?q={encoded_query}&hl=no&gl=NO&ceid=NO:no"

        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True

        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    logger.error(f"Google News RSS returned status {response.status}")
                    return []

                content = await response.text()
                return GoogleNewsRSS._parse_rss(content, max_results)

        except asyncio.TimeoutError:
            logger.error("Google News RSS request timed out")
            return []
        except Exception as e:
            logger.error(f"Error fetching Google News: {e}")
            return []
        finally:
            if close_session:
                await session.close()

    @staticmethod
    def _parse_rss(content: str, max_results: int) -> list[dict]:
        """Parse RSS feed content."""
        news_items = []

        try:
            root = ET.fromstring(content)
            channel = root.find("channel")

            if channel is None:
                return []

            for item in channel.findall("item")[:max_results]:
                title = item.find("title")
                link = item.find("link")
                pub_date = item.find("pubDate")
                source = item.find("source")

                if title is not None and link is not None:
                    # Parse publication date
                    published = None
                    if pub_date is not None and pub_date.text:
                        try:
                            # Google News uses RFC 2822 format
                            from email.utils import parsedate_to_datetime
                            published = parsedate_to_datetime(pub_date.text).isoformat()
                        except Exception:
                            published = datetime.utcnow().isoformat()

                    news_items.append({
                        "title": title.text,
                        "url": link.text,
                        "published": published,
                        "source": source.text if source is not None else "Unknown",
                    })

        except ET.ParseError as e:
            logger.error(f"Error parsing RSS feed: {e}")

        logger.info(f"Parsed {len(news_items)} news items")
        return news_items


class OsloBorsNewsweb:
    """Oslo Børs Newsweb RSS feed for regulatory filings."""

    @staticmethod
    async def fetch_filings(
        ticker: str,
        max_results: int = 30,
        session: aiohttp.ClientSession = None,
    ) -> list[dict]:
        """
        Fetch regulatory filings from Oslo Børs Newsweb.

        Note: The ticker should be without the .OL suffix (e.g., "EQNR" not "EQNR.OL")
        """
        # Remove .OL suffix if present
        clean_ticker = ticker.replace(".OL", "").replace(".ol", "")

        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True

        try:
            # Try RSS feed first
            rss_url = f"https://newsweb.oslobors.no/rss?issuer={clean_ticker}"

            async with session.get(
                rss_url,
                timeout=aiohttp.ClientTimeout(total=30),
                headers={"User-Agent": "Mozilla/5.0 (compatible; StockAnalyzer/1.0)"}
            ) as response:
                if response.status != 200:
                    logger.warning(f"Newsweb RSS returned status {response.status}, trying HTML scrape")
                    return await OsloBorsNewsweb._scrape_html(clean_ticker, max_results, session)

                content = await response.text()
                filings = OsloBorsNewsweb._parse_rss(content, max_results)

                if not filings:
                    # Fallback to HTML scraping if RSS is empty
                    return await OsloBorsNewsweb._scrape_html(clean_ticker, max_results, session)

                return filings

        except Exception as e:
            logger.error(f"Error fetching Newsweb filings: {e}")
            return []
        finally:
            if close_session:
                await session.close()

    @staticmethod
    def _parse_rss(content: str, max_results: int) -> list[dict]:
        """Parse Newsweb RSS feed."""
        filings = []

        try:
            root = ET.fromstring(content)
            channel = root.find("channel")

            if channel is None:
                return []

            for item in channel.findall("item")[:max_results]:
                title = item.find("title")
                link = item.find("link")
                pub_date = item.find("pubDate")
                description = item.find("description")
                category = item.find("category")

                if title is not None and link is not None:
                    published = None
                    if pub_date is not None and pub_date.text:
                        try:
                            from email.utils import parsedate_to_datetime
                            published = parsedate_to_datetime(pub_date.text).isoformat()
                        except Exception:
                            published = datetime.utcnow().isoformat()

                    filings.append({
                        "title": title.text,
                        "url": link.text,
                        "published": published,
                        "filing_type": category.text if category is not None else "Filing",
                        "description": description.text if description is not None else "",
                    })

        except ET.ParseError as e:
            logger.error(f"Error parsing Newsweb RSS: {e}")

        logger.info(f"Parsed {len(filings)} filings from Newsweb RSS")
        return filings

    @staticmethod
    async def _scrape_html(
        ticker: str,
        max_results: int,
        session: aiohttp.ClientSession,
    ) -> list[dict]:
        """Fallback HTML scraping for Newsweb."""
        url = f"https://newsweb.oslobors.no/search?category=&issuer={ticker}&fromDate=&toDate=&market=&messageTitle="

        try:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=30),
                headers={"User-Agent": "Mozilla/5.0 (compatible; StockAnalyzer/1.0)"}
            ) as response:
                if response.status != 200:
                    logger.error(f"Newsweb HTML returned status {response.status}")
                    return []

                html = await response.text()

                # Simple regex-based extraction (basic fallback)
                pattern = r'href="(/message/\d+)"[^>]*>([^<]+)</a>'
                matches = re.findall(pattern, html)

                filings = []
                for path, title in matches[:max_results]:
                    filings.append({
                        "title": title.strip(),
                        "url": f"https://newsweb.oslobors.no{path}",
                        "published": datetime.utcnow().isoformat(),
                        "filing_type": "Filing",
                        "description": "",
                    })

                logger.info(f"Scraped {len(filings)} filings from Newsweb HTML")
                return filings

        except Exception as e:
            logger.error(f"Error scraping Newsweb HTML: {e}")
            return []


class NorgesBankAPI:
    """Norges Bank API for key policy rate (macro context)."""

    BASE_URL = "https://data.norges-bank.no/api"

    @staticmethod
    async def get_key_policy_rate(session: aiohttp.ClientSession = None) -> dict | None:
        """Get the current key policy rate from Norges Bank."""
        url = f"{NorgesBankAPI.BASE_URL}/data/IR/B.KPRA.SD.NOK?format=sdmx-json&lastNObservations=1"

        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True

        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    logger.error(f"Norges Bank API returned status {response.status}")
                    return None

                data = await response.json()

                # Parse SDMX-JSON response
                observations = data.get("data", {}).get("dataSets", [{}])[0].get("series", {})
                if observations:
                    for series_key, series_data in observations.items():
                        obs = series_data.get("observations", {})
                        if obs:
                            latest_key = max(obs.keys())
                            value = obs[latest_key][0]
                            return {
                                "rate": value,
                                "type": "Key Policy Rate",
                                "currency": "NOK",
                            }

                return None

        except Exception as e:
            logger.error(f"Error fetching Norges Bank data: {e}")
            return None
        finally:
            if close_session:
                await session.close()


async def fetch_all_data(
    ticker: str,
    company_name: str = None,
    start_date: str = None,
) -> dict:
    """
    Fetch all data for a ticker in parallel.

    Args:
        ticker: Stock ticker (e.g., EQNR.OL)
        company_name: Company name for news search
        start_date: Start date for incremental fetch

    Returns:
        Dictionary with all fetched data
    """
    # Get stock info synchronously (yfinance doesn't support async well)
    stock_info = YFinanceSource.get_stock_info(ticker)

    if company_name is None:
        company_name = stock_info.get("name", ticker)

    # Get price history
    if start_date:
        prices = YFinanceSource.get_price_history(
            ticker,
            start_date=start_date,
            end_date=datetime.utcnow().strftime("%Y-%m-%d"),
        )
    else:
        prices = YFinanceSource.get_price_history(ticker, period="1y")

    # Fetch news and filings in parallel
    async with aiohttp.ClientSession() as session:
        news_task = GoogleNewsRSS.fetch_news(company_name, session=session)
        filings_task = OsloBorsNewsweb.fetch_filings(ticker, session=session)
        macro_task = NorgesBankAPI.get_key_policy_rate(session=session)

        news, filings, macro = await asyncio.gather(
            news_task, filings_task, macro_task,
            return_exceptions=True,
        )

        # Handle exceptions
        if isinstance(news, Exception):
            logger.error(f"News fetch failed: {news}")
            news = []
        if isinstance(filings, Exception):
            logger.error(f"Filings fetch failed: {filings}")
            filings = []
        if isinstance(macro, Exception):
            logger.error(f"Macro data fetch failed: {macro}")
            macro = None

    return {
        "stock_info": stock_info,
        "prices": prices,
        "news": news,
        "filings": filings,
        "macro": macro,
    }
