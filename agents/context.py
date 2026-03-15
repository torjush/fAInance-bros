"""Context Agent - Retrieves historical context from the database."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from data.storage import Storage

logger = logging.getLogger(__name__)


class ContextAgent:
    """
    Agent responsible for gathering historical context from the database.

    This agent queries SQLite for:
    - Previous insights about the ticker
    - Historical price data
    - Company information
    - Last analysis timestamp (for incremental updates)
    """

    def __init__(self, storage: Storage):
        self.storage = storage

    def get_context(self, ticker: str) -> dict[str, Any]:
        """
        Retrieve all relevant context for a ticker.

        Args:
            ticker: Stock ticker symbol (e.g., EQNR.OL)

        Returns:
            Dictionary containing historical context
        """
        logger.info(f"Gathering context for {ticker}")

        # Get company info
        company = self.storage.get_company(ticker)

        # Get last analysis timestamp
        last_analyzed = self.storage.get_last_analyzed(ticker)

        # Get previous insights (last few of each type)
        previous_insights = {
            "price_analysis": self.storage.get_insights(ticker, "price_analysis", limit=3),
            "sentiment_analysis": self.storage.get_insights(ticker, "sentiment_analysis", limit=3),
            "full_analysis": self.storage.get_insights(ticker, "full_analysis", limit=2),
        }

        # Load stored company profile and targeted news themes for reuse on incremental runs
        company_profile_rows = self.storage.get_insights(ticker, "company_profile", limit=1)
        company_profile = company_profile_rows[0]["content"] if company_profile_rows else None

        targeted_news_rows = self.storage.get_insights(ticker, "targeted_news", limit=1)
        targeted_news_themes = targeted_news_rows[0]["content"] if targeted_news_rows else None

        # Get price history (last 90 days for trend analysis)
        ninety_days_ago = (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%d")
        price_history = self.storage.get_prices(ticker, start_date=ninety_days_ago)

        # Get cached news
        cached_news = self.storage.get_cached_news(ticker, since=ninety_days_ago)

        # Get latest report
        latest_report = self.storage.get_latest_report(ticker)

        context = {
            "ticker": ticker,
            "company": company,
            "last_analyzed": last_analyzed.isoformat() if last_analyzed else None,
            "previous_insights": previous_insights,
            "price_history": price_history,
            "cached_news": cached_news,
            "latest_report": latest_report,
            "is_new_ticker": company is None,
            "company_profile": company_profile,
            "targeted_news_themes": targeted_news_themes,
        }

        # Calculate some summary stats for the context
        if price_history:
            prices = [p["close"] for p in price_history if p.get("close")]
            if prices:
                context["price_summary"] = {
                    "current": prices[0],  # Most recent (list is DESC)
                    "high_90d": max(prices),
                    "low_90d": min(prices),
                    "avg_90d": sum(prices) / len(prices),
                    "data_points": len(prices),
                }

        logger.info(
            f"Context gathered for {ticker}: "
            f"{'new ticker' if context['is_new_ticker'] else 'existing'}, "
            f"{len(price_history)} price records, "
            f"{len(cached_news)} cached news"
        )

        return context

    def get_incremental_start_date(self, ticker: str) -> str | None:
        """
        Determine the start date for incremental data fetching.

        Returns the day after the last analysis, or None if this is a new ticker.
        """
        last_analyzed = self.storage.get_last_analyzed(ticker)

        if last_analyzed:
            # Start from the day after last analysis
            start = last_analyzed + timedelta(days=1)
            return start.strftime("%Y-%m-%d")

        return None

    def format_previous_insights(self, insights: dict) -> str:
        """
        Format previous insights into a readable string for the analysis prompt.
        """
        if not any(insights.values()):
            return "No previous analysis available for this ticker."

        sections = []

        # Full analysis summaries
        if insights.get("full_analysis"):
            sections.append("### Previous Full Analyses")
            for insight in insights["full_analysis"]:
                timestamp = insight.get("timestamp", "Unknown date")
                summary = insight.get("summary", "No summary")
                sections.append(f"- **{timestamp}**: {summary}")

        # Price analysis
        if insights.get("price_analysis"):
            sections.append("\n### Previous Price Analyses")
            for insight in insights["price_analysis"][:2]:
                content = insight.get("content", {})
                if isinstance(content, dict):
                    trend = content.get("trend", "unknown")
                    summary = content.get("summary", "No details")
                    sections.append(f"- Trend: {trend} - {summary}")

        # Sentiment analysis
        if insights.get("sentiment_analysis"):
            sections.append("\n### Previous Sentiment Analyses")
            for insight in insights["sentiment_analysis"][:2]:
                content = insight.get("content", {})
                if isinstance(content, dict):
                    sentiment = content.get("overall_sentiment", "unknown")
                    themes = content.get("key_themes", [])
                    sections.append(f"- Sentiment: {sentiment}, Themes: {', '.join(themes[:3])}")

        return "\n".join(sections) if sections else "No previous analysis available."
