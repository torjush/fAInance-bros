"""Analysis Agent - Generates insights using Claude Sonnet."""

import json
import logging
from datetime import datetime, timezone
from typing import Any

import anthropic

from config import Config, PROMPTS
from data.storage import Storage
from utils import strip_code_blocks

logger = logging.getLogger(__name__)


class AnalyzerAgent:
    """
    Agent responsible for analyzing collected data and generating insights.

    Uses Claude Sonnet for deeper analysis and reasoning about:
    - Price trends and technical patterns
    - News sentiment and impact
    - Regulatory developments
    - Risk factors
    """

    def __init__(self, storage: Storage, config: Config):
        self.storage = storage
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    def analyze(
        self,
        ticker: str,
        context: dict[str, Any],
        collected_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform comprehensive analysis of the ticker.

        Args:
            ticker: Stock ticker symbol
            context: Historical context from ContextAgent
            collected_data: Fresh data from CollectorAgent

        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"Starting analysis for {ticker}")

        # Get company info
        stock_info = collected_data.get("stock_info", {})
        company_name = stock_info.get("name", ticker)
        sector = stock_info.get("sector", "Unknown")

        # Prepare price data for analysis
        prices = collected_data.get("prices", [])
        price_stats = self._calculate_price_stats(prices)
        recent_prices = self._format_recent_prices(prices[:30])  # Last 30 days

        # Prepare news data
        news_data = self._format_news_data(collected_data.get("news", []))

        # Get previous insights context
        previous_insights = context.get("previous_insights", {})
        previous_insights_text = self._format_previous_insights(previous_insights)

        # Build the analysis prompt
        prompt = PROMPTS["analyze_data"].format(
            ticker=ticker,
            company_name=company_name,
            previous_insights=previous_insights_text,
            price_data=recent_prices,
            price_stats=json.dumps(price_stats, indent=2),
            news_data=news_data,
        )

        # Call Claude Sonnet for analysis
        try:
            response = self.client.messages.create(
                model=self.config.analysis_model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.content[0].text.strip()
            content = strip_code_blocks(content)
            analysis = json.loads(content)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse analysis JSON: {e}")
            logger.error(f"Response content (first 500 chars): {content[:500]}")
            analysis = self._create_fallback_analysis(price_stats)

        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            analysis = self._create_fallback_analysis(price_stats)

        # Save insights to database
        self._save_insights(ticker, analysis)

        # Add metadata
        analysis["ticker"] = ticker
        analysis["company_name"] = company_name
        analysis["sector"] = sector
        analysis["analysis_timestamp"] = datetime.now(timezone.utc).isoformat()
        analysis["price_stats"] = price_stats
        analysis["recent_prices"] = prices[:30]

        logger.info(f"Analysis complete for {ticker}")
        return analysis

    def _calculate_price_stats(self, prices: list[dict]) -> dict:
        """Calculate comprehensive price statistics."""
        if not prices:
            return {"error": "No price data available"}

        closes = [p["close"] for p in prices if p.get("close")]
        highs = [p["high"] for p in prices if p.get("high")]
        lows = [p["low"] for p in prices if p.get("low")]
        volumes = [p["volume"] for p in prices if p.get("volume")]

        if not closes:
            return {"error": "No closing prices available"}

        current = closes[0]
        period_high = max(highs) if highs else current
        period_low = min(lows) if lows else current
        avg_price = sum(closes) / len(closes)

        # Returns
        returns = {}
        if len(closes) >= 2:
            returns["1d"] = round((closes[0] - closes[1]) / closes[1] * 100, 2)
        if len(closes) >= 6:
            returns["5d"] = round((closes[0] - closes[5]) / closes[5] * 100, 2)
        if len(closes) >= 23:
            returns["1m"] = round((closes[0] - closes[22]) / closes[22] * 100, 2)
        if len(closes) >= 66:
            returns["3m"] = round((closes[0] - closes[65]) / closes[65] * 100, 2)

        # Volatility (20-day rolling std of returns)
        if len(closes) >= 21:
            daily_returns = [(closes[i] - closes[i+1]) / closes[i+1] for i in range(20)]
            mean_return = sum(daily_returns) / len(daily_returns)
            variance = sum((r - mean_return)**2 for r in daily_returns) / len(daily_returns)
            volatility = variance ** 0.5 * (252 ** 0.5)  # Annualized
        else:
            volatility = 0

        # Simple moving averages
        sma = {}
        if len(closes) >= 10:
            sma["10d"] = round(sum(closes[:10]) / 10, 2)
        if len(closes) >= 20:
            sma["20d"] = round(sum(closes[:20]) / 20, 2)
        if len(closes) >= 50:
            sma["50d"] = round(sum(closes[:50]) / 50, 2)

        # Price relative to SMAs
        price_vs_sma = {}
        for period, value in sma.items():
            price_vs_sma[f"vs_{period}"] = round((current - value) / value * 100, 2)

        return {
            "current_price": round(current, 2),
            "period_high": round(period_high, 2),
            "period_low": round(period_low, 2),
            "average_price": round(avg_price, 2),
            "returns": returns,
            "volatility_annualized": round(volatility * 100, 2),
            "sma": sma,
            "price_vs_sma": price_vs_sma,
            "avg_volume": round(sum(volumes) / len(volumes)) if volumes else 0,
            "data_points": len(closes),
        }

    def _format_recent_prices(self, prices: list[dict]) -> str:
        """Format recent prices for the prompt."""
        if not prices:
            return "No price data available."

        lines = ["Date | Open | High | Low | Close | Volume"]
        lines.append("-" * 60)

        for p in prices[:15]:  # Show last 15 days in detail
            lines.append(
                f"{p.get('date', 'N/A')} | "
                f"{p.get('open', 'N/A')} | "
                f"{p.get('high', 'N/A')} | "
                f"{p.get('low', 'N/A')} | "
                f"{p.get('close', 'N/A')} | "
                f"{p.get('volume', 'N/A')}"
            )

        return "\n".join(lines)

    def _format_news_data(self, news: list[dict]) -> str:
        """Format news data for the prompt."""
        if not news:
            return "No recent news available."

        lines = []
        for item in news[:15]:  # Limit to 15 items
            title = item.get("title", "No title")
            source = item.get("source", "Unknown")
            published = item.get("published", "Unknown date")
            extracted = item.get("extracted_data") or item.get("extracted", {})

            lines.append(f"### {title}")
            lines.append(f"Source: {source} | Published: {published}")

            if extracted:
                if isinstance(extracted, dict):
                    sentiment = extracted.get("sentiment", "unknown")
                    relevance = extracted.get("relevance", "unknown")
                    summary = extracted.get("summary", "")
                    lines.append(f"Sentiment: {sentiment} | Relevance: {relevance}")
                    if summary:
                        lines.append(f"Summary: {summary}")

            lines.append("")

        return "\n".join(lines)

    def _format_previous_insights(self, insights: dict) -> str:
        """Format previous insights for context."""
        if not any(insights.values()):
            return "No previous analysis available for this ticker."

        sections = []

        for insight_type, items in insights.items():
            if not items:
                continue

            sections.append(f"### Previous {insight_type.replace('_', ' ').title()}")
            for item in items[:2]:
                timestamp = item.get("timestamp", "Unknown")
                summary = item.get("summary", "")
                if summary:
                    sections.append(f"- [{timestamp}] {summary}")

        return "\n".join(sections) if sections else "No previous analysis available."

    def _save_insights(self, ticker: str, analysis: dict):
        """Save analysis insights to database."""
        # Save individual insight types
        if "price_analysis" in analysis:
            self.storage.save_insight(
                ticker=ticker,
                insight_type="price_analysis",
                content=analysis["price_analysis"],
                summary=analysis["price_analysis"].get("summary", ""),
            )

        if "sentiment_analysis" in analysis:
            self.storage.save_insight(
                ticker=ticker,
                insight_type="sentiment_analysis",
                content=analysis["sentiment_analysis"],
                summary=analysis["sentiment_analysis"].get("summary", ""),
            )

        # Save full analysis
        self.storage.save_insight(
            ticker=ticker,
            insight_type="full_analysis",
            content=analysis,
            summary=analysis.get("outlook", "Analysis completed"),
        )

    def _create_fallback_analysis(self, price_stats: dict) -> dict:
        """Create a basic fallback analysis if LLM fails."""
        return {
            "price_analysis": {
                "trend": "neutral",
                "trend_strength": "weak",
                "support_levels": [],
                "resistance_levels": [],
                "volatility": "unknown",
                "summary": "Unable to complete full price analysis.",
            },
            "sentiment_analysis": {
                "overall_sentiment": "neutral",
                "confidence": 0.0,
                "key_themes": [],
                "summary": "Unable to complete sentiment analysis.",
            },
            "risk_factors": [],
            "key_observations": ["Analysis incomplete - please review raw data"],
            "outlook": "Unable to generate outlook due to analysis error.",
        }
