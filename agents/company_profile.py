"""Company Profile Agent - Extracts sectors, geographies, and targeted news queries."""

import json
import logging
from typing import Any

import anthropic

from config import Config, PROMPTS
from data.storage import Storage
from utils import strip_code_blocks

logger = logging.getLogger(__name__)


class CompanyProfileAgent:
    """
    Agent that extracts a structured company profile from yfinance stock info.

    Uses Claude Haiku to identify:
    - Key sectors/industries the company operates in
    - Key geographies/regions
    - Targeted news search queries (sector × geography combos)

    On incremental runs, reuses the stored profile to save API cost.
    """

    def __init__(self, storage: Storage, config: Config):
        self.storage = storage
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    def profile(
        self,
        ticker: str,
        stock_info: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Build a company profile from stock info.

        Reuses stored profile if available (saves Haiku call on incremental runs).
        Always extracts fresh profile on first run or when no profile exists.

        Returns:
            {sectors, geographies, search_queries}
        """
        logger.info(f"[Company Profile] Profiling {ticker}")

        # Check for existing profile in context (loaded by ContextAgent)
        existing = context.get("company_profile")
        if existing and existing.get("sectors") and existing.get("geographies"):
            logger.info(f"[Company Profile] Reusing stored profile for {ticker}")
            return existing

        # Extract fresh profile
        profile = self._extract_profile(ticker, stock_info)

        # Persist to DB
        summary = self._build_summary(profile)
        self.storage.save_insight(
            ticker=ticker,
            insight_type="company_profile",
            content=profile,
            summary=summary,
        )
        logger.info(f"[Company Profile] Saved profile for {ticker}: {summary}")

        return profile

    def _extract_profile(
        self, ticker: str, stock_info: dict[str, Any]
    ) -> dict[str, Any]:
        """Call Claude Haiku to extract sectors, geographies, and queries."""
        company_name = stock_info.get("name", ticker)
        sector = stock_info.get("sector", "Unknown")
        industry = stock_info.get("industry", "Unknown")
        country = stock_info.get("country", "Unknown")
        business_summary = stock_info.get("longBusinessSummary", "")

        prompt = PROMPTS["extract_company_profile"].format(
            company_name=company_name,
            sector=sector,
            industry=industry,
            country=country,
            business_summary=business_summary[:2000],  # Cap to avoid token bloat
        )

        try:
            response = self.client.messages.create(
                model=self.config.extraction_model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text.strip()
            content = strip_code_blocks(content)
            return json.loads(content)

        except json.JSONDecodeError as e:
            logger.warning(f"[Company Profile] Failed to parse JSON: {e}")
            return self._fallback_profile(stock_info)
        except Exception as e:
            logger.error(f"[Company Profile] Extraction error: {e}")
            return self._fallback_profile(stock_info)

    def _fallback_profile(self, stock_info: dict) -> dict:
        """Basic fallback when LLM extraction fails."""
        sector = stock_info.get("sector", "Unknown")
        country = stock_info.get("country", "Unknown")
        name = stock_info.get("name", "")
        queries = []
        if sector != "Unknown":
            queries.append(sector)
        if country != "Unknown" and sector != "Unknown":
            queries.append(f"{sector} {country}")
        return {
            "sectors": [sector] if sector != "Unknown" else [],
            "geographies": [country] if country != "Unknown" else [],
            "search_queries": queries,
        }

    def _build_summary(self, profile: dict) -> str:
        sectors = ", ".join(profile.get("sectors", []))
        geos = ", ".join(profile.get("geographies", []))
        return f"{sectors or 'Unknown sector'} — {geos or 'Unknown geography'}"
