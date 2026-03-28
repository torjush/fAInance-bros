"""Macro Advisor Agent - Recommends Oslo Børs stocks based on macro sector outlook."""

import json
import logging
from typing import Any

import anthropic

from config import Config, PROMPTS
from utils import strip_code_blocks

logger = logging.getLogger(__name__)


class MacroAdvisorAgent:
    """
    Agent that recommends specific Oslo Børs stocks in macro-favoured sectors.

    Takes the global_news output (including safer_sectors / avoid_sectors) plus
    the list of tickers already in the portfolio, and uses Claude Sonnet to
    suggest 3-5 new stocks worth considering.

    No external data is fetched — recommendations rely solely on Claude's
    training knowledge of Oslo Børs combined with the live macro context.
    """

    def __init__(self, config: Config):
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    def recommend(
        self,
        global_news: dict[str, Any],
        portfolio_tickers: list[str],
    ) -> dict[str, Any]:
        """
        Generate stock ideas for macro-safe sectors.

        Args:
            global_news: Output from GlobalNewsAgent.fetch() — must include
                         safer_sectors and avoid_sectors keys.
            portfolio_tickers: Current holdings (e.g. ['EQNR.OL', 'DNB.OL']).
                               Suggestions will exclude these.

        Returns:
            Dict with key "stock_ideas": list of idea dicts, or empty list on failure.
        """
        safer_sectors = global_news.get("safer_sectors", [])

        if not safer_sectors:
            logger.warning("MacroAdvisorAgent: no safer_sectors in global_news — skipping")
            return {"stock_ideas": []}

        prompt = PROMPTS["recommend_macro_stocks"].format(
            macro_summary=global_news.get("summary", ""),
            key_themes=", ".join(global_news.get("key_themes", [])),
            macro_events=", ".join(global_news.get("macro_events", [])),
            market_sentiment=global_news.get("market_sentiment", "neutral"),
            safer_sectors=", ".join(safer_sectors),
            avoid_sectors=", ".join(global_news.get("avoid_sectors", [])) or "none specified",
            portfolio_tickers=", ".join(portfolio_tickers) if portfolio_tickers else "none",
        )

        try:
            response = self.client.messages.create(
                model=self.config.analysis_model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text.strip()
            content = strip_code_blocks(content)
            result = json.loads(content)
            if "stock_ideas" not in result:
                result = {"stock_ideas": []}
        except Exception as e:
            logger.error(f"MacroAdvisorAgent recommendation failed: {e}")
            result = {"stock_ideas": []}

        logger.info(f"MacroAdvisorAgent: suggested {len(result.get('stock_ideas', []))} stocks")
        return result
