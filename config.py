"""Configuration settings and prompts for the finance agents."""

import os
from dataclasses import dataclass, field


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Required environment variable {name!r} is not set")
    return value


@dataclass
class Config:
    """Application configuration."""

    # API Keys
    anthropic_api_key: str = field(default_factory=lambda: _require_env("ANTHROPIC_API_KEY"))

    # Model settings
    extraction_model: str = "claude-haiku-4-5-20251001"  # Cheaper for extraction
    analysis_model: str = "claude-sonnet-4-5-20250929"   # Better reasoning

    # Database
    db_path: str = field(default_factory=lambda: _require_env("DB_PATH"))

    # Data fetching
    price_history_days: int = 365  # How far back to fetch prices initially
    news_max_age_days: int = 30    # How far back to fetch news

    # Rate limiting
    max_concurrent_requests: int = 5

    # Report output
    reports_dir: str = field(default_factory=lambda: _require_env("REPORTS_DIR"))


# Prompts - kept separate for easy modification
PROMPTS = {
    "extract_news": """Extract structured information from the following news article about {ticker}.

Article:
{article_text}

Return a JSON object with:
{{
    "headline": "article headline",
    "summary": "1-2 sentence summary",
    "sentiment": "positive" | "negative" | "neutral",
    "relevance": "high" | "medium" | "low",
    "topics": ["list", "of", "key", "topics"],
    "mentioned_companies": ["list of company names/tickers mentioned"],
    "key_figures": {{"any numerical data like revenue, prices, percentages"}}
}}

Only return valid JSON, no other text.""",

    "extract_global_news": """You are a macro financial analyst. Based on the following global financial news headlines, extract the key market themes, overall sentiment, and sector risk implications for Oslo Børs investors.

Headlines:
{headlines}

Return a JSON object:
{{
    "market_sentiment": "positive" | "negative" | "neutral",
    "key_themes": ["list of key market themes, e.g. 'Fed rate pause', 'Oil supply cut', 'China slowdown'"],
    "macro_events": ["list of significant macro events or decisions mentioned"],
    "summary": "2-3 sentence overview of the current global market environment",
    "safer_sectors": ["list of 2-4 Oslo Børs sectors that are relatively safer or favoured given the current macro environment, e.g. 'Energy', 'Financials', 'Consumer Staples'"],
    "avoid_sectors": ["list of 2-4 Oslo Børs sectors to underweight or avoid given current macro risks, e.g. 'Real Estate', 'Consumer Discretionary', 'Technology'"]
}}

Only return valid JSON, no other text.""",

    "recommend_macro_stocks": """You are a senior equity analyst specialising in Oslo Børs (Oslo Stock Exchange). Based on the current macro environment described below, recommend 3 to 5 specific Oslo Børs stocks that are well-positioned over the next 1-3 months.

## Current Macro Environment
Market Sentiment: {market_sentiment}
Summary: {macro_summary}
Key Themes: {key_themes}
Notable Macro Events: {macro_events}

## Sector Guidance
Sectors currently favoured by the macro backdrop: {safer_sectors}
Sectors to avoid or underweight: {avoid_sectors}

## Portfolio Exclusion
The following tickers are already in the investor's portfolio and must NOT be recommended:
{portfolio_tickers}

## Your Task
Recommend 3-5 Oslo Børs stocks that:
1. Operate primarily in one of the favoured sectors above
2. Are NOT listed in the portfolio exclusion list above
3. Have a credible near-term catalyst or macro tailwind

Use only your training knowledge of Oslo Børs companies — do not invent tickers.
All tickers must be real Oslo Børs tickers ending in .OL (e.g. EQNR.OL, DNB.OL, MOWI.OL).

Return a JSON object:
{{
    "stock_ideas": [
        {{
            "ticker": "XXXX.OL",
            "company": "Full Company Name",
            "sector": "Sector name",
            "rationale": "1-2 sentences on why this stock fits the macro outlook",
            "risk_note": "1 sentence on the main risk to this idea"
        }}
    ]
}}

Only return valid JSON, no other text.""",

    "analyze_data": """You are a fundamental financial analyst specializing in Oslo Stock Exchange companies. Your investment horizon is approximately one month.

Analyze the following data for {ticker} ({company_name}):

## Previous Context
{previous_insights}

## Recent Price Data (last 30 days)
{price_data}

## Price Statistics
{price_stats}

## Recent News
{news_data}

## Global Market Context
{global_context}

## Sector & Geographic Context
{targeted_context}

Provide an analysis covering the following, in order of importance:

1. **News & Catalysts**: What are the most significant recent news items and upcoming catalysts over the next month? Focus on company-specific events (earnings, contracts, management changes, regulatory decisions) and sector tailwinds/headwinds.

2. **Fundamental Outlook (1-month horizon)**: Based on news and macro context, what is the likely direction for this stock over the next ~4 weeks? Consider: sentiment momentum, sector trends, macro environment.

3. **Global & Sector Context**: How does the current macro and sector environment affect the 1-month outlook for this company?

4. **Risk Factors**: What could derail the 1-month thesis?

5. **Price Context** (secondary): Use price data only as supporting context — note if the stock is up/down significantly recently, but do not base the outlook on technical patterns or support/resistance levels.

6. **Key Observations**: 3-5 bullet points of the most important takeaways, focused on the 1-month outlook.

Note: Technical indicators (support/resistance, moving averages) are computed separately and are low-weight inputs. Ground your analysis primarily in news, fundamentals, and macro context.

Return a JSON object with your analysis:
{{
    "price_analysis": {{
        "trend": "bullish" | "bearish" | "neutral",
        "trend_strength": "strong" | "moderate" | "weak",
        "volatility": "high" | "medium" | "low",
        "summary": "1-2 sentence summary of recent price action as context only"
    }},
    "sentiment_analysis": {{
        "overall_sentiment": "positive" | "negative" | "neutral",
        "confidence": 0.0-1.0,
        "key_themes": ["list of themes"],
        "summary": "2-3 sentence summary focused on news-driven sentiment"
    }},
    "global_context_impact": "1-2 sentences on how the global macro environment affects the 1-month outlook for this stock",
    "risk_factors": [
        {{"risk": "description", "severity": "high" | "medium" | "low"}}
    ],
    "key_observations": ["bullet point 1", "bullet point 2", ...],
    "outlook": "1-2 sentence forward-looking statement for the next ~1 month, based primarily on news and macro — not technicals"
}}

Only return valid JSON, no other text.""",

    "extract_company_profile": """You are a financial analyst. Given the following company information, extract the key sectors/industries and geographies where the company operates, then generate targeted news search queries.

Company: {company_name}
Sector: {sector}
Industry: {industry}
Country: {country}
Business Summary: {business_summary}

Return a JSON object:
{{
    "sectors": ["list of up to 3 key sectors/industries, e.g. 'oil & gas', 'renewable energy', 'offshore drilling'"],
    "geographies": ["list of up to 5 key regions/countries where company operates, e.g. 'Norway', 'North Sea', 'United States', 'Brazil'"],
    "search_queries": ["list of up to 6 targeted news search queries combining sectors and geographies, e.g. 'oil gas North Sea', 'offshore drilling Norway', 'energy sector Brazil']"
}}

Be specific and practical. Queries should be 2-4 words optimized for Google News search.
Only return valid JSON, no other text.""",

    "extract_targeted_news": """You are a financial analyst. Based on the following news headlines fetched for targeted sector/geography queries, extract key themes and insights.

Headlines (grouped by query):
{headlines}

Return a JSON object:
{{
    "sector_themes": ["list of key themes related to the company's sectors, e.g. 'OPEC production cuts', 'offshore rig demand up'"],
    "geo_themes": ["list of key themes related to the company's geographies, e.g. 'Norway energy tax increase', 'Brazil oil licensing round'"],
    "summary": "2-3 sentence summary of the most important sector and geographic developments"
}}

Only return valid JSON, no other text.""",

    "generate_portfolio_report": """You are a fundamental financial analyst writing a consolidated portfolio analysis report. The investment horizon for all recommendations is approximately one month.

Today's date: {date}

## Global Market Context
{global_context}

## Portfolio Holdings
{stock_data}

Generate a professional markdown portfolio report with EXACTLY this structure:

# Portfolio Analysis — {date}

## 1. Market Overview
- Current global macro environment and what it means for Oslo Børs stocks over the next month
- Which sectors dominate this portfolio and their 1-month macro sensitivity
- Portfolio positioning relative to the current macro backdrop

## 2. Individual Stock Analysis

For EACH stock in the portfolio, write a subsection:

### {ticker_placeholder} — {company_placeholder}
**Sector:** {sector_placeholder}
**Current Price:** {price_placeholder}

[2-3 paragraphs of analysis covering: key news and upcoming catalysts over the next month, sector/macro tailwinds or headwinds, and what this means for the 1-month outlook. Mention recent price moves only as brief supporting context — do not anchor the analysis on technical patterns]

**Recommendation: Buy / Hold / Sell** *(1-month horizon)*

**Key Risks:** [2-3 bullet points]

## 3. Portfolio Summary

| Ticker | Company | Sector | Current Price | 1-Month Outlook | Recommendation | Rationale |
|--------|---------|--------|---------------|-----------------|----------------|-----------|
[one row per stock]

## Disclaimer

*This report is AI-generated for informational purposes only. It is not financial advice. Always conduct your own research and consult with a qualified financial advisor before making investment decisions.*

---

Rules:
- Investment horizon is ~1 month. Base recommendations on news, catalysts, and macro context — not on technical chart patterns or support/resistance levels
- Be direct and decisive — no hedging phrases like "may", "could potentially", "it remains to be seen"
- Base all recommendations strictly on the data provided — no external assumptions
- If multiple stocks are in the same sector, explicitly note concentration risk in the Market Overview
- 1-Month Outlook values in the table must be one of: Bullish / Bearish / Neutral
- Recommendation must be one of: Buy / Hold / Sell

Return ONLY the markdown content, no code blocks or backticks around it.""",

    "generate_report": """You are a fundamental financial analyst writing a report for {ticker} ({company_name}). The investment horizon is approximately one month.

Using the following analysis data, generate a professional markdown report.

## Company Info
- Ticker: {ticker}
- Name: {company_name}
- Sector: {sector}

## Analysis Date
{analysis_date}

## Price Analysis
{price_analysis}

## Sentiment Analysis
{sentiment_analysis}

## Global Market Context
{global_context}

## Sector & Geographic Context
{targeted_context}

## Risk Factors
{risk_factors}

## Key Observations
{key_observations}

## Outlook
{outlook}

## Recent Price Data
{recent_prices}

Generate a well-formatted markdown report that:
1. Starts with a header and executive summary focused on the 1-month outlook
2. Leads with news, catalysts, and macro context — these are the primary drivers
3. Includes a "Global Market Context" section on macro relevance to this stock — only if relevant
4. Includes a "Sector & Geographic Context" section on sector/geo developments — only if relevant
5. Includes a brief "Price Context" section — mention recent price moves as supporting context only, not as the basis for the outlook. Do not dwell on technical patterns or support/resistance levels
6. Presents recent price data in a readable table
7. Ends with a clear Buy / Hold / Sell recommendation for a ~1 month horizon, with a one-sentence rationale
8. Ends with a disclaimer about this being AI-generated analysis

Make it professional, direct, and actionable. No hedging language.

Return ONLY the markdown content, no code blocks or backticks around it.""",
}


def get_config() -> Config:
    """Get configuration instance."""
    return Config()
