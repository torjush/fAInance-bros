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

    "analyze_data": """You are a financial analyst specializing in Oslo Stock Exchange companies.

Analyze the following data for {ticker} ({company_name}):

## Previous Context
{previous_insights}

## Recent Price Data (last 30 days)
{price_data}

## Price Statistics
{price_stats}

## Recent News
{news_data}

Provide a comprehensive analysis covering:

1. **Price Action Analysis**: Analyze recent price movements, trends, support/resistance levels, and volume patterns.

2. **News Sentiment**: Summarize the overall news sentiment and highlight the most significant news items.

3. **Risk Factors**: Identify any emerging risks based on the data.

4. **Key Observations**: 3-5 bullet points of the most important takeaways.

Note: Support and resistance levels are calculated algorithmically and will be added separately. Focus your analysis on trend, volatility, and interpretation.

Return a JSON object with your analysis:
{{
    "price_analysis": {{
        "trend": "bullish" | "bearish" | "neutral",
        "trend_strength": "strong" | "moderate" | "weak",
        "volatility": "high" | "medium" | "low",
        "summary": "2-3 sentence summary"
    }},
    "sentiment_analysis": {{
        "overall_sentiment": "positive" | "negative" | "neutral",
        "confidence": 0.0-1.0,
        "key_themes": ["list of themes"],
        "summary": "2-3 sentence summary"
    }},
    "risk_factors": [
        {{"risk": "description", "severity": "high" | "medium" | "low"}}
    ],
    "key_observations": ["bullet point 1", "bullet point 2", ...],
    "outlook": "brief 1-2 sentence forward-looking statement"
}}

Only return valid JSON, no other text.""",

    "generate_report": """You are a financial report writer creating an analysis report for {ticker} ({company_name}).

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

## Risk Factors
{risk_factors}

## Key Observations
{key_observations}

## Outlook
{outlook}

## Recent Price Data
{recent_prices}

Generate a well-formatted markdown report that:
1. Starts with a header and executive summary
2. Includes all analysis sections with clear headings
3. Presents price data in a readable table format
4. Ends with a disclaimer about this being AI-generated analysis

Make it professional, clear, and actionable. Use appropriate markdown formatting including headers, bullet points, tables, and emphasis where appropriate.

Return ONLY the markdown content, no code blocks or backticks around it.""",
}


def get_config() -> Config:
    """Get configuration instance."""
    return Config()
