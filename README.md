# Finance Agents

AI-powered stock analysis tool for companies listed on the Oslo Stock Exchange (Oslo Børs). Uses LangGraph to orchestrate multiple AI agents that collect data, analyze trends, and generate comprehensive reports with charts.

## Features

- **Multi-agent architecture** using LangGraph for orchestrated analysis
- **Incremental analysis** - only fetches new data since last run
- **Multiple data sources**:
  - Stock prices via yfinance
  - News via Google News RSS
  - Regulatory filings via Oslo Børs Newsweb RSS
  - Macro data (key policy rate) from Norges Bank
- **Persistent storage** with SQLite for data reuse across runs
- **Cost-optimized LLM usage**:
  - Claude Haiku for data extraction (cheap, fast)
  - Claude Sonnet for analysis and report generation (better reasoning)
- **Visual technical analysis** with price charts, moving averages, and support/resistance levels
- **PDF reports** alongside markdown

## Architecture

```
┌─────────────────┐
│  CLI Entry      │
│  (analyze.py)   │
└────────┬────────┘
         │
    LangGraph Workflow
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ Context Agent   │────▶│ Data Collector  │
│ (SQLite lookup) │     │ (yfinance, RSS) │
└─────────────────┘     └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ Analyzer Agent  │
                        │ (Claude Sonnet) │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  Visualization  │
                        │ (Price Charts)  │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ Report Generator│
                        │  (MD + PDF)     │
                        └─────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Anthropic API key

### Setup

1. Clone this repository

2. Install uv (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Create a `.env` file with your API key:
   ```bash
   cp .env.example .env
   # Edit .env and add your ANTHROPIC_API_KEY
   ```

4. Install dependencies:
   ```bash
   uv sync
   ```

### Usage

Analyze a stock:
```bash
# Analyze Equinor
uv run python analyze.py EQNR.OL

# Analyze DNB (auto-adds .OL suffix)
uv run python analyze.py DNB

# With verbose logging
uv run python analyze.py TEL.OL --verbose

# Save to custom output file
uv run python analyze.py EQNR.OL --output my_report.md
```

Reports are saved to `./data/reports/` as both `.md` and `.pdf` files.

### Scheduling with Cron

Add to crontab for automated daily analysis:
```bash
# Run every weekday at 18:00 (after market close)
0 18 * * 1-5 cd /path/to/finance-agents && uv run python analyze.py EQNR.OL
```

## Project Structure

```
finance-agents/
├── analyze.py          # CLI entry point and LangGraph workflow
├── config.py           # Configuration and prompts
├── visualization.py    # Price chart generation
├── utils.py            # Utility functions
├── agents/
│   ├── context.py      # Historical context retrieval
│   ├── collector.py    # Data collection from external sources
│   ├── analyzer.py     # AI-powered analysis
│   └── reporter.py     # Report generation (MD + PDF)
├── data/
│   ├── sources.py      # API integrations (yfinance, RSS feeds)
│   └── storage.py      # SQLite database operations
├── pyproject.toml      # Project dependencies
├── uv.lock             # Locked dependency versions
└── .env                # Environment variables (not committed)
```

## Database Schema

The SQLite database stores:

- **companies**: Basic company info and last analysis timestamp
- **price_history**: Historical OHLCV data
- **insights**: Structured analysis results (JSON)
- **reports**: Generated markdown reports
- **news_cache**: Cached news with extracted data
- **filings_cache**: Cached regulatory filings

## Configuration

Edit `config.py` to modify:

- **Model selection**: Change Haiku/Sonnet models
- **Prompts**: Customize extraction and analysis prompts
- **Data fetching**: Adjust history depth, rate limits

Environment variables in `.env`:

- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `DB_PATH`: Path to SQLite database (default: `./data/finance_agents.db`)
- `REPORTS_DIR`: Directory for reports (default: `./data/reports`)

## Cost Optimization

The system is designed to minimize API costs:

1. **Incremental fetching**: Only fetches data newer than `last_analyzed`
2. **Caching**: News and filings are cached to avoid re-extraction
3. **Model tiering**: Uses cheaper Haiku for extraction, Sonnet only for analysis
4. **Structured storage**: Insights stored as JSON for reuse

Typical costs per analysis run:
- First run (full history): ~$0.05-0.15
- Subsequent runs (incremental): ~$0.01-0.05

## Supported Tickers

Any Oslo Stock Exchange ticker should work. Common examples:
- `EQNR.OL` - Equinor
- `DNB.OL` - DNB Bank
- `MOWI.OL` - Mowi
- `TEL.OL` - Telenor
- `ORK.OL` - Orkla
- `YAR.OL` - Yara International

## Troubleshooting

**"No price data found"**: The ticker might be incorrect or delisted. Verify on Yahoo Finance.

**RSS feed errors**: Google News and Newsweb occasionally change their feeds. Check the logs for specific errors.

**API errors**: Ensure your `ANTHROPIC_API_KEY` is valid and has sufficient credits.

**Database locked**: Only run one analysis at a time per database file.

## License

MIT License - use freely for personal or commercial purposes.

## Disclaimer

This tool generates AI-powered analysis for informational purposes only. It is not financial advice. Always conduct your own research and consult with qualified financial advisors before making investment decisions.
