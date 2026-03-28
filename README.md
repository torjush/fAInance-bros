# fAInance-bros

AI-powered stock analysis tool for companies listed on the Oslo Stock Exchange (Oslo Børs). Uses LangGraph to orchestrate multiple AI agents that collect data, analyze trends, and generate comprehensive reports with charts.

## Features

- **Multi-agent architecture** using LangGraph for orchestrated analysis
- **Portfolio analysis** - analyze all holdings at once with a single unified report (Market Overview, per-stock BHS, summary table, macro sector outlook & new stock ideas)
- **Incremental analysis** - only fetches new data since last run
- **Multiple data sources**:
  - Stock prices via yfinance
  - Company-specific news via Google News RSS
  - Sector/geography-targeted news based on company profile
  - Global macro news (markets, central banks, commodities) with sector risk assessment
  - Macro data (key policy rate) from Norges Bank
- **Persistent storage** with SQLite for data reuse across runs
- **Cost-optimized LLM usage**:
  - Claude Haiku for data extraction (cheap, fast)
  - Claude Sonnet for analysis and report generation (better reasoning)
- **Visual technical analysis** with price charts, moving averages, and support/resistance levels
- **PDF reports** alongside markdown

## Architecture

### Single-Stock Mode

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
                        │ Global News     │
                        │ (macro context) │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ Company Profile │
                        │ (sectors, geos) │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ Targeted News   │
                        │ (sector × geo)  │
                        └────────┬────────┘
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

### Portfolio Mode

```
┌──────────────────────┐
│  CLI Entry           │
│  (analyze.py -p)     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Portfolio Analyzer   │
│ (portfolio_analyzer) │
└──────┬───────────────┘
       │
       ├─ Fetch Global News (once, shared)
       │      └─ includes safer_sectors / avoid_sectors
       │
       ├─ MacroAdvisorAgent ── sector risk + stock ideas
       │
       ├─ ThreadPoolExecutor ──────────────────────────────┐
       │   ├─ StockAnalyzerWorkflow(EQNR.OL, no report)   │
       │   ├─ StockAnalyzerWorkflow(DNB.OL, no report)     │
       │   └─ StockAnalyzerWorkflow(MOWI.OL, no report)    │
       │                                                   │
       └─────────────────────────────────── collect ◀──────┘
           │
           ▼
┌──────────────────────┐
│ Portfolio Reporter   │
│ (Claude Sonnet)      │
│ - Market Overview    │
│ - Per-stock BHS      │
│ - Summary table      │
│ - Sector outlook     │
│ - New stock ideas    │
└──────────────────────┘
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

Analyze a single stock:
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

Analyze a whole portfolio:
```bash
# Create a portfolio file (one ticker per line, # for comments)
cat > portfolio.txt << EOF
# My Oslo Børs portfolio
EQNR.OL
DNB
MOWI.OL
EOF

# Run portfolio analysis
uv run python analyze.py --portfolio portfolio.txt --verbose

# Short flag
uv run python analyze.py -p portfolio.txt
```

Reports are saved to `./data/reports/` as both `.md` and `.pdf` files. Portfolio reports are named `portfolio_YYYYMMDD_HHMMSS.md`.

### Scheduling with Cron

Add to crontab for automated daily analysis:
```bash
# Analyze single stock every weekday at 18:00 (after market close)
0 18 * * 1-5 cd /path/to/fAInance-bros && uv run python analyze.py EQNR.OL

# Analyze full portfolio every weekday at 18:00
0 18 * * 1-5 cd /path/to/fAInance-bros && uv run python analyze.py --portfolio portfolio.txt
```

## Project Structure

```
fAInance-bros/
├── analyze.py              # CLI entry point and LangGraph workflow (single-stock)
├── portfolio_analyzer.py   # Portfolio orchestrator (parallel per-stock + unified report)
├── config.py               # Configuration and prompts
├── visualization.py        # Price chart generation
├── utils.py                # Utility functions
├── portfolio.txt           # Your portfolio (not committed — add tickers here)
├── agents/
│   ├── context.py              # Historical context retrieval
│   ├── collector.py            # Data collection from external sources
│   ├── global_news.py          # Global macro news (markets, rates, commodities) + sector risk
│   ├── macro_advisor.py        # Macro-driven Oslo Børs stock ideas (portfolio mode only)
│   ├── company_profile.py      # Extract sectors/geographies from stock info
│   ├── targeted_news.py        # Sector/geography-targeted news fetching
│   ├── analyzer.py             # AI-powered analysis
│   ├── reporter.py             # Single-stock report generation (MD + PDF)
│   └── portfolio_reporter.py   # Portfolio report generation (MD + PDF)
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

## Configuration

Edit `config.py` to modify:

- **Model selection**: Change Haiku/Sonnet models
- **Prompts**: Customize extraction and analysis prompts
- **Data fetching**: Adjust history depth, rate limits

Environment variables in `.env`:

- `ANTHROPIC_API_KEY`: Your Anthropic API key (required)
- `DB_PATH`: Path to SQLite database (required)
- `REPORTS_DIR`: Directory for reports (required)

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
