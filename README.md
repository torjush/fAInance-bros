# Finance Agents

AI-powered stock analysis tool for companies listed on the Oslo Stock Exchange (Oslo Børs). Uses LangGraph to orchestrate multiple AI agents that collect data, analyze trends, and generate comprehensive markdown reports.

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
- **Containerized** with Docker for easy deployment and scheduling

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
                        │ Report Generator│
                        │ (Claude Sonnet) │
                        └─────────────────┘
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Anthropic API key

### Setup

1. Clone or copy this directory
2. Create a `.env` file with your API key:
   ```bash
   echo "ANTHROPIC_API_KEY=your-key-here" > .env
   ```

3. Build the Docker image:
   ```bash
   docker-compose build
   ```

### Usage

Analyze a stock:
```bash
# Analyze Equinor
docker-compose run analyzer EQNR.OL

# Analyze DNB (auto-adds .OL suffix)
docker-compose run analyzer DNB

# With verbose logging
docker-compose run analyzer-verbose EQNR.OL

# Save output to file
docker-compose run analyzer EQNR.OL > reports/eqnr_report.md
```

### Scheduling with Cron

Add to crontab for automated daily analysis:
```bash
# Run every weekday at 18:00 (after market close)
0 18 * * 1-5 cd /path/to/finance-agents && docker-compose run analyzer EQNR.OL > reports/eqnr_$(date +\%Y\%m\%d).md 2>&1
```

## Running Without Docker

You can also run directly with Python:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export ANTHROPIC_API_KEY="your-key-here"
export DB_PATH="./data/finance_agents.db"
export REPORTS_DIR="./data/reports"

# Run analysis
python analyze.py EQNR.OL

# With options
python analyze.py EQNR.OL --verbose --output report.md
```

## Project Structure

```
finance-agents/
├── analyze.py          # CLI entry point and LangGraph workflow
├── config.py           # Configuration and prompts
├── agents/
│   ├── context.py      # Historical context retrieval
│   ├── collector.py    # Data collection from external sources
│   ├── analyzer.py     # AI-powered analysis
│   └── reporter.py     # Report generation
├── data/
│   ├── sources.py      # API integrations (yfinance, RSS feeds)
│   └── storage.py      # SQLite database operations
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
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
