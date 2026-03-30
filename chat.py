#!/usr/bin/env python3
"""
Interactive chat for follow-up questions about stock analysis reports.

Usage:
    python chat.py TICKER [OPTIONS]

Examples:
    python chat.py EQNR.OL
    python chat.py DNB --verbose
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import logging
import sys
from datetime import datetime, timezone, timedelta

import anthropic

from config import get_config, PROMPTS
from data.storage import Storage
from analyze import validate_ticker, StockAnalyzerWorkflow


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_context(ticker: str, storage: Storage) -> dict | None:
    """Load all available analysis data for a ticker from the database."""
    report = storage.get_latest_report(ticker)
    if not report:
        return None

    company = storage.get_company(ticker)
    insights = storage.get_insights(ticker, "full_analysis", limit=1)
    prices = storage.get_prices(ticker, limit=30)

    since_date = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
    news = storage.get_cached_news(ticker, since=since_date)

    return {
        "ticker": ticker,
        "company_name": company["name"] if company else ticker,
        "sector": company["sector"] if company else "Unknown",
        "report_markdown": report["report_markdown"],
        "report_timestamp": report["timestamp"],
        "full_analysis": insights[0]["content"] if insights else None,
        "prices": prices,
        "news": news,
    }


def _format_price_table(prices: list[dict]) -> str:
    if not prices:
        return "No price data available."
    header = "Date       | Close  | Open   | High   | Low    | Volume"
    sep = "-----------|--------|--------|--------|--------|----------"
    rows = [header, sep]
    for p in reversed(prices):
        rows.append(
            f"{p['date']} | {p['close']:6.2f} | {p['open']:6.2f} | "
            f"{p['high']:6.2f} | {p['low']:6.2f} | {p.get('volume', 0):,}"
        )
    return "\n".join(rows)


def _format_news_summary(news: list[dict]) -> str:
    if not news:
        return "No recent news available."
    items = []
    for item in news[:15]:
        ed = item.get("extracted_data") or {}
        sentiment = ed.get("sentiment", "")
        summary = ed.get("summary", "")
        date = (item.get("published") or "")[:10]
        line = f"- [{date}] {item.get('title', '')} ({item.get('source', '')})"
        if sentiment:
            line += f" [{sentiment}]"
        if summary:
            line += f"\n  {summary}"
        items.append(line)
    return "\n".join(items)


def format_system_prompt(context: dict) -> str:
    """Render the chat system prompt with all context data."""
    structured = ""
    if context["full_analysis"]:
        analysis = context["full_analysis"]
        structured = json.dumps(
            {
                k: analysis.get(k)
                for k in ("risk_factors", "key_observations", "outlook", "sentiment_analysis", "global_context_impact")
                if analysis.get(k)
            },
            indent=2,
        )

    return PROMPTS["chat_system_prompt"].format(
        today=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        ticker=context["ticker"],
        company_name=context["company_name"],
        sector=context["sector"],
        report_timestamp=context["report_timestamp"][:10],
        report_markdown=context["report_markdown"],
        structured_analysis=structured or "(not available)",
        price_table=_format_price_table(context["prices"]),
        news_summary=_format_news_summary(context["news"]),
    )


def run_chat_session(ticker: str, storage: Storage, config, verbose: bool = False):
    """Load context and run interactive chat loop."""
    context = build_context(ticker, storage)

    if context is None:
        print(f"No analysis found for {ticker}.")
        try:
            answer = input("Run analysis now? [y/N]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            sys.exit(0)

        if answer != "y":
            print("Run `uv run python analyze.py {ticker}` first.")
            sys.exit(0)

        print(f"Running analysis for {ticker}...")
        workflow = StockAnalyzerWorkflow(config, include_report=True)
        result = workflow.run(ticker)
        if result.get("error"):
            print(f"Analysis failed: {result['error']}")
            sys.exit(1)

        context = build_context(ticker, storage)
        if context is None:
            print("Analysis completed but no report was saved. Something went wrong.")
            sys.exit(1)

    # Staleness warning
    report_dt = None
    try:
        report_dt = datetime.fromisoformat(context["report_timestamp"].replace("Z", "+00:00"))
    except Exception:
        pass

    age_days = None
    if report_dt:
        if report_dt.tzinfo is None:
            report_dt = report_dt.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - report_dt).days

    print(f"\n--- Chat: {context['ticker']} — {context['company_name']} ({context['sector']}) ---")
    print(f"Report date: {context['report_timestamp'][:10]}", end="")
    if age_days is not None and age_days > 7:
        print(f"  [WARNING: report is {age_days} days old — consider re-running analysis]", end="")
    print()
    print("Type 'quit' or press Ctrl+C to exit.\n")

    system_prompt = format_system_prompt(context)
    client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    messages = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        messages.append({"role": "user", "content": user_input})

        print("\nAssistant: ", end="", flush=True)
        collected = []

        try:
            with client.messages.stream(
                model=config.analysis_model,
                max_tokens=2048,
                system=system_prompt,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                    collected.append(text)
        except anthropic.APIError as e:
            print(f"\n[API error: {e}]")
            messages.pop()  # remove the unanswered user message
            continue

        print("\n")
        messages.append({"role": "assistant", "content": "".join(collected)})


def main():
    parser = argparse.ArgumentParser(
        description="Interactive chat about Oslo Børs stock analysis reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s EQNR.OL          Chat about the most recent Equinor analysis
  %(prog)s DNB               Auto-adds .OL suffix
  %(prog)s MOWI --verbose    Enable verbose logging
        """,
    )
    parser.add_argument("ticker", help="Stock ticker (e.g. EQNR.OL or EQNR)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()
    setup_logging(args.verbose)

    try:
        config = get_config()
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)

    ticker = validate_ticker(args.ticker)
    storage = Storage(config.db_path)

    run_chat_session(ticker, storage, config, verbose=args.verbose)


if __name__ == "__main__":
    main()
