#!/usr/bin/env python3
"""
Stock Analyzer CLI - AI-powered analysis of Oslo Stock Exchange companies.

Usage:
    python analyze.py TICKER [OPTIONS]

Examples:
    python analyze.py EQNR.OL
    python analyze.py EQNR.OL --output report.md
    python analyze.py EQNR.OL --verbose
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END

from config import get_config, Config
from data.storage import Storage
from agents.context import ContextAgent
from agents.collector import CollectorAgent
from agents.global_news import GlobalNewsAgent
from agents.company_profile import CompanyProfileAgent
from agents.targeted_news import TargetedNewsAgent
from agents.analyzer import AnalyzerAgent
from agents.reporter import ReporterAgent
from visualization import plot_price_chart


# Configure logging
def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# State definition for LangGraph
class AnalysisState(TypedDict):
    """State passed between agents in the workflow."""
    ticker: str
    context: dict[str, Any]
    collected_data: dict[str, Any]
    global_news: dict[str, Any]
    company_profile: dict[str, Any]
    targeted_news: dict[str, Any]
    analysis: dict[str, Any]
    chart_path: str | None
    report: str
    error: str | None
    status: str


class StockAnalyzerWorkflow:
    """LangGraph workflow for stock analysis."""

    def __init__(self, config: Config, include_report: bool = True):
        self.config = config
        self.include_report = include_report
        self.storage = Storage(config.db_path)

        # Initialize agents
        self.context_agent = ContextAgent(self.storage)
        self.collector_agent = CollectorAgent(self.storage, config)
        self.global_news_agent = GlobalNewsAgent(config)
        self.company_profile_agent = CompanyProfileAgent(self.storage, config)
        self.targeted_news_agent = TargetedNewsAgent(self.storage, config)
        self.analyzer_agent = AnalyzerAgent(self.storage, config)
        self.reporter_agent = ReporterAgent(self.storage, config)

        # Build the workflow graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create the graph with our state schema
        workflow = StateGraph(AnalysisState)

        # Add nodes for each agent
        workflow.add_node("gather_context", self._gather_context)
        workflow.add_node("collect_data", self._collect_data)
        workflow.add_node("fetch_global_news", self._fetch_global_news)
        workflow.add_node("profile_company", self._profile_company)
        workflow.add_node("fetch_targeted_news", self._fetch_targeted_news)
        workflow.add_node("analyze", self._analyze)
        workflow.add_node("generate_chart", self._generate_chart)

        # Define edges (linear workflow)
        workflow.set_entry_point("gather_context")
        workflow.add_edge("gather_context", "collect_data")
        workflow.add_edge("collect_data", "fetch_global_news")
        workflow.add_edge("fetch_global_news", "profile_company")
        workflow.add_edge("profile_company", "fetch_targeted_news")
        workflow.add_edge("fetch_targeted_news", "analyze")
        workflow.add_edge("analyze", "generate_chart")

        if self.include_report:
            workflow.add_node("generate_report", self._generate_report)
            workflow.add_edge("generate_chart", "generate_report")
            workflow.add_edge("generate_report", END)
        else:
            workflow.add_edge("generate_chart", END)

        return workflow.compile()

    def _gather_context(self, state: AnalysisState) -> dict:
        """Node: Gather historical context from database."""
        logging.info(f"[Context Agent] Gathering context for {state['ticker']}")

        try:
            context = self.context_agent.get_context(state["ticker"])
            return {
                "context": context,
                "status": "context_gathered",
            }
        except Exception as e:
            logging.error(f"Context gathering failed: {e}")
            return {
                "context": {},
                "error": str(e),
                "status": "context_failed",
            }

    def _collect_data(self, state: AnalysisState) -> dict:
        """Node: Collect fresh data from external sources."""
        logging.info(f"[Collector Agent] Collecting data for {state['ticker']}")

        try:
            # Run async collection in sync context
            collected_data = asyncio.run(
                self.collector_agent.collect(
                    ticker=state["ticker"],
                    context=state["context"],
                )
            )
            return {
                "collected_data": collected_data,
                "status": "data_collected",
            }
        except Exception as e:
            logging.error(f"Data collection failed: {e}")
            return {
                "collected_data": {},
                "error": str(e),
                "status": "collection_failed",
            }

    def _fetch_global_news(self, state: AnalysisState) -> dict:
        """Node: Fetch global financial news for macro context."""
        if state.get("global_news"):
            logging.info("[Global News Agent] Using pre-loaded global news")
            return {"status": "global_news_preloaded"}

        logging.info("[Global News Agent] Fetching global news")

        try:
            global_news = asyncio.run(self.global_news_agent.fetch())
            return {
                "global_news": global_news,
                "status": "global_news_fetched",
            }
        except Exception as e:
            logging.error(f"Global news fetch failed: {e}")
            return {
                "global_news": {},
                "error": str(e),
                "status": "global_news_failed",
            }

    def _profile_company(self, state: AnalysisState) -> dict:
        """Node: Build company profile (sectors, geographies, search queries)."""
        logging.info(f"[Company Profile Agent] Profiling {state['ticker']}")

        try:
            stock_info = state.get("collected_data", {}).get("stock_info", {})
            company_profile = self.company_profile_agent.profile(
                ticker=state["ticker"],
                stock_info=stock_info,
                context=state["context"],
            )
            return {
                "company_profile": company_profile,
                "status": "company_profiled",
            }
        except Exception as e:
            logging.error(f"Company profiling failed: {e}")
            return {
                "company_profile": {},
                "error": str(e),
                "status": "profile_failed",
            }

    def _fetch_targeted_news(self, state: AnalysisState) -> dict:
        """Node: Fetch sector/geography-targeted news."""
        logging.info(f"[Targeted News Agent] Fetching targeted news for {state['ticker']}")

        try:
            targeted_news = asyncio.run(
                self.targeted_news_agent.fetch(
                    ticker=state["ticker"],
                    company_profile=state.get("company_profile", {}),
                )
            )
            return {
                "targeted_news": targeted_news,
                "status": "targeted_news_fetched",
            }
        except Exception as e:
            logging.error(f"Targeted news fetch failed: {e}")
            return {
                "targeted_news": {},
                "error": str(e),
                "status": "targeted_news_failed",
            }

    def _analyze(self, state: AnalysisState) -> dict:
        """Node: Analyze collected data."""
        logging.info(f"[Analyzer Agent] Analyzing {state['ticker']}")

        if not state.get("collected_data"):
            return {
                "analysis": {},
                "error": "No data to analyze",
                "status": "analysis_failed",
            }

        try:
            analysis = self.analyzer_agent.analyze(
                ticker=state["ticker"],
                context=state["context"],
                collected_data=state["collected_data"],
                global_news=state.get("global_news", {}),
                targeted_news=state.get("targeted_news", {}),
            )
            return {
                "analysis": analysis,
                "status": "analysis_complete",
            }
        except Exception as e:
            logging.error(f"Analysis failed: {e}")
            return {
                "analysis": {},
                "error": str(e),
                "status": "analysis_failed",
            }

    def _generate_chart(self, state: AnalysisState) -> dict:
        """Node: Generate price chart with technical analysis overlays."""
        logging.info(f"[Visualization] Generating chart for {state['ticker']}")

        try:
            # Get prices - prefer from analysis (which has recent_prices), fall back to context
            analysis = state.get("analysis", {})
            prices = analysis.get("recent_prices", [])

            # If no prices in analysis, try context (historical data from DB)
            if not prices:
                prices = state.get("context", {}).get("price_history", [])

            if not prices:
                logging.warning("No price data for chart")
                return {"chart_path": None, "status": "chart_skipped"}

            # Generate chart path
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            ticker_safe = state["ticker"].replace(".", "_")
            chart_path = f"{self.config.reports_dir}/{ticker_safe}_{timestamp}.png"

            # Generate the chart
            result_path = plot_price_chart(
                ticker=state["ticker"],
                prices=prices,
                analysis=analysis,
                output_path=chart_path,
            )

            return {
                "chart_path": result_path,
                "status": "chart_generated",
            }

        except Exception as e:
            logging.error(f"Chart generation failed: {e}", exc_info=True)
            return {
                "chart_path": None,
                "status": "chart_failed",
            }

    def _generate_report(self, state: AnalysisState) -> dict:
        """Node: Generate final report."""
        logging.info(f"[Reporter Agent] Generating report for {state['ticker']}")

        if not state.get("analysis"):
            return {
                "report": "# Error\n\nAnalysis failed - unable to generate report.",
                "error": "No analysis to report",
                "status": "report_failed",
            }

        try:
            report = self.reporter_agent.generate_report(
                ticker=state["ticker"],
                analysis=state["analysis"],
                chart_path=state.get("chart_path"),
            )
            return {
                "report": report,
                "status": "complete",
            }
        except Exception as e:
            logging.error(f"Report generation failed: {e}")
            return {
                "report": f"# Error\n\nReport generation failed: {e}",
                "error": str(e),
                "status": "report_failed",
            }

    def run(self, ticker: str, global_news: dict | None = None) -> AnalysisState:
        """Run the full analysis workflow."""
        initial_state: AnalysisState = {
            "ticker": ticker,
            "context": {},
            "collected_data": {},
            "global_news": global_news or {},
            "company_profile": {},
            "targeted_news": {},
            "analysis": {},
            "chart_path": None,
            "report": "",
            "error": None,
            "status": "starting",
        }

        # Execute the workflow
        final_state = self.graph.invoke(initial_state)
        return final_state


def validate_ticker(ticker: str) -> str:
    """Validate and normalize ticker symbol."""
    ticker = ticker.upper().strip()

    # Add .OL suffix if not present for Oslo Stock Exchange
    if not ticker.endswith(".OL"):
        ticker = f"{ticker}.OL"

    return ticker


def load_portfolio_file(path: str) -> list[str]:
    """Load and validate tickers from a portfolio file."""
    tickers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Strip inline comments
            line = line.split("#")[0].strip()
            if not line:
                continue
            tickers.append(validate_ticker(line))
    return tickers


def main():
    parser = argparse.ArgumentParser(
        description="AI-powered stock analyzer for Oslo Stock Exchange",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s EQNR.OL                         Analyze Equinor
  %(prog)s EQNR.OL --output report.md      Save report to file
  %(prog)s EQNR.OL --verbose               Enable verbose logging
  %(prog)s DNB                             Analyze DNB (auto-adds .OL suffix)
  %(prog)s --portfolio portfolio.txt       Analyze full portfolio
        """,
    )

    parser.add_argument(
        "ticker",
        nargs="?",
        help="Stock ticker symbol (e.g., EQNR.OL or EQNR)",
    )

    parser.add_argument(
        "-p", "--portfolio",
        help="Path to portfolio file (one ticker per line, # for comments)",
        type=str,
        default=None,
    )

    parser.add_argument(
        "-o", "--output",
        help="Output file for the report (default: stdout)",
        type=str,
        default=None,
    )

    parser.add_argument(
        "-v", "--verbose",
        help="Enable verbose logging",
        action="store_true",
    )

    parser.add_argument(
        "--db-path",
        help="Path to SQLite database",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Validate mutual exclusion
    if args.ticker and args.portfolio:
        logger.error("Specify either a ticker or --portfolio, not both")
        sys.exit(1)
    if not args.ticker and not args.portfolio:
        parser.print_help()
        sys.exit(1)

    # Get configuration
    config = get_config()

    # Override database path if provided
    if args.db_path:
        config.db_path = args.db_path

    # Validate API key
    if not config.anthropic_api_key:
        logger.error("ANTHROPIC_API_KEY environment variable is not set")
        sys.exit(1)

    # Portfolio mode
    if args.portfolio:
        from portfolio_analyzer import PortfolioAnalyzer
        try:
            tickers = load_portfolio_file(args.portfolio)
            if not tickers:
                logger.error(f"No valid tickers found in {args.portfolio}")
                sys.exit(1)
            logger.info(f"Loaded {len(tickers)} tickers from portfolio: {', '.join(tickers)}")
            report_path = PortfolioAnalyzer(config).run(tickers)
            logger.info(f"Portfolio report saved to {report_path}")
        except FileNotFoundError:
            logger.error(f"Portfolio file not found: {args.portfolio}")
            sys.exit(1)
        except KeyboardInterrupt:
            logger.info("Analysis interrupted by user")
            sys.exit(130)
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
        return

    # Single-ticker mode
    ticker = validate_ticker(args.ticker)
    logger.info(f"Starting analysis for {ticker}")

    try:
        workflow = StockAnalyzerWorkflow(config)
        result = workflow.run(ticker)

        # Output report to custom location if specified
        if args.output:
            report = result.get("report", "")
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(f"Report saved to {args.output}")

        # Log chart location
        if result.get("chart_path"):
            logger.info(f"Chart saved to {result['chart_path']}")

        # Check for errors
        if result.get("error"):
            logger.warning(f"Workflow completed with errors: {result['error']}")
            sys.exit(1)

        logger.info(f"Analysis complete for {ticker}")

    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(130)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
