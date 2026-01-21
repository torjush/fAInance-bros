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

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END

from config import get_config, Config
from data.storage import Storage
from agents.context import ContextAgent
from agents.collector import CollectorAgent
from agents.analyzer import AnalyzerAgent
from agents.reporter import ReporterAgent


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
    analysis: dict[str, Any]
    report: str
    error: str | None
    status: str


class StockAnalyzerWorkflow:
    """LangGraph workflow for stock analysis."""

    def __init__(self, config: Config):
        self.config = config
        self.storage = Storage(config.db_path)

        # Initialize agents
        self.context_agent = ContextAgent(self.storage)
        self.collector_agent = CollectorAgent(self.storage, config)
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
        workflow.add_node("analyze", self._analyze)
        workflow.add_node("generate_report", self._generate_report)

        # Define edges (linear workflow)
        workflow.set_entry_point("gather_context")
        workflow.add_edge("gather_context", "collect_data")
        workflow.add_edge("collect_data", "analyze")
        workflow.add_edge("analyze", "generate_report")
        workflow.add_edge("generate_report", END)

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

    def run(self, ticker: str) -> AnalysisState:
        """Run the full analysis workflow."""
        initial_state: AnalysisState = {
            "ticker": ticker,
            "context": {},
            "collected_data": {},
            "analysis": {},
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


def main():
    parser = argparse.ArgumentParser(
        description="AI-powered stock analyzer for Oslo Stock Exchange",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s EQNR.OL                    Analyze Equinor
  %(prog)s EQNR.OL --output report.md Save report to file
  %(prog)s EQNR.OL --verbose          Enable verbose logging
  %(prog)s DNB                        Analyze DNB (auto-adds .OL suffix)
        """,
    )

    parser.add_argument(
        "ticker",
        help="Stock ticker symbol (e.g., EQNR.OL or EQNR)",
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

    # Validate ticker
    ticker = validate_ticker(args.ticker)
    logger.info(f"Starting analysis for {ticker}")

    # Get configuration
    config = get_config()

    # Override database path if provided
    if args.db_path:
        config.db_path = args.db_path

    # Validate API key
    if not config.anthropic_api_key:
        logger.error("ANTHROPIC_API_KEY environment variable is not set")
        sys.exit(1)

    # Create and run workflow
    try:
        workflow = StockAnalyzerWorkflow(config)
        result = workflow.run(ticker)

        # Output report
        report = result.get("report", "")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(f"Report saved to {args.output}")
        else:
            print(report)

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
