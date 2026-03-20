"""Portfolio Analyzer - Runs per-stock workflows in parallel, then generates a unified report."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import Config
from agents.global_news import GlobalNewsAgent
from agents.portfolio_reporter import PortfolioReporter
from analyze import StockAnalyzerWorkflow, AnalysisState

logger = logging.getLogger(__name__)


class PortfolioAnalyzer:
    """
    Orchestrates parallel analysis of multiple stocks and produces a single portfolio report.

    Steps:
    1. Fetch global news once (shared across all stocks)
    2. Run StockAnalyzerWorkflow (without individual reports) in parallel threads
    3. Collect results and pass to PortfolioReporter for a unified report
    """

    def __init__(self, config: Config):
        self.config = config

    def run(self, tickers: list[str]) -> str:
        """
        Analyze a portfolio of tickers and generate a unified report.

        Args:
            tickers: List of validated ticker symbols (e.g. ['EQNR.OL', 'DNB.OL'])

        Returns:
            Path to the saved portfolio report (.md file)
        """
        logger.info(f"Starting portfolio analysis for {len(tickers)} stocks")

        # Step 1: Fetch global news once
        global_news = self._fetch_global_news()

        # Step 2: Run per-stock analysis in parallel
        states = self._analyze_all(tickers, global_news)

        if not states:
            raise RuntimeError("All stock analyses failed — cannot generate portfolio report")

        # Step 3: Generate unified portfolio report
        reporter = PortfolioReporter(self.config)
        md_path, _ = reporter.generate_report(states, global_news)

        return str(md_path)

    def _fetch_global_news(self) -> dict:
        """Fetch global macro news synchronously."""
        logger.info("Fetching global news for portfolio analysis")
        try:
            return asyncio.run(GlobalNewsAgent(self.config).fetch())
        except Exception as e:
            logger.warning(f"Global news fetch failed: {e} — continuing without macro context")
            return {}

    def _analyze_ticker(self, ticker: str, global_news: dict) -> AnalysisState | None:
        """Run a single-stock workflow in a thread. Returns state or None on failure."""
        try:
            logger.info(f"[{ticker}] Starting analysis")
            workflow = StockAnalyzerWorkflow(self.config, include_report=False)
            state = workflow.run(ticker, global_news=global_news)
            logger.info(f"[{ticker}] Analysis complete (status={state.get('status')})")
            return state
        except Exception as e:
            logger.error(f"[{ticker}] Analysis failed: {e}")
            return None

    def _analyze_all(self, tickers: list[str], global_news: dict) -> list[AnalysisState]:
        """Run all ticker analyses in parallel using a thread pool."""
        max_workers = min(len(tickers), self.config.max_concurrent_requests)
        results: list[AnalysisState] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(self._analyze_ticker, ticker, global_news): ticker
                for ticker in tickers
            }
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    state = future.result()
                    if state is not None:
                        results.append(state)
                    else:
                        logger.warning(f"[{ticker}] Skipped due to analysis failure")
                except Exception as e:
                    logger.error(f"[{ticker}] Unexpected error: {e}")

        return results
