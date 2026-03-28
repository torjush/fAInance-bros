"""Portfolio Reporter Agent - Generates a unified report for a portfolio of stocks."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
from markdown_pdf import MarkdownPdf, Section

from config import Config, PROMPTS

logger = logging.getLogger(__name__)


class PortfolioReporter:
    """
    Generates a single consolidated portfolio report using one Claude Sonnet call.

    The report includes:
    - Market Overview (macro context)
    - Individual Stock Analysis sections (with BHS recommendation per stock)
    - Portfolio Summary table
    - Disclaimer
    """

    def __init__(self, config: Config):
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    def generate_report(
        self,
        states: list[dict[str, Any]],
        global_news: dict[str, Any],
        macro_advice: dict[str, Any] | None = None,
    ) -> tuple[Path, Path]:
        """
        Generate the portfolio report.

        Args:
            states: List of completed AnalysisState dicts from per-stock workflows
            global_news: Pre-fetched global news dict
            macro_advice: Optional output from MacroAdvisorAgent with stock_ideas list

        Returns:
            Tuple of (md_path, pdf_path)
        """
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        stock_data = self._format_stock_data(states)
        global_context = self._format_global_context(global_news)
        macro_advice_text = self._format_macro_advice(macro_advice)

        prompt = PROMPTS["generate_portfolio_report"].format(
            date=date_str,
            global_context=global_context,
            stock_data=stock_data,
            macro_advice=macro_advice_text,
            # Placeholder tokens in the prompt template — replaced by Claude in output
            ticker_placeholder="TICKER",
            company_placeholder="Company Name",
            sector_placeholder="Sector",
            price_placeholder="XXX.XX NOK",
        )

        logger.info(f"Generating portfolio report for {len(states)} stocks")

        try:
            response = self.client.messages.create(
                model=self.config.analysis_model,
                max_tokens=8192,
                messages=[{"role": "user", "content": prompt}],
            )
            report = response.content[0].text.strip()
        except Exception as e:
            logger.error(f"Portfolio report generation failed: {e}")
            report = self._fallback_report(states, date_str)

        # Embed chart images after each stock heading
        report = self._embed_charts(report, states)

        return self._save_report(report)

    def _format_global_context(self, global_news: dict) -> str:
        """Format global news dict into a readable prompt block."""
        if not global_news:
            return "No global macro context available."

        lines = []
        if summary := global_news.get("summary"):
            lines.append(f"**Summary:** {summary}")
        if sentiment := global_news.get("market_sentiment"):
            lines.append(f"**Market Sentiment:** {sentiment}")
        if themes := global_news.get("key_themes"):
            lines.append("**Key Themes:** " + ", ".join(themes))
        if events := global_news.get("macro_events"):
            lines.append("**Macro Events:** " + ", ".join(events))
        if safer := global_news.get("safer_sectors"):
            lines.append("**Favoured Sectors:** " + ", ".join(safer))
        if avoid := global_news.get("avoid_sectors"):
            lines.append("**Sectors to Avoid:** " + ", ".join(avoid))
        return "\n".join(lines) if lines else "No global macro context available."

    def _format_macro_advice(self, macro_advice: dict[str, Any] | None) -> str:
        """Format MacroAdvisorAgent output for injection into the report prompt."""
        if not macro_advice:
            return "No macro stock ideas available."

        ideas = macro_advice.get("stock_ideas", [])
        if not ideas:
            return "No macro stock ideas available."

        lines = []
        for idea in ideas:
            ticker = idea.get("ticker", "?")
            company = idea.get("company", "")
            sector = idea.get("sector", "")
            rationale = idea.get("rationale", "")
            risk_note = idea.get("risk_note", "")
            lines.append(
                f"- **{ticker}** ({company}) | Sector: {sector}\n"
                f"  Rationale: {rationale}\n"
                f"  Risk: {risk_note}"
            )
        return "\n".join(lines)

    def _format_stock_data(self, states: list[dict[str, Any]]) -> str:
        """Format per-stock analysis data into prompt blocks."""
        blocks = []
        for state in states:
            ticker = state.get("ticker", "UNKNOWN")
            analysis = state.get("analysis", {})

            company_name = analysis.get("company_name", ticker)
            sector = analysis.get("sector", "Unknown")
            price_stats = analysis.get("price_stats", {})
            current_price = price_stats.get("current_price", "N/A")

            price_analysis = json.dumps(analysis.get("price_analysis", {}), indent=2)
            sentiment_analysis = json.dumps(analysis.get("sentiment_analysis", {}), indent=2)
            risk_factors = json.dumps(analysis.get("risk_factors", []), indent=2)
            key_obs = "\n".join(f"- {o}" for o in analysis.get("key_observations", []))
            outlook = analysis.get("outlook", "N/A")
            global_impact = analysis.get("global_context_impact", "N/A")
            targeted_ctx = analysis.get("targeted_news_context", "N/A")

            block = f"""---
## {ticker} — {company_name}
**Sector:** {sector}
**Current Price:** {current_price} NOK

### Price Analysis
{price_analysis}

### Sentiment Analysis
{sentiment_analysis}

### Global Context Impact
{global_impact}

### Sector/Geographic Context
{targeted_ctx}

### Risk Factors
{risk_factors}

### Key Observations
{key_obs}

### Outlook
{outlook}
"""
            blocks.append(block)

        return "\n".join(blocks)

    def _embed_charts(self, report: str, states: list[dict[str, Any]]) -> str:
        """Insert chart image references after each stock section heading."""
        for state in states:
            ticker = state.get("ticker", "")
            chart_path = state.get("chart_path")
            if not chart_path:
                continue
            chart_file = Path(chart_path)
            if not chart_file.exists():
                continue

            chart_md = f"\n![{ticker} Price Chart]({chart_file.name})\n"
            # Find the stock heading line and insert chart after it
            heading = f"### {ticker}"
            if heading in report:
                report = report.replace(heading, f"{heading}\n{chart_md}", 1)

        return report

    def _save_report(self, report: str) -> tuple[Path, Path]:
        """Save report as .md and .pdf, return both paths."""
        reports_dir = Path(self.config.reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        base = f"portfolio_{timestamp}"

        md_path = reports_dir / f"{base}.md"
        pdf_path = reports_dir / f"{base}.pdf"

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Portfolio report saved to {md_path}")

        try:
            pdf = MarkdownPdf()
            pdf.add_section(Section(report, root=str(reports_dir)))
            pdf.save(str(pdf_path))
            logger.info(f"Portfolio PDF saved to {pdf_path}")
        except Exception as e:
            logger.warning(f"PDF generation failed: {e}")

        return md_path, pdf_path

    def _fallback_report(self, states: list[dict[str, Any]], date_str: str) -> str:
        """Minimal fallback report if LLM call fails."""
        tickers = [s.get("ticker", "?") for s in states]
        return f"""# Portfolio Analysis — {date_str}

*Report generation encountered an error. Raw analysis data is available in the database.*

## Analyzed Stocks
{chr(10).join(f'- {t}' for t in tickers)}

---

**Disclaimer:** This analysis is AI-generated and not financial advice.
"""
