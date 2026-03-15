"""Report Generator Agent - Creates markdown reports from analysis."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
from markdown_pdf import MarkdownPdf, Section

from config import Config, PROMPTS
from data.storage import Storage

logger = logging.getLogger(__name__)


class ReporterAgent:
    """
    Agent responsible for generating markdown reports from analysis.

    Uses Claude Sonnet to create professional, readable reports.
    """

    def __init__(self, storage: Storage, config: Config):
        self.storage = storage
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    def generate_report(
        self,
        ticker: str,
        analysis: dict[str, Any],
        chart_path: str = None,
    ) -> str:
        """
        Generate a markdown report from the analysis.

        Args:
            ticker: Stock ticker symbol
            analysis: Analysis results from AnalyzerAgent
            chart_path: Path to the generated chart image

        Returns:
            Markdown formatted report string
        """
        logger.info(f"Generating report for {ticker}")

        # Extract data for the prompt
        company_name = analysis.get("company_name", ticker)
        sector = analysis.get("sector", "Unknown")
        analysis_date = analysis.get("analysis_timestamp", datetime.now(timezone.utc).isoformat())

        # Format analysis components
        price_analysis = json.dumps(analysis.get("price_analysis", {}), indent=2)
        sentiment_analysis = json.dumps(analysis.get("sentiment_analysis", {}), indent=2)
        risk_factors = json.dumps(analysis.get("risk_factors", []), indent=2)
        key_observations = "\n".join(f"- {obs}" for obs in analysis.get("key_observations", []))
        outlook = analysis.get("outlook", "No outlook available")
        global_context = analysis.get("global_context_impact", "No global context available.")
        targeted_context = analysis.get("targeted_news_context", "No sector/geography context available.")

        # Format recent prices
        recent_prices = self._format_price_table(analysis.get("recent_prices", []))

        # Build the prompt
        prompt = PROMPTS["generate_report"].format(
            ticker=ticker,
            company_name=company_name,
            sector=sector,
            analysis_date=analysis_date,
            price_analysis=price_analysis,
            sentiment_analysis=sentiment_analysis,
            global_context=global_context,
            targeted_context=targeted_context,
            risk_factors=risk_factors,
            key_observations=key_observations,
            outlook=outlook,
            recent_prices=recent_prices,
        )

        # Generate report with Claude Sonnet
        try:
            response = self.client.messages.create(
                model=self.config.analysis_model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )

            report = response.content[0].text.strip()

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            report = self._create_fallback_report(ticker, analysis)

        # Embed chart if available
        if chart_path:
            chart_section = self._create_chart_section(chart_path)
            if chart_section:
                # Insert chart after first heading
                lines = report.split('\n')
                insert_idx = 1
                for i, line in enumerate(lines):
                    if line.startswith('#') and i > 0:
                        insert_idx = i
                        break
                lines.insert(insert_idx, chart_section)
                report = '\n'.join(lines)

        # Save report to database
        self.storage.save_report(ticker, report)

        # Update last analyzed timestamp
        self.storage.update_last_analyzed(ticker)

        # Save to file if reports directory is configured
        self._save_report_file(ticker, report)

        logger.info(f"Report generated for {ticker}")
        return report

    def _create_chart_section(self, chart_path: str) -> str | None:
        """Create markdown section with chart image link."""
        path = Path(chart_path)
        if not path.exists():
            logger.warning(f"Chart file not found: {chart_path}")
            return None

        return f"""
## Price Chart

![Price Chart with Technical Analysis]({path.name})

"""

    def _format_price_table(self, prices: list[dict]) -> str:
        """Format prices as a markdown table."""
        if not prices:
            return "No price data available."

        lines = ["| Date | Open | High | Low | Close | Volume |"]
        lines.append("|------|------|------|-----|-------|--------|")

        for p in prices[:10]:  # Last 10 days
            lines.append(
                f"| {p.get('date', 'N/A')} | "
                f"{p.get('open', 'N/A')} | "
                f"{p.get('high', 'N/A')} | "
                f"{p.get('low', 'N/A')} | "
                f"{p.get('close', 'N/A')} | "
                f"{p.get('volume', 'N/A'):,} |" if p.get('volume') else
                f"| {p.get('date', 'N/A')} | "
                f"{p.get('open', 'N/A')} | "
                f"{p.get('high', 'N/A')} | "
                f"{p.get('low', 'N/A')} | "
                f"{p.get('close', 'N/A')} | N/A |"
            )

        return "\n".join(lines)

    def _create_fallback_report(self, ticker: str, analysis: dict) -> str:
        """Create a basic report if LLM generation fails."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        company_name = analysis.get("company_name", ticker)

        price_stats = analysis.get("price_stats", {})
        current_price = price_stats.get("current_price", "N/A")

        return f"""# Stock Analysis Report: {ticker}

**Company:** {company_name}
**Generated:** {timestamp}

---

## Executive Summary

This report was generated with limited analysis due to a processing error.

## Current Price

**{current_price}** NOK

## Price Statistics

```json
{json.dumps(price_stats, indent=2)}
```

## Key Observations

{chr(10).join('- ' + obs for obs in analysis.get('key_observations', ['No observations available']))}

## Outlook

{analysis.get('outlook', 'Unable to generate outlook')}

---

*This report was auto-generated. Please verify all information independently.*

**Disclaimer:** This analysis is generated by AI and should not be considered financial advice. Always conduct your own research and consult with qualified financial advisors before making investment decisions.
"""

    def _save_report_file(self, ticker: str, report: str):
        """Save report to file system as markdown and PDF."""
        try:
            reports_dir = Path(self.config.reports_dir)
            reports_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            base_filename = f"{ticker.replace('.', '_')}_{timestamp}"

            # Save markdown
            md_filepath = reports_dir / f"{base_filename}.md"
            with open(md_filepath, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(f"Report saved to {md_filepath}")

            # Generate PDF
            pdf_filepath = reports_dir / f"{base_filename}.pdf"
            self._generate_pdf(report, pdf_filepath, reports_dir)

        except Exception as e:
            logger.warning(f"Failed to save report file: {e}")

    def _generate_pdf(self, report: str, output_path: Path, base_dir: Path):
        """Convert markdown report to PDF."""
        try:
            pdf = MarkdownPdf()
            pdf.add_section(Section(report, root=str(base_dir)))
            pdf.save(str(output_path))
            logger.info(f"PDF saved to {output_path}")

        except Exception as e:
            logger.warning(f"Failed to generate PDF: {e}")
