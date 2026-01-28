"""Visualization module for price charts with technical analysis overlays."""

import logging
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

logger = logging.getLogger(__name__)


def plot_price_chart(
    ticker: str,
    prices: list[dict],
    analysis: dict = None,
    output_path: str = None,
) -> str | None:
    """
    Generate a price chart with moving averages and support/resistance levels.

    Args:
        ticker: Stock ticker symbol
        prices: List of price dicts with 'date', 'close', etc.
        analysis: Analysis dict containing support/resistance levels
        output_path: Where to save the chart (optional)

    Returns:
        Path to saved chart, or None if failed
    """
    logger.info(f"plot_price_chart called for {ticker} with {len(prices) if prices else 0} prices, output_path={output_path}")

    try:
        if not prices:
            logger.warning("No prices to plot")
            return None

        # Sort prices by date (oldest first for plotting) and filter out None values
        prices_sorted = sorted(prices, key=lambda x: x['date'])
        prices_valid = [p for p in prices_sorted if p.get('close') is not None]

        if not prices_valid:
            logger.warning("No valid close prices to plot")
            return None

        logger.info(f"Plotting {len(prices_valid)} valid price points")

        # Extract data
        dates = [datetime.strptime(p['date'], '%Y-%m-%d') for p in prices_valid]
        closes = [p['close'] for p in prices_valid]

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot price line
        ax.plot(dates, closes, label='Close Price', color='#2962FF', linewidth=1.5)

        # Calculate and plot moving averages
        ma_periods = [10, 20, 50]
        ma_colors = ['#FF6D00', '#00C853', '#AA00FF']

        for period, color in zip(ma_periods, ma_colors):
            if len(closes) >= period:
                ma = _calculate_ma(closes, period)
                # Align MA with dates (MA starts after 'period' days)
                ma_dates = dates[period-1:]
                ax.plot(ma_dates, ma, label=f'{period}-day MA', color=color, linewidth=1, alpha=0.8)

        # Extract support/resistance from analysis
        if analysis:
            price_analysis = analysis.get('price_analysis', {})
            support_levels = price_analysis.get('support_levels', [])
            resistance_levels = price_analysis.get('resistance_levels', [])

            logger.info(f"Support levels: {support_levels}, Resistance levels: {resistance_levels}")

            # Plot support levels (green dashed lines)
            for level in support_levels:
                if isinstance(level, (int, float)) and level > 0:
                    ax.axhline(y=level, color='#00C853', linestyle='--', linewidth=1.5, alpha=0.7)
                    ax.text(dates[-1], level, f' Support: {level}', va='center', fontsize=9, color='#00C853')

            # Plot resistance levels (red dashed lines)
            for level in resistance_levels:
                if isinstance(level, (int, float)) and level > 0:
                    ax.axhline(y=level, color='#FF1744', linestyle='--', linewidth=1.5, alpha=0.7)
                    ax.text(dates[-1], level, f' Resistance: {level}', va='center', fontsize=9, color='#FF1744')

        # Formatting
        ax.set_title(f'{ticker} - Price Chart with Technical Analysis', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Price (NOK)', fontsize=11)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.xticks(rotation=45, ha='right')

        # Add current price annotation
        if closes:
            current_price = closes[-1]
            ax.annotate(
                f'Current: {current_price:.2f}',
                xy=(dates[-1], current_price),
                xytext=(10, 0),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold',
                color='#2962FF',
            )

        plt.tight_layout()

        # Save or show
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving chart to {output_path}")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Chart saved successfully to {output_path}")
            return output_path
        else:
            plt.show()
            plt.close(fig)
            return None

    except Exception as e:
        logger.error(f"Error in plot_price_chart: {e}", exc_info=True)
        plt.close('all')  # Clean up any open figures
        return None


def _calculate_ma(prices: list[float], period: int) -> list[float]:
    """Calculate simple moving average."""
    ma = []
    for i in range(period - 1, len(prices)):
        window = prices[i - period + 1:i + 1]
        ma.append(sum(window) / period)
    return ma
