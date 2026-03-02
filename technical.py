"""Technical analysis module with algorithmic indicators."""

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


@dataclass
class SupportResistanceLevel:
    """A support or resistance level with metadata."""
    price: float
    strength: int  # Number of touches/points in cluster
    level_type: str  # "support" or "resistance"
    start_date: str | None = None  # Earliest date in cluster
    end_date: str | None = None  # Latest date in cluster


def calculate_support_resistance(
    prices: list[dict],
    n_clusters: int = 3,
    lookback_days: int | None = None,
) -> dict:
    """
    Calculate support and resistance levels using K-means clustering.

    Clusters lows separately (for support) and highs separately (for resistance).

    Args:
        prices: List of price dicts with 'high', 'low', 'close' keys (newest first).
        n_clusters: Number of clusters per type (support and resistance each).
        lookback_days: Only use this many recent days. None = use all.

    Returns:
        Dict with 'support_levels', 'resistance_levels', 'current_price', 'lookback_days'
    """
    if not prices:
        return {"support_levels": [], "resistance_levels": [], "current_price": None}

    # Apply lookback filter
    if lookback_days and len(prices) > lookback_days:
        prices = prices[:lookback_days]

    # Extract high and low prices with their dates
    highs = [(p["high"], p["date"]) for p in prices if p.get("high") is not None]
    lows = [(p["low"], p["date"]) for p in prices if p.get("low") is not None]

    if len(highs) == 0 or len(lows) == 0:
        return {"support_levels": [], "resistance_levels": [], "current_price": None}

    # Get current price
    current_price = prices[0].get("close") or prices[0].get("high")
    if current_price is None:
        return {"support_levels": [], "resistance_levels": [], "current_price": None}

    # Cluster lows for support levels
    support = _cluster_levels(lows, n_clusters, "support")

    # Cluster highs for resistance levels
    resistance = _cluster_levels(highs, n_clusters, "resistance")

    # Sort: support by price descending (closest first), resistance ascending (closest first)
    support.sort(key=lambda x: x.price, reverse=True)
    resistance.sort(key=lambda x: x.price)

    logger.info(f"Calculated S/R: {len(support)} support, {len(resistance)} resistance from {len(prices)} days")

    return {
        "support_levels": support,
        "resistance_levels": resistance,
        "current_price": current_price,
        "lookback_days": lookback_days or len(prices),
    }


def _cluster_levels(
    values_with_dates: list[tuple[float, str]],
    n_clusters: int,
    level_type: str,
) -> list[SupportResistanceLevel]:
    """Cluster price values and return S/R levels with date ranges."""
    values = np.array([v[0] for v in values_with_dates])
    dates = [v[1] for v in values_with_dates]

    n_clusters = min(n_clusters, len(np.unique(values)))
    if n_clusters < 1:
        return []

    values_2d = values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(values_2d)

    levels = []
    for i, center in enumerate(kmeans.cluster_centers_):
        # Find dates belonging to this cluster
        cluster_mask = kmeans.labels_ == i
        cluster_dates = [dates[j] for j in range(len(dates)) if cluster_mask[j]]

        levels.append(SupportResistanceLevel(
            price=round(float(center[0]), 2),
            strength=int(np.sum(cluster_mask)),
            level_type=level_type,
            start_date=min(cluster_dates) if cluster_dates else None,
            end_date=max(cluster_dates) if cluster_dates else None,
        ))

    return levels
