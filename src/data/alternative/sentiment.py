"""Sentiment data fetchers."""

from __future__ import annotations

import pandas as pd
import requests


def fetch_fear_greed_index(limit: int = 200) -> pd.DataFrame:
    """Fetch Fear & Greed index data from Alternative.me."""
    url = "https://api.alternative.me/fng/"
    params = {"limit": min(max(limit, 1), 2000), "format": "json"}

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    payload = response.json().get("data", [])
    df = pd.DataFrame(payload)
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.rename(columns={"value": "fear_greed"})
    return df[["timestamp", "fear_greed", "value_classification"]].dropna()


def normalize_sentiment_score(series: pd.Series) -> pd.Series:
    """Normalize a 0-100 sentiment series to -1..1."""
    return (series.astype(float) - 50.0) / 50.0
