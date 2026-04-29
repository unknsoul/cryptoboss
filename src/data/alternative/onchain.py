"""On-chain data fetchers."""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import requests


GLASSNODE_BASE_URL = "https://api.glassnode.com/v1/metrics"


def fetch_glassnode_metric(
    metric: str,
    asset: str = "BTC",
    api_key: Optional[str] = None,
    base_url: str = GLASSNODE_BASE_URL,
) -> pd.DataFrame:
    """Fetch a Glassnode metric as a timestamped DataFrame."""
    api_key = api_key or os.getenv("GLASSNODE_API_KEY")
    if not api_key:
        raise ValueError("GLASSNODE_API_KEY is required")

    url = f"{base_url}/{metric.lstrip('/')}"
    params = {"a": asset, "api_key": api_key}

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    payload = response.json() or []
    df = pd.DataFrame(payload)
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df.get("t"), unit="s", utc=True, errors="coerce")
    df["value"] = pd.to_numeric(df.get("v"), errors="coerce")
    return df[["timestamp", "value"]].dropna()


def fetch_exchange_netflow(
    asset: str = "BTC",
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch exchange netflow from Glassnode."""
    df = fetch_glassnode_metric("exchange/netflow", asset=asset, api_key=api_key)
    if df.empty:
        return df
    df = df.rename(columns={"value": "netflow"})
    return df
