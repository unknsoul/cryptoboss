"""Options flow and implied volatility fetchers."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import requests


DERIBIT_BASE_URL = "https://www.deribit.com"


def fetch_deribit_option_summaries(currency: str = "BTC", base_url: str = DERIBIT_BASE_URL) -> pd.DataFrame:
    """Fetch option summary data from Deribit."""
    url = f"{base_url}/api/v2/public/get_book_summary_by_currency"
    params = {"currency": currency, "kind": "option"}

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    payload = response.json().get("result", [])
    df = pd.DataFrame(payload)
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["creation_timestamp"], unit="ms", utc=True, errors="coerce")
    df["mark_iv"] = pd.to_numeric(df.get("mark_iv"), errors="coerce")
    df["option_type"] = df["instrument_name"].astype(str).str.split("-").str[-1].str[-1].map({"C": "call", "P": "put"})
    return df


def fetch_deribit_iv_index(
    currency: str = "BTC",
    start_time: Optional[object] = None,
    end_time: Optional[object] = None,
    base_url: str = DERIBIT_BASE_URL,
) -> pd.DataFrame:
    """Fetch volatility index data from Deribit."""
    url = f"{base_url}/api/v2/public/get_volatility_index_data"
    params = {"currency": currency}
    if start_time is not None:
        params["start_timestamp"] = int(pd.Timestamp(start_time).timestamp() * 1000)
    if end_time is not None:
        params["end_timestamp"] = int(pd.Timestamp(end_time).timestamp() * 1000)

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    payload = response.json().get("result", {})
    data = payload.get("data", [])

    df = pd.DataFrame(data, columns=["timestamp", "volatility"])
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["implied_volatility"] = pd.to_numeric(df["volatility"], errors="coerce")
    return df[["timestamp", "implied_volatility"]].dropna()


def fetch_deribit_iv(
    currency: str = "BTC",
    start_time: Optional[object] = None,
    end_time: Optional[object] = None,
    base_url: str = DERIBIT_BASE_URL,
) -> pd.DataFrame:
    """Fetch implied volatility index with a snapshot fallback."""
    try:
        df = fetch_deribit_iv_index(currency, start_time=start_time, end_time=end_time, base_url=base_url)
        if not df.empty:
            return df
    except Exception:
        pass

    summaries = fetch_deribit_option_summaries(currency, base_url=base_url)
    if summaries.empty:
        return summaries

    snapshot = summaries["mark_iv"].dropna().mean()
    timestamp = pd.Timestamp.utcnow()
    return pd.DataFrame([{"timestamp": timestamp, "implied_volatility": float(snapshot)}])
