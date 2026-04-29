"""Walk-forward ML training pipeline with triple-barrier labels."""

from __future__ import annotations

import argparse
from typing import Optional

import ccxt
import pandas as pd

from src.data.alternative.funding_rates import (
    fetch_binance_funding_rates,
    fetch_open_interest_history,
    fetch_liquidation_history,
)
from src.features.pipeline import FeaturePipeline
from src.labels.labeler import OutcomeLabeler, OutcomeLabelerConfig
from src.labels.triple_barrier import TripleBarrierConfig
from src.labels.meta_labeler import MetaLabeler
from src.models.train import MLPipeline
from src.models.registry import ModelRegistry


def fetch_ohlcv_ccxt(
    symbol: str,
    timeframe: str,
    total_bars: int,
    exchange_id: str = "binance",
) -> pd.DataFrame:
    """Fetch OHLCV data via ccxt with pagination."""
    exchange = getattr(ccxt, exchange_id)()
    limit = 1000
    since = None
    rows = []

    while len(rows) < total_bars:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=min(limit, total_bars - len(rows)))
        if not batch:
            break
        rows.extend(batch)
        since = batch[-1][0] + 1
        if len(batch) < limit:
            break

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def build_feature_set(
    ohlcv: pd.DataFrame,
    symbol: str,
    include_alternative: bool = True,
) -> pd.DataFrame:
    pipeline = FeaturePipeline()

    funding_rates = fetch_binance_funding_rates(symbol, limit=3000) if include_alternative else None
    open_interest = fetch_open_interest_history(symbol, limit=500) if include_alternative else None
    liquidations = fetch_liquidation_history(symbol, limit=500) if include_alternative else None

    return pipeline.build_features(
        ohlcv,
        funding_rates=funding_rates,
        open_interest=open_interest,
        liquidations=liquidations,
    )


def train_models(features: pd.DataFrame, labels: pd.Series, model_type: str) -> dict:
    pipeline = MLPipeline(model_type=model_type)

    walk_forward = pipeline.train_walk_forward_purged(
        features,
        labels,
        train_size=2000,
        test_size=400,
        step_size=200,
        gap=200,
        embargo=100,
    )

    model, scaler, feature_cols = pipeline.train_final_model(features, labels)

    meta_labeler = MetaLabeler()
    valid_mask = ~(features[feature_cols].isna().any(axis=1) | labels.isna())
    X_valid = features.loc[valid_mask, feature_cols]
    y_valid = labels.loc[valid_mask]
    primary_proba = model.predict_proba(scaler.transform(X_valid))[:, 1]
    meta_report = meta_labeler.fit(X_valid, primary_proba, y_valid)

    return {
        "walk_forward": walk_forward,
        "primary_model": model,
        "primary_scaler": scaler,
        "feature_cols": feature_cols,
        "meta_model": meta_labeler.model,
        "meta_report": meta_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CryptoBoss ML training v2")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--bars", type=int, default=30000)
    parser.add_argument("--model", default="xgboost", choices=["xgboost", "randomforest"])
    args = parser.parse_args()

    symbol = args.symbol
    symbol_api = symbol.replace("/", "")

    ohlcv = fetch_ohlcv_ccxt(symbol, args.timeframe, total_bars=args.bars)
    features = build_feature_set(ohlcv, symbol_api, include_alternative=True)

    labeler = OutcomeLabeler(
        OutcomeLabelerConfig(
            method="triple_barrier",
            triple_barrier=TripleBarrierConfig(tp_pct=0.02, sl_pct=0.01, max_holding_bars=50, side="both"),
        )
    )
    labels = labeler.create_labels(features)

    results = train_models(features, labels, args.model)

    registry = ModelRegistry()
    primary_id = registry.save_model(
        results["primary_model"],
        model_name="primary_signal_model",
        features=results["feature_cols"],
        scaler=results["primary_scaler"],
        metadata={
            "walk_forward_periods": results["walk_forward"].periods,
            "walk_forward_accuracy": results["walk_forward"].avg_accuracy,
            "symbol": symbol,
            "timeframe": args.timeframe,
        },
    )

    meta_feature_cols = results["feature_cols"] + ["primary_confidence"]
    registry.save_model(
        results["meta_model"],
        model_name="meta_filter_model",
        features=meta_feature_cols,
        scaler=None,
        metadata={
            "meta_auc": results["meta_report"].auc,
            "symbol": symbol,
            "timeframe": args.timeframe,
        },
    )

    print("Primary model saved:", primary_id)
    print("Walk-forward accuracy:", results["walk_forward"].avg_accuracy)


if __name__ == "__main__":
    main()
