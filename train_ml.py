"""
ML Training Script - Train on Historical Market Data
======================================================
This script fetches historical BTC/USDT candles from Binance,
generates features, creates labels, and trains the Ensemble Model.

Usage:
    python train_ml.py

The trained model will be saved to models/signal_filter_xgboost.pkl
and automatically loaded by run_trading_bot.py on next restart.
"""

import os
import sys
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, '.')

# Import ML components
from core.ml.ensemble_model import EnsembleModel
from core.ml.real_ml_trainer import RealMLTrainer
from core.ml.feature_engineering import AdvancedFeatureEngineer


def fetch_binance_klines(symbol: str = 'BTCUSDT', interval: str = '5m', limit: int = 1000) -> pd.DataFrame:
    """
    Fetch historical klines from Binance API.
    
    Args:
        symbol: Trading pair (default BTCUSDT)
        interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d)
        limit: Number of candles (max 1000 per request)
    
    Returns:
        DataFrame with OHLCV data
    """
    url = 'https://api.binance.com/api/v3/klines'
    
    all_data = []
    end_time = None
    total_fetched = 0
    target_candles = 5000  # Get 5000 candles for good training data
    
    print(f"Fetching {target_candles} {interval} candles for {symbol}...")
    
    while total_fetched < target_candles:
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': min(limit, target_candles - total_fetched)
        }
        if end_time:
            params['endTime'] = end_time
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
            
            all_data = data + all_data  # Prepend older data
            end_time = data[0][0] - 1  # Get candles before this
            total_fetched += len(data)
            
            print(f"  Fetched {total_fetched}/{target_candles} candles...")
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
    
    if not all_data:
        raise ValueError("No data fetched from Binance")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    # Convert types
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    # Keep only OHLCV
    df = df[['open', 'high', 'low', 'close', 'volume']]
    
    print(f"✓ Fetched {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    
    return df


def create_labels(df: pd.DataFrame, forward_periods: int = 12, threshold: float = 0.005) -> pd.Series:
    """
    Create labels based on forward returns.
    
    Labels:
        2 (BUY):  Forward return > threshold (good long opportunity)
        0 (SELL): Forward return < -threshold (good short opportunity)  
        1 (NEUTRAL): Forward return within threshold (avoid)
    
    Args:
        df: OHLCV DataFrame
        forward_periods: How many candles ahead to look (12 x 5min = 1 hour)
        threshold: Return threshold (0.5% = 0.005)
    
    Returns:
        Series with labels
    """
    # Calculate forward return
    forward_return = df['close'].pct_change(forward_periods).shift(-forward_periods)
    
    # Create labels
    labels = pd.Series(1, index=df.index)  # Default NEUTRAL
    labels[forward_return > threshold] = 2   # BUY signal was good
    labels[forward_return < -threshold] = 0  # SELL signal was good
    
    return labels


def train_model(df: pd.DataFrame, labels: pd.Series, model_path: Path):
    """
    Train the Ensemble Model on historical data.
    """
    print("\n" + "=" * 70)
    print("TRAINING ENSEMBLE MODEL")
    print("=" * 70)
    
    # Initialize feature engineer
    engineer = AdvancedFeatureEngineer()
    
    # Generate features
    print("\n[1/4] Generating features...")
    features = engineer.engineer_features(df)
    
    print(f"  Created {len(features.columns)} features")
    print(f"  Sample features: {list(features.columns[:10])}")
    
    # Align labels with features (dropna removes rows)
    common_idx = features.index.intersection(labels.dropna().index)
    X = features.loc[common_idx]
    y = labels.loc[common_idx]
    
    # Remove any remaining NaN
    valid_mask = ~X.isna().any(axis=1)
    X = X[valid_mask]
    y = y[X.index]
    
    print(f"  Final dataset: {len(X)} samples")
    
    # Label distribution
    print(f"\n  Label distribution:")
    for label, name in [(0, 'SELL'), (1, 'NEUTRAL'), (2, 'BUY')]:
        count = (y == label).sum()
        pct = count / len(y) * 100
        print(f"    {name}: {count} ({pct:.1f}%)")
    
    # Time-based train/test split (critical for time series!)
    print("\n[2/4] Splitting data (70% train, 30% test)...")
    split_idx = int(len(X) * 0.7)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Train Ensemble
    print("\n[3/4] Training Ensemble Model...")
    ensemble = EnsembleModel()
    ensemble.train(X_train, y_train, X_test, y_test)
    
    # Evaluate
    print("\n[4/4] Evaluating model...")
    results = ensemble.evaluate(X_test, y_test)
    
    # Save model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble.save(model_path)
    
    print("\n" + "=" * 70)
    print(f"✓ MODEL TRAINED AND SAVED TO: {model_path}")
    print("=" * 70)
    
    return ensemble, results


def main():
    print("=" * 70)
    print("ML TRAINING PIPELINE")
    print("=" * 70)
    print(f"Started at: {datetime.now()}")
    
    # Paths
    model_path = Path('models/signal_filter_xgboost.pkl')
    
    # Step 1: Fetch historical data
    print("\n[STEP 1] Fetching Historical Data")
    print("-" * 40)
    
    df = fetch_binance_klines(
        symbol='BTCUSDT',
        interval='5m',
        limit=1000
    )
    
    # Step 2: Create labels
    print("\n[STEP 2] Creating Labels")
    print("-" * 40)
    
    labels = create_labels(
        df,
        forward_periods=12,  # 12 x 5min = 1 hour lookahead
        threshold=0.003      # 0.3% move threshold
    )
    
    print(f"✓ Created {len(labels)} labels")
    
    # Step 3: Train model
    print("\n[STEP 3] Training Model")
    print("-" * 40)
    
    ensemble, results = train_model(df, labels, model_path)
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nModel saved to: {model_path}")
    print(f"\nEnsemble Accuracy: {results['ensemble']['accuracy']:.2%}")
    print(f"Ensemble F1 Score: {results['ensemble']['f1_macro']:.2%}")
    print("\nThe bot will automatically load this model on next restart.")
    print("Run: python run_trading_bot.py")
    print("=" * 70)


if __name__ == '__main__':
    main()
