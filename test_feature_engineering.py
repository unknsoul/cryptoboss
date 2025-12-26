"""
Test Feature Engineering Pipeline
Demonstrates systematic feature generation and consistency verification.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.data import FeatureEngine
from datetime import datetime


def generate_sample_data() -> pd.DataFrame:
    """Generate sample OHLCV data."""
    dates = pd.date_range(start='2024-01-01', periods=500, freq='1h')
    
    np.random.seed(42)
    prices = 40000 * (1 + np.random.normal(0.0001, 0.02, len(dates))).cumprod()
    
    df = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
        'high': prices * (1 + np.abs(np.random.normal(0.002, 0.003, len(dates)))),
        'low': prices * (1 - np.abs(np.random.normal(0.002, 0.003, len(dates)))),
        'close': prices,
        'volume': np.random.uniform(1000, 5000, len(dates))
    }, index=dates)
    
    return df


def main():
    """Test feature engineering pipeline."""
    print("=" * 70)
    print("FEATURE ENGINEERING PIPELINE - DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Initialize engine
    print("⚙️  Initializing Feature Engine...")
    engine = FeatureEngine()
    print(f"   Registered {len(engine.get_feature_list())} features")
    print()
    
    # Show feature categories
    metadata = engine.get_feature_metadata()
    print("📊 FEATURE CATEGORIES:")
    for category, features in metadata['categories'].items():
        print(f"   {category.capitalize()}: {len(features)} features")
    print()
    
    # Generate data
    print("📁 Generating sample data...")
    df = generate_sample_data()
    print(f"   Generated {len(df)} bars of OHLCV data")
    print()
    
    # Generate features
    print("🔧 Generating all features...")
    features_df = engine.generate_features(df)
    print(f"   Generated {len(features_df.columns)} total columns")
    print(f"   Feature columns: {len(features_df.columns) - 5}")  # Subtract OHLCV
    print()
    
    # Show sample features
    print("🔍 SAMPLE FEATURES (last 5 rows):")
    print("-" * 70)
    feature_cols = [col for col in features_df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
    sample_features = feature_cols[:5]
    print(features_df[sample_features].tail().to_string())
    print()
    
    # Save features
    print("💾 Saving features to Parquet...")
    save_path = Path('data/features/test_features.parquet')
    engine.save_features(
        features_df,
        str(save_path),
        metadata={
            'generated_at': datetime.now().isoformat(),
            'total_features': len(feature_cols),
            'symbol': 'BTCUSDT',
            'timeframe': '1h'
        }
    )
    print(f"   Saved to {save_path}")
    print()
    
    # Load features back
    print("📥 Loading features from Parquet...")
    loaded_features = engine.load_features(str(save_path))
    print(f"   Loaded {len(loaded_features)} rows")
    print()
    
    # Verify consistency
    print("✅ Verifying feature consistency...")
    # Simulate "live" data (last 100 bars)
    live_data = df.iloc[-100:]
    consistency = engine.verify_consistency(features_df, live_data)
    
    if consistency['consistent']:
        print("   ✅ PASS: Training and live features match perfectly")
    else:
        print("   ❌ FAIL: Feature mismatch detected")
        if consistency['missing_in_live']:
            print(f"      Missing in live: {consistency['missing_in_live']}")
        if consistency['extra_in_live']:
            print(f"      Extra in live: {consistency['extra_in_live']}")
        if consistency['value_mismatch']:
            print(f"      Value mismatch: {consistency['value_mismatch']}")
    print()
    
    # Feature statistics
    print("📈 FEATURE STATISTICS:")
    print("-" * 70)
    for feature in sample_features:
        values = features_df[feature].dropna()
        print(f"   {feature}:")
        print(f"      Mean:  {values.mean():.6f}")
        print(f"      Std:   {values.std():.6f}")
        print(f"      Min:   {values.min():.6f}")
        print(f"      Max:   {values.max():.6f}")
    print()
    
    print("=" * 70)
    print("✅ Feature engineering pipeline test complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
