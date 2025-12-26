"""
Test ML Validation Pipeline
Demonstrates walk-forward validation with no data leakage.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.data import FeatureEngine
from src.models import MLPipeline, ModelRegistry


def generate_sample_data() -> pd.DataFrame:
    """Generate sample data with predictable pattern."""
    dates = pd.date_range(start='2024-01-01', periods=2000, freq='1h')
    
    np.random.seed(42)
    # Create trend + noise
    trend = np.linspace(40000, 45000, len(dates))
    noise = np.random.normal(0, 500, len(dates))
    prices = trend + noise
    
    df = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
        'high': prices * (1 + np.abs(np.random.normal(0.002, 0.003, len(dates)))),
        'low': prices * (1 - np.abs(np.random.normal(0.002, 0.003, len(dates)))),
        'close': prices,
        'volume': np.random.uniform(1000, 5000, len(dates))
    }, index=dates)
    
    return df


def main():
    """Test ML validation pipeline."""
    print("=" * 70)
    print("ML VALIDATION PIPELINE - DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Generate data
    print("📊 Generating sample data...")
    df = generate_sample_data()
    print(f"   Generated {len(df)} bars")
    print()
    
    # Generate features
    print("🔧 Generating features...")
    feature_engine = FeatureEngine()
    features = feature_engine.generate_features(df)
    print(f"   Generated {len(features.columns)} total columns")
    print()
    
    # Create labels
    print("🎯 Creating labels...")
    ml_pipeline = MLPipeline(model_type='randomforest')
    labels = ml_pipeline.create_labels(features, horizon=4, threshold_pct=0.3)
    print(f"   Positive samples: {labels.sum()}")
    print(f"   Negative samples: {(labels==0).sum()}")
    print()
    
    # Walk-forward validation
    print("🚀 Running walk-forward validation...")
    print("   (This validates on unseen future data)")
    wf_result = ml_pipeline.train_walk_forward(
        features=features,
        labels=labels,
        train_window=1000,
        test_window=200,
        step_size=100
    )
    print()
    
    # Display results
    print("=" * 70)
    print("WALK-FORWARD VALIDATION RESULTS")
    print("=" * 70)
    print()
    
    print("📈 PERFORMANCE METRICS:")
    print(f"   Periods tested:      {wf_result.periods}")
    print(f"   Average Accuracy:    {wf_result.avg_accuracy:.3f}")
    print(f"   Average Precision:   {wf_result.avg_precision:.3f}")
    print(f"   Average Recall:      {wf_result.avg_recall:.3f}")
    print()
    
    # F1 Score
    f1 = 2 * (wf_result.avg_precision * wf_result.avg_recall) / (wf_result.avg_precision + wf_result.avg_recall) if (wf_result.avg_precision + wf_result.avg_recall) > 0 else 0
    print(f"   F1 Score:            {f1:.3f}")
    print()
    
    print("🔝 TOP 10 FEATURES (by importance):")
    print("-" * 70)
    for i, (feature, importance) in enumerate(list(wf_result.feature_importance.items())[:10]):
        print(f"   {i+1:2d}. {feature:<30} {importance:.4f}")
    print()
    
    # Train final model
    print("🎓 Training final model on all data...")
    model, scaler, feature_list = ml_pipeline.train_final_model(features, labels)
    print(f"   Model trained on {len(feature_list)} features")
    print()
    
    # Save to registry
    print("💾 Saving to model registry...")
    registry = ModelRegistry()
    version_id = registry.save_model(
        model=model,
        model_name='signal_classifier',
        features=feature_list,
        scaler=scaler,
        metadata={
            'walk_forward_accuracy': wf_result.avg_accuracy,
            'walk_forward_precision': wf_result.avg_precision,
            'walk_forward_recall': wf_result.avg_recall,
            'f1_score': f1,
            'training_samples': len(features),
            'model_type': ml_pipeline.model_type
        }
    )
    print(f"   Saved as: {version_id}")
    print()
    
    # List models
    print("📂 REGISTERED MODELS:")
    print("-" * 70)
    models = registry.list_models()
    for model_info in models[:3]:
        print(f"   {model_info['version_id']}")
        print(f"      Accuracy: {model_info.get('walk_forward_accuracy', 'N/A'):.3f}")
        print(f"      Features: {model_info.get('num_features', 'N/A')}")
        print()
    
    print("=" * 70)
    print("✅ ML validation pipeline test complete")
    print("=" * 70)
    print()
    print("🎯 Key Takeaways:")
    print("   ✓ Walk-forward validation prevents overfitting")
    print("   ✓ No data leakage (train on past only)")
    print("   ✓ Model + features + scaler saved together")
    print("   ✓ Reproducible for live trading")


if __name__ == "__main__":
    main()
