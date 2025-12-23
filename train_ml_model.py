"""
Training Script for ML Models
Run this to train models on real BTC data
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from core.ml.real_ml_trainer import RealMLTrainer
from core.data_manager import DataManager


def train_signal_filter():
    """Train signal filtering model on real BTC data"""
    
    print("=" * 70)
    print("TRAINING ML SIGNAL FILTER")
    print("=" * 70)
    
    # Load historical data
    print("\n1. Loading BTC data...")
    dm = DataManager()
    
    try:
        df = dm.load_data('btc_1h.csv')
        print(f"✓ Loaded {len(df)} candles")
    except:
        print("Downloading new data from Binance...")
        df = dm.download_binance_data(
            symbol='BTCUSDT',
            interval='1h',
            start_date='2023-01-01'
        )
        dm.save_data(df, 'btc_1h.csv')
        print(f"✓ Downloaded and saved {len(df)} candles")
    
    # Initialize trainer
    print("\n2. Initializing ML trainer...")
    trainer = RealMLTrainer(model_type='xgboost')  # or 'lightgbm', 'random_forest'
    
    # Prepare data
    print("\n3. Preparing training data...")
    X_train, y_train, X_test, y_test = trainer.prepare_training_data(
        df,
        forward_periods=24,  # Predict 24 hours ahead
        threshold=0.015,     # 1.5% threshold
        train_ratio=0.7
    )
    
    # Train model
    print("\n4. Training model...")
    trainer.train(X_train, y_train, X_val=X_test, y_val=y_test)
    
    # Evaluate
    print("\n5. Evaluating model...")
    metrics = trainer.evaluate(X_test, y_test)
    
    # Save model
    print("\n6. Saving model...")
    model_path = Path('models/signal_filter_production.pkl')
    trainer.save_model(model_path)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nModel saved to: {model_path}")
    print(f"Test accuracy: {metrics['accuracy']:.2%}")
    print(f"F1 Score: {metrics['f1_macro']:.4f}")
    
    print("\n📌 IMPORTANT:")
    print("  - This model is trained on historical data")
    print("  - Performance may degrade in different market conditions")
    print("  - Monitor accuracy and retrain periodically")
    print("  - Test thoroughly in paper trading before live use")
    
    return trainer


if __name__ == '__main__':
    train_signal_filter()
