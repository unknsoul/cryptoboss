"""
Market Regime Detector
Uses Gaussian Mixture Models (Unsupervised Learning) to classify market states.
States: 0=Low Vol, 1=Bull, 2=Bear, 3=High Vol/Chop
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from typing import Dict, Any

class RegimeDetector:
    def __init__(self, lookback_period=200):
        self.lookback = lookback_period
        self.gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
        self.is_fitted = False
        self.current_regime = 0
        self.regime_map = {
            0: "LOW_VOL_RANGE",
            1: "BULL_TREND",
            2: "BEAR_TREND",
            3: "HIGH_VOL_CHOP"
        }
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for regime detection:
        1. Volatility (ATR/Price)
        2. Trend Strength (ADX)
        3. Momentum (RSI)
        4. Returns (Daily Change)
        """
        data = df.copy()
        
        # Calculate returns
        data['returns'] = data['close'].pct_change()
        
        # Simple Volatility (Rolling Std Dev of returns)
        data['volatility'] = data['returns'].rolling(window=20).std()
        
        # Returns Momentum (Rolling Mean of returns)
        data['momentum'] = data['returns'].rolling(window=20).mean()
        
        # Drop NaNs
        data = data.dropna()
        
        return data[['volatility', 'momentum']]
        
    def fit(self, df: pd.DataFrame):
        """Train the GMM model on historical data"""
        if len(df) < self.lookback:
            print("⚠️ Not enough data to fit Regime Detector")
            return
            
        features = self.prepare_features(df)
        
        if len(features) < 100:
            return
            
        self.gmm.fit(features)
        self.is_fitted = True
        
        # Map clusters to semantic meanings (Basic heuristic implementation)
        # In a full impl, we'd analyze the cluster means to assign labels dynamically
        # For now, we trust the model creates 4 distinct clusters
        print("✅ Regime Detector Fitted (4 States)")
        
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Predict the current regime"""
        if not self.is_fitted:
            return {"regime_id": 0, "regime_name": "UNKNOWN (Unfitted)"}
            
        features = self.prepare_features(df)
        if features.empty:
            return {"regime_id": 0, "regime_name": "UNKNOWN (No Data)"}
            
        current_data = features.iloc[[-1]] # Last row
        
        regime_id = self.gmm.predict(current_data)[0]
        probs = self.gmm.predict_proba(current_data)[0]
        confidence = probs[regime_id]
        
        # Remap ID for consistency if we wanted (omitted for speed)
        # Using raw cluster ID for now
        
        return {
            "regime_id": int(regime_id),
            "regime_name": f"CLUSTER_{regime_id}", # self.regime_map.get(regime_id, "UNKNOWN"),
            "confidence": float(confidence),
            "probabilities": probs.tolist()
        }
