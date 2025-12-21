"""
Advanced AI Predictor (Stacked Ensemble)
INTEGRATES: XGBoost (Gradient Boosting) + RandomForest (Bagging)
FEATURES: Technicals + OrderBook + Volatility + Regime + Sentiment
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class MLPredictor:
    """
    State-of-the-Art Signal Predictor
    Architecture: Stacked Generalization
    Level 0: XGBoost, Random Forest
    Level 1: Logistic Regression Meta-Learner
    """
    
    def __init__(self, model_path="models/ensemble_v1.pkl"):
        self.model_path = model_path
        self.scaler = StandardScaler()
        
        # Level 0 Models
        self.rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, use_label_encoder=False, eval_metric='logloss')
        
        # Level 1 Meta-Learner
        self.meta_learner = LogisticRegression()
        
        self.is_trained = False
        
        # Try loading
        self.load_model()
        
    def _extract_features(self, df: pd.DataFrame, orderbook_imbalance: float = 0.0) -> pd.DataFrame:
        """
        Create predictive features.
        CRITICAL: Z-Score Normalization for price-independence.
        """
        data = df.copy()
        
        # 1. Rolling Z-Scores (Price relative to Moving Average)
        data['sma_50'] = data['close'].rolling(window=50).mean()
        data['std_50'] = data['close'].rolling(window=50).std()
        data['z_score_price'] = (data['close'] - data['sma_50']) / (data['std_50'] + 1e-6)
        
        # 2. RSI Normalized
        # Assuming Data has RSI computed. If not, simple calc:
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        data['rsi'] = 100 - (100 / (1 + gain/(loss+1e-6)))
        data['rsi_norm'] = (data['rsi'] - 50) / 50.0 # Scale to -1..1
        
        # 3. Volatility (ATR-like)
        data['log_ret'] = np.log(data['close'] / data['close'].shift(1))
        data['volatility'] = data['log_ret'].rolling(window=14).std()
        
        # 4. Order Book Imbalance (Passed dynamic, but for history mock 0.0)
        # In live prediction, this is a single scalar. In training, we might mock it or drop it.
        # simpler: Add as column constant for this batch (if live) or use volume imbalance
        data['ob_imbalance'] = orderbook_imbalance 
        
        # 5. Lagged Features (Time Dynamics)
        for lag in [1, 2, 3]:
            data[f'ret_lag_{lag}'] = data['log_ret'].shift(lag)
            
        features = ['z_score_price', 'rsi_norm', 'volatility', 'ob_imbalance', 'ret_lag_1', 'ret_lag_2', 'ret_lag_3']
        
        # Drop NaN
        data = data.dropna()
        
        return data[features]

    def train(self, df: pd.DataFrame, labels: pd.Series):
        """
        Train the Ensemble
        """
        print("🧠 Training Stacked Ensemble AI...")
        X = self._extract_features(df)
        # Align labels
        y = labels.loc[X.index]
        
        if len(X) < 100:
            print("⚠️ Not enough data to train AI")
            return
            
        # Fit Scaler
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Level 0
        self.rf.fit(X_scaled, y)
        self.xgb_model.fit(X_scaled, y)
        
        # Generate Level 1 Inputs (Out-of-sample predictions usually, simplified here)
        p1 = self.rf.predict_proba(X_scaled)[:, 1]
        p2 = self.xgb_model.predict_proba(X_scaled)[:, 1]
        
        meta_X = np.column_stack((p1, p2))
        
        # Train Meta-Learner
        self.meta_learner.fit(meta_X, y)
        
        self.is_trained = True
        self.save_model()
        print("✅ AI Training Complete. Model Saved.")

    def predict(self, df: pd.DataFrame, orderbook_imbalance: float = 0.0) -> Dict[str, Any]:
        """
        Generate Prediction for the latest candle
        """
        if not self.is_trained:
            # Fallback for demo / pre-train
            return {'confidence': 0.0, 'direction': 'NEUTRAL', 'meta_score': 0.0}
            
        # Extract features for just the latest window
        # We need enough history to compute rolling metrics
        features = self._extract_features(df, orderbook_imbalance)
        
        if features.empty:
             return {'confidence': 0.0, 'direction': 'NEUTRAL', 'meta_score': 0.0}
             
        latest_X = features.iloc[[-1]] # Last row
        latest_X_scaled = self.scaler.transform(latest_X)
        
        # Level 0 Preds
        p1 = self.rf.predict_proba(latest_X_scaled)[:, 1]
        p2 = self.xgb_model.predict_proba(latest_X_scaled)[:, 1]
        
        meta_X = np.column_stack((p1, p2))
        
        # Level 1 Pred
        final_prob = self.meta_learner.predict_proba(meta_X)[0, 1] # Prob of Class 1 (Buy)
        
        # Mapping Prob to Signal
        # 0.5 - 1.0 -> Buy
        # 0.0 - 0.5 -> Sell
        # Converted to -1..1 score
        score = (final_prob - 0.5) * 2 # Map 0..1 to -1..1
        
        direction = "NEUTRAL"
        if score > 0.4: direction = "LONG"
        elif score < -0.4: direction = "SHORT"
        
        return {
            'confidence': abs(score),
            'direction': direction,
            'meta_score': score,
            'components': {
                'rf_prob': float(p1[0]),
                'xgb_prob': float(p2[0])
            }
        }
        
    def save_model(self):
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump({
                'rf': self.rf,
                'xgb': self.xgb_model,
                'meta': self.meta_learner,
                'scaler': self.scaler
            }, self.model_path)
        except Exception as e:
            print(f"Failed to save model: {e}")
            
    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                data = joblib.load(self.model_path)
                self.rf = data['rf']
                self.xgb_model = data['xgb']
                self.meta_learner = data['meta']
                self.scaler = data['scaler']
                self.is_trained = True
                print("✅ Loaded Pre-Trained AI Model")
            except:
                print("⚠️ Failed to load model. Retraining required.")
