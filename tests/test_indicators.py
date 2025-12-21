
import numpy as np
from core.indicators import TechnicalIndicators

import pandas as pd

def test_indicators():
    print("Loading real data...")
    df = pd.read_csv("data/btc_1h.csv")
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    
    print(f"Data types: {highs.dtype}, {closes.dtype}")
    
    print("Testing EMA...")
    ema = TechnicalIndicators.ema(closes, 12)
    print(f"EMA type: {type(ema)}, shape: {ema.shape}")
    
    print("Testing MACD...")
    m, s, h = TechnicalIndicators.macd(closes, 12, 26, 9)
    print(f"MACD types: {type(m)}, {type(s)}, {type(h)}")
    
    print("Testing ATR...")
    atr = TechnicalIndicators.atr(highs, lows, closes, 14)
    print(f"ATR type: {type(atr)}")
    
    print("Testing RSI...")
    rsi = TechnicalIndicators.rsi(closes, 14)
    print(f"RSI type: {type(rsi)}")

if __name__ == "__main__":
    try:
        test_indicators()
        print("All indicators OK")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
