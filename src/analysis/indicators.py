"""Technical indicator engine for context-first strategy logic."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


class IndicatorEngine:
    """Compute trend, momentum, volatility, volume, and structure indicators."""

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame enriched with all core indicators."""
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Input DataFrame missing OHLCV columns: {sorted(missing)}")

        out = df.copy()

        # Trend
        out["EMA_20"] = out["close"].ewm(span=20, adjust=False).mean()
        out["EMA_50"] = out["close"].ewm(span=50, adjust=False).mean()
        out["EMA_200"] = out["close"].ewm(span=200, adjust=False).mean()
        out["SuperTrend_10_3"] = self._supertrend(out, period=10, multiplier=3.0)
        out["VWAP"] = self._vwap(out)

        tenkan = (out["high"].rolling(9).max() + out["low"].rolling(9).min()) / 2
        kijun = (out["high"].rolling(26).max() + out["low"].rolling(26).min()) / 2
        out["Ichimoku_tenkan"] = tenkan
        out["Ichimoku_kijun"] = kijun
        out["Ichimoku_span_a"] = (tenkan + kijun) / 2
        out["Ichimoku_span_b"] = (out["high"].rolling(52).max() + out["low"].rolling(52).min()) / 2

        # Momentum
        out["RSI_14"] = self._rsi(out["close"], 14)
        stoch_rsi = self._stoch_rsi(out["RSI_14"], 14)
        out["StochRSI_k"] = stoch_rsi.rolling(3).mean()
        out["StochRSI_d"] = out["StochRSI_k"].rolling(3).mean()

        macd_fast = out["close"].ewm(span=12, adjust=False).mean()
        macd_slow = out["close"].ewm(span=26, adjust=False).mean()
        out["MACD"] = macd_fast - macd_slow
        out["MACD_signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
        out["MACD_hist"] = out["MACD"] - out["MACD_signal"]

        out["MFI_14"] = self._mfi(out, period=14)
        out["CMF_20"] = self._cmf(out, period=20)

        # Volatility
        out["ATR_14"] = self._atr(out, period=14)
        bb_mid = out["close"].rolling(20).mean()
        bb_std = out["close"].rolling(20).std(ddof=0)
        out["BB_mid"] = bb_mid
        out["BB_upper"] = bb_mid + (2 * bb_std)
        out["BB_lower"] = bb_mid - (2 * bb_std)

        kc_mid = out["close"].ewm(span=20, adjust=False).mean()
        out["KC_mid"] = kc_mid
        out["KC_upper"] = kc_mid + (2 * out["ATR_14"])
        out["KC_lower"] = kc_mid - (2 * out["ATR_14"])

        log_returns = np.log(out["close"] / out["close"].shift(1))
        out["HV_20"] = log_returns.rolling(20).std(ddof=0) * np.sqrt(20)

        # Volume
        out["OBV"] = self._obv(out)
        out["VolumeProfile_24h"] = out["volume"].rolling(24).sum()
        out["CVD"] = np.sign(out["close"].diff().fillna(0.0)) * out["volume"]
        out["CVD"] = out["CVD"].cumsum()

        # Structure markers
        high_points = self.find_swing_highs(out, lookback=10)
        low_points = self.find_swing_lows(out, lookback=10)
        out["SwingHigh"] = False
        out["SwingLow"] = False
        for point in high_points:
            out.loc[point["index"], "SwingHigh"] = True
        for point in low_points:
            out.loc[point["index"], "SwingLow"] = True

        return out

    def find_swing_highs(self, df: pd.DataFrame, lookback: int = 10) -> List[Dict]:
        """Return swing high points as dictionaries."""
        highs = df["high"].reset_index(drop=True)
        swings: List[Dict] = []

        for i in range(lookback, len(highs) - lookback):
            center = highs.iloc[i]
            left = highs.iloc[i - lookback : i]
            right = highs.iloc[i + 1 : i + 1 + lookback]
            if center > left.max() and center > right.max():
                strength = float((center - max(left.max(), right.max())) / max(center, 1e-9))
                swings.append({"index": int(i), "price": float(center), "strength": strength})

        return swings

    def find_swing_lows(self, df: pd.DataFrame, lookback: int = 10) -> List[Dict]:
        """Return swing low points as dictionaries."""
        lows = df["low"].reset_index(drop=True)
        swings: List[Dict] = []

        for i in range(lookback, len(lows) - lookback):
            center = lows.iloc[i]
            left = lows.iloc[i - lookback : i]
            right = lows.iloc[i + 1 : i + 1 + lookback]
            if center < left.min() and center < right.min():
                strength = float((min(left.min(), right.min()) - center) / max(abs(center), 1e-9))
                swings.append({"index": int(i), "price": float(center), "strength": strength})

        return swings

    def find_order_blocks(self, df: pd.DataFrame) -> List[Dict]:
        """Detect simple bullish and bearish order blocks."""
        body = (df["close"] - df["open"]).abs()
        body_avg = body.rolling(20, min_periods=3).mean()
        vol_avg = df["volume"].rolling(20, min_periods=3).mean()

        blocks: List[Dict] = []
        for i in range(1, len(df)):
            impulse = body.iloc[i] > (body_avg.iloc[i] * 1.25)
            volume_confirm = df["volume"].iloc[i] >= vol_avg.iloc[i]
            if not (impulse and volume_confirm):
                continue

            direction = "BULLISH" if df["close"].iloc[i] > df["open"].iloc[i] else "BEARISH"
            search = range(i - 1, -1, -1)
            anchor_index = i - 1

            for j in search:
                is_bear = df["close"].iloc[j] < df["open"].iloc[j]
                is_bull = df["close"].iloc[j] > df["open"].iloc[j]
                if direction == "BULLISH" and is_bear:
                    anchor_index = j
                    break
                if direction == "BEARISH" and is_bull:
                    anchor_index = j
                    break

            blocks.append(
                {
                    "direction": direction,
                    "index": int(anchor_index),
                    "low": float(df["low"].iloc[anchor_index]),
                    "high": float(df["high"].iloc[anchor_index]),
                }
            )

        return blocks

    def find_fair_value_gaps(self, df: pd.DataFrame) -> List[Dict]:
        """Detect basic three-candle fair value gaps."""
        gaps: List[Dict] = []
        for i in range(1, len(df) - 1):
            prev_high = float(df["high"].iloc[i - 1])
            prev_low = float(df["low"].iloc[i - 1])
            next_high = float(df["high"].iloc[i + 1])
            next_low = float(df["low"].iloc[i + 1])

            if next_low > prev_high:
                gaps.append(
                    {
                        "type": "BULLISH",
                        "index": int(i),
                        "gap_low": prev_high,
                        "gap_high": next_low,
                    }
                )
            elif next_high < prev_low:
                gaps.append(
                    {
                        "type": "BEARISH",
                        "index": int(i),
                        "gap_low": next_high,
                        "gap_high": prev_low,
                    }
                )

        return gaps

    def compute_fibonacci_levels(self, swing_high: float, swing_low: float) -> Dict[str, float]:
        """Compute common Fibonacci retracement levels."""
        high = float(max(swing_high, swing_low))
        low = float(min(swing_high, swing_low))
        delta = high - low

        return {
            "0.236": high - (delta * 0.236),
            "0.382": high - (delta * 0.382),
            "0.500": high - (delta * 0.500),
            "0.618": high - (delta * 0.618),
            "0.786": high - (delta * 0.786),
        }

    def detect_candlestick_patterns(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Detect common candlestick patterns on the latest candles."""
        if len(df) < 2:
            return {"hammer": False, "bullish_engulfing": False, "bearish_engulfing": False, "doji": False}

        current = df.iloc[-1]
        previous = df.iloc[-2]

        body = abs(current["close"] - current["open"])
        candle_range = max(current["high"] - current["low"], 1e-9)
        lower_wick = min(current["open"], current["close"]) - current["low"]
        upper_wick = current["high"] - max(current["open"], current["close"])

        hammer = lower_wick > (body * 2) and upper_wick < body
        doji = body <= (candle_range * 0.1)

        bullish_engulfing = (
            previous["close"] < previous["open"]
            and current["close"] > current["open"]
            and current["close"] >= previous["open"]
            and current["open"] <= previous["close"]
        )
        bearish_engulfing = (
            previous["close"] > previous["open"]
            and current["close"] < current["open"]
            and current["open"] >= previous["close"]
            and current["close"] <= previous["open"]
        )

        return {
            "hammer": bool(hammer),
            "bullish_engulfing": bool(bullish_engulfing),
            "bearish_engulfing": bool(bearish_engulfing),
            "doji": bool(doji),
        }

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)

        avg_gain = up.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = down.ewm(alpha=1 / period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)

    @staticmethod
    def _stoch_rsi(rsi: pd.Series, period: int = 14) -> pd.Series:
        min_rsi = rsi.rolling(period).min()
        max_rsi = rsi.rolling(period).max()
        denominator = (max_rsi - min_rsi).replace(0, np.nan)
        return ((rsi - min_rsi) / denominator).clip(lower=0.0, upper=1.0).fillna(0.5)

    @staticmethod
    def _mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        raw_money_flow = typical_price * df["volume"]

        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0.0)
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0.0)

        pos_sum = positive_flow.rolling(period).sum()
        neg_sum = negative_flow.rolling(period).sum().replace(0, np.nan)
        money_ratio = pos_sum / neg_sum
        mfi = 100 - (100 / (1 + money_ratio))
        return mfi.fillna(50.0)

    @staticmethod
    def _cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
        high_low = (df["high"] - df["low"]).replace(0, np.nan)
        money_flow_multiplier = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / high_low
        money_flow_volume = money_flow_multiplier.fillna(0.0) * df["volume"]
        return money_flow_volume.rolling(period).sum() / df["volume"].rolling(period).sum()

    @staticmethod
    def _obv(df: pd.DataFrame) -> pd.Series:
        direction = np.sign(df["close"].diff()).fillna(0.0)
        return (direction * df["volume"]).cumsum()

    @staticmethod
    def _vwap(df: pd.DataFrame) -> pd.Series:
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        cumulative_tpv = (typical_price * df["volume"]).cumsum()
        cumulative_vol = df["volume"].replace(0, np.nan).cumsum()
        return cumulative_tpv / cumulative_vol

    def _supertrend(self, df: pd.DataFrame, period: int, multiplier: float) -> pd.Series:
        atr = self._atr(df, period=period)
        hl2 = (df["high"] + df["low"]) / 2

        basic_upper = hl2 + (multiplier * atr)
        basic_lower = hl2 - (multiplier * atr)

        final_upper = basic_upper.copy()
        final_lower = basic_lower.copy()

        for i in range(1, len(df)):
            if pd.notna(final_upper.iloc[i - 1]) and df["close"].iloc[i - 1] <= final_upper.iloc[i - 1]:
                final_upper.iloc[i] = min(basic_upper.iloc[i], final_upper.iloc[i - 1])
            if pd.notna(final_lower.iloc[i - 1]) and df["close"].iloc[i - 1] >= final_lower.iloc[i - 1]:
                final_lower.iloc[i] = max(basic_lower.iloc[i], final_lower.iloc[i - 1])

        supertrend = pd.Series(index=df.index, dtype=float)
        for i in range(1, len(df)):
            if pd.isna(supertrend.iloc[i - 1]):
                supertrend.iloc[i] = final_lower.iloc[i] if df["close"].iloc[i] > hl2.iloc[i] else final_upper.iloc[i]
            elif supertrend.iloc[i - 1] == final_upper.iloc[i - 1] and df["close"].iloc[i] <= final_upper.iloc[i]:
                supertrend.iloc[i] = final_upper.iloc[i]
            elif supertrend.iloc[i - 1] == final_upper.iloc[i - 1] and df["close"].iloc[i] > final_upper.iloc[i]:
                supertrend.iloc[i] = final_lower.iloc[i]
            elif supertrend.iloc[i - 1] == final_lower.iloc[i - 1] and df["close"].iloc[i] >= final_lower.iloc[i]:
                supertrend.iloc[i] = final_lower.iloc[i]
            else:
                supertrend.iloc[i] = final_upper.iloc[i]

        return supertrend.bfill()
