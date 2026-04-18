"""Quantified SMC detectors with explicit formulas for BOS/FVG/Order Blocks."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def detect_swing_points_fractal(df: pd.DataFrame, left: int = 2, right: int = 2) -> pd.DataFrame:
    """Detect fractal swing highs and lows.

    Swing high condition:
        high[i] is max over [i-left, i+right]
    Swing low condition:
        low[i] is min over [i-left, i+right]
    """
    frame = _prepare(df)
    frame["swing_high"] = 0
    frame["swing_low"] = 0

    if len(frame) < left + right + 1:
        return frame

    for i in range(left, len(frame) - right):
        high_window = frame["high"].iloc[i - left : i + right + 1]
        low_window = frame["low"].iloc[i - left : i + right + 1]

        if float(frame["high"].iloc[i]) >= float(high_window.max()):
            frame.at[frame.index[i], "swing_high"] = 1

        if float(frame["low"].iloc[i]) <= float(low_window.min()):
            frame.at[frame.index[i], "swing_low"] = 1

    return frame


def detect_bos(
    df: pd.DataFrame,
    lookback: int = 20,
    break_threshold: float = 0.0001,
    volume_spike_mult: float = 1.5,
) -> pd.DataFrame:
    """Detect BOS with explicit close break + volume spike conditions.

    Bullish BOS:
        close > previous_high + threshold and volume > avg_volume * volume_spike_mult
    Bearish BOS:
        close < previous_low  - threshold and volume > avg_volume * volume_spike_mult
    """
    frame = _prepare(df)
    frame["bos"] = 0
    frame["bos_level"] = np.nan
    frame["bos_volume_spike"] = 0

    if len(frame) < lookback + 2:
        return frame

    volume_avg = frame["tick_volume"].rolling(20, min_periods=5).mean()

    for i in range(lookback, len(frame)):
        prev_high = float(frame["high"].iloc[i - lookback : i].max())
        prev_low = float(frame["low"].iloc[i - lookback : i].min())
        close = float(frame["close"].iloc[i])

        threshold_value = break_threshold
        volume = float(frame["tick_volume"].iloc[i])
        avg_volume = float(volume_avg.iloc[i]) if not pd.isna(volume_avg.iloc[i]) else volume
        volume_spike = volume >= avg_volume * volume_spike_mult

        if close > prev_high + threshold_value and volume_spike:
            frame.at[frame.index[i], "bos"] = 1
            frame.at[frame.index[i], "bos_level"] = prev_high
            frame.at[frame.index[i], "bos_volume_spike"] = 1
        elif close < prev_low - threshold_value and volume_spike:
            frame.at[frame.index[i], "bos"] = -1
            frame.at[frame.index[i], "bos_level"] = prev_low
            frame.at[frame.index[i], "bos_volume_spike"] = 1

    return frame


def detect_fvg(df: pd.DataFrame, min_gap: float = 0.00005) -> pd.DataFrame:
    """Detect 3-candle fair value gaps.

    Bullish FVG:
        candle1_high < candle3_low and (candle3_low - candle1_high) > min_gap
    Bearish FVG:
        candle1_low  > candle3_high and (candle1_low - candle3_high) > min_gap
    """
    frame = _prepare(df)
    frame["fvg"] = 0
    frame["fvg_top"] = np.nan
    frame["fvg_bottom"] = np.nan
    frame["fvg_gap"] = np.nan

    if len(frame) < 3:
        return frame

    for i in range(2, len(frame)):
        c1_high = float(frame["high"].iloc[i - 2])
        c1_low = float(frame["low"].iloc[i - 2])
        c3_low = float(frame["low"].iloc[i])
        c3_high = float(frame["high"].iloc[i])

        bullish_gap = c3_low - c1_high
        bearish_gap = c1_low - c3_high

        if c1_high < c3_low and bullish_gap > min_gap:
            frame.at[frame.index[i], "fvg"] = 1
            frame.at[frame.index[i], "fvg_top"] = c3_low
            frame.at[frame.index[i], "fvg_bottom"] = c1_high
            frame.at[frame.index[i], "fvg_gap"] = bullish_gap
        elif c1_low > c3_high and bearish_gap > min_gap:
            frame.at[frame.index[i], "fvg"] = -1
            frame.at[frame.index[i], "fvg_top"] = c1_low
            frame.at[frame.index[i], "fvg_bottom"] = c3_high
            frame.at[frame.index[i], "fvg_gap"] = bearish_gap

    return frame


def detect_order_block(
    df: pd.DataFrame,
    impulse_multiplier: float = 2.0,
    lookback: int = 10,
    volume_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Detect practical order blocks.

    Rule:
        If current candle body > impulse_multiplier * avg_body(lookback)
        and volume > volume_multiplier * avg_volume(lookback),
        mark previous candle as order block:
            current bullish impulse -> previous bearish candle is bullish OB
            current bearish impulse -> previous bullish candle is bearish OB
    """
    frame = _prepare(df)
    frame["ob"] = 0
    frame["ob_top"] = np.nan
    frame["ob_bottom"] = np.nan
    frame["ob_strength"] = np.nan

    body = (frame["close"] - frame["open"]).abs()
    avg_body = body.rolling(lookback, min_periods=max(3, lookback // 2)).mean()
    avg_volume = frame["tick_volume"].rolling(lookback, min_periods=max(3, lookback // 2)).mean()

    for i in range(1, len(frame)):
        if pd.isna(avg_body.iloc[i]) or pd.isna(avg_volume.iloc[i]):
            continue

        current_body = float(body.iloc[i])
        current_volume = float(frame["tick_volume"].iloc[i])
        body_threshold = float(avg_body.iloc[i]) * impulse_multiplier
        volume_threshold = float(avg_volume.iloc[i]) * volume_multiplier

        if body_threshold <= 0:
            continue

        is_impulsive = current_body > body_threshold and current_volume > volume_threshold
        if not is_impulsive:
            continue

        impulse_ratio = current_body / body_threshold
        prev_idx = frame.index[i - 1]

        bullish_impulse = float(frame["close"].iloc[i]) > float(frame["open"].iloc[i])
        bearish_impulse = float(frame["close"].iloc[i]) < float(frame["open"].iloc[i])

        prev_bearish = float(frame["close"].iloc[i - 1]) < float(frame["open"].iloc[i - 1])
        prev_bullish = float(frame["close"].iloc[i - 1]) > float(frame["open"].iloc[i - 1])

        if bullish_impulse and prev_bearish:
            frame.at[prev_idx, "ob"] = 1
            frame.at[prev_idx, "ob_top"] = float(frame["high"].iloc[i - 1])
            frame.at[prev_idx, "ob_bottom"] = float(frame["low"].iloc[i - 1])
            frame.at[prev_idx, "ob_strength"] = impulse_ratio
        elif bearish_impulse and prev_bullish:
            frame.at[prev_idx, "ob"] = -1
            frame.at[prev_idx, "ob_top"] = float(frame["high"].iloc[i - 1])
            frame.at[prev_idx, "ob_bottom"] = float(frame["low"].iloc[i - 1])
            frame.at[prev_idx, "ob_strength"] = impulse_ratio

    return frame


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy().sort_index()
    if "tick_volume" not in frame.columns:
        if "volume" in frame.columns:
            frame["tick_volume"] = frame["volume"]
        else:
            frame["tick_volume"] = 1.0
    return frame
