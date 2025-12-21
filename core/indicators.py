"""
Comprehensive Technical Indicators Library
Professional-grade indicators for trading strategies
"""

import numpy as np


class TechnicalIndicators:
    """
    Complete technical indicators library with all common indicators
    """
    
    @staticmethod
    def ema(prices, period):
        """Calculate Exponential Moving Average"""
        prices = np.array(prices, dtype=float)
        ema_values = np.zeros_like(prices)
        multiplier = 2 / (period + 1)
        
        # Start with SMA
        ema_values[period - 1] = np.mean(prices[:period])
        
        # Calculate EMA
        for i in range(period, len(prices)):
            ema_values[i] = (prices[i] - ema_values[i-1]) * multiplier + ema_values[i-1]
        
        return ema_values
    
    @staticmethod
    def sma(prices, period):
        """Calculate Simple Moving Average"""
        prices = np.array(prices, dtype=float)
        sma_values = np.zeros_like(prices)
        sma_values[:period-1] = np.nan
        
        for i in range(period - 1, len(prices)):
            sma_values[i] = np.mean(prices[i - period + 1:i + 1])
        
        return sma_values
    
    @staticmethod
    def atr(highs, lows, closes, period=14):
        """Calculate Average True Range"""
        highs = np.array(highs, dtype=float)
        lows = np.array(lows, dtype=float)
        closes = np.array(closes, dtype=float)
        
        tr = np.zeros(len(closes))
        tr[0] = highs[0] - lows[0]
        
        for i in range(1, len(closes)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            tr[i] = max(hl, hc, lc)
        
        atr_values = np.zeros_like(tr)
        atr_values[period - 1] = np.mean(tr[:period])
        
        for i in range(period, len(tr)):
            atr_values[i] = (atr_values[i-1] * (period - 1) + tr[i]) / period
        
        return atr_values
    
    @staticmethod
    def bollinger_bands(prices, period=20, std_dev=2):
        """
        Calculate Bollinger Bands
        Returns: (upper_band, middle_band, lower_band, bandwidth, %b)
        """
        prices = np.array(prices, dtype=float)
        middle_band = TechnicalIndicators.sma(prices, period)
        
        std = np.zeros_like(prices)
        for i in range(period - 1, len(prices)):
            std[i] = np.std(prices[i - period + 1:i + 1])
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        # Bandwidth (volatility indicator)
        bandwidth = (upper_band - lower_band) / middle_band
        
        # %B (position within bands)
        percent_b = (prices - lower_band) / (upper_band - lower_band)
        
        return upper_band, middle_band, lower_band, bandwidth, percent_b
    
    @staticmethod
    def macd(prices, fast=12, slow=26, signal=9):
        """
        Calculate MACD
        Returns: (macd_line, signal_line, histogram)
        """
        prices = np.array(prices, dtype=float)
        
        # Calculate EMAs
        ema_fast = TechnicalIndicators.ema(prices, fast)
        ema_slow = TechnicalIndicators.ema(prices, slow)
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        
        # Histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def rsi(prices, period=14):
        """Calculate Relative Strength Index"""
        prices = np.array(prices, dtype=float)
        deltas = np.diff(prices)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.zeros(len(prices))
        avg_loss = np.zeros(len(prices))
        
        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])
        
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def adx(highs, lows, closes, period=14):
        """
        Calculate ADX (Average Directional Index)
        Returns: (adx_value, plus_di, minus_di)
        """
        highs = np.array(highs)
        lows = np.array(lows)
        closes = np.array(closes)
        
        plus_dm = np.zeros(len(closes))
        minus_dm = np.zeros(len(closes))
        
        for i in range(1, len(closes)):
            high_diff = highs[i] - highs[i-1]
            low_diff = lows[i-1] - lows[i]
            
            if high_diff > low_diff and high_diff > 0:
                plus_dm[i] = high_diff
            if low_diff > high_diff and low_diff > 0:
                minus_dm[i] = low_diff
        
        atr_vals = TechnicalIndicators.atr(highs, lows, closes, period)
        
        plus_di = 100 * TechnicalIndicators.ema(plus_dm, period) / (atr_vals + 1e-10)
        minus_di = 100 * TechnicalIndicators.ema(minus_dm, period) / (atr_vals + 1e-10)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx_value = TechnicalIndicators.ema(dx, period)
        
        return adx_value, plus_di, minus_di
    
    @staticmethod
    def stochastic(highs, lows, closes, k_period=14, d_period=3):
        """
        Calculate Stochastic Oscillator
        Returns: (%K, %D)
        """
        highs = np.array(highs)
        lows = np.array(lows)
        closes = np.array(closes)
        
        k_values = np.zeros(len(closes))
        
        for i in range(k_period - 1, len(closes)):
            highest_high = np.max(highs[i - k_period + 1:i + 1])
            lowest_low = np.min(lows[i - k_period + 1:i + 1])
            
            if highest_high != lowest_low:
                k_values[i] = 100 * (closes[i] - lowest_low) / (highest_high - lowest_low)
            else:
                k_values[i] = 50
        
        d_values = TechnicalIndicators.sma(k_values, d_period)
        
        return k_values, d_values
    
    @staticmethod
    def squeeze_detector(highs, lows, closes, bb_period=20, kc_period=20, kc_mult=1.5):
        """
        Detect Bollinger Band Squeeze (low volatility)
        Returns: True when in squeeze, False otherwise
        """
        # Calculate Bollinger Bands
        bb_upper, bb_middle, bb_lower, _, _ = TechnicalIndicators.bollinger_bands(closes, bb_period)
        
        # Calculate Keltner Channels
        atr_vals = TechnicalIndicators.atr(highs, lows, closes, kc_period)
        ema_vals = TechnicalIndicators.ema(closes, kc_period)
        kc_upper = ema_vals + (kc_mult * atr_vals)
        kc_lower = ema_vals - (kc_mult * atr_vals)
        
        # Squeeze occurs when BB is inside KC
        squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)
        
        return squeeze
    
    @staticmethod
    def volume_ma(volumes, period=20):
        """Calculate Volume Moving Average"""
        return TechnicalIndicators.sma(volumes, period)
    
    @staticmethod
    def volatility_percentile(current_atr, atr_history, lookback=100):
        """
        Calculate what percentile current ATR is in historical distribution
        Returns: 0-100, where higher = more volatile than usual
        """
        if len(atr_history) < lookback:
            lookback = len(atr_history)
        
        recent_atr = atr_history[-lookback:]
        percentile = (np.sum(recent_atr <= current_atr) / lookback) * 100
        
        return percentile
    
    @staticmethod
    def donchian_channels(highs, lows, period=20):
        """
        Calculate Donchian Channels
        Returns: (upper_channel, lower_channel, middle_channel)
        """
        highs = np.array(highs)
        lows = np.array(lows)
        
        upper_channel = np.zeros(len(highs))
        lower_channel = np.zeros(len(lows))
        
        for i in range(period - 1, len(highs)):
            upper_channel[i] = np.max(highs[i - period + 1:i + 1])
            lower_channel[i] = np.min(lows[i - period + 1:i + 1])
        
        middle_channel = (upper_channel + lower_channel) / 2
        
        return upper_channel, lower_channel, middle_channel
    
    @staticmethod
    def keltner_channels(highs, lows, closes, period=20, multiplier=2):
        """
        Calculate Keltner Channels
        Returns: (upper_channel, middle_channel, lower_channel)
        """
        atr_vals = TechnicalIndicators.atr(highs, lows, closes, period)
        ema_vals = TechnicalIndicators.ema(closes, period)
        
        upper_channel = ema_vals + (multiplier * atr_vals)
        lower_channel = ema_vals - (multiplier * atr_vals)
        
        return upper_channel, ema_vals, lower_channel
