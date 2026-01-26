"""
Advanced Analysis - Enterprise Features #50, #55, #59, #64
Volume Profile, Market Microstructure, Liquidity, Order Flow.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


class VolumeProfileAnalysis:
    """
    Feature #50: Volume Profile Analysis
    
    Analyzes volume distribution at price levels.
    """
    
    def __init__(self, num_bins: int = 50):
        """
        Initialize volume profile.
        
        Args:
            num_bins: Number of price bins
        """
        self.num_bins = num_bins
        self.profiles: Dict[str, Dict] = {}
        
        logger.info(f"Volume Profile initialized - Bins: {num_bins}")
    
    def build_profile(self, symbol: str, candles: List[Dict]) -> Dict:
        """
        Build volume profile from candle data.
        
        Args:
            symbol: Trading pair
            candles: OHLCV candle data
            
        Returns:
            Volume profile data
        """
        if not candles:
            return {}
        
        # Find price range
        all_prices = []
        for c in candles:
            all_prices.extend([c['high'], c['low']])
        
        min_price = min(all_prices)
        max_price = max(all_prices)
        bin_size = (max_price - min_price) / self.num_bins
        
        # Initialize bins
        bins = {i: {'volume': 0, 'trades': 0, 'price': min_price + i * bin_size}
                for i in range(self.num_bins)}
        
        # Distribute volume across bins
        for candle in candles:
            vol_per_level = candle.get('volume', 0) / max(1, int((candle['high'] - candle['low']) / bin_size) + 1)
            
            for price in [candle['open'], candle['high'], candle['low'], candle['close']]:
                bin_idx = min(self.num_bins - 1, int((price - min_price) / bin_size))
                bins[bin_idx]['volume'] += vol_per_level / 4
                bins[bin_idx]['trades'] += 1
        
        # Find POC (Point of Control)
        poc_bin = max(bins.items(), key=lambda x: x[1]['volume'])
        
        # Find Value Area (70% of volume)
        total_vol = sum(b['volume'] for b in bins.values())
        sorted_bins = sorted(bins.items(), key=lambda x: x[1]['volume'], reverse=True)
        
        va_vol = 0
        va_bins = []
        for idx, data in sorted_bins:
            va_bins.append(idx)
            va_vol += data['volume']
            if va_vol >= total_vol * 0.7:
                break
        
        va_high = max(bins[i]['price'] for i in va_bins) + bin_size
        va_low = min(bins[i]['price'] for i in va_bins)
        
        profile = {
            'symbol': symbol,
            'bins': bins,
            'poc': {'price': poc_bin[1]['price'], 'volume': poc_bin[1]['volume']},
            'value_area': {'high': va_high, 'low': va_low},
            'total_volume': total_vol,
            'price_range': {'min': min_price, 'max': max_price}
        }
        
        self.profiles[symbol] = profile
        return profile
    
    def get_support_resistance(self, symbol: str) -> Dict:
        """Get volume-based S/R levels."""
        profile = self.profiles.get(symbol)
        if not profile:
            return {}
        
        # High volume nodes = S/R
        bins = profile['bins']
        avg_vol = sum(b['volume'] for b in bins.values()) / len(bins)
        
        hvn = [bins[i]['price'] for i, b in bins.items() if b['volume'] > avg_vol * 1.5]
        lvn = [bins[i]['price'] for i, b in bins.items() if b['volume'] < avg_vol * 0.5]
        
        return {
            'high_volume_nodes': hvn[:5],
            'low_volume_nodes': lvn[:5],
            'poc': profile['poc']['price'],
            'value_area': profile['value_area']
        }


class MarketMicrostructure:
    """
    Feature #55: Market Microstructure
    
    Analyzes market microstructure patterns.
    """
    
    def __init__(self):
        """Initialize microstructure analyzer."""
        self.tick_data: List[Dict] = []
        
        logger.info("Market Microstructure initialized")
    
    def add_tick(self, price: float, volume: float, side: str, timestamp: datetime):
        """Add a tick for analysis."""
        self.tick_data.append({
            'price': price,
            'volume': volume,
            'side': side,
            'timestamp': timestamp.isoformat()
        })
        self.tick_data = self.tick_data[-10000:]
    
    def analyze_spread_dynamics(self, bids: List[List], asks: List[List]) -> Dict:
        """Analyze bid-ask spread dynamics."""
        if not bids or not asks:
            return {}
        
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        spread = best_ask - best_bid
        mid = (best_bid + best_ask) / 2
        
        # Calculate depth-weighted mid
        bid_depth = sum(b[1] for b in bids[:5])
        ask_depth = sum(a[1] for a in asks[:5])
        weighted_mid = (best_bid * ask_depth + best_ask * bid_depth) / (bid_depth + ask_depth) if (bid_depth + ask_depth) > 0 else mid
        
        return {
            'spread': spread,
            'spread_bps': round(spread / mid * 10000, 2),
            'mid_price': mid,
            'weighted_mid': weighted_mid,
            'bid_depth_5': bid_depth,
            'ask_depth_5': ask_depth,
            'depth_imbalance': round((bid_depth - ask_depth) / (bid_depth + ask_depth), 3) if (bid_depth + ask_depth) > 0 else 0
        }
    
    def calculate_kyle_lambda(self) -> float:
        """Calculate Kyle's Lambda (price impact coefficient)."""
        if len(self.tick_data) < 100:
            return 0
        
        # Simplified: price change per unit volume
        price_changes = []
        for i in range(1, len(self.tick_data)):
            dp = self.tick_data[i]['price'] - self.tick_data[i-1]['price']
            vol = self.tick_data[i]['volume']
            if vol > 0:
                price_changes.append(abs(dp) / vol)
        
        return round(sum(price_changes) / len(price_changes), 6) if price_changes else 0
    
    def get_trade_intensity(self, window_seconds: int = 60) -> Dict:
        """Calculate trade intensity metrics."""
        if not self.tick_data:
            return {}
        
        recent = self.tick_data[-1000:]
        
        buy_vol = sum(t['volume'] for t in recent if t['side'] == 'BUY')
        sell_vol = sum(t['volume'] for t in recent if t['side'] == 'SELL')
        
        return {
            'buy_volume': buy_vol,
            'sell_volume': sell_vol,
            'net_volume': buy_vol - sell_vol,
            'buy_ratio': round(buy_vol / (buy_vol + sell_vol), 3) if (buy_vol + sell_vol) > 0 else 0.5,
            'trade_count': len(recent)
        }


class LiquidityAnalysis:
    """
    Feature #59: Liquidity Analysis
    
    Analyzes market liquidity conditions.
    """
    
    def __init__(self):
        """Initialize liquidity analyzer."""
        self.snapshots: List[Dict] = []
        
        logger.info("Liquidity Analysis initialized")
    
    def analyze_orderbook(self, bids: List[List], asks: List[List]) -> Dict:
        """Analyze order book liquidity."""
        if not bids or not asks:
            return {'liquidity_score': 0}
        
        # Calculate liquidity at different depths
        liquidity_1pct = 0
        liquidity_2pct = 0
        mid = (bids[0][0] + asks[0][0]) / 2
        
        for bid in bids:
            if bid[0] >= mid * 0.99:
                liquidity_1pct += bid[1] * bid[0]
            if bid[0] >= mid * 0.98:
                liquidity_2pct += bid[1] * bid[0]
        
        for ask in asks:
            if ask[0] <= mid * 1.01:
                liquidity_1pct += ask[1] * ask[0]
            if ask[0] <= mid * 1.02:
                liquidity_2pct += ask[1] * ask[0]
        
        # Liquidity score (0-100)
        score = min(100, liquidity_1pct / 1000)  # Normalize
        
        return {
            'liquidity_1pct': round(liquidity_1pct, 2),
            'liquidity_2pct': round(liquidity_2pct, 2),
            'liquidity_score': round(score, 1),
            'bid_levels': len(bids),
            'ask_levels': len(asks),
            'mid_price': mid
        }
    
    def calculate_slippage_estimate(self, size: float, side: str, orderbook: Dict) -> Dict:
        """Estimate slippage for a given order size."""
        if side == 'BUY':
            levels = orderbook.get('asks', [])
        else:
            levels = orderbook.get('bids', [])
        
        if not levels:
            return {'slippage_pct': 0}
        
        remaining = size
        total_cost = 0
        
        for price, qty in levels:
            fill = min(remaining, qty)
            total_cost += fill * price
            remaining -= fill
            if remaining <= 0:
                break
        
        if size > 0:
            avg_price = total_cost / size
            reference = levels[0][0]
            slippage = abs(avg_price - reference) / reference * 100
        else:
            slippage = 0
        
        return {
            'order_size': size,
            'avg_fill_price': round(total_cost / size, 2) if size > 0 else 0,
            'slippage_pct': round(slippage, 4),
            'remaining_unfilled': remaining
        }


class OrderFlowAnalysis:
    """
    Feature #64: Order Flow Analysis
    
    Analyzes order flow for trading signals.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize order flow analyzer.
        
        Args:
            window_size: Analysis window size
        """
        self.window_size = window_size
        self.trades: List[Dict] = []
        self.delta_history: List[float] = []
        
        logger.info("Order Flow Analysis initialized")
    
    def add_trade(self, price: float, size: float, aggressor: str, timestamp: datetime):
        """Add a trade for analysis."""
        self.trades.append({
            'price': price,
            'size': size,
            'aggressor': aggressor,  # 'buyer' or 'seller'
            'timestamp': timestamp.isoformat()
        })
        self.trades = self.trades[-5000:]
    
    def calculate_delta(self) -> Dict:
        """Calculate volume delta (buy - sell volume)."""
        recent = self.trades[-self.window_size:]
        
        buy_vol = sum(t['size'] for t in recent if t['aggressor'] == 'buyer')
        sell_vol = sum(t['size'] for t in recent if t['aggressor'] == 'seller')
        delta = buy_vol - sell_vol
        
        self.delta_history.append(delta)
        self.delta_history = self.delta_history[-1000:]
        
        return {
            'delta': round(delta, 4),
            'buy_volume': round(buy_vol, 4),
            'sell_volume': round(sell_vol, 4),
            'imbalance_ratio': round(buy_vol / sell_vol, 3) if sell_vol > 0 else float('inf'),
            'signal': 'bullish' if delta > 0 else 'bearish'
        }
    
    def detect_absorption(self, threshold: float = 2.0) -> Dict:
        """Detect absorption patterns (large volume with small price change)."""
        if len(self.trades) < 50:
            return {'absorption_detected': False}
        
        recent = self.trades[-50:]
        
        total_vol = sum(t['size'] for t in recent)
        price_change = abs(recent[-1]['price'] - recent[0]['price'])
        
        # High volume with low price change = absorption
        vol_per_tick = total_vol / (price_change + 0.01)
        
        is_absorption = vol_per_tick > threshold
        
        return {
            'absorption_detected': is_absorption,
            'volume_per_tick': round(vol_per_tick, 2),
            'total_volume': round(total_vol, 4),
            'price_change': round(price_change, 2)
        }
    
    def get_cumulative_delta(self) -> Dict:
        """Get cumulative delta analysis."""
        if not self.delta_history:
            return {}
        
        cumulative = sum(self.delta_history)
        trend = 'accumulation' if cumulative > 0 else 'distribution'
        
        return {
            'cumulative_delta': round(cumulative, 4),
            'trend': trend,
            'recent_delta': round(self.delta_history[-1], 4) if self.delta_history else 0,
            'delta_history_size': len(self.delta_history)
        }


# Singletons
_volume_profile: Optional[VolumeProfileAnalysis] = None
_microstructure: Optional[MarketMicrostructure] = None
_liquidity: Optional[LiquidityAnalysis] = None
_orderflow: Optional[OrderFlowAnalysis] = None


def get_volume_profile() -> VolumeProfileAnalysis:
    global _volume_profile
    if _volume_profile is None:
        _volume_profile = VolumeProfileAnalysis()
    return _volume_profile


def get_microstructure() -> MarketMicrostructure:
    global _microstructure
    if _microstructure is None:
        _microstructure = MarketMicrostructure()
    return _microstructure


def get_liquidity_analyzer() -> LiquidityAnalysis:
    global _liquidity
    if _liquidity is None:
        _liquidity = LiquidityAnalysis()
    return _liquidity


def get_orderflow() -> OrderFlowAnalysis:
    global _orderflow
    if _orderflow is None:
        _orderflow = OrderFlowAnalysis()
    return _orderflow


if __name__ == '__main__':
    # Test volume profile
    vp = VolumeProfileAnalysis()
    candles = [
        {'open': 50000, 'high': 50100, 'low': 49900, 'close': 50050, 'volume': 100}
        for _ in range(50)
    ]
    profile = vp.build_profile('BTCUSDT', candles)
    print(f"POC: {profile['poc']}")
    
    # Test order flow
    of = OrderFlowAnalysis()
    for i in range(100):
        of.add_trade(50000 + i, 0.1, 'buyer' if i % 2 == 0 else 'seller', datetime.now())
    print(f"Delta: {of.calculate_delta()}")
