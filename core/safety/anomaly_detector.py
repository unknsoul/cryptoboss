"""
Anomaly Detector - Enterprise Safety Feature #156
Detects abnormal market conditions and triggers protective actions.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Detects abnormal market conditions that may warrant closing positions or halting trading.
    
    Anomalies detected:
    - Flash crash: Large price move in short time
    - Volume spike: Abnormally high volume
    - Spread widening: Bid-ask spread exceeds threshold
    - Volatility explosion: ATR spike beyond normal range
    """
    
    def __init__(
        self,
        flash_crash_pct: float = 3.0,        # 3% move in 1 min
        flash_crash_window: int = 60,         # seconds
        volume_spike_mult: float = 5.0,       # 5x average volume
        volume_window: int = 20,              # candles for average
        spread_threshold_pct: float = 0.3,    # 0.3% spread
        volatility_spike_mult: float = 3.0    # 3x normal volatility
    ):
        """
        Initialize anomaly detector.
        
        Args:
            flash_crash_pct: Minimum % price change to trigger flash crash
            flash_crash_window: Seconds window for flash crash detection
            volume_spike_mult: Multiplier above average for volume spike
            volume_window: Number of candles for volume average
            spread_threshold_pct: Maximum acceptable spread percentage
            volatility_spike_mult: Multiplier for volatility spike
        """
        self.flash_crash_pct = flash_crash_pct
        self.flash_crash_window = flash_crash_window
        self.volume_spike_mult = volume_spike_mult
        self.volume_window = volume_window
        self.spread_threshold_pct = spread_threshold_pct
        self.volatility_spike_mult = volatility_spike_mult
        
        # Price history for flash crash detection
        self.price_history: deque = deque(maxlen=120)  # 2 minutes of ticks
        
        # Recent anomalies
        self.recent_anomalies: List[Dict] = []
        
        logger.info(f"Anomaly Detector initialized - Flash: {flash_crash_pct}%, "
                   f"Volume: {volume_spike_mult}x, Spread: {spread_threshold_pct}%")
    
    def update_price(self, price: float, timestamp: Optional[datetime] = None):
        """Record a price tick for flash crash detection."""
        if timestamp is None:
            timestamp = datetime.now()
        self.price_history.append({'price': price, 'time': timestamp})
    
    def check_anomalies(
        self,
        current_price: float,
        candles: List[Dict],
        orderbook: Optional[Dict] = None
    ) -> Tuple[bool, List[str]]:
        """
        Check for all anomaly conditions.
        
        Args:
            current_price: Current market price
            candles: Recent candle data
            orderbook: Order book data with bids/asks
            
        Returns:
            Tuple of (has_anomaly, list of anomaly descriptions)
        """
        anomalies = []
        
        # Update price history
        self.update_price(current_price)
        
        # 1. Flash Crash Detection
        flash_crash = self._check_flash_crash(current_price)
        if flash_crash:
            anomalies.append(flash_crash)
        
        # 2. Volume Spike Detection
        volume_spike = self._check_volume_spike(candles)
        if volume_spike:
            anomalies.append(volume_spike)
        
        # 3. Spread Widening Detection
        if orderbook:
            spread_anomaly = self._check_spread(orderbook, current_price)
            if spread_anomaly:
                anomalies.append(spread_anomaly)
        
        # 4. Volatility Explosion
        vol_spike = self._check_volatility_spike(candles)
        if vol_spike:
            anomalies.append(vol_spike)
        
        # Record anomalies
        if anomalies:
            for anomaly in anomalies:
                self.recent_anomalies.append({
                    'timestamp': datetime.now().isoformat(),
                    'description': anomaly
                })
            # Keep only last 50 anomalies
            self.recent_anomalies = self.recent_anomalies[-50:]
        
        return len(anomalies) > 0, anomalies
    
    def _check_flash_crash(self, current_price: float) -> Optional[str]:
        """Detect rapid price moves."""
        if len(self.price_history) < 2:
            return None
        
        now = datetime.now()
        window_start = now - timedelta(seconds=self.flash_crash_window)
        
        # Get prices in window
        window_prices = [p['price'] for p in self.price_history if p['time'] >= window_start]
        
        if len(window_prices) < 2:
            return None
        
        max_price = max(window_prices)
        min_price = min(window_prices)
        
        if min_price <= 0:
            return None
        
        move_pct = (max_price - min_price) / min_price * 100
        
        if move_pct >= self.flash_crash_pct:
            direction = "DOWN" if current_price < window_prices[0] else "UP"
            return f"FLASH MOVE {direction}: {move_pct:.1f}% in {self.flash_crash_window}s"
        
        return None
    
    def _check_volume_spike(self, candles: List[Dict]) -> Optional[str]:
        """Detect abnormal volume."""
        if not candles or len(candles) < self.volume_window + 1:
            return None
        
        try:
            # Get recent volumes
            volumes = [c.get('volume', 0) for c in candles[-(self.volume_window + 1):]]
            
            if not volumes or volumes[-1] == 0:
                return None
            
            avg_volume = sum(volumes[:-1]) / len(volumes[:-1])
            current_volume = volumes[-1]
            
            if avg_volume <= 0:
                return None
            
            ratio = current_volume / avg_volume
            
            if ratio >= self.volume_spike_mult:
                return f"VOLUME SPIKE: {ratio:.1f}x average"
        except Exception:
            pass
        
        return None
    
    def _check_spread(self, orderbook: Dict, price: float) -> Optional[str]:
        """Detect wide bid-ask spread."""
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            if not bids or not asks:
                return None
            
            best_bid = float(bids[0][0]) if isinstance(bids[0], list) else bids[0]
            best_ask = float(asks[0][0]) if isinstance(asks[0], list) else asks[0]
            
            spread = best_ask - best_bid
            spread_pct = spread / price * 100
            
            if spread_pct >= self.spread_threshold_pct:
                return f"SPREAD WARNING: {spread_pct:.2f}% (${spread:.2f})"
        except Exception:
            pass
        
        return None
    
    def _check_volatility_spike(self, candles: List[Dict]) -> Optional[str]:
        """Detect volatility explosion."""
        if not candles or len(candles) < 25:
            return None
        
        try:
            # Calculate recent ATR vs average ATR
            recent_atr = self._calculate_atr(candles[-5:])
            avg_atr = self._calculate_atr(candles[-25:-5])
            
            if avg_atr <= 0:
                return None
            
            ratio = recent_atr / avg_atr
            
            if ratio >= self.volatility_spike_mult:
                return f"VOLATILITY SPIKE: {ratio:.1f}x normal"
        except Exception:
            pass
        
        return None
    
    def _calculate_atr(self, candles: List[Dict]) -> float:
        """Calculate Average True Range."""
        if len(candles) < 2:
            return 0
        
        trs = []
        for i in range(1, len(candles)):
            high = candles[i].get('high', 0)
            low = candles[i].get('low', 0)
            prev_close = candles[i-1].get('close', 0)
            
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)
        
        return sum(trs) / len(trs) if trs else 0
    
    def should_close_position(self, current_price: float, candles: List[Dict]) -> Tuple[bool, str]:
        """
        Check if positions should be closed due to anomalies.
        
        Returns:
            Tuple of (should_close, reason)
        """
        has_anomaly, anomalies = self.check_anomalies(current_price, candles)
        
        # Only flash crashes and volatility spikes warrant position closing
        critical_anomalies = [a for a in anomalies if 'FLASH' in a or 'VOLATILITY SPIKE' in a]
        
        if critical_anomalies:
            return True, critical_anomalies[0]
        
        return False, ""
    
    def get_status(self) -> Dict:
        """Get current anomaly detector status."""
        return {
            'recent_anomalies': self.recent_anomalies[-5:],
            'anomaly_count': len(self.recent_anomalies),
            'price_history_size': len(self.price_history)
        }


if __name__ == '__main__':
    # Test anomaly detector
    detector = AnomalyDetector(flash_crash_pct=2.0)
    
    # Simulate normal prices
    for i in range(10):
        detector.update_price(50000 + i * 10)
    
    # Simulate flash crash
    detector.update_price(48000)  # 4% drop
    
    has_anomaly, anomalies = detector.check_anomalies(48000, [])
    print(f"Anomaly detected: {has_anomaly}")
    print(f"Anomalies: {anomalies}")
