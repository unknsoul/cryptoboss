"""
Volume Profile Analyzer
Identifies high-volume price levels for support/resistance
"""

from typing import List, Dict, Tuple
import numpy as np


class VolumeProfileAnalyzer:
    """
    UPGRADE 5: Volume Profile Analysis
    Identifies price levels with high trading volume to use as support/resistance
    """
    
    def __init__(self, num_bins: int = 50, hvn_threshold: float = 1.5):
        """
        Initialize Volume Profile Analyzer
        
        Args:
            num_bins: Number of price bins for volume distribution
            hvn_threshold: Multiplier above average to consider High Volume Node
        """
        self.num_bins = num_bins
        self.hvn_threshold = hvn_threshold
        self.volume_profile = {}
        self.poc_price = None
        self.high_volume_nodes = []
    
    def calculate_profile(self, candles: List[Dict]) -> Dict:
        """
        Calculate volume profile from candle data
        
        Args:
            candles: List of candle dicts with 'high', 'low', 'close', 'volume'
            
        Returns:
            Volume profile with POC and HVN levels
        """
        if not candles or len(candles) < 20:
            return {'poc': None, 'hvn_levels': [], 'valid': False}
        
        try:
            # Get price range
            all_highs = [c['high'] for c in candles]
            all_lows = [c['low'] for c in candles]
            
            price_min = min(all_lows)
            price_max = max(all_highs)
            price_range = price_max - price_min
            
            if price_range <= 0:
                return {'poc': None, 'hvn_levels': [], 'valid': False}
            
            # Create price bins
            bin_size = price_range / self.num_bins
            volume_at_price = {}
            
            for c in candles:
                # Distribute volume across price range of candle
                candle_high = c['high']
                candle_low = c['low']
                candle_volume = c.get('volume', 0)
                
                # Find which bins this candle touches
                start_bin = int((candle_low - price_min) / bin_size)
                end_bin = int((candle_high - price_min) / bin_size)
                
                # Distribute volume evenly across touched bins
                num_bins_touched = max(1, end_bin - start_bin + 1)
                vol_per_bin = candle_volume / num_bins_touched
                
                for bin_idx in range(start_bin, end_bin + 1):
                    bin_price = price_min + (bin_idx + 0.5) * bin_size
                    volume_at_price[bin_idx] = volume_at_price.get(bin_idx, 0) + vol_per_bin
            
            if not volume_at_price:
                return {'poc': None, 'hvn_levels': [], 'valid': False}
            
            # Find POC (Point of Control) - highest volume price level
            max_vol_bin = max(volume_at_price.keys(), key=lambda x: volume_at_price[x])
            self.poc_price = price_min + (max_vol_bin + 0.5) * bin_size
            
            # Find High Volume Nodes (above average)
            avg_volume = sum(volume_at_price.values()) / len(volume_at_price)
            threshold = avg_volume * self.hvn_threshold
            
            self.high_volume_nodes = []
            for bin_idx, vol in volume_at_price.items():
                if vol >= threshold:
                    hvn_price = price_min + (bin_idx + 0.5) * bin_size
                    self.high_volume_nodes.append({
                        'price': round(hvn_price, 2),
                        'volume': vol,
                        'strength': vol / avg_volume
                    })
            
            # Sort by strength
            self.high_volume_nodes.sort(key=lambda x: x['strength'], reverse=True)
            
            return {
                'poc': round(self.poc_price, 2),
                'hvn_levels': self.high_volume_nodes[:5],  # Top 5 HVN
                'valid': True
            }
            
        except Exception as e:
            return {'poc': None, 'hvn_levels': [], 'valid': False}
    
    def is_near_support(self, price: float, tolerance_pct: float = 0.5) -> Tuple[bool, float]:
        """
        Check if price is near a high volume node (potential support/resistance)
        
        Returns:
            (is_near_hvn, distance_pct)
        """
        if not self.high_volume_nodes:
            return False, 999
        
        for hvn in self.high_volume_nodes:
            hvn_price = hvn['price']
            distance_pct = abs(price - hvn_price) / price * 100
            
            if distance_pct <= tolerance_pct:
                return True, distance_pct
        
        return False, 999
    
    def get_nearest_hvn(self, price: float, direction: str = 'below') -> float:
        """
        Get nearest high volume node in specified direction
        
        Args:
            price: Current price
            direction: 'above' or 'below'
            
        Returns:
            Nearest HVN price or None
        """
        if not self.high_volume_nodes:
            return None
        
        if direction == 'below':
            below_hvns = [h['price'] for h in self.high_volume_nodes if h['price'] < price]
            return max(below_hvns) if below_hvns else None
        else:
            above_hvns = [h['price'] for h in self.high_volume_nodes if h['price'] > price]
            return min(above_hvns) if above_hvns else None


if __name__ == '__main__':
    # Test
    analyzer = VolumeProfileAnalyzer()
    
    # Fake candles
    candles = [
        {'high': 100, 'low': 98, 'close': 99, 'volume': 1000},
        {'high': 101, 'low': 99, 'close': 100, 'volume': 1500},
        {'high': 102, 'low': 100, 'close': 101, 'volume': 2000},
        {'high': 101, 'low': 99, 'close': 100, 'volume': 1800},
    ] * 10
    
    result = analyzer.calculate_profile(candles)
    print(f"POC: {result['poc']}")
    print(f"HVN Levels: {result['hvn_levels']}")
