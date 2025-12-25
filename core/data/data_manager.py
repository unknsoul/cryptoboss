"""
Data Management - Enterprise Features #31, #35, #38, #42
Multi-Timeframe Data, Validation, Aggregation, and Export.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from collections import defaultdict
import json
import csv
import os

logger = logging.getLogger(__name__)


class MultiTimeframeDataManager:
    """
    Feature #31: Multi-Timeframe Data Manager
    
    Manages and synchronizes data across multiple timeframes.
    """
    
    TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']
    
    def __init__(self):
        """Initialize multi-timeframe data manager."""
        self.data: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
        self.last_update: Dict[str, datetime] = {}
        
        logger.info("Multi-Timeframe Data Manager initialized")
    
    def add_candle(self, symbol: str, timeframe: str, candle: Dict):
        """Add a candle to the specified timeframe."""
        self.data[symbol][timeframe].append(candle)
        self.data[symbol][timeframe] = self.data[symbol][timeframe][-500:]  # Keep last 500
        self.last_update[f"{symbol}_{timeframe}"] = datetime.now()
    
    def get_candles(self, symbol: str, timeframe: str, limit: int = 100) -> List[Dict]:
        """Get candles for symbol and timeframe."""
        return self.data[symbol][timeframe][-limit:]
    
    def aggregate_to_higher_tf(self, symbol: str, from_tf: str, to_tf: str) -> List[Dict]:
        """Aggregate lower timeframe to higher timeframe."""
        tf_minutes = {'1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440}
        
        from_min = tf_minutes.get(from_tf, 1)
        to_min = tf_minutes.get(to_tf, 60)
        ratio = to_min // from_min
        
        source = self.data[symbol][from_tf]
        if len(source) < ratio:
            return []
        
        aggregated = []
        for i in range(0, len(source) - ratio + 1, ratio):
            chunk = source[i:i + ratio]
            if chunk:
                agg_candle = {
                    'open': chunk[0]['open'],
                    'high': max(c['high'] for c in chunk),
                    'low': min(c['low'] for c in chunk),
                    'close': chunk[-1]['close'],
                    'volume': sum(c.get('volume', 0) for c in chunk),
                    'timestamp': chunk[0].get('timestamp')
                }
                aggregated.append(agg_candle)
        
        return aggregated
    
    def get_aligned_data(self, symbol: str, timeframes: List[str]) -> Dict[str, List[Dict]]:
        """Get aligned data across multiple timeframes."""
        result = {}
        for tf in timeframes:
            result[tf] = self.get_candles(symbol, tf)
        return result
    
    def get_mtf_signal(self, symbol: str) -> Dict:
        """Get multi-timeframe trend alignment signal."""
        trends = {}
        
        for tf in ['5m', '15m', '1h', '4h']:
            candles = self.get_candles(symbol, tf, 20)
            if len(candles) >= 10:
                closes = [c['close'] for c in candles]
                sma = sum(closes[-10:]) / 10
                trends[tf] = 'UP' if closes[-1] > sma else 'DOWN'
            else:
                trends[tf] = 'NEUTRAL'
        
        # Calculate alignment
        up_count = sum(1 for t in trends.values() if t == 'UP')
        down_count = sum(1 for t in trends.values() if t == 'DOWN')
        
        if up_count >= 3:
            alignment = 'BULLISH'
        elif down_count >= 3:
            alignment = 'BEARISH'
        else:
            alignment = 'MIXED'
        
        return {
            'trends': trends,
            'alignment': alignment,
            'confidence': max(up_count, down_count) / len(trends)
        }


class DataValidationPipeline:
    """
    Feature #35: Data Validation Pipeline
    
    Validates incoming market data for quality and consistency.
    """
    
    def __init__(self):
        """Initialize validation pipeline."""
        self.validators: List[Callable] = []
        self.error_log: List[Dict] = []
        
        # Register default validators
        self._register_default_validators()
        
        logger.info("Data Validation Pipeline initialized")
    
    def _register_default_validators(self):
        """Register default validation rules."""
        self.add_validator(self._validate_ohlc_logic)
        self.add_validator(self._validate_positive_values)
        self.add_validator(self._validate_reasonable_price)
    
    def add_validator(self, validator_fn: Callable[[Dict], tuple]):
        """Add a validation function."""
        self.validators.append(validator_fn)
    
    def _validate_ohlc_logic(self, candle: Dict) -> tuple:
        """Validate OHLC logic (high >= low, etc.)"""
        o, h, l, c = candle.get('open', 0), candle.get('high', 0), candle.get('low', 0), candle.get('close', 0)
        
        if h < l:
            return False, "High is less than Low"
        if h < o or h < c:
            return False, "High is not highest"
        if l > o or l > c:
            return False, "Low is not lowest"
        
        return True, "OK"
    
    def _validate_positive_values(self, candle: Dict) -> tuple:
        """Validate all values are positive."""
        for key in ['open', 'high', 'low', 'close', 'volume']:
            if key in candle and candle[key] < 0:
                return False, f"Negative {key} value"
        return True, "OK"
    
    def _validate_reasonable_price(self, candle: Dict) -> tuple:
        """Validate price is within reasonable range."""
        close = candle.get('close', 0)
        if close <= 0:
            return False, "Zero or negative price"
        if close > 1000000:  # BTC unlikely to exceed $1M soon
            return False, "Unreasonably high price"
        return True, "OK"
    
    def validate(self, candle: Dict) -> Dict:
        """Run all validators on a candle."""
        errors = []
        
        for validator in self.validators:
            try:
                is_valid, message = validator(candle)
                if not is_valid:
                    errors.append(message)
            except Exception as e:
                errors.append(f"Validator error: {e}")
        
        result = {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'candle': candle
        }
        
        if errors:
            self.error_log.append({
                'timestamp': datetime.now().isoformat(),
                'errors': errors,
                'candle': candle
            })
            self.error_log = self.error_log[-100:]
        
        return result
    
    def validate_batch(self, candles: List[Dict]) -> Dict:
        """Validate a batch of candles."""
        valid = []
        invalid = []
        
        for candle in candles:
            result = self.validate(candle)
            if result['is_valid']:
                valid.append(candle)
            else:
                invalid.append(result)
        
        return {
            'total': len(candles),
            'valid': len(valid),
            'invalid': len(invalid),
            'valid_candles': valid,
            'invalid_results': invalid
        }


class OHLCVAggregator:
    """
    Feature #38: OHLCV Aggregator
    
    Aggregates tick data into OHLCV candles.
    """
    
    def __init__(self, interval_seconds: int = 60):
        """
        Initialize OHLCV aggregator.
        
        Args:
            interval_seconds: Candle interval in seconds
        """
        self.interval = interval_seconds
        self.current_candle: Optional[Dict] = None
        self.completed_candles: List[Dict] = []
        self.tick_count = 0
        
        logger.info(f"OHLCV Aggregator initialized - {interval_seconds}s candles")
    
    def add_tick(self, price: float, volume: float = 0, timestamp: Optional[datetime] = None) -> Optional[Dict]:
        """
        Add a price tick and potentially complete a candle.
        
        Returns:
            Completed candle if interval passed, else None
        """
        ts = timestamp or datetime.now()
        candle_start = ts.replace(second=0, microsecond=0)
        
        # Start new candle if needed
        if self.current_candle is None or self.current_candle['timestamp'] != candle_start:
            # Complete previous candle
            completed = None
            if self.current_candle:
                self.completed_candles.append(self.current_candle)
                completed = self.current_candle
            
            # Start new candle
            self.current_candle = {
                'timestamp': candle_start,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': volume,
                'tick_count': 1
            }
            
            return completed
        
        # Update current candle
        self.current_candle['high'] = max(self.current_candle['high'], price)
        self.current_candle['low'] = min(self.current_candle['low'], price)
        self.current_candle['close'] = price
        self.current_candle['volume'] += volume
        self.current_candle['tick_count'] += 1
        self.tick_count += 1
        
        return None
    
    def get_candles(self, limit: int = 100) -> List[Dict]:
        """Get completed candles."""
        return self.completed_candles[-limit:]
    
    def get_current(self) -> Optional[Dict]:
        """Get current in-progress candle."""
        return self.current_candle


class DataExportSystem:
    """
    Feature #42: Data Export System
    
    Exports trading data in various formats.
    """
    
    def __init__(self, export_dir: str = 'exports'):
        """
        Initialize export system.
        
        Args:
            export_dir: Directory for exports
        """
        self.export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)
        
        logger.info(f"Data Export System initialized - Dir: {export_dir}")
    
    def export_trades_csv(self, trades: List[Dict], filename: str = None) -> str:
        """Export trades to CSV."""
        filename = filename or f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(self.export_dir, filename)
        
        if not trades:
            return ""
        
        fieldnames = list(trades[0].keys())
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trades)
        
        logger.info(f"Exported {len(trades)} trades to {filepath}")
        return filepath
    
    def export_candles_csv(self, candles: List[Dict], symbol: str, timeframe: str) -> str:
        """Export candles to CSV."""
        filename = f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.csv"
        filepath = os.path.join(self.export_dir, filename)
        
        fieldnames = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for candle in candles:
                row = {k: candle.get(k, '') for k in fieldnames}
                writer.writerow(row)
        
        return filepath
    
    def export_json(self, data: Any, filename: str) -> str:
        """Export any data to JSON."""
        filepath = os.path.join(self.export_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return filepath
    
    def export_performance_report(
        self,
        trades: List[Dict],
        equity_curve: List[float],
        metrics: Dict
    ) -> str:
        """Export comprehensive performance report."""
        filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': metrics,
            'trade_count': len(trades),
            'trades': trades,
            'equity_curve_sample': equity_curve[::max(1, len(equity_curve)//100)]
        }
        
        return self.export_json(report, filename)


# Singletons
_mtf_manager: Optional[MultiTimeframeDataManager] = None
_validator: Optional[DataValidationPipeline] = None
_aggregator: Optional[OHLCVAggregator] = None
_exporter: Optional[DataExportSystem] = None


def get_mtf_manager() -> MultiTimeframeDataManager:
    global _mtf_manager
    if _mtf_manager is None:
        _mtf_manager = MultiTimeframeDataManager()
    return _mtf_manager


def get_data_validator() -> DataValidationPipeline:
    global _validator
    if _validator is None:
        _validator = DataValidationPipeline()
    return _validator


def get_aggregator() -> OHLCVAggregator:
    global _aggregator
    if _aggregator is None:
        _aggregator = OHLCVAggregator()
    return _aggregator


def get_exporter() -> DataExportSystem:
    global _exporter
    if _exporter is None:
        _exporter = DataExportSystem()
    return _exporter


if __name__ == '__main__':
    # Test MTF manager
    mtf = MultiTimeframeDataManager()
    for i in range(20):
        mtf.add_candle('BTCUSDT', '5m', {
            'open': 50000 + i * 10,
            'high': 50050 + i * 10,
            'low': 49950 + i * 10,
            'close': 50020 + i * 10,
            'volume': 100
        })
    
    signal = mtf.get_mtf_signal('BTCUSDT')
    print(f"MTF Signal: {signal}")
    
    # Test validator
    validator = DataValidationPipeline()
    result = validator.validate({'open': 50000, 'high': 50100, 'low': 49900, 'close': 50050})
    print(f"Validation: {result}")
    
    # Bad data
    result = validator.validate({'open': 50000, 'high': 49000, 'low': 49900, 'close': 50050})
    print(f"Bad data: {result}")
