"""
Live Signal Engine
Real-time strategy execution and signal generation
"""

import logging

logger = logging.getLogger(__name__)


class LiveSignalEngine:
    """
    Executes strategies in real-time and generates trading signals
    
    Features:
    - Multi-strategy support
    - Confidence-based signal selection
    - Signal filtering and validation
    """
    
    def __init__(self, strategies, data_manager, min_candles=200):
        """
        Initialize signal engine
        
        Args:
            strategies: List of strategy instances
            data_manager: LiveDataManager instance
            min_candles: Minimum candles required before trading
        """
        self.strategies = strategies if isinstance(strategies, list) else [strategies]
        self.data_manager = data_manager
        self.min_candles = min_candles
        
        self.signal_history = []
        
        logger.info(f"Signal engine initialized with {len(self.strategies)} strategies")
    
    def on_candle_close(self, interval):
        """
        Called when new candle closes - generate signals
        
        Args:
            interval: Timeframe that closed
        
        Returns:
            Signal dict or None
        """
        # Check if enough data
        if not self.data_manager.is_ready(interval, self.min_candles):
            logger.debug(f"Not enough data for {interval} "
                        f"({self.data_manager.get_buffer_size(interval)}/{self.min_candles})")
            return None
        
        # Get latest data
        data = self.data_manager.get_data(interval, lookback=500)
        if not data:
            return None
        
        # Run all strategies
        signals = []
        for strategy in self.strategies:
            try:
                signal = strategy.signal(
                    data['highs'],
                    data['lows'],
                    data['closes'],
                    data['volumes']
                )
                
                if signal:
                    signal['strategy_name'] = strategy.name if hasattr(strategy, 'name') else strategy.__class__.__name__
                    signal['interval'] = interval
                    signals.append(signal)
                    
                    logger.info(f"📊 {signal['strategy_name']} generated {signal['action']} signal")
            
            except Exception as e:
                logger.error(f"Error in strategy {strategy.__class__.__name__}: {e}")
        
        # Select best signal (highest confidence)
        if signals:
            best_signal = self._select_best_signal(signals)
            self.signal_history.append({
                'timestamp': data['timestamps'][-1],
                'signal': best_signal,
                'interval': interval
            })
            return best_signal
        
        return None
    
    def _select_best_signal(self, signals):
        """
        Select best signal from multiple strategies
        
        Args:
            signals: List of signal dicts
        
        Returns:
            Best signal (highest confidence)
        """
        if len(signals) == 1:
            return signals[0]
        
        # Sort by confidence
        sorted_signals = sorted(signals, key=lambda s: s.get('confidence', 0.5), reverse=True)
        
        best = sorted_signals[0]
        logger.info(f"Selected {best['strategy_name']} (confidence: {best.get('confidence', 0.5):.2f})")
        
        return best
    
    def get_recent_signals(self, limit=10):
        """Get recent signals"""
        return self.signal_history[-limit:]


# Multi-timeframe signal engine
class MultiTimeframeSignalEngine(LiveSignalEngine):
    """
    Signal engine for multi-timeframe strategies
    """
    
    def __init__(self, strategy, data_manager, primary_interval='1h'):
        """
        Initialize MTF signal engine
        
        Args:
            strategy: Multi-timeframe strategy instance
            data_manager: LiveDataManager with multiple intervals
            primary_interval: Main interval for signal generation
        """
        super().__init__(strategy, data_manager)
        self.primary_interval = primary_interval
    
    def on_candle_close(self, interval):
        """Generate signal using multi-timeframe data"""
        # Only generate signals on primary interval close
        if interval != self.primary_interval:
            return None
        
        # Check all intervals have enough data
        for intv in self.data_manager.intervals:
            if not self.data_manager.is_ready(intv, self.min_candles):
                logger.debug(f"MTF: Waiting for {intv} data")
                return None
        
        # Get all timeframe data
        all_data = self.data_manager.get_all_data()
        
        # Run MTF strategy
        try:
            strategy = self.strategies[0]
            signal = strategy.signal(all_data)
            
            if signal:
                signal['strategy_name'] = strategy.name if hasattr(strategy, 'name') else 'MultiTimeframe'
                signal['interval'] = self.primary_interval
                
                logger.info(f"📊 MTF Signal: {signal['action']} (all timeframes aligned)")
                
                self.signal_history.append({
                    'timestamp': all_data[self.primary_interval]['timestamps'][-1],
                    'signal': signal,
                    'interval': self.primary_interval
                })
                
                return signal
        
        except Exception as e:
            logger.error(f"Error in MTF strategy: {e}")
        
        return None
