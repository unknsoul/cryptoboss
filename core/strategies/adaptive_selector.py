"""
Adaptive Strategy Selector
Automatically selects the best strategy based on market conditions
Uses market regime, volatility, news, and real-time performance
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum
from core.monitoring.logger import get_logger
from core.monitoring.metrics import get_metrics


logger = get_logger()
metrics = get_metrics()


class MarketRegime(Enum):
    """Market regime classifications"""
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGING_LOW_VOL = "ranging_low_vol"
    RANGING_HIGH_VOL = "ranging_high_vol"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    HIGH_VOLATILITY_CHOP = "high_volatility_chop"
    NEWS_DRIVEN = "news_driven"


class StrategyPerformanceTracker:
    """
    Tracks real-time performance of each strategy
    Used for adaptive selection
    """
    
    def __init__(self, lookback_trades: int = 50):
        self.lookback = lookback_trades
        self.performance_history: Dict[str, List[Dict]] = {}
    
    def record_trade(self, strategy_name: str, trade_result: Dict[str, Any]):
        """Record a trade result for a strategy"""
        if strategy_name not in self.performance_history:
            self.performance_history[strategy_name] = []
        
        trade_record = {
            'timestamp': datetime.now(),
            'pnl': trade_result.get('pnl', 0),
            'win': trade_result.get('pnl', 0) > 0,
            'return_pct': trade_result.get('return_pct', 0)
        }
        
        self.performance_history[strategy_name].append(trade_record)
        
        # Keep only recent trades
        if len(self.performance_history[strategy_name]) > self.lookback:
            self.performance_history[strategy_name] = self.performance_history[strategy_name][-self.lookback:]
    
    def get_recent_performance(self, strategy_name: str, window: int = 20) -> Dict[str, float]:
        """Get recent performance metrics for a strategy"""
        if strategy_name not in self.performance_history:
            return {'win_rate': 0.5, 'avg_return': 0, 'sharpe': 0, 'profit_factor': 1.0}
        
        trades = self.performance_history[strategy_name][-window:]
        
        if not trades:
            return {'win_rate': 0.5, 'avg_return': 0, 'sharpe': 0, 'profit_factor': 1.0}
        
        wins = [t for t in trades if t['win']]
        losses = [t for t in trades if not t['win']]
        
        win_rate = len(wins) / len(trades)
        avg_return = np.mean([t['return_pct'] for t in trades])
        
        # Sharpe ratio (simplified)
        returns = [t['return_pct'] for t in trades]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) if len(returns) > 1 else 0
        
        # Profit factor
        total_wins = sum(t['pnl'] for t in wins)
        total_losses = abs(sum(t['pnl'] for t in losses))
        profit_factor = total_wins / (total_losses + 1e-8)
        
        return {
            'win_rate': win_rate,
            'avg_return': avg_return,
            'sharpe': sharpe,
            'profit_factor': profit_factor,
            'num_trades': len(trades)
        }


class AdaptiveStrategySelector:
    """
    Intelligent strategy selection based on:
    1. Market regime (trending, ranging, volatile)
    2. Volatility level
    3. News sentiment
    4. Recent strategy performance
    5. Time of day / liquidity
    """
    
    def __init__(self):
        self.performance_tracker = StrategyPerformanceTracker()
        self.current_strategy = None
        self.strategy_switches = []
        
        # Strategy -> Regime mapping (which strategies work best in which regimes)
        self.strategy_regime_map = {
            'statistical_arbitrage': [
                MarketRegime.RANGING_LOW_VOL,
                MarketRegime.RANGING_HIGH_VOL
            ],
            'trend_following': [
                MarketRegime.TRENDING_BULL,
                MarketRegime.TRENDING_BEAR
            ],
            'breakout_momentum': [
                MarketRegime.BREAKOUT,
                MarketRegime.BREAKDOWN
            ],
            'mean_reversion': [
                MarketRegime.RANGING_LOW_VOL,
                MarketRegime.RANGING_HIGH_VOL
            ],
            'volatility_breakout': [
                MarketRegime.BREAKOUT,
                MarketRegime.HIGH_VOLATILITY_CHOP
            ],
            'news_event_trading': [
                MarketRegime.NEWS_DRIVEN,
                MarketRegime.HIGH_VOLATILITY_CHOP
            ],
            'market_making': [
                MarketRegime.RANGING_LOW_VOL
            ],
            'volume_profile_trading': [
                MarketRegime.RANGING_HIGH_VOL,
                MarketRegime.TRENDING_BULL,
                MarketRegime.TRENDING_BEAR
            ]
        }
    
    def detect_market_regime(self, market_data: Dict[str, Any]) -> MarketRegime:
        """
        Detect current market regime using multiple factors
        
        Args:
            market_data: Dictionary containing:
                - prices: Recent price series
                - volume: Volume data
                - volatility: Current volatility
                - avg_volatility: Average volatility
                - sentiment_score: News sentiment (-1 to 1)
                - trend_strength: R-squared of trend
                - adx: ADX indicator value
        
        Returns:
            MarketRegime enum
        """
        prices = market_data.get('prices', [])
        volatility = market_data.get('volatility', 0.02)
        avg_volatility = market_data.get('avg_volatility', 0.02)
        sentiment = market_data.get('sentiment_score', 0)
        trend_strength = market_data.get('trend_strength', 0)
        adx = market_data.get('adx', 20)
        
        if len(prices) < 50:
            return MarketRegime.RANGING_LOW_VOL  # Default
        
        # Calculate metrics
        vol_ratio = volatility / (avg_volatility + 1e-8)
        
        # Recent price momentum
        momentum_20 = (prices[-1] / prices[-20] - 1) if len(prices) >= 20 else 0
        momentum_50 = (prices[-1] / prices[-50] - 1) if len(prices) >= 50 else 0
        
        # 1. News-driven regime (strong sentiment + high volatility)
        if abs(sentiment) > 0.6 and vol_ratio > 1.5:
            logger.info("Market regime detected: NEWS_DRIVEN", sentiment=sentiment, vol_ratio=vol_ratio)
            return MarketRegime.NEWS_DRIVEN
        
        # 2. High volatility chop (high vol, no clear trend)
        if vol_ratio > 2.0 and trend_strength < 0.3:
            logger.info("Market regime detected: HIGH_VOLATILITY_CHOP", vol_ratio=vol_ratio)
            return MarketRegime.HIGH_VOLATILITY_CHOP
        
        # 3. Breakout (price breaking above recent range + volume)
        recent_high = np.max(prices[-50:-1])
        if prices[-1] > recent_high * 1.02 and vol_ratio > 1.2:
            logger.info("Market regime detected: BREAKOUT", price=prices[-1], recent_high=recent_high)
            return MarketRegime.BREAKOUT
        
        # 4. Breakdown (price breaking below recent range)
        recent_low = np.min(prices[-50:-1])
        if prices[-1] < recent_low * 0.98 and vol_ratio > 1.2:
            logger.info("Market regime detected: BREAKDOWN", price=prices[-1], recent_low=recent_low)
            return MarketRegime.BREAKDOWN
        
        # 5. Trending Bull (strong upward trend)
        if adx > 25 and trend_strength > 0.6 and momentum_20 > 0.03 and momentum_50 > 0.05:
            logger.info("Market regime detected: TRENDING_BULL", adx=adx, trend_strength=trend_strength)
            return MarketRegime.TRENDING_BULL
        
        # 6. Trending Bear (strong downward trend)
        if adx > 25 and trend_strength > 0.6 and momentum_20 < -0.03 and momentum_50 < -0.05:
            logger.info("Market regime detected: TRENDING_BEAR", adx=adx, trend_strength=trend_strength)
            return MarketRegime.TRENDING_BEAR
        
        # 7. Ranging Low Vol (sideways, stable)
        if vol_ratio < 0.8 and trend_strength < 0.4:
            logger.info("Market regime detected: RANGING_LOW_VOL", vol_ratio=vol_ratio)
            return MarketRegime.RANGING_LOW_VOL
        
        # 8. Ranging High Vol (sideways, volatile)
        if vol_ratio >= 0.8 and trend_strength < 0.4:
            logger.info("Market regime detected: RANGING_HIGH_VOL", vol_ratio=vol_ratio)
            return MarketRegime.RANGING_HIGH_VOL
        
        # Default
        return MarketRegime.RANGING_LOW_VOL
    
    def select_best_strategy(self, market_data: Dict[str, Any], 
                            available_strategies: List[str]) -> str:
        """
        Select the best strategy based on current market conditions
        
        Args:
            market_data: Market condition data
            available_strategies: List of available strategy names
        
        Returns:
            Selected strategy name
        """
        # 1. Detect current market regime
        regime = self.detect_market_regime(market_data)
        
        # 2. Get strategies suitable for this regime
        suitable_strategies = []
        for strategy in available_strategies:
            if strategy in self.strategy_regime_map:
                if regime in self.strategy_regime_map[strategy]:
                    suitable_strategies.append(strategy)
        
        # If no suitable strategies found, use all
        if not suitable_strategies:
            suitable_strategies = available_strategies
        
        # 3. Score each suitable strategy based on recent performance
        strategy_scores = {}
        for strategy in suitable_strategies:
            perf = self.performance_tracker.get_recent_performance(strategy, window=20)
            
            # Scoring formula (can be tuned)
            score = (
                perf['win_rate'] * 40 +  # Win rate weight
                (perf['sharpe'] * 10 if perf['sharpe'] > 0 else 0) +  # Sharpe weight
                (perf['profit_factor'] * 20 if perf['profit_factor'] > 1 else 0) +  # PF weight
                (perf['avg_return'] * 100 * 30 if perf['avg_return'] > 0 else 0)  # Return weight
            )
            
            # Bonus for strategies that are good for current regime
            if regime in self.strategy_regime_map.get(strategy, []):
                score *= 1.3  # 30% bonus
            
            strategy_scores[strategy] = score
        
        # 4. Select best strategy
        if not strategy_scores:
            best_strategy = suitable_strategies[0] if suitable_strategies else available_strategies[0]
        else:
            best_strategy = max(strategy_scores, key=strategy_scores.get)
        
        # 5. Check if we should switch strategies
        should_switch = self._should_switch_strategy(best_strategy, regime)
        
        if should_switch and best_strategy != self.current_strategy:
            logger.info(
                f"🔄 Strategy switch: {self.current_strategy} → {best_strategy}",
                regime=regime.value,
                old_strategy=self.current_strategy,
                new_strategy=best_strategy,
                reason="regime_change"
            )
            
            self.strategy_switches.append({
                'timestamp': datetime.now(),
                'from_strategy': self.current_strategy,
                'to_strategy': best_strategy,
                'regime': regime,
                'scores': strategy_scores
            })
            
            self.current_strategy = best_strategy
            metrics.increment(f"strategy_switch_{best_strategy}")
        
        return best_strategy
    
    def _should_switch_strategy(self, proposed_strategy: str, regime: MarketRegime) -> bool:
        """
        Determine if we should switch to proposed strategy
        Prevents excessive switching
        """
        # Always switch if no current strategy
        if self.current_strategy is None:
            return True
        
        # Don't switch if proposed is same as current
        if proposed_strategy == self.current_strategy:
            return False
        
        # Check recent switches (prevent rapid switching)
        recent_switches = [
            s for s in self.strategy_switches
            if s['timestamp'] > datetime.now() - timedelta(hours=2)
        ]
        
        if len(recent_switches) >= 3:
            logger.warning("Too many recent strategy switches, holding current strategy")
            return False
        
        # Check if current strategy is performing well
        current_perf = self.performance_tracker.get_recent_performance(self.current_strategy, window=10)
        proposed_perf = self.performance_tracker.get_recent_performance(proposed_strategy, window=10)
        
        # Only switch if proposed is significantly better (20%+ improvement)
        if current_perf.get('num_trades', 0) >= 5:
            current_score = current_perf['win_rate'] * current_perf.get('profit_factor', 1)
            proposed_score = proposed_perf['win_rate'] * proposed_perf.get('profit_factor', 1)
            
            if proposed_score < current_score * 1.2:  # Not 20% better
                logger.info(
                    "Proposed strategy not significantly better, keeping current",
                    current_score=current_score,
                    proposed_score=proposed_score
                )
                return False
        
        return True
    
    def get_strategy_allocation(self, market_data: Dict[str, Any],
                               available_strategies: List[str]) -> Dict[str, float]:
        """
        Alternative approach: Return allocation weights for multiple strategies
        Can run strategies in parallel with different position sizes
        
        Returns:
            Dictionary of strategy_name: weight (0-1, sum=1)
        """
        regime = self.detect_market_regime(market_data)
        allocations = {}
        
        for strategy in available_strategies:
            # Base allocation
            weight = 0.1  # Minimum weight
            
            # Regime bonus
            if regime in self.strategy_regime_map.get(strategy, []):
                weight += 0.4
            
            # Performance bonus
            perf = self.performance_tracker.get_recent_performance(strategy, window=20)
            if perf['win_rate'] > 0.6:
                weight += 0.3
            if perf.get('sharpe', 0) > 1.5:
                weight += 0.2
            
            allocations[strategy] = weight
        
        # Normalize to sum to 1
        total = sum(allocations.values())
        if total > 0:
            allocations = {k: v/total for k, v in allocations.items()}
        
        return allocations
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get selector statistics"""
        return {
            'current_strategy': self.current_strategy,
            'total_switches': len(self.strategy_switches),
            'recent_switches': [
                s for s in self.strategy_switches
                if s['timestamp'] > datetime.now() - timedelta(days=7)
            ],
            'strategy_performance': {
                name: self.performance_tracker.get_recent_performance(name)
                for name in self.performance_tracker.performance_history.keys()
            }
        }


if __name__ == "__main__":
    # Test adaptive selector
    selector = AdaptiveStrategySelector()
    
    # Simulate market data
    prices = [45000 + i*10 + np.random.randn()*50 for i in range(100)]
    
    market_data = {
        'prices': prices,
        'volume': 1000,
        'volatility': 0.025,
        'avg_volatility': 0.020,
        'sentiment_score': 0.3,
        'trend_strength': 0.75,
        'adx': 28
    }
    
    available_strategies = [
        'statistical_arbitrage',
        'trend_following',
        'mean_reversion',
        'breakout_momentum',
        'market_making'
    ]
    
    # Test strategy selection
    best_strategy = selector.select_best_strategy(market_data, available_strategies)
    
    print("=" * 70)
    print("ADAPTIVE STRATEGY SELECTOR TEST")
    print("=" * 70)
    print(f"\nSelected Strategy: {best_strategy}")
    print(f"Market Regime: {selector.detect_market_regime(market_data)}")
    
    # Test allocation approach
    allocations = selector.get_strategy_allocation(market_data, available_strategies)
    print(f"\nStrategy Allocations:")
    for strategy, weight in sorted(allocations.items(), key=lambda x: x[1], reverse=True):
        print(f"  {strategy}: {weight:.1%}")
    
    print("\n✅ Adaptive selector test complete")
