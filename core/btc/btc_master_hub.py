"""
BTC Master Hub
Integrates all BTC-specific analysis for trading decisions

Features:
- Unified signal generation from all BTC modules
- Strategy performance tracking
- Circuit breaker for max daily loss
- Conviction-based position sizing
- Configuration management
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class StrategyPerformanceTracker:
    """
    Track performance per strategy
    
    Identifies which strategies are profitable
    """
    
    def __init__(self):
        self.strategy_stats = {}
        
    def record_trade(self, strategy_name: str, pnl: float, pnl_pct: float):
        """Record a trade for a strategy"""
        if strategy_name not in self.strategy_stats:
            self.strategy_stats[strategy_name] = {
                'total_trades': 0,
                'winners': 0,
                'losers': 0,
                'total_pnl': 0,
                'gross_profit': 0,
                'gross_loss': 0
            }
        
        stats = self.strategy_stats[strategy_name]
        stats['total_trades'] += 1
        stats['total_pnl'] += pnl
        
        if pnl > 0:
            stats['winners'] += 1
            stats['gross_profit'] += pnl
        else:
            stats['losers'] += 1
            stats['gross_loss'] += abs(pnl)
    
    def get_strategy_ranking(self) -> List[Dict]:
        """Get strategies ranked by performance"""
        rankings = []
        
        for name, stats in self.strategy_stats.items():
            if stats['total_trades'] > 0:
                win_rate = stats['winners'] / stats['total_trades'] * 100
                
                # Profit factor
                if stats['gross_loss'] > 0:
                    profit_factor = stats['gross_profit'] / stats['gross_loss']
                else:
                    profit_factor = 999 if stats['gross_profit'] > 0 else 0
                
                rankings.append({
                    'strategy': name,
                    'trades': stats['total_trades'],
                    'win_rate': round(win_rate, 1),
                    'total_pnl': round(stats['total_pnl'], 2),
                    'profit_factor': round(profit_factor, 2)
                })
        
        # Sort by profit factor
        rankings.sort(key=lambda x: x['profit_factor'], reverse=True)
        return rankings
    
    def get_best_strategies(self, min_trades: int = 10) -> List[str]:
        """Get names of profitable strategies"""
        rankings = self.get_strategy_ranking()
        return [
            r['strategy'] for r in rankings 
            if r['trades'] >= min_trades and r['profit_factor'] > 1.5
        ]


class CircuitBreaker:
    """
    Circuit breaker for max daily loss protection
    
    Stops trading when daily loss exceeds threshold
    """
    
    def __init__(self, max_daily_loss_pct: float = 5.0, max_consecutive_losses: int = 5):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_consecutive_losses = max_consecutive_losses
        
        self.daily_pnl = 0
        self.daily_start_equity = 0
        self.consecutive_losses = 0
        self.current_date = date.today()
        self.is_tripped = False
        self.trip_reason = None
        
    def reset_daily(self, current_equity: float):
        """Reset daily tracking"""
        self.daily_pnl = 0
        self.daily_start_equity = current_equity
        self.current_date = date.today()
        self.is_tripped = False
        self.trip_reason = None
        logger.info(f"Circuit breaker reset. Daily start equity: ${current_equity:,.2f}")
    
    def record_trade(self, pnl: float) -> bool:
        """
        Record trade and check if circuit breaker should trip
        
        Returns:
            True if circuit breaker is now tripped
        """
        # Check if new day
        if date.today() != self.current_date:
            # Auto-reset not done here - should be called externally
            pass
        
        self.daily_pnl += pnl
        
        # Track consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Check max daily loss
        if self.daily_start_equity > 0:
            daily_loss_pct = abs(self.daily_pnl) / self.daily_start_equity * 100
            
            if self.daily_pnl < 0 and daily_loss_pct >= self.max_daily_loss_pct:
                self.is_tripped = True
                self.trip_reason = f"Max daily loss ({daily_loss_pct:.1f}% > {self.max_daily_loss_pct}%)"
                logger.critical(f"🛑 CIRCUIT BREAKER TRIPPED: {self.trip_reason}")
                return True
        
        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.is_tripped = True
            self.trip_reason = f"Max consecutive losses ({self.consecutive_losses})"
            logger.critical(f"🛑 CIRCUIT BREAKER TRIPPED: {self.trip_reason}")
            return True
        
        return False
    
    def can_trade(self) -> tuple:
        """Check if trading is allowed"""
        if self.is_tripped:
            return False, self.trip_reason
        return True, "OK"
    
    def get_status(self) -> Dict:
        """Get circuit breaker status"""
        daily_loss_pct = 0
        if self.daily_start_equity > 0 and self.daily_pnl < 0:
            daily_loss_pct = abs(self.daily_pnl) / self.daily_start_equity * 100
        
        return {
            'is_tripped': self.is_tripped,
            'reason': self.trip_reason,
            'daily_pnl': round(self.daily_pnl, 2),
            'daily_loss_pct': round(daily_loss_pct, 2),
            'consecutive_losses': self.consecutive_losses,
            'remaining_loss_budget': round(self.max_daily_loss_pct - daily_loss_pct, 2)
        }


class ConvictionSizer:
    """
    Position sizing based on trade conviction
    
    A+ setups get more size, B/C setups get less
    """
    
    def __init__(self, base_size: float = 1.0):
        self.base_size = base_size
        
    def calculate_size_multiplier(self, signals: Dict) -> float:
        """
        Calculate size multiplier based on signal alignment
        
        Args:
            signals: Dict of various signal sources and their values
            
        Returns:
            Multiplier from 0.5 to 2.0
        """
        conviction_score = 0
        max_score = 0
        
        # Weight different signals
        weights = {
            'trend_alignment': 2.0,
            'funding_alignment': 1.5,
            'oi_confirmation': 1.5,
            'session_quality': 1.0,
            'order_block': 1.5,
            'fvg_confirmation': 1.0,
            'cme_gap_trade': 1.5,
            'dominance_alignment': 1.0
        }
        
        for signal_name, weight in weights.items():
            max_score += weight
            if signal_name in signals:
                signal_value = signals[signal_name]
                if isinstance(signal_value, bool):
                    conviction_score += weight if signal_value else 0
                elif isinstance(signal_value, (int, float)):
                    conviction_score += weight * min(max(signal_value, 0), 1)
        
        # Normalize to 0-1 range
        normalized = conviction_score / max_score if max_score > 0 else 0.5
        
        # Map to 0.5x to 2.0x multiplier
        # 0 = 0.5x, 0.5 = 1.0x, 1.0 = 2.0x
        multiplier = 0.5 + normalized * 1.5
        
        return round(multiplier, 2)
    
    def get_position_grade(self, multiplier: float) -> str:
        """Get letter grade for position"""
        if multiplier >= 1.8:
            return 'A+'
        elif multiplier >= 1.5:
            return 'A'
        elif multiplier >= 1.2:
            return 'B+'
        elif multiplier >= 1.0:
            return 'B'
        elif multiplier >= 0.75:
            return 'C'
        else:
            return 'D'


class BTCMasterConfig:
    """
    External configuration management
    
    Load settings from config file instead of hardcoding
    """
    
    DEFAULT_CONFIG = {
        'risk': {
            'max_daily_loss_pct': 5.0,
            'max_position_risk_pct': 1.0,
            'max_consecutive_losses': 5,
            'min_rr_ratio': 2.0
        },
        'signals': {
            'min_confidence': 0.50,
            'min_conviction_score': 0.4
        },
        'sessions': {
            'trade_asia': True,
            'trade_london': True,
            'trade_ny': True,
            'avoid_sunday': True
        },
        'features': {
            'use_cme_gaps': True,
            'use_funding_rate': True,
            'use_order_blocks': True,
            'use_fvg': True,
            'use_dominance': True
        }
    }
    
    def __init__(self, config_path: str = "config/btc_master.json"):
        self.config_path = Path(config_path)
        self.config = self.DEFAULT_CONFIG.copy()
        self.load()
        
    def load(self):
        """Load config from file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    loaded = json.load(f)
                    # Deep merge
                    for key, value in loaded.items():
                        if isinstance(value, dict) and key in self.config:
                            self.config[key].update(value)
                        else:
                            self.config[key] = value
                logger.info(f"✓ Config loaded from {self.config_path}")
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
    
    def save(self):
        """Save config to file"""
        try:
            self.config_path.parent.mkdir(exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Config saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Could not save config: {e}")
    
    def get(self, section: str, key: str, default=None):
        """Get config value"""
        return self.config.get(section, {}).get(key, default)
    
    def set(self, section: str, key: str, value):
        """Set config value"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value


class BTCMasterHub:
    """
    Central hub for all BTC-specific analysis
    
    Integrates all modules and provides unified signals
    """
    
    def __init__(self):
        # Load config
        self.config = BTCMasterConfig()
        
        # Initialize components
        self.strategy_tracker = StrategyPerformanceTracker()
        self.circuit_breaker = CircuitBreaker(
            max_daily_loss_pct=self.config.get('risk', 'max_daily_loss_pct', 5.0),
            max_consecutive_losses=self.config.get('risk', 'max_consecutive_losses', 5)
        )
        self.conviction_sizer = ConvictionSizer()
        
        # Load BTC analysis modules
        try:
            from core.btc.btc_analysis import (
                get_cme_tracker, get_dominance_tracker, get_funding_analyzer,
                get_oi_tracker, get_ob_detector, get_fvg_detector,
                get_liq_tracker, get_session_analyzer
            )
            
            self.cme = get_cme_tracker()
            self.dominance = get_dominance_tracker()
            self.funding = get_funding_analyzer()
            self.oi = get_oi_tracker()
            self.ob = get_ob_detector()
            self.fvg = get_fvg_detector()
            self.liquidations = get_liq_tracker()
            self.sessions = get_session_analyzer()
            
            self.modules_loaded = True
            logger.info("✓ BTC Master Hub: All modules loaded")
            
        except ImportError as e:
            logger.warning(f"Some BTC modules not loaded: {e}")
            self.modules_loaded = False
    
    def analyze(self, current_price: float, candles: List[Dict] = None) -> Dict:
        """
        Run full BTC analysis
        
        Returns unified analysis with all signals
        """
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'price': current_price,
            'signals': {},
            'adjustments': {},
            'warnings': []
        }
        
        if not self.modules_loaded:
            analysis['warnings'].append('BTC modules not fully loaded')
            return analysis
        
        # Session analysis
        session = self.sessions.get_current_session()
        day_bias = self.sessions.get_day_bias()
        analysis['session'] = session
        analysis['day_bias'] = day_bias
        
        if session['strategy_bias'] == 'AVOID':
            analysis['warnings'].append('Avoid trading in current session')
        
        # CME Gap
        cme_signal = self.cme.get_gap_trade_signal(current_price)
        if cme_signal:
            analysis['signals']['cme_gap'] = cme_signal
        
        # Funding
        funding_signal = self.funding.get_signal()
        if funding_signal:
            analysis['signals']['funding'] = funding_signal
        
        # Order blocks
        if candles:
            self.ob.detect_order_blocks(candles)
            ob_signal = self.ob.check_price_at_ob(current_price)
            if ob_signal:
                analysis['signals']['order_block'] = ob_signal
            
            # FVGs
            self.fvg.detect_fvgs(candles)
            fvg_signal = self.fvg.check_fvg_fill(current_price)
            if fvg_signal:
                analysis['signals']['fvg'] = fvg_signal
        
        # Liquidation levels
        self.liquidations.estimate_liquidation_levels(current_price, self.funding.current_rate)
        nearest_liq = self.liquidations.get_nearest_liquidation(current_price)
        if nearest_liq and nearest_liq['is_close']:
            analysis['warnings'].append(f"Near liquidation level: ${nearest_liq['price']:,.0f}")
        
        # Calculate conviction score
        conviction_signals = {}
        if session['volatility'] in ['MEDIUM_HIGH', 'HIGH']:
            conviction_signals['session_quality'] = 1.0
        if 'cme_gap' in analysis['signals']:
            conviction_signals['cme_gap_trade'] = 1.0
        if 'order_block' in analysis['signals']:
            conviction_signals['order_block'] = 1.0
        
        analysis['conviction_multiplier'] = self.conviction_sizer.calculate_size_multiplier(conviction_signals)
        analysis['position_grade'] = self.conviction_sizer.get_position_grade(analysis['conviction_multiplier'])
        
        return analysis
    
    def can_trade(self) -> Tuple:
        """Check if trading is allowed"""
        return self.circuit_breaker.can_trade()
    
    def record_trade(self, strategy: str, pnl: float, pnl_pct: float):
        """Record a completed trade"""
        self.strategy_tracker.record_trade(strategy, pnl, pnl_pct)
        self.circuit_breaker.record_trade(pnl)
    
    def get_summary(self) -> Dict:
        """Get summary of all analysis"""
        return {
            'circuit_breaker': self.circuit_breaker.get_status(),
            'strategy_rankings': self.strategy_tracker.get_strategy_ranking()[:5],
            'best_strategies': self.strategy_tracker.get_best_strategies()
        }


# Import fix for Tuple
from typing import Tuple


# Singletons
_btc_master = None
_circuit_breaker = None
_strategy_tracker = None
_conviction_sizer = None


def get_btc_master() -> BTCMasterHub:
    global _btc_master
    if _btc_master is None:
        _btc_master = BTCMasterHub()
    return _btc_master


def get_circuit_breaker() -> CircuitBreaker:
    global _circuit_breaker
    if _circuit_breaker is None:
        _circuit_breaker = CircuitBreaker()
    return _circuit_breaker


def get_strategy_tracker() -> StrategyPerformanceTracker:
    global _strategy_tracker
    if _strategy_tracker is None:
        _strategy_tracker = StrategyPerformanceTracker()
    return _strategy_tracker


if __name__ == '__main__':
    print("=" * 70)
    print("BTC MASTER HUB - TEST")
    print("=" * 70)
    
    # Test strategy tracker
    print("\n📊 Testing Strategy Performance Tracker...")
    tracker = StrategyPerformanceTracker()
    
    # Simulate trades
    strategies = ['momentum', 'mean_reversion', 'breakout']
    import random
    for _ in range(50):
        strategy = random.choice(strategies)
        pnl = random.gauss(10 if strategy == 'momentum' else 5, 30)
        tracker.record_trade(strategy, pnl, pnl / 100)
    
    rankings = tracker.get_strategy_ranking()
    print("  Strategy Rankings:")
    for r in rankings:
        print(f"    {r['strategy']}: WR {r['win_rate']}%, PF {r['profit_factor']:.2f}")
    
    # Test circuit breaker
    print("\n🛑 Testing Circuit Breaker...")
    cb = CircuitBreaker(max_daily_loss_pct=5.0)
    cb.reset_daily(10000)
    
    # Simulate losses
    for i in range(5):
        pnl = -100  # $100 loss each
        tripped = cb.record_trade(pnl)
        print(f"  Trade {i+1}: P&L ${pnl}, Tripped: {tripped}")
        if tripped:
            break
    
    status = cb.get_status()
    print(f"  Status: Daily P&L ${status['daily_pnl']}, Can trade: {not status['is_tripped']}")
    
    # Test conviction sizer
    print("\n💪 Testing Conviction Sizer...")
    sizer = ConvictionSizer()
    
    # Strong signals
    signals = {
        'trend_alignment': 1.0,
        'funding_alignment': 0.8,
        'order_block': True,
        'session_quality': 1.0
    }
    mult = sizer.calculate_size_multiplier(signals)
    grade = sizer.get_position_grade(mult)
    print(f"  Strong signals: {mult}x size (Grade: {grade})")
    
    # Weak signals
    weak_signals = {
        'trend_alignment': 0.3,
        'funding_alignment': 0.2
    }
    mult = sizer.calculate_size_multiplier(weak_signals)
    grade = sizer.get_position_grade(mult)
    print(f"  Weak signals: {mult}x size (Grade: {grade})")
    
    print("\n" + "=" * 70)
    print("✅ BTC Master Hub working!")
    print("=" * 70)
