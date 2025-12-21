"""
Updated Strategy Factory with Adaptive Selection
Includes all new institutional-grade strategi es
"""

from .enhanced_momentum import EnhancedMomentumStrategy
from .mean_reversion import MeanReversionStrategy
from .scalping import ScalpingStrategy
from .macd_crossover import MACDCrossoverStrategy
from .bollinger_breakout import BollingerBreakoutStrategy
from .professional_trend import ProfessionalTrendStrategyWrapper

# New advanced strategies
from .advanced_strategies import (
    StatisticalArbitrageStrategy,
    VolumeProfileStrategy,
    BreakoutMomentumStrategy
)
from .event_driven_strategies import (
    NewsEventTradingStrategy,
    LiquidityGrabStrategy,
    OrderFlowImbalanceStrategy
)


class AdvancedStrategyFactory:
    """
    Enhanced strategy factory with institutional-grade algorithms
    """
    
    STRATEGIES = {
        # Original strategies
        'enhanced_momentum': EnhancedMomentumStrategy,
        'mean_reversion': MeanReversionStrategy,
        'scalping': ScalpingStrategy,
        'macd_crossover': MACDCrossoverStrategy,
        'bollinger_breakout': BollingerBreakoutStrategy,
        'professional_trend': ProfessionalTrendStrategyWrapper,
        
        # Advanced institutional strategies
        'statistical_arbitrage': StatisticalArbitrageStrategy,
        'volume_profile_trading': VolumeProfileStrategy,
        'breakout_momentum': BreakoutMomentumStrategy,
        'news_event_trading': NewsEventTradingStrategy,
        'liquidity_grab': LiquidityGrabStrategy,
        'order_flow_imbalance': OrderFlowImbalanceStrategy,
    }
    
    @classmethod
    def create(cls, strategy_name: str, **kwargs):
        """
        Create a strategy instance
        
        Args:
            strategy_name: Name of strategy to create
            **kwargs: Strategy-specific parameters
        
        Returns:
            Strategy instance
        """
        if strategy_name not in cls.STRATEGIES:
            available = ', '.join(cls.STRATEGIES.keys())
            raise ValueError(
                f"Unknown strategy: {strategy_name}. "
                f"Available strategies: {available}"
            )
        
        strategy_class = cls.STRATEGIES[strategy_name]
        return strategy_class(**kwargs)
    
    @classmethod
    def get_all_strategies(cls):
        """Get list of all available strategy names"""
        return list(cls.STRATEGIES.keys())
    
    @classmethod
    def get_strategy_info(cls, strategy_name: str):
        """Get information about a strategy"""
        if strategy_name not in cls.STRATEGIES:
            return None
        
        strategy_class = cls.STRATEGIES[strategy_name]
        instance = strategy_class()
        
        return {
            'name': instance.name,
            'type': instance.strategy_type,
            'parameters': instance.parameters,
            'class': strategy_class.__name__
        }
    
    @classmethod
    def create_strategy_suite(cls, capital_per_strategy: float = 2000):
        """
        Create a suite of strategies for portfolio trading
        
        Returns:
            Dictionary of strategy_name: instance
        """
        suite = {}
        
        # Diverse strategy mix
        strategy_mix = [
            'statistical_arbitrage',     # Mean reversion
            'breakout_momentum',          # Trend following
            'volume_profile_trading',     # Support/resistance
            'enhanced_momentum',          # Momentum
            'news_event_trading',         # Event-driven
            'liquidity_grab'              # Reversal
        ]
        
        for strategy_name in strategy_mix:
            suite[strategy_name] = cls.create(strategy_name)
        
        return suite


# Legacy support
class StrategyFactory(AdvancedStrategyFactory):
    """Backward compatibility"""
    pass


if __name__ == "__main__":
    print("=" * 70)
    print("ADVANCED STRATEGY FACTORY")
    print("=" * 70)
    
    # List all strategies
    print("\nAvailable Strategies:")
    for i, name in enumerate(AdvancedStrategyFactory.get_all_strategies(), 1):
        info = AdvancedStrategyFactory.get_strategy_info(name)
        print(f"  {i}. {name}")
        print(f"     Type: {info['type']}")
        print(f"     Class: {info['class']}")
    
    # Create strategy suite
    print("\n" + "=" * 70)
    print("STRATEGY SUITE")
    print("=" * 70)
    
    suite = AdvancedStrategyFactory.create_strategy_suite()
    print(f"\nCreated {len(suite)} strategies:")
    for name, strategy in suite.items():
        print(f"  - {name}: {strategy.name}")
    
    print("\n✅ Strategy factory test complete")
