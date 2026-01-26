"""
Unit Tests for Professional Trading Architecture

Tests for:
- MarketContextEngine
- BiasEngine
- TradePermissionFilter
- DecisionLogger
- EntryProposal system
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

from src.core.market_context_engine import (
    MarketContextEngine, MarketRegime, get_market_context_engine
)
from src.core.bias_engine import (
    BiasEngine, TradeBias, get_bias_engine
)
from src.core.trade_permission_filter import (
    TradePermissionFilter, PermissionDenialReason
)
from src.core.decision_logger import (
    DecisionLogger, DecisionType
)
from src.core.entry_proposal import (
    EntryProposal, ProposalRanker
)
from src.core.risk_guardian import RiskGuardian, RiskLimits


class TestMarketContextEngine:
    """Test market context analysis."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range('2024-01-01', periods=300, freq='1h')
        
        # Trending data
        trend = np.cumsum(np.random.randn(300) * 10) + 60000
        
        df = pd.DataFrame({
            'open': trend + np.random.randn(300) * 50,
            'high': trend + np.abs(np.random.randn(300) * 100),
            'low': trend - np.abs(np.random.randn(300) * 100),
            'close': trend,
            'volume': np.random.uniform(100, 1000, 300)
        }, index=dates)
        
        return df
    
    def test_context_engine_initialization(self):
        """Test context engine can be initialized."""
        engine = MarketContextEngine()
        assert engine is not None
        assert engine.max_spread_bps == 10.0
    
    def test_get_current_context(self, sample_data):
        """Test context analysis returns valid result."""
        engine = MarketContextEngine()
        
        context = engine.get_current_context(
            df_1h=sample_data,
            current_price=60000
        )
        
        assert context is not None
        assert context.regime in MarketRegime
        assert 0.0 <= context.confidence <= 1.0
        assert context.trend_1h in ['up', 'down', 'neutral']
        assert context.volatility_regime in ['low', 'normal', 'high', 'extreme']
        assert context.trend_strength in ['weak', 'moderate', 'strong']
        assert isinstance(context.trading_allowed, bool)
    
    def test_context_blocks_on_extreme_volatility(self, sample_data):
        """Test context blocks trading during extreme volatility."""
        engine = MarketContextEngine(extreme_vol_percentile=50.0)
        
        # Add extreme volatility
        volatile_data = sample_data.copy()
        volatile_data['high'] = volatile_data['close'] + 5000
        volatile_data['low'] = volatile_data['close'] - 5000
        
        context = engine.get_current_context(
            df_1h=volatile_data,
            current_price=60000
        )
        
        # Should detect high volatility
        assert context.volatility_regime in ['high', 'extreme']


class TestBiasEngine:
    """Test directional bias determination."""
    
    @pytest.fixture
    def mock_context(self):
        """Create mock market context."""
        from src.core.market_context_engine import MarketContext, LiquidityMetrics
        
        return MarketContext(
            timestamp=datetime.now(),
            symbol="BTC/USDT",
            regime=MarketRegime.TRENDING_UP,
            confidence=0.8,
            trend_1h='up',
            trend_4h='up',
            trend_1d='up',
            atr_percentile=55.0,
            volatility_regime='normal',
            adx_value=28.0,
            trend_strength='moderate',
            liquidity=LiquidityMetrics(
                spread_bps=5.0,
                bid_size=10.0,
                ask_size=10.0,
                volume_24h=1000.0,
                is_acceptable=True
            ),
            trading_allowed=True,
            reason="Trending market",
            metadata={}
        )
    
    @pytest.fixture
    def trending_data(self):
        """Create trending price data."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        prices = np.linspace(60000, 65000, 100)  # Clear uptrend
        
        return pd.DataFrame({
            'close': prices,
            'high': prices + 100,
            'low': prices - 100,
            'volume': 100
        }, index=dates)
    
    def test_bias_engine_initialization(self):
        """Test bias engine initialization."""
        engine = BiasEngine()
        assert engine is not None
        assert engine.min_flip_interval_hours == 4
    
    def test_get_current_bias_trending_up(self, trending_data, mock_context):
        """Test bias detection in uptrend."""
        engine = BiasEngine()
        
        bias = engine.get_current_bias(trending_data, mock_context)
        
        assert bias.bias == TradeBias.LONG_BIAS
        assert bias.conviction > 0.5
        assert bias.higher_tf_trend == 'up'
    
    def test_bias_neutral_when_context_blocked(self, trending_data, mock_context):
        """Test bias is neutral when context blocks trading."""
        engine = BiasEngine()
        
        # Block trading in context
        mock_context.trading_allowed = False
        mock_context.reason = "Low liquidity"
        
        bias = engine.get_current_bias(trending_data, mock_context)
        
        assert bias.bias == TradeBias.NEUTRAL
        assert "Context blocked" in bias.reason
    
    def test_bias_flip_rate_limiting(self, trending_data, mock_context):
        """Test bias cannot flip too rapidly."""
        engine = BiasEngine(min_flip_interval_hours=4)
        
        # First bias
        bias1 = engine.get_current_bias(trending_data, mock_context)
        assert bias1.bias == TradeBias.LONG_BIAS
        
        # Try to flip immediately (should be prevented)
        mock_context.trend_1h = 'down'
        mock_context.trend_4h = 'down'
        mock_context.trend_1d = 'down'
        
        bias2 = engine.get_current_bias(trending_data, mock_context)
        
        # Should still be LONG_BIAS due to flip prevention
        assert bias2.bias == TradeBias.LONG_BIAS
        assert "flip blocked" in bias2.reason.lower()


class TestTradePermissionFilter:
    """Test trade permission filtering."""
    
    @pytest.fixture
    def risk_guardian(self):
        """Create risk guardian."""
        return RiskGuardian(portfolio_value=10000)
    
    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        from src.core.market_context_engine import MarketContext, LiquidityMetrics
        
        return MarketContext(
            timestamp=datetime.now(),
            symbol="BTC/USDT",
            regime=MarketRegime.TRENDING_UP,
            confidence=0.8,
            trend_1h='up',
            trend_4h='up',
            trend_1d='up',
            atr_percentile=55.0,
            volatility_regime='normal',
            adx_value=28.0,
            trend_strength='moderate',
            liquidity=LiquidityMetrics(
                spread_bps=5.0,
                bid_size=10.0,
                ask_size=10.0,
                volume_24h=1000.0,
                is_acceptable=True
            ),
            trading_allowed=True,
            reason="Approved",
            metadata={}
        )
    
    @pytest.fixture
    def mock_bias(self):
        """Create mock bias."""
        from src.core.bias_engine import BiasState
        
        return BiasState(
            bias=TradeBias.LONG_BIAS,
            conviction=0.8,
            timestamp=datetime.now(),
            higher_tf_trend='up',
            momentum_direction='bullish',
            volatility_expansion=False,
            last_flip_time=None,
            flips_in_window=0,
            reason="Strong uptrend",
            metadata={}
        )
    
    def test_permission_filter_initialization(self, risk_guardian):
        """Test permission filter initialization."""
        filter = TradePermissionFilter(risk_guardian)
        assert filter is not None
    
    def test_permission_approved_when_all_checks_pass(
        self, risk_guardian, mock_context, mock_bias
    ):
        """Test permission approved when all checks pass."""
        filter = TradePermissionFilter(risk_guardian)
        
        permission = filter.check_permission(
            context=mock_context,
            bias=mock_bias,
            direction="LONG"
        )
        
        assert permission.approved is True
        assert all(permission.checks_passed.values())
    
    def test_permission_denied_when_bias_neutral(
        self, risk_guardian, mock_context, mock_bias
    ):
        """Test permission denied when bias is neutral."""
        filter = TradePermissionFilter(risk_guardian)
        
        mock_bias.bias = TradeBias.NEUTRAL
        
        permission = filter.check_permission(
            context=mock_context,
            bias=mock_bias,
            direction="LONG"
        )
        
        assert permission.approved is False
        assert permission.denial_category == PermissionDenialReason.BIAS_NEUTRAL
    
    def test_permission_denied_when_direction_misaligned(
        self, risk_guardian, mock_context, mock_bias
    ):
        """Test permission denied when direction conflicts with bias."""
        filter = TradePermissionFilter(risk_guardian)
        
        # Bias is LONG, but trying to SHORT
        permission = filter.check_permission(
            context=mock_context,
            bias=mock_bias,
            direction="SHORT"
        )
        
        assert permission.approved is False
        assert permission.denial_category == PermissionDenialReason.BIAS_MISALIGNED


class TestDecisionLogger:
    """Test decision logging."""
    
    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary log directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_logger_initialization(self, temp_log_dir):
        """Test logger initialization."""
        logger = DecisionLogger(log_dir=temp_log_dir)
        assert logger is not None
        assert Path(temp_log_dir).exists()
    
    def test_log_context_decision(self, temp_log_dir):
        """Test logging context decision."""
        from src.core.market_context_engine import MarketContext, LiquidityMetrics
        
        logger = DecisionLogger(log_dir=temp_log_dir)
        
        context = MarketContext(
            timestamp=datetime.now(),
            symbol="BTC/USDT",
            regime=MarketRegime.TRENDING_UP,
            confidence=0.8,
            trend_1h='up',
            trend_4h='up',
            trend_1d='up',
            atr_percentile=55.0,
            volatility_regime='normal',
            adx_value=28.0,
            trend_strength='moderate',
            liquidity=LiquidityMetrics(5.0, 10.0, 10.0, 1000.0, True),
            trading_allowed=True,
            reason="Test",
            metadata={}
        )
        
        logger.log_context_decision(
            symbol="BTC/USDT",
            context=context,
            approved=True,
            reason="Test logging"
        )
        
        assert len(logger.log_buffer) == 1
        assert logger.log_buffer[0].decision_type == DecisionType.CONTEXT_DECISION
    
    def test_get_decision_stats(self, temp_log_dir):
        """Test getting decision statistics."""
        logger = DecisionLogger(log_dir=temp_log_dir)
        
        # Log some decisions
        for i in range(10):
            logger.log_no_trade_period("BTC/USDT", f"Test {i}")
        
        stats = logger.get_decision_stats(hours=24)
        
        assert stats['total_decisions'] == 10
        assert DecisionType.NO_TRADE_PERIOD.value in stats['decision_type_counts']


class TestEntryProposal:
    """Test entry proposal system."""
    
    def test_proposal_creation(self):
        """Test creating an entry proposal."""
        proposal = EntryProposal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            timestamp=datetime.now(),
            direction="LONG",
            entry_price=60000,
            size=0.01,
            stop_loss=58000,
            take_profit=62000,
            reasoning="Test proposal",
            confidence=0.8,
            context_alignment=0.9,
            bias_alignment=1.0,
            metadata={}
        )
        
        assert proposal.strategy_id == "test_strategy"
        assert proposal.direction == "LONG"
    
    def test_risk_reward_ratio_calculation(self):
        """Test R:R ratio calculation."""
        proposal = EntryProposal(
            strategy_id="test",
            symbol="BTC/USDT",
            timestamp=datetime.now(),
            direction="LONG",
            entry_price=60000,
            size=0.01,
            stop_loss=58000,  # Risk: 2000
            take_profit=64000,  # Reward: 4000
            reasoning="Test",
            confidence=0.8,
            context_alignment=0.9,
            bias_alignment=1.0,
            metadata={}
        )
        
        rr = proposal.get_risk_reward_ratio()
        assert rr == 2.0  # 4000 / 2000
    
    def test_overall_score_calculation(self):
        """Test overall score calculation."""
        proposal = EntryProposal(
            strategy_id="test",
            symbol="BTC/USDT",
            timestamp=datetime.now(),
            direction="LONG",
            entry_price=60000,
            size=0.01,
            stop_loss=58000,
            take_profit=64000,
            reasoning="Test",
            confidence=0.8,
            context_alignment=0.9,
            bias_alignment=1.0,
            metadata={}
        )
        
        score = proposal.get_overall_score()
        assert 0.0 <= score <= 1.0
        assert score > 0.7  # Should be high with good parameters


class TestProposalRanker:
    """Test proposal ranking system."""
    
    def test_rank_proposals(self):
        """Test ranking multiple proposals."""
        proposals = [
            EntryProposal(
                strategy_id="low_score",
                symbol="BTC/USDT",
                timestamp=datetime.now(),
                direction="LONG",
                entry_price=60000,
                size=0.01,
                stop_loss=58000,
                take_profit=61000,
                reasoning="Low confidence",
                confidence=0.3,
                context_alignment=0.4,
                bias_alignment=0.5,
                metadata={}
            ),
            EntryProposal(
                strategy_id="high_score",
                symbol="BTC/USDT",
                timestamp=datetime.now(),
                direction="LONG",
                entry_price=60000,
                size=0.01,
                stop_loss=58000,
                take_profit=64000,
                reasoning="High confidence",
                confidence=0.9,
                context_alignment=0.95,
                bias_alignment=1.0,
                metadata={}
            )
        ]
        
        ranked = ProposalRanker.rank_proposals(proposals)
        
        assert len(ranked) == 2
        assert ranked[0].strategy_id == "high_score"
        assert ranked[1].strategy_id == "low_score"
    
    def test_select_best_proposal(self):
        """Test selecting best proposal."""
        proposals = [
            EntryProposal(
                strategy_id="strategy_1",
                symbol="BTC/USDT",
                timestamp=datetime.now(),
                direction="LONG",
                entry_price=60000,
                size=0.01,
                stop_loss=58000,
                take_profit=62000,
                reasoning="Test",
                confidence=0.6,
                context_alignment=0.7,
                bias_alignment=0.8,
                metadata={}
            ),
            EntryProposal(
                strategy_id="strategy_2",
                symbol="BTC/USDT",
                timestamp=datetime.now(),
                direction="LONG",
                entry_price=60000,
                size=0.01,
                stop_loss=58000,
                take_profit=64000,
                reasoning="Test",
                confidence=0.9,
                context_alignment=0.95,
                bias_alignment=1.0,
                metadata={}
            )
        ]
        
        best = ProposalRanker.select_best_proposal(proposals)
        
        assert best is not None
        assert best.strategy_id == "strategy_2"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
