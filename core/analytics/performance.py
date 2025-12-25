"""
Performance Analytics Engine
Advanced metrics calculation and visualization.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class PerformanceAnalytics:
    """
    Advanced performance analytics.
    
    Metrics:
    - Sharpe & Sortino ratios
    - Maximum Adverse Excursion (MAE)
    - Win/Loss distribution
    - Rolling performance
    - Risk-adjusted returns
    """
    
    def __init__(self):
        self.trades: List[Dict] = []
        logger.info("Performance Analytics initialized")
    
    def add_trade(self, trade: Dict):
        """Add trade to analytics."""
        self.trades.append(trade)
    
    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # Annualized
    
    def calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio (downside deviation only)."""
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate
        
        # Downside deviation (only negative returns)
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return 0.0
        
        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0.0
        
        return np.mean(excess_returns) / downside_std * np.sqrt(252)
    
    def calculate_mae_mfe(self) -> Dict:
        """
        Calculate Maximum Adverse/Favorable Excursion.
        
        Returns:
            Dict with MAE/MFE statistics
        """
        if not self.trades:
           return {'mae': 0, 'mfe': 0, 'mae_avg': 0, 'mfe_avg': 0}
        
        # For now, use simple approximation
        # In real implementation, would track intra-trade price movement
        wins = [t for t in self.trades if t.get('pnl', 0) > 0]
        losses = [t for t in self.trades if t.get('pnl', 0) < 0]
        
        mae = abs(np.mean([t.get('pnl', 0) for t in losses])) if losses else 0
        mfe = np.mean([t.get('pnl', 0) for t in wins]) if wins else 0
        
        return {
            'mae': mae,
            'mfe': mfe,
            'mae_avg': mae,
            'mfe_avg': mfe,
            'total_trades': len(self.trades)
        }
    
    def get_trade_distribution(self) -> Dict:
        """Analyze win/loss distribution."""
        if not self.trades:
            return {'wins': 0, 'losses': 0, 'avg_win': 0, 'avg_loss': 0}
        
        wins = [t['pnl'] for t in self.trades if t.get('pnl', 0) > 0]
        losses = [t['pnl'] for t in self.trades if t.get('pnl', 0) < 0]
        
        return {
            'wins': len(wins),
            'losses': len(losses),
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'largest_win': max(wins) if wins else 0,
            'largest_loss': min(losses) if losses else 0,
            'win_stddev': np.std(wins) if len(wins) > 1 else 0,
            'loss_stddev': np.std(losses) if len(losses) > 1 else 0
        }
    
    def calculate_rolling_metrics(self, window: int = 20) -> Dict:
        """Calculate rolling performance metrics."""
        if len(self.trades) < window:
            return {}
        
        # Convert to DataFrame for rolling calculations
        df = pd.DataFrame(self.trades)
        if 'pnl' not in df.columns:
            return {}
        
        rolling_win_rate = df['pnl'].rolling(window).apply(
            lambda x: (x > 0).sum() / len(x) * 100
        )
        
        rolling_avg_pnl = df['pnl'].rolling(window).mean()
        
        return {
            'current_win_rate': rolling_win_rate.iloc[-1] if len(rolling_win_rate) > 0 else 0,
            'current_avg_pnl': rolling_avg_pnl.iloc[-1] if len(rolling_avg_pnl) > 0 else 0,
            'win_rate_trend': 'up' if rolling_win_rate.iloc[-1] > rolling_win_rate.iloc[-5] else 'down'
        }
    
    def export_to_csv(self, filepath: Path):
        """Export trades to CSV for analysis."""
        if not self.trades:
            logger.warning("No trades to export")
            return
        
        df = pd.DataFrame(self.trades)
        df.to_csv(filepath, index=False)
        logger.info(f"Exported {len(self.trades)} trades to {filepath}")
    
    def generate_report(self) -> Dict:
        """Generate comprehensive performance report."""
        if not self.trades:
            return {'error': 'No trades to analyze'}
        
        returns = [t.get('pnl_pct', 0) for t in self.trades if 'pnl_pct' in t]
        
        sharpe = self.calculate_sharpe_ratio(returns)
        sortino = self.calculate_sortino_ratio(returns)
        mae_mfe = self.calculate_mae_mfe()
        distribution = self.get_trade_distribution()
        rolling = self.calculate_rolling_metrics()
        
        return {
            'total_trades': len(self.trades),
            'sharpe_ratio': round(sharpe, 2),
            'sortino_ratio': round(sortino, 2),
            'mae': round(mae_mfe['mae'], 2),
            'mfe': round(mae_mfe['mfe'], 2),
            'win_rate': round(distribution['wins'] / len(self.trades) * 100, 2) if self.trades else 0,
            'avg_win': round(distribution['avg_win'], 2),
            'avg_loss': round(distribution['avg_loss'], 2),
            'profit_factor': round(abs(distribution['avg_win'] / distribution['avg_loss']), 2) if distribution['avg_loss'] != 0 else 0,
            'rolling_metrics': rolling
        }
