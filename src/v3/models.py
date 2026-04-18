"""Shared data models for CryptoBoss v3.0 microservices."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class SignalOutput:
    action: str
    confidence: float
    reason: str
    direction: str = "neutral"
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskDecision:
    approved: bool
    reason: str
    position_size: float = 0.0
    risk_pct: float = 0.0
    rr_ratio: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionReport:
    accepted: bool
    status: str
    order_id: Optional[str] = None
    action: str = "HOLD"
    symbol: str = ""
    order_type: str = "market"
    requested_price: Optional[float] = None
    filled_price: Optional[float] = None
    slippage: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeRecordV3:
    trade_id: str
    symbol: str
    strategy_used: str
    action: str
    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    reason_for_trade: str
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestSummaryV3:
    win_rate: float
    profit_factor: float
    drawdown: float
    sharpe_ratio: float
    total_trades: int
    net_pnl: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyArtifact:
    strategy_id: str
    name: str
    rules: Dict[str, Any]
    indicators: List[str]
    smc_conditions: List[str]
    risk_rules: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
