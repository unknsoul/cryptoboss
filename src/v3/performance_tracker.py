"""Performance tracker service for v3 microservice architecture."""

from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional

from .config import PerformanceTrackerConfig
from .models import TradeRecordV3


class PerformanceTracker:
    """Tracks trade lifecycle and computes real-time strategy statistics."""

    def __init__(self, config: Optional[PerformanceTrackerConfig] = None):
        self.config = config or PerformanceTrackerConfig()
        self._trades: Dict[str, TradeRecordV3] = {}
        self._logs: List[Dict[str, object]] = []

    def log_trade_entry(
        self,
        symbol: str,
        strategy_used: str,
        action: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        reason_for_trade: str,
        metadata: Optional[Dict[str, object]] = None,
    ) -> str:
        trade_id = str(uuid.uuid4())[:12]
        trade = TradeRecordV3(
            trade_id=trade_id,
            symbol=symbol,
            strategy_used=strategy_used,
            action=action,
            entry_time=datetime.utcnow(),
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason_for_trade=reason_for_trade,
            metadata=metadata or {},
        )
        self._trades[trade_id] = trade

        self._logs.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "event": "trade_entry",
                "trade_id": trade_id,
                "symbol": symbol,
                "strategy_used": strategy_used,
                "reason_for_trade": reason_for_trade,
            }
        )
        return trade_id

    def log_trade_exit(
        self,
        trade_id: str,
        exit_price: float,
        reason_for_trade: str,
        metadata: Optional[Dict[str, object]] = None,
    ) -> bool:
        trade = self._trades.get(trade_id)
        if trade is None:
            return False

        trade.exit_time = datetime.utcnow()
        trade.exit_price = exit_price
        if trade.action == "BUY":
            trade.pnl = exit_price - trade.entry_price
        else:
            trade.pnl = trade.entry_price - exit_price
        trade.metadata.update(metadata or {})

        self._logs.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "event": "trade_exit",
                "trade_id": trade_id,
                "symbol": trade.symbol,
                "strategy_used": trade.strategy_used,
                "reason_for_trade": reason_for_trade,
                "pnl": trade.pnl,
            }
        )
        return True

    def stats(self) -> Dict[str, object]:
        closed = [trade for trade in self._trades.values() if trade.exit_time is not None and trade.pnl is not None]
        wins = [trade for trade in closed if float(trade.pnl or 0.0) > 0]
        losses = [trade for trade in closed if float(trade.pnl or 0.0) <= 0]

        total = len(closed)
        win_rate = (len(wins) / total) if total else 0.0
        gross_profit = sum(float(trade.pnl or 0.0) for trade in wins)
        gross_loss = abs(sum(float(trade.pnl or 0.0) for trade in losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0

        return {
            "total_trades": total,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "net_pnl": gross_profit - gross_loss,
            "open_trades": len([trade for trade in self._trades.values() if trade.exit_time is None]),
            "real_time_stats": self.config.real_time_stats,
        }

    def dashboard_snapshot(self) -> Dict[str, object]:
        return {
            "dashboard": self.config.dashboard,
            "stats": self.stats(),
            "recent_logs": self._logs[-100:],
        }

    def trade_history(self) -> List[Dict[str, object]]:
        return [asdict(trade) for trade in self._trades.values()]

    def logs(self) -> List[Dict[str, object]]:
        return list(self._logs)
