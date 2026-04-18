"""Performance tracker service for v3 microservice architecture."""

from __future__ import annotations

import json
import os
import sqlite3
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
        self._db_path = os.getenv("V3_PERF_DB_PATH", "data/v3_trades.db")
        self._init_db()
        self._load_open_trades()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_db(self) -> None:
        directory = os.path.dirname(self._db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS v3_trades (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    strategy_used TEXT NOT NULL,
                    action TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    reason_for_trade TEXT,
                    exit_time TEXT,
                    exit_price REAL,
                    pnl REAL,
                    exit_reason TEXT,
                    metadata_json TEXT
                )
                """
            )
            conn.commit()

    def _load_open_trades(self) -> None:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    trade_id,
                    symbol,
                    strategy_used,
                    action,
                    entry_time,
                    entry_price,
                    stop_loss,
                    take_profit,
                    reason_for_trade,
                    metadata_json
                FROM v3_trades
                WHERE exit_time IS NULL
                """
            ).fetchall()

        for row in rows:
            metadata = {}
            if row[9]:
                try:
                    metadata = json.loads(row[9])
                except json.JSONDecodeError:
                    metadata = {}

            self._trades[row[0]] = TradeRecordV3(
                trade_id=row[0],
                symbol=row[1],
                strategy_used=row[2],
                action=row[3],
                entry_time=datetime.fromisoformat(row[4]),
                entry_price=float(row[5]),
                stop_loss=float(row[6]),
                take_profit=float(row[7]),
                reason_for_trade=row[8],
                metadata=metadata,
            )

    @staticmethod
    def _metadata_json(metadata: Optional[Dict[str, object]]) -> str:
        return json.dumps(metadata or {}, default=str)

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

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO v3_trades (
                    trade_id,
                    symbol,
                    strategy_used,
                    action,
                    entry_time,
                    entry_price,
                    stop_loss,
                    take_profit,
                    reason_for_trade,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade.trade_id,
                    trade.symbol,
                    trade.strategy_used,
                    trade.action,
                    trade.entry_time.isoformat(),
                    float(trade.entry_price),
                    float(trade.stop_loss),
                    float(trade.take_profit),
                    trade.reason_for_trade,
                    self._metadata_json(trade.metadata),
                ),
            )
            conn.commit()

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

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE v3_trades
                SET
                    exit_time = ?,
                    exit_price = ?,
                    pnl = ?,
                    exit_reason = ?,
                    metadata_json = ?
                WHERE trade_id = ?
                """,
                (
                    trade.exit_time.isoformat(),
                    float(exit_price),
                    float(trade.pnl),
                    reason_for_trade,
                    self._metadata_json(trade.metadata),
                    trade_id,
                ),
            )
            conn.commit()

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
