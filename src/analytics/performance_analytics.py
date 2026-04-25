"""Trade analytics helpers used by the dashboard API and analytics views."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from src.analysis.performance_analytics import PerformanceAnalyticsEngine


class TradeAnalyticsService:
    """Builds dashboard-ready analytics from raw trade records."""

    def __init__(self, engine: PerformanceAnalyticsEngine | None = None) -> None:
        self.engine = engine or PerformanceAnalyticsEngine()

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            if value in (None, ""):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_datetime(value: Any, fallback: datetime | None = None) -> datetime:
        if isinstance(value, datetime):
            return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
        if isinstance(value, str) and value:
            parsed = pd.to_datetime(value, errors="coerce", utc=True)
            if not pd.isna(parsed):
                return parsed.to_pydatetime()
        return fallback or datetime.now(UTC)

    @staticmethod
    def _normalize_direction(value: Any) -> str:
        text = str(value or "long").strip().lower()
        if text in {"sell", "short"}:
            return "short"
        return "long"

    @staticmethod
    def _session_name(timestamp: datetime) -> str:
        hour = timestamp.astimezone(UTC).hour
        if 7 <= hour < 10:
            return "LONDON"
        if 12 <= hour < 15:
            return "NEW_YORK"
        if hour >= 23 or hour < 2:
            return "ASIA"
        if 13 <= hour < 16:
            return "OVERLAP"
        return "OFF_SESSION"

    def _normalize_trade_row(self, trade: dict[str, Any]) -> dict[str, Any]:
        entry_price = self._to_float(trade.get("entry_price", trade.get("entryPrice", trade.get("price"))))
        exit_price = self._to_float(trade.get("exit_price", trade.get("exitPrice", trade.get("price", entry_price))))
        quantity = self._to_float(trade.get("quantity", trade.get("size", 0.0)))
        pnl = self._to_float(
            trade.get("pnl_usdt", trade.get("realizedPnL", trade.get("realized_pnl", trade.get("net_pnl", trade.get("pnl", 0.0)))))
        )
        closed_at = self._to_datetime(
            trade.get("closed_at", trade.get("exit_time", trade.get("exitTime", trade.get("timestamp"))))
        )
        entry_at = self._to_datetime(
            trade.get("entry_time", trade.get("entryTime", trade.get("opened_at", trade.get("timestamp")))),
            fallback=closed_at,
        )
        hold_minutes = self._to_float(trade.get("hold_minutes"))
        if hold_minutes == 0.0:
            hold_minutes = max((closed_at - entry_at).total_seconds() / 60.0, 0.0)

        notional = entry_price * quantity
        pnl_pct = self._to_float(trade.get("return_pct", trade.get("pnl_pct", trade.get("pnlPercent"))))
        if pnl_pct == 0.0 and notional > 0:
            pnl_pct = (pnl / notional) * 100.0

        strategy = str(trade.get("strategy", trade.get("strategy_id", "unknown")) or "unknown")
        session = str(trade.get("session") or self._session_name(closed_at))

        return {
            "trade_id": str(trade.get("trade_id", trade.get("id", f"trade-{closed_at.timestamp()}"))),
            "symbol": str(trade.get("symbol", "UNKNOWN")),
            "direction": self._normalize_direction(trade.get("direction", trade.get("side"))),
            "strategy": strategy,
            "session": session,
            "regime": str(trade.get("regime", trade.get("market_context", "unknown")) or "unknown"),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "price": exit_price or entry_price,
            "quantity": quantity,
            "pnl_usdt": pnl,
            "return_pct": pnl_pct,
            "rr_achieved": self._to_float(trade.get("rr_achieved", trade.get("rr", trade.get("rrAchieved")))),
            "rr_planned": self._to_float(trade.get("rr_planned", trade.get("planned_rr", trade.get("rrPlanned")))),
            "hold_minutes": hold_minutes,
            "closed_at": closed_at,
            "entry_time": entry_at,
        }

    def trade_frame(self, trades: list[dict[str, Any]]) -> pd.DataFrame:
        """Normalize raw trade records into a dataframe."""
        if not trades:
            return pd.DataFrame(
                columns=[
                    "trade_id",
                    "symbol",
                    "direction",
                    "strategy",
                    "session",
                    "regime",
                    "entry_price",
                    "exit_price",
                    "price",
                    "quantity",
                    "pnl_usdt",
                    "return_pct",
                    "rr_achieved",
                    "rr_planned",
                    "hold_minutes",
                    "closed_at",
                    "entry_time",
                ]
            )

        frame = pd.DataFrame([self._normalize_trade_row(trade) for trade in trades])
        frame["closed_at"] = pd.to_datetime(frame["closed_at"], utc=True, errors="coerce")
        frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True, errors="coerce")
        frame = frame.sort_values("closed_at").reset_index(drop=True)
        frame["hour"] = frame["closed_at"].dt.hour
        frame["weekday"] = frame["closed_at"].dt.dayofweek
        frame["weekday_name"] = frame["closed_at"].dt.day_name().str.slice(0, 3)
        week_source = frame["closed_at"].dt.tz_convert("UTC").dt.tz_localize(None)
        frame["week_start"] = week_source.dt.to_period("W-MON").dt.start_time.dt.tz_localize("UTC")
        return frame

    @staticmethod
    def _trade_card(row: pd.Series | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "trade_id": str(row.get("trade_id", "")),
            "symbol": str(row.get("symbol", "UNKNOWN")),
            "direction": str(row.get("direction", "long")).upper(),
            "pnl_usdt": float(row.get("pnl_usdt", 0.0)),
            "rr_achieved": float(row.get("rr_achieved", 0.0)),
            "duration_minutes": float(row.get("hold_minutes", 0.0)),
            "closed_at": pd.Timestamp(row.get("closed_at")).isoformat(),
        }

    def today_summary(
        self,
        trades: list[dict[str, Any]],
        initial_capital: float = 10000.0,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Return today's trading summary."""
        frame = self.trade_frame(trades)
        current_time = (now or datetime.now(UTC)).astimezone(UTC)
        today_mask = frame["closed_at"].dt.date == current_time.date() if not frame.empty else []
        today_frame = frame.loc[today_mask].copy() if not frame.empty else frame
        today_trades = today_frame.to_dict("records")
        snapshot = self.engine.compute_snapshot(today_trades, initial_capital=initial_capital, timestamp=current_time)

        best_row = today_frame.loc[today_frame["pnl_usdt"].idxmax()] if not today_frame.empty else None
        worst_row = today_frame.loc[today_frame["pnl_usdt"].idxmin()] if not today_frame.empty else None

        return {
            "date": current_time.date().isoformat(),
            "trades": snapshot.total_trades,
            "win_rate": snapshot.metrics["win_rate"],
            "profit_factor": snapshot.metrics["profit_factor"],
            "avg_rr": snapshot.metrics["avg_rr_achieved"],
            "total_pnl_usdt": snapshot.metrics["total_pnl_usdt"],
            "best_trade": self._trade_card(best_row),
            "worst_trade": self._trade_card(worst_row),
            "avg_hold_duration_minutes": snapshot.metrics["avg_hold_duration_minutes"],
        }

    def hourly_performance(
        self,
        trades: list[dict[str, Any]],
        days: int = 30,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Return hourly and heatmap performance for the recent trading window."""
        frame = self.trade_frame(trades)
        current_time = (now or datetime.now(UTC)).astimezone(UTC)
        cutoff = current_time - timedelta(days=days)
        recent = frame[frame["closed_at"] >= cutoff].copy() if not frame.empty else frame

        hourly_rows: list[dict[str, Any]] = []
        for hour in range(24):
            chunk = recent[recent["hour"] == hour]
            trades_count = int(len(chunk))
            hourly_rows.append(
                {
                    "hour": hour,
                    "trades": trades_count,
                    "win_rate": float((chunk["pnl_usdt"] > 0).mean()) if trades_count else 0.0,
                    "total_pnl_usdt": float(chunk["pnl_usdt"].sum()) if trades_count else 0.0,
                    "avg_pnl_usdt": float(chunk["pnl_usdt"].mean()) if trades_count else 0.0,
                }
            )

        heatmap_rows: list[dict[str, Any]] = []
        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for weekday, label in enumerate(day_labels):
            day_chunk = recent[recent["weekday"] == weekday]
            hours: list[dict[str, Any]] = []
            for hour in range(24):
                slot = day_chunk[day_chunk["hour"] == hour]
                trades_count = int(len(slot))
                hours.append(
                    {
                        "hour": hour,
                        "trades": trades_count,
                        "win_rate": float((slot["pnl_usdt"] > 0).mean()) if trades_count else 0.0,
                        "total_pnl_usdt": float(slot["pnl_usdt"].sum()) if trades_count else 0.0,
                    }
                )
            heatmap_rows.append({"day": label, "hours": hours})

        return {
            "window_days": days,
            "hourly": hourly_rows,
            "heatmap": heatmap_rows,
        }

    def symbol_performance(self, trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return per-symbol trade stats."""
        frame = self.trade_frame(trades)
        if frame.empty:
            return []

        rows: list[dict[str, Any]] = []
        for symbol, chunk in frame.groupby("symbol", dropna=False):
            rows.append(
                {
                    "symbol": str(symbol),
                    "trades": int(len(chunk)),
                    "win_rate": float((chunk["pnl_usdt"] > 0).mean()),
                    "total_pnl_usdt": float(chunk["pnl_usdt"].sum()),
                    "avg_pnl_usdt": float(chunk["pnl_usdt"].mean()),
                    "avg_rr": float(chunk["rr_achieved"].mean()) if not chunk.empty else 0.0,
                }
            )
        return sorted(rows, key=lambda item: item["total_pnl_usdt"], reverse=True)

    def strategy_breakdown(self, trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return per-strategy trade stats."""
        frame = self.trade_frame(trades)
        if frame.empty:
            return []

        rows: list[dict[str, Any]] = []
        for strategy, chunk in frame.groupby("strategy", dropna=False):
            wins = chunk[chunk["pnl_usdt"] > 0]["pnl_usdt"].sum()
            losses = abs(chunk[chunk["pnl_usdt"] < 0]["pnl_usdt"].sum())
            rows.append(
                {
                    "strategy": str(strategy),
                    "trades": int(len(chunk)),
                    "win_rate": float((chunk["pnl_usdt"] > 0).mean()),
                    "profit_factor": float(wins / losses) if losses else 0.0,
                    "total_pnl_usdt": float(chunk["pnl_usdt"].sum()),
                    "avg_rr": float(chunk["rr_achieved"].mean()) if not chunk.empty else 0.0,
                }
            )
        return sorted(rows, key=lambda item: item["total_pnl_usdt"], reverse=True)

    def weekly_equity(
        self,
        trades: list[dict[str, Any]],
        initial_capital: float = 10000.0,
    ) -> dict[str, Any]:
        """Return weekly equity curve data."""
        frame = self.trade_frame(trades)
        if frame.empty:
            return {"initial_capital": initial_capital, "points": []}

        weekly = (
            frame.groupby("week_start", dropna=False)["pnl_usdt"]
            .sum()
            .reset_index(name="weekly_pnl_usdt")
            .sort_values("week_start")
            .reset_index(drop=True)
        )
        weekly["cumulative_pnl_usdt"] = weekly["weekly_pnl_usdt"].cumsum()
        weekly["equity"] = initial_capital + weekly["cumulative_pnl_usdt"]

        return {
            "initial_capital": initial_capital,
            "points": [
                {
                    "week_start": pd.Timestamp(row.week_start).isoformat(),
                    "weekly_pnl_usdt": float(row.weekly_pnl_usdt),
                    "cumulative_pnl_usdt": float(row.cumulative_pnl_usdt),
                    "equity": float(row.equity),
                }
                for row in weekly.itertuples(index=False)
            ],
        }

    def drawdown_periods(
        self,
        trades: list[dict[str, Any]],
        initial_capital: float = 10000.0,
    ) -> list[dict[str, Any]]:
        """Return drawdown periods derived from the closed-trade equity curve."""
        frame = self.trade_frame(trades)
        if frame.empty:
            return []

        equity = initial_capital + frame["pnl_usdt"].cumsum()
        peaks = equity.cummax()
        drawdown_pct = ((equity - peaks) / peaks.replace(0, pd.NA)) * 100.0

        periods: list[dict[str, Any]] = []
        in_drawdown = False
        start_index = 0
        worst_depth = 0.0

        for index, value in enumerate(drawdown_pct.fillna(0.0).tolist()):
            if value < 0 and not in_drawdown:
                in_drawdown = True
                start_index = index
                worst_depth = value
            elif value < 0 and in_drawdown:
                worst_depth = min(worst_depth, value)
            elif value >= 0 and in_drawdown:
                start_row = frame.iloc[start_index]
                end_row = frame.iloc[index - 1]
                periods.append(
                    {
                        "start": pd.Timestamp(start_row["closed_at"]).isoformat(),
                        "end": pd.Timestamp(end_row["closed_at"]).isoformat(),
                        "depth_pct": abs(float(worst_depth)),
                        "duration_trades": int(index - start_index),
                    }
                )
                in_drawdown = False

        if in_drawdown:
            start_row = frame.iloc[start_index]
            end_row = frame.iloc[len(frame) - 1]
            periods.append(
                {
                    "start": pd.Timestamp(start_row["closed_at"]).isoformat(),
                    "end": pd.Timestamp(end_row["closed_at"]).isoformat(),
                    "depth_pct": abs(float(worst_depth)),
                    "duration_trades": int(len(frame) - start_index),
                }
            )

        return sorted(periods, key=lambda item: item["depth_pct"], reverse=True)
