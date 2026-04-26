"""Performance analytics engine for rolling portfolio and trade metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from statistics import mean
from typing import Any

import pandas as pd


@dataclass(slots=True)
class PerformanceSnapshot:
    """Single analytics snapshot."""

    timestamp: datetime
    total_trades: int
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


class PerformanceAnalyticsEngine:
    """Computes production-grade performance metrics from trades."""

    def __init__(self, rolling_windows: list[int] | None = None) -> None:
        self.rolling_windows = rolling_windows or [5, 10, 20, 50]

    def compute_snapshot(
        self,
        trades: list[dict[str, Any]],
        initial_capital: float = 10000.0,
        timestamp: datetime | None = None,
    ) -> PerformanceSnapshot:
        """Compute a full metrics snapshot from closed trades."""
        ts = timestamp or datetime.utcnow()
        trade_df = self._normalize_trade_frame(trades)

        if trade_df.empty:
            empty_metrics = {
                "total_pnl_usdt": 0.0,
                "total_pnl_pct": 0.0,
                "win_rate": 0.0,
                "loss_rate": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown_pct": 0.0,
                "max_drawdown_duration_bars": 0,
                "calmar_ratio": 0.0,
                "avg_rr_achieved": 0.0,
                "avg_rr_planned": 0.0,
                "expectancy_per_trade": 0.0,
                "best_trade_usdt": 0.0,
                "worst_trade_usdt": 0.0,
                "consecutive_wins_max": 0,
                "consecutive_losses_max": 0,
                "long_win_rate": 0.0,
                "short_win_rate": 0.0,
                "pnl_by_symbol": {},
                "pnl_by_strategy": {},
                "pnl_by_session": {},
                "pnl_by_regime": {},
                "avg_hold_duration_minutes": 0.0,
                "trades_per_day_avg": 0.0,
                "monthly_returns": {},
            }
            return PerformanceSnapshot(timestamp=ts, total_trades=0, metrics=empty_metrics)

        total_pnl = float(trade_df["pnl_usdt"].sum())
        total_trades = int(len(trade_df))

        wins = trade_df[trade_df["pnl_usdt"] > 0]
        losses = trade_df[trade_df["pnl_usdt"] < 0]

        win_rate = float(len(wins) / total_trades)
        loss_rate = float(len(losses) / total_trades)

        avg_win = float(wins["pnl_usdt"].mean()) if not wins.empty else 0.0
        avg_loss = float(losses["pnl_usdt"].mean()) if not losses.empty else 0.0

        gross_profit = float(wins["pnl_usdt"].sum()) if not wins.empty else 0.0
        gross_loss = abs(float(losses["pnl_usdt"].sum())) if not losses.empty else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)

        returns = trade_df["return_pct"].astype(float).tolist()
        sharpe = self._sharpe_ratio(returns)
        sortino = self._sortino_ratio(returns)

        equity_curve = initial_capital + trade_df["pnl_usdt"].cumsum()
        max_dd_pct, dd_duration = self._max_drawdown_stats(equity_curve)
        calmar = self._safe_div((total_pnl / initial_capital), max_dd_pct / 100.0) if max_dd_pct > 0 else 0.0

        rr_achieved = float(trade_df["rr_achieved"].mean()) if "rr_achieved" in trade_df else 0.0
        rr_planned = float(trade_df["rr_planned"].mean()) if "rr_planned" in trade_df else 0.0

        long_trades = trade_df[trade_df["direction"] == "long"]
        short_trades = trade_df[trade_df["direction"] == "short"]

        long_win_rate = (
            float((long_trades["pnl_usdt"] > 0).mean()) if not long_trades.empty else 0.0
        )
        short_win_rate = (
            float((short_trades["pnl_usdt"] > 0).mean()) if not short_trades.empty else 0.0
        )

        metrics: dict[str, Any] = {
            "total_pnl_usdt": total_pnl,
            "total_pnl_pct": (total_pnl / initial_capital) * 100.0,
            "win_rate": win_rate,
            "loss_rate": loss_rate,
            "total_trades": total_trades,
            "avg_win_usdt": avg_win,
            "avg_loss_usdt": avg_loss,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown_pct": max_dd_pct,
            "max_drawdown_duration_bars": dd_duration,
            "calmar_ratio": calmar,
            "avg_rr_achieved": rr_achieved,
            "avg_rr_planned": rr_planned,
            "expectancy_per_trade": expectancy,
            "best_trade_usdt": float(trade_df["pnl_usdt"].max()),
            "worst_trade_usdt": float(trade_df["pnl_usdt"].min()),
            "consecutive_wins_max": self._max_streak(trade_df["pnl_usdt"].tolist(), positive=True),
            "consecutive_losses_max": self._max_streak(trade_df["pnl_usdt"].tolist(), positive=False),
            "long_win_rate": long_win_rate,
            "short_win_rate": short_win_rate,
            "pnl_by_symbol": self._group_sum(trade_df, "symbol", "pnl_usdt"),
            "pnl_by_strategy": self._group_sum(trade_df, "strategy", "pnl_usdt"),
            "pnl_by_session": self._group_sum(trade_df, "session", "pnl_usdt"),
            "pnl_by_regime": self._group_sum(trade_df, "regime", "pnl_usdt"),
            "avg_hold_duration_minutes": float(trade_df["hold_minutes"].mean()),
            "trades_per_day_avg": self._trades_per_day(trade_df),
            "monthly_returns": self._monthly_returns(trade_df, initial_capital),
            "rolling": self._rolling_metrics(trade_df),
        }

        return PerformanceSnapshot(timestamp=ts, total_trades=total_trades, metrics=metrics)

    @staticmethod
    def _normalize_trade_frame(trades: list[dict[str, Any]]) -> pd.DataFrame:
        if not trades:
            return pd.DataFrame()

        df = pd.DataFrame(trades).copy()

        if "pnl_usdt" not in df.columns:
            df["pnl_usdt"] = 0.0
        if "return_pct" not in df.columns:
            # If missing explicit return, infer from planned_rr and pnl sign where possible.
            df["return_pct"] = (df["pnl_usdt"].astype(float) / 100.0).fillna(0.0)

        if "direction" not in df.columns:
            df["direction"] = "long"
        df["direction"] = df["direction"].astype(str).str.lower()

        for key in ["symbol", "strategy", "session", "regime"]:
            if key not in df.columns:
                df[key] = "unknown"
            df[key] = df[key].fillna("unknown").astype(str)

        if "hold_minutes" not in df.columns:
            df["hold_minutes"] = 0.0
        df["hold_minutes"] = pd.to_numeric(df["hold_minutes"], errors="coerce").fillna(0.0)

        if "rr_achieved" not in df.columns:
            df["rr_achieved"] = 0.0
        if "rr_planned" not in df.columns:
            df["rr_planned"] = 0.0

        # Normalize close timestamp for monthly/day groupings.
        if "closed_at" not in df.columns:
            if "timestamp" in df.columns:
                df["closed_at"] = df["timestamp"]
            else:
                df["closed_at"] = datetime.utcnow().isoformat()

        df["closed_at"] = pd.to_datetime(df["closed_at"], errors="coerce")
        df = df.sort_values("closed_at").reset_index(drop=True)

        return df

    @staticmethod
    def _safe_div(numerator: float, denominator: float) -> float:
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def _rolling_metrics(self, trade_df: pd.DataFrame) -> dict[str, dict[str, float]]:
        rolling: dict[str, dict[str, float]] = {}

        for window in self.rolling_windows:
            chunk = trade_df.tail(window)
            if chunk.empty:
                rolling[str(window)] = {
                    "win_rate": 0.0,
                    "avg_pnl": 0.0,
                    "profit_factor": 0.0,
                }
                continue

            wins = chunk[chunk["pnl_usdt"] > 0]
            losses = chunk[chunk["pnl_usdt"] < 0]
            gp = float(wins["pnl_usdt"].sum()) if not wins.empty else 0.0
            gl = abs(float(losses["pnl_usdt"].sum())) if not losses.empty else 0.0

            rolling[str(window)] = {
                "win_rate": float((chunk["pnl_usdt"] > 0).mean()),
                "avg_pnl": float(chunk["pnl_usdt"].mean()),
                "profit_factor": self._safe_div(gp, gl),
            }

        return rolling

    @staticmethod
    def _group_sum(df: pd.DataFrame, group_col: str, value_col: str) -> dict[str, float]:
        grouped = df.groupby(group_col, dropna=False)[value_col].sum().to_dict()
        return {str(k): float(v) for k, v in grouped.items()}

    @staticmethod
    def _max_streak(values: list[float], positive: bool) -> int:
        best = 0
        curr = 0
        for value in values:
            condition = value > 0 if positive else value < 0
            if condition:
                curr += 1
                if curr > best:
                    best = curr
            else:
                curr = 0
        return best

    @staticmethod
    def _sharpe_ratio(returns: list[float]) -> float:
        if len(returns) < 2:
            return 0.0
        avg_ret = mean(returns)
        std = pd.Series(returns).std(ddof=1)
        if std == 0 or pd.isna(std):
            return 0.0
        return float((avg_ret / std) * (len(returns) ** 0.5))

    @staticmethod
    def _sortino_ratio(returns: list[float]) -> float:
        if len(returns) < 2:
            return 0.0
        avg_ret = mean(returns)
        downside = [r for r in returns if r < 0]
        if not downside:
            return 0.0
        downside_std = pd.Series(downside).std(ddof=1)
        if downside_std == 0 or pd.isna(downside_std):
            return 0.0
        return float((avg_ret / downside_std) * (len(returns) ** 0.5))

    @staticmethod
    def _max_drawdown_stats(equity_curve: pd.Series) -> tuple[float, int]:
        running_peak = equity_curve.cummax()
        drawdown = ((equity_curve - running_peak) / running_peak) * 100.0
        max_dd = abs(float(drawdown.min())) if not drawdown.empty else 0.0

        duration = 0
        best_duration = 0
        for value in drawdown.tolist():
            if value < 0:
                duration += 1
                if duration > best_duration:
                    best_duration = duration
            else:
                duration = 0

        return max_dd, best_duration

    @staticmethod
    def _trades_per_day(df: pd.DataFrame) -> float:
        by_day = df.groupby(df["closed_at"].dt.date).size()
        if by_day.empty:
            return 0.0
        return float(by_day.mean())

    @staticmethod
    def _monthly_returns(df: pd.DataFrame, initial_capital: float) -> dict[str, float]:
        monthly_source = df["closed_at"]
        if getattr(monthly_source.dt, "tz", None) is not None:
            monthly_source = monthly_source.dt.tz_convert("UTC").dt.tz_localize(None)
        monthly_pnl = df.groupby(monthly_source.dt.to_period("M"))["pnl_usdt"].sum()
        return {
            str(period): float((pnl / initial_capital) * 100.0)
            for period, pnl in monthly_pnl.items()
        }
