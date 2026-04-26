"""Aggressive scalper strategy with weighted confluence scoring and live sizing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from .base_strategy import BaseStrategy, SignalResult, StrategyConfig
from ..risk.scalper_risk_engine import ScalperRiskConfig, ScalperRiskEngine

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "aggressive_scalper.yaml"


@dataclass
class ScalperParams:
    """Runtime parameters for the aggressive scalper."""

    rsi_period: int = 7
    rsi_fast_period: int = 4
    rsi_oversold: float = 28.0
    rsi_overbought: float = 72.0
    macd_fast: int = 6
    macd_slow: int = 13
    macd_signal: int = 4
    ema_fast: int = 8
    ema_slow: int = 21
    volume_ma_period: int = 14
    volume_spike_threshold: float = 1.4
    adx_period: int = 10
    adx_min_trend: float = 18.0
    atr_period: int = 10
    stop_loss_atr_mult: float = 0.6
    take_profit_1_mult: float = 1.5
    take_profit_2_mult: float = 2.8
    take_profit_3_mult: float = 4.2
    max_trades_per_hour: int = 8
    max_concurrent_positions: int = 4
    daily_loss_halt_pct: float = 4.0
    weekly_loss_halt_pct: float = 8.0
    cooldown_seconds: int = 45
    min_confluence: int = 3
    max_position_pct: float = 8.0
    max_risk_pct_per_trade: float = 0.01
    auto_partial_close: bool = True
    partial_close_pct_at_tp1: float = 0.5
    move_sl_to_be_at_tp1: bool = True
    trailing_stop_after_tp1: bool = True
    trailing_atr_mult: float = 1.0


class AggressiveScalper(BaseStrategy):
    """Fast scalper that blends momentum, trend, and volume into entry signals."""

    STRATEGY_ID = "aggressive_scalper_v1"
    VERSION = "2.0"
    MIN_ROWS = 30

    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        params: Optional[ScalperParams] = None,
    ) -> None:
        """Initialize strategy state and load runtime configuration."""
        file_payload = self._load_file_config()
        resolved_params = params or self._build_params(file_payload)
        resolved_config = config or self._build_strategy_config(file_payload, resolved_params)
        super().__init__(resolved_config)

        self.params = resolved_params
        self.risk_engine = ScalperRiskEngine(
            ScalperRiskConfig(
                max_risk_pct=self.params.max_risk_pct_per_trade,
                max_daily_loss_pct=self.params.daily_loss_halt_pct / 100.0,
                max_positions=self.params.max_concurrent_positions,
                partial_exit_pct=self.params.partial_close_pct_at_tp1,
                break_even_rr_trigger=1.0,
                trailing_atr_mult=self.params.trailing_atr_mult,
            )
        )
        self._account_balance = 10000.0
        self._daily_pnl = 0.0
        self._weekly_pnl = 0.0
        self._trade_timestamps: List[datetime] = []
        self._last_signal_at: Optional[datetime] = None
        self._last_watch_signal: Dict[str, Any] = {}
        self._last_signal_snapshot: Dict[str, Any] = {}

    @staticmethod
    def _load_file_config() -> Dict[str, Any]:
        """Load the strategy YAML config if it exists."""
        if not _CONFIG_PATH.exists():
            return {}

        try:
            payload = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            logger.warning("Failed to load aggressive scalper config: %s", exc)
            return {}

        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _build_strategy_config(payload: Dict[str, Any], params: ScalperParams) -> StrategyConfig:
        """Build the shared base strategy config from YAML payload."""
        strategy_payload = payload.get("strategy", {}) if isinstance(payload, dict) else {}
        risk_payload = payload.get("risk", {}) if isinstance(payload, dict) else {}
        guards_payload = payload.get("guards", {}) if isinstance(payload, dict) else {}
        symbols = payload.get("symbols", ["BTC/USDT"]) if isinstance(payload, dict) else ["BTC/USDT"]
        primary_symbol = symbols[0] if symbols else "BTC/USDT"

        return StrategyConfig(
            strategy_id=str(strategy_payload.get("id", AggressiveScalper.STRATEGY_ID)),
            version=str(strategy_payload.get("version", AggressiveScalper.VERSION)),
            symbol=str(primary_symbol),
            enabled=bool(strategy_payload.get("enabled", True)),
            max_position_pct=float(risk_payload.get("max_position_pct", params.max_position_pct)),
            min_confidence=0.55,
            cooldown_seconds=int(guards_payload.get("cooldown_seconds", params.cooldown_seconds)),
            metadata={
                "symbols": symbols,
                "timeframe": payload.get("timeframe", "5m"),
                "secondary_timeframe": payload.get("secondary_timeframe", "15m"),
                "leverage": float(risk_payload.get("leverage", 1.0)),
            },
        )

    @staticmethod
    def _build_params(payload: Dict[str, Any]) -> ScalperParams:
        """Build scalper parameters from the YAML payload."""
        rsi_payload = payload.get("rsi", {}) if isinstance(payload, dict) else {}
        macd_payload = payload.get("macd", {}) if isinstance(payload, dict) else {}
        ema_payload = payload.get("ema", {}) if isinstance(payload, dict) else {}
        atr_payload = payload.get("atr", {}) if isinstance(payload, dict) else {}
        volume_payload = payload.get("volume", {}) if isinstance(payload, dict) else {}
        adx_payload = payload.get("adx", {}) if isinstance(payload, dict) else {}
        risk_payload = payload.get("risk", {}) if isinstance(payload, dict) else {}
        guards_payload = payload.get("guards", {}) if isinstance(payload, dict) else {}
        scaling_payload = payload.get("scaling", {}) if isinstance(payload, dict) else {}

        return ScalperParams(
            rsi_period=int(rsi_payload.get("period", 7)),
            rsi_fast_period=int(rsi_payload.get("fast_period", 4)),
            rsi_oversold=float(rsi_payload.get("oversold", 28.0)),
            rsi_overbought=float(rsi_payload.get("overbought", 72.0)),
            macd_fast=int(macd_payload.get("fast", 6)),
            macd_slow=int(macd_payload.get("slow", 13)),
            macd_signal=int(macd_payload.get("signal", 4)),
            ema_fast=int(ema_payload.get("fast", 8)),
            ema_slow=int(ema_payload.get("slow", 21)),
            volume_ma_period=int(volume_payload.get("ma_period", 14)),
            volume_spike_threshold=float(volume_payload.get("spike_threshold", 1.4)),
            adx_period=int(adx_payload.get("period", 10)),
            adx_min_trend=float(adx_payload.get("min_trend", 18.0)),
            atr_period=int(atr_payload.get("period", 10)),
            stop_loss_atr_mult=float(risk_payload.get("stop_loss_atr_mult", risk_payload.get("stop_loss_pct", 0.6))),
            take_profit_1_mult=float(risk_payload.get("take_profit_1_mult", risk_payload.get("take_profit_pct", 1.5))),
            take_profit_2_mult=float(risk_payload.get("take_profit_2_mult", 2.8)),
            take_profit_3_mult=float(risk_payload.get("take_profit_3_mult", 4.2)),
            max_trades_per_hour=int(guards_payload.get("max_trades_per_hour", 8)),
            max_concurrent_positions=int(guards_payload.get("max_concurrent_positions", 4)),
            daily_loss_halt_pct=float(guards_payload.get("daily_loss_halt_pct", 4.0)),
            weekly_loss_halt_pct=float(guards_payload.get("weekly_loss_halt_pct", 8.0)),
            cooldown_seconds=int(guards_payload.get("cooldown_seconds", 45)),
            min_confluence=int(guards_payload.get("min_confluence", 3)),
            max_position_pct=float(risk_payload.get("max_position_pct", 8.0)),
            max_risk_pct_per_trade=float(risk_payload.get("max_risk_pct_per_trade", 0.01)),
            auto_partial_close=bool(scaling_payload.get("auto_partial_close", True)),
            partial_close_pct_at_tp1=float(scaling_payload.get("partial_close_pct_at_tp1", 0.5)),
            move_sl_to_be_at_tp1=bool(scaling_payload.get("move_sl_to_be_at_tp1", True)),
            trailing_stop_after_tp1=bool(scaling_payload.get("trailing_stop_after_tp1", True)),
            trailing_atr_mult=float(scaling_payload.get("trailing_atr_mult", 1.0)),
        )

    def set_account_balance(self, balance: float) -> None:
        """Update the capital reference used for risk sizing and loss halts."""
        if balance > 0:
            self._account_balance = float(balance)

    def set_daily_pnl(self, pnl: float) -> None:
        """Update current daily profit and loss."""
        self._daily_pnl = float(pnl)

    def set_weekly_pnl(self, pnl: float) -> None:
        """Update current weekly profit and loss."""
        self._weekly_pnl = float(pnl)

    def _slice_frame(self, df: pd.DataFrame, i: Optional[int]) -> pd.DataFrame:
        """Return the portion of the market frame visible to the strategy."""
        if i is None or i < 0:
            return df.copy()
        return df.iloc[: i + 1].copy()

    def _trim_trade_timestamps(self, now: datetime) -> None:
        """Keep only trade timestamps from the rolling one-hour window."""
        threshold = now.timestamp() - 3600.0
        self._trade_timestamps = [
            ts for ts in self._trade_timestamps if ts.timestamp() >= threshold
        ]

    def _cooldown_remaining_seconds(self, now: datetime) -> int:
        """Return the remaining cooldown before another entry can fire."""
        if self._last_signal_at is None:
            return 0
        elapsed = (now - self._last_signal_at).total_seconds()
        return max(int(self.params.cooldown_seconds - elapsed), 0)

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        """Compute an RSI series using simple rolling averages."""
        delta = series.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta.clip(upper=0.0)).abs()
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0.0, pd.NA)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)

    @staticmethod
    def _ema(series: pd.Series, period: int) -> pd.Series:
        """Compute an exponential moving average."""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 10) -> pd.Series:
        """Compute Average True Range for the supplied market frame."""
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(window=period, min_periods=period).mean()

    def _macd_histogram(self, close: pd.Series) -> pd.Series:
        """Compute MACD histogram values for the close series."""
        ema_fast = self._ema(close, self.params.macd_fast)
        ema_slow = self._ema(close, self.params.macd_slow)
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=self.params.macd_signal, adjust=False).mean()
        return macd - signal

    @staticmethod
    def _adx(df: pd.DataFrame, period: int) -> pd.Series:
        """Compute a lightweight ADX series for trend strength scoring."""
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)

        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(window=period, min_periods=period).mean()
        plus_di = 100.0 * plus_dm.rolling(window=period, min_periods=period).mean() / atr.replace(0.0, pd.NA)
        minus_di = 100.0 * minus_dm.rolling(window=period, min_periods=period).mean() / atr.replace(0.0, pd.NA)
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, pd.NA)) * 100.0
        return dx.rolling(window=period, min_periods=period).mean().fillna(0.0)

    def _ema_crossover_signal(self, close: pd.Series) -> int:
        """Return the most recent EMA crossover direction."""
        if len(close) < self.params.ema_slow + 2:
            return 0
        ema_fast = self._ema(close, self.params.ema_fast)
        ema_slow = self._ema(close, self.params.ema_slow)
        previous = float(ema_fast.iloc[-2] - ema_slow.iloc[-2])
        current = float(ema_fast.iloc[-1] - ema_slow.iloc[-1])
        if previous <= 0 and current > 0:
            return 1
        if previous >= 0 and current < 0:
            return -1
        return 0

    def _rsi_momentum(self, close: pd.Series) -> int:
        """Return short-term RSI momentum direction."""
        rsi_fast = self._rsi(close, self.params.rsi_fast_period)
        if len(rsi_fast) < 2:
            return 0
        current = float(rsi_fast.iloc[-1])
        previous = float(rsi_fast.iloc[-2])
        if current < 35.0 and current > previous:
            return 1
        if current > 65.0 and current < previous:
            return -1
        return 0

    @staticmethod
    def _volume_delta(df: pd.DataFrame) -> float:
        """Estimate directional volume pressure from the latest candle."""
        if df.empty:
            return 0.0
        candle = df.iloc[-1]
        candle_range = float(candle["high"]) - float(candle["low"]) + 0.0001
        body_gain = max(float(candle["close"]) - float(candle["open"]), 0.0)
        volume = float(candle.get("volume", 0.0) or 0.0)
        if volume <= 0:
            return 0.0
        buy_volume = volume * (body_gain / candle_range)
        sell_volume = max(volume - buy_volume, 0.0)
        return (buy_volume - sell_volume) / volume

    @staticmethod
    def _price_structure(close: pd.Series, lookback: int = 10) -> int:
        """Estimate whether price is trending up, down, or sideways."""
        if len(close) < lookback + 1:
            return 0
        window = close.tail(lookback).astype(float)
        diff = window.diff().dropna()
        if diff.empty:
            return 0
        higher_lows = all(a <= b for a, b in zip(window[:-1], window[1:]))
        lower_highs = all(a >= b for a, b in zip(window[:-1], window[1:]))
        if higher_lows and diff.gt(0).sum() >= len(diff) * 0.7:
            return 1
        if lower_highs and diff.lt(0).sum() >= len(diff) * 0.7:
            return -1
        return 0

    def _build_hold(self, reason: str, metadata: Optional[Dict[str, Any]] = None) -> SignalResult:
        """Build a HOLD signal with optional metadata."""
        snapshot = metadata.copy() if metadata else {}
        snapshot.setdefault("status", "WAIT")
        self._last_signal_snapshot = snapshot
        return SignalResult(action="HOLD", reason=reason, confidence=0.0, metadata=snapshot)

    def _finalize_signal(
        self,
        action: str,
        current_price: float,
        confidence: float,
        atr_value: float,
        metadata: Dict[str, Any],
        now: datetime,
    ) -> SignalResult:
        """Create a fully sized entry signal from the computed confluence state."""
        is_buy = action == "BUY"
        stop_distance = atr_value * self.params.stop_loss_atr_mult
        stop_loss = current_price - stop_distance if is_buy else current_price + stop_distance
        tp1 = current_price + (atr_value * self.params.take_profit_1_mult) if is_buy else current_price - (atr_value * self.params.take_profit_1_mult)
        tp2 = current_price + (atr_value * self.params.take_profit_2_mult) if is_buy else current_price - (atr_value * self.params.take_profit_2_mult)
        tp3 = current_price + (atr_value * self.params.take_profit_3_mult) if is_buy else current_price - (atr_value * self.params.take_profit_3_mult)

        size = self.risk_engine.compute_position_size(
            account_balance=self._account_balance,
            entry_price=current_price,
            stop_loss=stop_loss,
        )
        max_size = (self._account_balance * (self.params.max_position_pct / 100.0)) / max(current_price, 0.0001)
        size = round(min(size, max_size), 6)

        metadata.update(
            {
                "status": "CONFIRMED",
                "tp1": round(tp1, 6),
                "tp2": round(tp2, 6),
                "tp3": round(tp3, 6),
                "atr": round(atr_value, 6),
            }
        )
        self._trade_timestamps.append(now)
        self._last_signal_at = now
        self._last_signal_snapshot = metadata.copy()

        return SignalResult(
            action=action,
            reason=f"{action.lower()}_signal_confirmed",
            confidence=round(confidence, 4),
            size=size,
            price=round(current_price, 6),
            stop_loss=round(stop_loss, 6),
            take_profit=round(tp2, 6),
            signal_strength=round(confidence, 4),
            metadata=metadata,
        )

    def generate_signal(
        self,
        df: pd.DataFrame,
        i: Optional[int] = None,
        current_price: Optional[float] = None,
    ) -> SignalResult:
        """Generate a live scalping signal from the latest available candles."""
        if df is None or df.empty:
            return self._build_hold("insufficient_data")

        frame = self._slice_frame(df, i)
        if len(frame) < self.MIN_ROWS:
            return self._build_hold("insufficient_data")

        now = datetime.utcnow()
        self._trim_trade_timestamps(now)

        if not self.config.enabled:
            return self._build_hold("strategy_disabled")

        if self.risk_engine.daily_loss_halt(
            self._daily_pnl,
            self._account_balance,
            self.params.daily_loss_halt_pct,
        ):
            return self._build_hold("daily_loss_halt")

        if len(self._trade_timestamps) >= self.params.max_trades_per_hour:
            return self._build_hold("hourly_trade_cap_reached")

        cooldown_remaining = self._cooldown_remaining_seconds(now)
        if cooldown_remaining > 0:
            return self._build_hold(
                "cooldown_active",
                {"cooldown_remaining_seconds": cooldown_remaining},
            )

        close = frame["close"].astype(float)
        price_now = float(current_price if current_price is not None else close.iloc[-1])
        volume = frame["volume"].astype(float) if "volume" in frame.columns else pd.Series([0.0] * len(frame))
        rsi = self._rsi(close, self.params.rsi_period)
        rsi_fast = self._rsi(close, self.params.rsi_fast_period)
        macd_hist = self._macd_histogram(close)
        atr = self._atr(frame, self.params.atr_period)
        adx = self._adx(frame, self.params.adx_period)

        atr_value = float(atr.iloc[-1]) if not atr.empty and not pd.isna(atr.iloc[-1]) else 0.0
        adx_value = float(adx.iloc[-1]) if not adx.empty and not pd.isna(adx.iloc[-1]) else 0.0
        if atr_value <= 0:
            return self._build_hold("insufficient_volatility")

        volume_ma = volume.rolling(window=self.params.volume_ma_period, min_periods=self.params.volume_ma_period).mean()
        volume_ratio = float(volume.iloc[-1] / volume_ma.iloc[-1]) if not volume_ma.empty and not pd.isna(volume_ma.iloc[-1]) and float(volume_ma.iloc[-1]) > 0 else 0.0
        volume_delta = self._volume_delta(frame)
        ema_signal = self._ema_crossover_signal(close)
        structure_signal = self._price_structure(close)
        momentum_signal = self._rsi_momentum(close)

        buy_score = 0.0
        sell_score = 0.0
        components: Dict[str, Any] = {}

        rsi_now = float(rsi.iloc[-1])
        rsi_fast_now = float(rsi_fast.iloc[-1])
        rsi_fast_prev = float(rsi_fast.iloc[-2])
        if rsi_now < self.params.rsi_oversold and rsi_fast_now > rsi_fast_prev and momentum_signal > 0:
            buy_score += 2.0
            components["rsi_momentum"] = "buy"
        if rsi_now > self.params.rsi_overbought and rsi_fast_now < rsi_fast_prev and momentum_signal < 0:
            sell_score += 2.0
            components["rsi_momentum"] = "sell"

        macd_current = float(macd_hist.iloc[-1])
        macd_prev = float(macd_hist.iloc[-2])
        if macd_prev <= 0 and macd_current > 0:
            buy_score += 2.0
            components["macd_flip"] = "buy"
        if macd_prev >= 0 and macd_current < 0:
            sell_score += 2.0
            components["macd_flip"] = "sell"

        if ema_signal > 0:
            buy_score += 1.0
            components["ema_cross"] = "buy"
        elif ema_signal < 0:
            sell_score += 1.0
            components["ema_cross"] = "sell"

        if volume_ratio >= self.params.volume_spike_threshold and volume_delta > 0.2:
            buy_score += 2.0
            components["volume"] = "buy"
        elif volume_ratio >= self.params.volume_spike_threshold and volume_delta < -0.2:
            sell_score += 2.0
            components["volume"] = "sell"

        if structure_signal > 0:
            buy_score += 1.0
            components["structure"] = "buy"
        elif structure_signal < 0:
            sell_score += 1.0
            components["structure"] = "sell"

        if adx_value >= self.params.adx_min_trend:
            if buy_score > 0:
                buy_score += 1.0
            if sell_score > 0:
                sell_score += 1.0
            components["adx"] = "trend_confirmed"
        else:
            buy_score *= 0.5
            sell_score *= 0.5
            components["adx"] = "trend_weak"

        dominant_action = "BUY" if buy_score > sell_score else "SELL"
        dominant_score = buy_score if dominant_action == "BUY" else sell_score
        confidence = dominant_score / 9.0

        metadata = {
            "status": "WAIT",
            "buy_score": round(buy_score, 3),
            "sell_score": round(sell_score, 3),
            "components": components,
            "rsi": round(rsi_now, 3),
            "rsi_fast": round(rsi_fast_now, 3),
            "macd_hist": round(macd_current, 6),
            "volume_ratio": round(volume_ratio, 3),
            "volume_delta": round(volume_delta, 3),
            "price_structure": structure_signal,
            "adx": round(adx_value, 3),
        }

        if dominant_score >= 5.0 and confidence >= 0.55:
            return self._finalize_signal(
                action=dominant_action,
                current_price=price_now,
                confidence=confidence,
                atr_value=atr_value,
                metadata=metadata,
                now=now,
            )

        if 4.0 <= dominant_score < 5.0:
            metadata.update(
                {
                    "status": "WATCH",
                    "watch_direction": dominant_action.lower(),
                    "watch_confidence": round(confidence, 4),
                }
            )
            self._last_watch_signal = metadata.copy()
            self._last_signal_snapshot = metadata.copy()
            return SignalResult(
                action="HOLD",
                reason="watch_signal",
                confidence=round(confidence, 4),
                price=round(price_now, 6),
                metadata=metadata,
            )

        self._last_signal_snapshot = metadata.copy()
        return SignalResult(
            action="HOLD",
            reason="no_confluence",
            confidence=round(confidence, 4),
            price=round(price_now, 6),
            metadata=metadata,
        )

    def get_status(self) -> Dict[str, Any]:
        """Return current runtime state for the aggressive scalper."""
        now = datetime.utcnow()
        self._trim_trade_timestamps(now)
        halted = self.risk_engine.daily_loss_halt(
            self._daily_pnl,
            self._account_balance,
            self.params.daily_loss_halt_pct,
        )
        daily_loss_pct = 0.0
        if self._account_balance > 0 and self._daily_pnl < 0:
            daily_loss_pct = (-self._daily_pnl / self._account_balance) * 100.0

        return {
            "strategy_id": self.strategy_id,
            "version": self.version,
            "enabled": self.config.enabled,
            "halted": halted,
            "account_balance": round(self._account_balance, 2),
            "daily_pnl": round(self._daily_pnl, 4),
            "weekly_pnl": round(self._weekly_pnl, 4),
            "daily_loss_pct": round(daily_loss_pct, 4),
            "trades_last_hour": len(self._trade_timestamps),
            "cooldown_seconds": self.params.cooldown_seconds,
            "cooldown_remaining_seconds": self._cooldown_remaining_seconds(now),
            "last_signal_at": self._last_signal_at.isoformat() if self._last_signal_at else None,
            "last_signal": self._last_signal_snapshot,
            "last_watch_signal": self._last_watch_signal,
            "max_trades_per_hour": self.params.max_trades_per_hour,
            "stop_loss_atr_mult": self.params.stop_loss_atr_mult,
            "take_profit_1_mult": self.params.take_profit_1_mult,
            "take_profit_2_mult": self.params.take_profit_2_mult,
            "max_position_pct": self.params.max_position_pct,
            "leverage": float(self.config.metadata.get("leverage", 1.0)),
        }
