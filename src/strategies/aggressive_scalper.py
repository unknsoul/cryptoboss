"""Aggressive scalper strategy v3.0 — MTF + SMC + ML + session-aware scoring."""

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

SESSIONS = {
    "london_open":  {"start": 7,  "end": 10,  "weight": 1.3},
    "ny_open":      {"start": 13, "end": 16,  "weight": 1.4},
    "london_ny":    {"start": 10, "end": 13,  "weight": 1.1},
    "asia":         {"start": 0,  "end": 7,   "weight": 0.6},
    "ny_close":     {"start": 20, "end": 23,  "weight": 0.7},
    "evening":      {"start": 16, "end": 20,  "weight": 0.9},
}

MAX_POSSIBLE_SCORE = 10.0
ENTRY_THRESHOLD = 4.8
CONFIDENCE_THRESHOLD = 0.48


@dataclass
class ScalperParams:
    """Runtime parameters for the aggressive scalper."""

    rsi_period: int = 7
    rsi_fast_period: int = 4
    rsi_oversold: float = 32.0
    rsi_overbought: float = 68.0
    macd_fast: int = 6
    macd_slow: int = 13
    macd_signal: int = 4
    ema_fast: int = 8
    ema_slow: int = 21
    volume_ma_period: int = 14
    volume_spike_threshold: float = 1.3
    adx_period: int = 10
    adx_min_trend: float = 15.0
    atr_period: int = 10
    stop_loss_atr_mult: float = 0.55
    take_profit_1_mult: float = 1.8
    take_profit_2_mult: float = 3.0
    take_profit_3_mult: float = 5.0
    max_trades_per_hour: int = 12
    max_concurrent_positions: int = 3
    daily_loss_halt_pct: float = 3.0
    weekly_loss_halt_pct: float = 6.0
    cooldown_seconds: int = 30
    min_confluence: int = 3
    max_position_pct: float = 6.0
    max_risk_pct_per_trade: float = 0.0075
    auto_partial_close: bool = True
    partial_close_pct_at_tp1: float = 0.4
    move_sl_to_be_at_tp1: bool = True
    trailing_stop_after_tp1: bool = True
    trailing_atr_mult: float = 1.0


class AggressiveScalper(BaseStrategy):
    """Fast scalper: MTF-confirmed, SMC-refined, ML-gated, session-aware."""

    STRATEGY_ID = "aggressive_scalper_v1"
    VERSION = "3.0"
    MIN_ROWS = 30

    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        params: Optional[ScalperParams] = None,
    ) -> None:
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
        self._consecutive_losses = 0
        self._cooldown_until: Optional[datetime] = None

        self._smc_engine: Optional[Any] = None
        self._ml_filter: Optional[Any] = None
        self._init_smc_engine()
        self._init_ml_filter()

    def _init_smc_engine(self) -> None:
        try:
            from ..smc.smc_engine import SMCEngine
            self._smc_engine = SMCEngine(
                timeframes=["1m", "5m", "15m"],
                min_confluence=0.4,
            )
        except Exception as exc:
            logger.warning("SMC engine unavailable: %s", exc)

    def _init_ml_filter(self) -> None:
        try:
            from ..ml.signal_filter import SignalQualityFilter
            self._ml_filter = SignalQualityFilter(min_quality_score=55.0)
        except Exception as exc:
            logger.warning("ML signal filter unavailable: %s", exc)

    @staticmethod
    def _load_file_config() -> Dict[str, Any]:
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
            min_confidence=CONFIDENCE_THRESHOLD,
            cooldown_seconds=int(guards_payload.get("cooldown_seconds", params.cooldown_seconds)),
            metadata={
                "symbols": symbols,
                "timeframe": payload.get("timeframe", "1m"),
                "secondary_timeframe": payload.get("secondary_timeframe", "5m"),
                "tertiary_timeframe": payload.get("tertiary_timeframe", "15m"),
                "leverage": float(risk_payload.get("leverage", 1.0)),
            },
        )

    @staticmethod
    def _build_params(payload: Dict[str, Any]) -> ScalperParams:
        rsi_p = payload.get("rsi", {}) if isinstance(payload, dict) else {}
        macd_p = payload.get("macd", {}) if isinstance(payload, dict) else {}
        ema_p = payload.get("ema", {}) if isinstance(payload, dict) else {}
        atr_p = payload.get("atr", {}) if isinstance(payload, dict) else {}
        vol_p = payload.get("volume", {}) if isinstance(payload, dict) else {}
        adx_p = payload.get("adx", {}) if isinstance(payload, dict) else {}
        risk_p = payload.get("risk", {}) if isinstance(payload, dict) else {}
        guards_p = payload.get("guards", {}) if isinstance(payload, dict) else {}
        scale_p = payload.get("scaling", {}) if isinstance(payload, dict) else {}

        return ScalperParams(
            rsi_period=int(rsi_p.get("period", 7)),
            rsi_fast_period=int(rsi_p.get("fast_period", 4)),
            rsi_oversold=float(rsi_p.get("oversold", 32.0)),
            rsi_overbought=float(rsi_p.get("overbought", 68.0)),
            macd_fast=int(macd_p.get("fast", 6)),
            macd_slow=int(macd_p.get("slow", 13)),
            macd_signal=int(macd_p.get("signal", 4)),
            ema_fast=int(ema_p.get("fast", 8)),
            ema_slow=int(ema_p.get("slow", 21)),
            volume_ma_period=int(vol_p.get("ma_period", 14)),
            volume_spike_threshold=float(vol_p.get("spike_threshold", 1.3)),
            adx_period=int(adx_p.get("period", 10)),
            adx_min_trend=float(adx_p.get("min_trend", 15.0)),
            atr_period=int(atr_p.get("period", 10)),
            stop_loss_atr_mult=float(risk_p.get("stop_loss_atr_mult", 0.55)),
            take_profit_1_mult=float(risk_p.get("take_profit_1_mult", 1.8)),
            take_profit_2_mult=float(risk_p.get("take_profit_2_mult", 3.0)),
            take_profit_3_mult=float(risk_p.get("take_profit_3_mult", 5.0)),
            max_trades_per_hour=int(guards_p.get("max_trades_per_hour", 12)),
            max_concurrent_positions=int(guards_p.get("max_concurrent_positions", 3)),
            daily_loss_halt_pct=float(guards_p.get("daily_loss_halt_pct", 3.0)),
            weekly_loss_halt_pct=float(guards_p.get("weekly_loss_halt_pct", 6.0)),
            cooldown_seconds=int(guards_p.get("cooldown_seconds", 30)),
            min_confluence=int(guards_p.get("min_confluence", 3)),
            max_position_pct=float(risk_p.get("max_position_pct", 6.0)),
            max_risk_pct_per_trade=float(risk_p.get("max_risk_pct_per_trade", 0.0075)),
            auto_partial_close=bool(scale_p.get("auto_partial_close", True)),
            partial_close_pct_at_tp1=float(scale_p.get("partial_close_pct_at_tp1", 0.4)),
            move_sl_to_be_at_tp1=bool(scale_p.get("move_sl_to_be_at_tp1", True)),
            trailing_stop_after_tp1=bool(scale_p.get("trailing_stop_after_tp1", True)),
            trailing_atr_mult=float(scale_p.get("trailing_atr_mult", 1.0)),
        )

    def set_account_balance(self, balance: float) -> None:
        if balance > 0:
            self._account_balance = float(balance)

    def set_daily_pnl(self, pnl: float) -> None:
        self._daily_pnl = float(pnl)

    def set_weekly_pnl(self, pnl: float) -> None:
        self._weekly_pnl = float(pnl)

    def record_trade_result(self, pnl: float) -> None:
        """Track consecutive losses for circuit breaker."""
        if pnl < 0:
            self._consecutive_losses += 1
            if self._consecutive_losses >= 4:
                self._cooldown_until = datetime.utcnow().__class__(
                    *datetime.utcnow().timetuple()[:5],
                    second=0,
                )
                from datetime import timedelta
                self._cooldown_until = datetime.utcnow() + timedelta(minutes=15)
                logger.warning("4 consecutive losses — 15-min cooldown activated")
        else:
            self._consecutive_losses = 0

    def _slice_frame(self, df: pd.DataFrame, i: Optional[int]) -> pd.DataFrame:
        if i is None or i < 0:
            return df.copy()
        return df.iloc[: i + 1].copy()

    def _trim_trade_timestamps(self, now: datetime) -> None:
        threshold = now.timestamp() - 3600.0
        self._trade_timestamps = [
            ts for ts in self._trade_timestamps if ts.timestamp() >= threshold
        ]

    def _cooldown_remaining_seconds(self, now: datetime) -> int:
        if self._last_signal_at is None:
            return 0
        elapsed = (now - self._last_signal_at).total_seconds()
        return max(int(self.params.cooldown_seconds - elapsed), 0)

    @staticmethod
    def _session_weight(now: datetime) -> Tuple[float, str]:
        hour = now.hour
        for name, s in SESSIONS.items():
            if s["start"] <= hour < s["end"]:
                return s["weight"], name
        return 1.0, "default"

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> pd.Series:
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
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 10) -> pd.Series:
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        tr = pd.concat(
            [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1)
        return tr.rolling(window=period, min_periods=period).mean()

    def _macd_histogram(self, close: pd.Series) -> pd.Series:
        ema_fast = self._ema(close, self.params.macd_fast)
        ema_slow = self._ema(close, self.params.macd_slow)
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=self.params.macd_signal, adjust=False).mean()
        return macd - signal

    @staticmethod
    def _adx(df: pd.DataFrame, period: int) -> pd.Series:
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        tr = pd.concat(
            [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(window=period, min_periods=period).mean()
        plus_di = 100.0 * plus_dm.rolling(window=period, min_periods=period).mean() / atr.replace(0.0, pd.NA)
        minus_di = 100.0 * minus_dm.rolling(window=period, min_periods=period).mean() / atr.replace(0.0, pd.NA)
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, pd.NA)) * 100.0
        return dx.rolling(window=period, min_periods=period).mean().fillna(0.0)

    def _ema_crossover_signal(self, close: pd.Series) -> int:
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

    @staticmethod
    def _volume_delta(df: pd.DataFrame) -> float:
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
        if len(close) < lookback + 1:
            return 0
        window = close.tail(lookback).astype(float)
        diff = window.diff().dropna()
        if diff.empty:
            return 0
        if diff.gt(0).sum() >= len(diff) * 0.6:
            return 1
        if diff.lt(0).sum() >= len(diff) * 0.6:
            return -1
        return 0

    def _confirm_higher_timeframe(
        self,
        df_5m: Optional[pd.DataFrame],
        df_15m: Optional[pd.DataFrame],
        direction: str,
    ) -> Tuple[bool, float]:
        if df_5m is None or len(df_5m) < 25:
            return True, 0.5
        if df_15m is None or len(df_15m) < 25:
            return True, 0.5

        close_15m = df_15m["close"].astype(float)
        ema21_15m = self._ema(close_15m, 21).iloc[-1]
        price_15m = float(close_15m.iloc[-1])

        close_5m = df_5m["close"].astype(float)
        ema8_5m = self._ema(close_5m, 8)

        if len(ema8_5m) < 4:
            return True, 0.5

        slope = (float(ema8_5m.iloc[-1]) - float(ema8_5m.iloc[-3])) / max(float(ema8_5m.iloc[-3]), 0.0001)

        if direction == "BUY":
            if price_15m < ema21_15m * 0.997:
                return False, 0.0
            if slope > 0.0001:
                return True, 1.0
            return True, 0.4
        else:
            if price_15m > ema21_15m * 1.003:
                return False, 0.0
            if slope < -0.0001:
                return True, 1.0
            return True, 0.4

    def _get_smc_precision(
        self,
        multi_tf_data: Dict[str, pd.DataFrame],
        direction: str,
        current_price: float,
    ) -> Tuple[float, Optional[float], Optional[float]]:
        if self._smc_engine is None:
            return 0.0, None, None
        try:
            setups = self._smc_engine.get_active_setups(multi_tf_data, limit=5)
        except Exception:
            return 0.0, None, None

        dir_key = "long" if direction == "BUY" else "short"
        aligned = [s for s in setups if s.direction == dir_key]
        if not aligned:
            return 0.0, None, None

        best = aligned[0]
        proximity_pct = abs(current_price - best.entry_price) / max(current_price, 0.0001)

        if proximity_pct < 0.002:
            return 1.5, best.entry_price, best.stop_loss
        elif proximity_pct < 0.005:
            return 0.75, None, None
        return 0.25, None, None

    def _ml_quality_gate(
        self,
        direction: str,
        confidence: float,
        volume_ratio: float,
        atr_value: float,
        adx_value: float,
        htf_score: float,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        if self._ml_filter is None:
            return True, 1.0, {}

        avg_atr = atr_value * 1.1
        signal_data = {
            "ml_confidence": confidence,
            "direction": "LONG" if direction == "BUY" else "SHORT",
            "volume": volume_ratio,
            "avg_volume": 1.0,
            "volatility": atr_value,
            "avg_volatility": avg_atr,
            "orderbook_imbalance": 0.0,
            "sentiment_score": 0.0,
            "timeframe_alignment": max(int(htf_score * 4), 1),
            "trend_strength": min(adx_value / 50.0, 1.0),
        }
        try:
            quality = self._ml_filter.calculate_quality_score(signal_data)
        except Exception:
            return True, 1.0, {}

        passed = quality.get("quality_score", 0) >= self._ml_filter.min_quality_score
        multiplier = quality.get("position_multiplier", 1.0)
        return passed, multiplier, quality

    def _build_hold(self, reason: str, metadata: Optional[Dict[str, Any]] = None) -> SignalResult:
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
        session_weight: float = 1.0,
        ml_size_mult: float = 1.0,
        smc_sl: Optional[float] = None,
    ) -> SignalResult:
        is_buy = action == "BUY"
        stop_distance = atr_value * self.params.stop_loss_atr_mult
        stop_loss = current_price - stop_distance if is_buy else current_price + stop_distance

        if smc_sl is not None:
            if is_buy and smc_sl < current_price:
                stop_loss = smc_sl
            elif not is_buy and smc_sl > current_price:
                stop_loss = smc_sl

        tp1 = current_price + (atr_value * self.params.take_profit_1_mult) if is_buy else current_price - (atr_value * self.params.take_profit_1_mult)
        tp2 = current_price + (atr_value * self.params.take_profit_2_mult) if is_buy else current_price - (atr_value * self.params.take_profit_2_mult)
        tp3 = current_price + (atr_value * self.params.take_profit_3_mult) if is_buy else current_price - (atr_value * self.params.take_profit_3_mult)

        size = self.risk_engine.compute_position_size(
            account_balance=self._account_balance,
            entry_price=current_price,
            stop_loss=stop_loss,
            session_weight=session_weight,
        )
        size = size * ml_size_mult
        max_size = (self._account_balance * (self.params.max_position_pct / 100.0)) / max(current_price, 0.0001)
        size = round(min(size, max_size), 6)

        metadata.update({
            "status": "CONFIRMED",
            "tp1": round(tp1, 6),
            "tp2": round(tp2, 6),
            "tp3": round(tp3, 6),
            "atr": round(atr_value, 6),
            "session_weight": round(session_weight, 2),
        })
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
        df_5m: Optional[pd.DataFrame] = None,
        df_15m: Optional[pd.DataFrame] = None,
    ) -> SignalResult:
        """Generate a scalping signal using 10-point weighted confluence scoring."""
        if df is None or df.empty:
            return self._build_hold("insufficient_data")

        frame = self._slice_frame(df, i)
        if len(frame) < self.MIN_ROWS:
            return self._build_hold("insufficient_data")

        now = datetime.utcnow()
        self._trim_trade_timestamps(now)

        if not self.config.enabled:
            return self._build_hold("strategy_disabled")

        if self._cooldown_until and now < self._cooldown_until:
            return self._build_hold("consecutive_loss_cooldown")

        if self.risk_engine.daily_loss_halt(
            self._daily_pnl, self._account_balance, self.params.daily_loss_halt_pct,
        ):
            return self._build_hold("daily_loss_halt")

        if len(self._trade_timestamps) >= self.params.max_trades_per_hour:
            return self._build_hold("hourly_trade_cap_reached")

        cooldown_remaining = self._cooldown_remaining_seconds(now)
        if cooldown_remaining > 0:
            return self._build_hold("cooldown_active", {"cooldown_remaining_seconds": cooldown_remaining})

        session_weight, session_name = self._session_weight(now)

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

        # --- 10-POINT SCORING (decoupled components) ---
        buy_score = 0.0
        sell_score = 0.0
        components: Dict[str, Any] = {}

        rsi_now = float(rsi.iloc[-1])
        rsi_fast_now = float(rsi_fast.iloc[-1])
        rsi_fast_prev = float(rsi_fast.iloc[-2]) if len(rsi_fast) >= 2 else rsi_fast_now

        # Component 1: RSI Level (0-1 pts) — BUG-03 FIX: decoupled from momentum
        if rsi_now < self.params.rsi_oversold:
            buy_score += 1.0
            components["rsi_level"] = "oversold"
        elif rsi_now < 42:
            buy_score += 0.5
            components["rsi_level"] = "low"
        if rsi_now > self.params.rsi_overbought:
            sell_score += 1.0
            components["rsi_level"] = "overbought"
        elif rsi_now > 58:
            sell_score += 0.5
            components["rsi_level"] = "high"

        # Component 2: RSI Momentum (0-1 pts) — independent of level
        if rsi_fast_now > rsi_fast_prev and rsi_now < 55:
            buy_score += 1.0
            components["rsi_momentum"] = "rising"
        if rsi_fast_now < rsi_fast_prev and rsi_now > 45:
            sell_score += 1.0
            components["rsi_momentum"] = "falling"

        # Component 3: MACD Flip (0-2 pts)
        macd_current = float(macd_hist.iloc[-1])
        macd_prev = float(macd_hist.iloc[-2]) if len(macd_hist) >= 2 else 0.0
        if macd_prev <= 0 and macd_current > 0:
            buy_score += 2.0
            components["macd_flip"] = "bullish"
        elif macd_current > 0 and macd_current > macd_prev:
            buy_score += 1.0
            components["macd_flip"] = "bullish_momentum"
        if macd_prev >= 0 and macd_current < 0:
            sell_score += 2.0
            components["macd_flip"] = "bearish"
        elif macd_current < 0 and macd_current < macd_prev:
            sell_score += 1.0
            components["macd_flip"] = "bearish_momentum"

        # Component 4: EMA Cross (0-1 pts)
        if ema_signal > 0:
            buy_score += 1.0
            components["ema_cross"] = "bullish"
        elif ema_signal < 0:
            sell_score += 1.0
            components["ema_cross"] = "bearish"

        # Component 5: Volume Spike (0-1.5 pts)
        if volume_ratio >= self.params.volume_spike_threshold:
            if volume_delta > 0.15:
                buy_score += 1.5
                components["volume"] = "buy_spike"
            elif volume_delta < -0.15:
                sell_score += 1.5
                components["volume"] = "sell_spike"
            else:
                buy_score += 0.5
                sell_score += 0.5
                components["volume"] = "neutral_spike"

        # Component 6: Price Structure (0-1 pts)
        if structure_signal > 0:
            buy_score += 1.0
            components["structure"] = "bullish"
        elif structure_signal < 0:
            sell_score += 1.0
            components["structure"] = "bearish"

        # Component 7: ADX Trend (multiplier, not gate)
        if adx_value >= self.params.adx_min_trend:
            components["adx"] = "trending"
        else:
            buy_score *= 0.7
            sell_score *= 0.7
            components["adx"] = "weak"

        # Determine dominant direction before HTF/SMC
        dominant_action = "BUY" if buy_score >= sell_score else "SELL"
        base_score = buy_score if dominant_action == "BUY" else sell_score

        # Component 8: HTF Alignment (0-2 pts) — BUG-02 FIX
        htf_confirmed, htf_weight = self._confirm_higher_timeframe(df_5m, df_15m, dominant_action)
        if not htf_confirmed:
            return self._build_hold("htf_rejection", {
                "direction": dominant_action, "base_score": round(base_score, 2),
                "session": session_name,
            })
        htf_bonus = htf_weight * 1.5
        base_score += htf_bonus
        components["htf_alignment"] = round(htf_weight, 2)

        # Normalize — BUG-04 FIX
        dominant_score = base_score
        confidence = min(dominant_score / MAX_POSSIBLE_SCORE, 1.0)

        metadata = {
            "status": "WAIT",
            "buy_score": round(buy_score, 3),
            "sell_score": round(sell_score, 3),
            "dominant_score": round(dominant_score, 3),
            "components": components,
            "rsi": round(rsi_now, 3),
            "rsi_fast": round(rsi_fast_now, 3),
            "macd_hist": round(macd_current, 6),
            "volume_ratio": round(volume_ratio, 3),
            "volume_delta": round(volume_delta, 3),
            "price_structure": structure_signal,
            "adx": round(adx_value, 3),
            "session": session_name,
            "session_weight": round(session_weight, 2),
            "htf_weight": round(htf_weight, 2),
        }

        # Entry gate with session-adjusted threshold
        effective_threshold = ENTRY_THRESHOLD
        if session_weight < 0.7:
            effective_threshold = ENTRY_THRESHOLD + 1.5

        if dominant_score >= effective_threshold and confidence >= CONFIDENCE_THRESHOLD:
            # SMC precision layer
            multi_tf_data: Dict[str, pd.DataFrame] = {"1m": frame}
            if df_5m is not None:
                multi_tf_data["5m"] = df_5m
            if df_15m is not None:
                multi_tf_data["15m"] = df_15m

            smc_bonus, smc_entry, smc_sl = self._get_smc_precision(
                multi_tf_data, dominant_action, price_now,
            )
            dominant_score += smc_bonus
            confidence = min(dominant_score / MAX_POSSIBLE_SCORE, 1.0)
            if smc_bonus > 0:
                components["smc"] = round(smc_bonus, 2)
                metadata["smc_bonus"] = round(smc_bonus, 2)

            # ML quality gate (soft — reduces size, doesn't block)
            ml_passed, ml_mult, ml_info = self._ml_quality_gate(
                dominant_action, confidence, volume_ratio, atr_value, adx_value, htf_weight,
            )
            if not ml_passed:
                ml_mult = 0.5
            metadata["ml_quality"] = ml_info.get("quality_score", 0)
            metadata["ml_grade"] = ml_info.get("grade", "N/A")

            return self._finalize_signal(
                action=dominant_action,
                current_price=price_now,
                confidence=confidence,
                atr_value=atr_value,
                metadata=metadata,
                now=now,
                session_weight=session_weight,
                ml_size_mult=ml_mult,
                smc_sl=smc_sl,
            )

        if dominant_score >= (effective_threshold - 1.5) and adx_value >= self.params.adx_min_trend:
            metadata.update({
                "status": "WATCH",
                "watch_direction": dominant_action.lower(),
                "watch_confidence": round(confidence, 4),
            })
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
        now = datetime.utcnow()
        self._trim_trade_timestamps(now)
        halted = self.risk_engine.daily_loss_halt(
            self._daily_pnl, self._account_balance, self.params.daily_loss_halt_pct,
        )
        daily_loss_pct = 0.0
        if self._account_balance > 0 and self._daily_pnl < 0:
            daily_loss_pct = (-self._daily_pnl / self._account_balance) * 100.0

        session_weight, session_name = self._session_weight(now)

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
            "take_profit_3_mult": self.params.take_profit_3_mult,
            "max_position_pct": self.params.max_position_pct,
            "leverage": float(self.config.metadata.get("leverage", 1.0)),
            "consecutive_losses": self._consecutive_losses,
            "session": session_name,
            "session_weight": round(session_weight, 2),
            "smc_enabled": self._smc_engine is not None,
            "ml_filter_enabled": self._ml_filter is not None,
            "scoring_version": "10-point_v3",
            "entry_threshold": ENTRY_THRESHOLD,
        }
