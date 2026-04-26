"""Intraday SMC scalper strategy for CryptoBoss v12.0."""

from dataclasses import dataclass
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .base_strategy import BaseStrategy, SignalResult, StrategyConfig
from .session_manager_pro import SessionManagerPro, SessionWindow
from ..risk.scalper_risk_engine import ScalperRiskConfig, ScalperRiskEngine
from ..smc.smc_engine import SMCEngine, SMCSetup


@dataclass
class ScalperState:
    last_setup_id: Optional[str] = None
    active_setups: List[SMCSetup] = None

    def __post_init__(self) -> None:
        if self.active_setups is None:
            self.active_setups = []


class IntradayScalper(BaseStrategy):
    """SMC confluence-based intraday scalper with session and risk filters."""

    STRATEGY_ID = "intraday_scalper_v1"
    VERSION = "12.0"

    def __init__(self, config: StrategyConfig):
        super().__init__(config)

        bias_tf = config.metadata.get("bias_timeframe", "1h")
        mid_tf = config.metadata.get("mid_timeframe", "15m")
        entry_tf = config.metadata.get("entry_timeframe", "5m")

        self.entry_timeframe = entry_tf
        self.mid_timeframe = mid_tf
        self.bias_timeframe = bias_tf

        self.smc_engine = SMCEngine(
            timeframes=[bias_tf, mid_tf, entry_tf],
            min_confluence=float(config.metadata.get("min_confluence", 0.6)),
            atr_sl_multiplier=float(config.metadata.get("atr_sl_mult", 0.8)),
            tp_ratios=list(config.metadata.get("tp_ratios", [1.5, 2.5, 4.0])),
        )

        self.min_rr = float(config.metadata.get("min_rr", 1.5))
        self.session_filter = bool(config.metadata.get("session_filter", True))
        self.kill_zone_only = bool(config.metadata.get("kill_zone_only", False))

        self.session_manager = SessionManagerPro()
        self.risk_engine = ScalperRiskEngine(
            config=ScalperRiskConfig(
                max_risk_pct=float(config.metadata.get("max_risk_pct", 0.005)),
                max_positions=int(config.metadata.get("max_positions", 3)),
                partial_exit_pct=float(config.metadata.get("partial_exit_pct", 0.5)),
            )
        )

        self.state = ScalperState()

    def _compose_data(self, df: pd.DataFrame, i: int) -> Dict[str, pd.DataFrame]:
        # BaseStrategy calls are single-timeframe; map that slice to all configured TF keys.
        end = i + 1 if i is not None and i >= 0 else len(df)
        sliced = df.iloc[:end].copy()
        return {
            self.bias_timeframe: sliced,
            self.mid_timeframe: sliced,
            self.entry_timeframe: sliced,
        }

    def _current_session(self, now_utc: Optional[time] = None) -> Optional[SessionWindow]:
        return self.session_manager.current_session(now_utc=now_utc)

    def _in_kill_zone(self, now_utc: Optional[time] = None) -> Tuple[bool, str]:
        return self.session_manager.in_kill_zone(now_utc=now_utc)

    def generate_signal(
        self,
        df: pd.DataFrame,
        i: int,
        current_price: float,
        multi_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> SignalResult:
        """BaseStrategy-compatible entry point for engine pipeline usage."""
        data = multi_tf_data or self._compose_data(df, i)
        return self.generate_multi_timeframe_signal(data=data, account_balance=10000.0)

    def generate_multi_timeframe_signal(
        self,
        data: Dict[str, pd.DataFrame],
        account_balance: float = 10000.0,
    ) -> SignalResult:
        """Primary v12 API: generates SMC scalper signal from multi-timeframe data."""
        session = self._current_session(datetime.utcnow().time())
        if self.session_filter and session is None:
            return SignalResult(
                action="HOLD",
                reason="Outside active session - no trading",
                confidence=0.0,
            )

        in_kz, kz_name = self._in_kill_zone(datetime.utcnow().time())
        if self.kill_zone_only and not in_kz:
            return SignalResult(
                action="HOLD",
                reason="Kill zone filter enabled and no active kill zone",
                confidence=0.0,
            )

        setups = self.smc_engine.analyze(data)
        self.state.active_setups = setups

        if not setups:
            return SignalResult(
                action="HOLD",
                reason="No SMC setup detected",
                confidence=0.0,
            )

        best = setups[0]
        self.state.last_setup_id = best.setup_id

        if best.confidence < self.config.min_confidence:
            return SignalResult(
                action="HOLD",
                reason=(
                    f"Setup confidence {best.confidence:.2f} below minimum "
                    f"{self.config.min_confidence:.2f}"
                ),
                confidence=best.confidence,
            )

        if best.risk_reward < self.min_rr:
            return SignalResult(
                action="HOLD",
                reason=f"RR {best.risk_reward:.2f} below minimum {self.min_rr:.2f}",
                confidence=best.confidence,
            )

        if not self.risk_engine.can_open_position(current_open_positions=0):
            return SignalResult(
                action="HOLD",
                reason="Risk engine: maximum concurrent positions reached",
                confidence=best.confidence,
            )

        position_size = self.risk_engine.compute_position_size(
            account_balance=account_balance,
            entry_price=best.entry_price,
            stop_loss=best.stop_loss,
            session_weight=session.weight if session else 1.0,
        )

        action = "BUY" if best.is_long else "SELL"
        components_str = ", ".join(best.components.keys()) if best.components else "none"
        session_name = session.name if session else "Unknown"
        killzone_suffix = f" | Kill Zone: {kz_name}" if in_kz else ""

        return SignalResult(
            action=action,
            reason=(
                f"SMC {best.setup_type.value} | TF: {best.timeframe} | "
                f"Session: {session_name}{killzone_suffix} | "
                f"Confluence: {components_str} | RR: {best.risk_reward:.2f} | "
                f"Confidence: {best.confidence:.2f}"
            ),
            confidence=best.confidence,
            size=position_size,
            price=best.entry_price,
            stop_loss=best.stop_loss,
            take_profit=best.take_profit_1,
            signal_strength=best.confidence,
            metadata={
                "setup_id": best.setup_id,
                "setup_type": best.setup_type.value,
                "direction": best.direction,
                "entry": best.entry_price,
                "sl": best.stop_loss,
                "tp1": best.take_profit_1,
                "tp2": best.take_profit_2,
                "tp3": best.take_profit_3,
                "rr": best.risk_reward,
                "invalidation": best.invalidation,
                "components": best.components,
                "session": session_name,
                "kill_zone": kz_name if in_kz else None,
                "smc_state": self.smc_engine.get_state_dict(),
            },
        )

    def analyze_current_market(self, data: Dict[str, pd.DataFrame]) -> Dict:
        setups = self.smc_engine.analyze(data)
        state = self.smc_engine.get_state_dict()
        session = self._current_session(datetime.utcnow().time())
        in_kz, kz_name = self._in_kill_zone(datetime.utcnow().time())

        return {
            "session": session.name if session else None,
            "in_kill_zone": in_kz,
            "kill_zone_name": kz_name,
            "setups_count": len(setups),
            "top_setups": [
                {
                    "id": setup.setup_id,
                    "type": setup.setup_type.value,
                    "direction": setup.direction,
                    "timeframe": setup.timeframe,
                    "confidence": setup.confidence,
                    "rr": setup.risk_reward,
                    "entry": setup.entry_price,
                    "sl": setup.stop_loss,
                    "tp1": setup.take_profit_1,
                    "components": list(setup.components.keys()),
                }
                for setup in setups[:5]
            ],
            "smc_state": state,
        }
