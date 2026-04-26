"""Unified Smart Money Concepts (SMC) confluence engine."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

from .bos_choch import MarketTrend, StructureAnalyzer
from .fvg import FVGType, FairValueGapDetector
from .liquidity import LiquidityHunter, LiquidityType
from .order_blocks import OBType, OrderBlockDetector


class SetupType(Enum):
    BULLISH_OB_RETEST = "bullish_ob_retest"
    BEARISH_OB_RETEST = "bearish_ob_retest"
    BULLISH_FVG_FILL = "bullish_fvg_fill"
    BEARISH_FVG_FILL = "bearish_fvg_fill"
    CHOCH_REVERSAL = "choch_reversal"
    BOS_CONTINUATION = "bos_continuation"
    LIQUIDITY_SWEEP = "liquidity_sweep"
    OPTIMAL_ENTRY = "optimal_entry"


@dataclass
class SMCSetup:
    setup_id: str
    setup_type: SetupType
    direction: str
    timeframe: str
    timestamp: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    risk_reward: float
    confidence: float
    components: Dict
    invalidation: float
    notes: str = ""
    metadata: Dict = field(default_factory=dict)

    @property
    def is_long(self) -> bool:
        return self.direction == "long"

    @property
    def is_short(self) -> bool:
        return self.direction == "short"


class SMCEngine:
    """Runs OB/FVG/structure/liquidity detectors and builds trade setups."""

    def __init__(
        self,
        timeframes: Optional[List[str]] = None,
        min_confluence: float = 0.6,
        atr_sl_multiplier: float = 1.0,
        tp_ratios: Optional[List[float]] = None,
    ) -> None:
        self.timeframes = timeframes or ["1h", "15m", "5m"]
        self.min_confluence = min_confluence
        self.atr_sl_multiplier = atr_sl_multiplier
        self.tp_ratios = tp_ratios or [1.5, 2.5, 4.0]
        self._counter = 0
        self.use_body_only_ob = self._load_use_body_only_flag()

        self.ob_detectors = {
            tf: OrderBlockDetector(timeframe=tf, use_body_only=self.use_body_only_ob)
            for tf in self.timeframes
        }
        self.fvg_detectors = {tf: FairValueGapDetector(timeframe=tf) for tf in self.timeframes}
        self.structure_analyzers = {tf: StructureAnalyzer(timeframe=tf) for tf in self.timeframes}
        self.liquidity_hunters = {tf: LiquidityHunter(timeframe=tf) for tf in self.timeframes}

    @staticmethod
    def _load_use_body_only_flag() -> bool:
        """Read the order-block body-only setting from scalper config."""
        config_path = Path(__file__).resolve().parents[2] / "configs" / "scalper_config.yaml"
        if not config_path.exists():
            return False
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception:
            return False
        smc_payload = payload.get("smc", {}) if isinstance(payload, dict) else {}
        return bool(smc_payload.get("use_body_only_ob", False))

    def analyze(self, data: Dict[str, pd.DataFrame]) -> List[SMCSetup]:
        setups: List[SMCSetup] = []

        for timeframe, df in data.items():
            if timeframe not in self.timeframes or df is None or len(df) < 30:
                continue

            self.ob_detectors[timeframe].detect(df)
            self.ob_detectors[timeframe].update_status(df)

            self.fvg_detectors[timeframe].detect(df)
            self.fvg_detectors[timeframe].update_fill_status(df)

            self.structure_analyzers[timeframe].analyze(df)

            self.liquidity_hunters[timeframe].detect_equal_highs_lows(df)
            self.liquidity_hunters[timeframe].detect_stop_hunts(df)
            self.liquidity_hunters[timeframe].update_sweep_status(float(df["close"].iloc[-1]))

            current_price = float(df["close"].iloc[-1])
            atr = self._compute_atr(df)

            setups.extend(
                self._generate_setups(
                    tf=timeframe,
                    current_price=current_price,
                    atr=atr,
                )
            )

        setups.sort(key=lambda item: item.confidence, reverse=True)
        return setups

    def get_active_setups(
        self,
        data: Dict[str, pd.DataFrame],
        timeframe: Optional[str] = None,
        limit: int = 5,
    ) -> List[SMCSetup]:
        """Return the highest-confidence active setups for the requested timeframe."""
        setups = self.analyze(data)
        if timeframe is not None:
            setups = [setup for setup in setups if setup.timeframe == timeframe]
        return setups[: max(limit, 0)]

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr = pd.concat(
            [
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_series = tr.rolling(period).mean()
        atr_value = atr_series.iloc[-1]
        if pd.isna(atr_value) or float(atr_value) <= 0:
            return 0.01
        return float(atr_value)

    def _generate_setups(self, tf: str, current_price: float, atr: float) -> List[SMCSetup]:
        setups: List[SMCSetup] = []

        ob_detector = self.ob_detectors[tf]
        fvg_detector = self.fvg_detectors[tf]
        structure = self.structure_analyzers[tf]
        liquidity = self.liquidity_hunters[tf]

        trend = structure.current_trend
        latest_bos = structure.latest_bos
        latest_choch = structure.latest_choch

        if trend in (MarketTrend.BULLISH, MarketTrend.RANGING):
            bullish_obs = ob_detector.get_active_obs(OBType.BULLISH)
            bullish_fvgs = fvg_detector.get_open_fvgs(FVGType.BULLISH)

            for ob in bullish_obs:
                if not (ob.bottom <= current_price <= ob.top):
                    continue

                confluence = self._score_bullish_confluence(
                    ob=ob,
                    fvgs=bullish_fvgs,
                    bos=latest_bos,
                    choch=latest_choch,
                    liquidity=liquidity,
                    price=current_price,
                )
                if confluence["score"] < self.min_confluence:
                    continue

                stop_loss = ob.bottom - atr * self.atr_sl_multiplier
                risk = current_price - stop_loss
                if risk <= 0:
                    continue

                tp1 = current_price + risk * self.tp_ratios[0]
                tp2 = current_price + risk * self.tp_ratios[1]
                tp3 = current_price + risk * self.tp_ratios[2]

                self._counter += 1
                setups.append(
                    SMCSetup(
                        setup_id=f"SETUP_{self._counter}",
                        setup_type=SetupType.OPTIMAL_ENTRY if confluence["score"] > 0.8 else SetupType.BULLISH_OB_RETEST,
                        direction="long",
                        timeframe=tf,
                        timestamp=pd.Timestamp.utcnow(),
                        entry_price=current_price,
                        stop_loss=round(stop_loss, 4),
                        take_profit_1=round(tp1, 4),
                        take_profit_2=round(tp2, 4),
                        take_profit_3=round(tp3, 4),
                        risk_reward=round((tp1 - current_price) / risk, 2),
                        confidence=round(confluence["score"], 3),
                        components=confluence["components"],
                        invalidation=round(stop_loss, 4),
                        notes=f"Bullish OB retest on {tf} | Trend: {trend.value}",
                    )
                )

        if trend in (MarketTrend.BEARISH, MarketTrend.RANGING):
            bearish_obs = ob_detector.get_active_obs(OBType.BEARISH)
            bearish_fvgs = fvg_detector.get_open_fvgs(FVGType.BEARISH)

            for ob in bearish_obs:
                if not (ob.bottom <= current_price <= ob.top):
                    continue

                confluence = self._score_bearish_confluence(
                    ob=ob,
                    fvgs=bearish_fvgs,
                    bos=latest_bos,
                    choch=latest_choch,
                    liquidity=liquidity,
                    price=current_price,
                )
                if confluence["score"] < self.min_confluence:
                    continue

                stop_loss = ob.top + atr * self.atr_sl_multiplier
                risk = stop_loss - current_price
                if risk <= 0:
                    continue

                tp1 = current_price - risk * self.tp_ratios[0]
                tp2 = current_price - risk * self.tp_ratios[1]
                tp3 = current_price - risk * self.tp_ratios[2]

                self._counter += 1
                setups.append(
                    SMCSetup(
                        setup_id=f"SETUP_{self._counter}",
                        setup_type=SetupType.OPTIMAL_ENTRY if confluence["score"] > 0.8 else SetupType.BEARISH_OB_RETEST,
                        direction="short",
                        timeframe=tf,
                        timestamp=pd.Timestamp.utcnow(),
                        entry_price=current_price,
                        stop_loss=round(stop_loss, 4),
                        take_profit_1=round(tp1, 4),
                        take_profit_2=round(tp2, 4),
                        take_profit_3=round(tp3, 4),
                        risk_reward=round((current_price - tp1) / risk, 2),
                        confidence=round(confluence["score"], 3),
                        components=confluence["components"],
                        invalidation=round(stop_loss, 4),
                        notes=f"Bearish OB retest on {tf} | Trend: {trend.value}",
                    )
                )

        return setups

    @staticmethod
    def _score_bullish_confluence(ob, fvgs, bos, choch, liquidity, price: float) -> Dict:
        score = 0.0
        components: Dict = {}

        score += 0.3
        components["order_block"] = {"id": ob.ob_id, "strength": ob.strength, "score": 0.3}
        score += ob.strength * 0.1

        for fvg in fvgs:
            if fvg.is_open and fvg.bottom <= price <= fvg.top:
                score += 0.2
                components["fvg"] = {"id": fvg.fvg_id, "strength": fvg.strength, "score": 0.2}
                break

        if bos and bos.direction == "bullish":
            score += 0.2
            components["bos"] = {"id": bos.break_id, "score": 0.2}

        if choch and choch.direction == "bullish":
            score += 0.25
            components["choch"] = {"id": choch.break_id, "score": 0.25}

        nearby_sell_liquidity = liquidity.get_nearest_liquidity(price, LiquidityType.SELLSIDE)
        if nearby_sell_liquidity and not nearby_sell_liquidity.is_active:
            score += 0.15
            components["liquidity_swept"] = {
                "level": nearby_sell_liquidity.price,
                "score": 0.15,
            }

        return {"score": min(score, 1.0), "components": components}

    @staticmethod
    def _score_bearish_confluence(ob, fvgs, bos, choch, liquidity, price: float) -> Dict:
        score = 0.0
        components: Dict = {}

        score += 0.3
        components["order_block"] = {"id": ob.ob_id, "strength": ob.strength, "score": 0.3}
        score += ob.strength * 0.1

        for fvg in fvgs:
            if fvg.is_open and fvg.bottom <= price <= fvg.top:
                score += 0.2
                components["fvg"] = {"id": fvg.fvg_id, "strength": fvg.strength, "score": 0.2}
                break

        if bos and bos.direction == "bearish":
            score += 0.2
            components["bos"] = {"id": bos.break_id, "score": 0.2}

        if choch and choch.direction == "bearish":
            score += 0.25
            components["choch"] = {"id": choch.break_id, "score": 0.25}

        nearby_buy_liquidity = liquidity.get_nearest_liquidity(price, LiquidityType.BUYSIDE)
        if nearby_buy_liquidity and not nearby_buy_liquidity.is_active:
            score += 0.15
            components["liquidity_swept"] = {
                "level": nearby_buy_liquidity.price,
                "score": 0.15,
            }

        return {"score": min(score, 1.0), "components": components}

    def get_state_dict(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict:
        if data is not None:
            # Refresh state before exposing if caller supplied data.
            self.analyze(data)

        state: Dict[str, Dict] = {}
        for timeframe in self.timeframes:
            state[timeframe] = {
                "order_blocks": self.ob_detectors[timeframe].to_dict_list(),
                "fvgs": self.fvg_detectors[timeframe].to_dict_list(),
                "structure": self.structure_analyzers[timeframe].to_dict_list(),
                "liquidity": self.liquidity_hunters[timeframe].to_dict_list(),
                "trend": self.structure_analyzers[timeframe].current_trend.value,
            }
        return state
