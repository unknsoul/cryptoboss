"""Smart money concepts engine for v3 architecture."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from src.smc.liquidity import LiquidityHunter

from .config import SmartMoneyConfig
from .quantified_smc import detect_fvg, detect_order_block


class SmartMoneyEngine:
    """Detects order blocks, FVGs, and liquidity behavior with quantified rules."""

    def __init__(self, config: Optional[SmartMoneyConfig] = None):
        self.config = config or SmartMoneyConfig()
        self._liq_hunters: Dict[str, LiquidityHunter] = {}

    def analyze(self, frame: pd.DataFrame, timeframe: str = "5m") -> Dict[str, object]:
        if frame is None or frame.empty:
            raise ValueError("Frame is empty for smart money analysis")

        hunter_liq = self._liq_hunters.setdefault(timeframe, LiquidityHunter(timeframe=timeframe))

        df = frame.copy().sort_index()
        current_price = float(df["close"].iloc[-1])

        quantified = detect_order_block(
            df,
            impulse_multiplier=self.config.order_block_impulse_multiplier,
            lookback=self.config.order_block_lookback,
            volume_multiplier=self.config.order_block_volume_multiplier,
        )
        quantified = detect_fvg(quantified, min_gap=self.config.fvg_min_gap)

        hunter_liq.detect_equal_highs_lows(df)
        stop_hunts = hunter_liq.detect_stop_hunts(df)
        hunter_liq.update_sweep_status(current_price)

        order_blocks = self._extract_order_blocks(quantified)
        fvgs = self._extract_fvgs(quantified)

        price_at_ob, ob_signal, current_ob = self._price_in_order_block(current_price, order_blocks)
        price_in_fvg, fvg_signal, current_fvg = self._price_in_fvg(current_price, fvgs)

        previous_high = float(df["high"].iloc[-50:].max())
        previous_low = float(df["low"].iloc[-50:].min())
        liquidity_zones = {
            "equal_highs_lows": hunter_liq.to_dict_list(),
            "previous_high": previous_high,
            "previous_low": previous_low,
        }

        return {
            "order_blocks": order_blocks,
            "fvg": fvgs,
            "liquidity": {
                "zones": self.config.liquidity_zones,
                "levels": liquidity_zones,
                "stop_hunts": [
                    {
                        "id": hunt.hunt_id,
                        "direction": hunt.direction,
                        "swept_level": hunt.swept_level,
                        "confirmed": hunt.is_confirmed,
                    }
                    for hunt in stop_hunts
                ],
            },
            "price_at_ob": price_at_ob,
            "price_in_fvg": price_in_fvg,
            "ob_signal": ob_signal,
            "fvg_signal": fvg_signal,
            "current_ob": current_ob,
            "current_fvg": current_fvg,
            "liquidity_sweep": any(hunt.is_confirmed for hunt in stop_hunts),
            "current_price": current_price,
            "definitions": {
                "order_block": {
                    "candle_range": "last opposite candle before impulse",
                    "impulse_threshold": f"body > {self.config.order_block_impulse_multiplier}x avg_body",
                    "volume_condition": f"volume > {self.config.order_block_volume_multiplier}x avg_volume",
                    "mitigation": "price_return_within_zone",
                },
                "fvg": {
                    "definition": "3 candle imbalance",
                    "conditions": [
                        "candle1_high < candle3_low (bullish)",
                        "candle1_low > candle3_high (bearish)",
                    ],
                    "min_gap": self.config.fvg_min_gap,
                },
            },
        }

    def analyze_multi_timeframe(self, frames: Dict[str, pd.DataFrame]) -> Dict[str, object]:
        result: Dict[str, object] = {}
        for timeframe, frame in frames.items():
            if frame is None or frame.empty:
                continue
            result[timeframe] = self.analyze(frame, timeframe=timeframe)

        return result

    def _extract_order_blocks(self, frame: pd.DataFrame) -> List[Dict[str, object]]:
        blocks: List[Dict[str, object]] = []

        for idx in frame.index:
            ob_value = int(frame.at[idx, "ob"])
            if ob_value == 0:
                continue

            strength = float(frame.at[idx, "ob_strength"]) if not pd.isna(frame.at[idx, "ob_strength"]) else 0.0
            if strength < self.config.impulse_strength_threshold:
                continue

            top = float(frame.at[idx, "ob_top"]) if not pd.isna(frame.at[idx, "ob_top"]) else float(frame.at[idx, "high"])
            bottom = float(frame.at[idx, "ob_bottom"]) if not pd.isna(frame.at[idx, "ob_bottom"]) else float(frame.at[idx, "low"])

            blocks.append(
                {
                    "id": f"OBQ_{len(blocks) + 1}",
                    "timestamp": idx,
                    "type": "bullish" if ob_value > 0 else "bearish",
                    "status": "active",
                    "top": top,
                    "bottom": bottom,
                    "strength": strength,
                    "mitigation_pending": True,
                    "definition": (
                        "last bearish candle before bullish impulse"
                        if ob_value > 0
                        else "last bullish candle before bearish impulse"
                    ),
                }
            )

        return blocks[-100:]

    @staticmethod
    def _extract_fvgs(frame: pd.DataFrame) -> List[Dict[str, object]]:
        gaps: List[Dict[str, object]] = []

        for idx in frame.index:
            fvg_value = int(frame.at[idx, "fvg"])
            if fvg_value == 0:
                continue

            top = float(frame.at[idx, "fvg_top"]) if not pd.isna(frame.at[idx, "fvg_top"]) else 0.0
            bottom = float(frame.at[idx, "fvg_bottom"]) if not pd.isna(frame.at[idx, "fvg_bottom"]) else 0.0
            gap = float(frame.at[idx, "fvg_gap"]) if not pd.isna(frame.at[idx, "fvg_gap"]) else 0.0

            gaps.append(
                {
                    "id": f"FVGQ_{len(gaps) + 1}",
                    "timestamp": idx,
                    "type": "bullish" if fvg_value > 0 else "bearish",
                    "status": "open",
                    "top": top,
                    "bottom": bottom,
                    "size": gap,
                    "definition": "3 candle imbalance",
                }
            )

        return gaps[-150:]

    @staticmethod
    def _price_in_order_block(price: float, blocks: List[Dict[str, object]]) -> tuple[bool, int, Optional[Dict[str, object]]]:
        for block in reversed(blocks):
            if float(block["bottom"]) <= price <= float(block["top"]):
                signal = 1 if str(block["type"]) == "bullish" else -1
                return True, signal, block
        return False, 0, None

    @staticmethod
    def _price_in_fvg(price: float, gaps: List[Dict[str, object]]) -> tuple[bool, int, Optional[Dict[str, object]]]:
        for gap in reversed(gaps):
            lower = min(float(gap["bottom"]), float(gap["top"]))
            upper = max(float(gap["bottom"]), float(gap["top"]))
            if lower <= price <= upper:
                signal = 1 if str(gap["type"]) == "bullish" else -1
                return True, signal, gap
        return False, 0, None
