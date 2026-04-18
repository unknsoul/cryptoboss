"""Orchestrator v4 wiring dual-source price feed and Binance execution into v3 pipeline."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.exchange.binance_unified import BinanceUnified
from src.exchange.exchange_factory import ExchangeFactory
from src.exchange.price_feed import PriceFeed
from src.smc.multi_timeframe_analyzer import MultiTimeframeAnalyzer
from src.strategies.pro_strategy_builder import ProStrategyBuilder

from .config_v4 import V4SystemConfig
from .orchestrator import IntradayScalperV3System

logger = logging.getLogger(__name__)


class OrchestratorV4:
    """v4 orchestrator for MT5-price + Binance-execution architecture."""

    def __init__(self, config: Optional[V4SystemConfig] = None):
        self.config = config or V4SystemConfig.from_env()
        self.exchange = ExchangeFactory.from_env()
        self.price_feed: PriceFeed = self.exchange.price_feed
        self.executor: BinanceUnified = self.exchange.executor
        self._v3 = IntradayScalperV3System()
        self.strategy_builder = ProStrategyBuilder()
        self.mtf_analyzer = MultiTimeframeAnalyzer()

        logger.info(
            "OrchestratorV4 initialized | price_source=%s | execution_mode=%s",
            self.price_feed.active_source,
            self.executor.mode,
        )

    def run_cycle(
        self,
        symbol: str,
        timeframes: Optional[List[str]] = None,
        limit: int = 500,
        strategy_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        tf = timeframes or ["1m", "5m", "15m"]
        frames = self.price_feed.get_multi_timeframe(symbol=symbol, timeframes=tf, limit=limit)
        mtf_result = self.mtf_analyzer.analyze(frames)

        try:
            balance = self.executor.get_balance()
            account_state = {
                "equity": float(balance.get("total_usdt", 0.0)),
                "balance": float(balance.get("free_usdt", 0.0)),
                "last_price": float(self.price_feed.get_last_price(symbol)),
            }
        except Exception as exc:
            logger.warning("Could not fetch Binance balance: %s", exc)
            account_state = {
                "equity": 0.0,
                "balance": 0.0,
                "last_price": float(self.price_feed.get_last_price(symbol)),
            }

        cycle = self._v3.run_cycle(
            symbol=symbol,
            frames_by_timeframe=frames,
            account_state=account_state,
            strategy_used=strategy_id or "v4_default",
        )
        cycle["mtf_alignment"] = mtf_result
        cycle["price_source"] = self.price_feed.active_source
        cycle["execution_mode"] = self.executor.mode
        cycle["spread"] = self.price_feed.get_spread(symbol)
        return cycle

    def build_strategy(self, name: str, symbol: str = "BTC/USDT", timeframe: str = "5m") -> str:
        return self.strategy_builder.new(name=name, symbol=symbol, timeframe=timeframe)

    def load_preset(self, preset_name: str, symbol: str = "BTC/USDT", timeframe: str = "5m") -> str:
        return self.strategy_builder.load_preset(preset_name, symbol=symbol, timeframe=timeframe)

    def score_strategy(self, strategy_id: str):
        return self.strategy_builder.score(strategy_id)

    def validate_strategy(self, strategy_id: str):
        return self.strategy_builder.validate(strategy_id)

    def export_strategy(self, strategy_id: str) -> str:
        return self.strategy_builder.export_json(strategy_id)

    def import_strategy(self, json_str: str) -> str:
        return self.strategy_builder.import_json(json_str)

    def get_canvas(self, strategy_id: str) -> Dict[str, Any]:
        return self.strategy_builder.get_canvas_data(strategy_id)

    def status(self) -> Dict[str, Any]:
        return {
            "config": self.config.summary(),
            "exchange": self.exchange.status(),
            "strategies": self.strategy_builder.list(),
        }

    def __repr__(self) -> str:
        return f"OrchestratorV4(price={self.price_feed.active_source}, execution={self.executor.mode})"
