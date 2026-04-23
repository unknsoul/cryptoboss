"""
Mock Price Generator — Paper Mode Price Feed

Generates realistic cryptocurrency price ticks using geometric Brownian motion.
Used exclusively in paper trading mode when no exchange client is available.

Usage:
    gen = MockPriceGenerator(symbols={"BTC/USDT": 67000.0, "ETH/USDT": 3200.0})
    async for symbol, price, ts in gen.stream(interval=1.0):
        print(f"{symbol}: ${price:.2f}")
"""

import asyncio
import logging
import math
import random
import time
from datetime import datetime
from typing import AsyncGenerator, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class MockPriceGenerator:
    """
    Generates realistic price ticks for paper trading.

    Uses geometric Brownian motion with mean-reversion
    to produce believable price action for scalping strategies.

    Parameters:
        symbols: Dict of symbol -> starting price
        volatility: Per-tick volatility (default 0.0008 = 0.08%)
        mean_reversion: Strength of pull back to base price (0.0 = none, 1.0 = strong)
        tick_interval: Seconds between price ticks
    """

    def __init__(
        self,
        symbols: Optional[Dict[str, float]] = None,
        volatility: float = 0.0008,
        mean_reversion: float = 0.001,
        tick_interval: float = 1.0,
    ):
        self.symbols = symbols or {
            "BTC/USDT": 67000.0,
            "ETH/USDT": 3200.0,
        }
        self.volatility = volatility
        self.mean_reversion = mean_reversion
        self.tick_interval = tick_interval

        # Current prices (evolve over time)
        self._prices: Dict[str, float] = dict(self.symbols)
        # Base prices (for mean reversion)
        self._base_prices: Dict[str, float] = dict(self.symbols)

        self._running = False
        self._tick_count = 0

        logger.info(
            f"MockPriceGenerator initialized: {list(self.symbols.keys())} "
            f"(vol={volatility}, interval={tick_interval}s)"
        )

    def _next_price(self, symbol: str) -> float:
        """Generate next price tick using geometric Brownian motion with mean reversion."""
        current = self._prices[symbol]
        base = self._base_prices[symbol]

        # Random component (geometric Brownian motion)
        drift = 0.0
        shock = random.gauss(0, self.volatility)

        # Mean reversion pull
        log_ratio = math.log(current / base)
        reversion = -self.mean_reversion * log_ratio

        # Apply price change
        pct_change = drift + shock + reversion
        new_price = current * math.exp(pct_change)

        # Ensure price stays positive and within reasonable bounds
        new_price = max(new_price, base * 0.5)
        new_price = min(new_price, base * 2.0)

        self._prices[symbol] = new_price
        return round(new_price, 2)

    async def stream(
        self,
        interval: Optional[float] = None,
    ) -> AsyncGenerator[Tuple[str, float, datetime], None]:
        """
        Async generator that yields price ticks.

        Yields:
            Tuple of (symbol, price, timestamp)
        """
        interval = interval or self.tick_interval
        self._running = True

        logger.info(f"MockPriceGenerator streaming started (interval={interval}s)")

        try:
            while self._running:
                for symbol in self.symbols:
                    price = self._next_price(symbol)
                    ts = datetime.now()
                    self._tick_count += 1
                    yield (symbol, price, ts)

                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("MockPriceGenerator stream cancelled")
        finally:
            self._running = False
            logger.info(
                f"MockPriceGenerator stopped after {self._tick_count} ticks"
            )

    def stop(self):
        """Stop the price stream."""
        self._running = False

    def get_price(self, symbol: str) -> float:
        """Get the current mock price for a symbol."""
        return self._prices.get(symbol, 0.0)

    def get_all_prices(self) -> Dict[str, float]:
        """Get all current mock prices."""
        return dict(self._prices)

    @property
    def is_running(self) -> bool:
        return self._running
