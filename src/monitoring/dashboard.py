"""Prometheus metrics publishing."""

from __future__ import annotations

from typing import Dict, Iterable

from prometheus_client import Gauge, start_http_server


class MetricsPublisher:
    """Expose runtime metrics via Prometheus gauges."""

    def __init__(self) -> None:
        self.pnl_usdt = Gauge("cryptoboss_pnl_usdt", "Current PnL in USDT")
        self.win_rate = Gauge("cryptoboss_win_rate", "Win rate percentage")
        self.open_positions = Gauge("cryptoboss_open_positions", "Open positions count")
        self.slippage_bps = Gauge("cryptoboss_slippage_bps", "Average slippage in bps")
        self.regime_state = Gauge("cryptoboss_regime_state", "Current regime", ["regime"])

    def update(self, snapshot: Dict[str, float], regime: str | None = None, regimes: Iterable[str] = ()) -> None:
        """Update metrics from a snapshot dictionary."""
        if "pnl_usdt" in snapshot:
            self.pnl_usdt.set(float(snapshot["pnl_usdt"]))
        if "win_rate" in snapshot:
            self.win_rate.set(float(snapshot["win_rate"]))
        if "open_positions" in snapshot:
            self.open_positions.set(float(snapshot["open_positions"]))
        if "slippage_bps" in snapshot:
            self.slippage_bps.set(float(snapshot["slippage_bps"]))

        if regime:
            labels = set(regimes) | {regime}
            for label in labels:
                self.regime_state.labels(regime=label).set(1.0 if label == regime else 0.0)


def start_metrics_server(port: int = 9090) -> None:
    """Start Prometheus metrics server."""
    start_http_server(port)
