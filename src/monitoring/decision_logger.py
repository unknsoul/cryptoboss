"""Decision audit logging."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict


@dataclass
class TradeDecision:
    timestamp: str
    symbol: str
    side: str
    entry_price: float
    size_usdt: float
    risk_pct: float
    sl_price: float
    tp_price: float
    rr_ratio: float
    regime: str
    ml_confidence: float
    meta_confidence: float
    smc_signal: str
    funding_rate: float
    strategy: str
    kelly_fraction: float
    exit_price: float = 0.0
    exit_reason: str = ""
    actual_slippage_bps: float = 0.0
    fees_paid: float = 0.0
    net_pnl_usdt: float = 0.0
    decision_reason: str = ""
    extra: Dict = field(default_factory=dict)

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()


class DecisionLogger:
    """Append structured trade decisions to JSONL."""

    def __init__(self, path: str = "logs/decisions.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, decision: TradeDecision) -> None:
        payload = asdict(decision)
        if payload.get("extra"):
            payload.update(payload.pop("extra"))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")
"""Decision audit logging."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, Optional


@dataclass
class TradeDecision:
    timestamp: str
    symbol: str
    side: str
    entry_price: float
    size_usdt: float
    risk_pct: float
    sl_price: float
    tp_price: float
    rr_ratio: float
    regime: str
    ml_confidence: float
    meta_confidence: float
    smc_signal: str
    funding_rate: float
    strategy: str
    kelly_fraction: float
    exit_price: float = 0.0
    exit_reason: str = ""
    actual_slippage_bps: float = 0.0
    fees_paid: float = 0.0
    net_pnl_usdt: float = 0.0
    decision_reason: str = ""
    extra: Dict = field(default_factory=dict)

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()


class DecisionLogger:
    """Append structured trade decisions to JSONL."""

    def __init__(self, path: str = "logs/decisions.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, decision: TradeDecision) -> None:
        payload = asdict(decision)
        if payload.get("extra"):
            payload.update(payload.pop("extra"))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")
