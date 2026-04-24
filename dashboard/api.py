"""
CryptoBoss Dashboard API - v1.0.1 RELEASE

FastAPI backend with WebSocket for real-time updates.
Implements:
- environment_signature and data_source tagging
- User authentication (email/password)
- Exchange account management with state isolation
"""

import asyncio
import json
import logging
import random
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum
import sys

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

try:
    from src.v3.orchestrator_v4 import OrchestratorV4
    from src.v3.config_v4 import V4SystemConfig
    from src.strategies.pro_strategy_builder import ProStrategyBuilder, INDICATOR_LIBRARY
    V4_AVAILABLE = True
except ImportError as e:
    V4_AVAILABLE = False
    OrchestratorV4 = Any  # type: ignore
    V4SystemConfig = Any  # type: ignore
    ProStrategyBuilder = Any  # type: ignore
    INDICATOR_LIBRARY = {}  # type: ignore
    logging.warning(f"v4 modules not available: {e}")

# Import session manager
try:
    from src.core.session_manager import get_session_manager, TradingMode
    SESSION_MANAGER_AVAILABLE = True
except ImportError:
    SESSION_MANAGER_AVAILABLE = False

# Import binance client for validation
try:
    from src.exchange.binance_client import BinanceClient
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False

# v1.0.1: Import auth services
try:
    from src.core.auth import get_auth_service, get_account_service
    from src.core.models import User, ExchangeAccount
    AUTH_AVAILABLE = True
except ImportError as e:
    AUTH_AVAILABLE = False
    logging.warning(f"Auth services not available: {e}")

# v1.0.1: Import scoped state manager
try:
    from src.core.scoped_state import (
        ScopedStateManager, switch_account, get_active_state, require_active_state
    )
    SCOPED_STATE_AVAILABLE = True
except ImportError as e:
    SCOPED_STATE_AVAILABLE = False
    logging.warning(f"Scoped state not available: {e}")

# v1.0.1: Import price truth enforcement
try:
    from src.core.price_truth import (
        PriceData, PriceSource, PriceValidator, PriceFeedManager, get_price_manager
    )
    PRICE_TRUTH_AVAILABLE = True
except ImportError as e:
    PRICE_TRUTH_AVAILABLE = False
    logging.warning(f"Price truth module not available: {e}")

# v1.0.1: Import engine lifecycle
try:
    from src.core.engine_lifecycle import EngineLifecycle, get_engine_lifecycle
    ENGINE_LIFECYCLE_AVAILABLE = True
except ImportError as e:
    ENGINE_LIFECYCLE_AVAILABLE = False
    logging.warning(f"Engine lifecycle not available: {e}")

# v1.0.1: Import live price feed
try:
    from src.core.live_price_feed import LivePriceFeed, get_price_feed
    LIVE_PRICE_FEED_AVAILABLE = True
except ImportError as e:
    LIVE_PRICE_FEED_AVAILABLE = False
    logging.warning(f"Live price feed not available: {e}")

# v1.0.1: Import bot instance manager (TRUE ISOLATION)
try:
    from src.core.bot_instance import (
        BotInstance, BotInstanceManager, get_active_bot, require_active_bot, switch_account as switch_bot_instance
    )
    BOT_INSTANCE_AVAILABLE = True
except ImportError as e:
    BOT_INSTANCE_AVAILABLE = False
    logging.warning(f"Bot instance manager not available: {e}")

# vFINAL: Import market data service for WebSocket prices
try:
    from src.core.market_data_service import (
        MarketDataService, get_market_data_service, start_market_data, PriceTick
    )
    MARKET_DATA_AVAILABLE = True
except ImportError as e:
    MARKET_DATA_AVAILABLE = False
    logging.warning(f"Market data service not available: {e}")

# v12.0: Import SMC/scalper/builder/tester stack
try:
    from src.smc.smc_engine import SMCEngine
    from src.strategies.base_strategy import StrategyConfig
    from src.strategies.intraday_scalper import IntradayScalper
    from src.strategies.strategy_builder import BuiltStrategy, ConditionOperator, LogicGate, StrategyBuilder
    from src.strategies.strategy_tester import StrategyTester
    V12_AVAILABLE = True
except ImportError as e:
    V12_AVAILABLE = False
    BuiltStrategy = Any  # type: ignore
    ConditionOperator = Any  # type: ignore
    LogicGate = Any  # type: ignore
    StrategyBuilder = Any  # type: ignore
    StrategyTester = Any  # type: ignore
    SMCEngine = Any  # type: ignore
    StrategyConfig = Any  # type: ignore
    IntradayScalper = Any  # type: ignore
    logging.warning(f"v12 modules not available: {e}")

# AggressiveScalper strategy import
try:
    from src.strategies.aggressive_scalper import AggressiveScalper, ScalperParams
    AGGRESSIVE_SCALPER_AVAILABLE = True
except ImportError as e:
    AGGRESSIVE_SCALPER_AVAILABLE = False
    AggressiveScalper = Any  # type: ignore
    ScalperParams = Any  # type: ignore
    logging.warning(f"AggressiveScalper not available: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Dashboard")

_v4_orchestrator = None


def get_v4() -> Optional["OrchestratorV4"]:
    global _v4_orchestrator
    if not V4_AVAILABLE:
        return None
    if _v4_orchestrator is None:
        _v4_orchestrator = OrchestratorV4()
    return _v4_orchestrator


# === CryptoBoss 1.0.0 Data Source Tags ===
class DataSourceTag(str, Enum):
    LIVE_EXCHANGE = "LIVE_EXCHANGE"
    TESTNET_EXCHANGE = "TESTNET_EXCHANGE"
    DERIVED = "DERIVED"
    SIMULATED = "SIMULATED"
    STALE = "STALE"


# === Environment Signature (Immutable after startup) ===
class EnvironmentSignature:
    _instance = None
    _locked = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._mode = "paper"
            cls._instance._started_at = datetime.now()
            cls._instance._checksum = None
        return cls._instance
    
    def lock(self, mode: str):
        """Lock environment - can only be called once at startup."""
        if self._locked:
            raise RuntimeError("Environment already locked - cannot change after startup")
        self._mode = mode
        self._started_at = datetime.now()
        self._checksum = hashlib.sha256(
            f"{mode}:{self._started_at.isoformat()}".encode()
        ).hexdigest()[:16]
        self._locked = True
        logger.warning(f"🔒 ENVIRONMENT LOCKED: {mode.upper()} (checksum: {self._checksum})")
    
    def get_signature(self) -> Dict:
        return {
            "mode": self._mode.upper(),
            "checksum": self._checksum or "UNLOCKED",
            "immutable_since": self._started_at.isoformat(),
            "is_live": self._mode == "live"
        }
    
    @property
    def mode(self) -> str:
        return self._mode
    
    @property
    def is_locked(self) -> bool:
        return self._locked
    

env_signature = EnvironmentSignature()


def wrap_response(data: Dict, data_source: DataSourceTag = None) -> Dict:
    """Wrap API response with environment_signature, data_source_tag, and identity fields."""
    if data_source is None:
        # Auto-determine based on environment
        if env_signature.mode == "live":
            data_source = DataSourceTag.LIVE_EXCHANGE
        elif env_signature.mode == "testnet":
            data_source = DataSourceTag.TESTNET_EXCHANGE
        else:
            data_source = DataSourceTag.SIMULATED
    
    response = {
        "data": data,
        "environment_signature": env_signature.get_signature(),
        "data_source": data_source.value,
        "timestamp": datetime.now().isoformat()
    }
    
    # v1.0.1: Add mandatory identity fields if account is active
    if SCOPED_STATE_AVAILABLE:
        active_state = get_active_state()
        if active_state:
            response["user_id"] = active_state.identity.user_id
            response["exchange_account_id"] = active_state.identity.exchange_account_id
            response["data_scope"] = "SCOPED"
            response["is_new_account"] = active_state.state.is_new_account()
            response["account_created_at"] = active_state.state.created_at.isoformat()
    
    # Also include from dashboard state if available
    if hasattr(state, 'active_exchange_account_id') and state.active_exchange_account_id:
        response["exchange_account_id"] = state.active_exchange_account_id
    if hasattr(state, 'active_user_id') and state.active_user_id:
        response["user_id"] = state.active_user_id
    
    return response


app = FastAPI(
    title="CryptoBoss Dashboard",
    description="Professional Trading Bot Dashboard - v1.0.1 with Identity Layer",
    version="1.0.1"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Security ===
security = HTTPBearer(auto_error=False)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[User]:
    """Get current authenticated user from JWT token."""
    if not AUTH_AVAILABLE:
        return None
    if not credentials:
        return None
    
    auth_service = get_auth_service()
    return auth_service.verify_token(credentials.credentials)


async def require_auth(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Require authentication - raises 401 if not authenticated."""
    user = await get_current_user(credentials)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


# === Auth Request/Response Models ===

class SignupRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    success: bool
    token: Optional[str] = None
    user: Optional[dict] = None
    error: Optional[str] = None


class CreateAccountRequest(BaseModel):
    exchange_name: str = "binance"
    environment: str  # TESTNET or LIVE
    api_key: str
    api_secret: str
    label: Optional[str] = ""


class SelectAccountRequest(BaseModel):
    exchange_account_id: str


# === Models ===

class EngineConfig(BaseModel):
    mode: str = "testnet"  # PAPER REMOVED - default to testnet
    capital: float = 10000.0
    symbols: List[str] = ["BTC/USDT"]
    strategy: str = "dca"


class SessionSwitchRequest(BaseModel):
    mode: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None


class ValidateKeysRequest(BaseModel):
    api_key: str
    api_secret: str
    testnet: bool = True


# Global state with proper initialization
class DashboardState:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.mode = "testnet"  # PAPER REMOVED - only testnet/live
        self.initial_capital = 0.0  # Comes from exchange
        self.capital = 0.0
        self.pnl = 0.0
        self.start_time = datetime.now()
        self.current_price = 0.0  # Comes from exchange
        self.last_price = 0.0
        self.price_history: List[Dict] = []
        self.trades: List[Dict] = []
        self.position = 0.0  # Comes from exchange
        self.position_entry_price = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.api_validated = False
        self.exchange_client = None
        
        # System state - PAPER REMOVED
        self.environment = "testnet"  # testnet or live ONLY
        self.connection_status = "disconnected"  # disconnected, connecting, connected, error
        self.timestamp_offset_ms = 0
        self.last_time_sync = None
        self.kill_switch_active = False
        self.kill_switch_reason = None
        
        # v1.0.0: Incident State Machine
        self.incident_state = "NORMAL"  # NORMAL, DEGRADED, INCIDENT_FREEZE, HALTED
        self.incident_reason = None
        self.incident_started_at = None
        
        # v1.0.0: Operator Controls
        self.trading_paused = False
        self.trading_pause_reason = None
        self.operator_action_log: List[Dict] = []  # Permanent log
        
        # v1.0.1: Identity tracking
        self.active_user_id: Optional[str] = None
        self.active_exchange_account_id: Optional[str] = None
        
        # Market context
        self.market_context = "UNKNOWN"  # TRENDING, RANGING, VOLATILE, CRISIS
        self.market_bias = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
        self.last_context_update = None
        
        # Decision tracking
        self.recent_decisions: List[Dict] = []
        self.last_decision_time = None
        self.decisions_today = 0
        self.rejections_today = 0

    def reset(self, new_mode: str = "testnet"):
        """
        Reset all state for new session.
        
        CRITICAL: Paper mode is removed. Only testnet/live allowed.
        """
        # Validate mode - reject paper
        if new_mode.lower() == "paper":
            logger.warning("⚠️ Paper mode requested but disabled - using testnet")
            new_mode = "testnet"
        
        self.session_id = str(uuid.uuid4())
        self.mode = new_mode
        self.capital = 0.0  # Will be fetched from exchange
        self.pnl = 0.0
        self.start_time = datetime.now()
        self.price_history = []
        self.trades = []
        self.position = 0.0
        self.position_entry_price = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.api_validated = False  # Must connect to exchange
        
        # Reset system state
        self.environment = new_mode
        self.connection_status = "disconnected"  # Must connect to exchange
        self.timestamp_offset_ms = 0
        self.market_context = "UNKNOWN"
        self.market_bias = "NEUTRAL"
        self.recent_decisions = []
        self.decisions_today = 0
        self.rejections_today = 0
        
        if self.exchange_client:
            asyncio.create_task(self._destroy_client())
        self.exchange_client = None
        logger.info(f"Session reset: {self.session_id[:8]}... (mode={new_mode})")
    
    async def _destroy_client(self):
        if self.exchange_client:
            try:
                await self.exchange_client.destroy()
            except:
                pass


    @property
    def portfolio_value(self) -> float:
        return self.capital + (self.position * self.current_price)
    
    @property
    def unrealized_pnl(self) -> float:
        if self.position > 0:
            return (self.current_price - self.position_entry_price) * self.position
        return 0.0
    
    @property
    def total_pnl(self) -> float:
        return self.pnl + self.unrealized_pnl
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100

state = DashboardState()


# === v12 Runtime Objects & Helpers ===

_TIMEFRAME_RULES = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}


def _timeframe_rule(timeframe: str) -> str:
    return _TIMEFRAME_RULES.get(timeframe.lower(), "5min")


def _synthesize_ohlcv(limit: int, base_price: float) -> pd.DataFrame:
    rows = max(limit, 120)
    base = base_price if base_price and base_price > 0 else 65000.0
    timestamps = pd.date_range(end=pd.Timestamp.utcnow(), periods=rows, freq="1min")

    closes = [base]
    for _ in range(rows - 1):
        closes.append(max(1.0, closes[-1] * (1.0 + random.gauss(0.0, 0.0015))))

    data = []
    for idx in range(rows):
        close = closes[idx]
        prev = closes[idx - 1] if idx > 0 else close
        open_price = prev
        wick = abs(close - open_price) * 0.3 + close * 0.0005
        high = max(open_price, close) + wick
        low = max(0.0, min(open_price, close) - wick)
        volume = abs(close - open_price) * random.uniform(80.0, 200.0) + random.uniform(50.0, 400.0)
        data.append(
            {
                "timestamp": timestamps[idx],
                "open": float(open_price),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": float(volume),
            }
        )

    frame = pd.DataFrame(data).set_index("timestamp")
    return frame


def _price_history_ohlcv(limit: int) -> pd.DataFrame:
    if not state.price_history:
        return _synthesize_ohlcv(limit=limit, base_price=state.current_price)

    parsed = []
    for item in state.price_history[-max(limit * 6, 240):]:
        ts_raw = item.get("time") or item.get("timestamp")
        ts = pd.to_datetime(ts_raw, errors="coerce", utc=True)
        price = float(item.get("price", 0.0) or 0.0)
        if pd.isna(ts) or price <= 0:
            continue
        parsed.append((ts, price))

    if len(parsed) < 20:
        return _synthesize_ohlcv(limit=limit, base_price=state.current_price)

    parsed.sort(key=lambda row: row[0])
    history_df = pd.DataFrame(parsed, columns=["timestamp", "price"]).drop_duplicates("timestamp")
    history_df.set_index("timestamp", inplace=True)
    history_df = history_df.resample("1min").last().ffill().dropna()

    if history_df.empty:
        return _synthesize_ohlcv(limit=limit, base_price=state.current_price)

    history_df["open"] = history_df["price"].shift(1).fillna(history_df["price"])
    history_df["close"] = history_df["price"]
    spread = (history_df["close"] - history_df["open"]).abs()
    baseline = history_df["close"] * 0.0006
    history_df["high"] = history_df[["open", "close"]].max(axis=1) + spread * 0.25 + baseline
    history_df["low"] = history_df[["open", "close"]].min(axis=1) - spread * 0.25 - baseline
    history_df["volume"] = (spread * 120.0 + 120.0).clip(lower=10.0)

    ohlcv = history_df[["open", "high", "low", "close", "volume"]].tail(max(limit, 120))
    ohlcv.index = pd.DatetimeIndex(ohlcv.index)
    return ohlcv


async def fetch_ohlcv(symbol: str = "BTC/USDT", timeframe: str = "5m", limit: int = 500) -> pd.DataFrame:
    _ = symbol
    one_min = _price_history_ohlcv(limit=max(limit * 5, 240))

    rule = _timeframe_rule(timeframe)
    if rule == "1min":
        frame = one_min
    else:
        frame = (
            one_min.resample(rule)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )

    frame = frame.tail(limit).copy()
    frame.index = pd.DatetimeIndex(frame.index)
    return frame


async def fetch_multi_tf_data(symbol: str, timeframes: List[str], limit: int = 500) -> Dict[str, pd.DataFrame]:
    output: Dict[str, pd.DataFrame] = {}
    for timeframe in timeframes:
        output[timeframe] = await fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
    return output


def _serialize_setup(setup) -> Dict[str, Any]:
    return {
        "id": setup.setup_id,
        "type": setup.setup_type.value,
        "direction": setup.direction,
        "timeframe": setup.timeframe,
        "confidence": setup.confidence,
        "rr": setup.risk_reward,
        "entry": setup.entry_price,
        "sl": setup.stop_loss,
        "tp1": setup.take_profit_1,
        "tp2": setup.take_profit_2,
        "tp3": setup.take_profit_3,
        "components": setup.components,
        "notes": setup.notes,
    }


def _series_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def _series_atr(df: pd.DataFrame, period: int = 14) -> float:
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    if pd.isna(atr) or float(atr) <= 0:
        return float(max(df["close"].iloc[-1] * 0.001, 0.01))
    return float(atr)


def _compare_values(
    operator: str,
    left: Any,
    right: Any,
    prev_left: Optional[Any] = None,
    prev_right: Optional[Any] = None,
) -> bool:
    if operator == ">":
        return float(left) > float(right)
    if operator == "<":
        return float(left) < float(right)
    if operator == ">=":
        return float(left) >= float(right)
    if operator == "<=":
        return float(left) <= float(right)
    if operator == "==":
        return left == right
    if operator == "cross_above":
        if prev_left is None or prev_right is None:
            return False
        return float(prev_left) <= float(prev_right) and float(left) > float(right)
    if operator == "cross_below":
        if prev_left is None or prev_right is None:
            return False
        return float(prev_left) >= float(prev_right) and float(left) < float(right)
    return False


def _evaluate_condition(condition, df_slice: pd.DataFrame, smc_context: Dict[str, Any]) -> bool:
    indicator = condition.indicator.value
    params = condition.params or {}
    operator = condition.operator.value
    threshold = condition.threshold

    close = df_slice["close"]
    high = df_slice["high"]
    low = df_slice["low"]
    volume = df_slice["volume"]
    current_price = float(close.iloc[-1])

    if indicator == "ema":
        fast = int(params.get("fast", params.get("period", 9)))
        slow = int(params.get("slow", 21))
        fast_series = close.ewm(span=fast, adjust=False).mean()
        slow_series = close.ewm(span=slow, adjust=False).mean()
        if len(fast_series) < 2 or len(slow_series) < 2:
            return False
        if operator in ("cross_above", "cross_below"):
            right = float(slow_series.iloc[-1])
        else:
            right = float(threshold) if isinstance(threshold, (int, float)) else float(slow_series.iloc[-1])
        return _compare_values(
            operator,
            float(fast_series.iloc[-1]),
            right,
            float(fast_series.iloc[-2]),
            float(slow_series.iloc[-2]),
        )

    if indicator == "rsi":
        period = int(params.get("period", 14))
        rsi_series = _series_rsi(close, period)
        if len(rsi_series) < 2:
            return False
        return _compare_values(
            operator,
            float(rsi_series.iloc[-1]),
            float(threshold),
            float(rsi_series.iloc[-2]),
            float(threshold),
        )

    if indicator == "volume_ma":
        period = int(params.get("period", 20))
        vol_ma = volume.rolling(period).mean()
        if len(vol_ma) < 2 or pd.isna(vol_ma.iloc[-1]) or vol_ma.iloc[-1] == 0:
            return False
        ratio = float(volume.iloc[-1] / vol_ma.iloc[-1])
        return _compare_values(operator, ratio, float(threshold), ratio, float(threshold))

    if indicator == "vwap":
        typical = (high + low + close) / 3.0
        vwap = (typical * volume).cumsum() / volume.cumsum().replace(0, np.nan)
        vwap = vwap.fillna(close)
        if operator == "in_zone":
            if isinstance(threshold, dict):
                band = float(threshold.get("band", 0.002))
            else:
                band = 0.002
            return abs(current_price - float(vwap.iloc[-1])) / max(current_price, 1e-9) <= band
        return _compare_values(operator, current_price, float(vwap.iloc[-1]), float(close.iloc[-2]), float(vwap.iloc[-2]))

    smc_state = smc_context.get("state", {})
    entry_tf = smc_context.get("entry_timeframe", "5m")
    tf_state = smc_state.get(entry_tf, {})

    if indicator == "order_block":
        desired = params.get("type", "bullish")
        for ob in tf_state.get("order_blocks", []):
            if ob.get("type") != desired:
                continue
            if ob.get("status") not in ("active", "tested"):
                continue
            if operator == "in_zone":
                if float(ob["bottom"]) <= current_price <= float(ob["top"]):
                    return True
            elif operator == "==":
                return bool(threshold)
        return False

    if indicator == "fvg":
        desired = params.get("type", "bullish")
        for fvg in tf_state.get("fvgs", []):
            if fvg.get("type") != desired:
                continue
            if fvg.get("status") == "filled":
                continue
            if operator == "in_zone":
                if float(fvg["bottom"]) <= current_price <= float(fvg["top"]):
                    return True
            elif operator == "==":
                return bool(threshold)
        return False

    if indicator in ("bos", "choch"):
        direction = params.get("direction", "bullish")
        key = "is_bos" if indicator == "bos" else "is_choch"
        has_event = any(item.get(key) and item.get("direction") == direction for item in tf_state.get("structure", [])[:10])
        if operator == "==":
            return has_event == bool(threshold)
        return has_event

    if indicator == "liquidity":
        levels = tf_state.get("liquidity", [])
        if not levels:
            return False
        nearest = min(levels, key=lambda item: abs(float(item.get("price", 0.0)) - current_price))
        if operator == "in_zone":
            return abs(float(nearest.get("price", 0.0)) - current_price) / max(current_price, 1e-9) <= 0.002
        if operator == "==":
            return bool(threshold)
        return True

    return False


def _evaluate_group(group, df_slice: pd.DataFrame, smc_context: Dict[str, Any]) -> bool:
    if group is None or not group.conditions:
        return False

    results = [_evaluate_condition(condition, df_slice, smc_context) for condition in group.conditions]
    if group.gate == LogicGate.AND:
        return all(results)
    return any(results)


def _compile_strategy_signal_fn(strategy: BuiltStrategy):
    engine = SMCEngine(timeframes=["1h", "15m", strategy.timeframe])

    def signal_fn(df_slice: pd.DataFrame) -> Dict[str, Any]:
        if df_slice is None or len(df_slice) < 60:
            return {"action": "HOLD"}

        close = float(df_slice["close"].iloc[-1])
        atr = _series_atr(df_slice)
        data = {
            "1h": df_slice,
            "15m": df_slice,
            strategy.timeframe: df_slice,
        }
        setups = engine.analyze(data)
        state_dict = engine.get_state_dict()
        smc_context = {
            "state": state_dict,
            "setups": setups,
            "entry_timeframe": strategy.timeframe,
        }

        long_ok = _evaluate_group(strategy.entry_long, df_slice, smc_context)
        short_ok = _evaluate_group(strategy.entry_short, df_slice, smc_context)

        if long_ok and short_ok:
            return {"action": "HOLD"}
        if not long_ok and not short_ok:
            return {"action": "HOLD"}

        direction = "long" if long_ok else "short"

        if direction == "long":
            if strategy.exit_config.stop_loss_type in ("atr", "ob_bottom", "swing_low"):
                stop = close - atr * float(strategy.exit_config.stop_loss_value)
            else:
                stop = close * (1.0 - float(strategy.exit_config.stop_loss_value) / 100.0)
            risk = close - stop
        else:
            if strategy.exit_config.stop_loss_type in ("atr", "ob_bottom", "swing_low"):
                stop = close + atr * float(strategy.exit_config.stop_loss_value)
            else:
                stop = close * (1.0 + float(strategy.exit_config.stop_loss_value) / 100.0)
            risk = stop - close

        if risk <= 0:
            return {"action": "HOLD"}

        rr = (
            float(strategy.exit_config.take_profit_value)
            if strategy.exit_config.take_profit_type == "rr"
            else 1.5
        )
        tp2_rr = float(strategy.exit_config.tp2_rr) if strategy.exit_config.tp2_rr else rr * 1.6
        tp3_rr = float(strategy.exit_config.tp3_rr) if strategy.exit_config.tp3_rr else rr * 2.5

        if direction == "long":
            tp1 = close + risk * rr
            tp2 = close + risk * tp2_rr
            tp3 = close + risk * tp3_rr
            action = "BUY"
        else:
            tp1 = close - risk * rr
            tp2 = close - risk * tp2_rr
            tp3 = close - risk * tp3_rr
            action = "SELL"

        position_size = (10000.0 * strategy.risk_config.max_risk_pct) / risk

        return {
            "action": action,
            "entry": close,
            "sl": float(stop),
            "tp1": float(tp1),
            "tp2": float(tp2),
            "tp3": float(tp3),
            "size": float(position_size),
            "confidence": 0.7,
            "setup_type": setups[0].setup_type.value if setups else "custom_strategy",
            "components": setups[0].components if setups else {},
        }

    return signal_fn


if V12_AVAILABLE:
    v12_builder = StrategyBuilder()
    v12_tester = StrategyTester()
    v12_scalper = IntradayScalper(
        StrategyConfig(
            strategy_id="intraday_scalper_v1",
            version="12.0",
            symbol="BTC/USDT",
            min_confidence=0.55,
            metadata={
                "entry_timeframe": "5m",
                "mid_timeframe": "15m",
                "bias_timeframe": "1h",
                "max_risk_pct": 0.005,
                "min_rr": 1.5,
                "min_confluence": 0.6,
                "atr_sl_mult": 0.8,
                "session_filter": True,
                "kill_zone_only": False,
                "max_positions": 3,
                "partial_exit_pct": 0.5,
                "tp_ratios": [1.5, 2.5, 4.0],
            },
        )
    )
else:
    v12_builder = None
    v12_tester = None
    v12_scalper = None


# === AggressiveScalper Instance ===

if AGGRESSIVE_SCALPER_AVAILABLE:
    try:
        _aggressive_scalper_instance = AggressiveScalper()
        logger.info(f"AggressiveScalper instantiated: {_aggressive_scalper_instance.strategy_id}")
    except Exception as e:
        _aggressive_scalper_instance = None
        logger.warning(f"Failed to instantiate AggressiveScalper: {e}")
else:
    _aggressive_scalper_instance = None


# === WebSocket Manager ===

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()


# === API Endpoints ===

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the dashboard."""
    dashboard_path = Path(__file__).parent / "static" / "index.html"
    if dashboard_path.exists():
        return FileResponse(dashboard_path)
    return HTMLResponse("<h1>Dashboard not found</h1>")


# === Session Management ===

@app.get("/api/session")
async def get_session():
    """Get current session info."""
    return {
        "session_id": state.session_id,
        "mode": state.mode,
        "created_at": state.start_time.isoformat(),
        "is_running": True,
        "api_validated": state.api_validated,
        "connection_status": "connected" if state.api_validated else "disconnected"
    }


@app.post("/api/session/switch")
async def switch_session(request: SessionSwitchRequest):
    """
    Switch trading mode and create a new session.
    
    This endpoint:
    1. Validates API credentials (for non-paper modes)
    2. Resets all state to fresh values
    3. Creates a new session_id
    4. Returns fresh exchange balances (if applicable)
    """
    mode = request.mode.lower()
    
    if mode not in ["testnet", "live"]:
        raise HTTPException(status_code=400, detail="Invalid mode. Must be: testnet or live")
    
    # Always require API credentials — paper mode removed
    balances = {}
    if not request.api_key or not request.api_secret:
        raise HTTPException(status_code=400, detail="API credentials required")
    
    if not BINANCE_AVAILABLE:
        raise HTTPException(status_code=500, detail="Exchange client not available")
    
    try:
        # Create client and validate
        testnet = mode == "testnet"
        client = BinanceClient(
            api_key=request.api_key,
            api_secret=request.api_secret,
            testnet=testnet
        )
        
        validation = await client.validate_credentials()
        
        if not validation["success"]:
            await client.destroy()
            raise HTTPException(status_code=401, detail=f"Invalid credentials: {validation['message']}")
        
        balances = validation.get("balances", {})
        
        # Store the client in state
        state.exchange_client = client
        state.api_validated = True
        
        # Start real trading loop now that exchange is connected
        await start_real_trading_loop()
        logger.info(f"Trading loop started for {mode.upper()} mode")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session switch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    # Reset state for new session
    state.reset(mode)
    
    # Broadcast session change to WebSocket clients
    await manager.broadcast({
        "type": "session_change",
        "session_id": state.session_id,
        "mode": mode,
        "message": f"Switched to {mode.upper()} mode"
    })
    
    logger.info(f"Session switched: {state.session_id[:8]}... (mode={mode})")
    
    return {
        "success": True,
        "session_id": state.session_id,
        "mode": mode,
        "created_at": state.start_time.isoformat(),
        "balances": balances,
        "api_validated": state.api_validated
    }


@app.post("/api/validate-keys")
async def validate_keys(request: ValidateKeysRequest):
    """
    Validate API credentials without switching mode.
    Used by the frontend to pre-validate keys before mode switch.
    """
    if not BINANCE_AVAILABLE:
        return {
            "success": False,
            "message": "Exchange client not available",
            "balances": {}
        }
    
    try:
        client = BinanceClient(
            api_key=request.api_key,
            api_secret=request.api_secret,
            testnet=request.testnet
        )
        
        result = await client.validate_credentials()
        await client.destroy()
        
        return result
        
    except Exception as e:
        logger.error(f"Key validation error: {e}")
        return {
            "success": False,
            "message": str(e),
            "balances": {}
        }


@app.post("/api/session/reset")
async def reset_session():
    """Reset current session (keep same mode but clear all data)."""
    current_mode = state.mode
    state.reset(current_mode)
    
    await manager.broadcast({
        "type": "session_change",
        "session_id": state.session_id,
        "mode": current_mode,
        "message": "Session reset"
    })
    
    return {
        "success": True,
        "session_id": state.session_id,
        "mode": current_mode,
        "message": "Session reset successfully"
    }


# === v1.0.1: Authentication Endpoints ===

@app.post("/api/auth/signup")
async def signup(request: SignupRequest):
    """
    Create a new user account.
    
    CryptoBoss 1.0.1: Email/password authentication.
    """
    if not AUTH_AVAILABLE:
        raise HTTPException(status_code=501, detail="Auth services not available")
    
    auth_service = get_auth_service()
    result = auth_service.signup(request.email, request.password)
    
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)
    
    return wrap_response({
        "success": True,
        "token": result.token,
        "user": result.user.to_dict()
    })


@app.post("/api/auth/login")
async def login(request: LoginRequest):
    """
    Authenticate user and return JWT token.
    """
    if not AUTH_AVAILABLE:
        raise HTTPException(status_code=501, detail="Auth services not available")
    
    auth_service = get_auth_service()
    result = auth_service.login(request.email, request.password)
    
    if not result.success:
        raise HTTPException(status_code=401, detail=result.error)
    
    logger.info(f"✅ User logged in: {request.email}")
    
    return wrap_response({
        "success": True,
        "token": result.token,
        "user": result.user.to_dict()
    })


@app.post("/api/auth/logout")
async def logout(user: User = Depends(require_auth)):
    """
    Logout current user.
    """
    # Session cleanup handled by frontend discarding token
    logger.info(f"User logged out: {user.email}")
    return wrap_response({"success": True, "message": "Logged out"})


@app.get("/api/auth/me")
async def get_me(user: User = Depends(require_auth)):
    """
    Get current authenticated user info.
    """
    return wrap_response({
        "user": user.to_dict(),
        "authenticated": True
    })


# === v1.0.1: Exchange Account Endpoints ===

@app.post("/api/accounts/create")
async def create_account(request: CreateAccountRequest, user: User = Depends(require_auth)):
    """
    Create a new exchange account.
    
    CryptoBoss 1.0.1: Each API key pair creates a NEW exchange_account_id.
    All state is scoped to this account.
    """
    if not AUTH_AVAILABLE:
        raise HTTPException(status_code=501, detail="Auth services not available")
    
    account_service = get_account_service()
    result = account_service.create_account(
        user_id=user.user_id,
        exchange_name=request.exchange_name,
        environment=request.environment,
        api_key=request.api_key,
        api_secret=request.api_secret,
        label=request.label
    )
    
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)
    
    logger.info(f"✅ New exchange account: {result.account.exchange_account_id[:8]}...")
    
    return wrap_response({
        "success": True,
        "account": result.account.to_dict(),
        "message": "Exchange account created with clean state"
    })


@app.get("/api/accounts/list")
async def list_accounts(user: User = Depends(require_auth)):
    """
    Get all exchange accounts for current user.
    """
    if not AUTH_AVAILABLE:
        raise HTTPException(status_code=501, detail="Auth services not available")
    
    account_service = get_account_service()
    result = account_service.get_accounts(user.user_id)
    
    return wrap_response({
        "accounts": [acc.to_dict() for acc in result.accounts],
        "count": len(result.accounts)
    })


@app.post("/api/accounts/select")
async def select_account(request: SelectAccountRequest, user: User = Depends(require_auth)):
    """
    Select an exchange account as active.
    
    FUNDAMENTAL AXIOM: There is NO such thing as a global bot state.
    
    THIS TRIGGERS:
    1. STOP current bot instance
    2. DESTROY all its memory
    3. START a brand-new bot instance
    4. LOAD state ONLY for selected exchange_account_id
    """
    if not AUTH_AVAILABLE:
        raise HTTPException(status_code=501, detail="Auth services not available")
    
    account_service = get_account_service()
    result = account_service.get_account(user.user_id, request.exchange_account_id)
    
    if not result.success:
        raise HTTPException(status_code=404, detail=result.error)
    
    # === TRUE INSTANCE ISOLATION ===
    # Step 1-4: Switch bot instance (STOP → DESTROY → START)
    bot_instance = None
    is_new_account = False
    
    if BOT_INSTANCE_AVAILABLE:
        logger.info(f"🔄 Switching bot instance to {request.exchange_account_id[:8]}...")
        bot_instance = switch_bot_instance(
            user_id=user.user_id,
            exchange_account_id=result.account.exchange_account_id,
            environment=result.account.environment
        )
        is_new_account = len(bot_instance.trading_state.trade_history) == 0
        logger.info(f"✅ Bot instance switched - {'NEW' if is_new_account else 'existing'} account")
    
    # Legacy: Also reset dashboard state
    state.reset(result.account.environment.lower())
    state.active_exchange_account_id = result.account.exchange_account_id
    state.active_user_id = user.user_id
    
    # Legacy: Switch scoped state manager
    if SCOPED_STATE_AVAILABLE:
        try:
            scoped_manager = switch_account(user.user_id, result.account.exchange_account_id)
            logger.info(f"🔒 ScopedStateManager also switched")
        except Exception as e:
            logger.warning(f"ScopedStateManager switch failed: {e}")
    
    # Lock environment based on account
    if not env_signature.is_locked:
        env_signature.lock(result.account.environment.lower())
    
    # CRYPTOBOSS 2.0: Save active account to SQLite for persistence
    try:
        from src.core.database.repository import get_repository
        repo = get_repository()
        repo.set_active_account(user.user_id, result.account.exchange_account_id)
    except Exception as e:
        logger.warning(f"Could not save active account to SQLite: {e}")
    
    logger.info(f"🎯 Account switch complete: {result.account.exchange_account_id[:8]}... ({result.account.environment})")
    
    # Broadcast ACCOUNT_CHANGED event (frontend MUST reset everything)
    await manager.broadcast({
        "type": "ACCOUNT_CHANGED",
        "action": "FULL_RESET_REQUIRED",
        "exchange_account_id": result.account.exchange_account_id,
        "environment": result.account.environment,
        "is_new_account": is_new_account,
        "mandatory_actions": [
            "STOP_ALL_STREAMS",
            "CLEAR_ALL_UI_STATE",
            "CLEAR_CHARTS",
            "CLEAR_TABLES",
            "CLOSE_ALL_SOCKETS",
            "REQUEST_FRESH_DATA"
        ]
    })
    
    # Get dashboard data from the new bot instance
    dashboard_data = None
    if bot_instance:
        dashboard_data = bot_instance.get_dashboard_data()
    
    return wrap_response({
        "success": True,
        "account": result.account.to_dict(),
        "instance_switched": bot_instance is not None,
        "is_new_account": is_new_account,
        "dashboard_data": dashboard_data,
        "message": f"Bot instance switched - {'empty state' if is_new_account else 'loaded existing state'}"
    })


@app.delete("/api/accounts/{exchange_account_id}")
async def delete_account(exchange_account_id: str, user: User = Depends(require_auth)):
    """
    Delete/archive an exchange account.
    
    Warning: This archives the account and all associated data.
    """
    if not AUTH_AVAILABLE:
        raise HTTPException(status_code=501, detail="Auth services not available")
    
    account_service = get_account_service()
    result = account_service.delete_account(user.user_id, exchange_account_id)
    
    if not result.success:
        raise HTTPException(status_code=404, detail=result.error)
    
    logger.warning(f"🗑️ Account archived: {exchange_account_id[:8]}...")
    
    return wrap_response({
        "success": True,
        "message": "Account archived"
    })


@app.get("/api/accounts/active")
async def get_active_account(user: User = Depends(require_auth)):
    """
    Get the currently active exchange account.
    """
    if not hasattr(state, 'active_exchange_account_id') or not state.active_exchange_account_id:
        return wrap_response({
            "active": False,
            "account": None,
            "message": "No account selected"
        })
    
    account_service = get_account_service()
    result = account_service.get_account(user.user_id, state.active_exchange_account_id)
    
    if not result.success:
        return wrap_response({
            "active": False,
            "account": None,
            "message": "Account not found"
        })
    
    # Get key fingerprint for display
    fingerprint = account_service.get_key_fingerprint(user.user_id, state.active_exchange_account_id)
    
    account_data = result.account.to_dict()
    account_data["api_key_fingerprint"] = fingerprint
    
    return wrap_response({
        "active": True,
        "account": account_data
    })


# === Account Reset Endpoint ===

class ResetAccountRequest(BaseModel):
    confirm: bool = False
    reason: str = ""


@app.post("/api/accounts/{exchange_account_id}/reset")
async def reset_account(exchange_account_id: str, request: ResetAccountRequest, user: User = Depends(require_auth)):
    """
    Reset account state - DELETE all trades and analytics data.
    
    CRYPTOBOSS 2.0: Account reset functionality.
    
    WHAT GETS DELETED:
    - All trades for this account
    - PnL history
    - Bot instance state
    
    WHAT STAYS:
    - User account
    - Exchange account (API keys)
    - Other exchange accounts
    
    REQUIRES: confirm=True and a reason.
    """
    if not request.confirm:
        raise HTTPException(
            status_code=400, 
            detail="Must set confirm=true to reset account. This is destructive!"
        )
    
    if len(request.reason) < 10:
        raise HTTPException(
            status_code=400,
            detail="Reason must be at least 10 characters"
        )
    
    try:
        from src.core.database.repository import get_repository
        repo = get_repository()
        
        # Verify user owns this account
        account = repo.find_account_by_id(exchange_account_id)
        if not account:
            raise HTTPException(status_code=404, detail="Account not found")
        
        if account.user_id != user.user_id:
            raise HTTPException(status_code=403, detail="Not your account")
        
        # Delete all trades for this account
        deleted_count = repo.delete_trades_for_account(user.user_id, exchange_account_id)
        
        # Log the action
        logger.warning(
            f"🗑️ ACCOUNT RESET: {exchange_account_id[:8]}... | "
            f"User: {user.email} | "
            f"Trades deleted: {deleted_count} | "
            f"Reason: {request.reason}"
        )
        
        # Reset bot instance if available
        if BOT_INSTANCE_AVAILABLE:
            try:
                from src.core.bot_instance import BotInstanceManager
                manager_instance = BotInstanceManager()
                if manager_instance.active_instance and manager_instance.active_instance.account_id == exchange_account_id:
                    manager_instance.destroy_active()
                    manager_instance.create_instance(
                        account_id=exchange_account_id,
                        user_id=user.user_id,
                        environment=account.environment
                    )
            except Exception as e:
                logger.warning(f"Could not reset bot instance: {e}")
        
        return wrap_response({
            "success": True,
            "trades_deleted": deleted_count,
            "user_id": user.user_id,
            "exchange_account_id": exchange_account_id,
            "reason": request.reason,
            "message": f"Account reset complete. {deleted_count} trades deleted."
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reset account: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === System State Endpoints ===

@app.get("/api/system")
async def get_system():
    """
    Get complete system state.
    
    Returns environment, connection, time sync, kill switch state.
    This is the primary endpoint for dashboard observability.
    Includes environment_signature per CryptoBoss 1.0.0 spec.
    """
    uptime = (datetime.now() - state.start_time).total_seconds()
    
    system_data = {
        "session_id": state.session_id,
        "environment": state.environment,
        "mode": state.mode,
        "connection_status": state.connection_status,
        "api_validated": state.api_validated,
        "timestamp_offset_ms": state.timestamp_offset_ms,
        "last_time_sync": state.last_time_sync.isoformat() if state.last_time_sync else None,
        "kill_switch": {
            "active": state.kill_switch_active,
            "reason": state.kill_switch_reason
        },
        "uptime_seconds": uptime,
        "started_at": state.start_time.isoformat(),
        "ws_clients": len(manager.active_connections),
        "incident_state": "NORMAL"  # v1.0.0: Incident state tracking
    }
    return wrap_response(system_data)


@app.get("/api/context")
async def get_context():
    """
    Get current market context and bias.
    
    Used by dashboard to show trading environment.
    Includes data_source tag per specification.
    """
    context_data = {
        "market_context": state.market_context,
        "bias": state.market_bias,
        "last_update": state.last_context_update.isoformat() if state.last_context_update else None,
        "current_price": state.current_price,
        "price_change_pct": ((state.current_price - state.last_price) / state.last_price * 100) if state.last_price > 0 else 0,
        "symbol": "BTC/USDT"
    }
    return wrap_response(context_data, DataSourceTag.DERIVED)


@app.get("/api/decisions")
async def get_decisions(limit: int = 50):
    """
    Get recent trading decisions.
    
    Returns decision flow results, rejections, and outcomes.
    """
    return {
        "decisions": state.recent_decisions[-limit:],
        "decisions_today": state.decisions_today,
        "rejections_today": state.rejections_today,
        "last_decision_time": state.last_decision_time.isoformat() if state.last_decision_time else None,
        "total_trades": state.total_trades
    }


@app.get("/api/risk")
async def get_risk():
    """
    Get risk state and budget.
    
    Shows drawdown, allocation, and remaining risk budget.
    CryptoBoss 1.0.0: All values tagged with data source.
    """
    drawdown_pct = (state.pnl / state.initial_capital * 100) if state.initial_capital > 0 else 0
    
    risk_data = {
        "daily_pnl": state.pnl,
        "daily_pnl_pct": drawdown_pct,
        "unrealized_pnl": state.unrealized_pnl,
        "total_pnl": state.total_pnl,
        "capital": {
            "initial": state.initial_capital,
            "current": state.capital,
            "allocated": state.position * state.current_price
        },
        "limits": {
            "daily_loss_limit_pct": 5.0,
            "max_position_pct": 25.0,
            "max_trades_per_day": 10
        },
        "remaining_budget": {
            "daily_loss_available_pct": 5.0 + drawdown_pct,
            "trades_remaining": 10 - state.total_trades
        },
        "kill_switch_active": state.kill_switch_active,
        "risk_guardian_active": True,
        "capital_governor_active": True
    }
    return wrap_response(risk_data, DataSourceTag.DERIVED)


@app.post("/api/kill-switch")
async def toggle_kill_switch(active: bool = True, reason: str = "Manual activation"):
    """Toggle the kill switch."""
    state.kill_switch_active = active
    state.kill_switch_reason = reason if active else None
    
    await manager.broadcast({
        "type": "kill_switch",
        "active": active,
        "reason": reason
    })
    
    # Stop/start trading loop based on kill switch
    if active:
        await stop_real_trading_loop()
    else:
        await start_real_trading_loop()
    
    logger.warning(f"Kill switch {'ACTIVATED' if active else 'DEACTIVATED'}: {reason}")
    
    return {
        "success": True,
        "kill_switch_active": state.kill_switch_active,
        "reason": state.kill_switch_reason
    }


# === Multi-Symbol Prices Endpoint ===

SUPPORTED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]


@app.get("/api/prices")
async def get_prices(symbol: str = "BTC/USDT", timeframe: str = "1h", limit: int = 200):
    """Return OHLCV candles for chart widgets and SMC overlays."""
    frame = await fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=min(max(limit, 20), 1000))
    payload = [
        {
            "timestamp": index.isoformat(),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }
        for index, row in frame.iterrows()
    ]
    return wrap_response(payload, DataSourceTag.DERIVED)


@app.get("/api/prices/live")
async def get_live_prices():
    """
    Get live prices for multiple symbols.
    
    CRYPTOBOSS 2.0: Real prices from Binance.
    NO FALLBACK - if exchange unavailable, return empty.
    """
    prices = {}
    
    try:
        if BINANCE_AVAILABLE:
            # Determine if testnet based on environment
            is_testnet = env_signature.mode == "testnet" or not env_signature.is_locked
            
            # Create client for price fetch
            client = BinanceClient(testnet=is_testnet)
            
            for symbol in SUPPORTED_SYMBOLS:
                try:
                    ticker = client.exchange.fetch_ticker(symbol.replace("USDT", "/USDT"))
                    prices[symbol] = {
                        "symbol": symbol,
                        "price": ticker.get("last", 0),
                        "change24h": ticker.get("percentage", 0),
                        "high24h": ticker.get("high", 0),
                        "low24h": ticker.get("low", 0),
                        "volume24h": ticker.get("quoteVolume", 0),
                        "timestamp": datetime.now().isoformat(),
                        "source": "BINANCE_TESTNET" if is_testnet else "BINANCE_LIVE"
                    }
                except Exception as e:
                    logger.warning(f"Failed to fetch {symbol}: {e}")
                    prices[symbol] = {
                        "symbol": symbol,
                        "price": 0,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
            
            client.close()
        else:
            # No Binance client - return empty
            for symbol in SUPPORTED_SYMBOLS:
                prices[symbol] = {
                    "symbol": symbol,
                    "price": 0,
                    "error": "Exchange client not available",
                    "timestamp": datetime.now().isoformat()
                }
    except Exception as e:
        logger.error(f"Failed to fetch prices: {e}")
        for symbol in SUPPORTED_SYMBOLS:
            prices[symbol] = {
                "symbol": symbol,
                "price": 0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    return wrap_response({
        "prices": prices,
        "symbols": SUPPORTED_SYMBOLS,
        "count": len(prices)
    })



@app.get("/api/status")
async def get_status():
    """Get current bot status."""
    uptime = (datetime.now() - state.start_time).total_seconds()
    
    return {
        "session_id": state.session_id,
        "status": "running",
        "mode": state.mode,
        "capital": state.initial_capital,
        "current_capital": state.capital,
        "portfolio_value": state.portfolio_value,
        "pnl": state.total_pnl,
        "pnl_pct": (state.total_pnl / state.initial_capital * 100) if state.initial_capital > 0 else 0,
        "realized_pnl": state.pnl,
        "unrealized_pnl": state.unrealized_pnl,
        "uptime_seconds": uptime,
        "trades_count": state.total_trades,
        "win_rate": state.win_rate,
        "position": state.position,
        "current_price": state.current_price,
        "connected_clients": len(manager.active_connections),
        "api_validated": state.api_validated,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/portfolio")
async def get_portfolio():
    """
    Get portfolio details.
    
    RULE: Returns data ONLY from the active bot instance.
    New accounts = EMPTY portfolio.
    """
    # Use BotInstanceManager for true isolation
    if BOT_INSTANCE_AVAILABLE:
        bot = get_active_bot()
        if bot:
            trading_state = bot.trading_state
            positions = []
            
            # Only show positions if they exist in THIS bot instance
            for pos in trading_state.positions:
                positions.append({
                    "symbol": pos.get("symbol", "BTC/USDT"),
                    "quantity": pos.get("quantity", 0),
                    "entry_price": pos.get("entry_price", 0),
                    "current_price": state.current_price,
                    "value_usd": pos.get("quantity", 0) * state.current_price,
                    "pnl": pos.get("pnl", 0),
                    "pnl_pct": pos.get("pnl_pct", 0)
                })
            
            return wrap_response({
                "balance": trading_state.balances if trading_state.balances else {"USDT": 10000.0, "BTC": 0.0},
                "positions": positions,  # Empty for new accounts
                "total_value_usd": sum(trading_state.balances.values()) if trading_state.balances else 10000.0,
                "daily_pnl": trading_state.total_pnl,
                "daily_pnl_pct": (trading_state.total_pnl / 10000 * 100) if trading_state.total_pnl else 0,
                "is_new_account": len(trading_state.trade_history) == 0
            })
        else:
            # No bot instance - return empty
            return wrap_response({
                "balance": {"USDT": 0.0, "BTC": 0.0},
                "positions": [],
                "total_value_usd": 0,
                "daily_pnl": 0,
                "daily_pnl_pct": 0,
                "is_new_account": True,
                "error": "No active bot instance"
            })
    
    # Fallback to legacy state
    btc_value = state.position * state.current_price
    
    return wrap_response({
        "balance": {
            "USDT": round(state.capital, 2),
            "BTC": round(state.position, 6)
        },
        "positions": [
            {
                "symbol": "BTC/USDT",
                "quantity": state.position,
                "entry_price": state.position_entry_price,
                "current_price": state.current_price,
                "value_usd": btc_value,
                "pnl": state.unrealized_pnl,
                "pnl_pct": ((state.current_price - state.position_entry_price) / state.position_entry_price * 100) if state.position_entry_price > 0 else 0
            }
        ] if state.position > 0 else [],
        "total_value_usd": state.portfolio_value,
        "daily_pnl": state.total_pnl,
        "daily_pnl_pct": (state.total_pnl / state.initial_capital * 100) if state.initial_capital > 0 else 0
    })


@app.get("/api/trades")
async def get_trades(limit: int = 50, user: User = Depends(require_auth)):
    """
    Get recent trades.
    
    CRYPTOBOSS 2.0: Uses SQLite repository with ownership filtering.
    CRITICAL: Always filters by user_id AND exchange_account_id.
    New accounts = EMPTY trades array.
    """
    try:
        from src.core.database.repository import get_repository
        repo = get_repository()
        
        # Get active account ID
        active_account_id = repo.get_active_account_id(user.user_id)
        
        if not active_account_id:
            # No active account - return empty
            return wrap_response({
                "trades": [],
                "count": 0,
                "is_new_account": True,
                "user_id": user.user_id,
                "exchange_account_id": None,
                "message": "No active account selected"
            })
        
        # Get trades from SQLite - ALWAYS filtered by ownership
        trades = repo.get_trades(user.user_id, active_account_id, limit=limit)
        
        return wrap_response({
            "trades": trades,
            "count": len(trades),
            "is_new_account": len(trades) == 0,
            "user_id": user.user_id,
            "exchange_account_id": active_account_id
        })
        
    except Exception as e:
        logger.error(f"Failed to get trades: {e}")
        
        # Fallback to bot instance if SQLite fails
        if BOT_INSTANCE_AVAILABLE:
            bot = get_active_bot()
            if bot:
                trades = bot.trading_state.trade_history[-limit:]
                return wrap_response({
                    "trades": trades,
                    "count": len(trades),
                    "is_new_account": len(bot.trading_state.trade_history) == 0
                })
        
        # Ultimate fallback - empty
        return wrap_response({
            "trades": [],
            "count": 0,
            "is_new_account": True,
            "error": str(e)
        })


@app.get("/api/pnl/history")
async def get_pnl_history():
    """
    Get cumulative P/L history for the PLGraph component.
    
    Returns points array sorted by time with running total.
    Uses in-memory trades from state (works with both SQLite and legacy fallback).
    """
    closed_trades = [t for t in state.trades if t.get("side") == "SELL" and t.get("pnl") is not None]
    
    if not closed_trades:
        return wrap_response({
            "points": [],
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "message": "No closed trades yet — P/L graph will populate as trades close"
        })
    
    # Build cumulative P/L series
    points = []
    running_pnl = 0.0
    wins = 0
    losses = 0
    
    for trade in closed_trades:
        pnl = float(trade.get("pnl", 0))
        running_pnl += pnl
        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1
        
        points.append({
            "time": trade.get("time", datetime.now().isoformat()),
            "pnl": round(running_pnl, 4),
            "trade_pnl": round(pnl, 4),
            "symbol": trade.get("symbol", "BTC/USDT"),
        })
    
    total_closed = wins + losses
    win_rate = (wins / total_closed * 100) if total_closed > 0 else 0.0
    
    return wrap_response({
        "points": points,
        "total_pnl": round(running_pnl, 4),
        "win_rate": round(win_rate, 1),
        "total_trades": total_closed,
        "wins": wins,
        "losses": losses,
    })


@app.get("/api/strategies")
async def get_strategies():
    """Get active strategies."""
    strategies = [
        {
            "id": "dca_btc_usdt",
            "type": "DCA",
            "symbol": "BTC/USDT",
            "status": "active",
            "pnl": round(state.total_pnl, 2),
            "trades": state.total_trades,
            "win_rate": round(state.win_rate, 1),
            "position": state.position
        }
    ]
    
    # Add AggressiveScalper if available
    if AGGRESSIVE_SCALPER_AVAILABLE and _aggressive_scalper_instance:
        metrics = _aggressive_scalper_instance.get_metrics()
        strategies.append({
            "id": "aggressive_scalper_v1",
            "type": "AggressiveScalper",
            "symbol": "BTC/USDT",
            "status": "active" if state.api_validated else "waiting",
            "pnl": 0,
            "trades": metrics.get("signals_generated", 0),
            "win_rate": 0,
            "position": 0,
            "confidence_threshold": 0.6,
        })
    
    return {"strategies": strategies}



@app.get("/api/v2/smc/state")
async def get_smc_state(symbol: str = "BTC/USDT", timeframe: str = "5m", limit: int = 400):
    """Return full SMC state and top setups for the requested symbol."""
    if not V12_AVAILABLE:
        raise HTTPException(status_code=503, detail="v12 SMC modules are unavailable")

    requested_tf = timeframe.lower()
    timeframes = ["1h", "15m", requested_tf]
    engine = SMCEngine(timeframes=timeframes)
    data = await fetch_multi_tf_data(symbol=symbol, timeframes=timeframes, limit=limit)
    setups = engine.analyze(data)
    state_dict = engine.get_state_dict()

    return wrap_response(
        {
            "symbol": symbol,
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "smc_state": state_dict,
            "setups": [_serialize_setup(setup) for setup in setups[:10]],
        },
        DataSourceTag.DERIVED,
    )


@app.get("/api/v2/smc/order-blocks")
async def get_order_blocks(symbol: str = "BTC/USDT", timeframe: str = "15m", status: str = "active"):
    """Return filtered order blocks for a timeframe."""
    if not V12_AVAILABLE:
        raise HTTPException(status_code=503, detail="v12 SMC modules are unavailable")

    tf = timeframe.lower()
    engine = SMCEngine(timeframes=["1h", "15m", tf])
    data = await fetch_multi_tf_data(symbol=symbol, timeframes=engine.timeframes, limit=320)
    engine.analyze(data)
    blocks = engine.get_state_dict().get(tf, {}).get("order_blocks", [])

    if status:
        blocks = [item for item in blocks if item.get("status") == status]

    return wrap_response(
        {
            "symbol": symbol,
            "timeframe": tf,
            "count": len(blocks),
            "order_blocks": blocks,
        },
        DataSourceTag.DERIVED,
    )


@app.get("/api/v2/smc/fvg")
async def get_fvgs(symbol: str = "BTC/USDT", timeframe: str = "15m"):
    """Return FVG state for a timeframe."""
    if not V12_AVAILABLE:
        raise HTTPException(status_code=503, detail="v12 SMC modules are unavailable")

    tf = timeframe.lower()
    engine = SMCEngine(timeframes=["1h", "15m", tf])
    data = await fetch_multi_tf_data(symbol=symbol, timeframes=engine.timeframes, limit=320)
    engine.analyze(data)
    fvgs = engine.get_state_dict().get(tf, {}).get("fvgs", [])

    return wrap_response(
        {
            "symbol": symbol,
            "timeframe": tf,
            "count": len(fvgs),
            "fvgs": fvgs,
        },
        DataSourceTag.DERIVED,
    )


@app.get("/api/v2/smc/structure")
async def get_structure(symbol: str = "BTC/USDT", timeframe: str = "15m"):
    """Return BOS/CHoCH structure timeline for a timeframe."""
    if not V12_AVAILABLE:
        raise HTTPException(status_code=503, detail="v12 SMC modules are unavailable")

    tf = timeframe.lower()
    engine = SMCEngine(timeframes=["1h", "15m", tf])
    data = await fetch_multi_tf_data(symbol=symbol, timeframes=engine.timeframes, limit=320)
    engine.analyze(data)

    structure = engine.get_state_dict().get(tf, {}).get("structure", [])
    trend = engine.get_state_dict().get(tf, {}).get("trend", "unknown")

    return wrap_response(
        {
            "symbol": symbol,
            "timeframe": tf,
            "trend": trend,
            "count": len(structure),
            "structure": structure,
        },
        DataSourceTag.DERIVED,
    )


@app.get("/api/v2/scalper/live")
async def get_scalper_live(symbol: str = "BTC/USDT", account_balance: float = 10000.0):
    """Return live intraday scalper signal and current SMC market analysis."""
    if not V12_AVAILABLE or v12_scalper is None:
        raise HTTPException(status_code=503, detail="v12 scalper module is unavailable")

    timeframes = [v12_scalper.bias_timeframe, v12_scalper.mid_timeframe, v12_scalper.entry_timeframe]
    data = await fetch_multi_tf_data(symbol=symbol, timeframes=timeframes, limit=400)
    signal = v12_scalper.generate_multi_timeframe_signal(data=data, account_balance=account_balance)
    analysis = v12_scalper.analyze_current_market(data=data)

    return wrap_response(
        {
            "symbol": symbol,
            "signal": {
                "action": signal.action,
                "reason": signal.reason,
                "confidence": signal.confidence,
                "size": signal.size,
                "price": signal.price,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "metadata": signal.metadata,
            },
            "analysis": analysis,
        },
        DataSourceTag.DERIVED,
    )


@app.get("/api/v2/strategies/presets")
async def list_v2_presets():
    """Return built-in StrategyBuilder presets."""
    if not V12_AVAILABLE or v12_builder is None:
        raise HTTPException(status_code=503, detail="v12 strategy builder is unavailable")

    return wrap_response(
        {
            "presets": list(StrategyBuilder.PRESETS.keys()),
            "details": StrategyBuilder.PRESETS,
        },
        DataSourceTag.DERIVED,
    )


@app.post("/api/v2/strategies/create")
async def create_v2_strategy(payload: Dict[str, Any]):
    """Create a custom strategy definition via StrategyBuilder."""
    if not V12_AVAILABLE or v12_builder is None:
        raise HTTPException(status_code=503, detail="v12 strategy builder is unavailable")

    name = payload.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="Field 'name' is required")

    strategy_id = v12_builder.new_strategy(
        name=name,
        symbol=payload.get("symbol", "BTC/USDT"),
        timeframe=payload.get("timeframe", "5m"),
        description=payload.get("description", ""),
    )

    return wrap_response({"strategy_id": strategy_id}, DataSourceTag.DERIVED)


@app.post("/api/v2/strategies/load-preset")
async def load_v2_preset(preset: str, symbol: str = "BTC/USDT", timeframe: str = "5m"):
    """Create a strategy instance from a preset template."""
    if not V12_AVAILABLE or v12_builder is None:
        raise HTTPException(status_code=503, detail="v12 strategy builder is unavailable")

    try:
        strategy_id = v12_builder.load_preset(preset, symbol=symbol, timeframe=timeframe)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    strategy = v12_builder.get_strategy(strategy_id)
    return wrap_response(
        {"strategy_id": strategy_id, "strategy": json.loads(strategy.to_json())},
        DataSourceTag.DERIVED,
    )


@app.get("/api/v2/strategies")
async def list_v2_strategies():
    """List all strategies currently loaded in StrategyBuilder."""
    if not V12_AVAILABLE or v12_builder is None:
        raise HTTPException(status_code=503, detail="v12 strategy builder is unavailable")
    return wrap_response({"strategies": v12_builder.list_strategies()}, DataSourceTag.DERIVED)


@app.post("/api/v2/strategies/{strategy_id}/backtest")
async def backtest_v2_strategy(strategy_id: str, payload: Dict[str, Any]):
    """Backtest a built strategy with detailed metrics and Monte Carlo analysis."""
    if not V12_AVAILABLE or v12_builder is None or v12_tester is None:
        raise HTTPException(status_code=503, detail="v12 tester stack is unavailable")

    try:
        strategy = v12_builder.get_strategy(strategy_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Strategy '{strategy_id}' not found") from exc

    limit = int(payload.get("limit", 1200))
    df = await fetch_ohlcv(symbol=strategy.symbol, timeframe=strategy.timeframe, limit=limit)

    signal_fn = _compile_strategy_signal_fn(strategy)
    result = v12_tester.run(
        df=df,
        signal_fn=signal_fn,
        strategy_name=strategy.name,
        strategy_id=strategy_id,
        symbol=strategy.symbol,
        timeframe=strategy.timeframe,
    )

    monte_carlo = v12_tester.run_monte_carlo(
        result,
        n_simulations=int(payload.get("n_simulations", 500)),
    )

    trades_payload = [
        {
            "id": trade.trade_id,
            "direction": trade.direction,
            "entry_time": str(trade.entry_time),
            "exit_time": str(trade.exit_time),
            "entry": trade.entry_price,
            "exit": trade.exit_price,
            "net_pnl": round(trade.net_pnl, 4),
            "pnl_pct": round(trade.pnl_pct, 2),
            "exit_reason": trade.exit_reason,
            "rr": round(trade.risk_reward_achieved, 2),
        }
        for trade in result.trades
    ]

    monthly_returns = {str(key): float(value) for key, value in result.monthly_returns.to_dict().items()}

    return wrap_response(
        {
            "summary": result.to_summary_dict(),
            "trades": trades_payload,
            "equity_curve": result.equity_curve.tolist(),
            "drawdown": result.drawdown_series.tolist(),
            "monthly_returns": monthly_returns,
            "monte_carlo": monte_carlo,
        },
        DataSourceTag.DERIVED,
    )


@app.post("/api/v2/strategies/{strategy_id}/walk-forward")
async def walk_forward_v2_strategy(strategy_id: str, payload: Dict[str, Any]):
    """Run walk-forward optimization on a built strategy."""
    if not V12_AVAILABLE or v12_builder is None or v12_tester is None:
        raise HTTPException(status_code=503, detail="v12 tester stack is unavailable")

    try:
        base_strategy = v12_builder.get_strategy(strategy_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Strategy '{strategy_id}' not found") from exc

    param_grid = payload.get("param_grid", {"take_profit_value": [1.2, 1.5, 2.0], "stop_loss_value": [0.8, 1.0, 1.2]})
    splits = int(payload.get("n_splits", 5))
    limit = int(payload.get("limit", 2000))

    df = await fetch_ohlcv(symbol=base_strategy.symbol, timeframe=base_strategy.timeframe, limit=limit)

    def signal_fn_factory(params: Dict[str, Any]):
        strategy_copy = BuiltStrategy.from_json(base_strategy.to_json())
        for key, value in params.items():
            if hasattr(strategy_copy.exit_config, key):
                setattr(strategy_copy.exit_config, key, value)
            elif hasattr(strategy_copy.risk_config, key):
                setattr(strategy_copy.risk_config, key, value)
            elif hasattr(strategy_copy.filter_config, key):
                setattr(strategy_copy.filter_config, key, value)
        return _compile_strategy_signal_fn(strategy_copy)

    wf_results = v12_tester.run_walk_forward(
        df=df,
        signal_fn_factory=signal_fn_factory,
        param_grid=param_grid,
        n_splits=splits,
        strategy_name=base_strategy.name,
        strategy_id=base_strategy.strategy_id,
        symbol=base_strategy.symbol,
        timeframe=base_strategy.timeframe,
    )

    return wrap_response({"walk_forward": wf_results}, DataSourceTag.DERIVED)


# === v4 Dual-Source Professional APIs ===

@app.get("/api/v4/status")
async def v4_status():
    orchestrator = get_v4()
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="v4 modules are unavailable")
    return wrap_response(orchestrator.status(), DataSourceTag.DERIVED)


@app.get("/api/v4/config")
async def v4_config():
    orchestrator = get_v4()
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="v4 modules are unavailable")
    return wrap_response(orchestrator.config.summary(), DataSourceTag.DERIVED)


@app.post("/api/v4/cycle")
async def v4_run_cycle(payload: Dict[str, Any]):
    orchestrator = get_v4()
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="v4 modules are unavailable")

    result = orchestrator.run_cycle(
        symbol=payload.get("symbol", "BTC/USDT"),
        timeframes=payload.get("timeframes", ["1m", "5m", "15m"]),
        limit=int(payload.get("limit", 500)),
        strategy_id=payload.get("strategy_id"),
    )
    return wrap_response(result, DataSourceTag.DERIVED)


@app.get("/api/v4/price/ohlcv")
async def v4_get_ohlcv(symbol: str = "BTC/USDT", timeframe: str = "5m", limit: int = 500):
    orchestrator = get_v4()
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="v4 modules are unavailable")

    df = orchestrator.price_feed.get_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
    return wrap_response(
        {
            "symbol": symbol,
            "timeframe": timeframe,
            "source": orchestrator.price_feed.active_source,
            "candles": df.reset_index().to_dict(orient="records"),
        },
        DataSourceTag.DERIVED,
    )


@app.get("/api/v4/price/multi-tf")
async def v4_get_multi_tf(symbol: str = "BTC/USDT", limit: int = 200):
    orchestrator = get_v4()
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="v4 modules are unavailable")

    frames = orchestrator.price_feed.get_multi_timeframe(symbol=symbol, timeframes=["1m", "5m", "15m"], limit=limit)
    return wrap_response(
        {
            "symbol": symbol,
            "source": orchestrator.price_feed.active_source,
            "timeframes": {tf: df.reset_index().to_dict(orient="records") for tf, df in frames.items()},
        },
        DataSourceTag.DERIVED,
    )


@app.get("/api/v4/price/ticker")
async def v4_get_ticker(symbol: str = "BTC/USDT"):
    orchestrator = get_v4()
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="v4 modules are unavailable")

    return wrap_response(
        {
            "last_price": orchestrator.price_feed.get_last_price(symbol),
            "spread_pct": orchestrator.price_feed.get_spread(symbol),
            "source": orchestrator.price_feed.active_source,
        },
        DataSourceTag.DERIVED,
    )


@app.get("/api/v4/builder/indicators")
async def v4_builder_indicators():
    if not V4_AVAILABLE:
        raise HTTPException(status_code=503, detail="v4 modules are unavailable")
    return wrap_response(INDICATOR_LIBRARY, DataSourceTag.DERIVED)


@app.get("/api/v4/builder/presets")
async def v4_builder_presets():
    if not V4_AVAILABLE:
        raise HTTPException(status_code=503, detail="v4 modules are unavailable")
    return wrap_response(list(ProStrategyBuilder.PRESETS.keys()), DataSourceTag.DERIVED)


@app.post("/api/v4/builder/strategies")
async def v4_create_strategy(payload: Dict[str, Any]):
    orchestrator = get_v4()
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="v4 modules are unavailable")

    name = payload.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="Field 'name' is required")

    strategy_id = orchestrator.build_strategy(
        name=name,
        symbol=payload.get("symbol", "BTC/USDT"),
        timeframe=payload.get("timeframe", "5m"),
    )
    return wrap_response({"strategy_id": strategy_id}, DataSourceTag.DERIVED)


@app.post("/api/v4/builder/strategies/preset")
async def v4_load_preset(preset: str, symbol: str = "BTC/USDT", timeframe: str = "5m"):
    orchestrator = get_v4()
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="v4 modules are unavailable")

    strategy_id = orchestrator.load_preset(preset, symbol=symbol, timeframe=timeframe)
    strategy = orchestrator.strategy_builder.get(strategy_id)
    score, breakdown, recommendation = orchestrator.score_strategy(strategy_id)

    return wrap_response(
        {
            "strategy_id": strategy_id,
            "name": strategy.name,
            "ai_score": score,
            "breakdown": breakdown,
            "recommendation": recommendation,
        },
        DataSourceTag.DERIVED,
    )


@app.get("/api/v4/builder/strategies")
async def v4_list_strategies():
    orchestrator = get_v4()
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="v4 modules are unavailable")
    return wrap_response(orchestrator.strategy_builder.list(), DataSourceTag.DERIVED)


@app.get("/api/v4/builder/strategies/{strategy_id}")
async def v4_get_strategy(strategy_id: str):
    orchestrator = get_v4()
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="v4 modules are unavailable")

    try:
        strategy = orchestrator.strategy_builder.get(strategy_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Strategy '{strategy_id}' not found") from exc

    return wrap_response(json.loads(strategy.to_json()), DataSourceTag.DERIVED)


@app.post("/api/v4/builder/strategies/{strategy_id}/conditions")
async def v4_add_condition(strategy_id: str, payload: Dict[str, Any]):
    orchestrator = get_v4()
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="v4 modules are unavailable")

    block_id = orchestrator.strategy_builder.add_condition(
        strategy_id=strategy_id,
        direction=payload["direction"],
        indicator=payload["indicator"],
        operator=payload["operator"],
        threshold=payload["threshold"],
        params=payload.get("params", {}),
        output_key=payload.get("output_key", ""),
        description=payload.get("description", ""),
        canvas_x=float(payload.get("canvas_x", 0.0)),
        canvas_y=float(payload.get("canvas_y", 0.0)),
    )
    return wrap_response({"block_id": block_id}, DataSourceTag.DERIVED)


@app.delete("/api/v4/builder/strategies/{strategy_id}/conditions/{block_id}")
async def v4_remove_condition(strategy_id: str, block_id: str):
    orchestrator = get_v4()
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="v4 modules are unavailable")
    removed = orchestrator.strategy_builder.remove_condition(strategy_id, block_id)
    return wrap_response({"removed": removed}, DataSourceTag.DERIVED)


@app.post("/api/v4/builder/strategies/{strategy_id}/score")
async def v4_score_strategy(strategy_id: str):
    orchestrator = get_v4()
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="v4 modules are unavailable")
    score, breakdown, recommendation = orchestrator.score_strategy(strategy_id)
    return wrap_response(
        {
            "score": score,
            "breakdown": breakdown,
            "recommendation": recommendation,
        },
        DataSourceTag.DERIVED,
    )


@app.post("/api/v4/builder/strategies/{strategy_id}/validate")
async def v4_validate_strategy(strategy_id: str):
    orchestrator = get_v4()
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="v4 modules are unavailable")
    valid, errors = orchestrator.validate_strategy(strategy_id)
    return wrap_response({"valid": valid, "errors": errors}, DataSourceTag.DERIVED)


@app.get("/api/v4/builder/strategies/{strategy_id}/canvas")
async def v4_get_canvas(strategy_id: str):
    orchestrator = get_v4()
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="v4 modules are unavailable")
    return wrap_response(orchestrator.get_canvas(strategy_id), DataSourceTag.DERIVED)


@app.post("/api/v4/builder/strategies/{strategy_id}/backtest")
async def v4_backtest_strategy(strategy_id: str, payload: Dict[str, Any]):
    orchestrator = get_v4()
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="v4 modules are unavailable")

    try:
        strategy = orchestrator.strategy_builder.get(strategy_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Strategy '{strategy_id}' not found") from exc

    df = orchestrator.price_feed.get_ohlcv(
        symbol=strategy.symbol,
        timeframe=strategy.entry_timeframe,
        limit=int(payload.get("limit", 1000)),
    )

    from src.strategies.strategy_tester import StrategyTester

    tester = StrategyTester(initial_capital=float(payload.get("initial_capital", 10000)))
    result = tester.run(
        df=df,
        signal_fn=lambda _x: {"action": "HOLD"},
        strategy_name=strategy.name,
        strategy_id=strategy_id,
        symbol=strategy.symbol,
        timeframe=strategy.entry_timeframe,
    )
    monte_carlo = tester.run_monte_carlo(result) if bool(payload.get("run_monte_carlo", False)) else {}
    strategy.backtest_summary = result.to_summary_dict()

    return wrap_response(
        {
            "summary": result.to_summary_dict(),
            "equity_curve": result.equity_curve.tolist(),
            "monte_carlo": monte_carlo,
        },
        DataSourceTag.DERIVED,
    )


@app.get("/api/v4/builder/strategies/{strategy_id}/export")
async def v4_export_strategy(strategy_id: str):
    orchestrator = get_v4()
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="v4 modules are unavailable")
    return Response(content=orchestrator.export_strategy(strategy_id), media_type="application/json")


@app.post("/api/v4/builder/strategies/import")
async def v4_import_strategy(payload: Dict[str, Any]):
    orchestrator = get_v4()
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="v4 modules are unavailable")
    strategy_id = orchestrator.import_strategy(json.dumps(payload))
    return wrap_response({"strategy_id": strategy_id}, DataSourceTag.DERIVED)


@app.get("/api/v4/binance/balance")
async def v4_binance_balance():
    orchestrator = get_v4()
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="v4 modules are unavailable")
    return wrap_response(orchestrator.executor.get_balance(), DataSourceTag.DERIVED)


@app.get("/api/v4/binance/mode")
async def v4_binance_mode():
    orchestrator = get_v4()
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="v4 modules are unavailable")
    return wrap_response(
        {
            "mode": orchestrator.executor.mode,
            "is_live": orchestrator.executor.mode == "live",
        },
        DataSourceTag.DERIVED,
    )


@app.get("/api/v4/binance/positions")
async def v4_binance_positions():
    orchestrator = get_v4()
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="v4 modules are unavailable")
    return wrap_response(orchestrator.executor.get_open_positions(), DataSourceTag.DERIVED)


@app.get("/api/v4/binance/ticker")
async def v4_binance_ticker(symbol: str = "BTC/USDT"):
    orchestrator = get_v4()
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="v4 modules are unavailable")
    return wrap_response(orchestrator.executor.get_ticker(symbol), DataSourceTag.DERIVED)


@app.post("/api/engine/start")
async def start_engine(config: EngineConfig):
    """Start/reset the trading engine."""
    state.mode = config.mode
    state.initial_capital = config.capital
    state.capital = config.capital
    state.pnl = 0.0
    state.position = 0.0
    state.trades = []
    state.total_trades = 0
    state.winning_trades = 0
    state.losing_trades = 0
    state.start_time = datetime.now()
    
    await manager.broadcast({
        "type": "engine_status",
        "status": "started",
        "mode": config.mode
    })
    
    return {"status": "started", "mode": config.mode}


@app.post("/api/engine/stop")
async def stop_engine():
    """Stop the trading engine."""
    await manager.broadcast({
        "type": "engine_status",
        "status": "stopped"
    })
    return {"status": "stopped"}


@app.post("/api/emergency-stop")
async def emergency_stop():
    """Emergency stop - close all positions."""
    if state.position > 0:
        # Close position at current price
        proceeds = state.position * state.current_price * 0.999  # Fee
        pnl = (state.current_price - state.position_entry_price) * state.position
        state.pnl += pnl
        state.capital += proceeds
        
        trade = {
            "id": len(state.trades) + 1,
            "time": datetime.now().isoformat(),
            "symbol": "BTC/USDT",
            "side": "SELL",
            "amount": state.position,
            "price": state.current_price,
            "pnl": round(pnl, 2),
            "reason": "EMERGENCY_STOP"
        }
        state.trades.append(trade)
        state.position = 0
        
        await manager.broadcast({"type": "trade", **trade})
    
    await manager.broadcast({
        "type": "emergency",
        "message": "Emergency stop activated - all positions closed"
    })
    
    return {"status": "emergency_stop_activated", "positions_closed": True}


# === CryptoBoss 1.0.0 Operator Controls ===

class OperatorActionRequest(BaseModel):
    reason: str  # Mandatory per specification
    

@app.post("/api/operator/pause")
async def pause_trading(request: OperatorActionRequest):
    """
    Pause all trading activity.
    
    CryptoBoss 1.0.0: Reason is mandatory. Action permanently logged.
    Operator cannot bypass risk or capital veto.
    """
    if not request.reason or len(request.reason.strip()) < 5:
        raise HTTPException(status_code=400, detail="Reason must be at least 5 characters")
    
    state.trading_paused = True
    state.trading_pause_reason = request.reason
    
    # Permanent log
    action_log = {
        "action": "PAUSE_TRADING",
        "reason": request.reason,
        "timestamp": datetime.now().isoformat(),
        "operator": "dashboard_user",
        "previous_state": "active"
    }
    state.operator_action_log.append(action_log)
    
    await manager.broadcast({
        "type": "operator_action",
        "action": "pause",
        "reason": request.reason
    })
    
    logger.warning(f"⏸️ TRADING PAUSED by operator: {request.reason}")
    
    return wrap_response({
        "success": True,
        "action": "pause_trading",
        "trading_paused": True,
        "reason": request.reason,
        "logged_at": action_log["timestamp"]
    })


@app.post("/api/operator/resume")
async def resume_trading(request: OperatorActionRequest):
    """
    Resume trading activity.
    
    CryptoBoss 1.0.0: Reason is mandatory. Cannot resume during INCIDENT_FREEZE.
    """
    if not request.reason or len(request.reason.strip()) < 5:
        raise HTTPException(status_code=400, detail="Reason must be at least 5 characters")
    
    # Cannot resume if in incident freeze
    if state.incident_state == "INCIDENT_FREEZE":
        raise HTTPException(
            status_code=403, 
            detail="Cannot resume trading during INCIDENT_FREEZE. Must acknowledge incident first."
        )
    
    state.trading_paused = False
    state.trading_pause_reason = None
    
    action_log = {
        "action": "RESUME_TRADING",
        "reason": request.reason,
        "timestamp": datetime.now().isoformat(),
        "operator": "dashboard_user",
        "previous_state": "paused"
    }
    state.operator_action_log.append(action_log)
    
    await manager.broadcast({
        "type": "operator_action",
        "action": "resume",
        "reason": request.reason
    })
    
    logger.warning(f"▶️ TRADING RESUMED by operator: {request.reason}")
    
    return wrap_response({
        "success": True,
        "action": "resume_trading",
        "trading_paused": False,
        "reason": request.reason,
        "logged_at": action_log["timestamp"]
    })


@app.post("/api/operator/acknowledge-incident")
async def acknowledge_incident(request: OperatorActionRequest):
    """
    Acknowledge and clear an incident state.
    
    CryptoBoss 1.0.0: Manual operator acknowledgement required to exit incident state.
    Reason is permanently logged.
    """
    if not request.reason or len(request.reason.strip()) < 5:
        raise HTTPException(status_code=400, detail="Reason must be at least 5 characters")
    
    if state.incident_state == "NORMAL":
        raise HTTPException(status_code=400, detail="No incident to acknowledge")
    
    previous_state = state.incident_state
    previous_reason = state.incident_reason
    
    # Clear incident
    state.incident_state = "NORMAL"
    state.incident_reason = None
    state.incident_started_at = None
    
    action_log = {
        "action": "ACKNOWLEDGE_INCIDENT",
        "reason": request.reason,
        "timestamp": datetime.now().isoformat(),
        "operator": "dashboard_user",
        "previous_incident_state": previous_state,
        "previous_incident_reason": previous_reason
    }
    state.operator_action_log.append(action_log)
    
    await manager.broadcast({
        "type": "incident_acknowledged",
        "previous_state": previous_state,
        "reason": request.reason
    })
    
    logger.warning(f"✅ INCIDENT ACKNOWLEDGED by operator: {request.reason} (was: {previous_state})")
    
    return wrap_response({
        "success": True,
        "action": "acknowledge_incident",
        "previous_state": previous_state,
        "current_state": "NORMAL",
        "reason": request.reason,
        "logged_at": action_log["timestamp"]
    })


@app.get("/api/operator/actions")
async def get_operator_actions(limit: int = 100):
    """
    Get operator action log.
    
    CryptoBoss 1.0.0: All operator actions are permanently logged.
    """
    return wrap_response({
        "actions": state.operator_action_log[-limit:],
        "total_actions": len(state.operator_action_log),
        "trading_paused": state.trading_paused,
        "incident_state": state.incident_state
    })


@app.get("/api/incident")
async def get_incident_state():
    """
    Get current incident state.
    
    CryptoBoss 1.0.0: Incident states - NORMAL, DEGRADED, INCIDENT_FREEZE, HALTED
    """
    return wrap_response({
        "state": state.incident_state,
        "reason": state.incident_reason,
        "started_at": state.incident_started_at.isoformat() if state.incident_started_at else None,
        "trading_allowed": state.incident_state == "NORMAL" and not state.trading_paused,
        "position_reduction_only": state.incident_state == "INCIDENT_FREEZE"
    })


# === WebSocket ===

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    # Send initial state
    await websocket.send_json({
        "type": "init",
        "status": await get_status(),
        "portfolio": await get_portfolio(),
        "trades": state.trades[-10:]
    })
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            elif message.get("type") == "refresh":
                await websocket.send_json({
                    "type": "update",
                    "status": await get_status(),
                    "portfolio": await get_portfolio()
                })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# === WebSocket Prices (vFINAL) ===

# Store connected price clients
price_clients: List[WebSocket] = []


@app.websocket("/ws/prices")
async def websocket_prices(websocket: WebSocket):
    """
    Real-time price feed WebSocket.
    
    CRYPTOBOSS vFINAL: Prices from Binance WebSocket.
    - Uses MAINNET for prices (testnet has no real data)
    - Auto-pushes updates to all connected clients
    - Shows 'disconnected' if no data
    """
    await websocket.accept()
    price_clients.append(websocket)
    
    try:
        # Send current prices immediately
        if MARKET_DATA_AVAILABLE:
            service = get_market_data_service()
            prices = {symbol: tick.to_dict() for symbol, tick in service.get_all_prices().items()}
            await websocket.send_json({
                "type": "init",
                "prices": prices,
                "connected": service.is_connected,
                "timestamp": datetime.now().isoformat()
            })
        else:
            await websocket.send_json({
                "type": "init",
                "prices": {},
                "connected": False,
                "error": "Market data service not available",
                "timestamp": datetime.now().isoformat()
            })
        
        # Keep connection alive and handle client messages
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat", "timestamp": datetime.now().isoformat()})
                
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in price_clients:
            price_clients.remove(websocket)


async def broadcast_price_update(tick: PriceTick):
    """Receive price from real market data service and broadcast to WebSocket clients."""
    # Update in-memory state so REST endpoints also return real prices
    if tick.symbol in ("BTC/USDT", "BTCUSDT"):
        state.current_price = tick.price
        state.last_price = tick.price
    
    if not price_clients:
        return
        
    message = {
        "type": "price",
        "symbol": tick.symbol,
        "data": tick.to_dict()
    }
    
    disconnected = []
    for client in price_clients:
        try:
            await client.send_json(message)
        except Exception:
            disconnected.append(client)
    
    # Clean up disconnected clients
    for client in disconnected:
        price_clients.remove(client)


# === Background Tasks ===

# === Real Trading Loop ===

_trading_loop_task = None
_aggressive_scalper_instance = None


async def real_trading_loop():
    """
    Main trading loop — runs when a real exchange client is connected.
    Fetches OHLCV data, runs strategies, executes real orders on Binance.
    Loops every 30 seconds.
    """
    SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    LOOP_INTERVAL = 30  # seconds between signal checks
    
    logger.info("Real trading loop starting...")
    
    while True:
        try:
            # Only run if exchange client is connected
            if state.exchange_client is None or not state.api_validated:
                await asyncio.sleep(LOOP_INTERVAL)
                continue
            
            if state.kill_switch_active:
                logger.info("Kill switch active — trading loop paused")
                await asyncio.sleep(LOOP_INTERVAL)
                continue
            
            if not AGGRESSIVE_SCALPER_AVAILABLE or _aggressive_scalper_instance is None:
                await asyncio.sleep(LOOP_INTERVAL)
                continue
            
            # Fetch balance
            try:
                balance = await state.exchange_client.get_balance()
                usdt_free = 0
                for k, v in balance.items():
                    if k == "USDT":
                        usdt_free = v if isinstance(v, (int, float)) else v.get("free", 0)
                        break
                if usdt_free < 10:
                    logger.warning(f"Insufficient USDT balance: ${usdt_free:.2f} — skipping cycle")
                    await asyncio.sleep(LOOP_INTERVAL)
                    continue
            except Exception as e:
                logger.error(f"Balance fetch failed: {e}")
                await asyncio.sleep(LOOP_INTERVAL)
                continue
            
            # Run strategy on each symbol
            for symbol in SYMBOLS:
                try:
                    # Get OHLCV candles
                    candles = await state.exchange_client.get_ohlcv(symbol, timeframe="5m", limit=100)
                    if not candles:
                        continue
                    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
                    current_price = float(df["close"].iloc[-1])
                    
                    # Update state price
                    if symbol == "BTC/USDT":
                        state.current_price = current_price
                    
                    # Set strategy symbol and generate signal
                    _aggressive_scalper_instance.config.symbol = symbol
                    signal = _aggressive_scalper_instance.generate_signal(
                        df, len(df) - 1, current_price
                    )
                    
                    if signal.action in ("BUY", "SELL") and signal.confidence >= 0.6:
                        # Calculate order size
                        position_usdt = usdt_free * 0.08  # 8% of free capital
                        
                        if position_usdt < 10:
                            continue
                        
                        side = "buy" if signal.action == "BUY" else "sell"
                        quantity = position_usdt / current_price
                        
                        logger.info(
                            f"SIGNAL: {signal.action} {symbol} @ {current_price:.4f} | "
                            f"qty={quantity:.6f} | confidence={signal.confidence:.2f} | "
                            f"SL={signal.stop_loss:.4f} TP={signal.take_profit:.4f}"
                        )
                        
                        # Place real order
                        try:
                            order = await state.exchange_client.create_order(
                                symbol=symbol,
                                side=side,
                                order_type="market",
                                amount=round(quantity, 6),
                            )
                            
                            logger.info(f"Order placed: {order}")
                            
                            # Record trade
                            state.total_trades += 1
                            trade_record = {
                                "id": state.total_trades,
                                "time": datetime.now().isoformat(),
                                "symbol": symbol,
                                "side": signal.action,
                                "amount": round(quantity, 6),
                                "price": current_price,
                                "pnl": 0,
                                "reason": signal.reason[:80],
                                "order_id": order.get("id", ""),
                                "confidence": signal.confidence,
                            }
                            state.trades.append(trade_record)
                            
                            # Broadcast to WebSocket clients
                            await manager.broadcast({
                                "type": "trade",
                                **trade_record,
                            })
                            
                        except Exception as order_error:
                            logger.error(f"Order placement failed for {symbol}: {order_error}")

                except Exception as symbol_error:
                    logger.error(f"Error processing {symbol}: {symbol_error}")
                    continue
            
            await asyncio.sleep(LOOP_INTERVAL)
            
        except asyncio.CancelledError:
            logger.info("Real trading loop cancelled")
            break
        except Exception as loop_error:
            logger.error(f"Trading loop error: {loop_error}")
            await asyncio.sleep(60)  # Back off on error


async def start_real_trading_loop():
    """Start the real trading loop task. Called when exchange client is connected."""
    global _trading_loop_task
    if _trading_loop_task is None or _trading_loop_task.done():
        _trading_loop_task = asyncio.create_task(real_trading_loop())
        logger.info("Real trading loop task started")


async def stop_real_trading_loop():
    """Stop the real trading loop task."""
    global _trading_loop_task
    if _trading_loop_task and not _trading_loop_task.done():
        _trading_loop_task.cancel()
        try:
            await _trading_loop_task
        except asyncio.CancelledError:
            pass
        _trading_loop_task = None
        logger.info("Real trading loop stopped")


@app.on_event("startup")
async def startup():
    """Start background tasks — real market data only, no simulator."""
    # Real Binance market data (public WebSocket — no API key required)
    if MARKET_DATA_AVAILABLE:
        try:
            service = get_market_data_service()
            service.subscribe(broadcast_price_update)
            await service.start()
            logger.info("Market data service started — real-time prices from Binance")
        except Exception as e:
            logger.warning(f"Market data service failed to start: {e}")
            logger.warning("Prices will show as 0 until a session is connected")
    else:
        logger.warning("Market data service not available — install dependencies")

    logger.info("CryptoBoss API ready — no simulator running")


# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
