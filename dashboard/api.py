"""
CryptoBoss Dashboard API - Enhanced with Live Trading Simulation

FastAPI backend with WebSocket for real-time updates.
Now includes simulated trading activity to show dynamic values.
"""

import asyncio
import json
import logging
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Dashboard")

app = FastAPI(
    title="CryptoBoss Dashboard",
    description="Professional Trading Bot Dashboard",
    version="2.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Models ===

class EngineConfig(BaseModel):
    mode: str = "paper"
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
        self.mode = "paper"
        self.initial_capital = 10000.0
        self.capital = 10000.0
        self.pnl = 0.0
        self.start_time = datetime.now()
        self.current_price = 65000.0
        self.last_price = 65000.0
        self.price_history: List[Dict] = []
        self.trades: List[Dict] = []
        self.position = 0.0  # BTC held
        self.position_entry_price = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.api_validated = False
        self.exchange_client = None
        
        # System state
        self.environment = "paper"  # paper, testnet, live
        self.connection_status = "disconnected"  # disconnected, connecting, connected, error
        self.timestamp_offset_ms = 0
        self.last_time_sync = None
        self.kill_switch_active = False
        self.kill_switch_reason = None
        
        # Market context
        self.market_context = "UNKNOWN"  # TRENDING, RANGING, VOLATILE, CRISIS
        self.market_bias = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
        self.last_context_update = None
        
        # Decision tracking
        self.recent_decisions: List[Dict] = []
        self.last_decision_time = None
        self.decisions_today = 0
        self.rejections_today = 0

    def reset(self, new_mode: str = "paper"):
        """Reset all state for new session."""
        self.session_id = str(uuid.uuid4())
        self.mode = new_mode
        self.capital = self.initial_capital
        self.pnl = 0.0
        self.start_time = datetime.now()
        self.price_history = []
        self.trades = []
        self.position = 0.0
        self.position_entry_price = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.api_validated = new_mode == "paper"
        
        # Reset system state
        self.environment = new_mode
        self.connection_status = "disconnected" if new_mode != "paper" else "connected"
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
    
    if mode not in ["paper", "testnet", "live"]:
        raise HTTPException(status_code=400, detail="Invalid mode. Must be: paper, testnet, or live")
    
    # For non-paper modes, validate API credentials
    balances = {}
    if mode != "paper":
        if not request.api_key or not request.api_secret:
            raise HTTPException(status_code=400, detail="API credentials required for non-paper mode")
        
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


# === System State Endpoints ===

@app.get("/api/system")
async def get_system():
    """
    Get complete system state.
    
    Returns environment, connection, time sync, kill switch state.
    This is the primary endpoint for dashboard observability.
    """
    uptime = (datetime.now() - state.start_time).total_seconds()
    
    return {
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
        "ws_clients": len(manager.active_connections)
    }


@app.get("/api/context")
async def get_context():
    """
    Get current market context and bias.
    
    Used by dashboard to show trading environment.
    """
    return {
        "market_context": state.market_context,
        "bias": state.market_bias,
        "last_update": state.last_context_update.isoformat() if state.last_context_update else None,
        "current_price": state.current_price,
        "price_change_pct": ((state.current_price - state.last_price) / state.last_price * 100) if state.last_price > 0 else 0,
        "symbol": "BTC/USDT"
    }


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
    """
    drawdown_pct = (state.pnl / state.initial_capital * 100) if state.initial_capital > 0 else 0
    
    return {
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
        "kill_switch_active": state.kill_switch_active
    }


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
    
    logger.warning(f"Kill switch {'ACTIVATED' if active else 'DEACTIVATED'}: {reason}")
    
    return {
        "success": True,
        "kill_switch_active": state.kill_switch_active,
        "reason": state.kill_switch_reason
    }



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
    """Get portfolio details."""
    btc_value = state.position * state.current_price
    
    return {
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
    }


@app.get("/api/trades")
async def get_trades(limit: int = 50):
    """Get recent trades."""
    return state.trades[-limit:]


@app.get("/api/strategies")
async def get_strategies():
    """Get active strategies."""
    return {
        "strategies": [
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
    }


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


# === Background Tasks ===

async def price_and_trading_simulator():
    """Simulate price updates and DCA trading activity."""
    trade_cooldown = 0
    
    while True:
        try:
            # Random price walk
            change_pct = random.gauss(0, 0.001)  # 0.1% std dev
            state.last_price = state.current_price
            state.current_price = max(50000, min(80000, state.current_price * (1 + change_pct)))
            
            price_change_pct = ((state.current_price - state.last_price) / state.last_price) * 100
            
            # Update price history
            state.price_history.append({
                "time": datetime.now().isoformat(),
                "price": state.current_price
            })
            if len(state.price_history) > 100:
                state.price_history = state.price_history[-100:]
            
            # Broadcast price update
            await manager.broadcast({
                "type": "price",
                "symbol": "BTC/USDT",
                "price": round(state.current_price, 2),
                "change_pct": round(price_change_pct, 4),
                "timestamp": datetime.now().isoformat()
            })
            
            # Simulated DCA Trading Logic
            trade_cooldown -= 1
            
            if trade_cooldown <= 0:
                # Check if we should enter a position (price dip)
                if state.position == 0 and random.random() < 0.05:  # 5% chance to buy
                    # Buy some BTC
                    invest_amount = state.capital * 0.1  # 10% of capital
                    btc_amount = invest_amount / state.current_price
                    fee = invest_amount * 0.001
                    
                    state.capital -= (invest_amount + fee)
                    state.position = btc_amount
                    state.position_entry_price = state.current_price
                    state.total_trades += 1
                    
                    trade = {
                        "id": len(state.trades) + 1,
                        "time": datetime.now().isoformat(),
                        "symbol": "BTC/USDT",
                        "side": "BUY",
                        "amount": round(btc_amount, 6),
                        "price": round(state.current_price, 2),
                        "pnl": 0,
                        "reason": "DCA_ENTRY"
                    }
                    state.trades.append(trade)
                    
                    await manager.broadcast({"type": "trade", **trade})
                    trade_cooldown = random.randint(10, 30)  # Wait 10-30 seconds
                
                # Check if we should exit (take profit or stop loss)
                elif state.position > 0:
                    pnl_pct = ((state.current_price - state.position_entry_price) / state.position_entry_price) * 100
                    
                    # Take profit at +2% or stop loss at -3%
                    if pnl_pct >= 2.0 or pnl_pct <= -3.0 or random.random() < 0.02:
                        proceeds = state.position * state.current_price
                        fee = proceeds * 0.001
                        pnl = (state.current_price - state.position_entry_price) * state.position - fee
                        
                        state.capital += (proceeds - fee)
                        state.pnl += pnl
                        state.total_trades += 1
                        
                        if pnl > 0:
                            state.winning_trades += 1
                        else:
                            state.losing_trades += 1
                        
                        reason = "TAKE_PROFIT" if pnl_pct >= 2.0 else ("STOP_LOSS" if pnl_pct <= -3.0 else "SIGNAL")
                        
                        trade = {
                            "id": len(state.trades) + 1,
                            "time": datetime.now().isoformat(),
                            "symbol": "BTC/USDT",
                            "side": "SELL",
                            "amount": round(state.position, 6),
                            "price": round(state.current_price, 2),
                            "pnl": round(pnl, 2),
                            "reason": reason
                        }
                        state.trades.append(trade)
                        state.position = 0
                        state.position_entry_price = 0
                        
                        await manager.broadcast({"type": "trade", **trade})
                        trade_cooldown = random.randint(5, 15)
            
            # Broadcast status update every few seconds
            await manager.broadcast({
                "type": "status_update",
                "portfolio_value": round(state.portfolio_value, 2),
                "pnl": round(state.total_pnl, 2),
                "pnl_pct": round((state.total_pnl / state.initial_capital * 100), 2) if state.initial_capital > 0 else 0,
                "trades_count": state.total_trades,
                "win_rate": round(state.win_rate, 1),
                "position": round(state.position, 6)
            })
            
            await asyncio.sleep(1)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Simulator error: {e}")
            await asyncio.sleep(5)


@app.on_event("startup")
async def startup():
    """Start background tasks."""
    asyncio.create_task(price_and_trading_simulator())
    logger.info("Dashboard API started with trading simulator")


# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
