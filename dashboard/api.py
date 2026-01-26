"""
CryptoBoss Dashboard API - Enhanced with Live Trading Simulation

FastAPI backend with WebSocket for real-time updates.
Now includes simulated trading activity to show dynamic values.
"""

import asyncio
import json
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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


# Global state with proper initialization
class DashboardState:
    def __init__(self):
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


# === Models ===

class EngineConfig(BaseModel):
    mode: str = "paper"
    capital: float = 10000.0
    symbols: List[str] = ["BTC/USDT"]
    strategy: str = "dca"


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


@app.get("/api/status")
async def get_status():
    """Get current bot status."""
    uptime = (datetime.now() - state.start_time).total_seconds()
    
    return {
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
