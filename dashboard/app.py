"""
CryptoBoss Professional Trading Dashboard
Full-featured UI with real Binance data, multiple strategies, and trading controls

Run: python -m dashboard.app
Open: http://localhost:8000
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import asyncio
from typing import List, Dict, Any, Optional
import json
import logging
import signal

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Try importing core modules
try:
    from src.core import (
        get_state_manager, get_risk_guardian, get_event_bus
    )
    from src.strategies.dca_strategy import DCAStrategy
    from src.strategies.grid_strategy import GridTradingStrategy
    CORE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Core imports not available: {e}")
    CORE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CryptoBoss Pro",
    description="Professional Crypto Trading Dashboard",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared trading state
trading_state: Dict[str, Any] = {
    'system_status': 'running',
    'mode': 'paper',
    'uptime_start': datetime.now(),
    'equity': 10000.0,
    'initial_capital': 10000.0,
    'positions': [],
    'trades': [],
    'signals': [],
    'candles': [],
    'orderbook': {'bids': [], 'asks': []},
    'strategies': [
        {'id': 'dca_btc_usdt', 'name': 'DCA Strategy', 'symbol': 'BTC/USDT', 'type': 'DCA', 'status': 'active', 'pnl': 0},
        {'id': 'grid_btc_usdt', 'name': 'Grid Trading', 'symbol': 'BTC/USDT', 'type': 'GRID', 'status': 'paused', 'pnl': 0},
        {'id': 'mm_eth_usdt', 'name': 'Market Making', 'symbol': 'ETH/USDT', 'type': 'MM', 'status': 'active', 'pnl': 0},
    ],
    'metrics': {
        'total_return': 0.0,
        'win_rate': 0.0,
        'sharpe_ratio': 0.0,
        'max_drawdown': 0.0,
        'num_trades': 0,
        'profit_factor': 0.0
    },
    'sentiment': {
        'fear_greed_index': 50,
        'level': 'neutral'
    },
    'current_price': 0.0,
    'price_change_24h': 0.0
}

# WebSocket connections
active_connections: List[WebSocket] = []


# ==================== API ENDPOINTS ====================

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the professional trading dashboard"""
    return get_professional_dashboard_html()


@app.get("/audit")
async def audit_dashboard():
    """Serve the static codebase audit dashboard page."""
    audit_path = Path(__file__).parent / "static" / "audit.html"
    if not audit_path.exists():
        return HTMLResponse("Audit page not found", status_code=404)
    return FileResponse(str(audit_path))


@app.get("/api/status")
async def get_status():
    uptime = datetime.now() - trading_state['uptime_start']
    return {
        'status': trading_state['system_status'],
        'mode': trading_state['mode'],
        'uptime_seconds': int(uptime.total_seconds()),
        'uptime_formatted': str(uptime).split('.')[0],
        'timestamp': datetime.now().isoformat()
    }


@app.get("/api/performance")
async def get_performance():
    equity = trading_state['equity']
    initial = trading_state['initial_capital']
    return {
        'equity': round(equity, 2),
        'initial_capital': round(initial, 2),
        'total_return_pct': round((equity / initial - 1) * 100, 2),
        'total_return_usd': round(equity - initial, 2),
        'metrics': trading_state['metrics'],
        'timestamp': datetime.now().isoformat()
    }


@app.get("/api/positions")
async def get_positions():
    return {
        'positions': trading_state['positions'],
        'count': len(trading_state['positions'])
    }


@app.get("/api/trades")
async def get_trades(limit: int = 20):
    return {
        'trades': trading_state['trades'][-limit:],
        'total': len(trading_state['trades'])
    }


@app.get("/api/signals")
async def get_signals(limit: int = 10):
    return {'signals': trading_state['signals'][-limit:]}


@app.get("/api/strategies")
async def get_strategies():
    """Get all configured strategies"""
    return {'strategies': trading_state['strategies']}


@app.get("/api/candles")
async def get_candles():
    """Fetch real candle data from Binance API."""
    import requests
    
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {'symbol': 'BTCUSDT', 'interval': '1h', 'limit': 100}
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        candles = []
        for kline in data:
            candles.append({
                'time': int(kline[0] / 1000),
                'open': float(kline[1]),
                'high': float(kline[2]),
                'low': float(kline[3]),
                'close': float(kline[4])
            })
        
        if candles:
            trading_state['current_price'] = candles[-1]['close']
            trading_state['candles'] = candles
        
        return {'candles': candles}
    
    except Exception as e:
        logger.error(f"Failed to fetch candles: {e}")
        return {'candles': trading_state.get('candles', [])}


@app.get("/api/orderbook")
async def get_orderbook():
    """Fetch real orderbook data from Binance API."""
    import requests
    
    try:
        url = "https://api.binance.com/api/v3/depth"
        params = {'symbol': 'BTCUSDT', 'limit': 10}
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        bids = [[float(b[0]), float(b[1])] for b in data['bids']]
        asks = [[float(a[0]), float(a[1])] for a in data['asks']]
        
        spread = asks[0][0] - bids[0][0] if bids and asks else 0
        mid_price = (asks[0][0] + bids[0][0]) / 2 if bids and asks else 0
        
        return {
            'bids': bids,
            'asks': asks,
            'spread': round(spread, 2),
            'mid_price': round(mid_price, 2)
        }
    
    except Exception as e:
        logger.error(f"Failed to fetch orderbook: {e}")
        return {'bids': [], 'asks': [], 'spread': 0, 'mid_price': 0}


@app.get("/api/sentiment")
async def get_sentiment():
    """Fetch Fear & Greed Index"""
    import requests
    
    try:
        response = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
        data = response.json()
        
        if data.get('data'):
            value = int(data['data'][0]['value'])
            classification = data['data'][0]['value_classification']
            
            trading_state['sentiment'] = {
                'fear_greed_index': value,
                'level': classification.lower()
            }
    except:
        pass
    
    return {'sentiment': trading_state['sentiment']}


@app.get("/api/ticker")
async def get_ticker():
    """Get 24h ticker from Binance"""
    import requests
    
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        params = {'symbol': 'BTCUSDT'}
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        return {
            'price': float(data['lastPrice']),
            'change_24h': float(data['priceChangePercent']),
            'high_24h': float(data['highPrice']),
            'low_24h': float(data['lowPrice']),
            'volume_24h': float(data['volume'])
        }
    except Exception as e:
        return {'price': trading_state['current_price'], 'change_24h': 0}


@app.post("/api/emergency-stop")
async def emergency_stop():
    """Emergency kill switch"""
    trading_state['system_status'] = 'EMERGENCY_HALT'
    trading_state['emergency_halt'] = True
    trading_state['halt_time'] = datetime.now().isoformat()
    
    # Broadcast to all connected clients
    for conn in active_connections:
        try:
            await conn.send_json({
                'type': 'emergency',
                'message': 'Emergency stop activated!'
            })
        except:
            pass
    
    return {
        'status': 'halted',
        'message': 'Emergency stop activated. Trading halted.',
        'halt_time': trading_state['halt_time']
    }


@app.post("/api/resume-trading")
async def resume_trading():
    """Resume trading after emergency stop"""
    trading_state['system_status'] = 'running'
    trading_state['emergency_halt'] = False
    return {'status': 'resumed', 'message': 'Trading resumed'}


@app.post("/api/strategy/{strategy_id}/toggle")
async def toggle_strategy(strategy_id: str):
    """Toggle strategy on/off"""
    for s in trading_state['strategies']:
        if s['id'] == strategy_id:
            s['status'] = 'paused' if s['status'] == 'active' else 'active'
            return {'status': 'ok', 'strategy': s}
    return {'status': 'error', 'message': 'Strategy not found'}


@app.post("/api/reset-dashboard")
async def reset_dashboard():
    """Reset all dashboard data"""
    trading_state['trades'] = []
    trading_state['signals'] = []
    trading_state['positions'] = []
    trading_state['equity'] = 10000.0
    trading_state['metrics'] = {
        'total_return': 0.0,
        'win_rate': 0.0,
        'sharpe_ratio': 0.0,
        'max_drawdown': 0.0,
        'num_trades': 0,
        'profit_factor': 0.0
    }
    return {'status': 'success', 'message': 'Dashboard reset'}


# ==================== WEBSOCKET ====================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"Client connected. Total: {len(active_connections)}")
    
    try:
        # Send initial state
        await websocket.send_json({
            'type': 'init',
            'status': trading_state['system_status'],
            'equity': trading_state['equity'],
            'strategies': trading_state['strategies']
        })
        
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get('type') == 'ping':
                await websocket.send_json({'type': 'pong'})
                
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total: {len(active_connections)}")


# ==================== DASHBOARD HTML ====================

def get_professional_dashboard_html() -> str:
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CryptoBoss Pro Trading</title>
    <script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #0a0e17;
            --bg-card: rgba(16, 23, 39, 0.9);
            --glass-border: rgba(255, 255, 255, 0.08);
            --text-primary: #ffffff;
            --text-secondary: #64748b;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-blue: #3b82f6;
            --accent-purple: #8b5cf6;
            --accent-yellow: #f59e0b;
            --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
        }

        .bg-animation {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1;
            background: 
                radial-gradient(ellipse at 20% 80%, rgba(99, 102, 241, 0.15) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 20%, rgba(139, 92, 246, 0.1) 0%, transparent 50%);
        }

        .header {
            background: var(--bg-card);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid var(--glass-border);
            padding: 16px 32px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky; top: 0; z-index: 100;
        }

        .logo { display: flex; align-items: center; gap: 12px; }
        .logo-icon {
            width: 40px; height: 40px;
            background: var(--gradient-primary);
            border-radius: 12px;
            display: flex; align-items: center; justify-content: center;
            font-size: 20px;
        }
        .logo-text {
            font-size: 22px; font-weight: 700;
            background: linear-gradient(135deg, #fff 0%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .header-stats { display: flex; gap: 32px; }
        .header-stat { text-align: right; }
        .header-stat-label { font-size: 11px; color: var(--text-secondary); text-transform: uppercase; }
        .header-stat-value { font-size: 18px; font-weight: 600; }

        .status-badge {
            display: flex; align-items: center; gap: 8px;
            padding: 8px 16px;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 20px;
            font-size: 13px;
            color: var(--accent-green);
        }
        .status-badge.offline {
            background: rgba(239, 68, 68, 0.1);
            border-color: rgba(239, 68, 68, 0.3);
            color: var(--accent-red);
        }
        .status-dot {
            width: 8px; height: 8px;
            background: currentColor;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

        .main {
            display: grid;
            grid-template-columns: 1fr 320px;
            gap: 20px;
            padding: 20px 32px;
            max-width: 1800px;
            margin: 0 auto;
        }

        .card {
            background: var(--bg-card);
            backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 20px;
        }
        .card-title {
            font-size: 14px; font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 16px;
            display: flex; align-items: center; gap: 8px;
        }

        #chart-container { height: 400px; border-radius: 12px; overflow: hidden; }

        .orderbook { font-family: 'SF Mono', monospace; font-size: 12px; }
        .orderbook-row {
            display: grid; grid-template-columns: 1fr 1fr;
            gap: 8px; padding: 4px 8px; border-radius: 4px;
        }
        .orderbook-row.ask { background: rgba(239, 68, 68, 0.1); }
        .orderbook-row.bid { background: rgba(16, 185, 129, 0.1); }
        .orderbook-price.ask { color: var(--accent-red); }
        .orderbook-price.bid { color: var(--accent-green); }
        .orderbook-qty { color: var(--text-secondary); text-align: right; }
        .spread-indicator {
            text-align: center; padding: 8px;
            color: var(--text-secondary); font-size: 11px;
            border-top: 1px solid var(--glass-border);
            border-bottom: 1px solid var(--glass-border);
            margin: 4px 0;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 16px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 16px;
            text-align: center;
        }
        .metric-icon { font-size: 24px; margin-bottom: 8px; }
        .metric-value { font-size: 24px; font-weight: 700; margin-bottom: 4px; }
        .metric-value.positive { color: var(--accent-green); }
        .metric-value.negative { color: var(--accent-red); }
        .metric-label { font-size: 11px; color: var(--text-secondary); text-transform: uppercase; }

        .strategies-section { grid-column: 1 / -1; }
        .strategies-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
        .strategy-card {
            background: rgba(255,255,255,0.02);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 16px;
            display: flex; align-items: center; gap: 12px;
        }
        .strategy-icon {
            width: 48px; height: 48px;
            border-radius: 12px;
            display: flex; align-items: center; justify-content: center;
            font-size: 20px;
        }
        .strategy-icon.dca { background: rgba(59, 130, 246, 0.2); color: var(--accent-blue); }
        .strategy-icon.grid { background: rgba(139, 92, 246, 0.2); color: var(--accent-purple); }
        .strategy-icon.mm { background: rgba(245, 158, 11, 0.2); color: var(--accent-yellow); }
        .strategy-info { flex: 1; }
        .strategy-name { font-weight: 600; margin-bottom: 4px; }
        .strategy-symbol { font-size: 12px; color: var(--text-secondary); }
        .strategy-status {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 11px; font-weight: 600;
        }
        .strategy-status.active { background: rgba(16, 185, 129, 0.2); color: var(--accent-green); }
        .strategy-status.paused { background: rgba(245, 158, 11, 0.2); color: var(--accent-yellow); }

        .bottom-section {
            grid-column: 1 / -1;
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
        }

        .trades-table { width: 100%; border-collapse: collapse; }
        .trades-table th {
            text-align: left; padding: 12px 8px;
            font-size: 11px; font-weight: 500;
            color: var(--text-secondary);
            text-transform: uppercase;
            border-bottom: 1px solid var(--glass-border);
        }
        .trades-table td {
            padding: 10px 8px; font-size: 13px;
            border-bottom: 1px solid var(--glass-border);
        }
        .trade-side {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px; font-weight: 600;
        }
        .trade-side.long { background: rgba(16, 185, 129, 0.2); color: var(--accent-green); }
        .trade-side.short { background: rgba(239, 68, 68, 0.2); color: var(--accent-red); }
        .pnl-positive { color: var(--accent-green) !important; }
        .pnl-negative { color: var(--accent-red) !important; }

        .fg-gauge { text-align: center; padding: 30px; }
        .fg-value { font-size: 64px; font-weight: 700; line-height: 1; }
        .fg-value.fear { color: var(--accent-red); }
        .fg-value.neutral { color: var(--accent-yellow); }
        .fg-value.greed { color: var(--accent-green); }
        .fg-label { font-size: 14px; color: var(--text-secondary); margin-top: 8px; text-transform: capitalize; }

        .controls { display: flex; gap: 12px; }
        .btn {
            padding: 10px 20px;
            border-radius: 10px;
            font-size: 13px; font-weight: 600;
            cursor: pointer;
            border: none;
            transition: all 0.3s;
            display: flex; align-items: center; gap: 8px;
        }
        .btn-danger { background: var(--accent-red); color: white; }
        .btn-danger:hover { box-shadow: 0 0 20px rgba(239, 68, 68, 0.4); }
        .btn-secondary {
            background: rgba(255,255,255,0.05);
            color: var(--text-primary);
            border: 1px solid var(--glass-border);
        }

        @media (max-width: 1200px) {
            .main { grid-template-columns: 1fr; }
            .metrics-grid { grid-template-columns: repeat(3, 1fr); }
            .strategies-grid { grid-template-columns: 1fr; }
            .bottom-section { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="bg-animation"></div>

    <header class="header">
        <div class="logo">
            <div class="logo-icon"><i class="fas fa-robot"></i></div>
            <span class="logo-text">CryptoBoss Pro</span>
        </div>
        
        <div class="header-stats">
            <div class="header-stat">
                <div class="header-stat-label">BTC/USDT</div>
                <div class="header-stat-value" id="btcPrice">$--</div>
            </div>
            <div class="header-stat">
                <div class="header-stat-label">24h Change</div>
                <div class="header-stat-value" id="priceChange">--%</div>
            </div>
            <div class="header-stat">
                <div class="header-stat-label">Portfolio</div>
                <div class="header-stat-value" id="portfolioValue">$10,000</div>
            </div>
        </div>

        <div class="controls">
            <button class="btn btn-secondary" onclick="refreshAll()">
                <i class="fas fa-sync-alt"></i> Refresh
            </button>
            <button class="btn btn-danger" onclick="emergencyStop()">
                <i class="fas fa-stop"></i> Emergency Stop
            </button>
        </div>

        <div class="status-badge" id="statusBadge">
            <span class="status-dot"></span>
            <span id="statusText">Connecting...</span>
        </div>
    </header>

    <main class="main">
        <!-- Metrics Row -->
        <section class="card" style="grid-column: 1 / -1;">
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-icon">💰</div>
                    <div class="metric-value" id="totalReturn">$0.00</div>
                    <div class="metric-label">Total P&L</div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon">📊</div>
                    <div class="metric-value" id="returnPct">0.00%</div>
                    <div class="metric-label">Return %</div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon">🎯</div>
                    <div class="metric-value" id="winRate">--%</div>
                    <div class="metric-label">Win Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon">📉</div>
                    <div class="metric-value" id="maxDrawdown">0.00%</div>
                    <div class="metric-label">Max DD</div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon">📈</div>
                    <div class="metric-value" id="sharpeRatio">0.00</div>
                    <div class="metric-label">Sharpe</div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon">🔄</div>
                    <div class="metric-value" id="numTrades">0</div>
                    <div class="metric-label">Trades</div>
                </div>
            </div>
        </section>

        <!-- Chart -->
        <section class="card">
            <div class="card-title"><i class="fas fa-chart-line"></i> BTC/USDT Price Chart</div>
            <div id="chart-container"></div>
        </section>

        <!-- Orderbook -->
        <section class="card">
            <div class="card-title"><i class="fas fa-book"></i> Order Book</div>
            <div class="orderbook" id="orderbook">
                <!-- Asks -->
                <div id="asks"></div>
                <div class="spread-indicator" id="spread">Spread: $--</div>
                <!-- Bids -->
                <div id="bids"></div>
            </div>
        </section>

        <!-- Strategies -->
        <section class="card strategies-section">
            <div class="card-title"><i class="fas fa-cogs"></i> Active Strategies</div>
            <div class="strategies-grid" id="strategiesGrid"></div>
        </section>

        <!-- Bottom Section -->
        <section class="bottom-section">
            <!-- Trades -->
            <div class="card">
                <div class="card-title"><i class="fas fa-history"></i> Recent Trades</div>
                <table class="trades-table">
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Symbol</th>
                            <th>Side</th>
                            <th>Price</th>
                            <th>Size</th>
                            <th>P&L</th>
                        </tr>
                    </thead>
                    <tbody id="tradesBody">
                        <tr><td colspan="6" style="text-align:center;color:var(--text-secondary);">No trades yet</td></tr>
                    </tbody>
                </table>
            </div>

            <!-- Fear & Greed -->
            <div class="card">
                <div class="card-title"><i class="fas fa-gauge-high"></i> Fear & Greed Index</div>
                <div class="fg-gauge">
                    <div class="fg-value neutral" id="fgValue">50</div>
                    <div class="fg-label" id="fgLabel">Neutral</div>
                </div>
            </div>
        </section>
    </main>

    <script>
        let chart = null;
        let candleSeries = null;
        let ws = null;

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            initChart();
            connectWebSocket();
            loadAllData();
            
            // Auto-refresh every 30 seconds
            setInterval(loadAllData, 30000);
        });

        function initChart() {
            const container = document.getElementById('chart-container');
            chart = LightweightCharts.createChart(container, {
                layout: {
                    background: { type: 'solid', color: 'transparent' },
                    textColor: '#64748b',
                },
                grid: {
                    vertLines: { color: 'rgba(255,255,255,0.03)' },
                    horzLines: { color: 'rgba(255,255,255,0.03)' },
                },
                crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
                rightPriceScale: { borderColor: 'rgba(255,255,255,0.1)' },
                timeScale: { borderColor: 'rgba(255,255,255,0.1)', timeVisible: true },
            });

            candleSeries = chart.addCandlestickSeries({
                upColor: '#10b981',
                downColor: '#ef4444',
                borderUpColor: '#10b981',
                borderDownColor: '#ef4444',
                wickUpColor: '#10b981',
                wickDownColor: '#ef4444',
            });

            chart.timeScale().fitContent();
        }

        function connectWebSocket() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);
            
            ws.onopen = () => {
                document.getElementById('statusBadge').classList.remove('offline');
                document.getElementById('statusText').textContent = 'Live';
            };
            
            ws.onclose = () => {
                document.getElementById('statusBadge').classList.add('offline');
                document.getElementById('statusText').textContent = 'Disconnected';
                setTimeout(connectWebSocket, 3000);
            };
            
            ws.onmessage = (e) => {
                const data = JSON.parse(e.data);
                if (data.type === 'emergency') alert(data.message);
            };
        }

        async function loadAllData() {
            await Promise.all([
                loadTicker(),
                loadCandles(),
                loadOrderbook(),
                loadPerformance(),
                loadStrategies(),
                loadTrades(),
                loadSentiment()
            ]);
        }

        async function refreshAll() {
            await loadAllData();
        }

        async function loadTicker() {
            try {
                const res = await fetch('/api/ticker');
                const data = await res.json();
                
                document.getElementById('btcPrice').textContent = '$' + data.price.toLocaleString(undefined, {minimumFractionDigits: 2});
                
                const changeEl = document.getElementById('priceChange');
                changeEl.textContent = (data.change_24h >= 0 ? '+' : '') + data.change_24h.toFixed(2) + '%';
                changeEl.style.color = data.change_24h >= 0 ? '#10b981' : '#ef4444';
            } catch (e) { console.error('Ticker error:', e); }
        }

        async function loadCandles() {
            try {
                const res = await fetch('/api/candles');
                const data = await res.json();
                if (data.candles && data.candles.length > 0) {
                    candleSeries.setData(data.candles);
                    chart.timeScale().fitContent();
                }
            } catch (e) { console.error('Candles error:', e); }
        }

        async function loadOrderbook() {
            try {
                const res = await fetch('/api/orderbook');
                const data = await res.json();
                
                const asksHtml = data.asks.slice(0, 8).reverse().map(a => 
                    `<div class="orderbook-row ask">
                        <span class="orderbook-price ask">${a[0].toLocaleString()}</span>
                        <span class="orderbook-qty">${a[1].toFixed(4)}</span>
                    </div>`
                ).join('');
                
                const bidsHtml = data.bids.slice(0, 8).map(b => 
                    `<div class="orderbook-row bid">
                        <span class="orderbook-price bid">${b[0].toLocaleString()}</span>
                        <span class="orderbook-qty">${b[1].toFixed(4)}</span>
                    </div>`
                ).join('');
                
                document.getElementById('asks').innerHTML = asksHtml;
                document.getElementById('bids').innerHTML = bidsHtml;
                document.getElementById('spread').textContent = `Spread: $${data.spread.toFixed(2)} | Mid: $${data.mid_price.toLocaleString()}`;
            } catch (e) { console.error('Orderbook error:', e); }
        }

        async function loadPerformance() {
            try {
                const res = await fetch('/api/performance');
                const data = await res.json();
                
                document.getElementById('portfolioValue').textContent = '$' + data.equity.toLocaleString();
                
                const returnEl = document.getElementById('totalReturn');
                returnEl.textContent = (data.total_return_usd >= 0 ? '+' : '') + '$' + data.total_return_usd.toFixed(2);
                returnEl.classList.toggle('positive', data.total_return_usd >= 0);
                returnEl.classList.toggle('negative', data.total_return_usd < 0);
                
                const pctEl = document.getElementById('returnPct');
                pctEl.textContent = (data.total_return_pct >= 0 ? '+' : '') + data.total_return_pct.toFixed(2) + '%';
                pctEl.classList.toggle('positive', data.total_return_pct >= 0);
                pctEl.classList.toggle('negative', data.total_return_pct < 0);
                
                document.getElementById('winRate').textContent = (data.metrics.win_rate * 100).toFixed(1) + '%';
                document.getElementById('maxDrawdown').textContent = data.metrics.max_drawdown.toFixed(2) + '%';
                document.getElementById('sharpeRatio').textContent = data.metrics.sharpe_ratio.toFixed(2);
                document.getElementById('numTrades').textContent = data.metrics.num_trades;
            } catch (e) { console.error('Performance error:', e); }
        }

        async function loadStrategies() {
            try {
                const res = await fetch('/api/strategies');
                const data = await res.json();
                
                const icons = { DCA: 'dca', GRID: 'grid', MM: 'mm' };
                const iconSymbols = { DCA: 'fa-layer-group', GRID: 'fa-border-all', MM: 'fa-chart-bar' };
                
                document.getElementById('strategiesGrid').innerHTML = data.strategies.map(s => `
                    <div class="strategy-card">
                        <div class="strategy-icon ${icons[s.type] || 'dca'}">
                            <i class="fas ${iconSymbols[s.type] || 'fa-cog'}"></i>
                        </div>
                        <div class="strategy-info">
                            <div class="strategy-name">${s.name}</div>
                            <div class="strategy-symbol">${s.symbol}</div>
                        </div>
                        <span class="strategy-status ${s.status}">${s.status.toUpperCase()}</span>
                    </div>
                `).join('');
            } catch (e) { console.error('Strategies error:', e); }
        }

        async function loadTrades() {
            try {
                const res = await fetch('/api/trades?limit=10');
                const data = await res.json();
                
                if (data.trades.length === 0) {
                    document.getElementById('tradesBody').innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--text-secondary);padding:20px;">No trades yet</td></tr>';
                    return;
                }
                
                document.getElementById('tradesBody').innerHTML = data.trades.map(t => `
                    <tr>
                        <td>${new Date(t.time || t.timestamp).toLocaleTimeString()}</td>
                        <td>${t.symbol || 'BTC/USDT'}</td>
                        <td><span class="trade-side ${t.side?.toLowerCase() || 'long'}">${(t.side || 'LONG').toUpperCase()}</span></td>
                        <td>$${(t.price || 0).toLocaleString()}</td>
                        <td>${(t.size || t.amount || 0).toFixed(4)}</td>
                        <td class="${(t.pnl || 0) >= 0 ? 'pnl-positive' : 'pnl-negative'}">${(t.pnl || 0) >= 0 ? '+' : ''}$${(t.pnl || 0).toFixed(2)}</td>
                    </tr>
                `).join('');
            } catch (e) { console.error('Trades error:', e); }
        }

        async function loadSentiment() {
            try {
                const res = await fetch('/api/sentiment');
                const data = await res.json();
                
                const value = data.sentiment.fear_greed_index;
                const level = data.sentiment.level;
                
                const fgEl = document.getElementById('fgValue');
                fgEl.textContent = value;
                fgEl.className = 'fg-value ' + (value < 30 ? 'fear' : value > 70 ? 'greed' : 'neutral');
                
                document.getElementById('fgLabel').textContent = level;
            } catch (e) { console.error('Sentiment error:', e); }
        }

        async function emergencyStop() {
            if (confirm('⚠️ EMERGENCY STOP\\n\\nThis will halt ALL trading immediately.\\n\\nAre you sure?')) {
                try {
                    await fetch('/api/emergency-stop', { method: 'POST' });
                    location.reload();
                } catch (e) { alert('Failed to stop'); }
            }
        }
    </script>
</body>
</html>
'''


# Mount static files if exists
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
