"""
CryptoBoss Professional Trading Dashboard
Advanced UI with real-time charts, positions, and trading controls

Run: python -m dashboard.app
Open: http://localhost:8000
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json
import asyncio

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(
    title="CryptoBoss Pro",
    description="Professional Crypto Trading Dashboard",
    version="2.0.0"
)

# Shared trading state
trading_state: Dict[str, Any] = {
    'system_status': 'running',
    'uptime_start': datetime.now(),
    'equity': 10000.0,
    'initial_capital': 10000.0,
    'positions': [],
    'trades': [],
    'signals': [],
    'candles': [],
    'orderbook': {'bids': [], 'asks': []},
    'metrics': {
        'total_return': 0.0,
        'win_rate': 0.5,
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

# WebSocket connections for real-time updates
active_connections: List[WebSocket] = []


# ==================== API ENDPOINTS ====================

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the professional trading dashboard"""
    return get_professional_dashboard_html()


@app.get("/api/status")
async def get_status():
    uptime = datetime.now() - trading_state['uptime_start']
    return {
        'status': trading_state['system_status'],
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


@app.get("/api/candles")
async def get_candles():
    return {'candles': trading_state.get('candles', [])}


@app.get("/api/orderbook")
async def get_orderbook():
    return trading_state.get('orderbook', {'bids': [], 'asks': [], 'spread': 0})


@app.get("/api/sentiment")
async def get_sentiment():
    return {'sentiment': trading_state['sentiment']}


@app.get("/api/equity_history")
async def get_equity_history():
    import random
    history = []
    equity = trading_state['initial_capital']
    for i in range(100):
        change = random.uniform(-50, 70)
        equity += change
        history.append({
            'time': int((datetime.now() - timedelta(hours=100-i)).timestamp()),
            'equity': round(max(equity, 5000), 2)
        })
    trading_state['equity'] = equity
    return {'history': history}


# POST endpoints for updates
@app.post("/api/update/equity")
async def update_equity(request: Request):
    data = await request.json()
    if 'equity' in data:
        trading_state['equity'] = float(data['equity'])
    return {'status': 'ok'}


@app.post("/api/update/signal")
async def add_signal(request: Request):
    data = await request.json()
    trading_state['signals'].append({**data, 'timestamp': datetime.now().isoformat()})
    if len(trading_state['signals']) > 100:
        trading_state['signals'] = trading_state['signals'][-100:]
    return {'status': 'ok'}


@app.post("/api/update/trade")
async def add_trade(request: Request):
    data = await request.json()
    trading_state['trades'].append(data)
    return {'status': 'ok'}


@app.post("/api/update/metrics")
async def update_metrics(request: Request):
    data = await request.json()
    trading_state['metrics'].update(data)
    return {'status': 'ok'}


# ==================== PROFESSIONAL DASHBOARD HTML ====================

def get_professional_dashboard_html() -> str:
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CryptoBoss Pro Trading</title>
    <script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #0a0e17;
            --bg-card: rgba(16, 23, 39, 0.8);
            --bg-card-hover: rgba(26, 35, 55, 0.9);
            --glass: rgba(255, 255, 255, 0.03);
            --glass-border: rgba(255, 255, 255, 0.08);
            --text-primary: #ffffff;
            --text-secondary: #64748b;
            --text-muted: #475569;
            --accent-green: #10b981;
            --accent-green-glow: rgba(16, 185, 129, 0.3);
            --accent-red: #ef4444;
            --accent-red-glow: rgba(239, 68, 68, 0.3);
            --accent-blue: #3b82f6;
            --accent-purple: #8b5cf6;
            --accent-yellow: #f59e0b;
            --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --gradient-success: linear-gradient(135deg, #10b981 0%, #059669 100%);
            --gradient-danger: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* Animated background */
        .bg-animation {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            background: 
                radial-gradient(ellipse at 20% 80%, rgba(99, 102, 241, 0.15) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 20%, rgba(139, 92, 246, 0.1) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 50%, rgba(16, 185, 129, 0.05) 0%, transparent 70%);
        }

        /* Header */
        .header {
            background: var(--bg-card);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid var(--glass-border);
            padding: 16px 32px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .logo-icon {
            width: 40px;
            height: 40px;
            background: var(--gradient-primary);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
        }

        .logo-text {
            font-size: 22px;
            font-weight: 700;
            background: linear-gradient(135deg, #fff 0%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .header-stats {
            display: flex;
            gap: 32px;
        }

        .header-stat {
            text-align: right;
        }

        .header-stat-label {
            font-size: 11px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .header-stat-value {
            font-size: 18px;
            font-weight: 600;
        }

        .status-live {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 20px;
            font-size: 13px;
            color: var(--accent-green);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background: var(--accent-green);
            border-radius: 50%;
            animation: pulse 2s infinite;
            box-shadow: 0 0 10px var(--accent-green-glow);
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.7; transform: scale(1.1); }
        }

        /* Main Layout */
        .main {
            display: grid;
            grid-template-columns: 1fr 320px;
            grid-template-rows: auto auto;
            gap: 20px;
            padding: 20px 32px;
            max-width: 1800px;
            margin: 0 auto;
        }

        /* Glass Card */
        .card {
            background: var(--bg-card);
            backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 20px;
            transition: all 0.3s ease;
        }

        .card:hover {
            border-color: rgba(255, 255, 255, 0.15);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }

        .card-title {
            font-size: 14px;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* Chart Area */
        .chart-section {
            grid-column: 1;
            grid-row: 1;
        }

        #chart-container {
            height: 400px;
            border-radius: 12px;
            overflow: hidden;
        }

        /* Order Book */
        .orderbook-section {
            grid-column: 2;
            grid-row: 1;
        }

        .orderbook {
            display: flex;
            flex-direction: column;
            gap: 2px;
            font-family: 'SF Mono', 'Consolas', monospace;
            font-size: 12px;
        }

        .orderbook-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            padding: 4px 8px;
            border-radius: 4px;
            position: relative;
        }

        .orderbook-row.ask {
            background: linear-gradient(90deg, transparent 0%, rgba(239, 68, 68, 0.15) 100%);
        }

        .orderbook-row.bid {
            background: linear-gradient(90deg, rgba(16, 185, 129, 0.15) 0%, transparent 100%);
        }

        .orderbook-price { font-weight: 500; }
        .orderbook-price.ask { color: var(--accent-red); }
        .orderbook-price.bid { color: var(--accent-green); }
        .orderbook-qty { color: var(--text-secondary); text-align: right; }

        .spread-indicator {
            text-align: center;
            padding: 8px;
            color: var(--text-muted);
            font-size: 11px;
            border-top: 1px solid var(--glass-border);
            border-bottom: 1px solid var(--glass-border);
            margin: 4px 0;
        }

        /* Metrics Grid */
        .metrics-section {
            grid-column: 1 / -1;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 16px;
        }

        .metric-card {
            background: var(--glass);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 16px;
            text-align: center;
            transition: all 0.3s ease;
        }

        .metric-card:hover {
            transform: translateY(-2px);
            border-color: var(--accent-blue);
            box-shadow: 0 4px 20px rgba(59, 130, 246, 0.2);
        }

        .metric-icon {
            font-size: 24px;
            margin-bottom: 8px;
        }

        .metric-value {
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 4px;
        }

        .metric-value.positive { color: var(--accent-green); }
        .metric-value.negative { color: var(--accent-red); }

        .metric-label {
            font-size: 11px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Bottom Section */
        .bottom-section {
            grid-column: 1 / -1;
            display: grid;
            grid-template-columns: 1fr 1fr 300px;
            gap: 20px;
        }

        /* Trades Table */
        .trades-table {
            width: 100%;
            border-collapse: collapse;
        }

        .trades-table th {
            text-align: left;
            padding: 12px 8px;
            font-size: 11px;
            font-weight: 500;
            color: var(--text-secondary);
            text-transform: uppercase;
            border-bottom: 1px solid var(--glass-border);
        }

        .trades-table td {
            padding: 10px 8px;
            font-size: 13px;
            border-bottom: 1px solid var(--glass-border);
        }

        .trades-table tr:hover td {
            background: var(--glass);
        }

        .trade-side {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }

        .trade-side.long {
            background: rgba(16, 185, 129, 0.2);
            color: var(--accent-green);
        }

        .trade-side.short {
            background: rgba(239, 68, 68, 0.2);
            color: var(--accent-red);
        }

        /* P&L Colors - Green for profit, Red for loss */
        .pnl-positive,
        td.pnl-positive {
            color: #10b981 !important;
            font-weight: 700;
            text-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
        }

        .pnl-negative,
        td.pnl-negative {
            color: #ef4444 !important;
            font-weight: 700;
            text-shadow: 0 0 10px rgba(239, 68, 68, 0.5);
        }

        /* Pagination Controls */
        .pagination {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 16px;
            padding: 12px;
            border-top: 1px solid var(--glass-border);
            margin-top: 8px;
        }

        .page-btn {
            background: var(--glass);
            border: 1px solid var(--glass-border);
            color: var(--text-primary);
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s;
        }

        .page-btn:hover {
            background: var(--accent-purple);
            border-color: var(--accent-purple);
        }

        .page-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .page-info {
            color: var(--text-secondary);
            font-size: 13px;
        }

        /* Signals Feed */
        .signals-feed {
            max-height: 300px;
            overflow-y: auto;
        }

        .signal-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px;
            border-bottom: 1px solid var(--glass-border);
        }

        .signal-icon {
            width: 36px;
            height: 36px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
        }

        .signal-icon.buy {
            background: rgba(16, 185, 129, 0.2);
        }

        .signal-icon.sell {
            background: rgba(239, 68, 68, 0.2);
        }

        .signal-info { flex: 1; }

        .signal-title {
            font-size: 13px;
            font-weight: 500;
        }

        .signal-time {
            font-size: 11px;
            color: var(--text-secondary);
        }

        .signal-confidence {
            font-size: 12px;
            font-weight: 600;
            color: var(--accent-blue);
        }

        /* Fear & Greed Gauge */
        .fg-gauge {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .fg-value {
            font-size: 56px;
            font-weight: 700;
            line-height: 1;
        }

        .fg-value.fear { color: var(--accent-red); }
        .fg-value.neutral { color: var(--accent-yellow); }
        .fg-value.greed { color: var(--accent-green); }

        .fg-label {
            font-size: 14px;
            color: var(--text-secondary);
            margin-top: 8px;
        }

        .fg-bar {
            width: 100%;
            height: 8px;
            background: linear-gradient(90deg, var(--accent-red), var(--accent-yellow), var(--accent-green));
            border-radius: 4px;
            margin-top: 16px;
            position: relative;
        }

        .fg-marker {
            position: absolute;
            top: -4px;
            width: 16px;
            height: 16px;
            background: white;
            border-radius: 50%;
            transform: translateX(-50%);
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
        }

        ::-webkit-scrollbar-track {
            background: var(--bg-dark);
        }

        ::-webkit-scrollbar-thumb {
            background: var(--text-muted);
            border-radius: 3px;
        }

        @media (max-width: 1200px) {
            .main {
                grid-template-columns: 1fr;
            }
            .metrics-grid {
                grid-template-columns: repeat(3, 1fr);
            }
            .bottom-section {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="bg-animation"></div>

    <header class="header">
        <div class="logo">
            <div class="logo-icon">CB</div>
            <span class="logo-text">CryptoBoss Pro</span>
        </div>
        <div class="header-stats">
            <div class="header-stat">
                <div class="header-stat-label">BTC/USDT</div>
                <div class="header-stat-value" id="current-price">$43,250.00</div>
            </div>
            <div class="header-stat">
                <div class="header-stat-label">24h Change</div>
                <div class="header-stat-value positive" id="price-change">+2.45%</div>
            </div>
            <div class="header-stat">
                <div class="header-stat-label">Portfolio</div>
                <div class="header-stat-value" id="portfolio-value">$10,000.00</div>
            </div>
        </div>
        <div class="status-live">
            <div class="status-dot"></div>
            <span id="status-text">Live Trading</span>
        </div>
    </header>

    <main class="main">
        <!-- Chart Section -->
        <section class="card chart-section">
            <div class="card-title">
                <span>BTCUSDT</span>
                <span style="color: var(--text-muted); font-weight: normal;">1H</span>
            </div>
            <div id="chart-container"></div>
        </section>

        <!-- Order Book -->
        <section class="card orderbook-section">
            <div class="card-title">Order Book</div>
            <div class="orderbook" id="orderbook">
                <!-- Populated by JS -->
            </div>
        </section>

        <!-- Metrics -->
        <section class="metrics-section">
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-icon">&#128176;</div>
                    <div class="metric-value positive" id="total-return">+0.00%</div>
                    <div class="metric-label">Total Return</div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon">&#127919;</div>
                    <div class="metric-value" id="win-rate">0%</div>
                    <div class="metric-label">Win Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon">&#128200;</div>
                    <div class="metric-value" id="sharpe">0.00</div>
                    <div class="metric-label">Sharpe Ratio</div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon">&#128201;</div>
                    <div class="metric-value negative" id="max-dd">0.00%</div>
                    <div class="metric-label">Max Drawdown</div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon">&#128202;</div>
                    <div class="metric-value" id="profit-factor">0.00</div>
                    <div class="metric-label">Profit Factor</div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon">&#128176;</div>
                    <div class="metric-value" id="num-trades">0</div>
                    <div class="metric-label">Total Trades</div>
                </div>
            </div>
        </section>

        <!-- Bottom Section -->
        <section class="bottom-section">
            <!-- Recent Trades -->
            <div class="card">
                <div class="card-title">Recent Trades</div>
                <table class="trades-table">
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Side</th>
                            <th>Price</th>
                            <th>P&L</th>
                        </tr>
                    </thead>
                    <tbody id="trades-body">
                        <tr><td colspan="4" style="text-align:center;color:var(--text-muted)">No trades yet</td></tr>
                    </tbody>
                </table>
                <div class="pagination" id="trades-pagination">
                    <button class="page-btn" id="prev-page" onclick="changePage(-1)">← Prev</button>
                    <span class="page-info" id="page-info">Page 1 of 1</span>
                    <button class="page-btn" id="next-page" onclick="changePage(1)">Next →</button>
                </div>
            </div>

            <!-- Signal Feed -->
            <div class="card">
                <div class="card-title">Signal Feed</div>
                <div class="signals-feed" id="signals-feed">
                    <div class="signal-item">
                        <div class="signal-icon buy">&#8593;</div>
                        <div class="signal-info">
                            <div class="signal-title">LONG Signal Generated</div>
                            <div class="signal-time">Waiting for signals...</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Fear & Greed -->
            <div class="card">
                <div class="card-title">Market Sentiment</div>
                <div class="fg-gauge">
                    <div class="fg-value neutral" id="fg-value">50</div>
                    <div class="fg-label" id="fg-label">Neutral</div>
                    <div class="fg-bar">
                        <div class="fg-marker" id="fg-marker" style="left: 50%"></div>
                    </div>
                </div>
            </div>
        </section>
    </main>

    <script>
        // Initialize TradingView Chart
        const chartContainer = document.getElementById('chart-container');
        const chart = LightweightCharts.createChart(chartContainer, {
            layout: {
                background: { type: 'solid', color: 'transparent' },
                textColor: '#64748b',
            },
            grid: {
                vertLines: { color: 'rgba(255, 255, 255, 0.03)' },
                horzLines: { color: 'rgba(255, 255, 255, 0.03)' },
            },
            crosshair: {
                mode: LightweightCharts.CrosshairMode.Normal,
            },
            rightPriceScale: {
                borderColor: 'rgba(255, 255, 255, 0.1)',
            },
            timeScale: {
                borderColor: 'rgba(255, 255, 255, 0.1)',
                timeVisible: true,
            },
        });

        const candleSeries = chart.addCandlestickSeries({
            upColor: '#10b981',
            downColor: '#ef4444',
            borderUpColor: '#10b981',
            borderDownColor: '#ef4444',
            wickUpColor: '#10b981',
            wickDownColor: '#ef4444',
        });

        // Resize chart on window resize
        window.addEventListener('resize', () => {
            chart.applyOptions({ width: chartContainer.clientWidth });
        });

        // Update functions
        async function updateChart() {
            try {
                const res = await fetch('/api/candles');
                const data = await res.json();
                candleSeries.setData(data.candles);
                
                if (data.candles.length > 0) {
                    const lastCandle = data.candles[data.candles.length - 1];
                    document.getElementById('current-price').textContent = 
                        '$' + lastCandle.close.toLocaleString('en-US', {minimumFractionDigits: 2});
                }
            } catch (e) {
                console.error('Chart update error:', e);
            }
        }

        async function updateOrderbook() {
            try {
                const res = await fetch('/api/orderbook');
                const data = await res.json();
                
                let html = '';
                
                // Asks (reversed for display)
                data.asks.slice().reverse().forEach(ask => {
                    html += `<div class="orderbook-row ask">
                        <span class="orderbook-price ask">${ask[0].toFixed(2)}</span>
                        <span class="orderbook-qty">${ask[1].toFixed(4)}</span>
                    </div>`;
                });
                
                html += `<div class="spread-indicator">Spread: $${data.spread.toFixed(2)}</div>`;
                
                // Bids
                data.bids.forEach(bid => {
                    html += `<div class="orderbook-row bid">
                        <span class="orderbook-price bid">${bid[0].toFixed(2)}</span>
                        <span class="orderbook-qty">${bid[1].toFixed(4)}</span>
                    </div>`;
                });
                
                document.getElementById('orderbook').innerHTML = html;
            } catch (e) {
                console.error('Orderbook update error:', e);
            }
        }

        async function updateMetrics() {
            try {
                const res = await fetch('/api/performance');
                const data = await res.json();
                
                document.getElementById('portfolio-value').textContent = 
                    '$' + data.equity.toLocaleString('en-US', {minimumFractionDigits: 2});
                
                const returnEl = document.getElementById('total-return');
                returnEl.textContent = (data.total_return_pct >= 0 ? '+' : '') + data.total_return_pct.toFixed(2) + '%';
                returnEl.className = 'metric-value ' + (data.total_return_pct >= 0 ? 'positive' : 'negative');
                
                document.getElementById('win-rate').textContent = (data.metrics.win_rate * 100).toFixed(1) + '%';
                document.getElementById('sharpe').textContent = data.metrics.sharpe_ratio.toFixed(2);
                document.getElementById('max-dd').textContent = (data.metrics.max_drawdown * 100).toFixed(2) + '%';
                document.getElementById('profit-factor').textContent = data.metrics.profit_factor.toFixed(2);
                document.getElementById('num-trades').textContent = data.metrics.num_trades;
            } catch (e) {
                console.error('Metrics update error:', e);
            }
        }

        async function updateSentiment() {
            try {
                const res = await fetch('/api/sentiment');
                const data = await res.json();
                const fg = data.sentiment.fear_greed_index;
                
                const fgValue = document.getElementById('fg-value');
                const fgLabel = document.getElementById('fg-label');
                const fgMarker = document.getElementById('fg-marker');
                
                fgValue.textContent = fg;
                fgMarker.style.left = fg + '%';
                
                if (fg < 35) {
                    fgValue.className = 'fg-value fear';
                    fgLabel.textContent = 'Extreme Fear';
                } else if (fg < 45) {
                    fgValue.className = 'fg-value fear';
                    fgLabel.textContent = 'Fear';
                } else if (fg <= 55) {
                    fgValue.className = 'fg-value neutral';
                    fgLabel.textContent = 'Neutral';
                } else if (fg < 75) {
                    fgValue.className = 'fg-value greed';
                    fgLabel.textContent = 'Greed';
                } else {
                    fgValue.className = 'fg-value greed';
                    fgLabel.textContent = 'Extreme Greed';
                }
            } catch (e) {
                console.error('Sentiment update error:', e);
            }
        }

        // Pagination state
        let currentPage = 1;
        const tradesPerPage = 15;
        let allTrades = [];

        function changePage(delta) {
            const totalPages = Math.ceil(allTrades.length / tradesPerPage);
            const newPage = currentPage + delta;
            if (newPage >= 1 && newPage <= totalPages) {
                currentPage = newPage;
                renderTrades();
            }
        }

        function renderTrades() {
            const tradesBody = document.getElementById('trades-body');
            const totalPages = Math.max(1, Math.ceil(allTrades.length / tradesPerPage));
            
            // Update page info
            document.getElementById('page-info').textContent = `Page ${currentPage} of ${totalPages}`;
            document.getElementById('prev-page').disabled = currentPage <= 1;
            document.getElementById('next-page').disabled = currentPage >= totalPages;
            
            if (allTrades.length === 0) {
                tradesBody.innerHTML = '<tr><td colspan="4" style="text-align:center;color:var(--text-muted)">No trades yet</td></tr>';
                return;
            }
            
            // Get trades for current page
            const startIdx = (currentPage - 1) * tradesPerPage;
            const endIdx = startIdx + tradesPerPage;
            const pageTrades = allTrades.slice(startIdx, endIdx);
            
            let html = '';
            pageTrades.forEach(item => {
                if (item.isLive) {
                    // Live position
                    const pnl = item.pnl || 0;
                    const pnlClass = pnl >= 0 ? 'pnl-positive' : 'pnl-negative';
                    const pnlSign = pnl >= 0 ? '+' : '';
                    const sideClass = item.side === 'LONG' ? 'long' : 'short';
                    
                    html += `<tr style="background: rgba(99, 102, 241, 0.1);">
                        <td><span style="color:#a78bfa;">● LIVE</span></td>
                        <td><span class="trade-side ${sideClass}">${item.side}</span></td>
                        <td>$${item.price?.toLocaleString('en-US', {minimumFractionDigits: 2}) || 'N/A'}</td>
                        <td class="${pnlClass}">${pnlSign}$${Math.abs(pnl).toFixed(2)}</td>
                    </tr>`;
                } else {
                    // Closed trade
                    const pnl = item.pnl || 0;
                    const pnlClass = pnl >= 0 ? 'pnl-positive' : 'pnl-negative';
                    const pnlSign = pnl >= 0 ? '+' : '';
                    const sideClass = item.side === 'LONG' ? 'long' : 'short';
                    
                    html += `<tr>
                        <td>${item.time}</td>
                        <td><span class="trade-side ${sideClass}">${item.side}</span></td>
                        <td>$${item.price?.toLocaleString('en-US', {minimumFractionDigits: 2}) || 'N/A'}</td>
                        <td class="${pnlClass}">${pnlSign}$${Math.abs(pnl).toFixed(2)}</td>
                    </tr>`;
                }
            });
            
            tradesBody.innerHTML = html;
        }

        // Update trades table - shows both open positions and closed trades with pagination
        async function updateTrades() {
            try {
                // Fetch both trades and positions
                const [tradesRes, positionsRes] = await Promise.all([
                    fetch('/api/trades?limit=100'),
                    fetch('/api/positions')
                ]);
                const tradesData = await tradesRes.json();
                const positionsData = await positionsRes.json();
                
                allTrades = [];
                
                // First add OPEN positions (live trades)
                if (positionsData.positions && positionsData.positions.length > 0) {
                    positionsData.positions.forEach(pos => {
                        allTrades.push({
                            isLive: true,
                            side: pos.side,
                            price: pos.entry_price,
                            pnl: pos.unrealized_pnl || 0,
                            time: 'LIVE'
                        });
                    });
                }
                
                // Then add closed trades (reversed for newest first)
                if (tradesData.trades && tradesData.trades.length > 0) {
                    tradesData.trades.slice().reverse().forEach(trade => {
                        let timeStr = 'N/A';
                        if (trade.timestamp) {
                            const date = new Date(trade.timestamp);
                            timeStr = date.toLocaleTimeString('en-US', {hour: '2-digit', minute:'2-digit'});
                        }
                        
                        allTrades.push({
                            isLive: false,
                            side: trade.side,
                            price: trade.exit || trade.entry,
                            pnl: trade.pnl || 0,
                            time: timeStr
                        });
                    });
                }
                
                renderTrades();
            } catch (e) {
                console.error('Trades update error:', e);
            }
        }

        // Update signals feed
        async function updateSignals() {
            try {
                const res = await fetch('/api/signals?limit=10');
                const data = await res.json();
                const signalsFeed = document.getElementById('signals-feed');
                
                if (data.signals && data.signals.length > 0) {
                    let html = '';
                    data.signals.slice().reverse().forEach(signal => {
                        const action = signal.action || 'HOLD';
                        const iconClass = action === 'LONG' || action === 'BUY' ? 'buy' : 'sell';
                        const icon = action === 'LONG' || action === 'BUY' ? '&#8593;' : '&#8595;';
                        const confidence = signal.confidence || 0;
                        
                        // Format timestamp
                        let timeStr = 'Just now';
                        if (signal.timestamp) {
                            const date = new Date(signal.timestamp);
                            timeStr = date.toLocaleTimeString('en-US', {hour: '2-digit', minute:'2-digit'});
                        }
                        
                        // Get reasons
                        const reasons = signal.reasons || [];
                        const reasonStr = reasons.slice(0, 2).join(', ') || 'Signal generated';
                        
                        html += `<div class="signal-item">
                            <div class="signal-icon ${iconClass}">${icon}</div>
                            <div class="signal-info">
                                <div class="signal-title">${action} Signal @ $${signal.price?.toLocaleString('en-US', {minimumFractionDigits: 2}) || 'N/A'}</div>
                                <div class="signal-time">${timeStr} - ${reasonStr.substring(0, 50)}${reasonStr.length > 50 ? '...' : ''}</div>
                            </div>
                            <div class="signal-confidence">${(confidence * 100).toFixed(0)}%</div>
                        </div>`;
                    });
                    signalsFeed.innerHTML = html;
                } else {
                    signalsFeed.innerHTML = `<div class="signal-item">
                        <div class="signal-icon buy">&#8593;</div>
                        <div class="signal-info">
                            <div class="signal-title">Waiting for signals...</div>
                            <div class="signal-time">Bot is analyzing market</div>
                        </div>
                    </div>`;
                }
            } catch (e) {
                console.error('Signals update error:', e);
            }
        }

        // Initial load
        updateChart();
        updateOrderbook();
        updateMetrics();
        updateSentiment();
        updateTrades();
        updateSignals();

        // Periodic updates
        setInterval(updateChart, 30000);
        setInterval(updateOrderbook, 5000);
        setInterval(updateMetrics, 5000);
        setInterval(updateSentiment, 60000);
        setInterval(updateTrades, 5000);
        setInterval(updateSignals, 5000);
    </script>
</body>
</html>'''


def run_dashboard(host: str = "0.0.0.0", port: int = 8000):
    print(f"Starting CryptoBoss Pro Dashboard at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_dashboard()
