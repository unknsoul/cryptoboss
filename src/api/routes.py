"""
CryptoBoss REST API Routes
Production-Grade API v11.0 - Professional Trading Dashboard
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Import existing components
try:
    from src.core.storage.database import SQLiteManager
except ImportError:
    try:
        from core.storage.database import SQLiteManager
    except ImportError:
        SQLiteManager = None

# Import v11.0 components
try:
    from src.core import (
        get_decision_store, get_intent_registry,
        get_drawdown_governor, get_slippage_monitor,
        get_recovery_handler, get_safety_metrics,
        get_incident_state_machine, get_operator_control,
        get_risk_guardian, get_capital_governor,
        get_exchange_monitor, get_state_manager
    )
    from src.api.websocket import (
        get_websocket_manager, websocket_endpoint,
        StreamChannel
    )
except ImportError as e:
    logger.warning(f"v11.0 imports unavailable: {e}")
    get_decision_store = None
    get_websocket_manager = None

app = FastAPI(
    title="CryptoBoss API",
    version="11.1.0",
    description="Production-Grade Trading Platform API - FINAL-MAP"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# v11.1: Import environment and authenticity guards
try:
    from src.core.environment_guard import get_environment_guard
    from src.core.data_authenticity import DataSource
    from src.core.decision_narrative import get_narrative_engine
except ImportError:
    get_environment_guard = None
    DataSource = None
    get_narrative_engine = None

def get_env_signature() -> Optional[Dict]:
    """Get current environment signature for API responses."""
    if get_environment_guard is None:
        return None
    try:
        guard = get_environment_guard()
        if guard._initialized:
            return guard.get_signature().to_dict()
    except:
        pass
    return None

def attach_signature(response: Dict) -> Dict:
    """Attach environment signature to any API response."""
    sig = get_env_signature()
    if sig:
        response["_environment"] = sig
    response["_timestamp"] = datetime.utcnow().isoformat()
    return response

# Shared state
trading_state = {
    'mode': 'paper',
    'equity': 10000.0,
    'positions': [],
    'strategies': [],
    'kill_switch_active': False
}


# ============= PYDANTIC MODELS =============

class ModeRequest(BaseModel):
    mode: str

class OperatorActionRequest(BaseModel):
    action: str  # pause, resume, recover
    reason: str
    operator_id: str = "default"


# ============= V11.0 NEW ENDPOINTS =============

@app.get("/api/v11/decisions")
async def get_decisions(
    limit: int = Query(50, ge=1, le=500),
    status: str = Query(None),
    symbol: str = Query(None)
):
    """Get recent trade decisions."""
    try:
        if get_decision_store:
            store = get_decision_store()
            if symbol:
                decisions = store.get_by_symbol(symbol, limit)
            else:
                decisions = store.get_recent(limit)
            return [d.to_dict() for d in decisions]
        return []
    except Exception as e:
        logger.error(f"Failed to get decisions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v11/decisions/{decision_id}")
async def get_decision(decision_id: str):
    """Get a specific decision by ID."""
    try:
        if get_decision_store:
            store = get_decision_store()
            decision = store.get(decision_id)
            if decision:
                return decision.to_dict()
        raise HTTPException(status_code=404, detail="Decision not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get decision: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v11/decisions/stats")
async def get_decision_stats():
    """Get decision statistics."""
    try:
        if get_decision_store:
            store = get_decision_store()
            return store.get_stats()
        return {}
    except Exception as e:
        logger.error(f"Failed to get decision stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v11/intents")
async def get_intents(limit: int = Query(50, ge=1, le=200)):
    """Get trade intents."""
    try:
        if get_intent_registry:
            registry = get_intent_registry()
            intents = registry.get_pending_intents()
            return [i.to_dict() for i in intents[:limit]]
        return []
    except Exception as e:
        logger.error(f"Failed to get intents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v11/intents/stats")
async def get_intent_stats():
    """Get intent registry statistics."""
    try:
        if get_intent_registry:
            registry = get_intent_registry()
            return registry.get_stats()
        return {}
    except Exception as e:
        logger.error(f"Failed to get intent stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v11/risk/state")
async def get_risk_state():
    """Get current risk state."""
    try:
        result = {
            'timestamp': datetime.utcnow().isoformat(),
            'risk_state': 'safe',
            'kill_switch_active': trading_state.get('kill_switch_active', False),
        }
        
        if get_risk_guardian:
            guardian = get_risk_guardian()
            result.update({
                'daily_drawdown': guardian.limits.daily_drawdown_limit,
                'can_trade': guardian.can_trade(),
            })
        
        if get_incident_state_machine:
            ism = get_incident_state_machine()
            result['incident_state'] = ism.get_snapshot().to_dict()
            result['can_open_positions'] = ism.can_open_new_positions()
        
        return result
    except Exception as e:
        logger.error(f"Failed to get risk state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v11/risk/exposure")
async def get_risk_exposure():
    """Get portfolio exposure breakdown."""
    try:
        if get_capital_governor:
            governor = get_capital_governor()
            return governor.get_allocation_status()
        return {}
    except Exception as e:
        logger.error(f"Failed to get exposure: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v11/drawdown")
async def get_drawdown():
    """Get drawdown status across all timeframes."""
    try:
        if get_drawdown_governor:
            governor = get_drawdown_governor()
            return governor.get_status()
        return {
            'current_equity': trading_state.get('equity', 10000),
            'in_defensive_mode': False,
            'states': {}
        }
    except Exception as e:
        logger.error(f"Failed to get drawdown: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v11/drawdown/events")
async def get_drawdown_events(limit: int = Query(50, ge=1, le=200)):
    """Get recent drawdown events."""
    try:
        if get_drawdown_governor:
            governor = get_drawdown_governor()
            return governor.get_events(limit)
        return []
    except Exception as e:
        logger.error(f"Failed to get drawdown events: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v11/slippage")
async def get_slippage(hours: int = Query(24, ge=1, le=168)):
    """Get slippage statistics."""
    try:
        if get_slippage_monitor:
            monitor = get_slippage_monitor()
            return monitor.get_status()
        return {}
    except Exception as e:
        logger.error(f"Failed to get slippage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v11/slippage/records")
async def get_slippage_records(
    limit: int = Query(50, ge=1, le=200),
    symbol: str = Query(None)
):
    """Get recent slippage records."""
    try:
        if get_slippage_monitor:
            monitor = get_slippage_monitor()
            return monitor.get_recent_records(limit, symbol)
        return []
    except Exception as e:
        logger.error(f"Failed to get slippage records: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v11/recovery/status")
async def get_recovery_status():
    """Get exchange recovery handler status."""
    try:
        if get_recovery_handler:
            handler = get_recovery_handler()
            return handler.get_status()
        return {}
    except Exception as e:
        logger.error(f"Failed to get recovery status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v11/recovery/errors")
async def get_recovery_errors(limit: int = Query(50, ge=1, le=200)):
    """Get recent exchange errors."""
    try:
        if get_recovery_handler:
            handler = get_recovery_handler()
            return handler.get_recent_errors(limit)
        return []
    except Exception as e:
        logger.error(f"Failed to get recovery errors: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v11/safety")
async def get_safety_metrics():
    """Get safety metrics."""
    try:
        if get_safety_metrics:
            collector = get_safety_metrics()
            return collector.get_safety_metrics().to_dict()
        return {}
    except Exception as e:
        logger.error(f"Failed to get safety metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v11/operator/state")
async def get_operator_state():
    """Get operator control state."""
    try:
        if get_operator_control:
            operator = get_operator_control()
            return operator.get_state().to_dict()
        return {'is_paused': False, 'is_halted': False}
    except Exception as e:
        logger.error(f"Failed to get operator state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v11/operator/audit")
async def get_operator_audit(limit: int = Query(50, ge=1, le=200)):
    """Get operator action audit log."""
    try:
        if get_operator_control:
            operator = get_operator_control()
            return [log.to_dict() for log in operator.get_audit_log(limit)]
        return []
    except Exception as e:
        logger.error(f"Failed to get operator audit: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v11/operator/action")
async def operator_action(request: OperatorActionRequest):
    """Execute an operator action."""
    try:
        if not get_operator_control:
            raise HTTPException(status_code=503, detail="Operator control unavailable")
        
        operator = get_operator_control()
        
        if request.action == "pause":
            result = operator.pause(request.reason, request.operator_id)
        elif request.action == "resume":
            result = operator.resume(request.reason, request.operator_id)
        elif request.action == "recover":
            result = operator.recover_from_halt(request.reason, request.operator_id)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")
        
        return result.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Operator action failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v11/health")
async def get_comprehensive_health():
    """Get comprehensive system health."""
    try:
        health = {
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'healthy',
            'version': '11.0.0',
            'mode': trading_state.get('mode'),
            'components': {}
        }
        
        # Exchange health
        if get_exchange_monitor:
            monitor = get_exchange_monitor()
            snapshot = monitor.get_snapshot()
            health['components']['exchange'] = {
                'health_level': snapshot.health_level.value,
                'escalation_stage': snapshot.escalation_stage.value,
                'api_latency_ms': snapshot.api_latency_ms,
            }
        
        # Risk health
        if get_incident_state_machine:
            ism = get_incident_state_machine()
            snapshot = ism.get_snapshot()
            health['components']['risk'] = {
                'incident_state': snapshot.state.value,
                'can_trade': ism.can_open_new_positions(),
            }
        
        # Recovery handler
        if get_recovery_handler:
            handler = get_recovery_handler()
            status = handler.get_status()
            health['components']['recovery'] = {
                'in_paper_mode': status.get('in_paper_mode', False),
                'recent_errors': status.get('recent_error_count', 0),
            }
        
        # Determine overall health
        unhealthy = False
        for component, data in health['components'].items():
            if isinstance(data, dict):
                if data.get('health_level') in ['degraded', 'unhealthy']:
                    unhealthy = True
                if data.get('incident_state') in ['incident_freeze', 'halted']:
                    unhealthy = True
        
        health['status'] = 'degraded' if unhealthy else 'healthy'
        
        return health
    except Exception as e:
        logger.error(f"Failed to get health: {e}")
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'error',
            'error': str(e)
        }


# ============= WEBSOCKET V11.0 =============

@app.websocket("/ws/v11/stream")
async def websocket_stream(
    websocket: WebSocket,
    channels: str = Query(None),
    symbols: str = Query(None)
):
    """v11.0 WebSocket streaming endpoint."""
    if websocket_endpoint:
        await websocket_endpoint(websocket, channels, symbols)
    else:
        await websocket.accept()
        await websocket.send_json({'error': 'WebSocket manager not available'})
        await websocket.close()


# ============= LEGACY ENDPOINTS (maintained for compatibility) =============

@app.get("/api/prices")
async def get_prices(timeframe: str = "1h", limit: int = 100):
    """Get historical price data."""
    import numpy as np
    from datetime import timedelta
    
    now = datetime.now()
    prices = []
    base_price = 40000
    
    for i in range(limit):
        timestamp = now - timedelta(hours=limit - i)
        price = base_price + np.random.normal(0, 500)
        
        prices.append({
            'timestamp': timestamp.isoformat(),
            'open': price * (1 + np.random.normal(0, 0.001)),
            'high': price * (1 + abs(np.random.normal(0.002, 0.003))),
            'low': price * (1 - abs(np.random.normal(0.002, 0.003))),
            'close': price,
            'volume': 1000 + np.random.uniform(0, 500)
        })
    
    return prices


@app.get("/api/positions")
async def get_positions():
    """Get open positions."""
    return trading_state.get('positions', [])


@app.get("/api/orders")
async def get_orders(limit: int = 50):
    """Get order history."""
    try:
        if SQLiteManager:
            db = SQLiteManager()
            trades = db.get_recent_trades(limit=limit)
            return trades
        return []
    except Exception as e:
        logger.error(f"Failed to fetch orders: {e}")
        return []


@app.get("/api/performance")
async def get_performance():
    """Get performance metrics."""
    try:
        if SQLiteManager:
            db = SQLiteManager()
            trades = db.get_all_trades()
            
            if not trades:
                return {
                    'total_trades': 0,
                    'total_pnl': 0,
                    'win_rate': 0,
                    'profit_factor': 0
                }
            
            wins = [t for t in trades if t.get('pnl', 0) > 0]
            losses = [t for t in trades if t.get('pnl', 0) < 0]
            
            total_wins = sum(t['pnl'] for t in wins)
            total_losses = abs(sum(t['pnl'] for t in losses))
            
            return {
                'total_trades': len(trades),
                'total_pnl': sum(t.get('pnl', 0) for t in trades),
                'win_rate': (len(wins) / len(trades) * 100) if trades else 0,
                'profit_factor': (total_wins / total_losses) if total_losses > 0 else 0
            }
        return {'total_trades': 0, 'total_pnl': 0, 'win_rate': 0, 'profit_factor': 0}
    except Exception as e:
        logger.error(f"Failed to get performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/risk-metrics")
async def get_risk_metrics():
    """Get risk metrics."""
    try:
        # Use v11.0 components if available
        result = {
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'total_trades': 0,
            'daily_pnl': 0
        }
        
        if get_drawdown_governor:
            governor = get_drawdown_governor()
            status = governor.get_status()
            daily_state = status.get('states', {}).get('daily', {})
            result['max_drawdown'] = -daily_state.get('current_drawdown_pct', 0)
        
        return result
    except Exception as e:
        logger.error(f"Failed to get risk metrics: {e}")
        return {
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'total_trades': 0,
            'daily_pnl': 0
        }


@app.get("/api/strategies")
async def get_strategies():
    """Get available strategies."""
    return trading_state.get('strategies', [
        {
            'name': 'Momentum Strategy',
            'enabled': True,
            'performance': {'win_rate': 65.0, 'pnl': 450.25, 'trades': 82}
        },
        {
            'name': 'Mean Reversion',
            'enabled': True,
            'performance': {'win_rate': 58.5, 'pnl': 320.50, 'trades': 105}
        },
        {
            'name': 'Breakout Strategy',
            'enabled': False,
            'performance': {'win_rate': 52.0, 'pnl': -85.75, 'trades': 45}
        }
    ])


@app.post("/api/mode/paper")
async def switch_to_paper():
    """Switch to paper trading mode."""
    trading_state['mode'] = 'paper'
    
    # Notify via WebSocket if available
    if get_websocket_manager:
        mgr = get_websocket_manager()
        await mgr.broadcast_alert("mode_change", "Switched to paper trading", "info")
    
    logger.info("Switched to paper trading mode")
    return {'success': True, 'mode': 'paper'}


@app.post("/api/mode/live")
async def switch_to_live():
    """Switch to live trading mode (with warnings)."""
    logger.warning("⚠️ Switching to LIVE trading mode")
    trading_state['mode'] = 'live'
    
    # Notify via WebSocket if available
    if get_websocket_manager:
        mgr = get_websocket_manager()
        await mgr.broadcast_alert("mode_change", "Switched to LIVE trading - real capital at risk", "warning")
    
    return {'success': True, 'mode': 'live', 'warning': 'Live trading active - real capital at risk'}


# Health check
@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'mode': trading_state.get('mode'),
        'version': '11.0.0'
    }


# Legacy WebSocket endpoints
@app.websocket("/ws/prices")
async def websocket_prices(websocket: WebSocket):
    """Legacy WebSocket for real-time price updates."""
    await websocket.accept()
    try:
        while True:
            import numpy as np
            price = 40000 + np.random.normal(0, 500)
            
            data = {
                'type': 'price_update',
                'symbol': 'BTCUSDT',
                'price': price,
                'timestamp': datetime.now().isoformat()
            }
            
            await websocket.send_json(data)
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

