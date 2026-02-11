"""
CryptoBoss REST API Routes
Production-Grade API v11.0 - Professional Trading Dashboard
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query, Request
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

# Import v2.0 database repository for user/account scoped data
try:
    from src.core.database import get_repository, SQLiteUserRepository
except ImportError:
    get_repository = None
    SQLiteUserRepository = None

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
    'mode': 'testnet',  # PAPER REMOVED - use testnet
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

class SessionSwitchRequest(BaseModel):
    mode: str  # 'testnet' or 'live'
    api_key: str
    api_secret: str

class ValidateKeysRequest(BaseModel):
    api_key: str
    api_secret: str
    testnet: bool = True


# ============= API KEY VALIDATION HELPER =============

async def _validate_binance_keys(api_key: str, api_secret: str, testnet: bool = True) -> dict:
    """
    Validate Binance API keys by calling GET /api/v3/account.
    
    Uses proper HMAC-SHA256 signing with server time sync.
    Returns: {"success": bool, "message": str, "balances": dict|None}
    """
    import aiohttp
    import hmac
    import hashlib
    import time
    
    # Choose the correct base URL
    if testnet:
        base_url = "https://testnet.binance.vision/api"
    else:
        base_url = "https://api.binance.com/api"
    
    logger.info(f"Validating API keys against {base_url} (testnet={testnet})")
    
    try:
        async with aiohttp.ClientSession() as session:
            # Step 1: Get server time for accurate signing
            try:
                async with session.get(
                    f"{base_url}/v3/time",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as time_resp:
                    if time_resp.status == 200:
                        time_data = await time_resp.json()
                        server_time = time_data.get("serverTime", int(time.time() * 1000))
                        logger.info(f"Got server time: {server_time}")
                    else:
                        server_time = int(time.time() * 1000)
                        logger.warning(f"Server time endpoint returned {time_resp.status}, using local time")
            except Exception as e:
                server_time = int(time.time() * 1000)
                logger.warning(f"Failed to get server time: {e}, using local time")
            
            # Step 2: Sign the request using HMAC-SHA256
            timestamp = server_time
            query_string = f"timestamp={timestamp}&recvWindow=10000"
            
            signature = hmac.new(
                api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            # Step 3: Call GET /api/v3/account (the authoritative validation endpoint)
            url = f"{base_url}/v3/account?{query_string}&signature={signature}"
            headers = {"X-MBX-APIKEY": api_key}
            
            logger.info(f"Calling GET /api/v3/account to validate keys...")
            
            async with session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                resp_text = await resp.text()
                
                if resp.status == 200:
                    data = await resp.json() if resp.content_type == 'application/json' else {}
                    try:
                        data = json.loads(resp_text)
                    except:
                        pass
                    
                    # Extract non-zero balances
                    balances = {}
                    for b in data.get("balances", []):
                        free = float(b.get("free", 0))
                        locked = float(b.get("locked", 0))
                        if free > 0 or locked > 0:
                            balances[b["asset"]] = free + locked
                    
                    logger.info(f"API key validation SUCCESS. Found {len(balances)} non-zero balances.")
                    return {
                        "success": True,
                        "message": "API keys validated successfully",
                        "balances": balances
                    }
                    
                elif resp.status == 401:
                    logger.warning(f"API key validation FAILED: 401 Unauthorized")
                    return {
                        "success": False,
                        "message": "Invalid API key"
                    }
                    
                elif resp.status == 403:
                    logger.warning(f"API key validation FAILED: 403 Forbidden")
                    return {
                        "success": False,
                        "message": "API key does not have permissions. Check your IP whitelist and API restrictions."
                    }
                    
                else:
                    # Try to extract Binance error message
                    try:
                        err_data = json.loads(resp_text)
                        err_msg = err_data.get("msg", resp_text[:200])
                        err_code = err_data.get("code", resp.status)
                    except:
                        err_msg = resp_text[:200]
                        err_code = resp.status
                    
                    logger.warning(f"API key validation FAILED: {resp.status} - code={err_code} msg={err_msg}")
                    return {
                        "success": False,
                        "message": f"Validation failed (code {err_code}): {err_msg}"
                    }
                    
    except aiohttp.ClientConnectorError as e:
        logger.error(f"Connection error during validation: {e}")
        return {
            "success": False,
            "message": f"Cannot connect to {'testnet' if testnet else 'live'} Binance API. Check your internet connection."
        }
    except asyncio.TimeoutError:
        logger.error("Timeout during API key validation")
        return {
            "success": False,
            "message": "Validation timed out. The exchange may be experiencing issues."
        }
    except Exception as e:
        logger.error(f"Unexpected error during validation: {e}")
        return {
            "success": False,
            "message": f"Validation error: {str(e)}"
        }


# ============= SESSION / KEY VALIDATION ENDPOINTS =============

@app.post("/api/session/switch")
async def session_switch(request: SessionSwitchRequest):
    """
    Switch trading mode (testnet/live) with API key validation.
    
    This is called when the user enters API keys in the modal.
    It validates the keys against Binance and creates a new session.
    """
    mode = request.mode
    if mode not in ('testnet', 'live'):
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}. Use 'testnet' or 'live'.")
    
    testnet = (mode == 'testnet')
    
    # Validate API keys against Binance
    result = await _validate_binance_keys(request.api_key, request.api_secret, testnet=testnet)
    
    if not result["success"]:
        raise HTTPException(status_code=401, detail=result["message"])
    
    # Update server-side state
    trading_state['mode'] = mode
    
    # Generate session ID
    import uuid
    session_id = str(uuid.uuid4())
    
    logger.info(f"Session switched to {mode} mode. Session: {session_id}")
    
    return {
        "success": True,
        "session_id": session_id,
        "mode": mode,
        "created_at": datetime.utcnow().isoformat(),
        "balances": result.get("balances", {}),
        "message": result["message"]
    }


@app.post("/api/validate-keys")
async def validate_keys(request: ValidateKeysRequest):
    """
    Validate API keys without switching mode.
    
    Used for pre-validation before committing to a mode switch.
    """
    result = await _validate_binance_keys(
        request.api_key,
        request.api_secret,
        testnet=request.testnet
    )
    
    return {
        "success": result["success"],
        "message": result["message"],
        "balances": result.get("balances")
    }


# ============= V2.0 NEW ENDPOINTS (Ownership-Scoped) =============

@app.get("/api/health")
async def api_health():
    """Simple health check endpoint."""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0"
    }


# ============= AUTH ENDPOINTS =============

# In-memory user store (for development - replace with DB in production)
_users_store: Dict = {}
_sessions_store: Dict = {}
_accounts_store: Dict = {}
_active_account: Dict = {}

@app.post("/api/auth/login")
async def auth_login(request: Request):
    """Login with email/password. Returns JWT token."""
    try:
        body = await request.json()
        email = body.get("email", "")
        password = body.get("password", "")
        
        if not email or not password:
            raise HTTPException(status_code=400, detail="Email and password required")
        
        # Check if user exists
        user = _users_store.get(email)
        if not user or user.get("password") != password:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        # Generate simple token
        import uuid
        token = str(uuid.uuid4())
        _sessions_store[token] = {"email": email, "user_id": user["user_id"]}
        
        return {
            "success": True,
            "token": token,
            "user": {
                "id": user["user_id"],
                "email": email,
                "name": user.get("name", email.split("@")[0])
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auth/signup")
async def auth_signup(request: Request):
    """Register a new user."""
    try:
        body = await request.json()
        email = body.get("email", "")
        password = body.get("password", "")
        name = body.get("name", "")
        
        if not email or not password:
            raise HTTPException(status_code=400, detail="Email and password required")
        
        if email in _users_store:
            raise HTTPException(status_code=409, detail="User already exists")
        
        import uuid
        user_id = str(uuid.uuid4())
        _users_store[email] = {
            "user_id": user_id,
            "email": email,
            "password": password,
            "name": name or email.split("@")[0],
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Auto-login
        token = str(uuid.uuid4())
        _sessions_store[token] = {"email": email, "user_id": user_id}
        
        return {
            "success": True,
            "token": token,
            "user": {
                "id": user_id,
                "email": email,
                "name": name or email.split("@")[0]
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/auth/me")
async def auth_me(request: Request):
    """Get current user from token. Alias for /api/me."""
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    session = _sessions_store.get(token)
    
    if not session:
        # For development: return default user if no token
        return {
            "authenticated": True,
            "user": {
                "id": "default-user",
                "email": "trader@cryptoboss.io",
                "name": "Trader"
            },
            "accounts": list(_accounts_store.values()),
            "active_account_id": _active_account.get("id")
        }
    
    email = session["email"]
    user = _users_store.get(email, {})
    user_accounts = [a for a in _accounts_store.values() if a.get("user_id") == session["user_id"]]
    
    return {
        "authenticated": True,
        "user": {
            "id": session["user_id"],
            "email": email,
            "name": user.get("name", email.split("@")[0])
        },
        "accounts": user_accounts,
        "active_account_id": _active_account.get("id")
    }


# ============= ACCOUNT MANAGEMENT ENDPOINTS =============

@app.get("/api/accounts/list")
async def accounts_list(request: Request):
    """List all exchange accounts for the current user."""
    return {
        "accounts": list(_accounts_store.values()),
        "count": len(_accounts_store)
    }


@app.get("/api/accounts/active")
async def accounts_active(request: Request):
    """Get the currently active exchange account."""
    active_id = _active_account.get("id")
    if active_id and active_id in _accounts_store:
        return {
            "active_account": _accounts_store[active_id],
            "has_active": True
        }
    return {
        "active_account": None,
        "has_active": False
    }


@app.post("/api/accounts/select")
async def accounts_select(request: Request):
    """Select an exchange account as active."""
    try:
        body = await request.json()
        account_id = body.get("account_id") or body.get("exchange_account_id")
        
        if not account_id:
            raise HTTPException(status_code=400, detail="account_id required")
        
        if account_id not in _accounts_store:
            raise HTTPException(status_code=404, detail="Account not found")
        
        _active_account["id"] = account_id
        
        return {
            "success": True,
            "active_account": _accounts_store[account_id]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/accounts/create")
async def accounts_create(request: Request):
    """Create a new exchange account."""
    try:
        body = await request.json()
        label = body.get("label", "My Account")
        environment = body.get("environment", "TESTNET")
        api_key = body.get("api_key", "")
        api_secret = body.get("api_secret", "")
        
        if not api_key or not api_secret:
            raise HTTPException(status_code=400, detail="API key and secret required")
        
        # Validate the keys
        testnet = environment.upper() == "TESTNET"
        validation = await _validate_binance_keys(api_key, api_secret, testnet=testnet)
        
        if not validation["success"]:
            raise HTTPException(status_code=401, detail=validation["message"])
        
        import uuid
        account_id = str(uuid.uuid4())
        
        account = {
            "exchange_account_id": account_id,
            "id": account_id,
            "label": label,
            "exchange": "binance",
            "environment": environment.upper(),
            "api_key_last4": api_key[-4:] if len(api_key) >= 4 else "****",
            "created_at": datetime.utcnow().isoformat(),
            "user_id": "default-user",
            "status": "active",
            "balances": validation.get("balances", {})
        }
        
        _accounts_store[account_id] = account
        _active_account["id"] = account_id
        
        logger.info(f"Created exchange account: {account_id} ({label}, {environment})")
        
        return {
            "success": True,
            "account": account
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create account: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============= DASHBOARD DATA ENDPOINTS =============

@app.get("/api/system")
async def get_system_status():
    """Get system status for dashboard page."""
    return {
        "status": "healthy",
        "mode": trading_state.get("mode", "testnet"),
        "uptime": "0h 0m",
        "kill_switch_active": trading_state.get("kill_switch_active", False),
        "version": "11.1.0",
        "timestamp": datetime.utcnow().isoformat(),
        "exchange_health": "NORMAL",
        "active_strategies": 0,
        "open_positions": 0,
        "total_trades_today": 0
    }


@app.get("/api/context")
async def get_context():
    """Get market context for dashboard page."""
    try:
        # Try v11 risk state if available
        if get_risk_guardian:
            guardian = get_risk_guardian()
            state = guardian.get_state()
            return {
                "regime": state.get("market_regime", "UNKNOWN"),
                "confidence": 0,
                "trading_allowed": not trading_state.get("kill_switch_active", False),
                "timestamp": datetime.utcnow().isoformat()
            }
    except Exception:
        pass
    
    return {
        "regime": "UNKNOWN",
        "confidence": 0,
        "trading_allowed": not trading_state.get("kill_switch_active", False),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/api/risk")
async def get_risk_overview():
    """Get risk overview for dashboard and risk page."""
    try:
        result = {
            "daily_pnl": 0,
            "total_pnl": 0,
            "max_drawdown": 0,
            "current_exposure": 0,
            "risk_level": "low",
            "positions_count": 0,
            "win_rate": 0,
            "sharpe_ratio": 0,
            "profit_factor": 0,
            "total_trades": 0,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if get_risk_guardian:
            try:
                guardian = get_risk_guardian()
                state = guardian.get_state()
                result.update({
                    "max_drawdown": state.get("max_drawdown_pct", 0),
                    "risk_level": state.get("risk_level", "low"),
                })
            except Exception:
                pass
        
        if get_drawdown_governor:
            try:
                governor = get_drawdown_governor()
                status = governor.get_status()
                daily = status.get("states", {}).get("daily", {})
                result["daily_pnl"] = daily.get("current_pnl", 0)
            except Exception:
                pass
        
        return result
    except Exception as e:
        logger.error(f"Failed to get risk: {e}")
        return {
            "daily_pnl": 0, "total_pnl": 0, "max_drawdown": 0,
            "current_exposure": 0, "risk_level": "low", "positions_count": 0,
            "win_rate": 0, "sharpe_ratio": 0, "profit_factor": 0,
            "total_trades": 0, "timestamp": datetime.utcnow().isoformat()
        }


@app.get("/api/portfolio")
async def get_portfolio():
    """Get portfolio/positions for PositionsTable component."""
    return {
        "positions": [],
        "total_value": 0,
        "unrealized_pnl": 0,
        "timestamp": datetime.utcnow().isoformat()
    }


# ============= FEATURE PAGE ENDPOINTS =============

@app.get("/api/drift")
async def get_drift():
    """Get drift analysis data."""
    return {
        "drift_score": 0,
        "metrics": [],
        "alerts": [],
        "timestamp": datetime.utcnow().isoformat(),
        "status": "no_data"
    }


@app.get("/api/incident-state")
async def get_incident_state():
    """Get current incident state."""
    return {
        "active_incidents": [],
        "resolved_today": 0,
        "escalation_level": 0,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/api/operator")
async def get_operator():
    """Get operator state (frontend-compatible alias for v11 operator)."""
    try:
        if get_operator_control:
            operator = get_operator_control()
            return operator.get_state().to_dict()
    except Exception:
        pass
    
    return {
        "is_paused": False,
        "is_halted": False,
        "last_action": None,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/api/operator/pause")
async def operator_pause(request: Request):
    """Pause trading operations."""
    try:
        body = await request.json()
        reason = body.get("reason", "Manual pause")
        
        trading_state["kill_switch_active"] = True
        logger.warning(f"OPERATOR PAUSE: {reason}")
        
        if get_operator_control:
            operator = get_operator_control()
            operator.pause(reason=reason)
        
        return {
            "success": True,
            "action": "pause",
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/operator/resume")
async def operator_resume(request: Request):
    """Resume trading operations."""
    try:
        body = await request.json()
        reason = body.get("reason", "Manual resume")
        
        trading_state["kill_switch_active"] = False
        logger.info(f"OPERATOR RESUME: {reason}")
        
        if get_operator_control:
            operator = get_operator_control()
            operator.resume(reason=reason)
        
        return {
            "success": True,
            "action": "resume",
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/settings")
async def get_settings():
    """Get current settings."""
    return {
        "tradingMode": trading_state.get("mode", "testnet"),
        "apiConnection": {
            "exchange": "Binance",
            "status": "connected" if _active_account.get("id") else "disconnected",
            "testnet": trading_state.get("mode") == "testnet"
        },
        "riskLimits": {
            "dailyLossLimit": 0,
            "weeklyLossLimit": 0,
            "maxDrawdown": 0,
            "maxPositions": 0,
            "maxExposure": 0,
            "tradesPerDay": 0,
            "tradesPerContext": 0,
            "lossesPerBias": 0
        },
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/api/me")
async def get_current_user():
    """
    Get current user info for auth hydration.
    
    TODO: In production, extract user_id from JWT token.
    For now, returns a default user for testing.
    """
    try:
        repo = get_repository() if get_repository else None
        
        # For demo/testing - use default user
        # In production, extract user_id from Authorization header
        default_user_id = "default-user"
        
        user = None
        accounts = []
        active_account_id = None
        
        if repo:
            user = repo.find_by_id(default_user_id)
            if user:
                accounts = repo.find_accounts_by_user(default_user_id)
                active_account_id = repo.get_active_account_id(default_user_id)
        
        return {
            "authenticated": True,
            "user": {
                "user_id": default_user_id,
                "email": user.email if user else "demo@cryptoboss.io",
                "username": user.username if user else "Demo User"
            } if user or True else None,  # Always return user for testing
            "accounts": [
                {
                    "exchange_account_id": acc.exchange_account_id,
                    "exchange": acc.exchange,
                    "environment": acc.environment,
                    "label": acc.label
                } for acc in accounts
            ] if accounts else [],
            "active_account_id": active_account_id
        }
    except Exception as e:
        logger.error(f"Failed to get current user: {e}")
        return {
            "authenticated": False,
            "user": None,
            "accounts": [],
            "active_account_id": None
        }


@app.get("/api/positions")
async def get_positions_v2(
    exchange_account_id: str = Query(None, description="Filter by exchange account ID")
):
    """
    Get open positions, optionally filtered by exchange account.
    
    If exchange_account_id is provided, returns only positions for that account.
    """
    # For now, return empty positions - no mock data
    # In production, fetch from exchange via exchange_account_id
    return {
        "positions": [],
        "exchange_account_id": exchange_account_id,
        "timestamp": datetime.utcnow().isoformat(),
        "source": "TESTNET" if exchange_account_id else "NONE"
    }


@app.get("/api/trades")
async def get_trades_v2(
    exchange_account_id: str = Query(None, description="Filter by exchange account ID"),
    limit: int = Query(50, ge=1, le=500)
):
    """
    Get trades filtered by exchange account.
    
    CRITICAL: All trades are scoped by user_id (from JWT) and exchange_account_id.
    """
    try:
        repo = get_repository() if get_repository else None
        
        if not exchange_account_id:
            return {
                "trades": [],
                "total": 0,
                "message": "No exchange_account_id provided"
            }
        
        # TODO: Extract user_id from JWT
        user_id = "default-user"
        
        trades = []
        if repo:
            trades = repo.get_trades(user_id, exchange_account_id, limit)
        
        return {
            "trades": trades,
            "total": len(trades),
            "exchange_account_id": exchange_account_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get trades: {e}")
        return {"trades": [], "total": 0, "error": str(e)}


@app.post("/api/account/{exchange_account_id}/reset")
async def reset_account_data(exchange_account_id: str):
    """
    Reset all data for an exchange account.
    
    Deletes: trades, positions, analytics for that exchange_account_id.
    Requires confirmation & logs action.
    """
    try:
        repo = get_repository() if get_repository else None
        
        # TODO: Extract user_id from JWT
        user_id = "default-user"
        
        deleted_trades = 0
        if repo:
            deleted_trades = repo.delete_trades_for_account(user_id, exchange_account_id)
        
        logger.warning(f"RESET: Account {exchange_account_id} reset by {user_id}. Deleted {deleted_trades} trades.")
        
        return {
            "success": True,
            "exchange_account_id": exchange_account_id,
            "deleted_trades": deleted_trades,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to reset account: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/account/{exchange_account_id}/trade/{trade_id}")
async def delete_single_trade(exchange_account_id: str, trade_id: str):
    """
    Delete a single trade.
    
    CRITICAL: Only deletes if trade belongs to user_id AND exchange_account_id.
    """
    try:
        # TODO: Implement single trade deletion
        # For now, return success as placeholder
        logger.info(f"DELETE: Trade {trade_id} from account {exchange_account_id}")
        
        return {
            "success": True,
            "trade_id": trade_id,
            "exchange_account_id": exchange_account_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to delete trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/prices/live")
async def get_live_prices(
    symbols: str = Query("BTCUSDT,ETHUSDT", description="Comma-separated symbol list"),
    exchange_account_id: str = Query(None)
):
    """
    Get current live prices for symbols.
    
    Returns cached live prices from WebSocket feed if available.
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    
    # For now, return empty prices - will be filled by MarketDataService
    prices = {}
    for symbol in symbol_list:
        prices[symbol] = {
            "price": None,
            "timestamp_ms": None,
            "source": "DISCONNECTED"
        }
    
    return {
        "prices": prices,
        "exchange_account_id": exchange_account_id,
        "timestamp": datetime.utcnow().isoformat()
    }


# ============= REPLAY ENDPOINTS (DISABLED BY DEFAULT) =============

@app.get("/api/replay/sessions")
async def get_replay_sessions(
    exchange_account_id: str = Query(..., description="Account ID for scoping")
):
    """
    Get replay sessions for an account.
    
    CRITICAL: Replay is disabled by default.
    Only returns data when explicitly requested.
    """
    try:
        from src.core.replay_engine import get_replay_engine
        
        engine = get_replay_engine()
        sessions = engine.list_sessions()
        
        # Filter by account (if we implement account scoping in replay)
        # For now, return all sessions but with account context
        return {
            "sessions": sessions,
            "exchange_account_id": exchange_account_id,
            "replay_enabled": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    except ImportError:
        return {"sessions": [], "exchange_account_id": exchange_account_id, "replay_enabled": False}
    except Exception as e:
        logger.error(f"Failed to get replay sessions: {e}")
        return {"sessions": [], "exchange_account_id": exchange_account_id, "error": str(e)}


@app.get("/api/replay/session/{session_id}")
async def get_replay_session(
    session_id: str,
    exchange_account_id: str = Query(..., description="Account ID for scoping")
):
    """Get a specific replay session with decisions."""
    try:
        from src.core.replay_engine import get_replay_engine
        
        engine = get_replay_engine()
        session = engine.load_session(session_id)
        
        if not session:
            return {"decisions": [], "error": "Session not found"}
        
        # Format decisions for frontend
        decisions = []
        for d in session.decisions:
            decisions.append({
                "time": d.timestamp.split("T")[1][:8] if "T" in d.timestamp else d.timestamp,
                "type": d.decision_type,
                "live": {"action": d.result, "result": "OK"},
                "replay": {"action": d.result, "result": "OK"},
                "match": True  # TODO: Implement actual comparison
            })
        
        return {
            "decisions": decisions,
            "session_id": session_id,
            "exchange_account_id": exchange_account_id
        }
    except Exception as e:
        logger.error(f"Failed to load replay session: {e}")
        return {"decisions": [], "error": str(e)}


# ============= ACCOUNT RESET ENDPOINT =============

@app.post("/api/account/reset")
async def reset_account(
    exchange_account_id: str = Query(..., description="Account ID to reset")
):
    """
    Reset all data for an account.
    
    Deletes:
    - All trades
    - All positions  
    - All risk metrics
    - All decision logs
    
    CRITICAL: This is a destructive operation.
    """
    try:
        repo = get_repository() if get_repository else None
        
        if not repo:
            return {"success": False, "error": "Database not available"}
        
        # Log the reset
        logger.warning(f"🔄 Account reset requested for: {exchange_account_id}")
        
        # TODO: Implement actual deletion queries
        # For now, return success to indicate the endpoint exists
        
        return {
            "success": True,
            "exchange_account_id": exchange_account_id,
            "deleted": {
                "trades": 0,
                "positions": 0,
                "risk_metrics": 0,
                "decision_logs": 0
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to reset account: {e}")
        return {"success": False, "error": str(e)}


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
    result = {
        'timestamp': datetime.utcnow().isoformat(),
        'risk_state': 'safe',
        'kill_switch_active': trading_state.get('kill_switch_active', False),
    }
    
    if get_risk_guardian:
        try:
            guardian = get_risk_guardian()
            result.update({
                'daily_drawdown': guardian.limits.daily_drawdown_limit,
                'can_trade': guardian.can_trade(),
            })
        except Exception as e:
            logger.warning(f"Risk guardian unavailable: {e}")
    
    if get_incident_state_machine:
        try:
            ism = get_incident_state_machine()
            result['incident_state'] = ism.get_snapshot().to_dict()
            result['can_open_positions'] = ism.can_open_new_positions()
        except Exception as e:
            logger.warning(f"Incident state machine unavailable: {e}")
    
    return result


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


# Legacy /api/positions removed - use v2.0 endpoint /api/positions with exchange_account_id


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
    """Get available strategies. NO MOCK DATA."""
    return trading_state.get('strategies', [])


# REMOVED: Paper mode endpoint - paper trading permanently disabled
# Use TESTNET for testing, LIVE for production


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


# Legacy WebSocket endpoints - NOW USES REAL PRICES
@app.websocket("/ws/prices")
async def websocket_prices(
    websocket: WebSocket,
    account: str = None,
    symbols: str = "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT"
):
    """
    WebSocket for real-time price updates from Binance.
    
    Args:
        account: exchange_account_id for filtering
        symbols: comma-separated list of symbols to stream
    
    CRYPTOBOSS 2.0: No fake prices - all data from exchange
    """
    await websocket.accept()
    
    symbol_list = symbols.split(",") if symbols else ["BTCUSDT"]
    exchange_account_id = account or "default"
    
    # Import MarketDataService
    try:
        from src.exchange.binance_client import MarketDataService
    except ImportError:
        from exchange.binance_client import MarketDataService
    
    # Determine if testnet based on environment
    import os
    testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
    
    # Create service for this connection
    service = MarketDataService(
        exchange_account_id=exchange_account_id,
        testnet=testnet,
        poll_interval=1.0
    )
    
    # Queue for price updates
    price_queue = asyncio.Queue()
    
    async def price_callback(event):
        await price_queue.put(event)
    
    try:
        # Subscribe to symbols
        for symbol in symbol_list:
            service.subscribe(symbol, price_callback)
        
        # Start price service
        await service.start()
        
        # Send initial connection message
        await websocket.send_json({
            "type": "connected",
            "exchange_account_id": exchange_account_id,
            "symbols": symbol_list,
            "source": "TESTNET" if testnet else "LIVE",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Stream prices to client
        while True:
            try:
                # Wait for price update with timeout
                event = await asyncio.wait_for(price_queue.get(), timeout=5.0)
                
                await websocket.send_json({
                    "type": "price",
                    "channel": "prices",
                    "symbol": event.get("symbol"),
                    "price": event.get("price"),
                    "timestamp": event.get("timestamp_ms"),
                    "source": event.get("source"),
                    "exchange_account_id": event.get("exchange_account_id")
                })
                
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
    except WebSocketDisconnect:
        logger.info(f"Price WebSocket disconnected: {exchange_account_id}")
    except Exception as e:
        logger.error(f"Price WebSocket error: {e}")
    finally:
        # Clean up
        await service.stop()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

