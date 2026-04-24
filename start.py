"""
CryptoBoss Launcher — boots the REAL dashboard API.

Usage:
    python start.py                   # default: testnet mode, port 8000
    python start.py --port 9000       # custom port
    python start.py --host 0.0.0.0    # bind to all interfaces

All trading runs through dashboard/api.py which contains:
    - Real Binance exchange connectivity
    - Real-time market data (Binance mainnet WebSocket)
    - AggressiveScalper strategy loop
    - WebSocket broadcast to frontend
    - Auth, accounts, kill switch, risk management
"""

import os
import sys
import signal
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=str(env_path))

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("CryptoBoss")


def main():
    parser = argparse.ArgumentParser(description="CryptoBoss Trading Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    # Validate env
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    if api_key and api_secret:
        logger.info(f"Binance API key found: {api_key[:8]}...{api_key[-4:]}")
    else:
        logger.warning("No Binance API keys found in .env — exchange features will be unavailable")

    logger.info("=" * 60)
    logger.info("  CRYPTOBOSS TRADING DASHBOARD")
    logger.info("  Mode: TESTNET (connect via Settings page for LIVE)")
    logger.info(f"  Dashboard: http://{args.host}:{args.port}")
    logger.info(f"  Frontend:  http://localhost:3000")
    logger.info("=" * 60)

    try:
        import uvicorn
        uvicorn.run(
            "dashboard.api:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info",
        )
    except KeyboardInterrupt:
        logger.info("Shutting down CryptoBoss...")
    except Exception as e:
        logger.error(f"Failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
