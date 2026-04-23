"""
CryptoBoss Launch Script
Starts Backend API + Trading Bot in testnet mode
"""
import asyncio
import sys
import os
import threading
import time
import logging

# Fix Windows encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-5s | %(name)s - %(message)s'
)
logger = logging.getLogger("CryptoBoss.Launcher")


def start_api_server():
    """Start the FastAPI backend on port 8000."""
    try:
        import uvicorn
        logger.info("Starting API server on http://127.0.0.1:8000")
        uvicorn.run(
            "src.api.routes:app",
            host="127.0.0.1",
            port=8000,
            log_level="info",
            reload=False,
        )
    except Exception as e:
        logger.error(f"API server failed: {e}")


async def start_trading_bot():
    """Start the trading bot in testnet mode."""
    from dotenv import load_dotenv
    load_dotenv()
    
    from src.core.secrets_manager import get_secrets
    from src.core.engine import create_engine
    
    secrets = get_secrets()
    valid, missing = secrets.validate_required()
    
    if not valid:
        logger.error(f"Missing required secrets: {missing}")
        logger.error("Set BINANCE_API_KEY and BINANCE_API_SECRET in .env file")
        logger.error("Get testnet keys from: https://testnet.binance.vision/")
        return
    
    # Create exchange client for testnet
    try:
        from src.exchange.binance_client import BinanceClient
        creds = secrets.get_exchange_credentials("binance")
        
        logger.info("Connecting to Binance TESTNET...")
        client = BinanceClient(
            api_key=creds["api_key"],
            api_secret=creds["api_secret"],
            testnet=True,
        )
        await client.connect()
        
        # Test connection
        balance = await client.get_balance()
        usdt_balance = balance.get("USDT", {}).get("free", 0)
        btc_price = await client.get_price("BTC/USDT")
        
        logger.info(f"Connected to TESTNET!")
        logger.info(f"USDT Balance: ${usdt_balance:,.2f}")
        logger.info(f"BTC Price: ${btc_price:,.2f}")
        
    except Exception as e:
        logger.error(f"Exchange connection failed: {e}")
        logger.warning("Falling back to paper mode with simulated prices...")
        # Clean up the failed client
        try:
            if client and hasattr(client, '_exchange') and client._exchange:
                await client._exchange.close()
        except Exception:
            pass
        client = None
    
    # Create engine
    mode = "testnet" if client else "paper"
    engine = create_engine(
        mode=mode,
        portfolio_value=10000.0,
        exchange_client=client,
    )
    
    # Add strategy (DCAStrategy doesn't take symbol in __init__)
    from src.strategies.dca_strategy import DCAStrategy
    dca = DCAStrategy(base_order_size=500.0, safety_order_size=1000.0)
    engine.add_strategy("dca_btc_usdt", dca, symbol="BTC/USDT", allocation=10000.0)
    
    # Start engine
    engine.start()
    logger.info(f"Engine status: {engine.status.value}")
    
    # If no exchange client, use mock prices
    if not client:
        from src.core.mock_price_generator import MockPriceGenerator
        price_gen = MockPriceGenerator(
            symbols={"BTC/USDT": 67000.0},
            tick_interval=2.0,
        )
        
        tick_count = 0
        try:
            async for symbol, price, ts in price_gen.stream():
                await engine._process_price_update(symbol, price)
                tick_count += 1
                if tick_count % 30 == 0:
                    total_pnl = sum(s.get("pnl", 0) for s in engine.strategies.values())
                    logger.info(f"[{tick_count*2}s] {symbol}: ${price:,.2f} | P&L: ${total_pnl:.2f}")
        except asyncio.CancelledError:
            pass
        finally:
            price_gen.stop()
            engine.stop()
    else:
        # Live/testnet streaming via exchange WebSocket
        try:
            logger.info("Starting live price stream from TESTNET...")
            
            async def on_price(symbol, price, timestamp):
                await engine._process_price_update(symbol, price)
            
            # Register callback and stream
            client.subscribe_price("BTC/USDT", on_price)
            
            tick_count = 0
            while True:
                await asyncio.sleep(60)
                tick_count += 1
                total_pnl = sum(s.get("pnl", 0) for s in engine.strategies.values())
                current_price = client._prices.get("BTC/USDT", 0)
                logger.info(f"[{tick_count}m] BTC/USDT: ${current_price:,.2f} | P&L: ${total_pnl:.2f}")
                
        except asyncio.CancelledError:
            pass
        finally:
            await client.close()
            engine.stop()


def main():
    print("=" * 60)
    print("  CRYPTOBOSS TRADING SYSTEM")
    print("  Mode: TESTNET")
    print("=" * 60)
    print()
    print("  Backend API: http://127.0.0.1:8000")
    print("  Frontend UI: http://localhost:3000 (run 'npm run dev' in frontend/)")
    print("  API Docs:    http://127.0.0.1:8000/docs")
    print()
    print("=" * 60)
    
    # Start API server in background thread
    api_thread = threading.Thread(target=start_api_server, daemon=True)
    api_thread.start()
    logger.info("API server thread started")
    
    # Give API server a moment to start
    time.sleep(2)
    
    # Start trading bot in main event loop
    try:
        asyncio.run(start_trading_bot())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    
    print("\nCryptoBoss stopped.")


if __name__ == "__main__":
    main()
