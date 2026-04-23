#!/usr/bin/env python3
"""
CryptoBoss Trading Bot - Clean Entry Point

This is the main entry point for the trading system.
Uses the new institutional-grade architecture from src.core.

Usage:
    python run_trading_bot.py                    # Paper trading (default)
    python run_trading_bot.py --mode=live        # Live trading
"""

import asyncio
import argparse
import signal
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core import (
    create_engine,
    get_config,
    get_secrets,
    get_observability,
    get_shutdown_manager,
    get_integration_hub,
)
from src.core.integration_hub import TelegramIntegration, DiscordIntegration
from src.strategies.dca_strategy import DCAStrategy
from src.strategies.range_scalp import RangeScalpStrategy
from src.strategies.smc_scalper import SMCScalperStrategy
from src.strategies.smc_trend_follow import SMCTrendFollowStrategy
from src.exchange.binance_client import BinanceClient, create_binance_client

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s - %(message)s'
)
logger = logging.getLogger("CryptoBoss")


async def test_exchange_connection(testnet: bool = True) -> tuple[bool, dict]:
    """
    Test exchange connectivity and return status.
    
    Returns:
        tuple: (success: bool, info: dict with balance/prices/error)
    """
    print("\n  📡 Testing Exchange Connection...")
    
    try:
        client = create_binance_client(testnet=testnet)
        await client.connect()
        
        # Test balance fetch
        balance = await client.get_balance()
        usdt_balance = balance.get("USDT", {}).get("free", 0)
        
        # Test price fetch
        btc_price = await client.get_price("BTC/USDT")
        
        await client.close()
        
        info = {
            "connected": True,
            "testnet": testnet,
            "usdt_balance": usdt_balance,
            "btc_price": btc_price,
        }
        
        print(f"  ✅ Connected to {'Testnet' if testnet else 'Mainnet'}")
        print(f"  💰 USDT Balance: ${usdt_balance:,.2f}")
        print(f"  📊 BTC Price: ${btc_price:,.2f}")
        
        return True, info
        
    except Exception as e:
        error_msg = str(e)
        print(f"  ⚠️  Exchange connection failed: {error_msg}")
        return False, {"connected": False, "error": error_msg}


def parse_args():
    parser = argparse.ArgumentParser(description="CryptoBoss Trading Bot")
    parser.add_argument(
        "--mode", 
        choices=["paper", "live", "backtest"],
        default="paper",
        help="Trading mode (default: paper)"
    )
    parser.add_argument(
        "--env",
        choices=["dev", "staging", "prod"],
        default="dev",
        help="Environment (default: dev)"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Starting capital in USD (default: 10000)"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC/USDT"],
        help="Trading symbols (default: BTC/USDT)"
    )
    parser.add_argument(
        "--strategy",
        choices=["dca", "smc_trend_follow", "smc_scalper", "range_scalp", "all"],
        default="dca",
        help="Strategy to run (default: dca)"
    )
    return parser.parse_args()


async def setup_integrations():
    """Setup notification integrations."""
    hub = get_integration_hub()
    secrets = get_secrets()
    
    # Telegram (optional)
    telegram_config = secrets.get_telegram_config()
    if telegram_config.get("enabled"):
        try:
            hub.register(TelegramIntegration(
                bot_token=telegram_config["token"],
                authorized_users=telegram_config.get("user_ids", [])
            ))
            logger.info("Telegram integration registered")
        except Exception as e:
            logger.warning(f"Could not setup Telegram: {e}")
    
    # Discord (optional)
    discord_url = secrets.get("DISCORD_WEBHOOK_URL")
    if discord_url:
        try:
            hub.register(DiscordIntegration(webhook_url=discord_url))
            logger.info("Discord integration registered")
        except Exception as e:
            logger.warning(f"Could not setup Discord: {e}")
    
    await hub.start_all()
    return hub


def create_strategies(args, engine):
    """Create and add strategies to the engine."""
    selected_strategies = (
        ["dca", "smc_trend_follow", "smc_scalper", "range_scalp"]
        if args.strategy == "all"
        else [args.strategy]
    )
    allocation_per_strategy = args.capital / max(1, len(args.symbols) * len(selected_strategies))
    
    for symbol in args.symbols:
        if "dca" in selected_strategies:
            # DCA Strategy
            dca = DCAStrategy(
                base_order_size=allocation_per_strategy * 0.05,  # 5% per order
                safety_order_size=allocation_per_strategy * 0.10,
                max_safety_orders=5,
                price_step_pct=2.0,  # Fixed: was price_deviation_pct
                target_profit_pct=2.0,
                safety_order_volume_scale=1.5,
                stop_loss_pct=10.0
            )
            engine.add_strategy(
                strategy_id=f"dca_{symbol.replace('/', '_').lower()}",
                strategy=dca,
                symbol=symbol,
                allocation=allocation_per_strategy,
                auto_start=True
            )
            logger.info(f"Added DCA strategy for {symbol}")

        if "smc_trend_follow" in selected_strategies:
            trend = SMCTrendFollowStrategy(symbol=symbol)
            engine.add_strategy(
                strategy_id=f"smc_trend_{symbol.replace('/', '_').lower()}",
                strategy=trend,
                symbol=symbol,
                allocation=allocation_per_strategy,
                auto_start=True,
            )
            logger.info(f"Added SMC trend-follow strategy for {symbol}")

        if "smc_scalper" in selected_strategies:
            scalper = SMCScalperStrategy(symbol=symbol)
            engine.add_strategy(
                strategy_id=f"smc_scalper_{symbol.replace('/', '_').lower()}",
                strategy=scalper,
                symbol=symbol,
                allocation=allocation_per_strategy,
                auto_start=True,
            )
            logger.info(f"Added SMC scalper strategy for {symbol}")

        if "range_scalp" in selected_strategies:
            range_scalp = RangeScalpStrategy(symbol=symbol)
            engine.add_strategy(
                strategy_id=f"range_scalp_{symbol.replace('/', '_').lower()}",
                strategy=range_scalp,
                symbol=symbol,
                allocation=allocation_per_strategy,
                auto_start=True,
            )
            logger.info(f"Added range scalp strategy for {symbol}")


async def main():
    args = parse_args()
    
    # Banner
    print("=" * 60)
    print("  🤖 CRYPTOBOSS TRADING BOT")
    print("=" * 60)
    print(f"  Mode: {args.mode.upper()}")
    print(f"  Environment: {args.env}")
    print(f"  Capital: ${args.capital:,.2f}")
    print(f"  Symbols: {', '.join(args.symbols)}")
    print(f"  Strategy: {args.strategy}")
    print("=" * 60)
    
    # Load config and validate secrets
    config = get_config(env=args.env)
    secrets = get_secrets()
    
    # Validate secrets for live mode
    if args.mode == "live":
        valid, missing = secrets.validate_required()
        if not valid:
            logger.error(f"Missing required secrets for live trading: {missing}")
            sys.exit(1)
        
        # Production safety check
        if args.env != "prod":
            logger.error("Live trading requires --env=prod")
            sys.exit(1)
    
    # Test exchange connectivity
    use_testnet = args.env != "prod"
    connected, exchange_info = await test_exchange_connection(testnet=use_testnet)
    
    if not connected and args.mode == "live":
        logger.error("Cannot proceed with live trading - exchange connection failed")
        sys.exit(1)
    
    # Create Binance client for live price data
    exchange_client = None
    if connected:
        exchange_client = create_binance_client(testnet=use_testnet)
        await exchange_client.connect()
    
    # Create trading engine with exchange client
    engine = create_engine(
        mode=args.mode,
        portfolio_value=args.capital,
        exchange_client=exchange_client
    )
    
    # Setup strategies
    create_strategies(args, engine)
    
    # Setup integrations
    hub = await setup_integrations()
    
    # Setup graceful shutdown
    shutdown = get_shutdown_manager()
    
    async def cleanup():
        logger.info("Shutting down...")
        await hub.stop_all()
        engine.stop()
        if exchange_client:
            await exchange_client.close()
    
    shutdown.register_async(cleanup)
    
    # Start engine
    logger.info("Starting trading engine...")
    engine.start()
    
    # Broadcast startup
    await hub.broadcast(
        f"🚀 CryptoBoss started in {args.mode.upper()} mode with ${args.capital:,.2f}"
    )
    
    # Print status
    status = engine.get_status()
    logger.info(f"Engine status: {status['status']}")
    logger.info(f"Active strategies: {status['strategies']}")
    
    # Start live price feed if connected
    if exchange_client and connected:
        logger.info("Starting live price feed...")
        await exchange_client.start_price_feed(args.symbols, interval=5.0)
        
        # Subscribe engine to price updates
        for symbol in args.symbols:
            exchange_client.subscribe_price(symbol, lambda s, p: asyncio.create_task(
                engine._process_price_update(s, p)
            ))
    
    print("\n" + "=" * 60)
    print("  ✅ BOT STARTED SUCCESSFULLY")
    if connected:
        print(f"  📡 Live prices: {', '.join(args.symbols)}")
    print("=" * 60)
    print("  Press Ctrl+C to stop")
    print("=" * 60)
    
    try:
        # Main loop - show periodic status
        tick = 0
        while True:
            await asyncio.sleep(5)
            tick += 1
            
            # Every 60 seconds, print status
            if tick % 12 == 0:
                status = engine.get_status()
                pnl = status.get('total_pnl', 0)
                logger.info(f"[{tick*5}s] P&L: ${pnl:,.2f} | Active: {len([s for s in status['strategies'].values() if s['active']])}")
            
    except KeyboardInterrupt:
        pass
    finally:
        await cleanup()
    
    # Final summary
    final_status = engine.get_status()
    print("\n" + "=" * 60)
    print("  📊 FINAL SUMMARY")
    print("=" * 60)
    print(f"  Total P&L: ${final_status.get('total_pnl', 0):,.2f}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)
