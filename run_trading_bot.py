#!/usr/bin/env python3
"""
CryptoBoss Trading Bot - Clean Entry Point

This is the main entry point for the trading system.

Usage:
    python run_trading_bot.py                          # Paper trading (default, no API keys needed)
    python run_trading_bot.py --mode=live --env=prod   # Live trading (requires API keys)
    python run_trading_bot.py --mode=backtest           # Backtest mode
"""

import asyncio
import argparse
import signal
import sys
import logging
from pathlib import Path

# Fix Windows console encoding for emoji/unicode
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

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
from src.core.mock_price_generator import MockPriceGenerator
from src.strategies.dca_strategy import DCAStrategy
from src.strategies.range_scalp import RangeScalpStrategy
from src.strategies.smc_scalper import SMCScalperStrategy
from src.strategies.smc_trend_follow import SMCTrendFollowStrategy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s - %(message)s'
)
logger = logging.getLogger("CryptoBoss")


def parse_args():
    parser = argparse.ArgumentParser(description="CryptoBoss Trading Bot")
    parser.add_argument(
        "--mode",
        choices=["paper", "testnet", "live", "backtest"],
        default="testnet",
        help="Trading mode (default: testnet)"
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


async def setup_exchange_client(args):
    """
    Setup exchange client for TESTNET or LIVE mode.
    
    Returns:
        (exchange_client, exchange_info) or (None, None) for paper mode
        
    Raises:
        SystemExit if setup fails
    """
    if args.mode == "paper":
        return None, None
    
    # Testnet/Live mode requires API keys — fail fast if missing
    try:
        from src.exchange.binance_client import BinanceClient
    except ImportError:
        logger.error("Exchange client libraries not installed. Run: pip install ccxt python-binance")
        sys.exit(1)
    
    # Live mode safety: require --env=prod
    if args.mode == "live" and args.env != "prod":
        logger.error(
            "❌ Live trading requires --env=prod for safety.\n"
            "   Run: python run_trading_bot.py --mode=live --env=prod"
        )
        sys.exit(1)
    
    # Validate secrets
    secrets = get_secrets()
    valid, missing = secrets.validate_required()
    if not valid:
        logger.error(
            f"❌ Missing required secrets: {missing}\n"
            "   Set BINANCE_API_KEY and BINANCE_API_SECRET in .env file"
        )
        sys.exit(1)
    
    # Determine testnet vs live
    use_testnet = args.mode == "testnet"
    env_label = "TESTNET" if use_testnet else "LIVE MAINNET"
    
    print(f"\n  📡 Connecting to Binance {env_label}...")
    
    try:
        creds = secrets.get_exchange_credentials("binance")
        client = BinanceClient(
            api_key=creds["api_key"],
            api_secret=creds["api_secret"],
            testnet=use_testnet,
        )
        await client.connect()
        
        # Test balance fetch
        balance = await client.get_balance()
        usdt_balance = balance.get("USDT", {}).get("free", 0)
        
        # Test price fetch
        btc_price = await client.get_price("BTC/USDT")
        
        print(f"  ✅ Connected to {env_label}")
        print(f"  💰 USDT Balance: ${usdt_balance:,.2f}")
        print(f"  📊 BTC Price: ${btc_price:,.2f}")
        
        exchange_info = {
            "connected": True,
            "testnet": use_testnet,
            "usdt_balance": usdt_balance,
            "btc_price": btc_price,
        }
        
        return client, exchange_info
        
    except Exception as e:
        logger.error(f"❌ Exchange connection failed: {e}")
        if use_testnet:
            logger.error(
                "Make sure you have valid testnet keys in .env file.\n"
                "Get testnet keys from: https://testnet.binance.vision/"
            )
        else:
            logger.error("Cannot proceed with live trading — fix credentials and retry")
        sys.exit(1)


async def setup_integrations():
    """Setup notification integrations (optional, never blocks startup)."""
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
    
    try:
        await hub.start_all()
    except Exception as e:
        logger.warning(f"Integration startup warning: {e}")
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
            dca = DCAStrategy(
                base_order_size=allocation_per_strategy * 0.05,
                safety_order_size=allocation_per_strategy * 0.10,
                max_safety_orders=5,
                price_step_pct=2.0,
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


async def run_paper_mode(args, engine, hub):
    """Run bot in paper trading mode with mock price generator."""
    print("\n  📄 PAPER MODE — Using simulated prices (no API keys needed)")
    
    # Create mock price generator with configured symbols
    symbol_prices = {}
    for symbol in args.symbols:
        if "BTC" in symbol:
            symbol_prices[symbol] = 67000.0
        elif "ETH" in symbol:
            symbol_prices[symbol] = 3200.0
        elif "SOL" in symbol:
            symbol_prices[symbol] = 145.0
        else:
            symbol_prices[symbol] = 100.0
    
    price_gen = MockPriceGenerator(symbols=symbol_prices, tick_interval=2.0)
    
    tick = 0
    async for symbol, price, ts in price_gen.stream():
        tick += 1
        
        # Feed price to engine
        await engine._process_price_update(symbol, price)
        
        # Periodic status logging
        if tick % 30 == 0:  # Every ~60 seconds (2s interval * 30)
            status = engine.get_status()
            pnl = status.get('total_pnl', 0)
            active = len([s for s in status['strategies'].values() if s['active']])
            logger.info(
                f"[{tick * 2}s] {symbol}: ${price:,.2f} | "
                f"P&L: ${pnl:,.2f} | Active: {active}"
            )


async def run_live_mode(args, engine, exchange_client, hub):
    """Run bot in live trading mode with real exchange."""
    print("\n  🔴 LIVE MODE — Real money trading active!")
    
    # Start live price feed
    logger.info("Starting live price feed...")
    await exchange_client.start_price_feed(args.symbols, interval=5.0)
    
    # Subscribe engine to price updates
    for symbol in args.symbols:
        exchange_client.subscribe_price(symbol, lambda s, p: asyncio.create_task(
            engine._process_price_update(s, p)
        ))
    
    # Main loop
    tick = 0
    while True:
        await asyncio.sleep(5)
        tick += 1
        
        if tick % 12 == 0:  # Every 60 seconds
            status = engine.get_status()
            pnl = status.get('total_pnl', 0)
            active = len([s for s in status['strategies'].values() if s['active']])
            logger.info(f"[{tick * 5}s] P&L: ${pnl:,.2f} | Active: {active}")


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
    
    # === MODE-SPECIFIC SETUP ===
    exchange_client = None
    
    if args.mode == "paper":
        # Paper mode: no exchange client needed
        logger.info("Paper mode — skipping exchange connection")
        exchange_client = None
        
    elif args.mode in ("testnet", "live"):
        # Testnet/Live mode: require exchange client, fail fast
        exchange_client, exchange_info = await setup_exchange_client(args)
        
    elif args.mode == "backtest":
        logger.info("Backtest mode — loading historical data")
        print("\n  ⚠️  Backtest mode is not fully implemented yet.")
        print("  Use run_backtest.py for backtesting.")
        sys.exit(0)
    
    # Create trading engine
    engine = create_engine(
        mode=args.mode,
        portfolio_value=args.capital,
        exchange_client=exchange_client
    )
    
    # Setup strategies
    create_strategies(args, engine)
    
    # Setup integrations (optional, never blocks)
    hub = await setup_integrations()
    
    # Setup graceful shutdown
    shutdown = get_shutdown_manager()
    
    async def cleanup():
        logger.info("Shutting down...")
        try:
            await hub.stop_all()
        except Exception:
            pass
        engine.stop()
        if exchange_client:
            try:
                await exchange_client.close()
            except Exception:
                pass
    
    shutdown.register_async(cleanup)
    
    # Start engine
    logger.info("Starting trading engine...")
    engine.start()
    
    # Broadcast startup
    try:
        await hub.broadcast(
            f"🚀 CryptoBoss started in {args.mode.upper()} mode with ${args.capital:,.2f}"
        )
    except Exception:
        pass
    
    # Print status
    status = engine.get_status()
    logger.info(f"Engine status: {status['status']}")
    logger.info(f"Active strategies: {status['strategies']}")
    
    print("\n" + "=" * 60)
    print("  ✅ BOT STARTED SUCCESSFULLY")
    if args.mode == "paper":
        print("  📄 Paper mode — simulated prices")
    elif args.mode == "testnet":
        print(f"  🧪 Testnet mode — real orders on testnet")
        print(f"  📡 Symbols: {', '.join(args.symbols)}")
    elif args.mode == "live":
        print(f"  🔴 Live mode — REAL MONEY")
        print(f"  📡 Symbols: {', '.join(args.symbols)}")
    print("=" * 60)
    print("  Press Ctrl+C to stop")
    print("=" * 60)
    
    try:
        if args.mode == "paper":
            await run_paper_mode(args, engine, hub)
        elif args.mode in ("testnet", "live"):
            await run_live_mode(args, engine, exchange_client, hub)
            
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
