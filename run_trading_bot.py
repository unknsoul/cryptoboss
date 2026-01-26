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
        choices=["dca", "grid", "all"],
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
    allocation_per_strategy = args.capital / len(args.symbols)
    
    for symbol in args.symbols:
        if args.strategy in ("dca", "all"):
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
    
    # Create trading engine
    engine = create_engine(
        mode=args.mode,
        portfolio_value=args.capital
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
    
    # Simple run loop (no WebSocket for now)
    print("\n" + "=" * 60)
    print("  ✅ BOT STARTED SUCCESSFULLY")
    print("=" * 60)
    print("  Press Ctrl+C to stop")
    print("=" * 60)
    
    try:
        # Just wait for keyboard interrupt
        while True:
            await asyncio.sleep(1)
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
