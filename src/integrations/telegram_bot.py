"""
Telegram Trading Bot
Control your trading bot via Telegram messages.

Commands:
- /balance - Check portfolio balance
- /positions - View open positions  
- /pnl - Show profit & loss
- /start <strategy> - Start a strategy
- /stop <strategy> - Stop a strategy
- /buy <symbol> <amount> - Market buy
- /sell <symbol> <amount> - Market sell
- /alerts - Configure price alerts
- /help - Show all commands
"""

import logging
from typing import Dict, List, Optional
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    ContextTypes
)
import asyncio

logger = logging.getLogger(__name__)


class TelegramTradingBot:
    """
    Telegram bot for trading control and monitoring.
    
    Features:
    - Portfolio monitoring
    - Trade execution
    - Strategy control
    - Price alerts
    - Emergency controls
    """
    
    def __init__(
        self,
        token: str,
        authorized_users: List[int],
        trading_bot_instance,
        exchange_client
    ):
        """
        Initialize Telegram bot.
        
        Args:
            token: Telegram bot token from @BotFather
            authorized_users: List of authorized Telegram user IDs
            trading_bot_instance: Reference to main trading bot
            exchange_client: Exchange client for orders
        """
        self.token = token
        self.authorized_users = set(authorized_users)
        self.trading_bot = trading_bot_instance
        self.exchange = exchange_client
        
        # State
        self.price_alerts: Dict[str, List[Dict]] = {}
        
        # Create application
        self.app = Application.builder().token(token).build()
        
        # Register handlers
        self._register_handlers()
        
        logger.info(f"Telegram bot initialized with {len(authorized_users)} authorized users")
    
    def _check_authorization(self, user_id: int) -> bool:
        """Check if user is authorized."""
        return user_id in self.authorized_users
    
    def _register_handlers(self):
        """Register all command handlers."""
        # Commands
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("help", self.cmd_help))
        self.app.add_handler(CommandHandler("balance", self.cmd_balance))
        self.app.add_handler(CommandHandler("positions", self.cmd_positions))
        self.app.add_handler(CommandHandler("pnl", self.cmd_pnl))
        self.app.add_handler(CommandHandler("strategies", self.cmd_strategies))
        self.app.add_handler(CommandHandler("start_strategy", self.cmd_start_strategy))
        self.app.add_handler(CommandHandler("stop_strategy", self.cmd_stop_strategy))
        self.app.add_handler(CommandHandler("buy", self.cmd_buy))
        self.app.add_handler(CommandHandler("sell", self.cmd_sell))
        self.app.add_handler(CommandHandler("alerts", self.cmd_alerts))
        self.app.add_handler(CommandHandler("emergency_stop", self.cmd_emergency_stop))
        
        # Callback query handler for buttons
        self.app.add_handler(CallbackQueryHandler(self.button_callback))
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start command - welcome message."""
        user_id = update.effective_user.id
        
        if not self._check_authorization(user_id):
            await update.message.reply_text("⛔ Unauthorized. Contact bot owner.")
            return
        
        welcome_text = """
🤖 **CryptoBoss Trading Bot**

Welcome! I'll help you manage your trading bot via Telegram.

📱 **Quick Commands:**
/balance - Portfolio balance
/positions - Open positions
/pnl - Profit & Loss
/strategies - List strategies
/help - All commands

⚠️ **Emergency:**
/emergency_stop - Stop all trading immediately

Type /help for full command list.
        """
        
        await update.message.reply_text(welcome_text, parse_mode='Markdown')
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Help command - show all commands."""
        if not self._check_authorization(update.effective_user.id):
            return
        
        help_text = """
📚 **Command Reference**

**Portfolio:**
/balance - Show portfolio balance
/positions - List open positions
/pnl - Profit & Loss summary

**Trading:**
/buy <symbol> <amount> - Market buy
/sell <symbol> <amount> - Market sell

**Strategy Control:**
/strategies - List all strategies
/start_strategy <name> - Start strategy
/stop_strategy <name> - Stop strategy

**Alerts:**
/alerts - Manage price alerts

**Emergency:**
/emergency_stop - STOP ALL TRADING

**Info:**
/help - This message
        """
        
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Get portfolio balance."""
        if not self._check_authorization(update.effective_user.id):
            return
        
        try:
            # Get balance from exchange
            balance = await self.exchange.fetch_balance()
            
            # Format response
            response = "💰 **Portfolio Balance**\n\n"
            
            total_usd = 0
            for currency, amounts in balance['total'].items():
                if amounts > 0:
                    # Get current price in USD
                    try:
                        if currency == 'USDT':
                            price = 1.0
                        else:
                            ticker = await self.exchange.fetch_ticker(f"{currency}/USDT")
                            price = ticker['last']
                        
                        value_usd = amounts * price
                        total_usd += value_usd
                        
                        response += f"{currency}: {amounts:.4f} (${value_usd:,.2f})\n"
                    except:
                        response += f"{currency}: {amounts:.4f}\n"
            
            response += f"\n**Total**: ${total_usd:,.2f}"
            
            await update.message.reply_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show open positions."""
        if not self._check_authorization(update.effective_user.id):
            return
        
        try:
            # Get positions from trading bot
            if hasattr(self.trading_bot, 'get_positions'):
                positions = self.trading_bot.get_positions()
            else:
                await update.message.reply_text("No positions tracking available")
                return
            
            if not positions:
                await update.message.reply_text("📊 No open positions")
                return
            
            response = "📊 **Open Positions**\n\n"
            
            for pos in positions:
                response += f"**{pos['symbol']}**\n"
                response += f"  Side: {pos['side']}\n"
                response += f"  Entry: ${pos['entry_price']:,.2f}\n"
                response += f"  Size: {pos['quantity']:.4f}\n"
                response += f"  Unrealized P&L: ${pos['unrealized_pnl']:+,.2f}\n\n"
            
            await update.message.reply_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def cmd_pnl(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show profit & loss."""
        if not self._check_authorization(update.effective_user.id):
            return
        
        try:
            # Get P&L from trading bot
            if hasattr(self.trading_bot, 'get_performance'):
                perf = self.trading_bot.get_performance()
            else:
                await update.message.reply_text("No P&L tracking available")
                return
            
            response = "💵 **Profit & Loss**\n\n"
            response += f"Today: ${perf.get('daily_pnl', 0):+,.2f}\n"
            response += f"This Week: ${perf.get('weekly_pnl', 0):+,.2f}\n"
            response += f"This Month: ${perf.get('monthly_pnl', 0):+,.2f}\n"
            response += f"All Time: ${perf.get('total_pnl', 0):+,.2f}\n\n"
            response += f"Win Rate: {perf.get('win_rate', 0):.1f}%\n"
            response += f"Total Trades: {perf.get('total_trades', 0)}\n"
            
            await update.message.reply_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error fetching P&L: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def cmd_strategies(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """List all strategies."""
        if not self._check_authorization(update.effective_user.id):
            return
        
        try:
            # Get strategies from trading bot
            if hasattr(self.trading_bot, 'get_strategies'):
                strategies = self.trading_bot.get_strategies()
            else:
                await update.message.reply_text("No strategies available")
                return
            
            response = "📈 **Strategies**\n\n"
            
            for name, strategy in strategies.items():
                status = "🟢 Active" if strategy.get('active') else "🔴 Inactive"
                response += f"**{name}**: {status}\n"
                if 'pnl' in strategy:
                    response += f"  P&L: ${strategy['pnl']:+,.2f}\n"
                response += "\n"
            
            await update.message.reply_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error fetching strategies: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def cmd_buy(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Execute market buy."""
        if not self._check_authorization(update.effective_user.id):
            return
        
        try:
            # Parse command: /buy BTC 0.01
            args = context.args
            if len(args) < 2:
                await update.message.reply_text("Usage: /buy <symbol> <amount>")
                return
            
            symbol = args[0].upper()
            if '/' not in symbol:
                symbol = f"{symbol}/USDT"
            
            amount = float(args[1])
            
            # Confirm with user
            keyboard = [
                [
                    InlineKeyboardButton("✅ Confirm", callback_data=f"buy_confirm_{symbol}_{amount}"),
                    InlineKeyboardButton("❌ Cancel", callback_data="buy_cancel")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                f"Confirm market BUY:\n{amount} {symbol}",
                reply_markup=reply_markup
            )
            
        except Exception as e:
            logger.error(f"Error in buy command: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def cmd_sell(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Execute market sell."""
        if not self._check_authorization(update.effective_user.id):
            return
        
        try:
            # Parse command: /sell BTC 0.01
            args = context.args
            if len(args) < 2:
                await update.message.reply_text("Usage: /sell <symbol> <amount>")
                return
            
            symbol = args[0].upper()
            if '/' not in symbol:
                symbol = f"{symbol}/USDT"
            
            amount = float(args[1])
            
            # Confirm with user
            keyboard = [
                [
                    InlineKeyboardButton("✅ Confirm", callback_data=f"sell_confirm_{symbol}_{amount}"),
                    InlineKeyboardButton("❌ Cancel", callback_data="sell_cancel")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                f"Confirm market SELL:\n{amount} {symbol}",
                reply_markup=reply_markup
            )
            
        except Exception as e:
            logger.error(f"Error in sell command: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def cmd_emergency_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Emergency stop all trading."""
        if not self._check_authorization(update.effective_user.id):
            return
        
        try:
            # Confirm emergency stop
            keyboard = [
                [
                    InlineKeyboardButton("⚠️ CONFIRM STOP ALL", callback_data="emergency_confirm"),
                    InlineKeyboardButton("Cancel", callback_data="emergency_cancel")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                "⚠️ **EMERGENCY STOP**\n\nThis will:\n- Stop all strategies\n- Cancel all orders\n- Close all positions\n\nConfirm?",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            logger.error(f"Error in emergency stop: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button callbacks."""
        query = update.callback_query
        await query.answer()
        
        if not self._check_authorization(query.from_user.id):
            return
        
        data = query.data
        
        try:
            if data.startswith("buy_confirm"):
                # Execute buy order
                _, _, symbol, amount = data.split("_")
                order = await self.exchange.create_market_buy_order(symbol, float(amount))
                await query.edit_message_text(f"✅ Buy order executed:\n{order['id']}")
                
            elif data.startswith("sell_confirm"):
                # Execute sell order
                _, _, symbol, amount = data.split("_")
                order = await self.exchange.create_market_sell_order(symbol, float(amount))
                await query.edit_message_text(f"✅ Sell order executed:\n{order['id']}")
                
            elif data == "emergency_confirm":
                # Execute emergency stop
                if hasattr(self.trading_bot, 'emergency_stop'):
                    self.trading_bot.emergency_stop()
                await query.edit_message_text("🛑 **EMERGENCY STOP EXECUTED**\n\nAll trading halted.")
                
            elif data.endswith("cancel"):
                await query.edit_message_text("Cancelled")
                
        except Exception as e:
            logger.error(f"Error in callback: {e}")
            await query.edit_message_text(f"❌ Error: {str(e)}")
    
    async def cmd_start_strategy(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start a strategy."""
        if not self._check_authorization(update.effective_user.id):
            return
        
        try:
            args = context.args
            if not args:
                await update.message.reply_text("Usage: /start_strategy <name>")
                return
            
            strategy_name = " ".join(args)
            
            if hasattr(self.trading_bot, 'start_strategy'):
                success = self.trading_bot.start_strategy(strategy_name)
                if success:
                    await update.message.reply_text(f"✅ Started strategy: {strategy_name}")
                else:
                    await update.message.reply_text(f"❌ Failed to start: {strategy_name}")
            else:
                await update.message.reply_text("Strategy control not available")
                
        except Exception as e:
            logger.error(f"Error starting strategy: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def cmd_stop_strategy(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Stop a strategy."""
        if not self._check_authorization(update.effective_user.id):
            return
        
        try:
            args = context.args
            if not args:
                await update.message.reply_text("Usage: /stop_strategy <name>")
                return
            
            strategy_name = " ".join(args)
            
            if hasattr(self.trading_bot, 'stop_strategy'):
                success = self.trading_bot.stop_strategy(strategy_name)
                if success:
                    await update.message.reply_text(f"✅ Stopped strategy: {strategy_name}")
                else:
                    await update.message.reply_text(f"❌ Failed to stop: {strategy_name}")
            else:
                await update.message.reply_text("Strategy control not available")
                
        except Exception as e:
            logger.error(f"Error stopping strategy: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def cmd_alerts(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Manage price alerts."""
        if not self._check_authorization(update.effective_user.id):
            return
        
        await update.message.reply_text("Price alerts feature coming soon!")
    
    async def send_alert(self, user_ids: List[int], message: str):
        """Send alert to specified users."""
        for user_id in user_ids:
            try:
                await self.app.bot.send_message(chat_id=user_id, text=message, parse_mode='Markdown')
            except Exception as e:
                logger.error(f"Failed to send alert to {user_id}: {e}")
    
    def run(self):
        """Start the bot."""
        logger.info("Starting Telegram bot...")
        self.app.run_polling()
    
    async def start_async(self):
        """Start bot asynchronously."""
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()
    
    async def stop_async(self):
        """Stop bot asynchronously."""
        await self.app.updater.stop()
        await self.app.stop()
        await self.app.shutdown()


# Example usage
if __name__ == "__main__":
    # Configuration
    TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_FROM_BOTFATHER"
    AUTHORIZED_USERS = [123456789]  # Your Telegram user ID
    
    # Initialize (you need to provide actual trading_bot and exchange instances)
    bot = TelegramTradingBot(
        token=TELEGRAM_BOT_TOKEN,
        authorized_users=AUTHORIZED_USERS,
        trading_bot_instance=None,  # Your trading bot instance
        exchange_client=None  # Your exchange client
    )
    
    # Run
    bot.run()
