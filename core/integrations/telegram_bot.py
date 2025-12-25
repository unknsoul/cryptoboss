"""
Telegram Bot Integration
Remote control and notifications for trading bot.
"""
import logging
from typing import Optional, Callable
from datetime import datetime

try:
    from telegram import Update, Bot
    from telegram.ext import Application, CommandHandler, ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logging.warning("python-telegram-bot not installed. Telegram features disabled.")

logger = logging.getLogger(__name__)


class TelegramBotManager:
    """
    Telegram bot for remote monitoring and control.
    
    Commands:
    - /status - Current positions and equity
    - /stats - Performance metrics
    - /pause - Pause trading
    - /resume - Resume trading
    - /equity - Current equity
    """
    
    def __init__(self, token: str, chat_id: str, bot_instance=None):
        """
        Initialize Telegram bot.
        
        Args:
            token: Telegram bot token
            chat_id: Authorized chat ID
            bot_instance: Trading bot instance
        """
        if not TELEGRAM_AVAILABLE:
            raise ImportError("python-telegram-bot required. Install: pip install python-telegram-bot")
        
        self.token = token
        self.chat_id = chat_id
        self.bot_instance = bot_instance
        self.app = None
        self.telegram_bot = Bot(token=token)
        
        logger.info("Telegram Bot Manager initialized")
    
    def start(self):
        """Start Telegram bot."""
        self.app = Application.builder().token(self.token).build()
        
        # Register command handlers
        self.app.add_handler(CommandHandler("status", self._cmd_status))
        self.app.add_handler(CommandHandler("stats", self._cmd_stats))
        self.app.add_handler(CommandHandler("pause", self._cmd_pause))
        self.app.add_handler(CommandHandler("resume", self._cmd_resume))
        self.app.add_handler(CommandHandler("equity", self._cmd_equity))
        self.app.add_handler(CommandHandler("help", self._cmd_help))
        
        # Start polling in background
        self.app.run_polling()
        logger.info("Telegram bot started")
    
    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        if not self._is_authorized(update):
            return
        
        if not self.bot_instance:
            await update.message.reply_text("Bot instance not connected")
            return
        
        # Get current status
        position = self.bot_instance.position
        equity = self.bot_instance.equity
        
        if position:
            msg = f"🟢 *ACTIVE POSITION*\n\n"
            msg += f"Side: {position['side']}\n"
            msg += f"Entry: ${position['entry_price']:.2f}\n"
            msg += f"Size: {position['size']:.4f}\n"
            msg += f"Strategy: {position.get('strategy', 'N/A')}\n\n"
            msg += f"💰 Equity: ${equity:,.2f}"
        else:
            msg = f"⚪ NO POSITION\n\n"
            msg += f"💰 Equity: ${equity:,.2f}"
        
        await update.message.reply_text(msg, parse_mode='Markdown')
    
    async def _cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command."""
        if not self._is_authorized(update):
            return
        
        if not self.bot_instance:
            await update.message.reply_text("Bot instance not connected")
            return
        
        bot = self.bot_instance
        total_trades = len([t for t in bot.trades if t.get('status') == 'CLOSED'])
        wins = len([t for t in bot.trades if t.get('status') == 'CLOSED' and t.get('pnl', 0) > 0])
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        total_return = ((bot.equity - bot.initial_capital) / bot.initial_capital * 100)
        
        msg = f"📊 *PERFORMANCE STATS*\n\n"
        msg += f"Equity: ${bot.equity:,.2f}\n"
        msg += f"Return: {total_return:+.2f}%\n"
        msg += f"Trades: {total_trades}\n"
        msg += f"Win Rate: {win_rate:.1f}%\n"
        msg += f"Win Streak: {bot.win_streak}\n"
        msg += f"Loss Streak: {bot.loss_streak}"
        
        await update.message.reply_text(msg, parse_mode='Markdown')
    
    async def _cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pause command."""
        if not self._is_authorized(update):
            return
        
        # Set pause flag
        if hasattr(self.bot_instance, 'trading_paused'):
            self.bot_instance.trading_paused = True
            await update.message.reply_text("⏸️ Trading PAUSED")
        else:
            await update.message.reply_text("Pause feature not available")
    
    async def _cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /resume command."""
        if not self._is_authorized(update):
            return
        
        if hasattr(self.bot_instance, 'trading_paused'):
            self.bot_instance.trading_paused = False
            await update.message.reply_text("▶️ Trading RESUMED")
        else:
            await update.message.reply_text("Resume feature not available")
    
    async def _cmd_equity(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /equity command."""
        if not self._is_authorized(update):
            return
        
        if not self.bot_instance:
            await update.message.reply_text("Bot instance not connected")
            return
        
        equity = self.bot_instance.equity
        initial = self.bot_instance.initial_capital
        pnl = equity - initial
        pnl_pct = (pnl / initial * 100)
        
        msg = f"💰 *EQUITY*\n\n"
        msg += f"Current: ${equity:,.2f}\n"
        msg += f"Initial: ${initial:,.2f}\n"
        msg += f"P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)"
        
        await update.message.reply_text(msg, parse_mode='Markdown')
    
    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        msg = "🤖 *TRADING BOT COMMANDS*\n\n"
        msg += "/status - Current position\n"
        msg += "/stats - Performance metrics\n"
        msg += "/equity - Equity details\n"
        msg += "/pause - Pause trading\n"
        msg += "/resume - Resume trading\n"
        msg += "/help - This message"
        
        await update.message.reply_text(msg, parse_mode='Markdown')
    
    def _is_authorized(self, update: Update) -> bool:
        """Check if user is authorized."""
        if str(update.effective_chat.id) != str(self.chat_id):
            logger.warning(f"Unauthorized Telegram access attempt from {update.effective_chat.id}")
            return False
        return True
    
    async def send_notification(self, message: str):
        """Send push notification."""
        try:
            await self.telegram_bot.send_message(chat_id=self.chat_id, text=message, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
    
    async def notify_trade(self, trade: dict):
        """Send trade notification."""
        side = trade.get('side', 'UNKNOWN')
        entry = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price')
        pnl = trade.get('pnl', 0)
        
        if exit_price:
            # Trade closed
            emoji = "🟢" if pnl > 0 else "🔴"
            msg = f"{emoji} *TRADE CLOSED*\n\n"
            msg += f"Side: {side}\n"
            msg += f"Entry: ${entry:.2f}\n"
            msg += f"Exit: ${exit_price:.2f}\n"
            msg += f"P&L: ${pnl:+.2f}"
        else:
            # Trade opened
            msg = f"🔵 *TRADE OPENED*\n\n"
            msg += f"Side: {side}\n"
            msg += f"Entry: ${entry:.2f}"
        
        await self.send_notification(msg)
