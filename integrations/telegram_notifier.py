"""
Telegram Notification System
Real-time alerts for trading events via Telegram

Setup Instructions:
1. Create bot via @BotFather on Telegram
2. Get bot token
3. Get your chat_id by messaging @userinfobot
4. Set environment variables or update config
"""

import os
import logging
from typing import Optional, Dict, List
from datetime import datetime
import requests

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Professional Telegram notification system
    
    Features:
    - Trade execution alerts
    - Signal notifications  
    - Performance updates
    - Risk alerts (circuit breaker, drawdown)
    - Custom message formatting
    """
    
    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        """
        Initialize Telegram notifier
        
        Args:
            bot_token: Telegram bot token from @BotFather
            chat_id: Your Telegram chat ID (get from @userinfobot)
        """
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        
        self.enabled = bool(self.bot_token and self.chat_id)
        
        if not self.enabled:
            logger.warning("Telegram notifications disabled - missing bot_token or chat_id")
            logger.info("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables to enable")
        else:
            logger.info("✓ Telegram notifications enabled")
        
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
    
    def send_message(self, message: str, parse_mode: str = 'HTML') -> bool:
        """
        Send a message via Telegram
        
        Args:
            message: Message text (supports HTML/Markdown)
            parse_mode: 'HTML' or 'Markdown'
            
        Returns:
            True if sent successfully
        """
        if not self.enabled:
            logger.debug(f"Telegram disabled, would have sent: {message}")
            return False
        
        try:
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            
            response = requests.post(self.api_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.debug(f"Telegram message sent: {message[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    def notify_trade_execution(self, trade: Dict) -> bool:
        """
        Send trade execution alert
        
        Args:
            trade: Trade details dict
        """
        action = trade.get('action', 'UNKNOWN')
        price = trade.get('price', 0)
        size = trade.get('size', 0)
        confidence = trade.get('confidence', 0)
        pnl = trade.get('pnl', 0)
        
        # Emoji based on action
        emoji = "🟢" if action == 'LONG' else "🔴" if action == 'SHORT' else "⚪"
        
        message = f"""
{emoji} <b>TRADE EXECUTED</b>

<b>Action:</b> {action}
<b>Price:</b> ${price:,.2f}
<b>Size:</b> {size:.4f} BTC
<b>Value:</b> ${price * size:,.2f}
<b>Confidence:</b> {confidence*100:.1f}%
"""
        
        if pnl != 0:
            pnl_emoji = "💰" if pnl > 0 else "📉"
            message += f"""{pnl_emoji} <b>P&L:</b> ${pnl:,.2f} ({pnl/trade.get('entry_value', 1)*100:+.2f}%)
"""
        
        reasons = trade.get('reasons', [])
        if reasons:
            message += f"\n<b>Reasons:</b>\n"
            for reason in reasons[:3]:  # Limit to top 3
                message += f"• {reason}\n"
        
        timestamp = trade.get('timestamp', datetime.now().strftime('%H:%M:%S'))
        message += f"\n<i>Time: {timestamp}</i>"
        
        return self.send_message(message)
    
    def notify_signal(self, signal: Dict) -> bool:
        """
        Send signal alert
        
        Args:
            signal: Signal details dict
        """
        action = signal.get('action', 'HOLD')
        confidence = signal.get('confidence', 0)
        price = signal.get('price', 0)
        
        if action == 'HOLD' or confidence < 0.5:
            return False  # Don't spam with low-confidence signals
        
        emoji = "📈" if action == 'LONG' else "📉"
        
        message = f"""
{emoji} <b>SIGNAL GENERATED</b>

<b>Action:</b> {action}
<b>Price:</b> ${price:,.2f}
<b>Confidence:</b> {confidence*100:.1f}%

<i>Awaiting execution...</i>
"""
        
        return self.send_message(message)
    
    def notify_position_close(self, position: Dict) -> bool:
        """
        Send position close alert
        
        Args:
            position: Position details dict
        """
        pnl = position.get('pnl', 0)
        pnl_pct = position.get('pnl_pct', 0)
        hold_time = position.get('hold_time', 0)
        
        emoji = "✅" if pnl > 0 else "❌"
        
        message = f"""
{emoji} <b>POSITION CLOSED</b>

<b>Side:</b> {position.get('side', 'UNKNOWN')}
<b>Entry:</b> ${position.get('entry_price', 0):,.2f}
<b>Exit:</b> ${position.get('exit_price', 0):,.2f}
<b>Size:</b> {position.get('size', 0):.4f} BTC

<b>P&L:</b> ${pnl:,.2f} ({pnl_pct:+.2f}%)
<b>Hold Time:</b> {hold_time}

<i>{datetime.now().strftime('%H:%M:%S')}</i>
"""
        
        return self.send_message(message)
    
    def notify_performance_update(self, metrics: Dict) -> bool:
        """
        Send daily/periodic performance update
        
        Args:
            metrics: Performance metrics dict
        """
        equity = metrics.get('equity', 0)
        total_pnl = metrics.get('total_pnl', 0)
        win_rate = metrics.get('win_rate', 0)
        total_trades = metrics.get('total_trades', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        
        message = f"""
📊 <b>PERFORMANCE UPDATE</b>

<b>Equity:</b> ${equity:,.2f}
<b>Total P&L:</b> ${total_pnl:,.2f}
<b>Total Trades:</b> {total_trades}
<b>Win Rate:</b> {win_rate*100:.1f}%
<b>Sharpe Ratio:</b> {sharpe:.2f}

<i>{datetime.now().strftime('%Y-%m-%d %H:%M')}</i>
"""
        
        return self.send_message(message)
    
    def notify_risk_alert(self, alert_type: str, details: Dict) -> bool:
        """
        Send risk management alert
        
        Args:
            alert_type: 'circuit_breaker', 'drawdown', 'var_breach', etc.
            details: Alert-specific details
        """
        emoji_map = {
            'circuit_breaker': '🛑',
            'drawdown': '⚠️',
            'var_breach': '🚨',
            'high_slippage': '⚡',
            'api_error': '❗'
        }
        
        emoji = emoji_map.get(alert_type, '⚠️')
        
        message = f"""
{emoji} <b>RISK ALERT: {alert_type.upper().replace('_', ' ')}</b>

"""
        
        for key, value in details.items():
            message += f"<b>{key.replace('_', ' ').title()}:</b> {value}\n"
        
        message += f"\n<i>{datetime.now().strftime('%H:%M:%S')}</i>"
        
        return self.send_message(message)
    
    def notify_system_status(self, status: str, uptime: str = None) -> bool:
        """
        Send system status update
        
        Args:
            status: 'started', 'stopped', 'error'
            uptime: Optional uptime string
        """
        emoji_map = {
            'started': '🟢',
            'stopped': '🔴',
            'error': '❌',
            'restart': '🔄'
        }
        
        emoji = emoji_map.get(status, '⚪')
        
        message = f"""
{emoji} <b>SYSTEM {status.upper()}</b>

<b>Bot:</b> CryptoBoss Pro
<b>Status:</b> {status.title()}
"""
        
        if uptime:
            message += f"<b>Uptime:</b> {uptime}\n"
        
        message += f"\n<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"
        
        return self.send_message(message)
    
    def test_connection(self) -> bool:
        """
        Test Telegram connection
        
        Returns:
            True if connection works
        """
        if not self.enabled:
            print("❌ Telegram not enabled - set bot_token and chat_id")
            return False
        
        message = "✅ Telegram connection test successful!"
        result = self.send_message(message)
        
        if result:
            print("✅ Telegram test message sent successfully")
            return True
        else:
            print("❌ Failed to send test message")
            return False


# Singleton instance
_telegram_notifier: Optional[TelegramNotifier] = None


def get_telegram_notifier() -> TelegramNotifier:
    """Get singleton Telegram notifier"""
    global _telegram_notifier
    if _telegram_notifier is None:
        _telegram_notifier = TelegramNotifier()
    return _telegram_notifier


if __name__ == '__main__':
    # Test the Telegram notifier
    print("=" * 60)
    print("TELEGRAM NOTIFIER - SETUP & TEST")
    print("=" * 60)
    
    print("\n📱 Telegram Bot Setup Instructions:")
    print("1. Open Telegram and search for @BotFather")
    print("2. Send /newbot and follow instructions")
    print("3. Copy the bot token")
    print("4. Search for @userinfobot to get your chat_id")
    print("5. Set environment variables:")
    print("   export TELEGRAM_BOT_TOKEN='your_token_here'")
    print("   export TELEGRAM_CHAT_ID='your_chat_id_here'")
    
    # Initialize (will check environment variables)
    notifier = TelegramNotifier()
    
    if notifier.enabled:
        print("\n✅ Telegram configuration found!")
        print("\n🧪 Running connection test...")
        
        # Test connection
        if notifier.test_connection():
            print("\n📊 Sending sample notifications...")
            
            # Test trade notification
            sample_trade = {
                'action': 'LONG',
                'price': 87500.50,
                'size': 0.15,
                'confidence': 0.75,
                'reasons': ['Strong momentum', 'HTF trend UP', 'Volume confirmation'],
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }
            notifier.notify_trade_execution(sample_trade)
            
            # Test performance update
            sample_metrics = {
                'equity': 10234.56,
                'total_pnl': 234.56,
                'win_rate': 0.68,
                'total_trades': 47,
                'sharpe_ratio': 2.15
            }
            notifier.notify_performance_update(sample_metrics)
            
            print("\n✅ All test notifications sent!")
        
    else:
        print("\n⚠️  Telegram not configured (running in disabled mode)")
        print("Set environment variables to enable notifications")
    
    print("\n" + "=" * 60)
