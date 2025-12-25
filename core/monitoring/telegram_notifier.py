"""
Telegram Notifier & Alert System - Enterprise Features #247, #249
Real-time notifications for trades and risk alerts.
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional
import requests
from enum import Enum

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "ℹ️"
    SUCCESS = "✅"
    WARNING = "⚠️"
    CRITICAL = "🚨"
    TRADE_ENTRY = "📈"
    TRADE_EXIT = "📉"
    PROFIT = "💰"
    LOSS = "📛"


class TelegramNotifier:
    """
    Feature #247: Telegram Notification System
    
    Sends real-time alerts for:
    - Trade entries/exits
    - Profit/loss updates
    - Risk warnings
    - Circuit breaker triggers
    - Daily summaries
    """
    
    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        enabled: bool = True
    ):
        """
        Initialize Telegram notifier.
        
        Args:
            bot_token: Telegram bot token (from BotFather)
            chat_id: Target chat ID for notifications
            enabled: Whether notifications are enabled
        """
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        self.enabled = enabled and bool(self.bot_token) and bool(self.chat_id)
        
        self.message_queue: List[Dict] = []
        self.last_send_time: Optional[datetime] = None
        self.daily_message_count = 0
        self.max_daily_messages = 100
        
        if self.enabled:
            logger.info("Telegram Notifier ENABLED ✅")
        else:
            logger.info("Telegram Notifier disabled (no credentials)")
    
    def send(
        self,
        message: str,
        level: AlertLevel = AlertLevel.INFO,
        silent: bool = False
    ) -> bool:
        """
        Send a notification to Telegram.
        
        Args:
            message: Message content
            level: Alert severity level
            silent: If True, send without notification sound
            
        Returns:
            True if sent successfully
        """
        if not self.enabled:
            logger.debug(f"Telegram (disabled): {message}")
            return False
        
        if self.daily_message_count >= self.max_daily_messages:
            logger.warning("Daily Telegram message limit reached")
            return False
        
        formatted = f"{level.value} {message}"
        
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': formatted,
                'parse_mode': 'HTML',
                'disable_notification': silent
            }
            
            response = requests.post(url, json=payload, timeout=5)
            
            if response.status_code == 200:
                self.daily_message_count += 1
                self.last_send_time = datetime.now()
                return True
            else:
                logger.error(f"Telegram send failed: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Telegram error: {e}")
            return False
    
    def notify_trade_entry(
        self,
        side: str,
        price: float,
        size: float,
        confidence: float,
        reasons: List[str]
    ):
        """Send trade entry notification."""
        icon = AlertLevel.TRADE_ENTRY
        msg = (
            f"<b>TRADE OPENED</b>\n"
            f"Side: {side}\n"
            f"Price: ${price:,.2f}\n"
            f"Size: {size:.6f} BTC\n"
            f"Confidence: {confidence:.0%}\n"
            f"Reasons: {', '.join(reasons[:3])}"
        )
        self.send(msg, icon)
    
    def notify_trade_exit(
        self,
        side: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        return_pct: float,
        reason: str
    ):
        """Send trade exit notification."""
        icon = AlertLevel.PROFIT if pnl > 0 else AlertLevel.LOSS
        msg = (
            f"<b>TRADE CLOSED</b>\n"
            f"Side: {side}\n"
            f"Entry: ${entry_price:,.2f} → Exit: ${exit_price:,.2f}\n"
            f"P&L: {'+' if pnl >= 0 else ''}${pnl:.2f} ({return_pct:+.2f}%)\n"
            f"Reason: {reason}"
        )
        self.send(msg, icon)
    
    def notify_risk_alert(self, alert_type: str, message: str):
        """Send risk alert notification."""
        self.send(f"<b>RISK ALERT: {alert_type}</b>\n{message}", AlertLevel.WARNING)
    
    def notify_circuit_breaker(self, reason: str):
        """Send circuit breaker triggered notification."""
        self.send(f"<b>🛑 CIRCUIT BREAKER TRIGGERED</b>\n{reason}", AlertLevel.CRITICAL)
    
    def send_daily_summary(
        self,
        equity: float,
        daily_pnl: float,
        win_rate: float,
        trade_count: int
    ):
        """Send daily performance summary."""
        msg = (
            f"<b>📊 DAILY SUMMARY</b>\n"
            f"Equity: ${equity:,.2f}\n"
            f"Day P&L: {'+' if daily_pnl >= 0 else ''}${daily_pnl:.2f}\n"
            f"Win Rate: {win_rate:.0%}\n"
            f"Trades: {trade_count}"
        )
        self.send(msg, AlertLevel.INFO)
    
    def reset_daily_count(self):
        """Reset daily message counter."""
        self.daily_message_count = 0


class RiskAlerter:
    """
    Feature #249: Risk Alerts Dashboard
    Monitors risk metrics and generates alerts.
    """
    
    def __init__(
        self,
        notifier: Optional[TelegramNotifier] = None,
        drawdown_warning: float = 0.05,    # 5% DD warning
        drawdown_critical: float = 0.08,   # 8% DD critical
        loss_streak_warning: int = 3,
        daily_loss_warning: float = 0.03   # 3% daily loss
    ):
        """
        Initialize risk alerter.
        
        Args:
            notifier: Telegram notifier for sending alerts
            drawdown_warning: Drawdown level for warning
            drawdown_critical: Drawdown level for critical alert
            loss_streak_warning: Consecutive losses for warning
            daily_loss_warning: Daily loss for warning
        """
        self.notifier = notifier
        self.drawdown_warning = drawdown_warning
        self.drawdown_critical = drawdown_critical
        self.loss_streak_warning = loss_streak_warning
        self.daily_loss_warning = daily_loss_warning
        
        self.active_alerts: List[Dict] = []
        self.alert_history: List[Dict] = []
    
    def check_metrics(
        self,
        current_equity: float,
        peak_equity: float,
        daily_start_equity: float,
        consecutive_losses: int
    ) -> List[Dict]:
        """
        Check all risk metrics and generate alerts.
        
        Returns:
            List of triggered alerts
        """
        alerts = []
        now = datetime.now().isoformat()
        
        # Drawdown check
        if peak_equity > 0:
            dd = (peak_equity - current_equity) / peak_equity
            
            if dd >= self.drawdown_critical:
                alerts.append({
                    'type': 'DRAWDOWN_CRITICAL',
                    'level': 'critical',
                    'value': f"{dd:.1%}",
                    'threshold': f"{self.drawdown_critical:.1%}",
                    'timestamp': now
                })
            elif dd >= self.drawdown_warning:
                alerts.append({
                    'type': 'DRAWDOWN_WARNING',
                    'level': 'warning',
                    'value': f"{dd:.1%}",
                    'threshold': f"{self.drawdown_warning:.1%}",
                    'timestamp': now
                })
        
        # Daily loss check
        if daily_start_equity > 0:
            daily_loss = (daily_start_equity - current_equity) / daily_start_equity
            if daily_loss >= self.daily_loss_warning:
                alerts.append({
                    'type': 'DAILY_LOSS_WARNING',
                    'level': 'warning',
                    'value': f"{daily_loss:.1%}",
                    'threshold': f"{self.daily_loss_warning:.1%}",
                    'timestamp': now
                })
        
        # Loss streak check
        if consecutive_losses >= self.loss_streak_warning:
            alerts.append({
                'type': 'LOSS_STREAK',
                'level': 'warning',
                'value': str(consecutive_losses),
                'threshold': str(self.loss_streak_warning),
                'timestamp': now
            })
        
        # Update tracking
        self.active_alerts = alerts
        if alerts:
            self.alert_history.extend(alerts)
            self.alert_history = self.alert_history[-100:]
            
            # Send notifications
            if self.notifier:
                for alert in alerts:
                    self.notifier.notify_risk_alert(
                        alert['type'],
                        f"Current: {alert['value']} (Threshold: {alert['threshold']})"
                    )
        
        return alerts
    
    def get_status(self) -> Dict:
        """Get current alert status."""
        return {
            'active_alerts': self.active_alerts,
            'alert_count': len(self.active_alerts),
            'history_count': len(self.alert_history),
            'has_critical': any(a['level'] == 'critical' for a in self.active_alerts)
        }


# Singleton instances
_notifier: Optional[TelegramNotifier] = None
_risk_alerter: Optional[RiskAlerter] = None


def get_telegram_notifier() -> TelegramNotifier:
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier


def get_risk_alerter() -> RiskAlerter:
    global _risk_alerter
    if _risk_alerter is None:
        _risk_alerter = RiskAlerter(notifier=get_telegram_notifier())
    return _risk_alerter


if __name__ == '__main__':
    # Test (won't actually send without credentials)
    notifier = TelegramNotifier()
    notifier.notify_trade_entry('LONG', 50000, 0.01, 0.75, ['RSI oversold', 'Volume spike'])
    notifier.notify_trade_exit('LONG', 50000, 51000, 100, 2.0, 'Take Profit')
    
    alerter = RiskAlerter()
    alerts = alerter.check_metrics(
        current_equity=9500,
        peak_equity=10000,
        daily_start_equity=9800,
        consecutive_losses=4
    )
    print(f"Active alerts: {alerts}")
