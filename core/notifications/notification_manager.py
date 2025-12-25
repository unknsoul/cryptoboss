"""
Notifications & Alerts - Enterprise Features #230, #235, #240, #245
Discord, Email, Webhooks, and Alert Aggregation.
"""

import logging
import smtplib
import json
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from enum import Enum
import requests
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)


class AlertPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class DiscordWebhook:
    """
    Feature #230: Discord Webhook Integration
    
    Sends notifications to Discord channels.
    """
    
    def __init__(self, webhook_url: Optional[str] = None):
        """
        Initialize Discord webhook.
        
        Args:
            webhook_url: Discord webhook URL
        """
        self.webhook_url = webhook_url or os.getenv('DISCORD_WEBHOOK_URL')
        self.enabled = bool(self.webhook_url)
        self.message_count = 0
        
        if self.enabled:
            logger.info("Discord Webhook ENABLED ✅")
        else:
            logger.info("Discord Webhook disabled (no URL)")
    
    def send(
        self,
        content: str,
        username: str = "Trading Bot",
        embed: Optional[Dict] = None
    ) -> bool:
        """
        Send a message to Discord.
        
        Args:
            content: Message content
            username: Bot username
            embed: Optional embed object
            
        Returns:
            True if sent successfully
        """
        if not self.enabled:
            logger.debug(f"Discord (disabled): {content}")
            return False
        
        payload = {
            'username': username,
            'content': content
        }
        
        if embed:
            payload['embeds'] = [embed]
        
        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=5
            )
            self.message_count += 1
            return response.status_code in [200, 204]
        except Exception as e:
            logger.error(f"Discord send failed: {e}")
            return False
    
    def send_trade_alert(
        self,
        action: str,
        side: str,
        price: float,
        pnl: Optional[float] = None
    ):
        """Send formatted trade alert."""
        color = 0x00ff00 if side == 'LONG' else 0xff0000
        
        if pnl is not None:
            color = 0x00ff00 if pnl > 0 else 0xff0000
            title = f"💰 Trade Closed: {'+' if pnl > 0 else ''}${pnl:.2f}"
        else:
            title = f"📊 {action} {side}"
        
        embed = {
            'title': title,
            'color': color,
            'fields': [
                {'name': 'Side', 'value': side, 'inline': True},
                {'name': 'Price', 'value': f'${price:,.2f}', 'inline': True}
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        if pnl is not None:
            embed['fields'].append({'name': 'P&L', 'value': f'${pnl:.2f}', 'inline': True})
        
        self.send("", embed=embed)
    
    def send_risk_alert(self, alert_type: str, message: str):
        """Send risk warning alert."""
        embed = {
            'title': f"⚠️ Risk Alert: {alert_type}",
            'description': message,
            'color': 0xff6600,
            'timestamp': datetime.now().isoformat()
        }
        self.send("", embed=embed)


class EmailAlertSystem:
    """
    Feature #235: Email Alert System
    
    Sends email notifications for important events.
    """
    
    def __init__(
        self,
        smtp_server: Optional[str] = None,
        smtp_port: int = 587,
        username: Optional[str] = None,
        password: Optional[str] = None,
        from_email: Optional[str] = None,
        to_emails: Optional[List[str]] = None
    ):
        """
        Initialize email system.
        
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP port
            username: SMTP username
            password: SMTP password
            from_email: Sender email
            to_emails: Recipient emails
        """
        self.smtp_server = smtp_server or os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = smtp_port
        self.username = username or os.getenv('SMTP_USERNAME')
        self.password = password or os.getenv('SMTP_PASSWORD')
        self.from_email = from_email or os.getenv('ALERT_FROM_EMAIL')
        self.to_emails = to_emails or os.getenv('ALERT_TO_EMAILS', '').split(',')
        
        self.enabled = all([self.smtp_server, self.username, self.password, self.from_email])
        
        if self.enabled:
            logger.info("Email Alert System ENABLED ✅")
        else:
            logger.info("Email Alert System disabled (missing config)")
    
    def send(
        self,
        subject: str,
        body: str,
        priority: AlertPriority = AlertPriority.MEDIUM
    ) -> bool:
        """
        Send an email alert.
        
        Args:
            subject: Email subject
            body: Email body (HTML supported)
            priority: Alert priority
            
        Returns:
            True if sent successfully
        """
        if not self.enabled:
            logger.debug(f"Email (disabled): {subject}")
            return False
        
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{priority.name}] {subject}"
            msg['From'] = self.from_email
            msg['To'] = ', '.join([e for e in self.to_emails if e])
            
            # HTML body
            html = f"""
            <html>
            <body>
            <h2>{subject}</h2>
            <p>{body}</p>
            <hr>
            <small>Sent at {datetime.now().isoformat()}</small>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'plain'))
            msg.attach(MIMEText(html, 'html'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.from_email, self.to_emails, msg.as_string())
            
            logger.info(f"Email sent: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return False
    
    def send_daily_report(self, metrics: Dict):
        """Send daily performance report."""
        subject = f"Daily Trading Report - {datetime.now().strftime('%Y-%m-%d')}"
        
        body = f"""
        Daily Trading Summary:
        
        Equity: ${metrics.get('equity', 0):,.2f}
        Day P&L: ${metrics.get('daily_pnl', 0):,.2f}
        Trades: {metrics.get('trade_count', 0)}
        Win Rate: {metrics.get('win_rate', 0):.1%}
        """
        
        self.send(subject, body, AlertPriority.LOW)


class WebhookEventSystem:
    """
    Feature #240: Webhook Event System
    
    Sends events to custom webhook endpoints.
    """
    
    def __init__(self):
        """Initialize webhook system."""
        self.webhooks: Dict[str, Dict] = {}  # event_type -> webhook config
        self.event_log: List[Dict] = []
        
        logger.info("Webhook Event System initialized")
    
    def register_webhook(
        self,
        event_type: str,
        url: str,
        headers: Optional[Dict] = None,
        method: str = 'POST'
    ):
        """Register a webhook for an event type."""
        self.webhooks[event_type] = {
            'url': url,
            'headers': headers or {},
            'method': method
        }
        logger.info(f"Webhook registered for: {event_type}")
    
    def emit(self, event_type: str, data: Dict) -> bool:
        """
        Emit an event to registered webhook.
        
        Args:
            event_type: Type of event
            data: Event data
            
        Returns:
            True if sent successfully
        """
        if event_type not in self.webhooks:
            return False
        
        webhook = self.webhooks[event_type]
        
        payload = {
            'event': event_type,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        try:
            if webhook['method'] == 'POST':
                response = requests.post(
                    webhook['url'],
                    json=payload,
                    headers=webhook['headers'],
                    timeout=5
                )
            else:
                response = requests.get(
                    webhook['url'],
                    params={'payload': json.dumps(payload)},
                    headers=webhook['headers'],
                    timeout=5
                )
            
            success = response.status_code < 400
            
            self.event_log.append({
                'event': event_type,
                'timestamp': datetime.now().isoformat(),
                'success': success,
                'status_code': response.status_code
            })
            self.event_log = self.event_log[-100:]
            
            return success
            
        except Exception as e:
            logger.error(f"Webhook emit failed: {e}")
            return False
    
    def get_event_log(self, event_type: Optional[str] = None) -> List[Dict]:
        """Get event log, optionally filtered."""
        if event_type:
            return [e for e in self.event_log if e['event'] == event_type]
        return self.event_log


class AlertAggregator:
    """
    Feature #245: Alert Aggregation
    
    Aggregates and deduplicates alerts to prevent spam.
    """
    
    def __init__(
        self,
        dedup_window_minutes: int = 5,
        max_alerts_per_hour: int = 50
    ):
        """
        Initialize alert aggregator.
        
        Args:
            dedup_window_minutes: Minutes to deduplicate same alerts
            max_alerts_per_hour: Maximum alerts per hour
        """
        self.dedup_window = timedelta(minutes=dedup_window_minutes)
        self.max_per_hour = max_alerts_per_hour
        
        self.recent_alerts: Dict[str, datetime] = {}
        self.hourly_counts: List[datetime] = []
        self.pending_alerts: List[Dict] = []
        self.handlers: List[Callable] = []
        
        logger.info(f"Alert Aggregator initialized - Dedup: {dedup_window_minutes}min")
    
    def add_handler(self, handler: Callable[[Dict], None]):
        """Add an alert handler function."""
        self.handlers.append(handler)
    
    def submit(
        self,
        alert_type: str,
        message: str,
        priority: AlertPriority = AlertPriority.MEDIUM,
        data: Optional[Dict] = None
    ) -> bool:
        """
        Submit an alert for processing.
        
        Args:
            alert_type: Type of alert
            message: Alert message
            priority: Alert priority
            data: Additional data
            
        Returns:
            True if alert was sent (not deduplicated or rate-limited)
        """
        now = datetime.now()
        alert_key = f"{alert_type}:{message}"
        
        # Check deduplication
        if alert_key in self.recent_alerts:
            if now - self.recent_alerts[alert_key] < self.dedup_window:
                logger.debug(f"Alert deduplicated: {alert_type}")
                return False
        
        # Check rate limit (except critical)
        if priority != AlertPriority.CRITICAL:
            hour_ago = now - timedelta(hours=1)
            self.hourly_counts = [t for t in self.hourly_counts if t > hour_ago]
            
            if len(self.hourly_counts) >= self.max_per_hour:
                logger.warning("Alert rate limit reached")
                return False
        
        # Process alert
        self.recent_alerts[alert_key] = now
        self.hourly_counts.append(now)
        
        alert = {
            'type': alert_type,
            'message': message,
            'priority': priority.name,
            'timestamp': now.isoformat(),
            'data': data or {}
        }
        
        # Send to handlers
        for handler in self.handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
        
        return True
    
    def get_stats(self) -> Dict:
        """Get aggregator statistics."""
        hour_ago = datetime.now() - timedelta(hours=1)
        return {
            'alerts_last_hour': len([t for t in self.hourly_counts if t > hour_ago]),
            'max_per_hour': self.max_per_hour,
            'unique_alert_types': len(self.recent_alerts),
            'handler_count': len(self.handlers)
        }


# Singletons
_discord: Optional[DiscordWebhook] = None
_email: Optional[EmailAlertSystem] = None
_webhook: Optional[WebhookEventSystem] = None
_aggregator: Optional[AlertAggregator] = None


def get_discord() -> DiscordWebhook:
    global _discord
    if _discord is None:
        _discord = DiscordWebhook()
    return _discord


def get_email_alerts() -> EmailAlertSystem:
    global _email
    if _email is None:
        _email = EmailAlertSystem()
    return _email


def get_webhook_system() -> WebhookEventSystem:
    global _webhook
    if _webhook is None:
        _webhook = WebhookEventSystem()
    return _webhook


def get_alert_aggregator() -> AlertAggregator:
    global _aggregator
    if _aggregator is None:
        _aggregator = AlertAggregator()
    return _aggregator


if __name__ == '__main__':
    # Test alert aggregator
    agg = AlertAggregator(dedup_window_minutes=1)
    
    received = []
    agg.add_handler(lambda a: received.append(a))
    
    # First alert - should send
    result1 = agg.submit('TEST', 'Test message', AlertPriority.MEDIUM)
    print(f"First alert sent: {result1}")
    
    # Same alert - should be deduplicated
    result2 = agg.submit('TEST', 'Test message', AlertPriority.MEDIUM)
    print(f"Same alert sent: {result2}")
    
    # Different alert - should send
    result3 = agg.submit('TEST', 'Different message', AlertPriority.HIGH)
    print(f"Different alert sent: {result3}")
    
    print(f"Received alerts: {len(received)}")
    print(f"Stats: {agg.get_stats()}")
