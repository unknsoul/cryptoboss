"""
Alert Management System
Sends notifications for critical events via multiple channels
"""

import os
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from collections import defaultdict
import json


class AlertManager:
    """
    Multi-channel alerting system
    Supports: Email, Slack, Discord, Webhooks
    Features: Alert throttling, severity levels, rate limiting
    """
    
    SEVERITY_LEVELS = ["INFO", "WARNING", "CRITICAL"]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Alert history for throttling
        self.alert_history = defaultdict(list)
        self.throttle_minutes = self.config.get('throttle_minutes', 5)
        
        # Email configuration
        self.email_enabled = self.config.get('email_enabled', False)
        self.smtp_host = self.config.get('smtp_host', 'smtp.gmail.com')
        self.smtp_port = self.config.get('smtp_port', 587)
        self.smtp_user = self.config.get('smtp_user', os.getenv('SMTP_USER'))
        self.smtp_password = self.config.get('smtp_password', os.getenv('SMTP_PASSWORD'))
        self.email_to = self.config.get('email_to', [])
        
        # Slack configuration
        self.slack_enabled = self.config.get('slack_enabled', False)
        self.slack_webhook = self.config.get('slack_webhook', os.getenv('SLACK_WEBHOOK'))
        
        # Discord configuration
        self.discord_enabled = self.config.get('discord_enabled', False)
        self.discord_webhook = self.config.get('discord_webhook', os.getenv('DISCORD_WEBHOOK'))
        
        # Custom webhook
        self.webhook_enabled = self.config.get('webhook_enabled', False)
        self.webhook_url = self.config.get('webhook_url')
        
        # Alert rules
        self.rules = self._load_default_rules()
        
    def _load_default_rules(self) -> Dict[str, Dict[str, Any]]:
        """Default alert rules"""
        return {
            "position_stuck": {
                "severity": "WARNING",
                "channels": ["slack", "email"],
                "throttle": True
            },
            "daily_loss_limit": {
                "severity": "CRITICAL",
                "channels": ["email", "slack", "discord"],
                "throttle": False
            },
            "api_latency_high": {
                "severity": "WARNING",
                "channels": ["slack"],
                "throttle": True
            },
            "circuit_breaker": {
                "severity": "CRITICAL",
                "channels": ["email", "slack", "discord"],
                "throttle": False
            },
            "exchange_disconnected": {
                "severity": "CRITICAL",
                "channels": ["email", "slack"],
                "throttle": False
            },
            "abnormal_slippage": {
                "severity": "WARNING",
                "channels": ["slack"],
                "throttle": True
            },
            "ml_model_degradation": {
                "severity": "WARNING",
                "channels": ["email"],
                "throttle": True
            }
        }
    
    def _should_throttle(self, alert_type: str) -> bool:
        """Check if alert should be throttled"""
        if alert_type not in self.rules or not self.rules[alert_type].get("throttle", True):
            return False
        
        recent_alerts = self.alert_history[alert_type]
        cutoff_time = datetime.now() - timedelta(minutes=self.throttle_minutes)
        
        # Remove old alerts
        self.alert_history[alert_type] = [
            t for t in recent_alerts if t > cutoff_time
        ]
        
        # Check if we recently sent this alert
        return len(self.alert_history[alert_type]) > 0
    
    def send_alert(self, alert_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Send alert through configured channels
        
        Args:
            alert_type: Type of alert (must match rules)
            message: Alert message
            details: Additional context
        """
        if alert_type not in self.rules:
            print(f"⚠️ Unknown alert type: {alert_type}")
            return
        
        # Check throttling
        if self._should_throttle(alert_type):
            print(f"🔇 Alert throttled: {alert_type}")
            return
        
        rule = self.rules[alert_type]
        severity = rule["severity"]
        channels = rule["channels"]
        
        # Format message
        full_message = self._format_message(alert_type, severity, message, details)
        
        # Send to each channel
        for channel in channels:
            if channel == "email" and self.email_enabled:
                self._send_email(alert_type, severity, full_message)
            elif channel == "slack" and self.slack_enabled:
                self._send_slack(alert_type, severity, full_message)
            elif channel == "discord" and self.discord_enabled:
                self._send_discord(alert_type, severity, full_message)
            elif channel == "webhook" and self.webhook_enabled:
                self._send_webhook(alert_type, severity, full_message, details)
        
        # Record alert
        self.alert_history[alert_type].append(datetime.now())
        print(f"📢 Alert sent: {alert_type} ({severity})")
    
    def _format_message(self, alert_type: str, severity: str, message: str, 
                       details: Optional[Dict[str, Any]] = None) -> str:
        """Format alert message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        msg = f"🚨 [{severity}] Trading Bot Alert\n"
        msg += f"Time: {timestamp}\n"
        msg += f"Type: {alert_type}\n"
        msg += f"Message: {message}\n"
        
        if details:
            msg += "\nDetails:\n"
            for key, value in details.items():
                msg += f"  {key}: {value}\n"
        
        return msg
    
    def _send_email(self, alert_type: str, severity: str, message: str):
        """Send email alert"""
        if not self.smtp_user or not self.smtp_password:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_user
            msg['To'] = ', '.join(self.email_to)
            msg['Subject'] = f"[{severity}] Trading Bot Alert: {alert_type}"
            
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            server.starttls()
            server.login(self.smtp_user, self.smtp_password)
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            print(f"❌ Email send failed: {e}")
    
    def _send_slack(self, alert_type: str, severity: str, message: str):
        """Send Slack message"""
        if not self.slack_webhook:
            return
        
        try:
            # Color based on severity
            color = {
                "INFO": "#36a64f",
                "WARNING": "#ff9800",
                "CRITICAL": "#f44336"
            }.get(severity, "#808080")
            
            payload = {
                "attachments": [{
                    "color": color,
                    "title": f"Trading Bot Alert: {alert_type}",
                    "text": message,
                    "footer": "Trading Bot",
                    "ts": int(datetime.now().timestamp())
                }]
            }
            
            response = requests.post(
                self.slack_webhook,
                json=payload,
                timeout=5
            )
            response.raise_for_status()
            
        except Exception as e:
            print(f"❌ Slack send failed: {e}")
    
    def _send_discord(self, alert_type: str, severity: str, message: str):
        """Send Discord message"""
        if not self.discord_webhook:
            return
        
        try:
            # Color based on severity (Discord uses decimal colors)
            color = {
                "INFO": 3066993,    # Green
                "WARNING": 16761600, # Orange
                "CRITICAL": 15158332 # Red
            }.get(severity, 8421504)  # Gray
            
            payload = {
                "embeds": [{
                    "title": f"🚨 Trading Bot Alert",
                    "description": message,
                    "color": color,
                    "fields": [
                        {"name": "Alert Type", "value": alert_type, "inline": True},
                        {"name": "Severity", "value": severity, "inline": True}
                    ],
                    "timestamp": datetime.now().isoformat()
                }]
            }
            
            response = requests.post(
                self.discord_webhook,
                json=payload,
                timeout=5
            )
            response.raise_for_status()
            
        except Exception as e:
            print(f"❌ Discord send failed: {e}")
    
    def _send_webhook(self, alert_type: str, severity: str, message: str, 
                     details: Optional[Dict[str, Any]] = None):
        """Send to custom webhook"""
        if not self.webhook_url:
            return
        
        try:
            payload = {
                "alert_type": alert_type,
                "severity": severity,
                "message": message,
                "details": details or {},
                "timestamp": datetime.now().isoformat()
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=5
            )
            response.raise_for_status()
            
        except Exception as e:
            print(f"❌ Webhook send failed: {e}")
    
    def test_alerts(self):
        """Test all configured alert channels"""
        print("🧪 Testing alert channels...")
        
        self.send_alert(
            "position_stuck",
            "This is a test alert",
            {"test": True, "channel_test": "all"}
        )
        
        print("✅ Alert test complete")


# Global alert manager
_alert_instance: Optional[AlertManager] = None


def get_alerts(config: Optional[Dict[str, Any]] = None) -> AlertManager:
    """Get or create global alert manager"""
    global _alert_instance
    if _alert_instance is None:
        _alert_instance = AlertManager(config)
    return _alert_instance


if __name__ == "__main__":
    # Test alert manager
    config = {
        "slack_enabled": False,  # Set to True with webhook to test
        "email_enabled": False,  # Set to True with SMTP to test
        "discord_enabled": False # Set to True with webhook to test
    }
    
    alerts = get_alerts(config)
    
    print("\n📢 Simulating alerts...")
    alerts.send_alert(
        "daily_loss_limit",
        "Daily loss limit exceeded",
        {"current_loss": -5.2, "limit": -5.0}
    )
    
    print("\n✅ Alert manager test complete")
