"""Alert delivery utilities."""

from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from typing import Optional

import requests


class AlertManager:
    """Send alerts to Telegram, Discord, or email."""

    def __init__(self) -> None:
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_user_id = os.getenv("TELEGRAM_USER_ID")
        self.discord_webhook = os.getenv("DISCORD_WEBHOOK_URL")
        self.smtp_host = os.getenv("SMTP_HOST")
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_from = os.getenv("SMTP_FROM")
        self.smtp_to = os.getenv("SMTP_TO")

    def send_telegram(self, message: str) -> bool:
        if not self.telegram_token or not self.telegram_user_id:
            return False
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        response = requests.post(url, json={"chat_id": self.telegram_user_id, "text": message}, timeout=10)
        return response.ok

    def send_discord(self, message: str) -> bool:
        if not self.discord_webhook:
            return False
        response = requests.post(self.discord_webhook, json={"content": message}, timeout=10)
        return response.ok

    def send_email(self, subject: str, message: str) -> bool:
        if not (self.smtp_host and self.smtp_user and self.smtp_password and self.smtp_from and self.smtp_to):
            return False

        email = EmailMessage()
        email["Subject"] = subject
        email["From"] = self.smtp_from
        email["To"] = self.smtp_to
        email.set_content(message)

        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.starttls()
            server.login(self.smtp_user, self.smtp_password)
            server.send_message(email)
        return True

    def send_alert(self, message: str, subject: Optional[str] = None) -> None:
        """Send alert to all configured channels."""
        self.send_telegram(message)
        self.send_discord(message)
        if subject:
            self.send_email(subject, message)
