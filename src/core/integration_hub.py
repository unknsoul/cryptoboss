"""
Unified Integration Hub - Upgrade E

Central registration and lifecycle management for all external services:
- Telegram Bot
- Email Reporter
- Discord Bot
- Slack Webhook
- REST API
- WebSocket feeds

Benefits:
- Single start()/stop() for all integrations
- Consistent error handling
- Health monitoring
- Easy to add new channels
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class IntegrationStatus(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"
    STOPPING = "stopping"


@dataclass
class IntegrationHealth:
    """Health status for an integration."""
    name: str
    status: IntegrationStatus
    last_heartbeat: datetime
    error_count: int
    last_error: Optional[str]
    uptime_seconds: float


class BaseIntegration(ABC):
    """Base class for all integrations."""
    
    def __init__(self, name: str):
        self.name = name
        self.status = IntegrationStatus.STOPPED
        self.started_at: Optional[datetime] = None
        self.error_count = 0
        self.last_error: Optional[str] = None
        self.last_heartbeat = datetime.now()
    
    @abstractmethod
    async def start(self):
        """Start the integration."""
        pass
    
    @abstractmethod
    async def stop(self):
        """Stop the integration."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if integration is healthy."""
        pass
    
    def get_health(self) -> IntegrationHealth:
        uptime = 0.0
        if self.started_at:
            uptime = (datetime.now() - self.started_at).total_seconds()
        
        return IntegrationHealth(
            name=self.name,
            status=self.status,
            last_heartbeat=self.last_heartbeat,
            error_count=self.error_count,
            last_error=self.last_error,
            uptime_seconds=uptime
        )
    
    def record_error(self, error: str):
        self.error_count += 1
        self.last_error = error
        logger.error(f"[{self.name}] Error: {error}")
    
    def record_heartbeat(self):
        self.last_heartbeat = datetime.now()


class TelegramIntegration(BaseIntegration):
    """Telegram bot integration."""
    
    def __init__(self, bot_token: str, authorized_users: List[int], trading_bot=None):
        super().__init__("telegram")
        self.bot_token = bot_token
        self.authorized_users = authorized_users
        self.trading_bot = trading_bot
        self._bot = None
    
    async def start(self):
        self.status = IntegrationStatus.STARTING
        try:
            # Import here to avoid dependency if not used
            from src.integrations.telegram_bot import TelegramTradingBot
            
            self._bot = TelegramTradingBot(
                token=self.bot_token,
                authorized_users=self.authorized_users,
                trading_bot_instance=self.trading_bot,
                exchange_client=None
            )
            await self._bot.start_async()
            
            self.status = IntegrationStatus.RUNNING
            self.started_at = datetime.now()
            logger.info("Telegram integration started")
        except Exception as e:
            self.record_error(str(e))
            self.status = IntegrationStatus.ERROR
    
    async def stop(self):
        self.status = IntegrationStatus.STOPPING
        if self._bot:
            await self._bot.stop_async()
        self.status = IntegrationStatus.STOPPED
        logger.info("Telegram integration stopped")
    
    async def health_check(self) -> bool:
        self.record_heartbeat()
        return self.status == IntegrationStatus.RUNNING
    
    async def send_alert(self, message: str):
        if self._bot:
            await self._bot.send_alert(self.authorized_users, message)


class EmailIntegration(BaseIntegration):
    """Email notification integration."""
    
    def __init__(self, smtp_host: str, smtp_port: int, username: str, 
                 password: str, from_addr: str, to_addrs: List[str]):
        super().__init__("email")
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs
    
    async def start(self):
        self.status = IntegrationStatus.RUNNING
        self.started_at = datetime.now()
        logger.info("Email integration started")
    
    async def stop(self):
        self.status = IntegrationStatus.STOPPED
        logger.info("Email integration stopped")
    
    async def health_check(self) -> bool:
        self.record_heartbeat()
        return True
    
    async def send_email(self, subject: str, body: str):
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_addr
            msg['To'] = ", ".join(self.to_addrs)
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'html'))
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.from_addr, self.to_addrs, msg.as_string())
            
            logger.info(f"Email sent: {subject}")
        except Exception as e:
            self.record_error(str(e))


class DiscordIntegration(BaseIntegration):
    """Discord webhook integration."""
    
    def __init__(self, webhook_url: str):
        super().__init__("discord")
        self.webhook_url = webhook_url
    
    async def start(self):
        self.status = IntegrationStatus.RUNNING
        self.started_at = datetime.now()
        logger.info("Discord integration started")
    
    async def stop(self):
        self.status = IntegrationStatus.STOPPED
        logger.info("Discord integration stopped")
    
    async def health_check(self) -> bool:
        self.record_heartbeat()
        return True
    
    async def send_message(self, content: str, embed: Dict = None):
        import aiohttp
        
        payload = {"content": content}
        if embed:
            payload["embeds"] = [embed]
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as resp:
                    if resp.status != 204:
                        self.record_error(f"Discord webhook failed: {resp.status}")
        except Exception as e:
            self.record_error(str(e))


class SlackIntegration(BaseIntegration):
    """Slack webhook integration."""
    
    def __init__(self, webhook_url: str):
        super().__init__("slack")
        self.webhook_url = webhook_url
    
    async def start(self):
        self.status = IntegrationStatus.RUNNING
        self.started_at = datetime.now()
        logger.info("Slack integration started")
    
    async def stop(self):
        self.status = IntegrationStatus.STOPPED
        logger.info("Slack integration stopped")
    
    async def health_check(self) -> bool:
        self.record_heartbeat()
        return True
    
    async def send_message(self, text: str, blocks: List[Dict] = None):
        import aiohttp
        
        payload = {"text": text}
        if blocks:
            payload["blocks"] = blocks
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as resp:
                    if resp.status != 200:
                        self.record_error(f"Slack webhook failed: {resp.status}")
        except Exception as e:
            self.record_error(str(e))


class IntegrationHub:
    """
    Central hub for managing all integrations.
    
    Usage:
        hub = IntegrationHub()
        
        # Register integrations
        hub.register(TelegramIntegration(token="...", users=[123]))
        hub.register(DiscordIntegration(webhook="..."))
        
        # Start all
        await hub.start_all()
        
        # Send to all channels
        await hub.broadcast("Trade executed: BUY 0.1 BTC @ $65,000")
        
        # Get health
        health = hub.get_health_report()
        
        # Stop all
        await hub.stop_all()
    """
    
    def __init__(self):
        self.integrations: Dict[str, BaseIntegration] = {}
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False
        logger.info("IntegrationHub initialized")
    
    def register(self, integration: BaseIntegration):
        """Register an integration."""
        self.integrations[integration.name] = integration
        logger.info(f"Registered integration: {integration.name}")
    
    def unregister(self, name: str):
        """Unregister an integration."""
        if name in self.integrations:
            del self.integrations[name]
            logger.info(f"Unregistered integration: {name}")
    
    async def start_all(self):
        """Start all registered integrations."""
        self._running = True
        
        for name, integration in self.integrations.items():
            try:
                await integration.start()
                logger.info(f"Started integration: {name}")
            except Exception as e:
                logger.error(f"Failed to start {name}: {e}")
                integration.record_error(str(e))
        
        # Start health check loop
        self._health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def stop_all(self):
        """Stop all integrations."""
        self._running = False
        
        if self._health_check_task:
            self._health_check_task.cancel()
        
        for name, integration in self.integrations.items():
            try:
                await integration.stop()
                logger.info(f"Stopped integration: {name}")
            except Exception as e:
                logger.error(f"Failed to stop {name}: {e}")
    
    async def start_one(self, name: str):
        """Start a specific integration."""
        if name in self.integrations:
            await self.integrations[name].start()
    
    async def stop_one(self, name: str):
        """Stop a specific integration."""
        if name in self.integrations:
            await self.integrations[name].stop()
    
    async def broadcast(self, message: str, level: str = "info"):
        """Broadcast message to all active integrations."""
        for name, integration in self.integrations.items():
            if integration.status != IntegrationStatus.RUNNING:
                continue
            
            try:
                if isinstance(integration, TelegramIntegration):
                    await integration.send_alert(message)
                elif isinstance(integration, DiscordIntegration):
                    await integration.send_message(message)
                elif isinstance(integration, SlackIntegration):
                    await integration.send_message(message)
                elif isinstance(integration, EmailIntegration):
                    # Only email for important messages
                    if level in ("warning", "error", "critical"):
                        await integration.send_email(
                            subject=f"CryptoBoss Alert [{level.upper()}]",
                            body=message
                        )
            except Exception as e:
                logger.error(f"Failed to broadcast to {name}: {e}")
    
    async def _health_check_loop(self):
        """Periodic health check for all integrations."""
        while self._running:
            try:
                for name, integration in self.integrations.items():
                    try:
                        is_healthy = await integration.health_check()
                        if not is_healthy:
                            logger.warning(f"Integration {name} is unhealthy")
                    except Exception as e:
                        integration.record_error(str(e))
                
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
    
    def get_health_report(self) -> Dict[str, IntegrationHealth]:
        """Get health status of all integrations."""
        return {name: integration.get_health() 
                for name, integration in self.integrations.items()}
    
    def get(self, name: str) -> Optional[BaseIntegration]:
        """Get a specific integration."""
        return self.integrations.get(name)


# Singleton
_integration_hub: Optional[IntegrationHub] = None

def get_integration_hub() -> IntegrationHub:
    global _integration_hub
    if _integration_hub is None:
        _integration_hub = IntegrationHub()
    return _integration_hub
