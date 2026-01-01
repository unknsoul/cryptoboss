"""
Alerting Module
Provides alert management for the trading system.
"""

from typing import Dict, List, Optional, Callable
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class Alert:
    """Represents a single alert."""
    
    def __init__(
        self, 
        level: AlertLevel, 
        message: str, 
        source: str = "system",
        details: Dict = None
    ):
        self.level = level
        self.message = message
        self.source = source
        self.details = details or {}
        self.timestamp = datetime.now()
        self.acknowledged = False
    
    def to_dict(self) -> Dict:
        return {
            "level": self.level.value,
            "message": self.message,
            "source": self.source,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged
        }


class AlertManager:
    """
    Alert manager for trading system.
    Handles alert creation, storage, and notification dispatch.
    """
    
    def __init__(self, max_alerts: int = 100):
        self.alerts: List[Alert] = []
        self.max_alerts = max_alerts
        self.handlers: List[Callable[[Alert], None]] = []
    
    def send(
        self, 
        level: AlertLevel, 
        message: str, 
        source: str = "system",
        details: Dict = None
    ) -> Alert:
        """
        Send a new alert.
        
        Args:
            level: Alert severity level
            message: Alert message
            source: Source component
            details: Additional details dict
        
        Returns:
            Created Alert object
        """
        alert = Alert(level, message, source, details)
        self.alerts.append(alert)
        
        # Trim old alerts if exceeding max
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]
        
        # Notify handlers
        for handler in self.handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
        
        # Log the alert
        log_method = getattr(logger, level.value if level.value != "emergency" else "critical")
        log_method(f"[ALERT] {source}: {message}")
        
        return alert
    
    def info(self, message: str, source: str = "system", **details):
        """Send info level alert."""
        return self.send(AlertLevel.INFO, message, source, details)
    
    def warning(self, message: str, source: str = "system", **details):
        """Send warning level alert."""
        return self.send(AlertLevel.WARNING, message, source, details)
    
    def critical(self, message: str, source: str = "system", **details):
        """Send critical level alert."""
        return self.send(AlertLevel.CRITICAL, message, source, details)
    
    def emergency(self, message: str, source: str = "system", **details):
        """Send emergency level alert."""
        return self.send(AlertLevel.EMERGENCY, message, source, details)
    
    def add_handler(self, handler: Callable[[Alert], None]):
        """Add a handler to be called when alerts are sent."""
        self.handlers.append(handler)
    
    def get_recent(self, limit: int = 10, level: AlertLevel = None) -> List[Dict]:
        """
        Get recent alerts.
        
        Args:
            limit: Maximum number of alerts to return
            level: Filter by level (optional)
        
        Returns:
            List of alert dictionaries
        """
        alerts = self.alerts
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        return [a.to_dict() for a in alerts[-limit:]]
    
    def get_unacknowledged(self) -> List[Dict]:
        """Get all unacknowledged alerts."""
        return [a.to_dict() for a in self.alerts if not a.acknowledged]
    
    def acknowledge_all(self):
        """Acknowledge all alerts."""
        for alert in self.alerts:
            alert.acknowledged = True
    
    def clear(self):
        """Clear all alerts."""
        self.alerts = []


# Singleton instance
_alerts: Optional[AlertManager] = None


def get_alerts() -> AlertManager:
    """
    Get the singleton alert manager instance.
    
    Returns:
        AlertManager instance
    """
    global _alerts
    
    if _alerts is None:
        _alerts = AlertManager()
    
    return _alerts


__all__ = ['get_alerts', 'AlertManager', 'Alert', 'AlertLevel']
