"""
Professional Trading Logger
Structured JSON logging with rotation and multiple handlers
"""

import json
import logging
import logging.handlers
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class TradingLogger:
    """
    Centralized logging system for trading bot
    Features:
    - JSON structured logging
    - Multiple log levels
    - File rotation
    - Separate files for trades, errors, and general logs
    """
    
    def __init__(self, log_dir: str = "logs", level: str = "INFO"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create separate loggers
        self.main_logger = self._create_logger("trading_bot", "main.log", level)
        self.trade_logger = self._create_logger("trades", "trades.log", "INFO")
        self.error_logger = self._create_logger("errors", "errors.log", "ERROR")
        self.performance_logger = self._create_logger("performance", "performance.log", "INFO")
        
        # Console handler for important messages
        self._setup_console_handler()
        
    def _create_logger(self, name: str, filename: str, level: str) -> logging.Logger:
        """Create a logger with file rotation"""
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # Prevent duplicate handlers
        if logger.handlers:
            return logger
        
        # Rotating file handler (10MB per file, keep 10 files)
        file_path = self.log_dir / filename
        handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=10
        )
        
        # JSON formatter
        formatter = logging.Formatter(
            '{"timestamp":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":%(message)s}'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _setup_console_handler(self):
        """Add console output for main logger"""
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console.setFormatter(formatter)
        self.main_logger.addHandler(console)
    
    def _format_message(self, message: str, extra: Optional[Dict[str, Any]] = None) -> str:
        """Format message as JSON"""
        data = {"msg": message}
        if extra:
            data.update(extra)
        return json.dumps(data)
    
    # Main logging methods
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.main_logger.info(self._format_message(message, kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.main_logger.warning(self._format_message(message, kwargs))
    
    def error(self, message: str, **kwargs):
        """Log error message to both main and error logs"""
        formatted = self._format_message(message, kwargs)
        self.main_logger.error(formatted)
        self.error_logger.error(formatted)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.main_logger.debug(self._format_message(message, kwargs))
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        formatted = self._format_message(message, kwargs)
        self.main_logger.critical(formatted)
        self.error_logger.critical(formatted)
    
    # Specialized logging methods
    def log_trade(self, trade_data: Dict[str, Any]):
        """Log a trade execution"""
        trade_data['timestamp'] = datetime.now().isoformat()
        self.trade_logger.info(json.dumps(trade_data))
        self.info("Trade executed", **trade_data)
    
    def log_signal(self, signal_data: Dict[str, Any]):
        """Log a trading signal"""
        signal_data['timestamp'] = datetime.now().isoformat()
        self.main_logger.info(self._format_message("Signal generated", signal_data))
    
    def log_performance(self, metrics: Dict[str, Any]):
        """Log performance metrics"""
        metrics['timestamp'] = datetime.now().isoformat()
        self.performance_logger.info(json.dumps(metrics))
    
    def log_api_call(self, endpoint: str, method: str, status: str, latency_ms: float, error: Optional[str] = None):
        """Log API call details"""
        data = {
            "endpoint": endpoint,
            "method": method,
            "status": status,
            "latency_ms": round(latency_ms, 2),
            "timestamp": datetime.now().isoformat()
        }
        if error:
            data["error"] = error
            self.error(f"API call failed: {endpoint}", **data)
        else:
            self.debug(f"API call: {endpoint}", **data)
    
    def log_position_update(self, symbol: str, action: str, quantity: float, price: float, **kwargs):
        """Log position changes"""
        data = {
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": price,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.info(f"Position {action}: {symbol}", **data)
    
    def log_risk_event(self, event_type: str, severity: str, details: Dict[str, Any]):
        """Log risk management events"""
        data = {
            "event_type": event_type,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
            **details
        }
        
        if severity in ["HIGH", "CRITICAL"]:
            self.error(f"Risk event: {event_type}", **data)
        else:
            self.warning(f"Risk event: {event_type}", **data)


# Global logger instance
_logger_instance: Optional[TradingLogger] = None


def get_logger(log_dir: str = "logs", level: str = "INFO") -> TradingLogger:
    """Get or create global logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = TradingLogger(log_dir, level)
    return _logger_instance


if __name__ == "__main__":
    # Test the logger
    logger = get_logger()
    
    logger.info("Trading bot started")
    logger.log_trade({
        "symbol": "BTCUSDT",
        "side": "BUY",
        "quantity": 0.1,
        "price": 45000,
        "pnl": 150.50
    })
    logger.log_risk_event(
        "circuit_breaker_triggered",
        "CRITICAL",
        {"daily_loss": -5.2, "threshold": -5.0}
    )
    print("✅ Logger test complete. Check logs/ directory")
