"""
Logger Module
Provides structured logging for the trading system.
"""

import logging
import sys
from typing import Optional

# Configure default logging format
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_LEVEL = logging.INFO

# Cache for loggers
_loggers = {}


class TradingLogger:
    """
    Enhanced logger with structured output for trading operations.
    """
    
    def __init__(self, name: str = "cryptoboss", level: int = _DEFAULT_LEVEL):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Avoid duplicate handlers
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
            self.logger.addHandler(handler)
    
    def info(self, message: str, **kwargs):
        """Log info message with optional context."""
        if kwargs:
            message = f"{message} | {kwargs}"
        self.logger.info(message)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional context."""
        if kwargs:
            message = f"{message} | {kwargs}"
        self.logger.warning(message)
    
    def error(self, message: str, **kwargs):
        """Log error message with optional context."""
        if kwargs:
            message = f"{message} | {kwargs}"
        self.logger.error(message)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with optional context."""
        if kwargs:
            message = f"{message} | {kwargs}"
        self.logger.debug(message)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with optional context."""
        if kwargs:
            message = f"{message} | {kwargs}"
        self.logger.critical(message)


# Singleton instance
_default_logger: Optional[TradingLogger] = None


def get_logger(name: str = "cryptoboss") -> TradingLogger:
    """
    Get or create a logger instance.
    
    Args:
        name: Logger name (default: cryptoboss)
    
    Returns:
        TradingLogger instance
    """
    global _loggers
    
    if name not in _loggers:
        _loggers[name] = TradingLogger(name)
    
    return _loggers[name]


__all__ = ['get_logger', 'TradingLogger']
