"""
Enhanced Structured Logging System
Replaces basic logging.basicConfig with professional multi-file logging.
"""
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Optional

class EnhancedLogger:
    """
    Professional multi-file structured logging.
    
    Logs to:
    - trading_bot.log (all logs)
    - trades.log (trade-specific logs)
    - system.log (system/performance logs)
    - errors.log (errors and warnings only)
    """
    
    def __init__(self, name: str = "CryptoBoss", log_dir: Path = Path("data")):
        self.name = name
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        self.detailed_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        self.simple_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        self._setup_handlers()
    
   def _setup_handlers(self):
        """Setup all log handlers."""
        
        # 1. Main log file (all logs, rotating)
        main_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "trading_bot.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        main_handler.setLevel(logging.DEBUG)
        main_handler.setFormatter(self.detailed_formatter)
        self.logger.addHandler(main_handler)
        
        # 2. Trade log (trade-specific, rotating)
        trade_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "trades.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=10,
            encoding='utf-8'
        )
        trade_handler.setLevel(logging.INFO)
        trade_handler.setFormatter(self.detailed_formatter)
        trade_handler.addFilter(lambda record: 'TRADE' in record.getMessage().upper() or 'POSITION' in record.getMessage().upper())
        self.logger.addHandler(trade_handler)
        
        # 3. System log (performance/metrics)
        system_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "system.log",
            maxBytes=5*1024*1024,
            backupCount=3,
            encoding='utf-8'
        )
        system_handler.setLevel(logging.INFO)
        system_handler.setFormatter(self.detailed_formatter)
        system_handler.addFilter(lambda record: any(kw in record.getMessage().upper() for kw in ['CPU', 'MEMORY', 'LATENCY', 'METRICS']))
        self.logger.addHandler(system_handler)
        
        # 4. Error log (warnings and errors only)
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "errors.log",
            maxBytes=5*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(self.detailed_formatter)
        self.logger.addHandler(error_handler)
        
        # 5. Console handler (colored output if available)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self.simple_formatter)
        self.logger.addHandler(console_handler)
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger."""
        return self.logger
    
    def log_trade(self, trade_info: dict):
        """Convenience method for trade logging."""
        self.logger.info(f"TRADE: {trade_info}")
    
    def log_performance(self, metrics: dict):
        """Convenience method for performance metrics."""
        self.logger.info(f"METRICS: {metrics}")


# Global logger instance
_enhanced_logger: Optional[EnhancedLogger] = None


def get_enhanced_logger(name: str = "CryptoBoss", log_dir: Path = Path("data")) -> logging.Logger:
    """Get or create enhanced logger instance."""
    global _enhanced_logger
    if _enhanced_logger is None:
        _enhanced_logger = EnhancedLogger(name, log_dir)
    return _enhanced_logger.get_logger()
