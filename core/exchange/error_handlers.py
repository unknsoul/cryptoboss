"""
Advanced Error Handling and Recovery
Comprehensive exception handling with retry logic
"""

import time
import functools
from typing import Callable, Any, Optional, Type, Tuple
from core.monitoring.logger import get_logger
from core.monitoring.alerting import get_alerts


logger = get_logger()
alerts = get_alerts()


class TradingBotException(Exception):
    """Base exception for trading bot"""
    pass


class ExchangeException(TradingBotException):
    """Exchange-related errors"""
    pass


class OrderException(TradingBotException):
    """Order execution errors"""
    pass


class PositionException(TradingBotException):
    """Position management errors"""
    pass


class DataException(TradingBotException):
    """Data feed errors"""
    pass


class RiskException(TradingBotException):
    """Risk management violations"""
    pass


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable] = None
):
    """
    Decorator for retry logic with exponential backoff
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for each retry
        exceptions: Tuple of exceptions to catch
        on_retry: Callback function called on each retry
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}",
                            error=str(e),
                            retry_delay=delay
                        )
                        
                        if on_retry:
                            on_retry(attempt, e)
                        
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(
                            f"All {max_retries} retries exhausted for {func.__name__}",
                            error=str(e)
                        )
            
            # All retries failed
            raise last_exception
        
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    *args,
    default_return: Any = None,
    log_errors: bool = True,
    alert_on_error: bool = False,
    **kwargs
) -> Any:
    """
    Safely execute a function with error handling
    
    Args:
        func: Function to execute
        *args: Positional arguments
        default_return: Value to return on error
        log_errors: Whether to log errors
        alert_on_error: Whether to send alert on error
        **kwargs: Keyword arguments
    
    Returns:
        Function result or default_return on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.error(
                f"Error executing {func.__name__}",
                error=str(e),
                error_type=type(e).__name__
            )
        
        if alert_on_error:
            alerts.send_alert(
                "function_error",
                f"Error in {func.__name__}: {str(e)}",
                {"function": func.__name__, "error": str(e)}
            )
        
        return default_return


class ErrorHandler:
    """
    Centralized error handling for trading operations
    """
    
    def __init__(self):
        self.logger = get_logger()
        self.alerts = get_alerts()
        self.error_counts = {}
        self.consecutive_errors = {}
        
    def handle_exchange_error(self, error: Exception, operation: str) -> bool:
        """
        Handle exchange-related errors
        
        Returns:
            True if error is recoverable, False otherwise
        """
        error_msg = str(error).lower()
        
        # Rate limit errors - wait and retry
        if 'rate limit' in error_msg or '429' in error_msg:
            self.logger.warning(
                f"Rate limit hit during {operation}",
                operation=operation
            )
            time.sleep(60)  # Wait 1 minute
            return True
        
        # Network errors - retry
        elif 'network' in error_msg or 'timeout' in error_msg or 'connection' in error_msg:
            self.logger.warning(
                f"Network error during {operation}",
                operation=operation,
                error=str(error)
            )
            return True
        
        # Authentication errors - not recoverable without fixing credentials
        elif 'auth' in error_msg or 'api key' in error_msg:
            self.logger.critical(
                f"Authentication error during {operation}",
                operation=operation,
                error=str(error)
            )
            self.alerts.send_alert(
                "exchange_auth_error",
                f"Exchange authentication failed during {operation}",
                {"error": str(error)}
            )
            return False
        
        # Insufficient balance - alert but continue
        elif 'insufficient' in error_msg or 'balance' in error_msg:
            self.logger.error(
                f"Insufficient balance during {operation}",
                operation=operation
            )
            self.alerts.send_alert(
                "insufficient_balance",
                f"Insufficient balance for {operation}",
                {"error": str(error)}
            )
            return False
        
        # Unknown error
        else:
            self.logger.error(
                f"Unknown exchange error during {operation}",
                operation=operation,
                error=str(error),
                error_type=type(error).__name__
            )
            return True  # Assume retryable
    
    def handle_order_error(self, error: Exception, order_details: dict) -> bool:
        """
        Handle order execution errors
        
        Returns:
            True if should retry, False otherwise
        """
        error_msg = str(error).lower()
        
        # Invalid order parameters
        if 'invalid' in error_msg or 'parameter' in error_msg:
            self.logger.error(
                "Invalid order parameters",
                error=str(error),
                order=order_details
            )
            return False  # Don't retry invalid orders
        
        # Order would trigger immediately
        elif 'immediately' in error_msg or 'post-only' in error_msg:
            self.logger.warning(
                "Order would trigger immediately",
                order=order_details
            )
            return False
        
        # Price filter violation
        elif 'price' in error_msg and 'filter' in error_msg:
            self.logger.warning(
                "Price filter violation",
                error=str(error),
                order=order_details
            )
            return False
        
        # Size filter violation
        elif 'size' in error_msg or 'quantity' in error_msg or 'notional' in error_msg:
            self.logger.warning(
                "Size/quantity filter violation",
                error=str(error),
                order=order_details
            )
            return False
        
        # Other errors - assume retryable
        else:
            self.logger.error(
                "Order execution error",
                error=str(error),
                order=order_details
            )
            return True
    
    def track_consecutive_errors(self, operation: str, max_consecutive: int = 5) -> bool:
        """
        Track consecutive errors and trigger circuit breaker if needed
        
        Returns:
            True if within limits, False if circuit breaker should activate
        """
        if operation not in self.consecutive_errors:
            self.consecutive_errors[operation] = 0
        
        self.consecutive_errors[operation] += 1
        
        if self.consecutive_errors[operation] >= max_consecutive:
            self.logger.critical(
                f"Too many consecutive errors for {operation}",
                count=self.consecutive_errors[operation],
                limit=max_consecutive
            )
            self.alerts.send_alert(
                "consecutive_errors",
                f"Circuit breaker triggered: {max_consecutive} consecutive errors in {operation}",
                {"operation": operation, "count": self.consecutive_errors[operation]}
            )
            return False
        
        return True
    
    def reset_error_count(self, operation: str):
        """Reset error counter after successful operation"""
        if operation in self.consecutive_errors:
            self.consecutive_errors[operation] = 0


# Global error handler
_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get or create global error handler"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


# Example usage
if __name__ == "__main__":
    # Test retry decorator
    @retry_with_backoff(max_retries=3, initial_delay=0.5)
    def flaky_function(should_fail: bool = False):
        if should_fail:
            raise ExchangeException("Simulated API error")
        return "Success"
    
    # Test error handler
    handler = get_error_handler()
    
    try:
        result = flaky_function(should_fail=False)
        print(f"✅ Function succeeded: {result}")
    except Exception as e:
        print(f"❌ Function failed: {e}")
    
    # Test safe execute
    result = safe_execute(
        lambda: 1 / 0,
        default_return="Error occurred",
        log_errors=True
    )
    print(f"Safe execute result: {result}")
