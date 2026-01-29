"""
Exchange Recovery Handler - Robust Error Recovery for Exchange Operations

Handles exchange errors with:
- Retry with exponential backoff
- Error classification (transient vs permanent)
- Automatic failover to paper mode
- Order state reconciliation

v11.0 - Production-Grade Platform Upgrade
"""

import logging
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable, Any
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ErrorCategory(str, Enum):
    """Categories of exchange errors."""
    TRANSIENT = "transient"          # Temporary, retry likely to succeed
    RATE_LIMIT = "rate_limit"        # Rate limited, wait and retry
    AUTHENTICATION = "authentication"  # Auth error, needs investigation
    INSUFFICIENT_FUNDS = "insufficient_funds"  # Not enough balance
    INVALID_ORDER = "invalid_order"   # Order parameters invalid
    NETWORK = "network"               # Network connectivity issue
    EXCHANGE_ERROR = "exchange_error" # Exchange-side error
    TIMEOUT = "timeout"               # Request timed out
    UNKNOWN = "unknown"               # Unclassified error


class RecoveryAction(str, Enum):
    """Actions taken during recovery."""
    RETRY = "retry"
    WAIT_RETRY = "wait_retry"
    CANCEL_ORDER = "cancel_order"
    RECONCILE = "reconcile"
    FAILOVER_PAPER = "failover_paper"
    ABORT = "abort"
    ALERT = "alert"


@dataclass
class ErrorRecord:
    """Record of an exchange error."""
    timestamp: datetime
    operation: str
    error_type: str
    error_message: str
    category: ErrorCategory
    order_id: Optional[str] = None
    symbol: Optional[str] = None
    retry_count: int = 0
    recovered: bool = False
    recovery_action: Optional[RecoveryAction] = None
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'operation': self.operation,
            'error_type': self.error_type,
            'error_message': self.error_message,
            'category': self.category.value,
            'order_id': self.order_id,
            'symbol': self.symbol,
            'retry_count': self.retry_count,
            'recovered': self.recovered,
            'recovery_action': self.recovery_action.value if self.recovery_action else None,
        }


@dataclass 
class RecoveryResult:
    """Result of a recovery attempt."""
    success: bool
    action_taken: RecoveryAction
    attempts: int
    total_wait_ms: float
    error_category: ErrorCategory
    final_error: Optional[str] = None
    result_data: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'action_taken': self.action_taken.value,
            'attempts': self.attempts,
            'total_wait_ms': self.total_wait_ms,
            'error_category': self.error_category.value,
            'final_error': self.final_error,
            'result_data': self.result_data,
        }


class ExchangeRecoveryHandler:
    """
    Exchange Recovery Handler - Robust error handling for exchange operations.
    
    Features:
    - Automatic retry with exponential backoff
    - Error classification and appropriate handling
    - Automatic failover to paper mode on persistent issues
    - Order state reconciliation
    - Error tracking and alerting
    
    Usage:
        handler = ExchangeRecoveryHandler()
        
        # Execute with retry
        result = await handler.execute_with_retry(
            operation=exchange.create_order,
            args=("BTC/USDT", "limit", "buy", 0.1, 45000),
            operation_name="create_order"
        )
        
        # Check if should failover
        if handler.should_failover_to_paper():
            # Switch to paper mode
            pass
    """
    
    # Error patterns for classification
    ERROR_PATTERNS = {
        ErrorCategory.RATE_LIMIT: [
            'rate limit', 'too many requests', '429', 'throttle',
            'request limit', 'rate exceeded'
        ],
        ErrorCategory.AUTHENTICATION: [
            'auth', 'signature', 'api key', 'permission', 'forbidden',
            '401', '403', 'invalid key'
        ],
        ErrorCategory.INSUFFICIENT_FUNDS: [
            'insufficient', 'balance', 'not enough', 'margin',
            'available balance'
        ],
        ErrorCategory.INVALID_ORDER: [
            'invalid order', 'min notional', 'lot size', 'price filter',
            'invalid quantity', 'invalid price', 'invalid symbol'
        ],
        ErrorCategory.NETWORK: [
            'connection', 'network', 'dns', 'socket', 'refused',
            'unreachable', 'ssl', 'certificate'
        ],
        ErrorCategory.TIMEOUT: [
            'timeout', 'timed out', 'deadline', 'took too long'
        ],
        ErrorCategory.EXCHANGE_ERROR: [
            'exchange error', 'server error', '500', '502', '503', '504',
            'internal error', 'service unavailable', 'maintenance'
        ],
    }
    
    # Retry configuration
    DEFAULT_MAX_RETRIES = 5
    DEFAULT_BASE_DELAY_MS = 1000
    DEFAULT_MAX_DELAY_MS = 30000
    DEFAULT_BACKOFF_MULTIPLIER = 2.0
    
    # Failover thresholds
    FAILOVER_ERROR_THRESHOLD = 10  # Errors before considering failover
    FAILOVER_WINDOW_MINUTES = 5     # Time window for error counting
    
    def __init__(
        self,
        max_retries: int = 5,
        base_delay_ms: int = 1000,
        max_delay_ms: int = 30000,
        backoff_multiplier: float = 2.0,
        failover_threshold: int = 10,
        failover_window_minutes: int = 5,
        log_dir: str = "logs/recovery",
        on_error: Optional[Callable[[ErrorRecord], None]] = None,
        on_failover: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize ExchangeRecoveryHandler.
        
        Args:
            max_retries: Maximum retry attempts
            base_delay_ms: Initial delay before retry
            max_delay_ms: Maximum delay between retries
            backoff_multiplier: Multiplier for exponential backoff
            failover_threshold: Error count to trigger failover consideration
            failover_window_minutes: Window for error counting
            log_dir: Directory for recovery logs
            on_error: Callback for errors
            on_failover: Callback for failover events
        """
        self.max_retries = max_retries
        self.base_delay_ms = base_delay_ms
        self.max_delay_ms = max_delay_ms
        self.backoff_multiplier = backoff_multiplier
        self.failover_threshold = failover_threshold
        self.failover_window_minutes = failover_window_minutes
        self.log_dir = log_dir
        self.on_error = on_error
        self.on_failover = on_failover
        
        self._errors: List[ErrorRecord] = []
        self._max_errors = 10000
        self._in_paper_mode = False
        self._failover_reason: Optional[str] = None
        
        # Statistics
        self._stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_retries': 0,
            'recovered_after_retry': 0,
            'by_category': {cat.value: 0 for cat in ErrorCategory},
        }
        
        # Ensure log directory exists
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"ExchangeRecoveryHandler initialized: max_retries={max_retries}, "
            f"base_delay={base_delay_ms}ms, failover_threshold={failover_threshold}"
        )
    
    async def execute_with_retry(
        self,
        operation: Callable,
        args: tuple = (),
        kwargs: dict = None,
        operation_name: str = "operation",
        order_id: str = None,
        symbol: str = None,
        retry_on: List[ErrorCategory] = None
    ) -> RecoveryResult:
        """
        Execute an operation with automatic retry on failure.
        
        Args:
            operation: Async callable to execute
            args: Positional arguments
            kwargs: Keyword arguments
            operation_name: Name for logging
            order_id: Associated order ID
            symbol: Associated symbol
            retry_on: Error categories to retry on (default: transient errors)
            
        Returns:
            RecoveryResult with execution outcome
        """
        kwargs = kwargs or {}
        retry_on = retry_on or [
            ErrorCategory.TRANSIENT,
            ErrorCategory.RATE_LIMIT,
            ErrorCategory.NETWORK,
            ErrorCategory.TIMEOUT,
            ErrorCategory.EXCHANGE_ERROR,
        ]
        
        self._stats['total_operations'] += 1
        
        attempts = 0
        total_wait_ms = 0.0
        last_error = None
        last_category = ErrorCategory.UNKNOWN
        
        while attempts < self.max_retries:
            attempts += 1
            
            try:
                # Execute operation
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)
                
                # Success
                self._stats['successful_operations'] += 1
                if attempts > 1:
                    self._stats['recovered_after_retry'] += 1
                    logger.info(f"Operation {operation_name} succeeded after {attempts} attempts")
                
                return RecoveryResult(
                    success=True,
                    action_taken=RecoveryAction.RETRY if attempts > 1 else RecoveryAction.RETRY,
                    attempts=attempts,
                    total_wait_ms=total_wait_ms,
                    error_category=last_category,
                    result_data={'result': result} if not isinstance(result, dict) else result
                )
                
            except Exception as e:
                last_error = str(e)
                last_category = self.classify_error(e)
                
                # Record error
                error_record = ErrorRecord(
                    timestamp=datetime.utcnow(),
                    operation=operation_name,
                    error_type=type(e).__name__,
                    error_message=last_error,
                    category=last_category,
                    order_id=order_id,
                    symbol=symbol,
                    retry_count=attempts,
                )
                self._record_error(error_record)
                
                # Check if we should retry
                if last_category not in retry_on:
                    logger.warning(
                        f"Operation {operation_name} failed with non-retryable error: "
                        f"{last_category.value} - {last_error}"
                    )
                    break
                
                # Check if more retries allowed
                if attempts >= self.max_retries:
                    logger.error(
                        f"Operation {operation_name} failed after {attempts} attempts: {last_error}"
                    )
                    break
                
                # Calculate delay
                delay_ms = self._calculate_delay(attempts, last_category)
                total_wait_ms += delay_ms
                
                logger.warning(
                    f"Operation {operation_name} attempt {attempts} failed: {last_category.value}. "
                    f"Retrying in {delay_ms}ms..."
                )
                
                # Wait before retry
                await asyncio.sleep(delay_ms / 1000.0)
                self._stats['total_retries'] += 1
        
        # Failed after all retries
        self._stats['failed_operations'] += 1
        
        return RecoveryResult(
            success=False,
            action_taken=RecoveryAction.ABORT,
            attempts=attempts,
            total_wait_ms=total_wait_ms,
            error_category=last_category,
            final_error=last_error
        )
    
    def classify_error(self, error: Exception) -> ErrorCategory:
        """
        Classify an error into a category.
        
        Args:
            error: Exception to classify
            
        Returns:
            ErrorCategory
        """
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        combined = f"{error_type} {error_str}"
        
        for category, patterns in self.ERROR_PATTERNS.items():
            for pattern in patterns:
                if pattern in combined:
                    return category
        
        # Check for common exception types
        if 'timeout' in error_type:
            return ErrorCategory.TIMEOUT
        if 'connection' in error_type:
            return ErrorCategory.NETWORK
        
        # Default to transient for generic exceptions (safer for retry)
        if any(x in error_type for x in ['exception', 'error']):
            return ErrorCategory.TRANSIENT
        
        return ErrorCategory.UNKNOWN
    
    def should_failover_to_paper(self) -> bool:
        """
        Determine if exchange issues warrant failover to paper mode.
        
        Returns:
            True if failover is recommended
        """
        if self._in_paper_mode:
            return True
        
        # Count recent errors
        cutoff = datetime.utcnow() - timedelta(minutes=self.failover_window_minutes)
        recent_errors = [e for e in self._errors if e.timestamp >= cutoff]
        
        # Count by serious categories
        serious_categories = [
            ErrorCategory.EXCHANGE_ERROR,
            ErrorCategory.NETWORK,
            ErrorCategory.AUTHENTICATION,
        ]
        serious_count = sum(
            1 for e in recent_errors 
            if e.category in serious_categories
        )
        
        if serious_count >= self.failover_threshold:
            self._failover_reason = (
                f"{serious_count} serious errors in last {self.failover_window_minutes} minutes"
            )
            return True
        
        return False
    
    def activate_paper_mode(self, reason: str = "Manual activation") -> None:
        """Activate paper mode failover."""
        if not self._in_paper_mode:
            self._in_paper_mode = True
            self._failover_reason = reason
            
            logger.warning(f"PAPER MODE ACTIVATED: {reason}")
            
            if self.on_failover:
                try:
                    self.on_failover(reason)
                except Exception as e:
                    logger.error(f"Failover callback failed: {e}")
    
    def deactivate_paper_mode(self) -> None:
        """Deactivate paper mode failover."""
        if self._in_paper_mode:
            self._in_paper_mode = False
            self._failover_reason = None
            logger.info("Paper mode deactivated, returning to live trading")
    
    async def reconcile_order_state(
        self,
        order_id: str,
        symbol: str,
        get_order_func: Callable,
        cancel_order_func: Callable = None
    ) -> Dict:
        """
        Reconcile order state with exchange.
        
        Args:
            order_id: Order ID to check
            symbol: Trading symbol
            get_order_func: Function to get order status
            cancel_order_func: Optional function to cancel order
            
        Returns:
            Reconciled order state
        """
        try:
            # Get order from exchange
            result = await self.execute_with_retry(
                operation=get_order_func,
                args=(order_id, symbol),
                operation_name="get_order",
                order_id=order_id,
                symbol=symbol
            )
            
            if result.success:
                order_data = result.result_data.get('result', {})
                return {
                    'reconciled': True,
                    'order_id': order_id,
                    'status': order_data.get('status', 'unknown'),
                    'filled': order_data.get('filled', 0),
                    'remaining': order_data.get('remaining', 0),
                    'data': order_data
                }
            else:
                return {
                    'reconciled': False,
                    'order_id': order_id,
                    'error': result.final_error
                }
                
        except Exception as e:
            logger.error(f"Order reconciliation failed for {order_id}: {e}")
            return {
                'reconciled': False,
                'order_id': order_id,
                'error': str(e)
            }
    
    def get_recent_errors(self, limit: int = 100, category: ErrorCategory = None) -> List[Dict]:
        """Get recent errors."""
        errors = self._errors
        if category:
            errors = [e for e in errors if e.category == category]
        return [e.to_dict() for e in errors[-limit:]]
    
    def get_status(self) -> Dict:
        """Get handler status."""
        cutoff = datetime.utcnow() - timedelta(minutes=self.failover_window_minutes)
        recent_errors = [e for e in self._errors if e.timestamp >= cutoff]
        
        return {
            'in_paper_mode': self._in_paper_mode,
            'failover_reason': self._failover_reason,
            'should_failover': self.should_failover_to_paper(),
            'recent_error_count': len(recent_errors),
            'failover_threshold': self.failover_threshold,
            'stats': self._stats,
            'error_summary': {
                cat.value: sum(1 for e in recent_errors if e.category == cat)
                for cat in ErrorCategory
            }
        }
    
    def get_stats(self) -> Dict:
        """Get detailed statistics."""
        success_rate = (
            self._stats['successful_operations'] / self._stats['total_operations']
            if self._stats['total_operations'] > 0 else 0
        )
        recovery_rate = (
            self._stats['recovered_after_retry'] / self._stats['total_retries']
            if self._stats['total_retries'] > 0 else 0
        )
        
        return {
            **self._stats,
            'success_rate': success_rate,
            'recovery_rate': recovery_rate,
            'total_errors': len(self._errors),
        }
    
    def _calculate_delay(self, attempt: int, category: ErrorCategory) -> float:
        """Calculate delay before next retry."""
        # Base exponential backoff
        delay = self.base_delay_ms * (self.backoff_multiplier ** (attempt - 1))
        
        # Special handling for rate limits - use longer delays
        if category == ErrorCategory.RATE_LIMIT:
            delay = max(delay, 5000)  # At least 5 seconds for rate limits
        
        # Cap at maximum
        delay = min(delay, self.max_delay_ms)
        
        # Add small jitter (10%)
        import random
        jitter = delay * 0.1 * random.random()
        
        return delay + jitter
    
    def _record_error(self, error: ErrorRecord) -> None:
        """Record an error."""
        self._errors.append(error)
        self._stats['by_category'][error.category.value] += 1
        
        # Trim if needed
        if len(self._errors) > self._max_errors:
            self._errors = self._errors[-self._max_errors:]
        
        # Persist
        try:
            log_file = Path(self.log_dir) / f"errors_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
            with open(log_file, 'a') as f:
                f.write(json.dumps(error.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to persist error record: {e}")
        
        # Callback
        if self.on_error:
            try:
                self.on_error(error)
            except Exception as e:
                logger.error(f"Error callback failed: {e}")


# Singleton instance
_recovery_handler: Optional[ExchangeRecoveryHandler] = None


def get_recovery_handler() -> ExchangeRecoveryHandler:
    """Get the global ExchangeRecoveryHandler instance."""
    global _recovery_handler
    if _recovery_handler is None:
        _recovery_handler = ExchangeRecoveryHandler()
    return _recovery_handler
