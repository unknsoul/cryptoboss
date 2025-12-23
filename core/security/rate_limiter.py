"""
Rate Limiter for API Calls and Order Execution
Prevents hitting exchange rate limits
"""

from datetime import datetime, timedelta
from typing import Optional
import asyncio


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded"""
    pass


class RateLimiter:
    """
    Token bucket rate limiter for API calls
    
    Example:
        limiter = RateLimiter(max_requests=100, time_window=timedelta(minutes=1))
        await limiter.acquire()  # Blocks if rate limit reached
    """
    
    def __init__(self, max_requests: int, time_window: timedelta):
        """
        Initialize rate limiter
        
        Args:
            max_requests: Maximum number of requests allowed
            time_window: Time window for rate limiting
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: list[datetime] = []
        self._lock = asyncio.Lock()
    
    async def acquire(self, wait: bool = True) -> bool:
        """
        Acquire permission to make a request
        
        Args:
            wait: If True, wait until rate limit allows. If False, raise exception.
            
        Returns:
            True if acquired successfully
            
        Raises:
            RateLimitExceeded: If wait=False and rate limit exceeded
        """
        async with self._lock:
            now = datetime.now()
            
            # Remove expired requests
            self.requests = [
                req_time for req_time in self.requests
                if now - req_time < self.time_window
            ]
            
            # Check if we can proceed
            if len(self.requests) >= self.max_requests:
                if not wait:
                    raise RateLimitExceeded(
                        f"Rate limit exceeded: {self.max_requests} requests per {self.time_window}"
                    )
                
                # Calculate wait time
                oldest_request = self.requests[0]
                wait_until = oldest_request + self.time_window
                wait_seconds = (wait_until - now).total_seconds()
                
                if wait_seconds > 0:
                    await asyncio.sleep(wait_seconds + 0.1)  # Small buffer
                    return await self.acquire(wait=wait)
            
            # Add current request
            self.requests.append(now)
            return True
    
    def get_remaining_quota(self) -> int:
        """Get number of requests remaining in current window"""
        now = datetime.now()
        self.requests = [
            req_time for req_time in self.requests
            if now - req_time < self.time_window
        ]
        return max(0, self.max_requests - len(self.requests))
    
    def reset(self):
        """Reset rate limiter"""
        self.requests.clear()


class MultiTierRateLimiter:
    """
    Multi-tier rate limiter for different API endpoints
    
    Example:
        limiter = MultiTierRateLimiter()
        await limiter.acquire('order')  # Strict limit for orders
        await limiter.acquire('market_data')  # Looser limit for data
    """
    
    def __init__(self):
        self.limiters = {
            # Binance limits (conservative)
            'order': RateLimiter(10, timedelta(seconds=1)),  # 10 orders per second
            'market_data': RateLimiter(100, timedelta(minutes=1)),  # 100 per minute
            'account': RateLimiter(20, timedelta(seconds=1)),  # 20 per second
        }
    
    async def acquire(self, category: str = 'default', wait: bool = True) -> bool:
        """
        Acquire rate limit token for specific category
        
        Args:
            category: Rate limit category ('order', 'market_data', 'account')
            wait: Whether to wait if limit exceeded
            
        Returns:
            True if acquired
        """
        if category not in self.limiters:
            # Default limiter
            if 'default' not in self.limiters:
                self.limiters['default'] = RateLimiter(50, timedelta(seconds=1))
            category = 'default'
        
        return await self.limiters[category].acquire(wait=wait)
    
    def get_status(self) -> dict[str, int]:
        """Get remaining quota for all categories"""
        return {
            category: limiter.get_remaining_quota()
            for category, limiter in self.limiters.items()
        }


# Global rate limiter instance
_global_limiter: Optional[MultiTierRateLimiter] = None


def get_rate_limiter() -> MultiTierRateLimiter:
    """Get global rate limiter instance"""
    global _global_limiter
    if _global_limiter is None:
        _global_limiter = MultiTierRateLimiter()
    return _global_limiter


if __name__ == '__main__':
    # Test rate limiter
    async def test():
        limiter = RateLimiter(max_requests=5, time_window=timedelta(seconds=1))
        
        print("Making 5 requests rapidly...")
        for i in range(5):
            await limiter.acquire()
            print(f"Request {i+1} - Remaining: {limiter.get_remaining_quota()}")
        
        print("\nTrying 6th request (should wait)...")
        start = datetime.now()
        await limiter.acquire()
        elapsed = (datetime.now() - start).total_seconds()
        print(f"6th request completed after {elapsed:.2f}s wait")
    
    asyncio.run(test())
