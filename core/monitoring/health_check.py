"""
Health Check System
Monitor system status and component health
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, Callable
from datetime import datetime
import asyncio


class HealthStatus(Enum):
    """Component health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a single component"""
    name: str
    status: HealthStatus
    message: str
    last_check: datetime
    response_time_ms: Optional[float] = None
    details: Optional[Dict] = None


class HealthChecker:
    """
    Comprehensive health check system
    
    Monitors:
    - Exchange connectivity
    - Database connections
    - WebSocket status
    - API rate limits
    - System resources
    """
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.last_results: Dict[str, ComponentHealth] = {}
    
    def register_check(self, name: str, check_func: Callable):
        """
        Register a health check function
        
        Args:
            name: Check name
            check_func: Async function that returns (status, message, details)
        """
        self.checks[name] = check_func
    
    async def check_component(self, name: str) -> ComponentHealth:
        """
        Run health check for specific component
        
        Args:
            name: Component name
            
        Returns:
            ComponentHealth result
        """
        if name not in self.checks:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                message="No health check registered",
                last_check=datetime.now()
            )
        
        start_time = datetime.now()
        
        try:
            # Run check function (should return status, message, details)
            result = await self.checks[name]()
            
            if isinstance(result, tuple):
                if len(result) == 2:
                    status, message = result
                    details = None
                elif len(result) == 3:
                    status, message, details = result
                else:
                    raise ValueError(f"Invalid check result format from {name}")
            else:
                status = result
                message = "OK"
                details = None
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            health = ComponentHealth(
                name=name,
                status=status,
                message=message,
                last_check=datetime.now(),
                response_time_ms=response_time,
                details=details
            )
            
            self.last_results[name] = health
            return health
            
        except Exception as e:
            health = ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                last_check=datetime.now()
            )
            self.last_results[name] = health
            return health
    
    async def check_all(self) -> Dict[str, ComponentHealth]:
        """
        Run all registered health checks
        
        Returns:
            Dictionary of component health results
        """
        results = {}
        
        # Run all checks concurrently
        tasks = [self.check_component(name) for name in self.checks.keys()]
        health_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for name, result in zip(self.checks.keys(), health_results):
            if isinstance(result, Exception):
                results[name] = ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Exception: {str(result)}",
                    last_check=datetime.now()
                )
            else:
                results[name] = result
        
        return results
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status"""
        if not self.last_results:
            return HealthStatus.UNKNOWN
        
        statuses = [health.status for health in self.last_results.values()]
        
        # If any unhealthy, overall is unhealthy
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        
        # If any degraded, overall is degraded
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        
        # All healthy
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        
        return HealthStatus.UNKNOWN
    
    def print_health_report(self):
        """Print formatted health report"""
        overall = self.get_overall_status()
        
        print("\n" + "=" * 70)
        print(f"SYSTEM HEALTH CHECK - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        print(f"Overall Status: {overall.value.upper()}")
        print()
        
        for name, health in self.last_results.items():
            status_icon = {
                HealthStatus.HEALTHY: "✓",
                HealthStatus.DEGRADED: "⚠",
                HealthStatus.UNHEALTHY: "✗",
                HealthStatus.UNKNOWN: "?"
            }[health.status]
            
            print(f"{status_icon} {name:<25} {health.status.value:<12} {health.message}")
            
            if health.response_time_ms:
                print(f"  Response time: {health.response_time_ms:.1f}ms")
            
            if health.details:
                for key, value in health.details.items():
                    print(f"  {key}: {value}")
        
        print("=" * 70 + "\n")


# Example health check functions

async def check_exchange_connection(exchange_client=None) -> tuple:
    """Check if exchange connection is healthy"""
    try:
        if exchange_client:
            # Would actually ping exchange
            # await exchange_client.ping()
            pass
        
        # Simulated check
        await asyncio.sleep(0.05)  # Simulate network delay
        return (HealthStatus.HEALTHY, "Connected to Binance")
    
    except Exception as e:
        return (HealthStatus.UNHEALTHY, f"Connection failed: {e}")


async def check_websocket() -> tuple:
    """Check WebSocket connection status"""
    try:
        # Would check actual WebSocket
        await asyncio.sleep(0.01)
        return (HealthStatus.HEALTHY, "WebSocket active", {"messages_received": 1234})
    
    except Exception as e:
        return (HealthStatus.UNHEALTHY, f"WebSocket error: {e}")


async def check_database() -> tuple:
    """Check database connection"""
    try:
        # Would actually query database
        await asyncio.sleep(0.02)
        return (HealthStatus.HEALTHY, "Database connected")
    
    except Exception as e:
        return (HealthStatus.DEGRADED, f"Database slow: {e}")


async def check_rate_limits() -> tuple:
    """Check API rate limit status"""
    # Would check actual rate limiter
    used = 45
    limit = 100
    
    if used / limit > 0.9:
        return (HealthStatus.DEGRADED, f"Rate limit high: {used}/{limit}")
    else:
        return (HealthStatus.HEALTHY, f"Rate limit OK: {used}/{limit}")


async def check_system_resources() -> tuple:
    """Check system CPU and memory"""
    try:
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        
        if cpu_percent > 90 or memory_percent > 90:
            return (
                HealthStatus.DEGRADED,
                f"High resource usage",
                {"cpu": f"{cpu_percent}%", "memory": f"{memory_percent}%"}
            )
        else:
            return (
                HealthStatus.HEALTHY,
                "Resources OK",
                {"cpu": f"{cpu_percent}%", "memory": f"{memory_percent}%"}
            )
    
    except ImportError:
        return (HealthStatus.UNKNOWN, "psutil not installed")


if __name__ == '__main__':
    # Test health check system
    
    async def test():
        checker = HealthChecker()
        
        # Register checks
        checker.register_check("exchange", check_exchange_connection)
        checker.register_check("websocket", check_websocket)
        checker.register_check("database", check_database)
        checker.register_check("rate_limits", check_rate_limits)
        checker.register_check("system_resources", check_system_resources)
        
        # Run all checks
        results = await checker.check_all()
        
        # Print report
        checker.print_health_report()
        
        # Check overall status
        overall = checker.get_overall_status()
        if overall == HealthStatus.HEALTHY:
            print("✓ All systems operational!")
        else:
            print(f"⚠ System status: {overall.value}")
    
    asyncio.run(test())
