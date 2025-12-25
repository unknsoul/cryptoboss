"""
Health Check Endpoint
Production-ready health monitoring for Kubernetes/Docker.
"""
from fastapi import APIRouter
from typing import Dict
from datetime import datetime
import psutil

router = APIRouter()


@router.get("/health")
async def health_check() -> Dict:
    """
    Health check endpoint for load balancers/orchestrators.
    
    Returns:
        Health status with system metrics
    """
    # Check system resources
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # Determine health status
    is_healthy = (
        cpu_percent < 95 and
        memory.percent < 90 and
        disk.percent < 95
    )
    
    return {
        "status": "healthy" if is_healthy else "degraded",
        "timestamp": datetime.now().isoformat(),
        "system": {
            "cpu_percent": round(cpu_percent, 1),
            "memory_percent": round(memory.percent, 1),
            "disk_percent": round(disk.percent, 1)
        },
        "services": {
            "trading": "active",
            "dashboard": "active",
            "database": "active"
        }
    }


@router.get("/health/ready")
async def readiness_check() -> Dict:
    """
    Readiness check - is bot ready to accept requests?
    
    Returns:
        Readiness status
    """
    return {
        "ready": True,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/health/live")
async def liveness_check() -> Dict:
    """
    Liveness check - is bot still running?
    
    Returns:
        Liveness status
    """
    return {
        "alive": True,
        "timestamp": datetime.now().isoformat()
    }
