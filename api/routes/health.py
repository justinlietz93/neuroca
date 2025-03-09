"""
Health Check API Routes for NeuroCognitive Architecture (NCA)

This module provides health check endpoints for monitoring the NCA system's operational status.
It includes comprehensive health checks for all system components, memory tiers, and resource
utilization. The health endpoints follow standard practices for health check APIs, providing
both simple availability checks and detailed diagnostic information.

Usage:
    These routes are automatically registered with the FastAPI application and can be accessed
    at the /health endpoint. They provide both overall system health status and detailed
    component-level health information.

Security:
    Health endpoints may expose sensitive system information and should be properly secured
    in production environments. Consider implementing authentication for detailed health checks.
"""

import logging
import os
import platform
import time
from datetime import datetime
from typing import Dict, List, Optional, Union

import psutil
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from neuroca.config import settings
from neuroca.core.auth import get_optional_api_key
from neuroca.db.connection import get_db_status
from neuroca.memory import memory_manager
from neuroca.monitoring.metrics import get_system_metrics

# Configure logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/health",
    tags=["health"],
    responses={
        503: {"description": "Service Unavailable"},
        500: {"description": "Internal Server Error"},
        401: {"description": "Unauthorized"},
    },
)


class ComponentHealth(BaseModel):
    """Model representing the health status of a system component."""
    
    name: str = Field(..., description="Name of the component")
    status: str = Field(..., description="Status of the component (healthy, degraded, unhealthy)")
    latency_ms: Optional[float] = Field(None, description="Response latency in milliseconds")
    message: Optional[str] = Field(None, description="Additional status information")
    last_check: datetime = Field(..., description="Timestamp of the last health check")
    details: Optional[Dict] = Field(None, description="Detailed component-specific information")


class MemoryTierHealth(BaseModel):
    """Model representing the health status of a memory tier."""
    
    tier: str = Field(..., description="Memory tier name (working, episodic, semantic)")
    status: str = Field(..., description="Status of the memory tier")
    size: int = Field(..., description="Current size in bytes")
    capacity: int = Field(..., description="Maximum capacity in bytes")
    usage_percent: float = Field(..., description="Usage percentage")
    access_latency_ms: Optional[float] = Field(None, description="Access latency in milliseconds")
    details: Optional[Dict] = Field(None, description="Additional tier-specific details")


class ResourceUtilization(BaseModel):
    """Model representing system resource utilization."""
    
    cpu_percent: float = Field(..., description="CPU utilization percentage")
    memory_percent: float = Field(..., description="Memory utilization percentage")
    disk_percent: float = Field(..., description="Disk utilization percentage")
    network_io: Dict[str, int] = Field(..., description="Network IO statistics")
    process_count: int = Field(..., description="Number of running processes")


class DetailedHealthResponse(BaseModel):
    """Model for the detailed health check response."""
    
    status: str = Field(..., description="Overall system status (healthy, degraded, unhealthy)")
    version: str = Field(..., description="System version")
    environment: str = Field(..., description="Deployment environment")
    uptime_seconds: int = Field(..., description="System uptime in seconds")
    timestamp: datetime = Field(..., description="Current timestamp")
    components: List[ComponentHealth] = Field(..., description="Health status of system components")
    memory_tiers: List[MemoryTierHealth] = Field(..., description="Health status of memory tiers")
    resources: ResourceUtilization = Field(..., description="System resource utilization")
    host_info: Dict = Field(..., description="Host system information")


@router.get(
    "",
    summary="Basic health check",
    description="Simple health check endpoint that returns 200 OK if the service is running",
    status_code=status.HTTP_200_OK,
    response_description="Service is healthy",
    responses={
        503: {"description": "Service is unhealthy or starting up"},
    },
)
async def health_check(request: Request) -> Dict[str, str]:
    """
    Basic health check endpoint that returns a simple status response.
    
    This endpoint is designed for load balancers and simple monitoring systems
    that only need to know if the service is running.
    
    Returns:
        Dict[str, str]: A simple status response with the service status
    
    Raises:
        HTTPException: 503 status code if the service is unhealthy
    """
    try:
        # Check if the application is in startup mode
        app = request.app
        if hasattr(app, "is_startup_complete") and not app.is_startup_complete:
            logger.info("Health check during startup phase")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service is starting up",
            )
        
        # Basic DB connection check
        db_status = get_db_status()
        if not db_status["connected"]:
            logger.error(f"Database connection failed: {db_status.get('error')}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database connection failed",
            )
        
        return {"status": "healthy"}
    
    except Exception as e:
        if not isinstance(e, HTTPException):
            logger.exception("Health check failed with unexpected error")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Service is unhealthy: {str(e)}",
            )
        raise


@router.get(
    "/detailed",
    summary="Detailed health check",
    description="Comprehensive health check with detailed system status information",
    response_model=DetailedHealthResponse,
    status_code=status.HTTP_200_OK,
    response_description="Detailed health status",
    responses={
        401: {"description": "Unauthorized access"},
        503: {"description": "Service is unhealthy"},
    },
)
async def detailed_health_check(
    request: Request,
    response: Response,
    api_key: Optional[str] = Depends(get_optional_api_key),
) -> DetailedHealthResponse:
    """
    Detailed health check endpoint that provides comprehensive system status information.
    
    This endpoint requires authentication in production environments and provides
    detailed information about all system components, memory tiers, and resource utilization.
    
    Args:
        request: The incoming request
        response: The outgoing response
        api_key: Optional API key for authentication
    
    Returns:
        DetailedHealthResponse: Comprehensive health status information
    
    Raises:
        HTTPException: 401 if authentication fails, 503 if service is unhealthy
    """
    # Check authentication in production
    if settings.ENVIRONMENT == "production" and not api_key:
        logger.warning("Unauthorized access attempt to detailed health check")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required for detailed health check in production",
        )
    
    start_time = time.time()
    
    try:
        # Get system metrics
        metrics = get_system_metrics()
        
        # Check component health
        components = []
        
        # Database health
        db_status = get_db_status()
        db_latency = db_status.get("latency_ms", 0)
        components.append(
            ComponentHealth(
                name="database",
                status="healthy" if db_status["connected"] else "unhealthy",
                latency_ms=db_latency,
                message=db_status.get("message", ""),
                last_check=datetime.now(),
                details=db_status.get("details", {}),
            )
        )
        
        # Memory system health
        try:
            memory_status = memory_manager.get_health_status()
            components.append(
                ComponentHealth(
                    name="memory_system",
                    status=memory_status["status"],
                    latency_ms=memory_status.get("latency_ms"),
                    message=memory_status.get("message", ""),
                    last_check=datetime.now(),
                    details=memory_status.get("details", {}),
                )
            )
        except Exception as e:
            logger.error(f"Error checking memory system health: {str(e)}")
            components.append(
                ComponentHealth(
                    name="memory_system",
                    status="unhealthy",
                    message=f"Error checking health: {str(e)}",
                    last_check=datetime.now(),
                )
            )
        
        # Memory tier health
        memory_tiers = []
        for tier_name in ["working", "episodic", "semantic"]:
            try:
                tier_status = memory_manager.get_tier_status(tier_name)
                memory_tiers.append(
                    MemoryTierHealth(
                        tier=tier_name,
                        status=tier_status["status"],
                        size=tier_status["size"],
                        capacity=tier_status["capacity"],
                        usage_percent=tier_status["usage_percent"],
                        access_latency_ms=tier_status.get("access_latency_ms"),
                        details=tier_status.get("details", {}),
                    )
                )
            except Exception as e:
                logger.error(f"Error checking {tier_name} memory tier health: {str(e)}")
                memory_tiers.append(
                    MemoryTierHealth(
                        tier=tier_name,
                        status="unknown",
                        size=0,
                        capacity=0,
                        usage_percent=0,
                        details={"error": str(e)},
                    )
                )
        
        # Resource utilization
        resources = ResourceUtilization(
            cpu_percent=psutil.cpu_percent(),
            memory_percent=psutil.virtual_memory().percent,
            disk_percent=psutil.disk_usage('/').percent,
            network_io={
                "bytes_sent": psutil.net_io_counters().bytes_sent,
                "bytes_recv": psutil.net_io_counters().bytes_recv,
            },
            process_count=len(psutil.pids()),
        )
        
        # Host information
        host_info = {
            "hostname": platform.node(),
            "os": platform.system(),
            "os_version": platform.version(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "total_memory": psutil.virtual_memory().total,
        }
        
        # Determine overall status
        component_statuses = [c.status for c in components]
        overall_status = "healthy"
        if "unhealthy" in component_statuses:
            overall_status = "unhealthy"
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        elif "degraded" in component_statuses:
            overall_status = "degraded"
        
        # Calculate uptime
        process = psutil.Process(os.getpid())
        uptime_seconds = int(time.time() - process.create_time())
        
        # Construct response
        health_response = DetailedHealthResponse(
            status=overall_status,
            version=settings.VERSION,
            environment=settings.ENVIRONMENT,
            uptime_seconds=uptime_seconds,
            timestamp=datetime.now(),
            components=components,
            memory_tiers=memory_tiers,
            resources=resources,
            host_info=host_info,
        )
        
        # Log health check results
        execution_time = (time.time() - start_time) * 1000
        logger.info(f"Detailed health check completed in {execution_time:.2f}ms. Status: {overall_status}")
        
        return health_response
    
    except Exception as e:
        logger.exception("Detailed health check failed with unexpected error")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service is unhealthy: {str(e)}",
        )


@router.get(
    "/readiness",
    summary="Readiness probe",
    description="Checks if the service is ready to accept traffic",
    status_code=status.HTTP_200_OK,
    response_description="Service is ready",
    responses={
        503: {"description": "Service is not ready"},
    },
)
async def readiness_probe(request: Request) -> Dict[str, str]:
    """
    Readiness probe endpoint for Kubernetes and other orchestration systems.
    
    This endpoint checks if the service is ready to accept traffic by verifying
    that all required dependencies and components are available.
    
    Returns:
        Dict[str, str]: A simple status response indicating readiness
    
    Raises:
        HTTPException: 503 status code if the service is not ready
    """
    try:
        # Check if the application is in startup mode
        app = request.app
        if hasattr(app, "is_startup_complete") and not app.is_startup_complete:
            logger.info("Readiness check failed: startup not complete")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service startup not complete",
            )
        
        # Check database connection
        db_status = get_db_status()
        if not db_status["connected"]:
            logger.warning(f"Readiness check failed: database connection error: {db_status.get('error')}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database not ready",
            )
        
        # Check memory system initialization
        if not memory_manager.is_initialized():
            logger.warning("Readiness check failed: memory system not initialized")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Memory system not initialized",
            )
        
        return {"status": "ready"}
    
    except Exception as e:
        if not isinstance(e, HTTPException):
            logger.exception("Readiness check failed with unexpected error")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Service is not ready: {str(e)}",
            )
        raise


@router.get(
    "/liveness",
    summary="Liveness probe",
    description="Checks if the service is alive and functioning",
    status_code=status.HTTP_200_OK,
    response_description="Service is alive",
    responses={
        503: {"description": "Service is not functioning properly"},
    },
)
async def liveness_probe() -> Dict[str, str]:
    """
    Liveness probe endpoint for Kubernetes and other orchestration systems.
    
    This endpoint performs a minimal check to verify that the service is alive
    and functioning properly. It should be lightweight and fast.
    
    Returns:
        Dict[str, str]: A simple status response indicating liveness
    
    Raises:
        HTTPException: 503 status code if the service is not functioning properly
    """
    try:
        # Simple check to verify the service is running
        # This should be very lightweight and not depend on external services
        return {"status": "alive"}
    
    except Exception as e:
        logger.exception("Liveness check failed with unexpected error")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service is not functioning properly: {str(e)}",
        )