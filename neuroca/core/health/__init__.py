"""
Health monitoring and regulation system for the NeuroCognitive Architecture.

This module implements biologically-inspired health dynamics including:
- Component status tracking for system health monitoring
- Energy management for tracking resource usage across operations
- Attention allocation for focus and distraction mechanisms
- Homeostatic regulation for system stability
- Metrics collection for comprehensive health monitoring

The health system provides feedback loops that regulate cognitive processes
based on available resources, current load, and system priorities, similar
to biological systems that maintain homeostasis.
"""

# Import key components for easier access
from neuroca.core.health.component import ComponentStatus
from neuroca.core.health.monitor import HealthMonitor, HealthCheckResult
from neuroca.core.health.registry import HealthRegistry
from neuroca.core.health.monitoring import get_health_report, register_health_check

__all__ = [
    'ComponentStatus',
    'HealthCheckResult',
    'HealthMonitor',
    'HealthRegistry',
    'get_health_report',
    'register_health_check',
] 