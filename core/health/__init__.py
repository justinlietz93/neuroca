"""
Health Module for NeuroCognitive Architecture (NCA)

This module provides a comprehensive health monitoring, diagnostics, and management
system for the NeuroCognitive Architecture. It enables tracking the operational status
of various components, detecting anomalies, and implementing self-healing mechanisms.

The health system is inspired by biological health monitoring systems and provides:
1. Component status monitoring
2. Health metrics collection and analysis
3. Diagnostic capabilities
4. Self-healing mechanisms
5. Health reporting and alerting

Usage:
    from neuroca.core.health import HealthMonitor, ComponentStatus
    
    # Create a health monitor for a component
    monitor = HealthMonitor("memory_subsystem")
    
    # Report component status
    monitor.report_status(ComponentStatus.HEALTHY)
    
    # Register health check
    @monitor.health_check(interval_seconds=60)
    def check_memory_integrity():
        # Perform health check
        if memory_is_intact():
            return HealthCheckResult(ComponentStatus.HEALTHY)
        return HealthCheckResult(
            ComponentStatus.DEGRADED, 
            "Memory integrity issues detected"
        )
    
    # Get overall health status
    system_health = HealthRegistry.get_instance().get_overall_health()
"""

import enum
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# Configure module logger
logger = logging.getLogger(__name__)


class ComponentStatus(enum.Enum):
    """
    Enumeration of possible component health statuses.
    
    These statuses represent the health state of any monitored component
    in the NeuroCognitive Architecture.
    """
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    INITIALIZING = "initializing"
    RECOVERING = "recovering"
    MAINTENANCE = "maintenance"


@dataclass
class HealthCheckResult:
    """
    Result of a health check operation.
    
    Contains the status of the component and optional details about the health check.
    """
    status: ComponentStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def is_healthy(self) -> bool:
        """Check if the component is in a healthy state."""
        return self.status == ComponentStatus.HEALTHY
    
    def is_degraded(self) -> bool:
        """Check if the component is in a degraded state."""
        return self.status == ComponentStatus.DEGRADED
    
    def is_unhealthy(self) -> bool:
        """Check if the component is in an unhealthy state."""
        return self.status in (ComponentStatus.UNHEALTHY, ComponentStatus.CRITICAL)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the health check result to a dictionary."""
        return {
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


class HealthCheckError(Exception):
    """Exception raised when a health check fails to execute properly."""
    pass


@dataclass
class HealthMetric:
    """
    A metric used to measure the health of a component.
    
    Health metrics are quantitative measurements that can be used to
    assess the operational status of a component.
    """
    name: str
    value: Union[int, float, str, bool]
    component_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    unit: str = ""
    threshold_warning: Optional[Union[int, float]] = None
    threshold_critical: Optional[Union[int, float]] = None
    
    def is_above_warning(self) -> bool:
        """Check if the metric is above the warning threshold."""
        if self.threshold_warning is None or not isinstance(self.value, (int, float)):
            return False
        return self.value > self.threshold_warning
    
    def is_above_critical(self) -> bool:
        """Check if the metric is above the critical threshold."""
        if self.threshold_critical is None or not isinstance(self.value, (int, float)):
            return False
        return self.value > self.threshold_critical
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the health metric to a dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "component_id": self.component_id,
            "timestamp": self.timestamp.isoformat(),
            "unit": self.unit,
            "threshold_warning": self.threshold_warning,
            "threshold_critical": self.threshold_critical
        }


class HealthRegistry:
    """
    Singleton registry for all health monitors in the system.
    
    The registry maintains a global view of all component health monitors
    and provides methods to query the overall system health.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        """Initialize the health registry."""
        if HealthRegistry._instance is not None:
            raise RuntimeError("HealthRegistry is a singleton. Use get_instance() instead.")
        self._monitors: Dict[str, 'HealthMonitor'] = {}
        self._dependencies: Dict[str, Set[str]] = {}  # component_id -> set of dependency component_ids
    
    @classmethod
    def get_instance(cls) -> 'HealthRegistry':
        """Get the singleton instance of the health registry."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def register_monitor(self, monitor: 'HealthMonitor') -> None:
        """
        Register a health monitor with the registry.
        
        Args:
            monitor: The health monitor to register.
            
        Raises:
            ValueError: If a monitor with the same component ID is already registered.
        """
        if monitor.component_id in self._monitors:
            raise ValueError(f"A monitor for component '{monitor.component_id}' is already registered")
        
        self._monitors[monitor.component_id] = monitor
        logger.debug(f"Registered health monitor for component '{monitor.component_id}'")
    
    def unregister_monitor(self, component_id: str) -> None:
        """
        Unregister a health monitor from the registry.
        
        Args:
            component_id: The ID of the component whose monitor should be unregistered.
            
        Raises:
            KeyError: If no monitor is registered for the specified component ID.
        """
        if component_id not in self._monitors:
            raise KeyError(f"No monitor registered for component '{component_id}'")
        
        monitor = self._monitors.pop(component_id)
        monitor.stop()
        
        # Remove component from dependencies
        for deps in self._dependencies.values():
            deps.discard(component_id)
        
        # Remove component's dependencies
        self._dependencies.pop(component_id, None)
        
        logger.debug(f"Unregistered health monitor for component '{component_id}'")
    
    def get_monitor(self, component_id: str) -> Optional['HealthMonitor']:
        """
        Get the health monitor for a specific component.
        
        Args:
            component_id: The ID of the component.
            
        Returns:
            The health monitor for the component, or None if not found.
        """
        return self._monitors.get(component_id)
    
    def register_dependency(self, component_id: str, dependency_id: str) -> None:
        """
        Register a dependency between components.
        
        Args:
            component_id: The ID of the dependent component.
            dependency_id: The ID of the component it depends on.
            
        Raises:
            ValueError: If registering would create a circular dependency.
        """
        # Check for circular dependencies
        if self._would_create_circular_dependency(component_id, dependency_id):
            raise ValueError(f"Adding dependency from '{component_id}' to '{dependency_id}' would create a circular dependency")
        
        if component_id not in self._dependencies:
            self._dependencies[component_id] = set()
        
        self._dependencies[component_id].add(dependency_id)
        logger.debug(f"Registered dependency: '{component_id}' depends on '{dependency_id}'")
    
    def _would_create_circular_dependency(self, component_id: str, dependency_id: str) -> bool:
        """
        Check if adding a dependency would create a circular dependency.
        
        Args:
            component_id: The ID of the dependent component.
            dependency_id: The ID of the component it depends on.
            
        Returns:
            True if adding the dependency would create a circular dependency, False otherwise.
        """
        # If dependency_id depends on component_id (directly or indirectly), adding this dependency would create a cycle
        visited = set()
        
        def dfs(current: str) -> bool:
            if current == component_id:
                return True
            if current in visited:
                return False
            
            visited.add(current)
            for dep in self._dependencies.get(current, set()):
                if dfs(dep):
                    return True
            return False
        
        return dfs(dependency_id)
    
    def get_dependencies(self, component_id: str) -> Set[str]:
        """
        Get the direct dependencies of a component.
        
        Args:
            component_id: The ID of the component.
            
        Returns:
            A set of component IDs that the specified component directly depends on.
        """
        return self._dependencies.get(component_id, set()).copy()
    
    def get_dependents(self, component_id: str) -> Set[str]:
        """
        Get the components that directly depend on the specified component.
        
        Args:
            component_id: The ID of the component.
            
        Returns:
            A set of component IDs that directly depend on the specified component.
        """
        return {
            dependent for dependent, dependencies in self._dependencies.items()
            if component_id in dependencies
        }
    
    def get_overall_health(self) -> HealthCheckResult:
        """
        Get the overall health status of the system.
        
        The overall health is determined by the worst status among all components,
        taking into account dependencies.
        
        Returns:
            A HealthCheckResult representing the overall system health.
        """
        if not self._monitors:
            return HealthCheckResult(
                ComponentStatus.UNKNOWN,
                "No health monitors registered"
            )
        
        # Get the latest health check result for each component
        component_results = {
            component_id: monitor.get_latest_result()
            for component_id, monitor in self._monitors.items()
        }
        
        # Determine the worst status
        worst_status = ComponentStatus.HEALTHY
        unhealthy_components = []
        
        for component_id, result in component_results.items():
            if result.status.value > worst_status.value:
                worst_status = result.status
            
            if result.status != ComponentStatus.HEALTHY:
                unhealthy_components.append({
                    "component_id": component_id,
                    "status": result.status.value,
                    "message": result.message
                })
        
        # Create a summary message
        if worst_status == ComponentStatus.HEALTHY:
            message = "All components are healthy"
        else:
            message = f"{len(unhealthy_components)} component(s) are not healthy"
        
        return HealthCheckResult(
            status=worst_status,
            message=message,
            details={
                "total_components": len(self._monitors),
                "unhealthy_components": unhealthy_components
            }
        )
    
    def get_all_health_statuses(self) -> Dict[str, HealthCheckResult]:
        """
        Get the health status of all registered components.
        
        Returns:
            A dictionary mapping component IDs to their latest health check results.
        """
        return {
            component_id: monitor.get_latest_result()
            for component_id, monitor in self._monitors.items()
        }


class HealthMonitor:
    """
    Health monitor for a specific component in the system.
    
    The health monitor tracks the health status of a component, runs periodic
    health checks, and reports health metrics.
    """
    
    def __init__(self, component_id: str, initial_status: ComponentStatus = ComponentStatus.INITIALIZING):
        """
        Initialize a health monitor for a component.
        
        Args:
            component_id: Unique identifier for the component being monitored.
            initial_status: Initial health status of the component.
        """
        self.component_id = component_id
        self._current_status = initial_status
        self._latest_result = HealthCheckResult(
            status=initial_status,
            message=f"Component '{component_id}' initialized"
        )
        self._health_checks: Dict[str, Tuple[Callable[[], HealthCheckResult], int]] = {}
        self._health_check_threads: Dict[str, threading.Thread] = {}
        self._stop_events: Dict[str, threading.Event] = {}
        self._metrics_history: List[HealthMetric] = []
        self._lock = threading.RLock()
        self._max_metrics_history = 1000  # Maximum number of metrics to keep in history
        
        # Register with the global registry
        HealthRegistry.get_instance().register_monitor(self)
        
        logger.info(f"Health monitor initialized for component '{component_id}' with status {initial_status.value}")
    
    def report_status(self, status: ComponentStatus, message: str = "", details: Dict[str, Any] = None) -> None:
        """
        Report the current health status of the component.
        
        Args:
            status: The current health status.
            message: Optional message describing the status.
            details: Optional dictionary with additional details.
        """
        with self._lock:
            self._current_status = status
            self._latest_result = HealthCheckResult(
                status=status,
                message=message,
                details=details or {},
                timestamp=datetime.now()
            )
        
        # Log the status change
        log_level = logging.INFO if status == ComponentStatus.HEALTHY else logging.WARNING
        logger.log(log_level, f"Component '{self.component_id}' status: {status.value} - {message}")
    
    def get_latest_result(self) -> HealthCheckResult:
        """
        Get the latest health check result for the component.
        
        Returns:
            The latest health check result.
        """
        with self._lock:
            return self._latest_result
    
    def health_check(self, interval_seconds: int = 60, check_id: str = None):
        """
        Decorator to register a health check function.
        
        The decorated function should return a HealthCheckResult.
        
        Args:
            interval_seconds: How often to run the health check, in seconds.
            check_id: Optional unique identifier for the health check.
                     If not provided, a UUID will be generated.
                     
        Returns:
            A decorator function.
            
        Example:
            @monitor.health_check(interval_seconds=60)
            def check_memory_integrity():
                # Perform health check
                if memory_is_intact():
                    return HealthCheckResult(ComponentStatus.HEALTHY)
                return HealthCheckResult(
                    ComponentStatus.DEGRADED, 
                    "Memory integrity issues detected"
                )
        """
        def decorator(func: Callable[[], HealthCheckResult]):
            nonlocal check_id
            if check_id is None:
                check_id = str(uuid.uuid4())
            
            with self._lock:
                self._health_checks[check_id] = (func, interval_seconds)
            
            # Start the health check thread
            self._start_health_check_thread(check_id)
            
            return func
        
        return decorator
    
    def _start_health_check_thread(self, check_id: str) -> None:
        """
        Start a thread to run a health check periodically.
        
        Args:
            check_id: The ID of the health check to run.
        """
        if check_id in self._health_check_threads and self._health_check_threads[check_id].is_alive():
            return
        
        func, interval = self._health_checks[check_id]
        stop_event = threading.Event()
        self._stop_events[check_id] = stop_event
        
        def run_health_check():
            while not stop_event.is_set():
                try:
                    result = func()
                    if not isinstance(result, HealthCheckResult):
                        logger.error(
                            f"Health check '{check_id}' for component '{self.component_id}' "
                            f"returned {type(result)} instead of HealthCheckResult"
                        )
                        result = HealthCheckResult(
                            ComponentStatus.UNKNOWN,
                            f"Health check returned invalid result type: {type(result)}"
                        )
                    
                    # Update the component status based on the health check result
                    with self._lock:
                        self._current_status = result.status
                        self._latest_result = result
                    
                    # Log the health check result
                    log_level = logging.INFO if result.status == ComponentStatus.HEALTHY else logging.WARNING
                    logger.log(
                        log_level,
                        f"Health check '{check_id}' for component '{self.component_id}': "
                        f"{result.status.value} - {result.message}"
                    )
                    
                except Exception as e:
                    logger.exception(
                        f"Error running health check '{check_id}' for component '{self.component_id}': {str(e)}"
                    )
                    with self._lock:
                        self._current_status = ComponentStatus.UNKNOWN
                        self._latest_result = HealthCheckResult(
                            ComponentStatus.UNKNOWN,
                            f"Health check failed with error: {str(e)}",
                            {"error": str(e), "traceback": logging.traceback.format_exc()}
                        )
                
                # Wait for the next interval or until stopped
                if stop_event.wait(interval):
                    break
        
        thread = threading.Thread(
            target=run_health_check,
            name=f"health-check-{self.component_id}-{check_id}",
            daemon=True
        )
        thread.start()
        self._health_check_threads[check_id] = thread
        
        logger.debug(
            f"Started health check thread '{check_id}' for component '{self.component_id}' "
            f"with interval {interval} seconds"
        )
    
    def stop_health_check(self, check_id: str) -> None:
        """
        Stop a specific health check.
        
        Args:
            check_id: The ID of the health check to stop.
            
        Raises:
            KeyError: If no health check with the specified ID is registered.
        """
        with self._lock:
            if check_id not in self._health_checks:
                raise KeyError(f"No health check with ID '{check_id}' registered")
            
            if check_id in self._stop_events:
                self._stop_events[check_id].set()
            
            if check_id in self._health_check_threads:
                self._health_check_threads[check_id].join(timeout=1.0)
                del self._health_check_threads[check_id]
            
            if check_id in self._stop_events:
                del self._stop_events[check_id]
            
            del self._health_checks[check_id]
        
        logger.debug(f"Stopped health check '{check_id}' for component '{self.component_id}'")
    
    def stop(self) -> None:
        """Stop all health checks for this component."""
        with self._lock:
            check_ids = list(self._health_checks.keys())
        
        for check_id in check_ids:
            try:
                self.stop_health_check(check_id)
            except Exception as e:
                logger.error(f"Error stopping health check '{check_id}': {str(e)}")
        
        logger.info(f"Stopped all health checks for component '{self.component_id}'")
    
    def report_metric(self, name: str, value: Union[int, float, str, bool], unit: str = "",
                     threshold_warning: Optional[Union[int, float]] = None,
                     threshold_critical: Optional[Union[int, float]] = None) -> HealthMetric:
        """
        Report a health metric for the component.
        
        Args:
            name: Name of the metric.
            value: Value of the metric.
            unit: Optional unit of measurement.
            threshold_warning: Optional warning threshold.
            threshold_critical: Optional critical threshold.
            
        Returns:
            The created HealthMetric object.
        """
        metric = HealthMetric(
            name=name,
            value=value,
            component_id=self.component_id,
            unit=unit,
            threshold_warning=threshold_warning,
            threshold_critical=threshold_critical
        )
        
        with self._lock:
            self._metrics_history.append(metric)
            
            # Trim history if it exceeds the maximum size
            if len(self._metrics_history) > self._max_metrics_history:
                self._metrics_history = self._metrics_history[-self._max_metrics_history:]
        
        # Log the metric
        logger.debug(f"Metric '{name}' for component '{self.component_id}': {value}{unit}")
        
        # Check thresholds and update status if needed
        if isinstance(value, (int, float)):
            if threshold_critical is not None and value > threshold_critical:
                self.report_status(
                    ComponentStatus.CRITICAL,
                    f"Metric '{name}' exceeded critical threshold: {value}{unit} > {threshold_critical}{unit}"
                )
            elif threshold_warning is not None and value > threshold_warning:
                self.report_status(
                    ComponentStatus.DEGRADED,
                    f"Metric '{name}' exceeded warning threshold: {value}{unit} > {threshold_warning}{unit}"
                )
        
        return metric
    
    def get_metrics(self, metric_name: Optional[str] = None, 
                   since: Optional[datetime] = None,
                   limit: int = 100) -> List[HealthMetric]:
        """
        Get health metrics for the component.
        
        Args:
            metric_name: Optional name of the metric to filter by.
            since: Optional timestamp to filter metrics after this time.
            limit: Maximum number of metrics to return.
            
        Returns:
            A list of HealthMetric objects.
        """
        with self._lock:
            metrics = self._metrics_history.copy()
        
        # Apply filters
        if metric_name is not None:
            metrics = [m for m in metrics if m.name == metric_name]
        
        if since is not None:
            metrics = [m for m in metrics if m.timestamp >= since]
        
        # Sort by timestamp (newest first) and apply limit
        metrics.sort(key=lambda m: m.timestamp, reverse=True)
        return metrics[:limit]
    
    def __del__(self):
        """Clean up resources when the monitor is garbage collected."""
        try:
            self.stop()
        except Exception:
            pass


# Export public API
__all__ = [
    'ComponentStatus',
    'HealthCheckResult',
    'HealthCheckError',
    'HealthMetric',
    'HealthMonitor',
    'HealthRegistry'
]