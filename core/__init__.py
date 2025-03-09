"""
NeuroCognitive Architecture (NCA) Core Module
=============================================

This module serves as the foundation for the NCA system, providing core domain logic,
models, and utilities that implement the biological-inspired cognitive architecture.

The core module is responsible for:
1. Defining the fundamental cognitive components and their interactions
2. Implementing the base classes for the three-tiered memory system
3. Providing health dynamics and homeostasis mechanisms
4. Establishing the event system for internal communication
5. Defining core exceptions and error handling patterns

Usage:
------
Import core components directly from this module:

    from neuroca.core import CognitiveComponent, MemoryBase, HealthMonitor
    
    # Create a cognitive component
    component = CognitiveComponent(name="reasoning_module")
    
    # Access health monitoring
    health_status = component.health.status()

Version and Metadata:
--------------------
This module tracks version information and provides metadata about the core system.
"""

import logging
import sys
import typing
from importlib.metadata import version, PackageNotFoundError
from typing import Dict, Any, Optional, List, Tuple, Union, Callable

# Configure module-level logger
logger = logging.getLogger(__name__)

# Package version management
try:
    __version__ = version("neuroca")
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"
    logger.debug("Package metadata not found, using default version")

# Core module configuration
__all__ = [
    # Base classes
    "CognitiveComponent",
    "MemoryBase",
    "HealthMonitor",
    "EventSystem",
    
    # Exceptions
    "CoreException",
    "MemoryAccessError",
    "HealthCriticalError",
    "ConfigurationError",
    
    # Utilities
    "initialize_core",
    "get_core_status",
    "register_component",
    
    # Version info
    "__version__",
]

# Core exceptions
class CoreException(Exception):
    """Base exception for all core-related errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
        logger.error(f"CoreException: {message}", extra=self.details)


class MemoryAccessError(CoreException):
    """Exception raised for errors in memory access operations."""
    pass


class HealthCriticalError(CoreException):
    """Exception raised when a component's health reaches a critical state."""
    pass


class ConfigurationError(CoreException):
    """Exception raised for errors in configuration settings."""
    pass


# Core component registry
_component_registry: Dict[str, 'CognitiveComponent'] = {}


class CognitiveComponent:
    """
    Base class for all cognitive components in the NCA system.
    
    Cognitive components are the fundamental building blocks of the architecture,
    representing distinct functional units with specific cognitive responsibilities.
    Each component has its own health monitoring, memory access, and event handling.
    
    Attributes:
        name (str): Unique identifier for the component
        health (HealthMonitor): Health monitoring system for this component
        active (bool): Whether the component is currently active
        metadata (Dict[str, Any]): Additional metadata about the component
    """
    
    def __init__(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a new cognitive component.
        
        Args:
            name: Unique identifier for the component
            metadata: Optional metadata dictionary with component information
        
        Raises:
            ConfigurationError: If a component with the same name already exists
        """
        if name in _component_registry:
            raise ConfigurationError(f"Component with name '{name}' already exists")
        
        self.name = name
        self.metadata = metadata or {}
        self.active = False
        self.health = HealthMonitor(component_name=name)
        self._event_handlers: Dict[str, List[Callable]] = {}
        
        # Register the component
        _component_registry[name] = self
        logger.debug(f"Cognitive component '{name}' initialized")
    
    def activate(self) -> bool:
        """
        Activate the component, making it operational.
        
        Returns:
            bool: True if activation was successful, False otherwise
        """
        try:
            self.active = True
            logger.info(f"Component '{self.name}' activated")
            return True
        except Exception as e:
            logger.error(f"Failed to activate component '{self.name}': {str(e)}")
            self.active = False
            return False
    
    def deactivate(self) -> bool:
        """
        Deactivate the component, suspending its operations.
        
        Returns:
            bool: True if deactivation was successful, False otherwise
        """
        try:
            self.active = False
            logger.info(f"Component '{self.name}' deactivated")
            return True
        except Exception as e:
            logger.error(f"Failed to deactivate component '{self.name}': {str(e)}")
            return False
    
    def register_event_handler(self, event_type: str, handler: Callable) -> None:
        """
        Register a handler for a specific event type.
        
        Args:
            event_type: The type of event to handle
            handler: Callback function to execute when event occurs
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        
        self._event_handlers[event_type].append(handler)
        logger.debug(f"Event handler registered for '{event_type}' in component '{self.name}'")
    
    def __repr__(self) -> str:
        return f"CognitiveComponent(name='{self.name}', active={self.active})"


class MemoryBase:
    """
    Base class for all memory implementations in the three-tiered memory system.
    
    The NCA memory system is inspired by human memory structures, with different
    tiers for working memory, episodic memory, and semantic memory. This base class
    provides common functionality for all memory implementations.
    
    Attributes:
        capacity (int): Maximum capacity of this memory store
        persistence_level (float): How persistent memories are (0.0-1.0)
        access_count (int): Number of times this memory has been accessed
    """
    
    def __init__(self, capacity: int, persistence_level: float = 0.5):
        """
        Initialize a new memory component.
        
        Args:
            capacity: Maximum number of items this memory can store
            persistence_level: How persistent memories are (0.0-1.0)
        
        Raises:
            ConfigurationError: If invalid configuration parameters are provided
        """
        if capacity <= 0:
            raise ConfigurationError("Memory capacity must be greater than zero")
        
        if not 0.0 <= persistence_level <= 1.0:
            raise ConfigurationError("Persistence level must be between 0.0 and 1.0")
        
        self.capacity = capacity
        self.persistence_level = persistence_level
        self.access_count = 0
        self._storage: Dict[str, Any] = {}
        
        logger.debug(f"Memory initialized with capacity {capacity}")
    
    def store(self, key: str, value: Any) -> bool:
        """
        Store a value in memory.
        
        Args:
            key: Identifier for the stored value
            value: The data to store
            
        Returns:
            bool: True if storage was successful, False otherwise
            
        Raises:
            MemoryAccessError: If the memory is at capacity
        """
        if len(self._storage) >= self.capacity:
            raise MemoryAccessError("Memory at capacity, cannot store new item")
        
        try:
            self._storage[key] = value
            logger.debug(f"Stored item with key '{key}' in memory")
            return True
        except Exception as e:
            logger.error(f"Failed to store item in memory: {str(e)}")
            raise MemoryAccessError(f"Storage operation failed: {str(e)}")
    
    def retrieve(self, key: str) -> Any:
        """
        Retrieve a value from memory.
        
        Args:
            key: Identifier for the value to retrieve
            
        Returns:
            The stored value
            
        Raises:
            MemoryAccessError: If the key does not exist
        """
        self.access_count += 1
        
        if key not in self._storage:
            raise MemoryAccessError(f"Key '{key}' not found in memory")
        
        return self._storage[key]
    
    def __repr__(self) -> str:
        return f"MemoryBase(capacity={self.capacity}, items={len(self._storage)})"


class HealthMonitor:
    """
    Monitors and manages the health of a cognitive component.
    
    The health monitoring system is inspired by biological homeostasis, tracking
    various vital parameters and triggering responses when they fall outside
    acceptable ranges.
    
    Attributes:
        component_name (str): Name of the component being monitored
        vitals (Dict[str, float]): Current vital signs
        thresholds (Dict[str, Tuple[float, float]]): Min/max thresholds for vitals
    """
    
    def __init__(self, component_name: str):
        """
        Initialize a health monitor for a component.
        
        Args:
            component_name: Name of the component being monitored
        """
        self.component_name = component_name
        self.vitals = {
            "energy": 1.0,
            "stability": 1.0,
            "responsiveness": 1.0,
            "error_rate": 0.0,
        }
        self.thresholds = {
            "energy": (0.2, 1.0),
            "stability": (0.3, 1.0),
            "responsiveness": (0.4, 1.0),
            "error_rate": (0.0, 0.3),
        }
        logger.debug(f"Health monitor initialized for component '{component_name}'")
    
    def update_vital(self, vital_name: str, value: float) -> None:
        """
        Update a specific vital sign.
        
        Args:
            vital_name: Name of the vital to update
            value: New value for the vital
            
        Raises:
            ConfigurationError: If the vital name is invalid
            HealthCriticalError: If the vital exceeds critical thresholds
        """
        if vital_name not in self.vitals:
            raise ConfigurationError(f"Unknown vital sign: {vital_name}")
        
        self.vitals[vital_name] = value
        
        # Check if the vital is outside acceptable thresholds
        if vital_name in self.thresholds:
            min_val, max_val = self.thresholds[vital_name]
            if value < min_val or value > max_val:
                logger.warning(
                    f"Vital '{vital_name}' for component '{self.component_name}' "
                    f"outside acceptable range: {value} (range: {min_val}-{max_val})"
                )
                
                # If severely outside range, raise critical error
                if value < min_val * 0.5 or value > max_val * 1.5:
                    raise HealthCriticalError(
                        f"Critical health issue in component '{self.component_name}'",
                        details={"vital": vital_name, "value": value, "threshold": (min_val, max_val)}
                    )
    
    def status(self) -> Dict[str, Any]:
        """
        Get the current health status.
        
        Returns:
            Dict containing health status information
        """
        # Calculate overall health score (simple average)
        health_factors = [
            self.vitals["energy"],
            self.vitals["stability"],
            self.vitals["responsiveness"],
            1.0 - self.vitals["error_rate"]  # Invert error rate for scoring
        ]
        overall_health = sum(health_factors) / len(health_factors)
        
        return {
            "component": self.component_name,
            "vitals": self.vitals.copy(),
            "overall_health": overall_health,
            "status": "healthy" if overall_health > 0.7 else "degraded" if overall_health > 0.4 else "critical"
        }
    
    def __repr__(self) -> str:
        status = self.status()
        return f"HealthMonitor(component='{self.component_name}', status='{status['status']}')"


class EventSystem:
    """
    Central event system for communication between components.
    
    The event system allows for decoupled communication between components
    through a publish-subscribe pattern, enabling complex interactions without
    tight coupling.
    
    Attributes:
        subscribers (Dict): Mapping of event types to subscriber callbacks
    """
    
    def __init__(self):
        """Initialize a new event system."""
        self.subscribers: Dict[str, List[Callable]] = {}
        logger.debug("Event system initialized")
    
    def subscribe(self, event_type: str, callback: Callable) -> None:
        """
        Subscribe to an event type.
        
        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event occurs
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        self.subscribers[event_type].append(callback)
        logger.debug(f"Subscribed to event type: {event_type}")
    
    def publish(self, event_type: str, data: Any = None) -> int:
        """
        Publish an event to all subscribers.
        
        Args:
            event_type: Type of event to publish
            data: Data to pass to subscribers
            
        Returns:
            int: Number of subscribers notified
        """
        if event_type not in self.subscribers:
            logger.debug(f"No subscribers for event type: {event_type}")
            return 0
        
        count = 0
        for callback in self.subscribers[event_type]:
            try:
                callback(data)
                count += 1
            except Exception as e:
                logger.error(f"Error in event subscriber: {str(e)}")
        
        logger.debug(f"Published event '{event_type}' to {count} subscribers")
        return count
    
    def unsubscribe(self, event_type: str, callback: Callable) -> bool:
        """
        Unsubscribe from an event type.
        
        Args:
            event_type: Type of event to unsubscribe from
            callback: Callback to remove
            
        Returns:
            bool: True if successfully unsubscribed, False otherwise
        """
        if event_type not in self.subscribers:
            return False
        
        if callback in self.subscribers[event_type]:
            self.subscribers[event_type].remove(callback)
            logger.debug(f"Unsubscribed from event type: {event_type}")
            return True
        
        return False
    
    def __repr__(self) -> str:
        event_counts = {event: len(subs) for event, subs in self.subscribers.items()}
        return f"EventSystem(events={len(event_counts)}, subscribers={sum(event_counts.values())})"


# Global event system instance
event_system = EventSystem()


def initialize_core() -> Dict[str, Any]:
    """
    Initialize the core system and return status information.
    
    This function should be called at application startup to ensure
    the core system is properly initialized.
    
    Returns:
        Dict containing initialization status and information
    """
    logger.info(f"Initializing NeuroCognitive Architecture Core v{__version__}")
    
    # Perform any necessary startup tasks
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    initialization_status = {
        "version": __version__,
        "python_version": python_version,
        "components_registered": len(_component_registry),
        "status": "initialized",
        "timestamp": __import__("datetime").datetime.now().isoformat()
    }
    
    logger.info(f"Core system initialized successfully")
    return initialization_status


def get_core_status() -> Dict[str, Any]:
    """
    Get the current status of the core system.
    
    Returns:
        Dict containing status information about the core system
    """
    component_statuses = {}
    for name, component in _component_registry.items():
        component_statuses[name] = {
            "active": component.active,
            "health": component.health.status()
        }
    
    return {
        "version": __version__,
        "components": component_statuses,
        "component_count": len(_component_registry),
        "timestamp": __import__("datetime").datetime.now().isoformat()
    }


def register_component(component: CognitiveComponent) -> bool:
    """
    Register a cognitive component with the core system.
    
    Args:
        component: The component to register
        
    Returns:
        bool: True if registration was successful, False otherwise
    """
    if component.name in _component_registry:
        logger.warning(f"Component with name '{component.name}' already registered")
        return False
    
    _component_registry[component.name] = component
    logger.info(f"Component '{component.name}' registered with core system")
    return True


# Initialize logging for the core module
def _setup_logging():
    """Configure logging for the core module."""
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    module_logger = logging.getLogger("neuroca.core")
    module_logger.addHandler(handler)
    module_logger.setLevel(logging.INFO)


# Run logging setup when module is imported
_setup_logging()