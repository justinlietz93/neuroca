"""
Health Dynamics Module for NeuroCognitive Architecture (NCA)

This module implements the dynamic health modeling system for the NCA, providing
mechanisms to track, update, and manage the health state of cognitive components.
It models biological-inspired health dynamics including homeostasis, adaptation,
fatigue, recovery, and stress responses that affect cognitive performance.

The module provides:
1. Health state tracking and management
2. Dynamic health parameter evolution over time
3. Homeostatic regulation mechanisms
4. Stress and recovery modeling
5. Health impact on cognitive performance
6. Health event generation and handling

Usage:
    health_manager = HealthDynamicsManager()
    
    # Initialize component health
    component_health = health_manager.initialize_component_health("memory_module_1")
    
    # Update health based on activity
    health_manager.update_health_state(component_health, workload=0.8, duration=300)
    
    # Get current health status
    health_status = health_manager.get_health_status(component_health)
    
    # Apply recovery period
    health_manager.apply_recovery(component_health, duration=600)
"""

import time
import uuid
import logging
import math
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import random
from enum import Enum, auto

# Configure logging
logger = logging.getLogger(__name__)

class HealthParameterType(Enum):
    """Enumeration of health parameter types for cognitive components."""
    ENERGY = auto()
    STABILITY = auto()
    RESPONSIVENESS = auto()
    ACCURACY = auto()
    CAPACITY = auto()
    RESILIENCE = auto()
    ADAPTABILITY = auto()
    STRESS = auto()
    FATIGUE = auto()

class HealthState(Enum):
    """Enumeration of overall health states for cognitive components."""
    OPTIMAL = auto()
    NORMAL = auto()
    DEGRADED = auto()
    IMPAIRED = auto()
    CRITICAL = auto()
    FAILED = auto()

class HealthEventType(Enum):
    """Enumeration of health event types that can occur."""
    PARAMETER_THRESHOLD_CROSSED = auto()
    STATE_CHANGE = auto()
    RECOVERY_STARTED = auto()
    RECOVERY_COMPLETED = auto()
    STRESS_SPIKE = auto()
    ADAPTATION_OCCURRED = auto()
    MAINTENANCE_REQUIRED = auto()
    MAINTENANCE_COMPLETED = auto()
    UNEXPECTED_FLUCTUATION = auto()

@dataclass
class HealthParameter:
    """
    Represents a single health parameter for a cognitive component.
    
    Attributes:
        param_type: Type of health parameter
        current_value: Current value of the parameter (0.0 to 1.0)
        baseline_value: Baseline/optimal value of the parameter
        min_threshold: Threshold below which performance is affected
        critical_threshold: Threshold below which function is severely impaired
        decay_rate: Rate at which parameter decays under load
        recovery_rate: Rate at which parameter recovers during rest
        last_update_time: Timestamp of last parameter update
    """
    param_type: HealthParameterType
    current_value: float = 1.0
    baseline_value: float = 1.0
    min_threshold: float = 0.3
    critical_threshold: float = 0.1
    decay_rate: float = 0.01
    recovery_rate: float = 0.005
    last_update_time: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate parameter values after initialization."""
        if not 0 <= self.current_value <= 1:
            raise ValueError(f"Current value must be between 0 and 1, got {self.current_value}")
        if not 0 <= self.baseline_value <= 1:
            raise ValueError(f"Baseline value must be between 0 and 1, got {self.baseline_value}")
        if not 0 <= self.min_threshold <= 1:
            raise ValueError(f"Min threshold must be between 0 and 1, got {self.min_threshold}")
        if not 0 <= self.critical_threshold <= self.min_threshold:
            raise ValueError(f"Critical threshold must be between 0 and min_threshold, got {self.critical_threshold}")
        if self.decay_rate < 0:
            raise ValueError(f"Decay rate must be non-negative, got {self.decay_rate}")
        if self.recovery_rate < 0:
            raise ValueError(f"Recovery rate must be non-negative, got {self.recovery_rate}")

@dataclass
class HealthEvent:
    """
    Represents a health-related event that occurred in a component.
    
    Attributes:
        event_id: Unique identifier for the event
        component_id: ID of the component where the event occurred
        event_type: Type of health event
        timestamp: When the event occurred
        details: Additional event details
        severity: Event severity (0.0 to 1.0)
        resolved: Whether the event has been resolved
    """
    event_id: str
    component_id: str
    event_type: HealthEventType
    timestamp: datetime
    details: Dict[str, Any]
    severity: float
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate event data after initialization."""
        if not 0 <= self.severity <= 1:
            raise ValueError(f"Severity must be between 0 and 1, got {self.severity}")

@dataclass
class ComponentHealth:
    """
    Represents the overall health state of a cognitive component.
    
    Attributes:
        component_id: Unique identifier for the component
        parameters: Dictionary of health parameters by type
        current_state: Current overall health state
        events: List of health events for this component
        creation_time: When this health tracking was initialized
        last_maintenance_time: When maintenance was last performed
    """
    component_id: str
    parameters: Dict[HealthParameterType, HealthParameter] = field(default_factory=dict)
    current_state: HealthState = HealthState.OPTIMAL
    events: List[HealthEvent] = field(default_factory=list)
    creation_time: datetime = field(default_factory=datetime.now)
    last_maintenance_time: Optional[datetime] = None
    
    def add_event(self, event_type: HealthEventType, details: Dict[str, Any], severity: float) -> HealthEvent:
        """
        Add a new health event to this component.
        
        Args:
            event_type: Type of health event
            details: Additional event details
            severity: Event severity (0.0 to 1.0)
            
        Returns:
            The created HealthEvent object
        """
        event = HealthEvent(
            event_id=str(uuid.uuid4()),
            component_id=self.component_id,
            event_type=event_type,
            timestamp=datetime.now(),
            details=details,
            severity=severity
        )
        self.events.append(event)
        logger.info(f"Health event created for component {self.component_id}: {event_type.name}, severity: {severity:.2f}")
        return event
    
    def resolve_event(self, event_id: str) -> bool:
        """
        Mark a health event as resolved.
        
        Args:
            event_id: ID of the event to resolve
            
        Returns:
            True if event was found and resolved, False otherwise
        """
        for event in self.events:
            if event.event_id == event_id and not event.resolved:
                event.resolved = True
                event.resolution_time = datetime.now()
                logger.info(f"Health event {event_id} resolved for component {self.component_id}")
                return True
        return False
    
    def get_unresolved_events(self) -> List[HealthEvent]:
        """
        Get all unresolved health events for this component.
        
        Returns:
            List of unresolved HealthEvent objects
        """
        return [event for event in self.events if not event.resolved]

class HealthDynamicsManager:
    """
    Manages the health dynamics of cognitive components in the NCA system.
    
    This class provides methods to initialize, update, and monitor the health
    state of cognitive components, modeling biological-inspired dynamics.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the health dynamics manager.
        
        Args:
            config: Optional configuration dictionary for health parameters
        """
        self.config = config or {}
        self.components: Dict[str, ComponentHealth] = {}
        self.default_parameters = self._get_default_parameters()
        logger.info("HealthDynamicsManager initialized")
    
    def _get_default_parameters(self) -> Dict[HealthParameterType, Dict[str, float]]:
        """
        Get default parameter configurations for different health parameter types.
        
        Returns:
            Dictionary mapping parameter types to their default configurations
        """
        return {
            HealthParameterType.ENERGY: {
                "baseline_value": 1.0,
                "min_threshold": 0.3,
                "critical_threshold": 0.1,
                "decay_rate": 0.02,
                "recovery_rate": 0.01
            },
            HealthParameterType.STABILITY: {
                "baseline_value": 1.0,
                "min_threshold": 0.4,
                "critical_threshold": 0.15,
                "decay_rate": 0.01,
                "recovery_rate": 0.008
            },
            HealthParameterType.RESPONSIVENESS: {
                "baseline_value": 1.0,
                "min_threshold": 0.35,
                "critical_threshold": 0.12,
                "decay_rate": 0.015,
                "recovery_rate": 0.012
            },
            HealthParameterType.ACCURACY: {
                "baseline_value": 1.0,
                "min_threshold": 0.5,
                "critical_threshold": 0.2,
                "decay_rate": 0.008,
                "recovery_rate": 0.005
            },
            HealthParameterType.CAPACITY: {
                "baseline_value": 1.0,
                "min_threshold": 0.4,
                "critical_threshold": 0.15,
                "decay_rate": 0.005,
                "recovery_rate": 0.003
            },
            HealthParameterType.RESILIENCE: {
                "baseline_value": 1.0,
                "min_threshold": 0.3,
                "critical_threshold": 0.1,
                "decay_rate": 0.007,
                "recovery_rate": 0.004
            },
            HealthParameterType.ADAPTABILITY: {
                "baseline_value": 1.0,
                "min_threshold": 0.4,
                "critical_threshold": 0.15,
                "decay_rate": 0.01,
                "recovery_rate": 0.007
            },
            HealthParameterType.STRESS: {
                "baseline_value": 0.0,  # Stress starts at 0 and increases
                "min_threshold": 0.6,   # Higher values are worse for stress
                "critical_threshold": 0.8,
                "decay_rate": -0.015,   # Negative because stress increases under load
                "recovery_rate": 0.02   # Stress decreases during recovery
            },
            HealthParameterType.FATIGUE: {
                "baseline_value": 0.0,  # Fatigue starts at 0 and increases
                "min_threshold": 0.5,   # Higher values are worse for fatigue
                "critical_threshold": 0.75,
                "decay_rate": -0.01,    # Negative because fatigue increases under load
                "recovery_rate": 0.015  # Fatigue decreases during recovery
            }
        }
    
    def initialize_component_health(self, component_id: str, 
                                   parameter_types: Optional[List[HealthParameterType]] = None,
                                   custom_params: Optional[Dict[HealthParameterType, Dict[str, float]]] = None) -> ComponentHealth:
        """
        Initialize health tracking for a cognitive component.
        
        Args:
            component_id: Unique identifier for the component
            parameter_types: List of health parameter types to track (default: all types)
            custom_params: Custom parameter configurations by type
            
        Returns:
            Initialized ComponentHealth object
            
        Raises:
            ValueError: If component_id already exists
        """
        if component_id in self.components:
            raise ValueError(f"Component with ID {component_id} already exists")
        
        if parameter_types is None:
            parameter_types = list(HealthParameterType)
        
        component_health = ComponentHealth(component_id=component_id)
        
        # Initialize parameters
        for param_type in parameter_types:
            param_config = self.default_parameters.get(param_type, {}).copy()
            
            # Override with custom parameters if provided
            if custom_params and param_type in custom_params:
                param_config.update(custom_params[param_type])
            
            # Create parameter
            parameter = HealthParameter(
                param_type=param_type,
                current_value=param_config.get("baseline_value", 1.0),
                baseline_value=param_config.get("baseline_value", 1.0),
                min_threshold=param_config.get("min_threshold", 0.3),
                critical_threshold=param_config.get("critical_threshold", 0.1),
                decay_rate=param_config.get("decay_rate", 0.01),
                recovery_rate=param_config.get("recovery_rate", 0.005)
            )
            
            component_health.parameters[param_type] = parameter
        
        self.components[component_id] = component_health
        logger.info(f"Initialized health tracking for component {component_id} with {len(parameter_types)} parameters")
        return component_health
    
    def update_health_state(self, component_health: ComponentHealth, 
                           workload: float, duration: float,
                           parameter_weights: Optional[Dict[HealthParameterType, float]] = None) -> None:
        """
        Update the health state of a component based on workload and duration.
        
        Args:
            component_health: The component health object to update
            workload: Workload intensity (0.0 to 1.0)
            duration: Duration of the workload in seconds
            parameter_weights: Optional weights for different parameters (default: equal weights)
            
        Raises:
            ValueError: If workload is not between 0 and 1
        """
        if not 0 <= workload <= 1:
            raise ValueError(f"Workload must be between 0 and 1, got {workload}")
        
        if duration <= 0:
            raise ValueError(f"Duration must be positive, got {duration}")
        
        # Default weights if not provided
        if parameter_weights is None:
            parameter_weights = {param_type: 1.0 for param_type in component_health.parameters}
        
        # Track state changes
        old_state = component_health.current_state
        
        # Update each parameter
        for param_type, parameter in component_health.parameters.items():
            # Skip parameters not in weights
            if param_type not in parameter_weights:
                continue
                
            weight = parameter_weights[param_type]
            
            # Calculate time since last update
            now = datetime.now()
            time_delta = (now - parameter.last_update_time).total_seconds()
            
            # Apply workload effect with weight
            effective_workload = workload * weight
            
            # For stress and fatigue, they increase under load (negative decay rate)
            if param_type in [HealthParameterType.STRESS, HealthParameterType.FATIGUE]:
                # These parameters increase under load
                decay_amount = -parameter.decay_rate * effective_workload * duration
                new_value = min(1.0, parameter.current_value - decay_amount)  # Cap at 1.0
            else:
                # Other parameters decrease under load
                decay_amount = parameter.decay_rate * effective_workload * duration
                new_value = max(0.0, parameter.current_value - decay_amount)  # Floor at 0.0
            
            # Check for threshold crossings
            if (parameter.current_value > parameter.min_threshold and 
                new_value <= parameter.min_threshold):
                # Crossed minimum threshold
                component_health.add_event(
                    event_type=HealthEventType.PARAMETER_THRESHOLD_CROSSED,
                    details={
                        "parameter": param_type.name,
                        "threshold": "minimum",
                        "previous_value": parameter.current_value,
                        "new_value": new_value
                    },
                    severity=0.5
                )
            
            if (parameter.current_value > parameter.critical_threshold and 
                new_value <= parameter.critical_threshold):
                # Crossed critical threshold
                component_health.add_event(
                    event_type=HealthEventType.PARAMETER_THRESHOLD_CROSSED,
                    details={
                        "parameter": param_type.name,
                        "threshold": "critical",
                        "previous_value": parameter.current_value,
                        "new_value": new_value
                    },
                    severity=0.8
                )
            
            # Update parameter
            parameter.current_value = new_value
            parameter.last_update_time = now
            
            logger.debug(f"Updated {param_type.name} for component {component_health.component_id}: "
                        f"{parameter.current_value:.4f} (workload: {workload:.2f}, duration: {duration:.1f}s)")
        
        # Update overall health state
        self._update_overall_state(component_health)
        
        # Check for state change
        if component_health.current_state != old_state:
            component_health.add_event(
                event_type=HealthEventType.STATE_CHANGE,
                details={
                    "previous_state": old_state.name,
                    "new_state": component_health.current_state.name
                },
                severity=self._get_state_change_severity(old_state, component_health.current_state)
            )
            logger.info(f"Component {component_health.component_id} health state changed: "
                       f"{old_state.name} -> {component_health.current_state.name}")
    
    def apply_recovery(self, component_health: ComponentHealth, 
                      duration: float,
                      parameter_weights: Optional[Dict[HealthParameterType, float]] = None) -> None:
        """
        Apply recovery to a component's health parameters.
        
        Args:
            component_health: The component health object to update
            duration: Duration of the recovery period in seconds
            parameter_weights: Optional weights for different parameters (default: equal weights)
            
        Raises:
            ValueError: If duration is not positive
        """
        if duration <= 0:
            raise ValueError(f"Duration must be positive, got {duration}")
        
        # Default weights if not provided
        if parameter_weights is None:
            parameter_weights = {param_type: 1.0 for param_type in component_health.parameters}
        
        # Track state changes
        old_state = component_health.current_state
        
        # Add recovery started event
        component_health.add_event(
            event_type=HealthEventType.RECOVERY_STARTED,
            details={"duration": duration},
            severity=0.2
        )
        
        # Update each parameter
        for param_type, parameter in component_health.parameters.items():
            # Skip parameters not in weights
            if param_type not in parameter_weights:
                continue
                
            weight = parameter_weights[param_type]
            
            # Calculate time since last update
            now = datetime.now()
            
            # Apply recovery with weight
            recovery_amount = parameter.recovery_rate * weight * duration
            
            # For stress and fatigue, they decrease during recovery
            if param_type in [HealthParameterType.STRESS, HealthParameterType.FATIGUE]:
                # These parameters decrease during recovery (toward baseline of 0.0)
                new_value = max(0.0, parameter.current_value - recovery_amount)
            else:
                # Other parameters increase during recovery (toward baseline of 1.0)
                new_value = min(parameter.baseline_value, parameter.current_value + recovery_amount)
            
            # Check for threshold crossings (in the positive direction)
            if (parameter.current_value <= parameter.min_threshold and 
                new_value > parameter.min_threshold):
                # Crossed minimum threshold (recovery)
                component_health.add_event(
                    event_type=HealthEventType.PARAMETER_THRESHOLD_CROSSED,
                    details={
                        "parameter": param_type.name,
                        "threshold": "minimum",
                        "direction": "recovery",
                        "previous_value": parameter.current_value,
                        "new_value": new_value
                    },
                    severity=0.3
                )
            
            if (parameter.current_value <= parameter.critical_threshold and 
                new_value > parameter.critical_threshold):
                # Crossed critical threshold (recovery)
                component_health.add_event(
                    event_type=HealthEventType.PARAMETER_THRESHOLD_CROSSED,
                    details={
                        "parameter": param_type.name,
                        "threshold": "critical",
                        "direction": "recovery",
                        "previous_value": parameter.current_value,
                        "new_value": new_value
                    },
                    severity=0.4
                )
            
            # Update parameter
            parameter.current_value = new_value
            parameter.last_update_time = now
            
            logger.debug(f"Applied recovery to {param_type.name} for component {component_health.component_id}: "
                        f"{parameter.current_value:.4f} (duration: {duration:.1f}s)")
        
        # Update overall health state
        self._update_overall_state(component_health)
        
        # Check for state change
        if component_health.current_state != old_state:
            component_health.add_event(
                event_type=HealthEventType.STATE_CHANGE,
                details={
                    "previous_state": old_state.name,
                    "new_state": component_health.current_state.name,
                    "cause": "recovery"
                },
                severity=self._get_state_change_severity(old_state, component_health.current_state)
            )
            logger.info(f"Component {component_health.component_id} health state changed after recovery: "
                       f"{old_state.name} -> {component_health.current_state.name}")
        
        # Add recovery completed event
        component_health.add_event(
            event_type=HealthEventType.RECOVERY_COMPLETED,
            details={"duration": duration},
            severity=0.2
        )
    
    def perform_maintenance(self, component_health: ComponentHealth) -> None:
        """
        Perform maintenance on a component to restore its health parameters.
        
        Args:
            component_health: The component health object to maintain
        """
        # Track state changes
        old_state = component_health.current_state
        
        # Add maintenance event
        component_health.add_event(
            event_type=HealthEventType.MAINTENANCE_REQUIRED,
            details={},
            severity=0.4
        )
        
        # Reset all parameters to baseline
        for param_type, parameter in component_health.parameters.items():
            if param_type in [HealthParameterType.STRESS, HealthParameterType.FATIGUE]:
                # These parameters reset to 0.0
                parameter.current_value = 0.0
            else:
                # Other parameters reset to baseline (usually 1.0)
                parameter.current_value = parameter.baseline_value
            
            parameter.last_update_time = datetime.now()
            
            logger.debug(f"Reset {param_type.name} for component {component_health.component_id} to "
                        f"{parameter.current_value:.4f} during maintenance")
        
        # Update maintenance time
        component_health.last_maintenance_time = datetime.now()
        
        # Update overall health state
        self._update_overall_state(component_health)
        
        # Add maintenance completed event
        component_health.add_event(
            event_type=HealthEventType.MAINTENANCE_COMPLETED,
            details={},
            severity=0.2
        )
        
        # Check for state change
        if component_health.current_state != old_state:
            component_health.add_event(
                event_type=HealthEventType.STATE_CHANGE,
                details={
                    "previous_state": old_state.name,
                    "new_state": component_health.current_state.name,
                    "cause": "maintenance"
                },
                severity=self._get_state_change_severity(old_state, component_health.current_state)
            )
            logger.info(f"Component {component_health.component_id} health state changed after maintenance: "
                       f"{old_state.name} -> {component_health.current_state.name}")
    
    def apply_stress_event(self, component_health: ComponentHealth, 
                          intensity: float, 
                          affected_parameters: Optional[List[HealthParameterType]] = None) -> None:
        """
        Apply a sudden stress event to a component.
        
        Args:
            component_health: The component health object to update
            intensity: Intensity of the stress event (0.0 to 1.0)
            affected_parameters: List of parameters affected by the stress event
            
        Raises:
            ValueError: If intensity is not between 0 and 1
        """
        if not 0 <= intensity <= 1:
            raise ValueError(f"Intensity must be between 0 and 1, got {intensity}")
        
        # Default to all parameters if not specified
        if affected_parameters is None:
            affected_parameters = list(component_health.parameters.keys())
        
        # Track state changes
        old_state = component_health.current_state
        
        # Add stress event
        component_health.add_event(
            event_type=HealthEventType.STRESS_SPIKE,
            details={
                "intensity": intensity,
                "affected_parameters": [param.name for param in affected_parameters]
            },
            severity=intensity
        )
        
        # Apply stress to each affected parameter
        for param_type in affected_parameters:
            if param_type not in component_health.parameters:
                continue
                
            parameter = component_health.parameters[param_type]
            
            # Calculate stress impact
            if param_type in [HealthParameterType.STRESS, HealthParameterType.FATIGUE]:
                # These parameters increase with stress
                stress_impact = intensity * 0.5  # Scale the impact
                new_value = min(1.0, parameter.current_value + stress_impact)
            else:
                # Other parameters decrease with stress
                stress_impact = intensity * 0.3  # Scale the impact
                new_value = max(0.0, parameter.current_value - stress_impact)
            
            # Check for threshold crossings
            if (parameter.current_value > parameter.min_threshold and 
                new_value <= parameter.min_threshold):
                # Crossed minimum threshold
                component_health.add_event(
                    event_type=HealthEventType.PARAMETER_THRESHOLD_CROSSED,
                    details={
                        "parameter": param_type.name,
                        "threshold": "minimum",
                        "cause": "stress_event",
                        "previous_value": parameter.current_value,
                        "new_value": new_value
                    },
                    severity=0.6
                )
            
            if (parameter.current_value > parameter.critical_threshold and 
                new_value <= parameter.critical_threshold):
                # Crossed critical threshold
                component_health.add_event(
                    event_type=HealthEventType.PARAMETER_THRESHOLD_CROSSED,
                    details={
                        "parameter": param_type.name,
                        "threshold": "critical",
                        "cause": "stress_event",
                        "previous_value": parameter.current_value,
                        "new_value": new_value
                    },
                    severity=0.9
                )
            
            # Update parameter
            parameter.current_value = new_value
            parameter.last_update_time = datetime.now()
            
            logger.debug(f"Applied stress to {param_type.name} for component {component_health.component_id}: "
                        f"{parameter.current_value:.4f} (intensity: {intensity:.2f})")
        
        # Update overall health state
        self._update_overall_state(component_health)
        
        # Check for state change
        if component_health.current_state != old_state:
            component_health.add_event(
                event_type=HealthEventType.STATE_CHANGE,
                details={
                    "previous_state": old_state.name,
                    "new_state": component_health.current_state.name,
                    "cause": "stress_event"
                },
                severity=self._get_state_change_severity(old_state, component_health.current_state)
            )
            logger.info(f"Component {component_health.component_id} health state changed after stress event: "
                       f"{old_state.name} -> {component_health.current_state.name}")
    
    def get_health_status(self, component_health: ComponentHealth) -> Dict[str, Any]:
        """
        Get a comprehensive health status report for a component.
        
        Args:
            component_health: The component health object to report on
            
        Returns:
            Dictionary containing health status information
        """
        # Ensure state is up to date
        self._update_overall_state(component_health)
        
        # Calculate overall health score
        health_score = self._calculate_health_score(component_health)
        
        # Prepare parameter status
        parameter_status = {}
        for param_type, parameter in component_health.parameters.items():
            # For stress and fatigue, lower is better
            if param_type in [HealthParameterType.STRESS, HealthParameterType.FATIGUE]:
                normalized_value = 1.0 - parameter.current_value
            else:
                normalized_value = parameter.current_value
                
            parameter_status[param_type.name] = {
                "current_value": parameter.current_value,
                "normalized_value": normalized_value,
                "baseline_value": parameter.baseline_value,
                "min_threshold": parameter.min_threshold,
                "critical_threshold": parameter.critical_threshold,
                "status": self._get_parameter_status(parameter)
            }
        
        # Get recent events
        recent_events = component_health.events[-10:] if component_health.events else []
        recent_event_data = [
            {
                "event_id": event.event_id,
                "event_type": event.event_type.name,
                "timestamp": event.timestamp.isoformat(),
                "severity": event.severity,
                "resolved": event.resolved,
                "details": event.details
            }
            for event in recent_events
        ]
        
        # Calculate time since last maintenance
        time_since_maintenance = None
        if component_health.last_maintenance_time:
            time_since_maintenance = (datetime.now() - component_health.last_maintenance_time).total_seconds()
        
        # Prepare status report
        status_report = {
            "component_id": component_health.component_id,
            "health_state": component_health.current_state.name,
            "health_score": health_score,
            "parameters": parameter_status,
            "recent_events": recent_event_data,
            "unresolved_events_count": len(component_health.get_unresolved_events()),
            "creation_time": component_health.creation_time.isoformat(),
            "last_maintenance_time": component_health.last_maintenance_time.isoformat() if component_health.last_maintenance_time else None,
            "time_since_maintenance": time_since_maintenance,
            "maintenance_recommended": self._is_maintenance_recommended(component_health)
        }
        
        return status_report
    
    def _update_overall_state(self, component_health: ComponentHealth) -> None:
        """
        Update the overall health state of a component based on its parameters.
        
        Args:
            component_health: The component health object to update
        """
        # Count parameters in different states
        critical_count = 0
        impaired_count = 0
        degraded_count = 0
        total_count = len(component_health.parameters)
        
        for param_type, parameter in component_health.parameters.items():
            if param_type in [HealthParameterType.STRESS, HealthParameterType.FATIGUE]:
                # For stress and fatigue, higher values are worse
                if parameter.current_value >= parameter.critical_threshold:
                    critical_count += 1
                elif parameter.current_value >= parameter.min_threshold:
                    impaired_count += 1
                elif parameter.current_value >= parameter.baseline_value + 0.2:  # Some elevation
                    degraded_count += 1
            else:
                # For other parameters, lower values are worse
                if parameter.current_value <= parameter.critical_threshold:
                    critical_count += 1
                elif parameter.current_value <= parameter.min_threshold:
                    impaired_count += 1
                elif parameter.current_value <= parameter.baseline_value - 0.2:  # Some degradation
                    degraded_count += 1
        
        # Determine overall state
        if critical_count >= total_count * 0.3 or critical_count >= 2:
            new_state = HealthState.CRITICAL
        elif critical_count > 0 or impaired_count >= total_count * 0.3:
            new_state = HealthState.IMPAIRED
        elif impaired_count > 0 or degraded_count >= total_count * 0.5:
            new_state = HealthState.DEGRADED
        elif degraded_count > 0:
            new_state = HealthState.NORMAL
        else:
            new_state = HealthState.OPTIMAL
        
        # Update state
        component_health.current_state = new_state
    
    def _calculate_health_score(self, component_health: ComponentHealth) -> float:
        """
        Calculate an overall health score for a component (0.0 to 1.0).
        
        Args:
            component_health: The component health object to score
            
        Returns:
            Health score between 0.0 (worst) and 1.0 (best)
        """
        if not component_health.parameters:
            return 1.0
        
        total_score = 0.0
        
        for param_type, parameter in component_health.parameters.items():
            # For stress and fatigue, lower is better
            if param_type in [HealthParameterType.STRESS, HealthParameterType.FATIGUE]:
                param_score = 1.0 - parameter.current_value
            else:
                param_score = parameter.current_value
            
            total_score += param_score
        
        return total_score / len(component_health.parameters)
    
    def _get_parameter_status(self, parameter: HealthParameter) -> str:
        """
        Get a textual status for a health parameter.
        
        Args:
            parameter: The health parameter to evaluate
            
        Returns:
            Status string
        """
        if parameter.param_type in [HealthParameterType.STRESS, HealthParameterType.FATIGUE]:
            # For stress and fatigue, higher values are worse
            if parameter.current_value >= parameter.critical_threshold:
                return "CRITICAL"
            elif parameter.current_value >= parameter.min_threshold:
                return "HIGH"
            elif parameter.current_value >= parameter.baseline_value + 0.2:
                return "ELEVATED"
            else:
                return "NORMAL"
        else:
            # For other parameters, lower values are worse
            if parameter.current_value <= parameter.critical_threshold:
                return "CRITICAL"
            elif parameter.current_value <= parameter.min_threshold:
                return "LOW"
            elif parameter.current_value <= parameter.baseline_value - 0.2:
                return "DEGRADED"
            else:
                return "OPTIMAL"
    
    def _get_state_change_severity(self, old_state: HealthState, new_state: HealthState) -> float:
        """
        Calculate the severity of a state change.
        
        Args:
            old_state: Previous health state
            new_state: New health state
            
        Returns:
            Severity score between 0.0 and 1.0
        """
        # Map states to numeric values (higher is worse)
        state_values = {
            HealthState.OPTIMAL: 0,
            HealthState.NORMAL: 1,
            HealthState.DEGRADED: 2,
            HealthState.IMPAIRED: 3,
            HealthState.CRITICAL: 4,
            HealthState.FAILED: 5
        }
        
        # Calculate difference
        diff = state_values[new_state] - state_values[old_state]
        
        if diff > 0:
            # Worsening condition (more severe)
            return min(1.0, 0.5 + diff * 0.1)
        else:
            # Improving condition (less severe)
            return max(0.1, 0.3 + diff * 0.05)
    
    def _is_maintenance_recommended(self, component_health: ComponentHealth) -> bool:
        """
        Determine if maintenance is recommended for a component.
        
        Args:
            component_health: The component health object to evaluate
            
        Returns:
            True if maintenance is recommended, False otherwise
        """
        # Check overall health state
        if component_health.current_state in [HealthState.IMPAIRED, HealthState.CRITICAL]:
            return True
        
        # Check time since last maintenance
        if component_health.last_maintenance_time:
            time_since_maintenance = (datetime.now() - component_health.last_maintenance_time).total_seconds()
            if time_since_maintenance > 7 * 24 * 60 * 60:  # 7 days
                return True
        
        # Check parameter values
        critical_params = 0
        for param_type, parameter in component_health.parameters.items():
            if param_type in [HealthParameterType.STRESS, HealthParameterType.FATIGUE]:
                if parameter.current_value >= parameter.critical_threshold:
                    critical_params += 1
            else:
                if parameter.current_value <= parameter.critical_threshold:
                    critical_params += 1
        
        return critical_params > 0

    def simulate_adaptation(self, component_health: ComponentHealth, 
                           duration: float, 
                           adaptation_strength: float = 0.5) -> None:
        """
        Simulate biological adaptation to stress over time.
        
        Args:
            component_health: The component health object to update
            duration: Duration of the adaptation period in seconds
            adaptation_strength: Strength of adaptation effect (0.0 to 1.0)
            
        Raises:
            ValueError: If adaptation_strength is not between 0 and 1
        """
        if not 0 <= adaptation_strength <= 1:
            raise ValueError(f"Adaptation strength must be between 0 and 1, got {adaptation_strength}")
        
        if duration <= 0:
            raise ValueError(f"Duration must be positive, got {duration}")
        
        # Check if there's enough stress to adapt to
        stress_param = component_health.parameters.get(HealthParameterType.STRESS)
        if not stress_param or stress_param.current_value < 0.3:
            logger.debug(f"Not enough stress to trigger adaptation for component {component_health.component_id}")
            return
        
        # Calculate adaptation effect
        adaptation_effect = adaptation_strength * 0.2 * (duration / 3600)  # Scale by hours
        
        # Apply adaptation effects
        adaptability_param = component_health.parameters.get(HealthParameterType.ADAPTABILITY)
        if adaptability_param:
            # Higher adaptability means better adaptation
            adaptation_multiplier = 0.5 + adaptability_param.current_value * 0.5
            adaptation_effect *= adaptation_multiplier
        
        # Add adaptation event
        component_health.add_event(
            event_type=HealthEventType.ADAPTATION_OCCURRED,
            details={
                "duration": duration,
                "adaptation_strength": adaptation_strength,
                "adaptation_effect": adaptation_effect
            },
            severity=0.3
        )
        
        # Apply adaptation effects to parameters
        for param_type, parameter in component_health.parameters.items():
            if param_type == HealthParameterType.STRESS:
                # Stress decreases with adaptation
                new_value = max(0.0, parameter.current_value - adaptation_effect)
                parameter.current_value = new_value
            elif param_type == HealthParameterType.RESILIENCE:
                # Resilience increases with adaptation
                new_value = min(parameter.baseline_value + 0.1, parameter.current_value + adaptation_effect * 0.5)
                parameter.current_value = new_value
            elif param_type == HealthParameterType.ADAPTABILITY:
                # Adaptability increases slightly
                new_value = min(parameter.baseline_value + 0.15, parameter.current_value + adaptation_effect * 0.3)
                parameter.current_value = new_value
            
            parameter.last_update_time = datetime.now()
            
            logger.debug(f"Applied adaptation to {param_type.name} for component {component_health.component_id}: "
                        f"{parameter.current_value:.4f}")
        
        # Update overall health state
        self._update_overall_state(component_health)
        
        logger.info(f"Adaptation simulated for component {component_health.component_id} "
                   f"with strength {adaptation_strength:.2f} over {duration:.1f}s")

    def export_health_data(self, component_health: ComponentHealth) -> Dict[str, Any]:
        """
        Export complete health data for a component in a serializable format.
        
        Args:
            component_health: The component health object to export
            
        Returns:
            Dictionary containing complete health data
        """
        # Export parameters
        parameters_data = {}
        for param_type, parameter in component_health.parameters.items():
            parameters_data[param_type.name] = {
                "current_value": parameter.current_value,
                "baseline_value": parameter.baseline_value,
                "min_threshold": parameter.min_threshold,
                "critical_threshold": parameter.critical_threshold,
                "decay_rate": parameter.decay_rate,
                "recovery_rate": parameter.recovery_rate,
                "last_update_time": parameter.last_update_time.isoformat()
            }
        
        # Export events
        events_data = []
        for event in component_health.events:
            events_data.append({
                "event_id": event.event_id,
                "event_type": event.event_type.name,
                "timestamp": event.timestamp.isoformat(),
                "details": event.details,
                "severity": event.severity,
                "resolved": event.resolved,
                "resolution_time": event.resolution_time.isoformat() if event.resolution_time else None
            })
        
        # Prepare export data
        export_data = {
            "component_id": component_health.component_id,
            "current_state": component_health.current_state.name,
            "parameters": parameters_data,
            "events": events_data,
            "creation_time": component_health.creation_time.isoformat(),
            "last_maintenance_time": component_health.last_maintenance_time.isoformat() if component_health.last_maintenance_time else None,
            "export_time": datetime.now().isoformat()
        }
        
        return export_data
    
    def import_health_data(self, data: Dict[str, Any]) -> ComponentHealth:
        """
        Import health data for a component from a previously exported format.
        
        Args:
            data: Dictionary containing health data
            
        Returns:
            Reconstructed ComponentHealth object
            
        Raises:
            ValueError: If data format is invalid
        """
        try:
            component_id = data["component_id"]
            
            # Create component health object
            component_health = ComponentHealth(
                component_id=component_id,
                current_state=HealthState[data["current_state"]],
                creation_time=datetime.fromisoformat(data["creation_time"]),
                last_maintenance_time=datetime.fromisoformat(data["last_maintenance_time"]) if data["last_maintenance_time"] else None
            )
            
            # Import parameters
            for param_name, param_data in data["parameters"].items():
                param_type = HealthParameterType[param_name]
                parameter = HealthParameter(
                    param_type=param_type,
                    current_value=param_data["current_value"],
                    baseline_value=param_data["baseline_value"],
                    min_threshold=param_data["min_threshold"],
                    critical_threshold=param_data["critical_threshold"],
                    decay_rate=param_data["decay_rate"],
                    recovery_rate=param_data["recovery_rate"],
                    last_update_time=datetime.fromisoformat(param_data["last_update_time"])
                )
                component_health.parameters[param_type] = parameter
            
            # Import events
            for event_data in data["events"]:
                event = HealthEvent(
                    event_id=event_data["event_id"],
                    component_id=component_id,
                    event_type=HealthEventType[event_data["event_type"]],
                    timestamp=datetime.fromisoformat(event_data["timestamp"]),
                    details=event_data["details"],
                    severity=event_data["severity"],
                    resolved=event_data["resolved"],
                    resolution_time=datetime.fromisoformat(event_data["resolution_time"]) if event_data["resolution_time"] else None
                )
                component_health.events.append(event)
            
            # Register in manager
            self.components[component_id] = component_health
            
            logger.info(f"Imported health data for component {component_id} with "
                       f"{len(component_health.parameters)} parameters and {len(component_health.events)} events")
            
            return component_health
            
        except (KeyError, ValueError) as e:
            logger.error(f"Error importing health data: {str(e)}")
            raise ValueError(f"Invalid health data format: {str(e)}")