"""
Factory for creating memory system instances.

This module provides a unified way to create memory systems of different types,
abstracting away the implementation details and configuration. It helps prevent
circular dependencies by centralizing the instantiation of memory systems.
"""

from typing import Dict, Optional, Type, Any

from neuroca.core.memory.interfaces import MemorySystem

# Type registry for memory systems
_memory_system_registry: Dict[str, Type[MemorySystem]] = {}

def register_memory_system(name: str, cls: Type[MemorySystem]) -> None:
    """
    Register a memory system implementation.
    
    Args:
        name: The name to register the memory system under
        cls: The memory system class to register
    """
    _memory_system_registry[name] = cls

def create_memory_system(memory_type: str, **config) -> MemorySystem:
    """
    Create a memory system of the specified type.
    
    Args:
        memory_type: The type of memory system to create
        **config: Configuration parameters for the memory system
        
    Returns:
        MemorySystem: An instance of the requested memory system
        
    Raises:
        ValueError: If the requested memory type is not registered
    """
    # Normalize memory type name
    memory_type = memory_type.lower()
    
    # Map common aliases to standard names
    memory_type_map = {
        "working": "working",
        "stm": "working",  # Short-term memory
        "short_term": "working",
        "episodic": "episodic",
        "em": "episodic",
        "semantic": "semantic",
        "sm": "semantic",
        "long_term": "semantic",
        "ltm": "semantic",  # Long-term memory
    }
    
    # Resolve the memory type
    resolved_type = memory_type_map.get(memory_type, memory_type)
    
    # Check if the memory type is registered
    if resolved_type not in _memory_system_registry:
        raise ValueError(f"Unknown memory type: {memory_type}")
    
    # Create and return the memory system
    memory_system_cls = _memory_system_registry[resolved_type]
    return memory_system_cls(**config)

def get_registered_memory_systems() -> Dict[str, Type[MemorySystem]]:
    """
    Get all registered memory systems.
    
    Returns:
        Dict[str, Type[MemorySystem]]: A dictionary mapping memory system names to classes
    """
    return _memory_system_registry.copy()

# Import and register concrete implementations
# These imports are at the bottom to avoid circular dependencies
def _register_default_memory_systems() -> None:
    """Register the default memory system implementations."""
    try:
        # These will be implemented later
        from neuroca.core.memory.working_memory import WorkingMemory
        from neuroca.core.memory.episodic_memory import EpisodicMemory
        from neuroca.core.memory.semantic_memory import SemanticMemory
        
        register_memory_system("working", WorkingMemory)
        register_memory_system("episodic", EpisodicMemory)
        register_memory_system("semantic", SemanticMemory)
    except ImportError:
        # During development, some implementations might not exist yet
        pass

# Register default memory systems
_register_default_memory_systems() 