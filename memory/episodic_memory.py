"""
Episodic memory functionality for the NCA system.

This module handles episodic memories - memories of specific events, times, and places.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime

class EpisodicMemory:
    """Class representing an episodic memory in the system."""
    
    def __init__(self, content: str, timestamp: Optional[datetime] = None, location: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None, importance: float = 0.5):
        """
        Initialize a new episodic memory.
        
        Args:
            content: The main content of the memory
            timestamp: When the memory was created/occurred
            location: Where the memory occurred
            context: Additional contextual information
            importance: How important this memory is (0.0-1.0)
        """
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.location = location
        self.context = context or {}
        self.importance = max(0.0, min(1.0, importance))
        self.retrieval_count = 0
        self.last_accessed = self.timestamp
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the memory to a dictionary representation."""
        return {
            "type": "episodic",
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "location": self.location,
            "context": self.context,
            "importance": self.importance,
            "retrieval_count": self.retrieval_count,
            "last_accessed": self.last_accessed.isoformat()
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EpisodicMemory':
        """Create an episodic memory from a dictionary."""
        memory = cls(
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            location=data.get("location"),
            context=data.get("context", {}),
            importance=data.get("importance", 0.5)
        )
        memory.retrieval_count = data.get("retrieval_count", 0)
        memory.last_accessed = datetime.fromisoformat(data.get("last_accessed", memory.timestamp.isoformat()))
        return memory


def store_episodic_memory(content: str, **kwargs) -> Dict[str, Any]:
    """
    Store a new episodic memory.
    
    Args:
        content: Memory content
        **kwargs: Additional memory attributes
        
    Returns:
        The stored memory as a dictionary
    """
    memory = EpisodicMemory(content, **kwargs)
    # In a real implementation, this would store to a database
    return memory.to_dict()
    

def retrieve_episodic_memories(query: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
    """
    Retrieve episodic memories matching the query.
    
    Args:
        query: Search parameters
        limit: Maximum number of results
        
    Returns:
        List of matching memories
    """
    # This is a stub implementation
    return [
        EpisodicMemory(f"Memory {i}", importance=0.8-i*0.1).to_dict() 
        for i in range(min(5, limit))
    ] 