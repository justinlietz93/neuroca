"""
Semantic memory functionality for the NCA system.

This module handles semantic memories - factual knowledge not tied to specific events.
"""

from typing import Dict, List, Any, Optional, Set
from datetime import datetime

class SemanticMemory:
    """Class representing a semantic memory in the system."""
    
    def __init__(self, concept: str, facts: Optional[Dict[str, Any]] = None, 
                 related_concepts: Optional[Set[str]] = None, 
                 confidence: float = 1.0):
        """
        Initialize a new semantic memory.
        
        Args:
            concept: The main concept this memory represents
            facts: Dictionary of facts about this concept
            related_concepts: Set of related concept names
            confidence: Confidence level in this knowledge (0.0-1.0)
        """
        self.concept = concept
        self.facts = facts or {}
        self.related_concepts = related_concepts or set()
        self.confidence = max(0.0, min(1.0, confidence))
        self.created_at = datetime.now()
        self.last_updated = self.created_at
        self.access_count = 0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the memory to a dictionary representation."""
        return {
            "type": "semantic",
            "concept": self.concept,
            "facts": self.facts,
            "related_concepts": list(self.related_concepts),
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "access_count": self.access_count
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SemanticMemory':
        """Create a semantic memory from a dictionary."""
        memory = cls(
            concept=data["concept"],
            facts=data.get("facts", {}),
            related_concepts=set(data.get("related_concepts", [])),
            confidence=data.get("confidence", 1.0)
        )
        memory.created_at = datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        memory.last_updated = datetime.fromisoformat(data.get("last_updated", memory.created_at.isoformat()))
        memory.access_count = data.get("access_count", 0)
        return memory


def store_semantic_memory(concept: str, facts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Store a new semantic memory or update an existing one.
    
    Args:
        concept: The concept name
        facts: Facts about the concept
        
    Returns:
        The stored memory as a dictionary
    """
    memory = SemanticMemory(concept, facts)
    # In a real implementation, this would store to a database
    return memory.to_dict()
    

def retrieve_semantic_memory(concept: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a semantic memory by concept name.
    
    Args:
        concept: The concept to retrieve
        
    Returns:
        The memory data if found, None otherwise
    """
    # This is a stub implementation
    if concept:
        memory = SemanticMemory(
            concept=concept,
            facts={"sample": "This is sample data for " + concept},
            related_concepts={"related_" + concept}
        )
        return memory.to_dict()
    return None


def query_semantic_network(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Query the semantic network for concepts matching the query.
    
    Args:
        query: Search query
        limit: Maximum number of results
        
    Returns:
        List of matching semantic memories
    """
    # This is a stub implementation
    return [
        SemanticMemory(f"Concept_{i}", {"key": f"Value for {query} {i}"}).to_dict()
        for i in range(min(5, limit))
    ] 