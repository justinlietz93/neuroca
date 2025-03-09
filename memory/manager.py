"""
Memory Manager for the NeuroCognitive Architecture (NCA).

This module provides a comprehensive memory management system that orchestrates
interactions between different memory tiers (working, episodic, and semantic memory).
It handles memory operations including storage, retrieval, consolidation, and forgetting
according to biological-inspired cognitive principles.

The MemoryManager serves as the central coordinator for all memory operations in the NCA,
providing a unified interface for other components to interact with the memory system
while abstracting the complexity of the underlying memory implementations.

Usage:
    manager = MemoryManager()
    
    # Store information in memory
    manager.store(content="New information", memory_type=MemoryType.WORKING)
    
    # Retrieve information from memory
    results = manager.retrieve(query="information", memory_type=MemoryType.EPISODIC)
    
    # Consolidate memories from working to long-term storage
    manager.consolidate()
    
    # Perform memory maintenance
    manager.maintain()
"""

import datetime
import json
import logging
import os
import time
import uuid
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Configure logging
logger = logging.getLogger(__name__)

class MemoryType(Enum):
    """Enum representing different types of memory in the NCA system."""
    WORKING = auto()    # Short-term, limited capacity memory
    EPISODIC = auto()   # Event-based, autobiographical memory
    SEMANTIC = auto()   # Factual, conceptual knowledge memory


class MemoryPriority(Enum):
    """Priority levels for memory items affecting retention and retrieval."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class MemoryItem:
    """
    Represents a single item stored in memory with metadata.
    
    Attributes:
        id (str): Unique identifier for the memory item
        content (Any): The actual content/data of the memory
        created_at (datetime): When the memory was first created
        last_accessed (datetime): When the memory was last retrieved/modified
        access_count (int): Number of times this memory has been accessed
        priority (MemoryPriority): Importance level of this memory
        tags (Set[str]): Labels/categories associated with this memory
        metadata (Dict): Additional contextual information about the memory
        decay_factor (float): Rate at which this memory decays (forgetting curve)
        associations (Dict[str, float]): Related memory IDs and their strength
    """
    
    def __init__(
        self,
        content: Any,
        priority: MemoryPriority = MemoryPriority.MEDIUM,
        tags: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        decay_factor: float = 0.05
    ):
        """
        Initialize a new memory item.
        
        Args:
            content: The data to be stored in memory
            priority: Importance level affecting retention
            tags: Set of labels for categorization and retrieval
            metadata: Additional contextual information
            decay_factor: Rate at which memory decays (0.0-1.0)
        """
        self.id = str(uuid.uuid4())
        self.content = content
        self.created_at = datetime.datetime.now()
        self.last_accessed = self.created_at
        self.access_count = 0
        self.priority = priority
        self.tags = tags or set()
        self.metadata = metadata or {}
        self.decay_factor = max(0.0, min(1.0, decay_factor))  # Clamp between 0 and 1
        self.associations = {}  # {memory_id: strength}
        
        logger.debug(f"Created new memory item with ID {self.id}")
    
    def access(self) -> None:
        """
        Record an access to this memory item, updating metadata.
        This reinforces the memory according to spaced repetition principles.
        """
        self.last_accessed = datetime.datetime.now()
        self.access_count += 1
        logger.debug(f"Accessed memory {self.id}, new access count: {self.access_count}")
    
    def calculate_activation(self) -> float:
        """
        Calculate the current activation level of this memory based on
        recency, frequency, and priority.
        
        Returns:
            float: Activation level between 0.0 and 1.0
        """
        # Time-based decay (recency)
        time_since_access = (datetime.datetime.now() - self.last_accessed).total_seconds()
        recency_factor = 1.0 / (1.0 + self.decay_factor * time_since_access)
        
        # Frequency factor
        frequency_factor = min(1.0, 0.1 + (self.access_count / 10.0))
        
        # Priority factor
        priority_factor = self.priority.value / max(e.value for e in MemoryPriority)
        
        # Combined activation (weighted average)
        activation = (0.5 * recency_factor) + (0.3 * frequency_factor) + (0.2 * priority_factor)
        return max(0.0, min(1.0, activation))  # Ensure between 0 and 1
    
    def add_association(self, memory_id: str, strength: float = 0.5) -> None:
        """
        Create or strengthen an association with another memory item.
        
        Args:
            memory_id: ID of the memory to associate with
            strength: Association strength (0.0-1.0)
        """
        clamped_strength = max(0.0, min(1.0, strength))
        self.associations[memory_id] = clamped_strength
        logger.debug(f"Added association from {self.id} to {memory_id} with strength {clamped_strength}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert memory item to a dictionary for serialization.
        
        Returns:
            Dict containing all memory item attributes
        """
        return {
            "id": self.id,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "priority": self.priority.name,
            "tags": list(self.tags),
            "metadata": self.metadata,
            "decay_factor": self.decay_factor,
            "associations": self.associations
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        """
        Create a memory item from a dictionary representation.
        
        Args:
            data: Dictionary containing memory item attributes
            
        Returns:
            Reconstructed MemoryItem object
        """
        item = cls(
            content=data["content"],
            priority=MemoryPriority[data["priority"]],
            tags=set(data["tags"]),
            metadata=data["metadata"],
            decay_factor=data["decay_factor"]
        )
        
        # Restore the original ID and other attributes
        item.id = data["id"]
        item.created_at = datetime.datetime.fromisoformat(data["created_at"])
        item.last_accessed = datetime.datetime.fromisoformat(data["last_accessed"])
        item.access_count = data["access_count"]
        item.associations = data["associations"]
        
        return item


class MemoryStore:
    """
    Base class for memory storage implementations.
    
    This abstract class defines the interface that all memory stores must implement,
    regardless of the specific memory type (working, episodic, semantic).
    """
    
    def __init__(self, capacity: Optional[int] = None):
        """
        Initialize a memory store.
        
        Args:
            capacity: Maximum number of items this store can hold (None for unlimited)
        """
        self.capacity = capacity
        self.items: Dict[str, MemoryItem] = {}
        logger.info(f"Initialized memory store with capacity {capacity}")
    
    def add(self, item: MemoryItem) -> str:
        """
        Add a memory item to the store.
        
        Args:
            item: The memory item to store
            
        Returns:
            str: ID of the stored memory item
            
        Raises:
            MemoryCapacityError: If the store is at capacity
        """
        if self.capacity is not None and len(self.items) >= self.capacity:
            # If at capacity, we need to make room
            self._make_room()
            
            # Double-check we have room now
            if len(self.items) >= self.capacity:
                raise MemoryCapacityError(f"Memory store at capacity ({self.capacity})")
        
        self.items[item.id] = item
        logger.debug(f"Added item {item.id} to memory store")
        return item.id
    
    def get(self, item_id: str) -> Optional[MemoryItem]:
        """
        Retrieve a memory item by ID.
        
        Args:
            item_id: ID of the memory item to retrieve
            
        Returns:
            The memory item if found, None otherwise
        """
        item = self.items.get(item_id)
        if item:
            item.access()
            logger.debug(f"Retrieved item {item_id} from memory store")
        else:
            logger.debug(f"Item {item_id} not found in memory store")
        return item
    
    def remove(self, item_id: str) -> bool:
        """
        Remove a memory item from the store.
        
        Args:
            item_id: ID of the memory item to remove
            
        Returns:
            bool: True if item was removed, False if not found
        """
        if item_id in self.items:
            del self.items[item_id]
            logger.debug(f"Removed item {item_id} from memory store")
            return True
        logger.debug(f"Could not remove item {item_id} (not found)")
        return False
    
    def search(
        self, 
        query: Optional[str] = None, 
        tags: Optional[Set[str]] = None,
        min_activation: float = 0.0,
        limit: Optional[int] = None
    ) -> List[MemoryItem]:
        """
        Search for memory items matching criteria.
        
        Args:
            query: Text to search for in content
            tags: Set of tags to match
            min_activation: Minimum activation level
            limit: Maximum number of results to return
            
        Returns:
            List of matching memory items
        """
        results = []
        
        for item in self.items.values():
            # Check activation threshold
            if item.calculate_activation() < min_activation:
                continue
                
            # Check tags if specified
            if tags and not tags.issubset(item.tags):
                continue
                
            # Check content query if specified
            if query and not self._content_matches(item.content, query):
                continue
                
            # If we got here, the item matches all criteria
            results.append(item)
            item.access()  # Record this access
            
            # Check if we've reached the limit
            if limit and len(results) >= limit:
                break
                
        logger.debug(f"Search returned {len(results)} results")
        return results
    
    def _content_matches(self, content: Any, query: str) -> bool:
        """
        Check if content matches a search query.
        
        Args:
            content: The content to check
            query: The search query
            
        Returns:
            bool: True if content matches query
        """
        # Handle different content types
        if isinstance(content, str):
            return query.lower() in content.lower()
        elif isinstance(content, dict):
            # Search through dictionary values
            return any(self._content_matches(v, query) for v in content.values())
        elif isinstance(content, list):
            # Search through list items
            return any(self._content_matches(item, query) for item in content)
        else:
            # For other types, convert to string and search
            try:
                return query.lower() in str(content).lower()
            except:
                return False
    
    def _make_room(self) -> None:
        """
        Make room in the store by removing low-activation items.
        This is called when the store is at capacity.
        """
        if not self.items or self.capacity is None:
            return
            
        # Calculate activation for all items
        items_with_activation = [
            (item, item.calculate_activation()) 
            for item in self.items.values()
        ]
        
        # Sort by activation (lowest first)
        items_with_activation.sort(key=lambda x: x[1])
        
        # Remove the lowest activation item
        lowest_item = items_with_activation[0][0]
        self.remove(lowest_item.id)
        logger.info(f"Removed item {lowest_item.id} to make room (activation: {items_with_activation[0][1]:.2f})")
    
    def clear(self) -> None:
        """Clear all items from the store."""
        count = len(self.items)
        self.items.clear()
        logger.info(f"Cleared memory store ({count} items removed)")
    
    def get_all_ids(self) -> List[str]:
        """
        Get all memory item IDs in the store.
        
        Returns:
            List of memory item IDs
        """
        return list(self.items.keys())
    
    def count(self) -> int:
        """
        Get the number of items in the store.
        
        Returns:
            int: Number of items
        """
        return len(self.items)


class WorkingMemoryStore(MemoryStore):
    """
    Implementation of working memory with limited capacity and rapid decay.
    
    Working memory is characterized by:
    - Small capacity (typically 5-9 items)
    - Rapid decay of items
    - High accessibility for recent items
    """
    
    def __init__(self, capacity: int = 7):
        """
        Initialize working memory store.
        
        Args:
            capacity: Maximum number of items (default: 7, based on Miller's Law)
        """
        super().__init__(capacity=capacity)
        self.default_decay_factor = 0.2  # Higher decay for working memory
        logger.info(f"Initialized working memory store with capacity {capacity}")
    
    def add(self, item: MemoryItem) -> str:
        """
        Add an item to working memory, with higher default decay.
        
        Args:
            item: Memory item to add
            
        Returns:
            str: ID of the added item
        """
        # Apply working memory's higher decay factor if not explicitly set
        if item.decay_factor == 0.05:  # Check if it's the default from MemoryItem
            item.decay_factor = self.default_decay_factor
            
        return super().add(item)


class EpisodicMemoryStore(MemoryStore):
    """
    Implementation of episodic memory for event-based, autobiographical memories.
    
    Episodic memory is characterized by:
    - Temporal organization (events in sequence)
    - Context-rich storage
    - Moderate decay rate
    """
    
    def __init__(self, capacity: Optional[int] = 1000):
        """
        Initialize episodic memory store.
        
        Args:
            capacity: Maximum number of items (default: 1000)
        """
        super().__init__(capacity=capacity)
        self.default_decay_factor = 0.05  # Moderate decay for episodic memory
        logger.info(f"Initialized episodic memory store with capacity {capacity}")
    
    def add_episode(
        self, 
        content: Any, 
        timestamp: Optional[datetime.datetime] = None,
        location: Optional[str] = None,
        actors: Optional[List[str]] = None,
        tags: Optional[Set[str]] = None,
        priority: MemoryPriority = MemoryPriority.MEDIUM
    ) -> str:
        """
        Add an episodic memory with temporal and contextual information.
        
        Args:
            content: The content of the episode
            timestamp: When the episode occurred
            location: Where the episode occurred
            actors: Who was involved in the episode
            tags: Tags for categorization
            priority: Importance of the episode
            
        Returns:
            str: ID of the added episode
        """
        # Create metadata specific to episodic memory
        metadata = {
            "timestamp": timestamp.isoformat() if timestamp else datetime.datetime.now().isoformat(),
            "location": location,
            "actors": actors or [],
            "type": "episode"
        }
        
        # Create and add the memory item
        item = MemoryItem(
            content=content,
            priority=priority,
            tags=tags or set(),
            metadata=metadata,
            decay_factor=self.default_decay_factor
        )
        
        return self.add(item)
    
    def get_episodes_by_timeframe(
        self, 
        start_time: datetime.datetime, 
        end_time: datetime.datetime
    ) -> List[MemoryItem]:
        """
        Retrieve episodes that occurred within a specific timeframe.
        
        Args:
            start_time: Start of the timeframe
            end_time: End of the timeframe
            
        Returns:
            List of episodes within the timeframe
        """
        results = []
        
        for item in self.items.values():
            if "timestamp" not in item.metadata:
                continue
                
            try:
                timestamp = datetime.datetime.fromisoformat(item.metadata["timestamp"])
                if start_time <= timestamp <= end_time:
                    results.append(item)
                    item.access()  # Record this access
            except (ValueError, TypeError):
                logger.warning(f"Invalid timestamp format in episode {item.id}")
                
        logger.debug(f"Retrieved {len(results)} episodes in timeframe")
        return results


class SemanticMemoryStore(MemoryStore):
    """
    Implementation of semantic memory for factual, conceptual knowledge.
    
    Semantic memory is characterized by:
    - Organized by meaning rather than time
    - Interconnected concepts and facts
    - Slow decay rate
    """
    
    def __init__(self, capacity: Optional[int] = None):
        """
        Initialize semantic memory store.
        
        Args:
            capacity: Maximum number of items (default: unlimited)
        """
        super().__init__(capacity=capacity)
        self.default_decay_factor = 0.01  # Slow decay for semantic memory
        self.concept_network = {}  # Graph of related concepts
        logger.info(f"Initialized semantic memory store with capacity {capacity}")
    
    def add_fact(
        self, 
        fact: Any,
        concepts: List[str],
        confidence: float = 1.0,
        source: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        priority: MemoryPriority = MemoryPriority.MEDIUM
    ) -> str:
        """
        Add a factual memory with conceptual relationships.
        
        Args:
            fact: The factual content
            concepts: List of concepts this fact relates to
            confidence: Confidence level in this fact (0.0-1.0)
            source: Source of this information
            tags: Tags for categorization
            priority: Importance of the fact
            
        Returns:
            str: ID of the added fact
        """
        # Create metadata specific to semantic memory
        metadata = {
            "type": "fact",
            "concepts": concepts,
            "confidence": max(0.0, min(1.0, confidence)),
            "source": source
        }
        
        # Create and add the memory item
        item = MemoryItem(
            content=fact,
            priority=priority,
            tags=tags or set(),
            metadata=metadata,
            decay_factor=self.default_decay_factor
        )
        
        item_id = self.add(item)
        
        # Update concept network
        for concept in concepts:
            if concept not in self.concept_network:
                self.concept_network[concept] = set()
            self.concept_network[concept].add(item_id)
            
        return item_id
    
    def get_facts_by_concept(self, concept: str) -> List[MemoryItem]:
        """
        Retrieve facts related to a specific concept.
        
        Args:
            concept: The concept to retrieve facts for
            
        Returns:
            List of facts related to the concept
        """
        if concept not in self.concept_network:
            return []
            
        results = []
        for item_id in self.concept_network[concept]:
            item = self.get(item_id)
            if item:
                results.append(item)
                
        logger.debug(f"Retrieved {len(results)} facts for concept '{concept}'")
        return results
    
    def get_related_concepts(self, concept: str, max_distance: int = 2) -> Dict[str, int]:
        """
        Find concepts related to the given concept within a certain distance.
        
        Args:
            concept: The starting concept
            max_distance: Maximum distance in the concept graph
            
        Returns:
            Dictionary of related concepts and their distances
        """
        if concept not in self.concept_network:
            return {}
            
        # Breadth-first search through concept network
        visited = {concept: 0}
        queue = [(concept, 0)]
        
        while queue:
            current, distance = queue.pop(0)
            
            if distance >= max_distance:
                continue
                
            # Find all facts related to this concept
            for item_id in self.concept_network.get(current, set()):
                item = self.get(item_id)
                if not item:
                    continue
                    
                # Get concepts from this fact
                for related in item.metadata.get("concepts", []):
                    if related not in visited:
                        visited[related] = distance + 1
                        queue.append((related, distance + 1))
        
        # Remove the original concept
        del visited[concept]
        
        logger.debug(f"Found {len(visited)} concepts related to '{concept}'")
        return visited


class MemoryCapacityError(Exception):
    """Exception raised when a memory store is at capacity."""
    pass


class MemoryManager:
    """
    Central coordinator for the NCA memory system.
    
    The MemoryManager orchestrates interactions between different memory tiers,
    handles memory operations, and implements cognitive processes like
    consolidation, association, and forgetting.
    """
    
    def __init__(
        self,
        working_capacity: int = 7,
        episodic_capacity: int = 1000,
        semantic_capacity: Optional[int] = None,
        persistence_dir: Optional[str] = None
    ):
        """
        Initialize the memory manager with its component stores.
        
        Args:
            working_capacity: Capacity of working memory
            episodic_capacity: Capacity of episodic memory
            semantic_capacity: Capacity of semantic memory (None for unlimited)
            persistence_dir: Directory for persisting memory to disk
        """
        # Initialize memory stores
        self.working_memory = WorkingMemoryStore(capacity=working_capacity)
        self.episodic_memory = EpisodicMemoryStore(capacity=episodic_capacity)
        self.semantic_memory = SemanticMemoryStore(capacity=semantic_capacity)
        
        # Set up persistence
        self.persistence_dir = persistence_dir
        if persistence_dir:
            os.makedirs(persistence_dir, exist_ok=True)
            logger.info(f"Memory persistence enabled at {persistence_dir}")
        
        # Memory operation tracking
        self.last_consolidation = datetime.datetime.now()
        self.last_maintenance = datetime.datetime.now()
        
        logger.info("Memory manager initialized")
    
    def store(
        self, 
        content: Any, 
        memory_type: MemoryType,
        priority: MemoryPriority = MemoryPriority.MEDIUM,
        tags: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store content in the specified memory type.
        
        Args:
            content: The content to store
            memory_type: Which memory store to use
            priority: Importance of this memory
            tags: Tags for categorization
            metadata: Additional contextual information
            
        Returns:
            str: ID of the stored memory item
            
        Raises:
            MemoryCapacityError: If the target memory store is at capacity
            ValueError: If an invalid memory type is specified
        """
        # Create the memory item
        item = MemoryItem(
            content=content,
            priority=priority,
            tags=tags or set(),
            metadata=metadata or {}
        )
        
        # Store in the appropriate memory store
        if memory_type == MemoryType.WORKING:
            item_id = self.working_memory.add(item)
            logger.info(f"Stored item {item_id} in working memory")
        elif memory_type == MemoryType.EPISODIC:
            item_id = self.episodic_memory.add(item)
            logger.info(f"Stored item {item_id} in episodic memory")
        elif memory_type == MemoryType.SEMANTIC:
            item_id = self.semantic_memory.add(item)
            logger.info(f"Stored item {item_id} in semantic memory")
        else:
            raise ValueError(f"Invalid memory type: {memory_type}")
            
        return item_id
    
    def retrieve(
        self, 
        query: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        item_id: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        min_activation: float = 0.0,
        limit: Optional[int] = None
    ) -> List[MemoryItem]:
        """
        Retrieve memory items matching the specified criteria.
        
        Args:
            query: Text to search for in content
            memory_type: Which memory store to search (None for all)
            item_id: Specific item ID to retrieve
            tags: Tags to match
            min_activation: Minimum activation level
            limit: Maximum number of results
            
        Returns:
            List of matching memory items
        """
        results = []
        
        # If item_id is specified, try to retrieve that specific item
        if item_id:
            for store in self._get_stores(memory_type):
                item = store.get(item_id)
                if item:
                    results.append(item)
                    break
            return results
        
        # Otherwise, search based on criteria
        for store in self._get_stores(memory_type):
            store_results = store.search(
                query=query,
                tags=tags,
                min_activation=min_activation,
                limit=limit
            )
            results.extend(store_results)
            
            # Check if we've reached the overall limit
            if limit and len(results) >= limit:
                results = results[:limit]
                break
                
        logger.info(f"Retrieved {len(results)} items matching query criteria")
        return results
    
    def _get_stores(self, memory_type: Optional[MemoryType] = None) -> List[MemoryStore]:
        """
        Get the memory stores to operate on based on the specified type.
        
        Args:
            memory_type: Which memory store to use (None for all)
            
        Returns:
            List of memory stores
        """
        if memory_type == MemoryType.WORKING:
            return [self.working_memory]
        elif memory_type == MemoryType.EPISODIC:
            return [self.episodic_memory]
        elif memory_type == MemoryType.SEMANTIC:
            return [self.semantic_memory]
        else:
            # If no specific type, search all stores (working first for recency)
            return [self.working_memory, self.episodic_memory, self.semantic_memory]
    
    def consolidate(self, force: bool = False) -> int:
        """
        Consolidate memories from working memory to long-term stores.
        
        This process mimics how the brain transfers short-term memories to
        long-term storage, especially during periods of rest.
        
        Args:
            force: Whether to force consolidation regardless of timing
            
        Returns:
            int: Number of items consolidated
        """
        # Check if enough time has passed since last consolidation
        time_since_last = (datetime.datetime.now() - self.last_consolidation).total_seconds()
        if not force and time_since_last < 300:  # 5 minutes
            logger.debug(f"Skipping consolidation, only {time_since_last:.1f}s since last run")
            return 0
            
        logger.info("Beginning memory consolidation process")
        consolidated_count = 0
        
        # Get items from working memory above activation threshold
        items_to_consolidate = self.working_memory.search(min_activation=0.3)
        
        for item in items_to_consolidate:
            # Determine which long-term store to use based on content/metadata
            target_store = self._determine_consolidation_target(item)
            
            if target_store == self.episodic_memory:
                # Ensure it has episodic metadata
                if "type" not in item.metadata:
                    item.metadata["type"] = "episode"
                if "timestamp" not in item.metadata:
                    item.metadata["timestamp"] = item.created_at.isoformat()
                    
                # Add to episodic memory
                self.episodic_memory.add(item)
                logger.debug(f"Consolidated item {item.id} to episodic memory")
                consolidated_count += 1
                
            elif target_store == self.semantic_memory:
                # Ensure it has semantic metadata
                if "type" not in item.metadata:
                    item.metadata["type"] = "fact"
                if "concepts" not in item.metadata:
                    # Extract concepts from tags
                    item.metadata["concepts"] = list(item.tags)
                    
                # Add to semantic memory
                self.semantic_memory.add(item)
                logger.debug(f"Consolidated item {item.id} to semantic memory")
                consolidated_count += 1
        
        # Update consolidation timestamp
        self.last_consolidation = datetime.datetime.now()
        logger.info(f"Consolidation complete: {consolidated_count} items transferred to long-term memory")
        
        return consolidated_count
    
    def _determine_consolidation_target(self, item: MemoryItem) -> MemoryStore:
        """
        Determine which long-term memory store an item should be consolidated to.
        
        Args:
            item: The memory item to evaluate
            
        Returns:
            The target memory store (episodic or semantic)
        """
        # Check if item metadata specifies a type
        if "type" in item.metadata:
            if item.metadata["type"] == "episode":
                return self.episodic_memory
            elif item.metadata["type"] == "fact":
                return self.semantic_memory
        
        # Check for temporal markers suggesting an episode
        temporal_indicators = {"timestamp", "time", "when", "date", "day"}
        if any(key in item.metadata for key in temporal_indicators):
            return self.episodic_memory
            
        # Check for conceptual markers suggesting semantic memory
        conceptual_indicators = {"concept", "fact", "definition", "meaning"}
        if any(key in item.metadata for key in conceptual_indicators) or "concepts" in item.metadata:
            return self.semantic_memory
            
        # Default: use episodic for more specific/detailed content, semantic for general knowledge
        if isinstance(item.content, str) and len(item.content) > 200:
            # Longer content tends to be episodic
            return self.episodic_memory
        else:
            # Shorter, more factual content tends to be semantic
            return self.semantic_memory
    
    def associate(self, item_id1: str, item_id2: str, strength: float = 0.5) -> bool:
        """
        Create a bidirectional association between two memory items.
        
        Args:
            item_id1: ID of the first memory item
            item_id2: ID of the second memory item
            strength: Association strength (0.0-1.0)
            
        Returns:
            bool: True if association was created, False otherwise
        """
        # Find the items in all memory stores
        item1 = None
        item2 = None
        
        for store in self._get_stores():
            if not item1:
                item1 = store.get(item_id1)
            if not item2:
                item2 = store.get(item_id2)
            if item1 and item2:
                break
                
        if not item1 or not item2:
            logger.warning(f"Could not create association: items not found")
            return False
            
        # Create bidirectional association
        item1.add_association(item_id2, strength)
        item2.add_association(item_id1, strength)
        
        logger.info(f"Created association between items {item_id1} and {item_id2} with strength {strength}")
        return True
    
    def maintain(self, force: bool = False) -> None:
        """
        Perform memory maintenance tasks like forgetting and optimization.
        
        Args:
            force: Whether to force maintenance regardless of timing
        """
        # Check if enough time has passed since last maintenance
        time_since_last = (datetime.datetime.now() - self.last_maintenance).total_seconds()
        if not force and time_since_last < 3600:  # 1 hour
            logger.debug(f"Skipping maintenance, only {time_since_last:.1f}s since last run")
            return
            
        logger.info("Beginning memory maintenance process")
        
        # Perform forgetting in each memory store
        self._forget_inactive_memories(self.working_memory, threshold=0.2)
        self._forget_inactive_memories(self.episodic_memory, threshold=0.1)
        self._forget_inactive_memories(self.semantic_memory, threshold=0.05)
        
        # Persist memory to disk if enabled
        if self.persistence_dir:
            self.save()
            
        # Update maintenance timestamp
        self.last_maintenance = datetime.datetime.now()
        logger.info("Memory maintenance complete")
    
    def _forget_inactive_memories(self, store: MemoryStore, threshold: float) -> int:
        """
        Remove memories with activation below the threshold.
        
        Args:
            store: The memory store to maintain
            threshold: Activation threshold for forgetting
            
        Returns:
            int: Number of items forgotten
        """
        forgotten_count = 0
        items_to_forget = []
        
        # Identify items below the activation threshold
        for item_id, item in store.items.items():
            # Skip high-priority items
            if item.priority == MemoryPriority.CRITICAL:
                continue
                
            # Check activation level
            if item.calculate_activation() < threshold:
                items_to_forget.append(item_id)
        
        # Remove the identified items
        for item_id in items_to_forget:
            store.remove(item_id)
            forgotten_count += 1
            
        logger.debug(f"Forgot {forgotten_count} items from memory store")
        return forgotten_count
    
    def save(self) -> None:
        """
        Save all memory stores to disk for persistence.
        
        Raises:
            IOError: If there's an error writing to disk
        """
        if not self.persistence_dir:
            logger.warning("Cannot save memory: persistence directory not set")
            return
            
        try:
            # Save working memory
            working_path = os.path.join(self.persistence_dir, "working_memory.json")
            self._save_store(self.working_memory, working_path)
            
            # Save episodic memory
            episodic_path = os.path.join(self.persistence_dir, "episodic_memory.json")
            self._save_store(self.episodic_memory, episodic_path)
            
            # Save semantic memory
            semantic_path = os.path.join(self.persistence_dir, "semantic_memory.json")
            self._save_store(self.semantic_memory, semantic_path)
            
            # Save concept network
            concept_path = os.path.join(self.persistence_dir, "concept_network.json")
            with open(concept_path, 'w') as f:
                # Convert sets to lists for JSON serialization
                serializable_network = {k: list(v) for k, v in self.semantic_memory.concept_network.items()}
                json.dump(serializable_network, f, indent=2)
                
            logger.info(f"Memory successfully saved to {self.persistence_dir}")
            
        except Exception as e:
            logger.error(f"Error saving memory: {str(e)}")
            raise IOError(f"Failed to save memory: {str(e)}")
    
    def _save_store(self, store: MemoryStore, filepath: str) -> None:
        """
        Save a memory store to a JSON file.
        
        Args:
            store: The memory store to save
            filepath: Path to save the file
        """
        items_dict = {}
        for item_id, item in store.items.items():
            items_dict[item_id] = item.to_dict()
            
        with open(filepath, 'w') as f:
            json.dump(items_dict, f, indent=2)
    
    def load(self) -> bool:
        """
        Load memory stores from disk.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.persistence_dir:
            logger.warning("Cannot load memory: persistence directory not set")
            return False
            
        try:
            # Load working memory
            working_path = os.path.join(self.persistence_dir, "working_memory.json")
            if os.path.exists(working_path):
                self._load_store(self.working_memory, working_path)
            
            # Load episodic memory
            episodic_path = os.path.join(self.persistence_dir, "episodic_memory.json")
            if os.path.exists(episodic_path):
                self._load_store(self.episodic_memory, episodic_path)
            
            # Load semantic memory
            semantic_path = os.path.join(self.persistence_dir, "semantic_memory.json")
            if os.path.exists(semantic_path):
                self._load_store(self.semantic_memory, semantic_path)
            
            # Load concept network
            concept_path = os.path.join(self.persistence_dir, "concept_network.json")
            if os.path.exists(concept_path):
                with open(concept_path, 'r') as f:
                    network_data = json.load(f)
                    # Convert lists back to sets
                    self.semantic_memory.concept_network = {k: set(v) for k, v in network_data.items()}
                    
            logger.info(f"Memory successfully loaded from {self.persistence_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading memory: {str(e)}")
            return False
    
    def _load_store(self, store: MemoryStore, filepath: str) -> None:
        """
        Load a memory store from a JSON file.
        
        Args:
            store: The memory store to load into
            filepath: Path to the JSON file
        """
        with open(filepath, 'r') as f:
            items_dict = json.load(f)
            
        # Clear existing items
        store.items.clear()
        
        # Load items from file
        for item_id, item_data in items_dict.items():
            store.items[item_id] = MemoryItem.from_dict(item_data)
    
    def clear(self, memory_type: Optional[MemoryType] = None) -> None:
        """
        Clear memory stores.
        
        Args:
            memory_type: Which memory store to clear (None for all)
        """
        if memory_type == MemoryType.WORKING or memory_type is None:
            self.working_memory.clear()
            logger.info("Working memory cleared")
            
        if memory_type == MemoryType.EPISODIC or memory_type is None:
            self.episodic_memory.clear()
            logger.info("Episodic memory cleared")
            
        if memory_type == MemoryType.SEMANTIC or memory_type is None:
            self.semantic_memory.clear()
            self.semantic_memory.concept_network.clear()
            logger.info("Semantic memory cleared")
            
        logger.info("Memory cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory system.
        
        Returns:
            Dictionary of memory statistics
        """
        stats = {
            "working_memory": {
                "count": self.working_memory.count(),
                "capacity": self.working_memory.capacity,
                "utilization": self.working_memory.count() / self.working_memory.capacity if self.working_memory.capacity else 0
            },
            "episodic_memory": {
                "count": self.episodic_memory.count(),
                "capacity": self.episodic_memory.capacity,
                "utilization": self.episodic_memory.count() / self.episodic_memory.capacity if self.episodic_memory.capacity else 0
            },
            "semantic_memory": {
                "count": self.semantic_memory.count(),
                "capacity": self.semantic_memory.capacity,
                "utilization": self.semantic_memory.count() / self.semantic_memory.capacity if self.semantic_memory.capacity else 0,
                "concept_count": len(self.semantic_memory.concept_network)
            },
            "last_consolidation": self.last_consolidation.isoformat(),
            "last_maintenance": self.last_maintenance.isoformat(),
            "time_since_consolidation": (datetime.datetime.now() - self.last_consolidation).total_seconds(),
            "time_since_maintenance": (datetime.datetime.now() - self.last_maintenance).total_seconds()
        }
        
        return stats