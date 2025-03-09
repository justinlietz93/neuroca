"""
Working memory functionality for the NCA system.

This module implements the working memory component, which handles temporarily active
information that's currently being processed by the system.
"""

from typing import Dict, List, Any, Optional, Deque
from collections import deque
from datetime import datetime

class WorkingMemory:
    """Class managing the working memory system."""
    
    def __init__(self, capacity: int = 7):
        """
        Initialize the working memory system.
        
        Args:
            capacity: Maximum number of items that can be held in working memory
                    (default is 7, based on human working memory limitations)
        """
        self.capacity = capacity
        self.items: Deque[Dict[str, Any]] = deque(maxlen=capacity)
        self.activated_at: Dict[str, datetime] = {}
        
    def add_item(self, item_id: str, content: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add an item to working memory.
        
        Args:
            item_id: Unique identifier for the item
            content: The content to store
            metadata: Additional metadata about the item
            
        Returns:
            True if item was added, False if rejected
        """
        # Check if item already exists and update it
        for i, existing in enumerate(self.items):
            if existing.get("id") == item_id:
                self.items[i] = {
                    "id": item_id,
                    "content": content,
                    "metadata": metadata or {},
                    "activation_time": datetime.now()
                }
                self.activated_at[item_id] = datetime.now()
                return True
        
        # If working memory is full, we need to decide what to drop
        # This implements a simple recency-based policy
        if len(self.items) >= self.capacity:
            # Item added to the right side, old items removed from the left
            # maxlen property of deque will handle the removal
            pass
            
        # Add the new item
        self.items.append({
            "id": item_id,
            "content": content,
            "metadata": metadata or {},
            "activation_time": datetime.now()
        })
        self.activated_at[item_id] = datetime.now()
        return True
    
    def get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an item from working memory by ID.
        
        Args:
            item_id: The ID of the item to retrieve
            
        Returns:
            The item if found, None otherwise
        """
        for item in self.items:
            if item.get("id") == item_id:
                # Update activation time
                self.activated_at[item_id] = datetime.now()
                return item
        return None
    
    def remove_item(self, item_id: str) -> bool:
        """
        Remove an item from working memory.
        
        Args:
            item_id: The ID of the item to remove
            
        Returns:
            True if removed, False if not found
        """
        for i, item in enumerate(self.items):
            if item.get("id") == item_id:
                self.items.remove(item)
                self.activated_at.pop(item_id, None)
                return True
        return False
    
    def get_all_items(self) -> List[Dict[str, Any]]:
        """
        Get all items currently in working memory.
        
        Returns:
            List of all items
        """
        return list(self.items)
    
    def clear(self) -> None:
        """Clear all items from working memory."""
        self.items.clear()
        self.activated_at.clear()


# Create a singleton instance
working_memory = WorkingMemory()

def add_to_working_memory(item_id: str, content: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """
    Add an item to the working memory system.
    
    Args:
        item_id: Unique identifier for the item
        content: The content to store
        metadata: Additional metadata about the item
        
    Returns:
        True if added successfully, False otherwise
    """
    return working_memory.add_item(item_id, content, metadata)


def get_from_working_memory(item_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve an item from working memory.
    
    Args:
        item_id: The ID of the item to retrieve
        
    Returns:
        The item if found, None otherwise
    """
    return working_memory.get_item(item_id)


def get_all_working_memory() -> List[Dict[str, Any]]:
    """
    Get all items currently in working memory.
    
    Returns:
        List of all items in working memory
    """
    return working_memory.get_all_items()


def clear_working_memory() -> None:
    """Clear all items from working memory."""
    working_memory.clear() 