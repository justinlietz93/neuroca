"""
Memory Command Line Interface Module.

This module provides CLI commands for interacting with the NeuroCognitive Architecture's
memory system. It allows users to manage and interact with the three-tiered memory system,
including working memory, episodic memory, and semantic memory.

Usage:
    neuroca memory list [--tier=<tier>] [--format=<format>]
    neuroca memory get <memory_id> [--format=<format>]
    neuroca memory store <content> [--tier=<tier>] [--tags=<tags>]
    neuroca memory delete <memory_id> [--force]
    neuroca memory search <query> [--tier=<tier>] [--limit=<limit>]
    neuroca memory stats [--tier=<tier>]
    neuroca memory export [--tier=<tier>] [--format=<format>] [--output=<file>]
    neuroca memory import <file> [--tier=<tier>] [--merge-strategy=<strategy>]
    neuroca memory optimize [--tier=<tier>]
    neuroca memory clear [--tier=<tier>] [--force]

Examples:
    # List all memories in working memory
    neuroca memory list --tier=working

    # Store new content in episodic memory
    neuroca memory store "Meeting with team about project roadmap" --tier=episodic --tags="meeting,roadmap"

    # Search for memories related to a topic
    neuroca memory search "project goals" --tier=semantic --limit=10

    # Get memory statistics
    neuroca memory stats

    # Export memories to a file
    neuroca memory export --tier=all --format=json --output=memories.json
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import click
import yaml
from rich.console import Console
from rich.table import Table

from neuroca.core.exceptions import (
    MemoryAccessError,
    MemoryNotFoundError,
    MemoryOperationError,
    MemoryTierNotFoundError,
    ValidationError,
)
from neuroca.memory.factory import MemoryFactory
from neuroca.memory.interfaces import MemoryInterface
from neuroca.memory.models import MemoryItem, MemoryQuery, MemoryStats
from neuroca.memory.tiers import MemoryTier

# Configure logger
logger = logging.getLogger(__name__)

# Console for rich output
console = Console()

# Valid memory tiers
VALID_TIERS = ["working", "episodic", "semantic", "all"]
VALID_FORMATS = ["json", "yaml", "table", "text"]
VALID_MERGE_STRATEGIES = ["replace", "append", "skip_existing"]

# Helper functions
def get_memory_interface(tier: str = None) -> MemoryInterface:
    """
    Get the appropriate memory interface based on the specified tier.
    
    Args:
        tier: The memory tier to access. If None, returns the default memory interface.
        
    Returns:
        An instance of the appropriate MemoryInterface.
        
    Raises:
        MemoryTierNotFoundError: If the specified tier is invalid.
    """
    try:
        factory = MemoryFactory()
        if tier is None or tier == "all":
            return factory.get_default_memory()
        
        if tier not in VALID_TIERS:
            raise MemoryTierNotFoundError(f"Invalid memory tier: {tier}")
        
        tier_enum = getattr(MemoryTier, tier.upper())
        return factory.get_memory_by_tier(tier_enum)
    except Exception as e:
        logger.error(f"Failed to get memory interface: {str(e)}")
        raise MemoryOperationError(f"Failed to access memory system: {str(e)}")

def format_output(data: Any, format_type: str) -> str:
    """
    Format data according to the specified format type.
    
    Args:
        data: The data to format.
        format_type: The format to use (json, yaml, etc.).
        
    Returns:
        Formatted string representation of the data.
        
    Raises:
        ValidationError: If the format type is invalid.
    """
    if format_type not in VALID_FORMATS:
        raise ValidationError(f"Invalid format: {format_type}. Valid formats: {', '.join(VALID_FORMATS)}")
    
    if format_type == "json":
        return json.dumps(data, indent=2, default=str)
    elif format_type == "yaml":
        return yaml.dump(data, default_flow_style=False)
    elif format_type == "text":
        if isinstance(data, list):
            return "\n".join([str(item) for item in data])
        return str(data)
    else:  # table format is handled separately with rich
        return data

def validate_memory_id(memory_id: str) -> str:
    """
    Validate a memory ID.
    
    Args:
        memory_id: The memory ID to validate.
        
    Returns:
        The validated memory ID.
        
    Raises:
        ValidationError: If the memory ID is invalid.
    """
    if not memory_id or not isinstance(memory_id, str):
        raise ValidationError("Memory ID must be a non-empty string")
    return memory_id

def validate_tier(tier: str) -> Optional[str]:
    """
    Validate a memory tier.
    
    Args:
        tier: The memory tier to validate.
        
    Returns:
        The validated memory tier or None if not specified.
        
    Raises:
        ValidationError: If the tier is invalid.
    """
    if tier is None:
        return None
    
    if tier not in VALID_TIERS:
        raise ValidationError(f"Invalid tier: {tier}. Valid tiers: {', '.join(VALID_TIERS)}")
    
    return tier

def display_table(data: List[Dict], title: str) -> None:
    """
    Display data as a rich table.
    
    Args:
        data: List of dictionaries to display.
        title: Title for the table.
    """
    if not data:
        console.print(f"[yellow]No {title} found.[/yellow]")
        return
    
    table = Table(title=title)
    
    # Add columns based on the first item's keys
    if data and isinstance(data[0], dict):
        for key in data[0].keys():
            table.add_column(str(key).capitalize(), style="cyan")
        
        # Add rows
        for item in data:
            table.add_row(*[str(value) for value in item.values()])
    
    console.print(table)

# CLI command group
@click.group(name="memory")
def memory_cli():
    """
    Commands for interacting with the NeuroCognitive Architecture's memory system.
    
    The memory system consists of three tiers: working memory, episodic memory, and semantic memory.
    These commands allow you to manage and interact with these memory tiers.
    """
    pass

@memory_cli.command(name="list")
@click.option("--tier", type=click.Choice(VALID_TIERS), default=None, help="Memory tier to list from.")
@click.option("--format", "format_type", type=click.Choice(VALID_FORMATS), default="table", help="Output format.")
def list_memories(tier: Optional[str], format_type: str):
    """
    List memories from the specified tier or all tiers.
    
    Examples:
        neuroca memory list
        neuroca memory list --tier=working
        neuroca memory list --format=json
    """
    try:
        tier = validate_tier(tier)
        memory = get_memory_interface(tier)
        
        logger.debug(f"Listing memories from tier: {tier or 'all'}")
        memories = memory.list_memories()
        
        if not memories:
            console.print("[yellow]No memories found.[/yellow]")
            return
        
        # Convert to serializable format
        serializable_memories = [
            {
                "id": mem.id,
                "content": mem.content[:50] + "..." if len(mem.content) > 50 else mem.content,
                "tier": mem.tier.name.lower() if hasattr(mem, 'tier') else "unknown",
                "created_at": mem.created_at.isoformat() if hasattr(mem, 'created_at') else "unknown",
                "tags": ", ".join(mem.tags) if hasattr(mem, 'tags') and mem.tags else ""
            }
            for mem in memories
        ]
        
        if format_type == "table":
            display_table(serializable_memories, f"Memories ({tier or 'all'} tier)")
        else:
            formatted_output = format_output(serializable_memories, format_type)
            console.print(formatted_output)
            
    except (ValidationError, MemoryTierNotFoundError) as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        sys.exit(1)
    except MemoryOperationError as e:
        console.print(f"[red]Memory operation failed:[/red] {str(e)}")
        logger.error(f"Memory list operation failed: {str(e)}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        logger.exception("Unexpected error during memory list operation")
        sys.exit(1)

@memory_cli.command(name="get")
@click.argument("memory_id", required=True)
@click.option("--format", "format_type", type=click.Choice(VALID_FORMATS), default="text", help="Output format.")
def get_memory(memory_id: str, format_type: str):
    """
    Retrieve a specific memory by ID.
    
    Examples:
        neuroca memory get abc123
        neuroca memory get abc123 --format=json
    """
    try:
        memory_id = validate_memory_id(memory_id)
        memory = get_memory_interface()
        
        logger.debug(f"Retrieving memory with ID: {memory_id}")
        memory_item = memory.get_memory(memory_id)
        
        if not memory_item:
            console.print(f"[yellow]Memory with ID '{memory_id}' not found.[/yellow]")
            return
        
        # Convert to serializable format
        serializable_memory = {
            "id": memory_item.id,
            "content": memory_item.content,
            "tier": memory_item.tier.name.lower() if hasattr(memory_item, 'tier') else "unknown",
            "created_at": memory_item.created_at.isoformat() if hasattr(memory_item, 'created_at') else "unknown",
            "updated_at": memory_item.updated_at.isoformat() if hasattr(memory_item, 'updated_at') else "unknown",
            "tags": memory_item.tags if hasattr(memory_item, 'tags') else [],
            "metadata": memory_item.metadata if hasattr(memory_item, 'metadata') else {}
        }
        
        if format_type == "table":
            # For table format, we'll create a list of key-value pairs
            table_data = [{"Property": k, "Value": str(v)} for k, v in serializable_memory.items()]
            display_table(table_data, f"Memory {memory_id}")
        else:
            formatted_output = format_output(serializable_memory, format_type)
            console.print(formatted_output)
            
    except ValidationError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        sys.exit(1)
    except MemoryNotFoundError as e:
        console.print(f"[yellow]Memory not found:[/yellow] {str(e)}")
        sys.exit(1)
    except MemoryOperationError as e:
        console.print(f"[red]Memory operation failed:[/red] {str(e)}")
        logger.error(f"Memory get operation failed: {str(e)}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        logger.exception("Unexpected error during memory get operation")
        sys.exit(1)

@memory_cli.command(name="store")
@click.argument("content", required=True)
@click.option("--tier", type=click.Choice(VALID_TIERS[:-1]), default="working", help="Memory tier to store in.")
@click.option("--tags", type=str, default="", help="Comma-separated tags for the memory.")
def store_memory(content: str, tier: str, tags: str):
    """
    Store new content in the specified memory tier.
    
    Examples:
        neuroca memory store "Important information to remember"
        neuroca memory store "Meeting notes" --tier=episodic --tags=meeting,notes
    """
    try:
        tier = validate_tier(tier)
        if tier == "all":
            raise ValidationError("Cannot store to 'all' tiers at once. Please specify a single tier.")
        
        memory = get_memory_interface(tier)
        
        # Parse tags
        tag_list = [tag.strip() for tag in tags.split(",")] if tags else []
        
        # Create memory item
        memory_item = MemoryItem(
            content=content,
            tags=tag_list,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={"source": "cli", "timestamp": time.time()}
        )
        
        logger.debug(f"Storing memory in tier: {tier or 'default'}")
        result = memory.store_memory(memory_item)
        
        console.print(f"[green]Memory stored successfully with ID:[/green] {result.id}")
        console.print(f"[green]Tier:[/green] {tier}")
        if tag_list:
            console.print(f"[green]Tags:[/green] {', '.join(tag_list)}")
            
    except ValidationError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        sys.exit(1)
    except MemoryOperationError as e:
        console.print(f"[red]Memory operation failed:[/red] {str(e)}")
        logger.error(f"Memory store operation failed: {str(e)}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        logger.exception("Unexpected error during memory store operation")
        sys.exit(1)

@memory_cli.command(name="delete")
@click.argument("memory_id", required=True)
@click.option("--force", is_flag=True, help="Force deletion without confirmation.")
def delete_memory(memory_id: str, force: bool):
    """
    Delete a specific memory by ID.
    
    Examples:
        neuroca memory delete abc123
        neuroca memory delete abc123 --force
    """
    try:
        memory_id = validate_memory_id(memory_id)
        memory = get_memory_interface()
        
        # Confirm deletion unless force flag is used
        if not force:
            if not click.confirm(f"Are you sure you want to delete memory with ID '{memory_id}'?"):
                console.print("[yellow]Deletion cancelled.[/yellow]")
                return
        
        logger.debug(f"Deleting memory with ID: {memory_id}")
        result = memory.delete_memory(memory_id)
        
        if result:
            console.print(f"[green]Memory with ID '{memory_id}' deleted successfully.[/green]")
        else:
            console.print(f"[yellow]Memory with ID '{memory_id}' not found or could not be deleted.[/yellow]")
            
    except ValidationError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        sys.exit(1)
    except MemoryNotFoundError as e:
        console.print(f"[yellow]Memory not found:[/yellow] {str(e)}")
        sys.exit(1)
    except MemoryOperationError as e:
        console.print(f"[red]Memory operation failed:[/red] {str(e)}")
        logger.error(f"Memory delete operation failed: {str(e)}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        logger.exception("Unexpected error during memory delete operation")
        sys.exit(1)

@memory_cli.command(name="search")
@click.argument("query", required=True)
@click.option("--tier", type=click.Choice(VALID_TIERS), default=None, help="Memory tier to search in.")
@click.option("--limit", type=int, default=10, help="Maximum number of results to return.")
def search_memories(query: str, tier: Optional[str], limit: int):
    """
    Search for memories matching the query.
    
    Examples:
        neuroca memory search "project goals"
        neuroca memory search "meeting" --tier=episodic --limit=5
    """
    try:
        tier = validate_tier(tier)
        memory = get_memory_interface(tier)
        
        if limit < 1:
            raise ValidationError("Limit must be a positive integer")
        
        # Create search query
        memory_query = MemoryQuery(
            query_text=query,
            limit=limit,
            filters={"tier": tier} if tier and tier != "all" else {}
        )
        
        logger.debug(f"Searching memories with query: '{query}' in tier: {tier or 'all'}")
        results = memory.search_memories(memory_query)
        
        if not results:
            console.print(f"[yellow]No memories found matching query: '{query}'[/yellow]")
            return
        
        # Convert to serializable format
        serializable_results = [
            {
                "id": mem.id,
                "content": mem.content[:50] + "..." if len(mem.content) > 50 else mem.content,
                "tier": mem.tier.name.lower() if hasattr(mem, 'tier') else "unknown",
                "relevance": f"{mem.relevance:.2f}" if hasattr(mem, 'relevance') else "N/A",
                "tags": ", ".join(mem.tags) if hasattr(mem, 'tags') and mem.tags else ""
            }
            for mem in results
        ]
        
        display_table(serializable_results, f"Search Results for '{query}'")
            
    except ValidationError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        sys.exit(1)
    except MemoryOperationError as e:
        console.print(f"[red]Memory operation failed:[/red] {str(e)}")
        logger.error(f"Memory search operation failed: {str(e)}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        logger.exception("Unexpected error during memory search operation")
        sys.exit(1)

@memory_cli.command(name="stats")
@click.option("--tier", type=click.Choice(VALID_TIERS), default=None, help="Memory tier to get stats for.")
def memory_stats(tier: Optional[str]):
    """
    Get statistics about the memory system.
    
    Examples:
        neuroca memory stats
        neuroca memory stats --tier=semantic
    """
    try:
        tier = validate_tier(tier)
        memory = get_memory_interface(tier)
        
        logger.debug(f"Getting memory stats for tier: {tier or 'all'}")
        stats = memory.get_stats()
        
        if not stats:
            console.print("[yellow]No statistics available.[/yellow]")
            return
        
        # Display stats in a table
        table = Table(title=f"Memory Statistics ({tier or 'all'} tier)")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        # Add rows based on stats
        if isinstance(stats, dict):
            for key, value in stats.items():
                table.add_row(str(key).replace("_", " ").capitalize(), str(value))
        else:
            # If stats is a MemoryStats object
            for attr in dir(stats):
                if not attr.startswith("_") and not callable(getattr(stats, attr)):
                    value = getattr(stats, attr)
                    if isinstance(value, dict):
                        # Handle nested dictionaries
                        for sub_key, sub_value in value.items():
                            table.add_row(f"{attr} - {sub_key}".replace("_", " ").capitalize(), str(sub_value))
                    else:
                        table.add_row(attr.replace("_", " ").capitalize(), str(value))
        
        console.print(table)
            
    except ValidationError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        sys.exit(1)
    except MemoryOperationError as e:
        console.print(f"[red]Memory operation failed:[/red] {str(e)}")
        logger.error(f"Memory stats operation failed: {str(e)}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        logger.exception("Unexpected error during memory stats operation")
        sys.exit(1)

@memory_cli.command(name="export")
@click.option("--tier", type=click.Choice(VALID_TIERS), default=None, help="Memory tier to export.")
@click.option("--format", "format_type", type=click.Choice(["json", "yaml"]), default="json", help="Export format.")
@click.option("--output", type=str, default=None, help="Output file path. If not specified, prints to stdout.")
def export_memories(tier: Optional[str], format_type: str, output: Optional[str]):
    """
    Export memories to a file or stdout.
    
    Examples:
        neuroca memory export
        neuroca memory export --tier=episodic --format=json --output=memories.json
    """
    try:
        tier = validate_tier(tier)
        memory = get_memory_interface(tier)
        
        logger.debug(f"Exporting memories from tier: {tier or 'all'}")
        memories = memory.list_memories()
        
        if not memories:
            console.print("[yellow]No memories to export.[/yellow]")
            return
        
        # Convert to serializable format
        serializable_memories = [
            {
                "id": mem.id,
                "content": mem.content,
                "tier": mem.tier.name.lower() if hasattr(mem, 'tier') else "unknown",
                "created_at": mem.created_at.isoformat() if hasattr(mem, 'created_at') else None,
                "updated_at": mem.updated_at.isoformat() if hasattr(mem, 'updated_at') else None,
                "tags": mem.tags if hasattr(mem, 'tags') else [],
                "metadata": mem.metadata if hasattr(mem, 'metadata') else {}
            }
            for mem in memories
        ]
        
        # Format the output
        if format_type == "json":
            formatted_output = json.dumps(serializable_memories, indent=2, default=str)
        else:  # yaml
            formatted_output = yaml.dump(serializable_memories, default_flow_style=False)
        
        # Write to file or stdout
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(formatted_output)
            
            console.print(f"[green]Exported {len(memories)} memories to {output}[/green]")
        else:
            console.print(formatted_output)
            
    except ValidationError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        sys.exit(1)
    except MemoryOperationError as e:
        console.print(f"[red]Memory operation failed:[/red] {str(e)}")
        logger.error(f"Memory export operation failed: {str(e)}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        logger.exception("Unexpected error during memory export operation")
        sys.exit(1)

@memory_cli.command(name="import")
@click.argument("file", type=click.Path(exists=True, readable=True))
@click.option("--tier", type=click.Choice(VALID_TIERS[:-1]), default=None, help="Memory tier to import into.")
@click.option("--merge-strategy", type=click.Choice(VALID_MERGE_STRATEGIES), default="skip_existing", 
              help="Strategy for handling existing memories.")
def import_memories(file: str, tier: Optional[str], merge_strategy: str):
    """
    Import memories from a file.
    
    Examples:
        neuroca memory import memories.json
        neuroca memory import memories.json --tier=semantic --merge-strategy=replace
    """
    try:
        tier = validate_tier(tier)
        if tier == "all":
            raise ValidationError("Cannot import to 'all' tiers at once. Please specify a single tier.")
        
        memory = get_memory_interface(tier)
        
        # Read the file
        file_path = Path(file)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Parse the content based on file extension
        if file_path.suffix.lower() == ".json":
            memories_data = json.loads(content)
        elif file_path.suffix.lower() in [".yaml", ".yml"]:
            memories_data = yaml.safe_load(content)
        else:
            raise ValidationError(f"Unsupported file format: {file_path.suffix}. Use .json, .yaml, or .yml")
        
        if not isinstance(memories_data, list):
            raise ValidationError("Import file must contain a list of memory objects")
        
        # Convert to MemoryItem objects
        memory_items = []
        for mem_data in memories_data:
            # Handle date strings
            created_at = datetime.fromisoformat(mem_data.get("created_at")) if mem_data.get("created_at") else datetime.now()
            updated_at = datetime.fromisoformat(mem_data.get("updated_at")) if mem_data.get("updated_at") else datetime.now()
            
            memory_item = MemoryItem(
                id=mem_data.get("id"),  # May be None for new memories
                content=mem_data.get("content", ""),
                tags=mem_data.get("tags", []),
                created_at=created_at,
                updated_at=updated_at,
                metadata=mem_data.get("metadata", {})
            )
            memory_items.append(memory_item)
        
        logger.debug(f"Importing {len(memory_items)} memories to tier: {tier or 'default'} with strategy: {merge_strategy}")
        
        # Import the memories
        result = memory.import_memories(memory_items, merge_strategy=merge_strategy)
        
        console.print(f"[green]Successfully imported {result.get('imported', 0)} memories.[/green]")
        if result.get("skipped", 0) > 0:
            console.print(f"[yellow]Skipped {result.get('skipped', 0)} existing memories.[/yellow]")
        if result.get("errors", 0) > 0:
            console.print(f"[red]Failed to import {result.get('errors', 0)} memories.[/red]")
            
    except ValidationError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        sys.exit(1)
    except MemoryOperationError as e:
        console.print(f"[red]Memory operation failed:[/red] {str(e)}")
        logger.error(f"Memory import operation failed: {str(e)}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        logger.exception("Unexpected error during memory import operation")
        sys.exit(1)

@memory_cli.command(name="optimize")
@click.option("--tier", type=click.Choice(VALID_TIERS), default=None, help="Memory tier to optimize.")
def optimize_memory(tier: Optional[str]):
    """
    Optimize the memory system for better performance.
    
    Examples:
        neuroca memory optimize
        neuroca memory optimize --tier=semantic
    """
    try:
        tier = validate_tier(tier)
        memory = get_memory_interface(tier)
        
        logger.debug(f"Optimizing memory tier: {tier or 'all'}")
        
        with console.status(f"Optimizing {tier or 'all'} memory tier(s)..."):
            result = memory.optimize()
        
        if result:
            console.print(f"[green]Memory optimization completed successfully for {tier or 'all'} tier(s).[/green]")
            if isinstance(result, dict):
                for key, value in result.items():
                    console.print(f"[green]{key}:[/green] {value}")
        else:
            console.print("[yellow]Memory optimization completed with no changes.[/yellow]")
            
    except ValidationError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        sys.exit(1)
    except MemoryOperationError as e:
        console.print(f"[red]Memory operation failed:[/red] {str(e)}")
        logger.error(f"Memory optimization operation failed: {str(e)}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        logger.exception("Unexpected error during memory optimization operation")
        sys.exit(1)

@memory_cli.command(name="clear")
@click.option("--tier", type=click.Choice(VALID_TIERS), default=None, help="Memory tier to clear.")
@click.option("--force", is_flag=True, help="Force clearing without confirmation.")
def clear_memory(tier: Optional[str], force: bool):
    """
    Clear all memories from the specified tier or all tiers.
    
    Examples:
        neuroca memory clear --tier=working
        neuroca memory clear --tier=all --force
    """
    try:
        tier = validate_tier(tier)
        memory = get_memory_interface(tier)
        
        tier_display = tier or "all"
        
        # Confirm clearing unless force flag is used
        if not force:
            if not click.confirm(f"Are you sure you want to clear {tier_display} memory tier(s)? This action cannot be undone."):
                console.print("[yellow]Operation cancelled.[/yellow]")
                return
        
        logger.debug(f"Clearing memory tier: {tier_display}")
        result = memory.clear()
        
        if result:
            console.print(f"[green]Successfully cleared {tier_display} memory tier(s).[/green]")
            if isinstance(result, dict):
                for key, value in result.items():
                    console.print(f"[green]{key}:[/green] {value}")
        else:
            console.print("[yellow]No memories were cleared.[/yellow]")
            
    except ValidationError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        sys.exit(1)
    except MemoryOperationError as e:
        console.print(f"[red]Memory operation failed:[/red] {str(e)}")
        logger.error(f"Memory clear operation failed: {str(e)}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        logger.exception("Unexpected error during memory clear operation")
        sys.exit(1)

if __name__ == "__main__":
    memory_cli()