#!/usr/bin/env python3
"""
NeuroCognitive Architecture (NCA) Command Line Interface

This module serves as the main entry point for the NCA CLI, providing a comprehensive
interface for interacting with the NeuroCognitive Architecture system. It enables users
to manage, monitor, and control various aspects of the NCA system through a structured
command hierarchy.

The CLI is built using Click for command parsing and organization, with comprehensive
logging, error handling, and configuration management.

Usage:
    neuroca --help
    neuroca [command] [options]
    
Examples:
    neuroca --version
    neuroca init --config path/to/config.yaml
    neuroca memory list
    neuroca run --model gpt-4 --input "Process this text"
    
Environment Variables:
    NEUROCA_CONFIG_PATH: Path to configuration file
    NEUROCA_LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    NEUROCA_API_KEY: API key for LLM service integration
"""

import os
import sys
import logging
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Package version
__version__ = "0.1.0"

# Initialize rich console for pretty output
console = Console()

# Configure logging with rich handler
logging.basicConfig(
    level=os.environ.get("NEUROCA_LOG_LEVEL", "INFO").upper(),
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("neuroca.cli")


class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""
    pass


class NCAConfig:
    """
    Configuration manager for the NCA system.
    
    Handles loading, validation, and access to configuration settings from
    various sources (environment variables, config files, CLI parameters).
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path or os.environ.get("NEUROCA_CONFIG_PATH")
        self.config: Dict[str, Any] = {}
        
    def load(self) -> Dict[str, Any]:
        """
        Load configuration from file if specified.
        
        Returns:
            Dict containing configuration values
            
        Raises:
            ConfigurationError: If configuration file cannot be loaded or parsed
        """
        if not self.config_path:
            logger.debug("No configuration file specified, using defaults")
            return {}
            
        config_path = Path(self.config_path)
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {self.config_path}")
            
        try:
            if config_path.suffix.lower() in ('.yaml', '.yml'):
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                raise ConfigurationError(f"Unsupported configuration format: {config_path.suffix}")
                
            logger.debug(f"Loaded configuration from {self.config_path}")
            return self.config
            
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ConfigurationError(f"Failed to parse configuration file: {str(e)}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {str(e)}")
            
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key to retrieve
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)
        
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key to set
            value: Value to assign
        """
        self.config[key] = value
        
    def save(self, path: Optional[str] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            path: Optional path to save configuration to (defaults to self.config_path)
            
        Raises:
            ConfigurationError: If configuration cannot be saved
        """
        save_path = Path(path or self.config_path)
        if not save_path:
            raise ConfigurationError("No configuration path specified for saving")
            
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            if save_path.suffix.lower() in ('.yaml', '.yml'):
                with open(save_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            elif save_path.suffix.lower() == '.json':
                with open(save_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
            else:
                raise ConfigurationError(f"Unsupported configuration format: {save_path.suffix}")
                
            logger.debug(f"Saved configuration to {save_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {str(e)}")


# Common options for all commands
def common_options(func):
    """Common CLI options decorator for all commands."""
    func = click.option('--verbose', '-v', is_flag=True, help="Enable verbose output")(func)
    func = click.option('--config', '-c', help="Path to configuration file")(func)
    return func


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="NeuroCognitive Architecture (NCA)")
@common_options
@click.pass_context
def cli(ctx: click.Context, verbose: bool, config: Optional[str]) -> None:
    """
    NeuroCognitive Architecture (NCA) Command Line Interface.
    
    This tool provides commands to interact with the NCA system, manage memory tiers,
    control cognitive processes, and integrate with LLM models.
    """
    # Set up context object for sharing data between commands
    ctx.ensure_object(dict)
    
    # Configure logging level based on verbosity
    if verbose:
        logging.getLogger("neuroca").setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Initialize configuration
    try:
        ctx.obj['config'] = NCAConfig(config)
        ctx.obj['config'].load()
    except ConfigurationError as e:
        logger.error(f"Configuration error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        sys.exit(1)


@cli.command()
@click.option('--force', '-f', is_flag=True, help="Force initialization even if already initialized")
@click.option('--template', '-t', help="Template configuration to use")
@common_options
@click.pass_context
def init(ctx: click.Context, force: bool, template: Optional[str], verbose: bool, config: Optional[str]) -> None:
    """
    Initialize the NCA system with required configuration and resources.
    
    This command sets up the necessary directory structure, configuration files,
    and initial state for the NCA system to operate. It can use templates for
    different deployment scenarios.
    """
    logger.info("Initializing NeuroCognitive Architecture system...")
    
    try:
        # Load template configuration if specified
        template_config = {}
        if template:
            template_path = Path(template)
            if not template_path.exists():
                logger.error(f"Template configuration not found: {template}")
                sys.exit(1)
                
            try:
                if template_path.suffix.lower() in ('.yaml', '.yml'):
                    with open(template_path, 'r') as f:
                        template_config = yaml.safe_load(f)
                elif template_path.suffix.lower() == '.json':
                    with open(template_path, 'r') as f:
                        template_config = json.load(f)
                else:
                    logger.error(f"Unsupported template format: {template_path.suffix}")
                    sys.exit(1)
            except Exception as e:
                logger.error(f"Failed to load template: {str(e)}")
                sys.exit(1)
        
        # Create default configuration
        default_config = {
            "version": __version__,
            "memory": {
                "working_memory": {
                    "capacity": 7,
                    "decay_rate": 0.1
                },
                "short_term": {
                    "capacity": 100,
                    "retention_period": 3600  # 1 hour in seconds
                },
                "long_term": {
                    "storage_path": "data/long_term",
                    "indexing": "semantic"
                }
            },
            "llm": {
                "default_model": "gpt-3.5-turbo",
                "api_key_env": "NEUROCA_API_KEY",
                "timeout": 30
            },
            "health": {
                "energy_decay_rate": 0.05,
                "rest_recovery_rate": 0.1,
                "critical_threshold": 0.2
            }
        }
        
        # Merge with template if provided
        if template_config:
            # Deep merge the configurations
            def deep_merge(source, destination):
                for key, value in source.items():
                    if isinstance(value, dict):
                        node = destination.setdefault(key, {})
                        deep_merge(value, node)
                    else:
                        destination[key] = value
                return destination
                
            config_data = deep_merge(template_config, default_config)
        else:
            config_data = default_config
        
        # Determine configuration path
        config_path = config or ctx.obj['config'].config_path or "config/neuroca.yaml"
        config_file = Path(config_path)
        
        # Check if already initialized
        if config_file.exists() and not force:
            logger.warning(f"Configuration already exists at {config_file}. Use --force to overwrite.")
            sys.exit(1)
            
        # Create directory structure
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create data directories
        data_dirs = [
            Path("data/working_memory"),
            Path("data/short_term"),
            Path("data/long_term"),
            Path("logs")
        ]
        
        for directory in data_dirs:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
        
        # Save configuration
        ctx.obj['config'].config = config_data
        ctx.obj['config'].config_path = str(config_file)
        ctx.obj['config'].save()
        
        logger.info(f"NCA system initialized successfully. Configuration saved to {config_file}")
        
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        if verbose:
            logger.exception("Detailed error information:")
        sys.exit(1)


@cli.group()
@common_options
@click.pass_context
def memory(ctx: click.Context, verbose: bool, config: Optional[str]) -> None:
    """
    Manage the three-tiered memory system.
    
    Commands for interacting with working memory, short-term memory,
    and long-term memory components of the NCA system.
    """
    pass


@memory.command("list")
@click.option('--tier', '-t', type=click.Choice(['working', 'short', 'long', 'all']), 
              default='all', help="Memory tier to list")
@common_options
@click.pass_context
def memory_list(ctx: click.Context, tier: str, verbose: bool, config: Optional[str]) -> None:
    """
    List contents of memory tiers.
    
    Displays the current contents of the specified memory tier(s) with metadata
    such as creation time, access count, and decay status.
    """
    logger.info(f"Listing {tier} memory contents...")
    
    # This would connect to the actual memory subsystems in a real implementation
    # For now, we'll just show a sample output
    
    table = Table(title=f"{tier.title()} Memory Contents")
    table.add_column("ID", style="cyan")
    table.add_column("Content", style="green")
    table.add_column("Created", style="yellow")
    table.add_column("Last Access", style="yellow")
    table.add_column("Access Count", style="magenta")
    table.add_column("Status", style="blue")
    
    # Sample data - in a real implementation, this would come from the memory subsystems
    if tier in ['working', 'all']:
        table.add_row("wm-001", "Current task context: CLI development", "2023-10-15 14:30", 
                     "2023-10-15 15:45", "12", "Active")
        table.add_row("wm-002", "User preference: verbose output", "2023-10-15 14:32", 
                     "2023-10-15 15:40", "5", "Active")
    
    if tier in ['short', 'all']:
        table.add_row("stm-001", "Previous command results: memory stats", "2023-10-15 13:20", 
                     "2023-10-15 14:10", "3", "Decaying (0.7)")
        table.add_row("stm-002", "Error handling pattern for file operations", "2023-10-15 12:45", 
                     "2023-10-15 13:30", "2", "Decaying (0.5)")
    
    if tier in ['long', 'all']:
        table.add_row("ltm-001", "System initialization procedure", "2023-10-10 09:15", 
                     "2023-10-15 14:00", "8", "Consolidated")
        table.add_row("ltm-002", "User command history patterns", "2023-10-12 16:30", 
                     "2023-10-14 11:20", "15", "Consolidated")
    
    console.print(table)


@memory.command("clear")
@click.option('--tier', '-t', type=click.Choice(['working', 'short', 'long', 'all']), 
              required=True, help="Memory tier to clear")
@click.option('--force', '-f', is_flag=True, help="Force clearing without confirmation")
@common_options
@click.pass_context
def memory_clear(ctx: click.Context, tier: str, force: bool, verbose: bool, config: Optional[str]) -> None:
    """
    Clear contents of a memory tier.
    
    Removes all items from the specified memory tier. Use with caution,
    especially for long-term memory.
    """
    if not force:
        if not click.confirm(f"Are you sure you want to clear {tier} memory? This cannot be undone."):
            logger.info("Operation cancelled.")
            return
    
    logger.info(f"Clearing {tier} memory...")
    
    # This would connect to the actual memory subsystems in a real implementation
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Clearing {tier} memory...", total=None)
        # Simulate clearing operation
        import time
        time.sleep(1.5)
        progress.update(task, completed=True)
    
    logger.info(f"Successfully cleared {tier} memory")


@cli.group()
@common_options
@click.pass_context
def health(ctx: click.Context, verbose: bool, config: Optional[str]) -> None:
    """
    Monitor and manage system health dynamics.
    
    Commands for checking and controlling the biological-inspired health
    parameters of the NCA system, including energy levels and rest states.
    """
    pass


@health.command("status")
@common_options
@click.pass_context
def health_status(ctx: click.Context, verbose: bool, config: Optional[str]) -> None:
    """
    Display current health status of the NCA system.
    
    Shows energy levels, rest state, and other health-related metrics
    with recommendations for optimal performance.
    """
    logger.info("Retrieving system health status...")
    
    # This would connect to the actual health monitoring subsystem in a real implementation
    # For now, we'll just show a sample output
    
    # Sample health data
    health_data = {
        "energy": 0.72,
        "rest_state": "active",
        "continuous_operation": "4h 23m",
        "cognitive_load": 0.45,
        "memory_utilization": {
            "working": 0.68,
            "short_term": 0.41,
            "long_term": 0.22
        },
        "recommendations": [
            "Consider scheduling rest period within next 2 hours",
            "Working memory approaching high utilization"
        ]
    }
    
    # Display health information
    console.print("[bold]NCA System Health Status[/bold]")
    console.print(f"Energy Level: {health_data['energy']*100:.1f}%", 
                 style="green" if health_data['energy'] > 0.5 else "yellow" if health_data['energy'] > 0.2 else "red")
    console.print(f"Rest State: {health_data['rest_state'].title()}")
    console.print(f"Continuous Operation: {health_data['continuous_operation']}")
    console.print(f"Cognitive Load: {health_data['cognitive_load']*100:.1f}%")
    
    console.print("\n[bold]Memory Utilization[/bold]")
    for mem_type, util in health_data['memory_utilization'].items():
        console.print(f"{mem_type.replace('_', ' ').title()}: {util*100:.1f}%", 
                     style="green" if util < 0.6 else "yellow" if util < 0.8 else "red")
    
    if health_data['recommendations']:
        console.print("\n[bold]Recommendations[/bold]")
        for rec in health_data['recommendations']:
            console.print(f"• {rec}")


@cli.command()
@click.option('--model', '-m', help="LLM model to use for processing")
@click.option('--input', '-i', help="Input text or file path for processing")
@click.option('--output', '-o', help="Output file path for results")
@click.option('--interactive', is_flag=True, help="Run in interactive mode")
@common_options
@click.pass_context
def run(ctx: click.Context, model: Optional[str], input: Optional[str], 
        output: Optional[str], interactive: bool, verbose: bool, config: Optional[str]) -> None:
    """
    Run the NCA system with the specified input and configuration.
    
    This command executes the core NCA processing pipeline, integrating with
    the specified LLM model and applying the cognitive architecture to the input.
    """
    logger.info("Starting NCA processing...")
    
    # Get configuration
    cfg = ctx.obj['config']
    
    # Determine which model to use
    model_name = model or cfg.get("llm", {}).get("default_model", "gpt-3.5-turbo")
    logger.debug(f"Using model: {model_name}")
    
    # Handle input
    input_text = ""
    if input:
        input_path = Path(input)
        if input_path.exists() and input_path.is_file():
            try:
                with open(input_path, 'r') as f:
                    input_text = f.read()
                logger.debug(f"Loaded input from file: {input}")
            except Exception as e:
                logger.error(f"Failed to read input file: {str(e)}")
                sys.exit(1)
        else:
            input_text = input
            logger.debug("Using direct input text")
    
    # Interactive mode
    if interactive:
        logger.info("Starting interactive session. Type 'exit' or 'quit' to end.")
        console.print("[bold]NCA Interactive Mode[/bold] (Model: {})".format(model_name))
        console.print("Type your input and press Enter. Type 'exit' or 'quit' to end the session.")
        
        while True:
            try:
                user_input = console.input("\n[bold cyan]> [/bold cyan]")
                if user_input.lower() in ('exit', 'quit'):
                    break
                
                # This would process the input through the NCA system in a real implementation
                console.print("\n[dim]Processing...[/dim]")
                
                # Simulate processing
                import time
                time.sleep(1.5)
                
                # Sample response - in a real implementation, this would come from the NCA system
                response = f"NCA processed: {user_input}\n\nThis is a simulated response from the {model_name} model."
                console.print(f"\n[green]{response}[/green]")
                
            except KeyboardInterrupt:
                console.print("\nSession terminated by user.")
                break
            except Exception as e:
                logger.error(f"Error in interactive session: {str(e)}")
                if verbose:
                    logger.exception("Detailed error information:")
        
        logger.info("Interactive session ended")
        return
    
    # Non-interactive mode requires input
    if not input_text:
        logger.error("No input provided. Use --input option or --interactive mode.")
        sys.exit(1)
    
    # Process the input
    logger.info("Processing input...")
    
    # This would process the input through the NCA system in a real implementation
    # For now, we'll just show a sample output
    
    # Sample processing result
    result = f"NCA processed the input using {model_name} model.\n\n"
    result += "This is a simulated response that would normally contain the output from the NCA system."
    
    # Handle output
    if output:
        output_path = Path(output)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(result)
            logger.info(f"Results saved to {output}")
        except Exception as e:
            logger.error(f"Failed to write output file: {str(e)}")
            if verbose:
                logger.exception("Detailed error information:")
            sys.exit(1)
    else:
        # Print to console
        console.print("\n[bold]Processing Results:[/bold]")
        console.print(result)
    
    logger.info("NCA processing completed successfully")


@cli.command()
@click.argument('command', required=False)
@common_options
@click.pass_context
def help(ctx: click.Context, command: Optional[str], verbose: bool, config: Optional[str]) -> None:
    """
    Show help for a specific command or list all commands.
    
    If a command is specified, displays detailed help for that command.
    Otherwise, shows a list of all available commands with brief descriptions.
    """
    if command:
        # Get the command object
        cmd = cli.get_command(ctx, command)
        if cmd:
            # Show help for the specific command
            console.print(cmd.get_help(ctx))
        else:
            logger.error(f"Unknown command: {command}")
            console.print("\nAvailable commands:")
            for cmd_name in sorted(cli.list_commands(ctx)):
                cmd_obj = cli.get_command(ctx, cmd_name)
                console.print(f"  {cmd_name}: {cmd_obj.short_help}")
    else:
        # Show general help
        console.print(cli.get_help(ctx))


if __name__ == "__main__":
    try:
        cli(obj={})
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        logger.debug("Detailed error information:", exc_info=True)
        sys.exit(1)