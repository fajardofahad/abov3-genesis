# ABOV3 Genesis API Documentation

## Overview

This document provides comprehensive API documentation for ABOV3 Genesis, including all modules, classes, and their interfaces. ABOV3 Genesis is architected with clean separation of concerns and modular components.

## Table of Contents

1. [Core Modules](#core-modules)
2. [Agent System](#agent-system)
3. [Project Management](#project-management)
4. [Genesis Engine](#genesis-engine)
5. [Infrastructure](#infrastructure)
6. [UI Components](#ui-components)
7. [Utilities](#utilities)

## Core Modules

### `abov3.core.assistant.Assistant`

The main AI assistant that processes user requests and coordinates with other systems.

#### Constructor

```python
def __init__(self, agent=None, project_context: Dict[str, Any] = None, genesis_engine=None):
    """
    Initialize the Assistant
    
    Args:
        agent: Current active agent configuration
        project_context: Dictionary containing project-specific context
        genesis_engine: Reference to the Genesis Engine instance
    """
```

#### Methods

##### `async process(user_input: str, context: Dict[str, Any] = None) -> str`

Main processing method for user requests.

```python
async def process(self, user_input: str, context: Dict[str, Any] = None) -> str:
    """
    Process user input and return AI-generated response
    
    Args:
        user_input: The user's request or question
        context: Additional context for processing
        
    Returns:
        str: AI-generated response
        
    Raises:
        Exception: If processing fails
    """
```

##### `set_agent_manager(agent_manager: AgentManager)`

Sets the agent manager for automatic agent switching.

```python
def set_agent_manager(self, agent_manager: AgentManager):
    """
    Set agent manager for automatic agent switching
    
    Args:
        agent_manager: AgentManager instance
    """
```

---

### `abov3.core.ollama_client.OllamaClient`

Client for interacting with Ollama AI models.

#### Constructor

```python
def __init__(self, base_url: str = "http://localhost:11434"):
    """
    Initialize Ollama client
    
    Args:
        base_url: Ollama server URL
    """
```

#### Methods

##### `async is_available() -> bool`

Check if Ollama server is available.

```python
async def is_available(self) -> bool:
    """
    Check if Ollama server is available
    
    Returns:
        bool: True if server is available, False otherwise
    """
```

##### `async list_models() -> List[Dict[str, Any]]`

List all available Ollama models.

```python
async def list_models(self) -> List[Dict[str, Any]]:
    """
    List all available Ollama models
    
    Returns:
        List[Dict[str, Any]]: List of model information dictionaries
    """
```

##### `async chat(model: str, messages: List[Dict], **kwargs) -> Dict[str, Any]`

Send chat request to Ollama model.

```python
async def chat(self, model: str, messages: List[Dict], **kwargs) -> Dict[str, Any]:
    """
    Send chat request to Ollama model
    
    Args:
        model: Model name to use
        messages: List of message dictionaries
        **kwargs: Additional parameters
        
    Returns:
        Dict[str, Any]: Response from the model
    """
```

---

### `abov3.core.code_generator.CodeGenerator`

Generates and modifies code based on user requirements.

#### Constructor

```python
def __init__(self, project_path: Path):
    """
    Initialize CodeGenerator
    
    Args:
        project_path: Path to the project directory
    """
```

#### Methods

##### `async generate_code(prompt: str, language: str = None) -> str`

Generate code based on prompt.

```python
async def generate_code(self, prompt: str, language: str = None) -> str:
    """
    Generate code based on natural language prompt
    
    Args:
        prompt: Natural language description of desired code
        language: Programming language (auto-detected if None)
        
    Returns:
        str: Generated code
    """
```

##### `async modify_code(file_path: str, modifications: str) -> bool`

Modify existing code file.

```python
async def modify_code(self, file_path: str, modifications: str) -> bool:
    """
    Modify existing code based on requirements
    
    Args:
        file_path: Path to the code file to modify
        modifications: Description of desired modifications
        
    Returns:
        bool: True if modification was successful
    """
```

---

### `abov3.core.app_generator.FullApplicationGenerator`

Generates complete applications from high-level descriptions.

#### Constructor

```python
def __init__(self, project_path: Path):
    """
    Initialize FullApplicationGenerator
    
    Args:
        project_path: Path to the project directory
    """
```

#### Methods

##### `async generate_full_application(description: str, tech_stack: str = None) -> Dict[str, Any]`

Generate complete application.

```python
async def generate_full_application(self, description: str, tech_stack: str = None) -> Dict[str, Any]:
    """
    Generate a complete application from description
    
    Args:
        description: High-level description of the application
        tech_stack: Preferred technology stack
        
    Returns:
        Dict[str, Any]: Application generation results
    """
```

## Agent System

### `abov3.agents.manager.AgentManager`

Manages AI agents and their configurations.

#### Constructor

```python
def __init__(self, project_path: Path):
    """
    Initialize AgentManager
    
    Args:
        project_path: Path to the project directory
    """
```

#### Methods

##### `async create_agent(name: str, model: str, description: str, system_prompt: str) -> Agent`

Create a new AI agent.

```python
async def create_agent(self, name: str, model: str, description: str, system_prompt: str) -> Agent:
    """
    Create a new AI agent
    
    Args:
        name: Unique name for the agent
        model: Ollama model to use
        description: Human-readable description
        system_prompt: System prompt for the agent
        
    Returns:
        Agent: Created agent instance
    """
```

##### `async switch_agent(agent_name: str) -> bool`

Switch to a different agent.

```python
async def switch_agent(self, agent_name: str) -> bool:
    """
    Switch to a different agent
    
    Args:
        agent_name: Name of the agent to switch to
        
    Returns:
        bool: True if switch was successful
    """
```

##### `get_available_agents() -> List[Agent]`

Get list of all available agents.

```python
def get_available_agents(self) -> List[Agent]:
    """
    Get list of all available agents
    
    Returns:
        List[Agent]: List of available agents
    """
```

---

### `abov3.agents.manager.Agent`

Data class representing an AI agent configuration.

#### Attributes

```python
@dataclass
class Agent:
    name: str              # Unique agent name
    model: str             # Ollama model to use
    description: str       # Human-readable description
    system_prompt: str     # System prompt for the agent
    created: str          # Creation timestamp
    modified: str         # Last modification timestamp
    usage_count: int      # Number of times used
```

## Project Management

### `abov3.project.manager.ProjectManager`

Manages project configuration and context.

#### Constructor

```python
def __init__(self, project_path: Path):
    """
    Initialize ProjectManager
    
    Args:
        project_path: Path to the project directory
    """
```

#### Methods

##### `async initialize() -> bool`

Initialize project structure.

```python
async def initialize(self) -> bool:
    """
    Initialize project structure and configuration
    
    Returns:
        bool: True if initialization was successful
    """
```

##### `get_context() -> Dict[str, Any]`

Get project context information.

```python
def get_context(self) -> Dict[str, Any]:
    """
    Get current project context
    
    Returns:
        Dict[str, Any]: Project context dictionary
    """
```

##### `update_context(context: Dict[str, Any]) -> bool`

Update project context.

```python
def update_context(self, context: Dict[str, Any]) -> bool:
    """
    Update project context
    
    Args:
        context: New context data to merge
        
    Returns:
        bool: True if update was successful
    """
```

---

### `abov3.project.registry.ProjectRegistry`

Registry for managing multiple projects.

#### Constructor

```python
def __init__(self):
    """Initialize ProjectRegistry"""
```

#### Methods

##### `add_project(project_info: Dict[str, Any]) -> bool`

Add project to registry.

```python
def add_project(self, project_info: Dict[str, Any]) -> bool:
    """
    Add project to registry
    
    Args:
        project_info: Dictionary containing project information
        
    Returns:
        bool: True if project was added successfully
    """
```

##### `get_recent_projects(limit: int = 10) -> List[Dict[str, Any]]`

Get recently accessed projects.

```python
def get_recent_projects(self, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get recently accessed projects
    
    Args:
        limit: Maximum number of projects to return
        
    Returns:
        List[Dict[str, Any]]: List of recent projects
    """
```

## Genesis Engine

### `abov3.genesis.engine.GenesisEngine`

Core Genesis workflow engine.

#### Constructor

```python
def __init__(self, project_path: Path):
    """
    Initialize GenesisEngine
    
    Args:
        project_path: Path to the project directory
    """
```

#### Methods

##### `async start_genesis_workflow(idea: str) -> Dict[str, Any]`

Start the Genesis workflow for an idea.

```python
async def start_genesis_workflow(self, idea: str) -> Dict[str, Any]:
    """
    Start Genesis workflow to transform idea into reality
    
    Args:
        idea: User's idea or concept
        
    Returns:
        Dict[str, Any]: Workflow status and results
    """
```

##### `async get_genesis_stats() -> Dict[str, Any]`

Get Genesis statistics and progress.

```python
async def get_genesis_stats(self) -> Dict[str, Any]:
    """
    Get Genesis workflow statistics
    
    Returns:
        Dict[str, Any]: Statistics dictionary
    """
```

---

### `abov3.tasks.genesis_flow.GenesisFlow`

Manages the Genesis workflow phases.

#### Constructor

```python
def __init__(self, project_path: Path):
    """
    Initialize GenesisFlow
    
    Args:
        project_path: Path to the project directory
    """
```

#### Methods

##### `async update_phase(phase: str, status: str) -> bool`

Update Genesis phase status.

```python
async def update_phase(self, phase: str, status: str) -> bool:
    """
    Update Genesis workflow phase status
    
    Args:
        phase: Phase name (idea, design, build, test, deploy)
        status: New status (pending, in_progress, complete)
        
    Returns:
        bool: True if update was successful
    """
```

##### `get_current_phase() -> str`

Get current Genesis phase.

```python
def get_current_phase(self) -> str:
    """
    Get current Genesis workflow phase
    
    Returns:
        str: Current phase name
    """
```

## Infrastructure

### `abov3.infrastructure.orchestrator.InfrastructureOrchestrator`

Manages infrastructure components and resources.

#### Constructor

```python
def __init__(self):
    """Initialize InfrastructureOrchestrator"""
```

#### Methods

##### `start() -> bool`

Start infrastructure services.

```python
def start(self) -> bool:
    """
    Start all infrastructure services
    
    Returns:
        bool: True if services started successfully
    """
```

##### `stop() -> bool`

Stop infrastructure services.

```python
def stop(self) -> bool:
    """
    Stop all infrastructure services
    
    Returns:
        bool: True if services stopped successfully
    """
```

##### `get_health_status() -> Dict[str, Any]`

Get infrastructure health status.

```python
def get_health_status(self) -> Dict[str, Any]:
    """
    Get health status of all infrastructure components
    
    Returns:
        Dict[str, Any]: Health status dictionary
    """
```

---

### `abov3.infrastructure.monitoring.MonitoringSystem`

System monitoring and metrics collection.

#### Constructor

```python
def __init__(self):
    """Initialize MonitoringSystem"""
```

#### Methods

##### `collect_metrics() -> Dict[str, Any]`

Collect system metrics.

```python
def collect_metrics(self) -> Dict[str, Any]:
    """
    Collect current system metrics
    
    Returns:
        Dict[str, Any]: Metrics dictionary
    """
```

##### `get_performance_report() -> Dict[str, Any]`

Get performance report.

```python
def get_performance_report(self) -> Dict[str, Any]:
    """
    Generate performance report
    
    Returns:
        Dict[str, Any]: Performance report
    """
```

## UI Components

### `abov3.ui.display.UIManager`

Manages user interface elements and display.

#### Constructor

```python
def __init__(self):
    """Initialize UIManager"""
```

#### Methods

##### `show_banner(title: str, subtitle: str = None)`

Show application banner.

```python
def show_banner(self, title: str, subtitle: str = None):
    """
    Display application banner
    
    Args:
        title: Main title text
        subtitle: Optional subtitle text
    """
```

##### `show_progress(message: str, progress: float)`

Show progress indicator.

```python
def show_progress(self, message: str, progress: float):
    """
    Display progress indicator
    
    Args:
        message: Progress message
        progress: Progress value (0.0 to 1.0)
    """
```

---

### `abov3.ui.genz.GenZStatus`

Provides GenZ-style status messages and animations.

#### Constructor

```python
def __init__(self):
    """Initialize GenZStatus"""
```

#### Methods

##### `get_status(status_type: str) -> str`

Get GenZ-style status message.

```python
def get_status(self, status_type: str) -> str:
    """
    Get GenZ-style status message
    
    Args:
        status_type: Type of status (working, success, error, etc.)
        
    Returns:
        str: Formatted status message
    """
```

## Session Management

### `abov3.session.manager.SessionManager`

Manages user sessions and conversation history.

#### Constructor

```python
def __init__(self, project_path: Path):
    """
    Initialize SessionManager
    
    Args:
        project_path: Path to the project directory
    """
```

#### Methods

##### `create_new_session() -> str`

Create a new session.

```python
def create_new_session(self) -> str:
    """
    Create a new user session
    
    Returns:
        str: Session ID
    """
```

##### `async save_session() -> bool`

Save current session data.

```python
async def save_session(self) -> bool:
    """
    Save current session to storage
    
    Returns:
        bool: True if save was successful
    """
```

##### `async restore_session() -> Dict[str, Any]`

Restore previous session.

```python
async def restore_session(self) -> Dict[str, Any]:
    """
    Restore previous session data
    
    Returns:
        Dict[str, Any]: Session data or None if no session found
    """
```

## Utilities

### `abov3.core.error_handler.ErrorHandler`

Handles errors and provides recovery mechanisms.

#### Constructor

```python
def __init__(self):
    """Initialize ErrorHandler"""
```

#### Methods

##### `handle_error(error: Exception, context: Dict[str, Any]) -> str`

Handle and format errors.

```python
def handle_error(self, error: Exception, context: Dict[str, Any]) -> str:
    """
    Handle error and provide user-friendly message
    
    Args:
        error: Exception that occurred
        context: Error context information
        
    Returns:
        str: User-friendly error message
    """
```

---

### `abov3.core.validator.Validator`

Validates inputs and configurations.

#### Constructor

```python
def __init__(self):
    """Initialize Validator"""
```

#### Methods

##### `validate_project_path(path: str) -> bool`

Validate project path.

```python
def validate_project_path(self, path: str) -> bool:
    """
    Validate project directory path
    
    Args:
        path: Path to validate
        
    Returns:
        bool: True if path is valid
    """
```

##### `validate_agent_config(config: Dict[str, Any]) -> bool`

Validate agent configuration.

```python
def validate_agent_config(self, config: Dict[str, Any]) -> bool:
    """
    Validate agent configuration
    
    Args:
        config: Agent configuration dictionary
        
    Returns:
        bool: True if configuration is valid
    """
```

## Error Handling

All API methods follow consistent error handling patterns:

- **Validation Errors**: Raise `ValueError` for invalid inputs
- **Network Errors**: Raise `ConnectionError` for Ollama connectivity issues  
- **File System Errors**: Raise `IOError` for file operation failures
- **Configuration Errors**: Raise `ConfigurationError` for invalid configurations

## Response Formats

### Standard Response Format

Most API methods return structured data:

```python
{
    "status": "success" | "error",
    "data": Any,           # Response data
    "message": str,        # Human-readable message
    "timestamp": str,      # ISO format timestamp
    "error_code": int      # Error code (only for errors)
}
```

### Agent Response Format

```python
{
    "name": str,           # Agent name
    "model": str,          # AI model used
    "response": str,       # Generated response
    "context_used": bool,  # Whether context was utilized
    "tokens_used": int,    # Approximate token usage
    "processing_time": float  # Response time in seconds
}
```

### Project Context Format

```python
{
    "project_path": str,   # Absolute project path
    "project_name": str,   # Project name
    "language": str,       # Primary programming language
    "framework": str,      # Framework/technology used
    "files": List[str],    # List of project files
    "dependencies": List[str],  # Project dependencies
    "last_modified": str   # Last modification timestamp
}
```

## Rate Limits and Performance

- **Concurrent Requests**: Maximum 5 simultaneous requests
- **Token Limits**: Respects individual model token limits
- **Request Timeout**: 60 seconds default timeout
- **Memory Management**: Automatic conversation history pruning

## Authentication

ABOV3 Genesis runs locally and doesn't require authentication. However, when extending for multi-user scenarios:

- Use JWT tokens for session management
- Implement role-based access control for agents
- Secure API endpoints with authentication middleware

## Versioning

API follows semantic versioning:
- **Major Version**: Breaking changes
- **Minor Version**: New features (backward compatible)
- **Patch Version**: Bug fixes and improvements

Current version: `1.0.0`

## Examples

### Basic Usage

```python
from abov3.core.assistant import Assistant
from abov3.agents.manager import AgentManager
from pathlib import Path

# Initialize components
project_path = Path("/path/to/project")
agent_manager = AgentManager(project_path)
assistant = Assistant(agent_manager.current_agent)

# Process user request
response = await assistant.process("Create a Python function to calculate fibonacci")
print(response)
```

### Advanced Usage

```python
from abov3.genesis.engine import GenesisEngine
from abov3.core.app_generator import FullApplicationGenerator

# Initialize Genesis
genesis = GenesisEngine(project_path)
app_gen = FullApplicationGenerator(project_path)

# Generate complete application
result = await app_gen.generate_full_application(
    "E-commerce website with user authentication and payment processing"
)

# Start Genesis workflow
workflow_result = await genesis.start_genesis_workflow(
    "Build a modern task management application"
)
```

This API documentation provides a comprehensive reference for developers working with ABOV3 Genesis. For more detailed examples and tutorials, see the [User Guide](USER_GUIDE.md) and [Examples](examples/) directory.