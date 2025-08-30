#!/usr/bin/env python3
"""
ABOV3 Genesis - From Idea to Built Reality
Main entry point for the AI coding assistant
"""

import asyncio
import signal
import random
import sys
import os
import json
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import confirm

# Add the package root to Python path for development
package_root = Path(__file__).parent.parent
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

from abov3.genesis.engine import GenesisEngine
from abov3.project.manager import ProjectManager
from abov3.project.registry import ProjectRegistry
from abov3.core.assistant import Assistant
from abov3.agents.manager import AgentManager
from abov3.agents.commands import AgentCommandHandler
from abov3.tasks.manager import TaskManager
from abov3.tasks.genesis_flow import GenesisFlow
from abov3.session.manager import SessionManager
from abov3.permissions.manager import PermissionManager
from abov3.dependencies.detector import DependencyDetector
from abov3.ui.display import UIManager
from abov3.ui.genz import GenZStatus, AnimatedStatus
from abov3.security.integration import SecurityIntegration, initialize_abov3_security
from abov3.security.core import SecurityLevel

console = Console()

class ABOV3Genesis:
    """
    ABOV3 Genesis - Main application class
    From Idea to Built Reality
    """
    
    def __init__(self, project_path: Optional[Path] = None):
        self.version = "1.0.0"
        self.tagline = "From Idea to Built Reality"
        self.project_path = None
        self.project_manager = None
        self.registry = ProjectRegistry()
        self.genz = GenZStatus()
        self.animated_status = AnimatedStatus(console)
        self.ui = UIManager()
        self.genesis_engine = None
        
        # Configuration for persistent settings
        self.config_file = Path.home() / '.abov3' / 'config.yaml'
        self.current_model = self.load_saved_model()
        
        # Will be initialized after project selection
        self.agent_manager = None
        self.task_manager = None
        self.session_manager = None
        self.permission_manager = None
        self.dependency_detector = None
        self.assistant = None
        self.genesis_flow = None
        
        # Security integration
        self.security_integration = None
        
        # Processing state
        self.processing = False
        self.current_task = None
        self.task_queue = asyncio.Queue()
        self.interrupt_requested = False
        
        # Background tasks
        self.background_tasks = []
        
        # Initialize with project if provided
        if project_path:
            self.project_path = Path(project_path)
    
    def load_saved_model(self):
        """Load the saved AI model from config file"""
        try:
            if self.config_file.exists():
                import yaml
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
                return config.get('ai_model', 'llama3:latest')
        except Exception:
            pass  # Fall back to default if config can't be loaded
        return 'llama3:latest'
    
    def save_model_config(self, model_name: str):
        """Save the AI model selection to config file"""
        try:
            # Ensure config directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Load existing config or create new one
            config = {}
            if self.config_file.exists():
                import yaml
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
            
            # Update model setting
            config['ai_model'] = model_name
            self.current_model = model_name
            
            # Save config
            import yaml
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config, f, default_flow_style=False)
                
            console.print(f"[green]✅ Saved {model_name} as default AI model[/green]")
            
        except Exception as e:
            console.print(f"[yellow]⚠️  Could not save model config: {e}[/yellow]")
    
    async def initialize(self):
        """Initialize ABOV3 Genesis with project selection"""
        
        # Check if project was provided
        if not self.project_path:
            await self.select_project()
        
        if not self.project_path:
            console.print("[red]❌ No project selected. Cannot proceed without a project directory.[/red]")
            console.print("[yellow]ABOV3 Genesis requires a project directory to transform ideas into reality.[/yellow]")
            return False
        
        # Initialize project-specific components
        await self.initialize_project()
        return True
    
    async def select_project(self):
        """Interactive project selection with Genesis theme"""
        console.clear()
        
        # Show ABOV3 Genesis banner
        self.show_genesis_banner()
        
        # Check for recent projects
        recent_projects = self.registry.get_recent_projects(5)
        
        console.print("\n[bold yellow]🚨 No project directory specified![/bold yellow]")
        console.print("[dim]ABOV3 Genesis needs a home to build your reality[/dim]\n")
        
        options = []
        console.print("[bold]Choose your genesis path:[/bold]\n")
        
        if recent_projects:
            console.print("[cyan]📚 Continue Building (Recent Projects):[/cyan]")
            for i, proj in enumerate(recent_projects, 1):
                status = self.get_project_status(proj)
                console.print(f"  {i}. {proj['name']} {status}")
                console.print(f"     [dim]{proj['path']}[/dim]")
                options.append(('recent', proj['path']))
            
            next_num = len(recent_projects) + 1
        else:
            next_num = 1
        
        console.print(f"\n[cyan]🌟 Start Fresh:[/cyan]")
        console.print(f"  {next_num}. 💡 Create new project (Start your genesis)")
        options.append(('new', None))
        
        console.print(f"  {next_num + 1}. 📂 Open existing project")
        options.append(('existing', None))
        
        console.print(f"  {next_num + 2}. 🔍 Browse for project directory")
        options.append(('browse', None))
        
        console.print(f"  {next_num + 3}. ❌ Exit")
        options.append(('exit', None))
        
        # Get user choice
        while True:
            try:
                choice = console.input("\n[bold green]Choose your destiny >[/bold green] ")
                choice_idx = int(choice) - 1
                
                if 0 <= choice_idx < len(options):
                    action, path = options[choice_idx]
                    
                    if action == 'exit':
                        console.print("\n[yellow]The genesis awaits another day... ✨[/yellow]")
                        return
                    elif action == 'recent':
                        self.project_path = Path(path)
                        console.print(f"\n{self.genz.get_status('success')}")
                        console.print(f"[green]✓ Resuming genesis of: {self.project_path.name}[/green]")
                    elif action == 'new':
                        await self.create_new_project()
                    elif action == 'existing':
                        await self.open_existing_project()
                    elif action == 'browse':
                        await self.browse_for_project()
                    
                    break
                else:
                    console.print("[red]Invalid choice. Please try again.[/red]")
                    
            except ValueError:
                console.print("[red]Please enter a number.[/red]")
            except KeyboardInterrupt:
                console.print("\n[yellow]Genesis cancelled.[/yellow]")
                return
    
    def show_genesis_banner(self):
        """Display the ABOV3 Genesis banner"""
        # Manually crafted banner - each line carefully spaced to match top/bottom borders
        banner_text = """
╔══════════════════════════════════════════════════════╗
║              ABOV3 Genesis v1.0.0                    ║
║         From Idea to Built Reality                   ║
║                                                      ║
║   ✨ Transform your ideas into working code ✨       ║
║     💡 Idea → 📐 Design → 🔨 Build → 🚀 Ship         ║
╚══════════════════════════════════════════════════════╝
        """
        
        # Display the banner directly without Panel wrapper to avoid double borders
        console.print(Text(banner_text, style="cyan", justify="center"))
        
        # Show current AI model
        console.print(f"[dim]🤖 AI Model: [cyan]{self.current_model}[/cyan][/dim]")
        
        # Random Genesis motivation
        motivations = [
            "🌟 Every masterpiece starts with a single idea",
            "🚀 From zero to hero, one line at a time",
            "💫 Where imagination meets implementation",
            "🔥 Let's turn that spark into a wildfire",
            "✨ Your idea + ABOV3 = Reality"
        ]
        console.print(f"\n[italic dim]{random.choice(motivations)}[/italic dim]")
    
    def get_project_status(self, project: dict) -> str:
        """Get visual status of a project"""
        try:
            # Check project genesis status
            genesis_file = Path(project['path']) / '.abov3' / 'genesis.yaml'
            if genesis_file.exists():
                # Load genesis status
                import yaml
                with open(genesis_file) as f:
                    genesis = yaml.safe_load(f)
                    phase = genesis.get('current_phase', 'idea')
                    
                    phase_icons = {
                        'idea': '💡',
                        'design': '📐',
                        'build': '🔨',
                        'test': '🧪',
                        'deploy': '🚀',
                        'complete': '✅'
                    }
                    return f"[green]{phase_icons.get(phase, '📁')} {phase}[/green]"
        except Exception:
            pass
        return "[dim]📁 initialized[/dim]"
    
    async def create_new_project(self):
        """Create a new project with Genesis workflow"""
        console.print("\n[bold cyan]💡 Genesis: Create New Project[/bold cyan]")
        console.print("[dim]Let's transform your idea into reality[/dim]\n")
        
        # Get the idea first
        console.print("[yellow]First, tell me your idea:[/yellow]")
        idea = console.input("[dim](e.g., 'I want to build a task management app')[/dim]\n> ")
        
        if not idea.strip():
            console.print("[red]Every genesis needs an idea. Please try again.[/red]")
            return
        
        # Generate project name from idea
        suggested_name = self.generate_project_name(idea)
        
        # Get project name
        name_input = console.input(f"\nProject name [{suggested_name}]: ")
        name = name_input.strip() if name_input.strip() else suggested_name
        
        # Get project path
        default_path = Path.home() / "projects" / name
        path_input = console.input(f"Project location [{default_path}]: ")
        project_path = Path(path_input) if path_input.strip() else default_path
        
        # Create directory
        try:
            project_path.mkdir(parents=True, exist_ok=True)
            self.project_path = project_path
            
            # Show animated building status
            await self.animated_status.animate_building(2.0, "Creating your Genesis workspace...")
            
            # Create .abov3 directory structure
            abov3_dir = project_path / ".abov3"
            abov3_dir.mkdir(exist_ok=True)
            
            # Create subdirectories
            for subdir in ['agents', 'sessions', 'history', 'genesis_flow', 'tasks', 'permissions', 'dependencies', 'context']:
                (abov3_dir / subdir).mkdir(exist_ok=True)
            
            # Create genesis metadata
            genesis_data = {
                'idea': idea,
                'name': name,
                'created': datetime.now().isoformat(),
                'current_phase': 'idea',
                'phases': {
                    'idea': {'status': 'complete', 'timestamp': datetime.now().isoformat()},
                    'design': {'status': 'pending'},
                    'build': {'status': 'pending'},
                    'test': {'status': 'pending'},
                    'deploy': {'status': 'pending'}
                }
            }
            
            # Save genesis data
            import yaml
            genesis_file = abov3_dir / 'genesis.yaml'
            with open(genesis_file, 'w') as f:
                yaml.dump(genesis_data, f, default_flow_style=False)
            
            # Save idea as markdown
            idea_file = abov3_dir / 'genesis_flow' / 'idea.md'
            with open(idea_file, 'w') as f:
                f.write(f"# Project Genesis: {name}\n\n")
                f.write(f"## Original Idea\n{idea}\n\n")
                f.write(f"## Created\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Create basic project configuration
            project_config = {
                'name': name,
                'version': '0.1.0',
                'created': datetime.now().isoformat(),
                'genesis': True,
                'idea': idea
            }
            
            config_file = abov3_dir / 'project.yaml'
            with open(config_file, 'w') as f:
                yaml.dump(project_config, f, default_flow_style=False)
            
            # Add to registry
            self.registry.add_project({
                'name': name,
                'path': str(project_path),
                'idea': idea,
                'genesis': True
            })
            
            # Show animated project creation success
            await self.animated_status.show_completion_celebration(
                f"Genesis initiated for '{name}' - Your idea is now reality-bound!"
            )
            console.print(f"[green]✓ Genesis initiated for '{name}'[/green]")
            console.print(f"[green]💡 Idea captured: {idea[:50]}{'...' if len(idea) > 50 else ''}[/green]")
            console.print(f"\n[yellow]Ready to transform your idea into reality![/yellow]")
            
        except Exception as e:
            console.print(f"[red]Genesis failed: {e}[/red]")
            self.project_path = None
    
    async def open_existing_project(self):
        """Open an existing project directory"""
        console.print("\n[cyan]📂 Open Existing Project[/cyan]")
        path_input = console.input("Enter project path: ")
        
        if not path_input.strip():
            console.print("[red]Please provide a project path.[/red]")
            return
        
        project_path = Path(path_input.strip()).expanduser().resolve()
        
        if not project_path.exists():
            console.print(f"[red]Path does not exist: {project_path}[/red]")
            return
        
        if not project_path.is_dir():
            console.print(f"[red]Path is not a directory: {project_path}[/red]")
            return
        
        self.project_path = project_path
        
        # Add to registry if not exists
        self.registry.add_project({
            'name': project_path.name,
            'path': str(project_path),
            'genesis': False
        })
        
        console.print(f"[green]✓ Project opened: {project_path.name}[/green]")
    
    async def browse_for_project(self):
        """Browse for project directory (simplified implementation)"""
        console.print("\n[cyan]🔍 Browse for Project[/cyan]")
        console.print("[dim]Please enter the full path to your project directory:[/dim]")
        
        path_input = console.input("Project path: ")
        if path_input.strip():
            project_path = Path(path_input.strip()).expanduser().resolve()
            if project_path.exists() and project_path.is_dir():
                self.project_path = project_path
                console.print(f"[green]✓ Selected: {project_path.name}[/green]")
            else:
                console.print("[red]Invalid path or directory does not exist.[/red]")
    
    def generate_project_name(self, idea: str) -> str:
        """Generate a project name from an idea"""
        # Simple name generation from idea
        words = idea.lower().split()
        
        # Remove common words
        stop_words = {'i', 'want', 'to', 'build', 'create', 'make', 'a', 'an', 'the', 'app', 'application', 'system', 'tool'}
        words = [w.strip('.,!?;:()[]{}') for w in words if w not in stop_words and w.strip('.,!?;:()[]{}')]
        
        if words:
            # Take first 3 meaningful words
            name_words = words[:3]
            return '-'.join(name_words)
        return 'genesis-project'
    
    async def select_ai_model(self):
        """Select AI model from available Ollama models"""
        from abov3.core.ollama_client import OllamaClient
        
        try:
            # Initialize Ollama client
            ollama = OllamaClient()
            
            # Check if Ollama is available
            if not await ollama.is_available():
                console.print("[yellow]⚠️  Ollama not detected. Using default model configuration.[/yellow]")
                return
            
            # Get available models
            models = await ollama.list_models()
            
            if not models:
                console.print("[yellow]⚠️  No Ollama models found. Please pull a model first.[/yellow]")
                console.print("[dim]Example: ollama pull llama3[/dim]")
                return
            
            # Check if user wants to select a model
            console.print(f"\n[cyan]🤖 Found {len(models)} AI models available[/cyan]")
            
            # Show current default
            console.print(f"[dim]Current: {self.current_model}[/dim]")
            
            # Ask if they want to change
            change_model = console.input("\n[yellow]Would you like to select a different model? [y/N]: [/yellow]")
            
            if change_model.lower() in ['y', 'yes']:
                await self.show_model_selection(models)
                
        except Exception as e:
            console.print(f"[red]Error accessing Ollama models: {e}[/red]")
        finally:
            # Always close the ollama connection
            try:
                await ollama.close()
            except:
                pass  # Ignore cleanup errors
    
    async def show_model_selection(self, models):
        """Display model selection interface"""
        console.print("\n[bold]Available AI Models:[/bold]")
        
        # Create table of models
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=3)
        table.add_column("Model", style="cyan")
        table.add_column("Size", style="green")
        table.add_column("Modified", style="dim")
        
        for i, model in enumerate(models, 1):
            name = model.get('name', 'Unknown')
            size = self.format_model_size(model.get('size', 0))
            modified = self.format_modified_date(model.get('modified_at', ''))
            table.add_row(str(i), name, size, modified)
        
        console.print(table)
        
        # Get user choice
        try:
            choice = console.input(f"\n[yellow]Select model (1-{len(models)}) or Enter to skip: [/yellow]")
            
            if choice.strip():
                model_index = int(choice) - 1
                if 0 <= model_index < len(models):
                    selected_model = models[model_index]
                    model_name = selected_model.get('name', '')
                    
                    # Update the model in agent manager
                    if self.agent_manager and self.agent_manager.current_agent:
                        old_model = self.agent_manager.current_agent.model
                        self.agent_manager.current_agent.model = model_name
                        
                        # Save the updated agent (using the private method)
                        self.agent_manager._save_agent(self.agent_manager.current_agent)
                        self.agent_manager._save_current_agent()
                        
                        console.print(f"[green]✅ Model changed from {old_model} to {model_name}[/green]")
                    
                    # Save model preference to config file
                    self.save_model_config(model_name)
                    
                    # Update assistant default model
                    if hasattr(self, 'assistant') and self.assistant:
                        self.assistant.default_model = model_name
                else:
                    console.print("[red]Invalid selection[/red]")
                    
        except ValueError:
            console.print("[red]Invalid input[/red]")
    
    def format_model_size(self, size_bytes):
        """Format model size in human readable format"""
        if size_bytes == 0:
            return "Unknown"
        
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
    
    def format_modified_date(self, date_str):
        """Format modification date"""
        if not date_str:
            return "Unknown"
        
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return dt.strftime("%Y-%m-%d")
        except:
            return "Unknown"

    async def initialize_project(self):
        """Initialize project with Genesis capabilities"""
        # Show animated thinking status
        await self.animated_status.animate_thinking(2.0, "Initializing Genesis Engine...")
        
        # Ensure .abov3 directory exists
        abov3_dir = self.project_path / '.abov3'
        abov3_dir.mkdir(exist_ok=True)
        
        # Initialize security framework first
        console.print("[dim]🔒 Initializing Enterprise Security Framework...[/dim]")
        try:
            self.security_integration = await initialize_abov3_security(
                self.project_path, 
                SecurityLevel.HIGH
            )
            console.print("[green]✅ Security Framework initialized successfully[/green]")
        except Exception as e:
            console.print(f"[red]⚠️ Security Framework initialization failed: {e}[/red]")
            console.print("[yellow]⚠️ Continuing without security framework (NOT RECOMMENDED)[/yellow]")
        
        # Create subdirectories if they don't exist
        for subdir in ['agents', 'sessions', 'history', 'genesis_flow', 'tasks', 'permissions', 'dependencies', 'context']:
            (abov3_dir / subdir).mkdir(exist_ok=True)
        
        # Initialize Genesis engine
        self.genesis_engine = GenesisEngine(self.project_path)
        
        # Initialize project manager
        self.project_manager = ProjectManager(self.project_path)
        await self.project_manager.initialize()
        
        # Initialize all managers
        self.agent_manager = AgentManager(self.project_path)
        self.task_manager = TaskManager(self.project_path)
        self.genesis_flow = GenesisFlow(self.project_path)
        self.session_manager = SessionManager(self.project_path)
        self.permission_manager = PermissionManager(self.project_path)
        self.dependency_detector = DependencyDetector(self.project_path)
        
        # Load or create Genesis agents with animation
        await self.animated_status.animate_building(1.5, "Loading Genesis agents...")
        await self.load_genesis_agents()
        
        # Select AI model
        await self.select_ai_model()
        
        # Initialize assistant with Genesis capabilities
        self.assistant = Assistant(
            agent=self.agent_manager.current_agent if self.agent_manager else None,
            project_context=self.project_manager.get_context() if self.project_manager else {},
            genesis_engine=self.genesis_engine
        )
        
        # Set the saved AI model as default
        if self.assistant:
            self.assistant.default_model = self.current_model
        
        # Enable automatic agent switching
        if self.assistant and self.agent_manager:
            self.assistant.set_agent_manager(self.agent_manager)
        
        # Command handlers
        self.agent_handler = AgentCommandHandler(self.agent_manager) if self.agent_manager else None
        
        # Check for previous session
        if self.session_manager and self.session_manager.has_previous_session():
            await self.recover_session()
        else:
            # Check if this is a new Genesis project
            genesis_file = self.project_path / '.abov3' / 'genesis.yaml'
            if genesis_file.exists():
                console.print("\n[bold cyan]💡 Genesis Project Detected![/bold cyan]")
                console.print("[yellow]Let's continue transforming your idea into reality.[/yellow]")
            
            if self.session_manager:
                self.session_manager.create_new_session()
    
    async def load_genesis_agents(self):
        """Load Genesis-specific agents"""
        if not self.agent_manager:
            return
            
        # Check if Genesis agents exist
        if not self.agent_manager.has_agent('genesis-architect'):
            console.print("[dim]Loading Genesis agents...[/dim]")
            
            genesis_agents = [
                {
                    'name': 'genesis-architect',
                    'model': 'llama3:latest',
                    'description': 'Transforms ideas into system architecture',
                    'system_prompt': """You are the Genesis Architect. Your role is to transform 
                    ideas into comprehensive system architectures. You excel at:
                    - Breaking down ideas into components
                    - Designing scalable architectures
                    - Creating technical specifications
                    - Planning implementation phases
                    Focus on turning abstract concepts into concrete, buildable designs."""
                },
                {
                    'name': 'genesis-builder',
                    'model': 'codellama:latest',
                    'description': 'Builds complete applications from scratch',
                    'system_prompt': """You are the Genesis Builder. Your mission is to transform 
                    designs and ideas into working code. You specialize in:
                    - Writing production-ready code
                    - Implementing complete features
                    - Creating entire applications from specifications
                    - Following best practices and patterns
                    Your goal is to turn concepts into functional reality."""
                },
                {
                    'name': 'genesis-designer',
                    'model': 'llama3:latest',
                    'description': 'Creates beautiful UIs from descriptions',
                    'system_prompt': """You are the Genesis Designer. You transform UI ideas 
                    into beautiful, functional interfaces. You excel at:
                    - Creating intuitive user experiences
                    - Designing responsive layouts
                    - Implementing modern UI patterns
                    - Making applications visually appealing
                    Focus on creating interfaces that delight users."""
                }
            ]
            
            for agent_config in genesis_agents:
                await self.agent_manager.create_agent_from_config(agent_config)
    
    async def recover_session(self):
        """Recover previous session with Genesis theme"""
        console.print(f"{self.genz.get_status('working')}")
        console.print("[cyan]🔄 Detecting previous session for this project...[/cyan]")
        
        try:
            session_data = await self.session_manager.restore_session()
            
            if session_data:
                message_count = len(session_data.get('messages', []))
                task_count = len(session_data.get('tasks', []))
                agent_name = session_data.get('current_agent', 'default')
                
                console.print(f"\n{self.genz.get_status('success')}")
                console.print(f"[green]✓ Restored {message_count} messages[/green]")
                console.print(f"[cyan]📋 Active tasks: {task_count} tasks loaded[/cyan]")
                console.print(f"[magenta]🤖 Current agent: {agent_name}[/magenta]")
            
        except Exception as e:
            console.print(f"[yellow]⚠️  Could not restore session: {e}[/yellow]")
            console.print("[dim]Starting fresh...[/dim]")
    
    async def run(self):
        """Main REPL loop with Genesis capabilities"""
        
        # Initialize with project selection
        if not await self.initialize():
            return
        
        # Show project info with Genesis status
        await self.show_genesis_status()
        
        # Start background processors
        self.background_tasks.extend([
            asyncio.create_task(self.process_queue()),
            asyncio.create_task(self.auto_save()),
            asyncio.create_task(self.genz_status_rotator())
        ])
        
        # Setup interrupt handler
        def signal_handler(signum, frame):
            self.interrupt_requested = True
        signal.signal(signal.SIGINT, signal_handler)
        
        # Create prompt session
        session = PromptSession()
        
        console.print(f"\n[green]✨ ABOV3 Genesis is ready! Type your ideas or commands.[/green]")
        console.print("[dim]Type '/help' for commands or just start describing what you want to build![/dim]\n")
        
        while True:
            try:
                # Get current status
                if self.agent_manager and self.agent_manager.current_agent:
                    agent_name = self.agent_manager.current_agent.name
                else:
                    agent_name = "default"
                    
                project_name = self.project_path.name
                genesis_phase = self.genesis_flow.get_current_phase() if self.genesis_flow else ""
                status = "🔄" if self.processing else "✓"
                
                # Create Genesis-themed prompt
                if genesis_phase:
                    phase_icon = self.get_phase_icon(genesis_phase)
                    prompt_text = HTML(
                        f'<ansicyan>{status}</ansicyan> '
                        f'<ansigreen>ABOV3 Genesis</ansigreen> '
                        f'[<ansiyellow>{project_name}</ansiyellow>/'
                        f'<ansimagenta>{agent_name}</ansimagenta>/'
                        f'<ansicyan>{phase_icon}</ansicyan>]> '
                    )
                else:
                    prompt_text = HTML(
                        f'<ansicyan>{status}</ansicyan> '
                        f'<ansigreen>ABOV3 Genesis</ansigreen> '
                        f'[<ansiyellow>{project_name}</ansiyellow>/'
                        f'<ansimagenta>{agent_name}</ansimagenta>]> '
                    )
                
                # Get user input
                with patch_stdout():
                    user_input = await session.prompt_async(
                        prompt_text,
                        bottom_toolbar=self.get_toolbar
                    )
                
                if not user_input.strip():
                    continue
                
                # Handle special commands
                if user_input.startswith('/'):
                    await self.handle_command(user_input)
                    continue
                
                # Check for Genesis commands
                if user_input.lower().startswith(('build my idea', 'start genesis', 'transform my idea', 'continue genesis')):
                    await self.start_genesis_workflow()
                    continue
                
                # Process or queue input
                if self.processing:
                    await self.queue_input(user_input)
                else:
                    await self.process_input(user_input)
                    
            except KeyboardInterrupt:
                if self.interrupt_requested:
                    if await self.confirm_exit():
                        break
                    else:
                        self.interrupt_requested = False
            except EOFError:
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
        
        # Cleanup
        await self.cleanup()
    
    async def show_genesis_status(self):
        """Show Genesis project status"""
        genesis_file = self.project_path / '.abov3' / 'genesis.yaml'
        
        if genesis_file.exists():
            try:
                import yaml
                with open(genesis_file) as f:
                    genesis = yaml.safe_load(f)
                
                console.print(f"\n[bold cyan]📁 Genesis Project: {self.project_path.name}[/bold cyan]")
                console.print(f"[dim]Path: {self.project_path}[/dim]")
                
                # Show original idea
                idea = genesis.get('idea', 'No idea recorded')
                console.print(f"\n[yellow]💡 Original Idea:[/yellow]")
                console.print(f"   {idea}")
                
                # Show Genesis progress
                console.print(f"\n[cyan]Genesis Progress:[/cyan]")
                phases = genesis.get('phases', {})
                for phase_name, phase_data in phases.items():
                    icon = self.get_phase_icon(phase_name)
                    status = phase_data.get('status', 'pending')
                    status_icon = '✅' if status == 'complete' else '⏳' if status == 'in_progress' else '⏸️'
                    console.print(f"   {icon} {phase_name.capitalize()}: {status_icon}")
                
                console.print(f"\n[green]Ready to continue your genesis journey![/green]")
            except Exception as e:
                console.print(f"[yellow]⚠️  Could not load Genesis data: {e}[/yellow]")
                console.print(f"\n[bold cyan]📁 Project: {self.project_path.name}[/bold cyan]")
                console.print(f"[dim]Path: {self.project_path}[/dim]\n")
        else:
            console.print(f"\n[bold cyan]📁 Project: {self.project_path.name}[/bold cyan]")
            console.print(f"[dim]Path: {self.project_path}[/dim]\n")
        
        # Show current AI model at the end
        console.print(f"[dim]🤖 AI Model: [cyan]{self.current_model}[/cyan][/dim]\n")
    
    def get_phase_icon(self, phase: str) -> str:
        """Get icon for Genesis phase"""
        icons = {
            'idea': '💡',
            'design': '📐',
            'build': '🔨',
            'test': '🧪',
            'deploy': '🚀',
            'complete': '✅'
        }
        return icons.get(phase.lower(), '📁')
    
    async def start_genesis_workflow(self):
        """Start the Genesis workflow to transform idea into reality"""
        console.print("\n[bold cyan]🌟 Starting Genesis Workflow[/bold cyan]")
        console.print("[yellow]Transforming your idea into built reality...[/yellow]\n")
        
        # Load Genesis data
        genesis_file = self.project_path / '.abov3' / 'genesis.yaml'
        if not genesis_file.exists():
            console.print("[red]No Genesis idea found. Please create a new Genesis project first.[/red]")
            return
        
        try:
            import yaml
            with open(genesis_file) as f:
                genesis = yaml.safe_load(f)
            
            idea = genesis.get('idea')
            current_phase = genesis.get('current_phase', 'idea')
            
            console.print(f"[cyan]💡 Idea:[/cyan] {idea}")
            console.print(f"[cyan]📍 Current Phase:[/cyan] {current_phase}\n")
            
            # Start workflow based on current phase
            if current_phase == 'idea':
                await self.genesis_design_phase(idea)
            elif current_phase == 'design':
                await self.genesis_build_phase()
            elif current_phase == 'build':
                await self.genesis_test_phase()
            elif current_phase == 'test':
                await self.genesis_deploy_phase()
            else:
                console.print("[green]✅ Genesis complete! Your idea is now reality![/green]")
        except Exception as e:
            console.print(f"[red]Error loading Genesis workflow: {e}[/red]")
    
    async def genesis_design_phase(self, idea: str):
        """Genesis Design Phase - Create architecture from idea"""
        await self.animated_status.animate_building(duration=2.0, message="📐 Entering Design Phase...")
        console.print()
        
        # Switch to architect agent if available
        if self.agent_manager and self.agent_manager.has_agent('genesis-architect'):
            await self.agent_manager.switch_agent('genesis-architect')
        
        # Generate architecture
        design_prompt = f"""
        Transform this idea into a complete system architecture:
        {idea}
        
        Provide:
        1. System components
        2. Technology stack
        3. File structure
        4. Database design (if applicable)
        5. API design (if applicable)
        6. Implementation phases
        """
        
        await self.process_input(design_prompt)
        
        # Update Genesis status
        if self.genesis_flow:
            await self.genesis_flow.update_phase('design', 'complete')
            await self.genesis_flow.update_phase('build', 'in_progress')
        
        console.print("\n[green]✅ Design Phase Complete![/green]")
        console.print("[yellow]Ready for Build Phase. Type 'continue genesis' to proceed.[/yellow]")
    
    async def genesis_build_phase(self):
        """Genesis Build Phase - Create actual code"""
        await self.animated_status.animate_building(duration=2.5, message="🔨 Entering Build Phase...")
        console.print()
        
        # Switch to builder agent if available
        if self.agent_manager and self.agent_manager.has_agent('genesis-builder'):
            await self.agent_manager.switch_agent('genesis-builder')
        
        console.print("[yellow]Build phase would implement actual code generation...[/yellow]")
        console.print("[dim]This would use the design specifications to create working code.[/dim]")
        
        # Update Genesis status
        if self.genesis_flow:
            await self.genesis_flow.update_phase('build', 'complete')
            await self.genesis_flow.update_phase('test', 'in_progress')
    
    async def genesis_test_phase(self):
        """Genesis Test Phase - Test the application"""
        console.print(f"{self.genz.get_status('working')}")
        console.print("[cyan]🧪 Entering Test Phase...[/cyan]\n")
        
        console.print("[yellow]Test phase would implement automated testing...[/yellow]")
        console.print("[dim]This would create and run tests for the built application.[/dim]")
        
        # Update Genesis status
        if self.genesis_flow:
            await self.genesis_flow.update_phase('test', 'complete')
            await self.genesis_flow.update_phase('deploy', 'in_progress')
    
    async def genesis_deploy_phase(self):
        """Genesis Deploy Phase - Deploy the application"""
        console.print(f"{self.genz.get_status('working')}")
        console.print("[cyan]🚀 Entering Deploy Phase...[/cyan]\n")
        
        console.print("[yellow]Deploy phase would handle application deployment...[/yellow]")
        console.print("[dim]This would deploy the application to production environments.[/dim]")
        
        # Update Genesis status
        if self.genesis_flow:
            await self.genesis_flow.update_phase('deploy', 'complete')
        
        await self.animated_status.show_completion_celebration("✅ Genesis Complete! Your idea is now reality!")
        console.print("[green]✨ From idea to built reality - mission accomplished! ✨[/green]")
    
    def get_toolbar(self):
        """Get bottom toolbar with Genesis theme"""
        parts = []
        
        if self.processing:
            # Genesis-themed processing messages
            status = random.choice([
                "💭 Manifesting reality",
                "🔥 Forging your vision", 
                "⚡ Genesis in progress",
                "✨ Creating something beautiful",
                "🎮 Building your dream",
                "💅 Making it perfect"
            ])
            parts.append(f"{status}: {self.current_task or 'Processing...'}")
        else:
            parts.append("✓ Ready to build reality")
        
        parts.append(f"Queue: {self.task_queue.qsize()}")
        
        if self.task_manager:
            task_summary = self.task_manager.get_summary()
            if task_summary:
                parts.append(f"Tasks: {task_summary}")
        
        # Add Genesis phase if available
        if self.genesis_flow:
            phase = self.genesis_flow.get_current_phase()
            if phase:
                icon = self.get_phase_icon(phase)
                parts.append(f"{icon} {phase}")
        
        parts.append(f"📁 {self.project_path.name}")
        
        return " | ".join(parts)
    
    async def handle_command(self, command: str):
        """Handle special commands"""
        cmd_parts = command.strip().split()
        cmd = cmd_parts[0].lower()
        
        if cmd == '/help':
            self.show_help()
        elif cmd == '/clear':
            console.clear()
        elif cmd == '/exit':
            if await self.confirm_exit():
                raise KeyboardInterrupt()
        elif cmd == '/genesis':
            await self.show_genesis_status()
        elif cmd == '/vibe':
            console.print(f"\n{self.genz.get_status('success')}")
            console.print("[cyan]You've got this! Let's build something amazing! 🚀[/cyan]")
        elif cmd.startswith('/project'):
            await self.handle_project_command(cmd_parts[1:] if len(cmd_parts) > 1 else [])
        elif cmd.startswith('/agents'):
            await self.handle_agent_command(cmd_parts[1:] if len(cmd_parts) > 1 else [])
        elif cmd.startswith('/model'):
            await self.handle_model_command(cmd_parts[1:] if len(cmd_parts) > 1 else [])
        elif cmd == '/tasks':
            if self.task_manager:
                tasks = self.task_manager.get_active_tasks()
                console.print(f"\n[cyan]Active Tasks:[/cyan]")
                for i, task in enumerate(tasks, 1):
                    console.print(f"  {i}. {task}")
            else:
                console.print("[yellow]Task manager not available[/yellow]")
        elif cmd.startswith('/security'):
            await self.handle_security_command(cmd_parts[1:] if len(cmd_parts) > 1 else [])
        else:
            console.print(f"[red]Unknown command: {command}[/red]")
            console.print("[dim]Type '/help' for available commands[/dim]")
    
    async def handle_project_command(self, args: List[str]):
        """Handle project management commands"""
        if not args:
            # Show current project info
            console.print(f"\n[cyan]Current Project:[/cyan] {self.project_path.name}")
            console.print(f"[dim]Path: {self.project_path}[/dim]")
        elif args[0] == 'list':
            projects = self.registry.get_recent_projects(10)
            console.print(f"\n[cyan]Recent Projects:[/cyan]")
            for i, proj in enumerate(projects, 1):
                status = self.get_project_status(proj)
                console.print(f"  {i}. {proj['name']} {status}")
        elif args[0] == 'switch':
            if len(args) > 1:
                new_path = Path(args[1]).expanduser().resolve()
                if new_path.exists() and new_path.is_dir():
                    console.print(f"[yellow]Switching to project: {new_path.name}[/yellow]")
                    console.print("[dim]Please restart ABOV3 Genesis to switch projects properly.[/dim]")
                else:
                    console.print(f"[red]Project path not found: {new_path}[/red]")
            else:
                console.print("[red]Please provide a project path[/red]")
        else:
            console.print(f"[red]Unknown project command: {args[0]}[/red]")
    
    async def handle_agent_command(self, args: List[str]):
        """Handle agent management commands"""
        if not self.agent_manager:
            console.print("[red]Agent manager not available[/red]")
            return
        
        if not args:
            console.print(f"[cyan]Current Agent:[/cyan] {self.agent_manager.current_agent.name if self.agent_manager.current_agent else 'None'}")
        elif args[0] == 'list':
            agents = self.agent_manager.get_available_agents()
            console.print(f"\n[cyan]Available Agents:[/cyan]")
            for agent in agents:
                current_mark = "→ " if self.agent_manager.current_agent and agent.name == self.agent_manager.current_agent.name else "  "
                console.print(f"{current_mark}{agent.name}: {agent.description}")
        elif args[0] == 'switch':
            if len(args) > 1:
                agent_name = args[1]
                if await self.agent_manager.switch_agent(agent_name):
                    console.print(f"[green]✓ Switched to agent: {agent_name}[/green]")
                else:
                    console.print(f"[red]Agent not found: {agent_name}[/red]")
            else:
                console.print("[red]Please provide an agent name[/red]")
        else:
            console.print(f"[red]Unknown agent command: {args[0]}[/red]")
    
    async def handle_security_command(self, args: List[str]):
        """Handle security management commands"""
        if not self.security_integration:
            console.print("[red]🔒 Security framework not initialized[/red]")
            return
        
        if not args:
            # Show security status
            status = await self.security_integration.get_security_status()
            console.print(f"\n[cyan]🔒 Security Status:[/cyan]")
            console.print(f"Status: [green]{status['overall_status']}[/green]")
            console.print(f"Active Sessions: {status.get('active_sessions', 0)}")
            console.print(f"Blocked IPs: {status.get('blocked_ips', 0)}")
            
        elif args[0] == 'scan':
            scan_type = args[1] if len(args) > 1 else 'full'
            console.print(f"[cyan]🔍 Running {scan_type} vulnerability scan...[/cyan]")
            
            result = await self.security_integration.scan_for_vulnerabilities(scan_type)
            if result.get('success'):
                scan_results = result['results']
                summary = scan_results['summary']
                console.print(f"\n[green]✅ Scan completed[/green]")
                console.print(f"Files scanned: {summary['total_files']}")
                console.print(f"Vulnerabilities found: {summary['total_vulnerabilities']}")
                console.print(f"Critical: {summary['by_severity']['critical']}")
                console.print(f"High: {summary['by_severity']['high']}")
                console.print(f"Medium: {summary['by_severity']['medium']}")
                console.print(f"Low: {summary['by_severity']['low']}")
            else:
                console.print(f"[red]❌ Scan failed: {result.get('error')}[/red]")
                
        elif args[0] == 'report':
            days = int(args[1]) if len(args) > 1 and args[1].isdigit() else 7
            console.print(f"[cyan]📊 Generating security report for last {days} days...[/cyan]")
            
            report = await self.security_integration.generate_security_report(days)
            if 'error' not in report:
                summary = report['summary']
                console.print(f"\n[green]📋 Security Report ({days} days)[/green]")
                console.print(f"Total events: {summary['total_events']}")
                console.print(f"Security events: {summary['security_events']}")
                console.print(f"Failed logins: {summary['failed_authentications']}")
                console.print(f"Successful logins: {summary['successful_authentications']}")
                console.print(f"Permission denials: {summary['permission_denials']}")
                
                if report['recommendations']:
                    console.print(f"\n[yellow]🔍 Recommendations:[/yellow]")
                    for rec in report['recommendations'][:3]:
                        console.print(f"  • {rec}")
            else:
                console.print(f"[red]❌ Report generation failed: {report['error']}[/red]")
                
        elif args[0] == 'status':
            status = await self.security_integration.get_security_status()
            console.print(f"\n[cyan]🔒 Detailed Security Status:[/cyan]")
            console.print(json.dumps(status, indent=2))
            
        else:
            console.print(f"[red]Unknown security command: {args[0]}[/red]")
            console.print("[dim]Available: status, scan [type], report [days][/dim]")
    
    async def handle_model_command(self, args: List[str]):
        """Handle AI model management commands"""
        if not args:
            # Show current model info
            console.print(f"\n[cyan]🤖 Current AI Model:[/cyan] {self.current_model}")
            console.print("[dim]Use '/model switch' to change models[/dim]")
            return
        
        subcommand = args[0].lower()
        
        if subcommand == 'switch' or subcommand == 'select':
            await self.switch_ai_model()
        elif subcommand == 'list':
            await self.list_available_models()
        elif subcommand == 'help':
            self.show_model_help()
        else:
            console.print(f"[red]Unknown model command: {subcommand}[/red]")
            console.print("[dim]Use '/model help' for available commands[/dim]")
    
    async def switch_ai_model(self):
        """Interactive AI model switching"""
        from abov3.core.ollama_client import OllamaClient
        
        try:
            # Show animated thinking while loading models
            await self.animated_status.animate_thinking(1.5, "Loading available AI models...")
            
            ollama = OllamaClient()
            
            if not await ollama.is_available():
                console.print("[red]❌ Ollama is not available. Please start Ollama server first.[/red]")
                console.print("[dim]Run: ollama serve[/dim]")
                return
            
            models = await ollama.list_models()
            
            if not models:
                console.print("[red]❌ No AI models found.[/red]")
                console.print("[dim]Pull a model: ollama pull llama3[/dim]")
                await ollama.close()
                return
            
            # Show current model
            console.print(f"\n[cyan]🤖 Current Model:[/cyan] {self.current_model}")
            
            # Display model selection with animated interface
            await self.animated_status.animate_building(1.0, "Preparing model selection interface...")
            await self.show_model_selection(models)
            
        except Exception as e:
            console.print(f"[red]❌ Error switching models: {e}[/red]")
        finally:
            # Always close the ollama connection
            try:
                await ollama.close()
            except:
                pass  # Ignore cleanup errors
    
    async def list_available_models(self):
        """List all available AI models"""
        from abov3.core.ollama_client import OllamaClient
        
        try:
            ollama = OllamaClient()
            
            if not await ollama.is_available():
                console.print("[red]❌ Ollama is not available[/red]")
                return
            
            models = await ollama.list_models()
            
            if not models:
                console.print("[yellow]⚠️ No models available[/yellow]")
                await ollama.close()
                return
            
            console.print(f"\n[cyan]🤖 Available AI Models ({len(models)} total):[/cyan]")
            
            current_model = self.agent_manager.current_agent.model if self.agent_manager and self.agent_manager.current_agent else None
            
            for i, model in enumerate(models, 1):
                name = model.get('name', 'Unknown')
                size = self.format_model_size(model.get('size', 0))
                modified = self.format_modified_date(model.get('modified_at', ''))
                
                # Highlight current model
                if name == current_model:
                    console.print(f"  {i}. [bold green]✅ {name}[/bold green] ({size}) - [dim]{modified}[/dim] [green]← Current[/green]")
                else:
                    console.print(f"  {i}. {name} ({size}) - [dim]{modified}[/dim]")
            
            console.print(f"\n[dim]Use '/model switch' to change the active model[/dim]")
            
        except Exception as e:
            console.print(f"[red]❌ Error listing models: {e}[/red]")
        finally:
            # Always close the ollama connection
            try:
                await ollama.close()
            except:
                pass  # Ignore cleanup errors
    
    def show_model_help(self):
        """Show model command help"""
        help_text = """
╭─────────────────────────────────────────────────────╮
│         AI Model Management Commands                │
├─────────────────────────────────────────────────────┤
│ /model              - Show current model info       │
│ /model switch       - Switch to different model     │
│ /model select       - Same as switch                │  
│ /model list         - List all available models     │
│ /model help         - Show this help                │
├─────────────────────────────────────────────────────┤
│ Note: Requires Ollama server to be running          │
│ Start with: ollama serve                             │
│ Install models: ollama pull <model-name>            │
╰─────────────────────────────────────────────────────╯
        """
        console.print(help_text)
    
    async def process_input(self, user_input: str):
        """Process user input with Genesis capabilities"""
        if not self.assistant:
            console.print("[red]Assistant not available. Please check initialization.[/red]")
            return
        
        # Security validation if available
        # Only apply security to potentially dangerous inputs
        is_potentially_dangerous = any(keyword in user_input.lower() for keyword in [
            'rm -rf', 'delete system', 'format', 'drop table', 'drop database',
            '../..', 'etc/passwd', 'cmd.exe', 'powershell -c',
            '<script', 'javascript:', 'onclick', 'onerror',
            'exec(', 'eval(', '__import__', 'subprocess',
            'os.system', 'shell=true'
        ])
        
        if self.security_integration and is_potentially_dangerous:
            validation_result = await self.security_integration.secure_user_input(
                user_input, 'text', {'client_id': 'main_interface', 'authenticated': True}
            )
            
            if not validation_result.get('valid', True):
                console.print(f"[red]🔒 Security: Input blocked - {validation_result.get('reason', 'Security violation')}[/red]")
                console.print(f"[yellow]Debug: errors={validation_result.get('errors', [])}[/yellow]")
                return
            
            # Use sanitized input if available
            user_input = validation_result.get('processed_request', {}).get('input', user_input)
        
        self.processing = True
        self.current_task = "Processing your request"
        
        try:
            # Start background animation that runs continuously
            animation_task = asyncio.create_task(
                self.animated_status.start_background_animation("thinking")
            )
            
            # Prepare context for Assistant processing
            context = {
                'project_path': str(self.project_path),
                'agent': self.agent_manager.current_agent if self.agent_manager else None,
                'genesis': await self.genesis_engine.get_genesis_stats() if self.genesis_engine else None,
                'genesis_engine': self.genesis_engine,
                'agent_manager': self.agent_manager
            }
            
            # Process with assistant
            response = await self.assistant.process(user_input, context)
            
            # Stop the background animation
            self.animated_status.stop_background_animation("Processing complete!")
            
            # Cancel the animation task
            animation_task.cancel()
            try:
                await animation_task
            except asyncio.CancelledError:
                pass
            
            # Display response
            console.print(f"\n[bold green]Genesis Response:[/bold green]")
            console.print(response)
            
        except Exception as e:
            # Stop animation on error
            self.animated_status.stop_background_animation("Error occurred")
            
            # Cancel the animation task
            if 'animation_task' in locals():
                animation_task.cancel()
                try:
                    await animation_task
                except asyncio.CancelledError:
                    pass
            
            console.print(f"[red]Error processing request: {e}[/red]")
        finally:
            self.processing = False
            self.current_task = None
    
    async def queue_input(self, user_input: str):
        """Queue input for processing"""
        await self.task_queue.put(user_input)
        console.print(f"[yellow]📋 Request queued. Current queue size: {self.task_queue.qsize()}[/yellow]")
    
    async def process_queue(self):
        """Background task to process queue"""
        while True:
            try:
                if not self.processing and not self.task_queue.empty():
                    user_input = await self.task_queue.get()
                    await self.process_input(user_input)
                await asyncio.sleep(0.1)  # Small delay to prevent busy loop
            except Exception as e:
                console.print(f"[red]Queue processing error: {e}[/red]")
                await asyncio.sleep(1)
    
    async def auto_save(self):
        """Background task for auto-saving"""
        while True:
            try:
                await asyncio.sleep(30)  # Save every 30 seconds
                if self.session_manager:
                    await self.save_current_session()
            except Exception as e:
                # Silent fail for auto-save
                pass
    
    async def genz_status_rotator(self):
        """Background task to rotate GenZ status messages"""
        while True:
            try:
                await asyncio.sleep(5)  # Rotate every 5 seconds when processing
                if self.processing:
                    # Just update the current task description occasionally
                    pass
            except Exception:
                pass
    
    async def save_current_session(self):
        """Save current session state"""
        if self.session_manager:
            try:
                await self.session_manager.save_session()
            except Exception as e:
                # Silent fail for session save
                pass
    
    async def confirm_exit(self) -> bool:
        """Confirm exit with Genesis theme"""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: confirm("Save your genesis and exit?", default=True)
            )
        except:
            return True
    
    def show_help(self):
        """Show help with Genesis theme"""
        # Get terminal width for responsive borders
        width = min(console.size.width - 4, 70)  # Max 70 chars, min leaves padding
        
        # Create responsive borders
        top_border = "╭" + "─" * (width - 2) + "╮"
        bottom_border = "╰" + "─" * (width - 2) + "╯"
        separator = "├" + "─" * (width - 2) + "┤"
        
        def format_line(text):
            """Format a line to fit within borders"""
            if len(text) >= width - 2:
                return f"│ {text[:width-4]}... │"
            else:
                padding = width - len(text) - 3
                return f"│ {text}{' ' * padding}│"
        
        console.print(f"\n{top_border}")
        console.print(format_line("ABOV3 Genesis Commands"))
        console.print(format_line("From Idea to Built Reality"))
        console.print(separator)
        console.print(format_line("Genesis Commands:"))
        console.print(format_line("build my idea    - Start Genesis workflow"))
        console.print(format_line("start genesis    - Transform idea to reality"))
        console.print(format_line("continue genesis - Continue workflow"))
        console.print(format_line(""))
        console.print(format_line("Project Management:"))
        console.print(format_line("/project         - Current project info"))
        console.print(format_line("/project switch  - Switch projects"))
        console.print(format_line("/project list    - Show all projects"))
        console.print(format_line(""))
        console.print(format_line("Agent Management:"))
        console.print(format_line("/agents          - Current agent info"))
        console.print(format_line("/agents list     - View all agents"))
        console.print(format_line("/agents switch   - Switch active agent"))
        console.print(format_line(""))
        console.print(format_line("AI Model Management:"))
        console.print(format_line("/model           - Current model info"))
        console.print(format_line("/model list      - View all models"))
        console.print(format_line("/model switch    - Switch active model"))
        console.print(format_line(""))
        console.print(format_line("Security Commands:"))
        console.print(format_line("/security        - Security status"))
        console.print(format_line("/security scan   - Run vulnerability scan"))
        console.print(format_line("/security report - Generate security report"))
        console.print(format_line(""))
        console.print(format_line("Other Commands:"))
        console.print(format_line("/tasks           - View task progress"))
        console.print(format_line("/genesis         - Show Genesis status"))
        console.print(format_line("/clear           - Clear screen"))
        console.print(format_line("/vibe            - Get motivated"))
        console.print(format_line("/help            - Show this help"))
        console.print(format_line("/exit            - Save and exit"))
        console.print(format_line(""))
        console.print(format_line("Keyboard Shortcuts:"))
        console.print(format_line("Ctrl+C           - Interrupt/Priority"))
        console.print(format_line("Ctrl+D           - Exit"))
        console.print(f"{bottom_border}\n")
    
    async def cleanup(self):
        """Cleanup with Genesis theme"""
        await self.animated_status.animate_status("working", duration=1.0, message="Preserving your genesis...")
        console.print()
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        await self.save_current_session()
        
        # Update registry
        if self.registry and self.project_path:
            self.registry.update_last_accessed(str(self.project_path))
        
        # Update Genesis stats if available
        if self.genesis_flow:
            try:
                stats = self.genesis_flow.get_stats()
                console.print(f"\n[cyan]Genesis Statistics:[/cyan]")
                console.print(f"  Ideas transformed: {stats.get('ideas_completed', 0)}")
                console.print(f"  Lines generated: {stats.get('lines_generated', 0)}")
                console.print(f"  Files created: {stats.get('files_created', 0)}")
            except:
                pass
        
        await self.animated_status.animate_success(duration=2.0, message="✨ Your genesis continues... See you next time! ✨")
        console.print(f"[dim italic]{self.tagline}[/dim italic]")


@click.command()
@click.argument('project_path', required=False, type=click.Path())
@click.option('--new', is_flag=True, help='Start new Genesis project')
@click.option('--agent', default=None, help='Start with specific agent')
@click.option('--idea', default=None, help='Start with an idea')
@click.option('--version', is_flag=True, help='Show version and exit')
def main(project_path, new, agent, idea, version):
    """
    ABOV3 Genesis - From Idea to Built Reality
    
    Transform your ideas into working applications with AI-powered development.
    
    PROJECT_PATH: Optional path to project directory
    """
    
    if version:
        console.print("[cyan]ABOV3 Genesis v1.0.0[/cyan]")
        console.print("[dim]From Idea to Built Reality[/dim]")
        return
    
    # Create ABOV3 Genesis instance
    if project_path:
        app = ABOV3Genesis(Path(project_path))
    else:
        app = ABOV3Genesis()  # Will show project selector
    
    # Store initial idea if provided
    if idea:
        app.initial_idea = idea
    
    # Run main loop
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Genesis paused. Your reality awaits! ✌️[/yellow]")
    except Exception as e:
        console.print(f"[red]Genesis error: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")


if __name__ == "__main__":
    main()