"""
ABOV3 Genesis - UI Display Manager
Handles terminal UI, formatting, and visual elements
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.layout import Layout
from rich.live import Live
from rich.columns import Columns
from typing import Dict, List, Any, Optional
from pathlib import Path
import asyncio

class UIManager:
    """
    UI Manager for ABOV3 Genesis
    Handles all terminal display, formatting, and visual elements
    """
    
    def __init__(self):
        self.console = Console()
        self.theme_colors = {
            'primary': 'cyan',
            'secondary': 'bright_blue', 
            'success': 'green',
            'warning': 'yellow',
            'error': 'red',
            'accent': 'magenta',
            'dim': 'bright_black'
        }
        
        # Progress tracking
        self.progress_tasks = {}
        self.current_progress = None
    
    def show_banner(self, version: str = "1.0.0", tagline: str = "From Idea to Built Reality"):
        """Show the ABOV3 Genesis banner"""
        banner_text = f"""
╔══════════════════════════════════════════════════════╗
║              ABOV3 Genesis v{version:<6}              ║
║         {tagline:<36}         ║
║                                                       ║
║    ✨ Transform your ideas into working code ✨      ║
║       💡 Idea → 📐 Design → 🔨 Build → 🚀 Ship       ║
╚══════════════════════════════════════════════════════╝
        """.strip()
        
        panel = Panel(
            Text(banner_text, style=self.theme_colors['primary'], justify="center"),
            border_style=self.theme_colors['secondary'],
            padding=(1, 2)
        )
        self.console.print(panel)
    
    def show_project_status(self, project_info: Dict[str, Any]):
        """Display project status information"""
        name = project_info.get('name', 'Unknown Project')
        path = project_info.get('path', '')
        genesis_info = project_info.get('genesis', {})
        
        # Create project info table
        table = Table(title=f"📁 Project: {name}", show_header=False, box=None)
        table.add_column("Field", style=self.theme_colors['dim'])
        table.add_column("Value")
        
        table.add_row("Path", str(path))
        
        if genesis_info:
            idea = genesis_info.get('idea', 'No idea recorded')
            current_phase = genesis_info.get('current_phase', 'idea')
            
            table.add_row("Type", "Genesis Project")
            table.add_row("Original Idea", idea[:80] + "..." if len(idea) > 80 else idea)
            table.add_row("Current Phase", f"{self.get_phase_icon(current_phase)} {current_phase}")
        
        self.console.print(table)
    
    def show_genesis_progress(self, phases: Dict[str, Dict[str, Any]]):
        """Display Genesis workflow progress"""
        self.console.print("\n[cyan]Genesis Progress:[/cyan]")
        
        phase_order = ['idea', 'design', 'build', 'test', 'deploy']
        
        for phase_name in phase_order:
            phase_data = phases.get(phase_name, {})
            status = phase_data.get('status', 'pending')
            icon = self.get_phase_icon(phase_name)
            status_icon = self.get_status_icon(status)
            
            # Color based on status
            color = {
                'complete': 'green',
                'in_progress': 'yellow', 
                'pending': 'dim',
                'failed': 'red'
            }.get(status, 'dim')
            
            self.console.print(
                f"   {icon} [{color}]{phase_name.capitalize()}: {status_icon}[/{color}]"
            )
    
    def show_task_list(self, tasks: List[Dict[str, Any]], title: str = "Tasks"):
        """Display a list of tasks with their status"""
        if not tasks:
            self.console.print(f"[dim]No {title.lower()} to show[/dim]")
            return
        
        table = Table(title=title, show_header=True)
        table.add_column("Status", width=8)
        table.add_column("Task", flex=1)
        table.add_column("Progress", width=10)
        
        for task in tasks:
            status = task.get('status', 'pending')
            name = task.get('name', 'Unnamed task')
            progress = task.get('progress', 0)
            
            status_icon = self.get_status_icon(status)
            progress_bar = self.create_mini_progress_bar(progress)
            
            table.add_row(status_icon, name, progress_bar)
        
        self.console.print(table)
    
    def show_agent_list(self, agents: List[Dict[str, Any]], current_agent: str = None):
        """Display available agents"""
        if not agents:
            self.console.print("[dim]No agents available[/dim]")
            return
        
        table = Table(title="Available Agents", show_header=True)
        table.add_column("Active", width=6)
        table.add_column("Name")
        table.add_column("Description", flex=1)
        table.add_column("Model", width=15)
        
        for agent in agents:
            name = agent.get('name', 'Unknown')
            description = agent.get('description', 'No description')
            model = agent.get('model', 'Unknown')
            
            active_marker = "🤖" if name == current_agent else ""
            
            table.add_row(active_marker, name, description, model)
        
        self.console.print(table)
    
    def show_error(self, error: str, details: str = None):
        """Display an error message"""
        error_panel = Panel(
            f"[red]{error}[/red]" + (f"\n\n[dim]{details}[/dim]" if details else ""),
            title="❌ Error",
            border_style="red",
            padding=(1, 2)
        )
        self.console.print(error_panel)
    
    def show_success(self, message: str, details: str = None):
        """Display a success message"""
        success_panel = Panel(
            f"[green]{message}[/green]" + (f"\n\n[dim]{details}[/dim]" if details else ""),
            title="✅ Success", 
            border_style="green",
            padding=(1, 2)
        )
        self.console.print(success_panel)
    
    def show_warning(self, message: str, details: str = None):
        """Display a warning message"""
        warning_panel = Panel(
            f"[yellow]{message}[/yellow]" + (f"\n\n[dim]{details}[/dim]" if details else ""),
            title="⚠️ Warning",
            border_style="yellow", 
            padding=(1, 2)
        )
        self.console.print(warning_panel)
    
    def show_info(self, message: str, details: str = None):
        """Display an info message"""
        info_panel = Panel(
            f"[cyan]{message}[/cyan]" + (f"\n\n[dim]{details}[/dim]" if details else ""),
            title="ℹ️ Info",
            border_style="cyan",
            padding=(1, 2)
        )
        self.console.print(info_panel)
    
    def create_progress_bar(self, description: str = "Processing") -> Progress:
        """Create a progress bar for long-running operations"""
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console
        )
        return progress
    
    def show_project_selection(self, recent_projects: List[Dict[str, Any]]):
        """Display project selection UI"""
        self.console.print("\n[bold]Choose your genesis path:[/bold]\n")
        
        options = []
        
        if recent_projects:
            self.console.print("[cyan]📚 Continue Building (Recent Projects):[/cyan]")
            for i, proj in enumerate(recent_projects, 1):
                status = self.get_project_visual_status(proj)
                self.console.print(f"  {i}. {proj['name']} {status}")
                self.console.print(f"     [dim]{proj['path']}[/dim]")
                options.append(('recent', proj['path']))
            
            next_num = len(recent_projects) + 1
        else:
            next_num = 1
        
        self.console.print(f"\n[cyan]🌟 Start Fresh:[/cyan]")
        self.console.print(f"  {next_num}. 💡 Create new project (Start your genesis)")
        options.append(('new', None))
        
        self.console.print(f"  {next_num + 1}. 📂 Open existing project")
        options.append(('existing', None))
        
        self.console.print(f"  {next_num + 2}. 🔍 Browse for project directory")
        options.append(('browse', None))
        
        self.console.print(f"  {next_num + 3}. ❌ Exit")
        options.append(('exit', None))
        
        return options
    
    def show_help(self):
        """Display help information"""
        help_content = """
[bold cyan]Genesis Commands:[/bold cyan]
• build my idea    - Start Genesis workflow
• start genesis    - Transform idea to reality  
• continue genesis - Continue workflow

[bold cyan]Project Management:[/bold cyan]
• /project         - Current project info
• /project switch  - Switch projects
• /project list    - Show all projects

[bold cyan]Agent Management:[/bold cyan]
• /agents          - Current agent info
• /agents list     - View all agents
• /agents switch   - Switch active agent

[bold cyan]Other Commands:[/bold cyan]
• /tasks           - View task progress
• /genesis         - Show Genesis status
• /clear           - Clear screen
• /vibe            - Get motivated
• /help            - Show this help
• /exit            - Save and exit

[bold cyan]Keyboard Shortcuts:[/bold cyan]
• Ctrl+C           - Interrupt/Priority
• Ctrl+D           - Exit
        """
        
        help_panel = Panel(
            help_content,
            title="ABOV3 Genesis - Commands & Help",
            title_align="center",
            border_style=self.theme_colors['secondary'],
            padding=(1, 2)
        )
        self.console.print(help_panel)
    
    def get_phase_icon(self, phase: str) -> str:
        """Get icon for a Genesis phase"""
        icons = {
            'idea': '💡',
            'design': '📐',
            'build': '🔨',
            'test': '🧪',
            'deploy': '🚀',
            'complete': '✅'
        }
        return icons.get(phase.lower(), '📁')
    
    def get_status_icon(self, status: str) -> str:
        """Get icon for a status"""
        icons = {
            'complete': '✅',
            'in_progress': '⏳',
            'pending': '⏸️',
            'failed': '❌',
            'skipped': '⏭️'
        }
        return icons.get(status.lower(), '⏸️')
    
    def get_project_visual_status(self, project: dict) -> str:
        """Get visual status for a project in selection"""
        try:
            genesis_file = Path(project['path']) / '.abov3' / 'genesis.yaml'
            if genesis_file.exists():
                import yaml
                with open(genesis_file) as f:
                    genesis = yaml.safe_load(f)
                    phase = genesis.get('current_phase', 'idea')
                    phase_icon = self.get_phase_icon(phase)
                    return f"[green]{phase_icon} {phase}[/green]"
        except Exception:
            pass
        return "[dim]📁 initialized[/dim]"
    
    def create_mini_progress_bar(self, progress: float) -> str:
        """Create a mini text-based progress bar"""
        width = 8
        filled = int(width * (progress / 100))
        bar = "█" * filled + "░" * (width - filled)
        return f"[green]{bar}[/green] {progress:.0f}%"
    
    def show_genesis_stats(self, stats: Dict[str, Any]):
        """Display Genesis statistics"""
        table = Table(title="Genesis Statistics", show_header=False)
        table.add_column("Metric", style=self.theme_colors['dim'])
        table.add_column("Value")
        
        for key, value in stats.items():
            # Format key nicely
            formatted_key = key.replace('_', ' ').title()
            table.add_row(formatted_key, str(value))
        
        self.console.print(table)
    
    def clear_screen(self):
        """Clear the console screen"""
        self.console.clear()
    
    def print(self, *args, **kwargs):
        """Wrapper for console.print"""
        return self.console.print(*args, **kwargs)
    
    def input(self, prompt: str = "> ") -> str:
        """Wrapper for console.input"""
        return self.console.input(prompt)
    
    async def show_live_progress(self, tasks: List[str], update_callback: callable = None):
        """Show live updating progress for multiple tasks"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console
        ) as progress:
            
            # Create tasks
            task_ids = []
            for task_desc in tasks:
                task_id = progress.add_task(task_desc, total=100)
                task_ids.append(task_id)
            
            # Update progress
            if update_callback:
                await update_callback(progress, task_ids)
    
    def create_layout(self) -> Layout:
        """Create a rich layout for complex displays"""
        layout = Layout()
        return layout
    
    def show_columns(self, items: List[Any], title: str = None):
        """Display items in columns"""
        if title:
            self.console.print(f"\n[bold]{title}[/bold]")
        
        columns = Columns(items, equal=True, expand=True)
        self.console.print(columns)