#!/usr/bin/env python3
"""
Quick fix for ABOV3 Genesis freezing issue
"""

import sys
import os
import asyncio
from pathlib import Path

# Set UTF-8 encoding for Windows
if os.name == 'nt':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

def main():
    print("🚀 ABOV3 Genesis - Quick Fix Version")
    print("🔧 This version bypasses potential hanging issues")
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()
    
    # Add the project root to Python path
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    
    try:
        # Import only essential components
        print("📦 Loading essential components...")
        from rich.console import Console
        from rich.panel import Panel
        
        console = Console()
        
        # Show welcome message
        console.print(Panel.fit(
            "[bold cyan]ABOV3 Genesis v1.0.0[/bold cyan]\n"
            "[dim]Enterprise AI Coding Assistant[/dim]\n\n"
            "[yellow]⚠️  Quick Fix Mode Active[/yellow]\n"
            "This version bypasses the main event loop to avoid freezing."
        ), title="🚀 ABOV3 Genesis")
        
        # Simple command loop without async
        console.print("\n💡 Available commands:")
        console.print("  • [cyan]help[/cyan] - Show this help")
        console.print("  • [cyan]version[/cyan] - Show version info")  
        console.print("  • [cyan]status[/cyan] - Show system status")
        console.print("  • [cyan]debug[/cyan] - Run debug analysis")
        console.print("  • [cyan]exit[/cyan] - Exit ABOV3")
        
        while True:
            try:
                user_input = input("\n🤖 ABOV3 Genesis > ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['exit', 'quit', 'q']:
                    console.print("[yellow]👋 Goodbye! Your ideas await...[/yellow]")
                    break
                    
                elif user_input.lower() == 'help':
                    console.print("\n📚 ABOV3 Genesis Help:")
                    console.print("  • [cyan]version[/cyan] - Show version and system info")
                    console.print("  • [cyan]status[/cyan] - Check system status")
                    console.print("  • [cyan]debug[/cyan] - Run diagnostic checks")
                    console.print("  • [cyan]exit[/cyan] - Exit the application")
                    
                elif user_input.lower() == 'version':
                    console.print(f"\n[bold cyan]ABOV3 Genesis v1.0.0[/bold cyan]")
                    console.print(f"[dim]Enterprise AI Coding Assistant[/dim]")
                    console.print(f"[dim]Python: {sys.version}[/dim]")
                    console.print(f"[dim]Platform: {os.name}[/dim]")
                    console.print(f"[dim]Path: {script_dir}[/dim]")
                    
                elif user_input.lower() == 'status':
                    console.print("\n📊 System Status:")
                    
                    # Check basic imports
                    try:
                        import aiohttp
                        console.print("  • [green]✓[/green] aiohttp available")
                    except ImportError:
                        console.print("  • [red]✗[/red] aiohttp not available")
                        
                    try:
                        import yaml
                        console.print("  • [green]✓[/green] pyyaml available")
                    except ImportError:
                        console.print("  • [red]✗[/red] pyyaml not available")
                        
                    # Check ABOV3 modules
                    try:
                        from abov3.core.assistant import Assistant
                        console.print("  • [green]✓[/green] ABOV3 core modules loaded")
                    except ImportError as e:
                        console.print(f"  • [red]✗[/red] ABOV3 core error: {e}")
                        
                elif user_input.lower() == 'debug':
                    console.print("\n🔍 Running diagnostic checks...")
                    
                    # Test imports that might cause hanging
                    problematic_imports = [
                        ('docker', 'Docker SDK'),
                        ('kubernetes', 'Kubernetes client'),
                        ('redis', 'Redis client'),
                        ('ollama', 'Ollama client')
                    ]
                    
                    for module_name, description in problematic_imports:
                        try:
                            __import__(module_name)
                            console.print(f"  • [green]✓[/green] {description} available")
                        except ImportError:
                            console.print(f"  • [yellow]⚠[/yellow] {description} not installed (optional)")
                        except Exception as e:
                            console.print(f"  • [red]✗[/red] {description} error: {e}")
                    
                    console.print("\n🔧 If you're experiencing freezing:")
                    console.print("  1. Install minimal requirements: pip install -r requirements-minimal.txt")
                    console.print("  2. Check if Docker/Kubernetes are causing issues")
                    console.print("  3. Use this quick fix version for basic functionality")
                    
                else:
                    console.print(f"[yellow]Unknown command: {user_input}[/yellow]")
                    console.print("Type [cyan]help[/cyan] for available commands")
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
                continue
            except EOFError:
                break
                
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\n🔧 Please install dependencies:")
        print("pip install rich prompt_toolkit")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()