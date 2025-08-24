#!/usr/bin/env python3
"""
Test script for animated GenZ status messages and model selection
"""

import asyncio
import sys
import os
from pathlib import Path

# Set UTF-8 encoding for Windows
if os.name == 'nt':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from abov3.ui.genz import GenZStatus, AnimatedStatus
from abov3.core.ollama_client import OllamaClient
from rich.console import Console

console = Console()

async def test_animated_status():
    """Test animated status messages"""
    print("🎬 Testing Animated GenZ Status Messages\n")
    
    # Initialize animated status
    animated = AnimatedStatus(console)
    
    # Test different animation types
    console.print("[bold]Testing thinking animation:[/bold]")
    await animated.animate_thinking(3.0, "Processing your genius idea...")
    
    console.print("\n[bold]Testing building animation:[/bold]")  
    await animated.animate_building(3.0, "Constructing your digital empire...")
    
    console.print("\n[bold]Testing success animation:[/bold]")
    await animated.animate_success(2.0, "Absolutely slayed that implementation!")
    
    console.print("\n[bold]Testing phase transition:[/bold]")
    await animated.animate_phase_transition("idea", "design", 2.5)
    
    console.print("\n[bold]Testing completion celebration:[/bold]")
    await animated.show_completion_celebration("From idea to reality - no cap!")
    
    console.print("\n[bold]Testing progress steps:[/bold]")
    steps = [
        "Analyzing your vision",
        "Designing the architecture", 
        "Building core features",
        "Adding the finishing touches"
    ]
    await animated.animate_progress(steps, "working", 1.5)

async def test_model_selection():
    """Test Ollama model detection and display"""
    print("\n\n🤖 Testing Model Selection Features\n")
    
    try:
        # Test Ollama connection
        ollama = OllamaClient()
        
        if not await ollama.is_available():
            console.print("[yellow]⚠️  Ollama not running - model selection test skipped[/yellow]")
            return
        
        # Get available models
        models = await ollama.list_models()
        
        console.print(f"[green]✅ Ollama detected with {len(models)} models[/green]")
        
        if models:
            console.print("[cyan]Available models:[/cyan]")
            for i, model in enumerate(models[:5], 1):  # Show first 5 models
                name = model.get('name', 'Unknown')
                size = format_size(model.get('size', 0))
                console.print(f"  {i}. [bold]{name}[/bold] ({size})")
        
        await ollama.close()
        
    except Exception as e:
        console.print(f"[red]❌ Model selection test failed: {e}[/red]")

def format_size(size_bytes):
    """Format model size in human readable format"""
    if size_bytes == 0:
        return "Unknown size"
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"

async def main():
    """Run all tests"""
    console.print("[bold magenta]🎭 ABOV3 Genesis - Animation & Model Selection Demo[/bold magenta]\n")
    
    # Test animated status
    await test_animated_status()
    
    # Test model selection
    await test_model_selection()
    
    console.print("\n[bold green]🎉 All demo tests completed![/bold green]")
    console.print("[dim]These features will make ABOV3 Genesis feel alive and engaging![/dim]")

if __name__ == "__main__":
    asyncio.run(main())