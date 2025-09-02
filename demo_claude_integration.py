#!/usr/bin/env python3
"""
ABOV3 Genesis - Claude Integration Demo
Demonstrates the three core Claude Coder-level features
"""

import asyncio
import sys
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from abov3.core.claude_integration import (
    ClaudeIntegration,
    ClaudeIntegrationConfig,
    initialize_claude_integration
)
from abov3.core.memory_manager import MemoryType, Priority

async def demo_memory_system():
    """Demonstrate the advanced memory system"""
    print("🧠 MEMORY SYSTEM DEMO")
    print("=" * 50)
    
    # Initialize integration
    project_path = Path.cwd() / "demo_project"
    project_path.mkdir(exist_ok=True)
    
    config = ClaudeIntegrationConfig(
        max_memory_mb=100,
        context_window_size=10000,
        debug_mode=True
    )
    
    integration = await initialize_claude_integration(project_path, config)
    memory = integration.memory_manager
    
    print("📝 Storing conversation history...")
    
    # Store some conversation history
    conversations = [
        ("How do I create a Python web app?", "You can use Flask or Django..."),
        ("What's the best way to handle databases?", "SQLAlchemy is a great choice..."),
        ("How do I deploy to production?", "Consider using Docker and AWS..."),
        ("Can you help with error handling?", "Use try-except blocks and logging..."),
        ("What about testing?", "pytest is the standard for Python testing...")
    ]
    
    for i, (user_input, assistant_response) in enumerate(conversations, 1):
        await integration.store_conversation_context(
            user_input, 
            assistant_response,
            {"session_id": "demo", "turn": i}
        )
        print(f"  ✅ Stored conversation turn {i}")
    
    # Store code context
    print("\n💾 Storing code context...")
    await memory.store(
        {
            'file_path': 'app.py',
            'code': 'from flask import Flask\napp = Flask(__name__)',
            'language': 'python',
            'purpose': 'main application file'
        },
        MemoryType.CODE_CONTEXT,
        Priority.HIGH,
        tags={'python', 'flask', 'main'}
    )
    
    # Demonstrate intelligent retrieval
    print("\n🔍 Retrieving relevant context...")
    context = await integration.retrieve_relevant_context("flask web app", limit=3)
    
    for item in context:
        print(f"  📄 {item['type']}: {str(item['content'])[:50]}...")
    
    # Show memory stats
    stats = memory.get_memory_stats()
    print(f"\n📊 Memory Statistics:")
    print(f"  Total Entries: {stats['total_entries']}")
    print(f"  Memory Usage: {stats['memory_usage_mb']:.1f} MB")
    print(f"  Session Duration: {stats['session_duration_hours']:.1f} hours")
    
    await integration.shutdown()
    print("✅ Memory system demo completed!")

async def demo_feedback_system():
    """Demonstrate the write-run-debug feedback loop"""
    print("\n🔄 FEEDBACK SYSTEM DEMO")
    print("=" * 50)
    
    # Initialize integration
    project_path = Path.cwd() / "demo_project"
    project_path.mkdir(exist_ok=True)
    
    config = ClaudeIntegrationConfig(
        max_execution_time=10.0,
        max_iterations=3,
        debug_mode=True
    )
    
    integration = await initialize_claude_integration(project_path, config)
    feedback = integration.feedback_loop
    
    # Create test files with different scenarios
    test_files = []
    
    # 1. Successful script
    success_file = project_path / "success_demo.py"
    with open(success_file, 'w') as f:
        f.write('''#!/usr/bin/env python3
print("🎯 ABOV3 Genesis Feedback Demo")
print("This script runs successfully!")

# Simple calculation
result = 10 * 5
print(f"10 * 5 = {result}")

# Success message
print("✅ Demo completed successfully!")
''')
    test_files.append(("Successful Script", success_file))
    
    # 2. Script with syntax error (to demonstrate error detection)
    error_file = project_path / "error_demo.py"
    with open(error_file, 'w') as f:
        f.write('''#!/usr/bin/env python3
print("🚨 This script has a syntax error")

# Missing closing parenthesis
print("This will cause a syntax error"
result = 5 + 5
print(f"Result: {result}")
''')
    test_files.append(("Script with Error", error_file))
    
    # 3. Script with import error
    import_error_file = project_path / "import_demo.py"
    with open(import_error_file, 'w') as f:
        f.write('''#!/usr/bin/env python3
import nonexistent_module  # This will fail

print("This won't be reached")
''')
    test_files.append(("Import Error Script", import_error_file))
    
    # Execute each test file
    for test_name, file_path in test_files:
        print(f"\n🧪 Testing: {test_name}")
        
        try:
            # Execute with feedback
            result = await integration.execute_with_feedback(
                file_path,
                auto_iterate=False,
                progress_callback=lambda msg, progress: print(f"  📊 {msg} ({progress:.0%})")
            )
            
            print(f"  📈 Result: {'✅ Success' if result['success'] else '❌ Failed'}")
            print(f"  🔄 Iterations: {result['iterations']}")
            print(f"  ⏱️  Total Time: {result['total_time']:.2f}s")
            
            if result.get('improvements'):
                print(f"  🔧 Improvements Made: {len(result['improvements'])}")
            
            if result.get('error_log'):
                print(f"  ⚠️  Errors: {result['error_log'][0] if result['error_log'] else 'None'}")
                
        except Exception as e:
            print(f"  💥 Execution failed: {e}")
    
    # Show feedback statistics
    stats = feedback.get_cycle_metrics()
    print(f"\n📊 Feedback Loop Statistics:")
    print(f"  Total Executions: {stats['total_executions']}")
    print(f"  Successful Fixes: {stats['successful_fixes']}")
    print(f"  Average Iterations: {stats['average_iterations_to_success']:.1f}")
    
    if stats['common_errors']:
        print(f"  Most Common Errors:")
        for error, count in list(stats['common_errors'].items())[:3]:
            print(f"    • {error}: {count}")
    
    await integration.shutdown()
    print("✅ Feedback system demo completed!")

async def demo_keyboard_integration():
    """Demonstrate keyboard controls integration"""
    print("\n⌨️  KEYBOARD INTEGRATION DEMO")
    print("=" * 50)
    
    # Initialize integration
    project_path = Path.cwd() / "demo_project"
    project_path.mkdir(exist_ok=True)
    
    config = ClaudeIntegrationConfig(
        enable_keyboard_controls=True,
        debug_mode=True
    )
    
    integration = await initialize_claude_integration(project_path, config)
    keyboard = integration.keyboard_handler
    
    if not keyboard:
        print("⚠️  Keyboard controls not available on this platform")
        return
    
    print("🎮 Keyboard controls are active!")
    print("   ESC     → Emergency interrupt")
    print("   Ctrl+T  → Toggle todo list")
    print("   Ctrl+C  → Graceful interrupt")
    
    # Demonstrate operation context tracking
    print("\n🔄 Testing operation context tracking...")
    
    async with integration.operation_context("Demo Operation"):
        print(f"  📍 Active Operation: {integration.active_operation}")
        
        # Simulate some work
        for i in range(5):
            print(f"  ⏳ Working... step {i+1}/5")
            await asyncio.sleep(0.5)
            
            # Check for interrupt
            if keyboard.is_interrupt_requested():
                print("  🚨 Operation interrupted!")
                break
        else:
            print("  ✅ Operation completed normally")
    
    print(f"  📍 Active Operation: {integration.active_operation or 'None'}")
    
    # Show keyboard metrics
    metrics = keyboard.get_performance_metrics()
    print(f"\n📊 Keyboard Handler Metrics:")
    print(f"  Keys Processed: {metrics['keys_processed']}")
    print(f"  Interrupts Handled: {metrics['interrupt_count']}")
    print(f"  Average Response Time: {metrics['average_response_time_ms']:.2f}ms")
    print(f"  Todo Visible: {'✅' if metrics['todo_visible'] else '❌'}")
    
    await integration.shutdown()
    print("✅ Keyboard integration demo completed!")

async def demo_complete_integration():
    """Demonstrate all systems working together"""
    print("\n🎯 COMPLETE INTEGRATION DEMO")
    print("=" * 50)
    
    # Initialize full integration
    project_path = Path.cwd() / "demo_project"
    project_path.mkdir(exist_ok=True)
    
    config = ClaudeIntegrationConfig(
        enable_keyboard_controls=True,
        max_memory_mb=100,
        max_execution_time=10.0,
        background_processing=True,
        debug_mode=True
    )
    
    integration = await initialize_claude_integration(project_path, config)
    
    print("🌟 All systems initialized and integrated!")
    
    # Show system status
    status = integration.get_system_status()
    print(f"\n📊 System Status:")
    print(f"  Initialized: {'✅' if status['initialized'] else '❌'}")
    print(f"  Running: {'✅' if status['running'] else '❌'}")
    print(f"  Active Operation: {status['active_operation'] or 'None'}")
    
    modules = status['modules']
    print(f"\n🔧 Active Modules:")
    print(f"  Keyboard Controls: {'✅' if modules['keyboard'] else '❌'}")
    print(f"  Memory Management: {'✅' if modules['memory'] else '❌'}")
    print(f"  Feedback Loop: {'✅' if modules['feedback'] else '❌'}")
    print(f"  Background Tasks: {status['background_tasks']}")
    
    # Demonstrate integrated workflow
    print(f"\n🔄 Integrated Workflow Demo:")
    
    # 1. Store context in memory
    await integration.store_conversation_context(
        "Create a simple calculator",
        "I'll create a calculator with basic operations",
        {"workflow": "demo"}
    )
    print("  📝 Stored conversation context in memory")
    
    # 2. Create and execute code with feedback
    calc_file = project_path / "calculator_demo.py"
    with open(calc_file, 'w') as f:
        f.write('''#!/usr/bin/env python3
"""Simple Calculator Demo"""

def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        return "Error: Division by zero"
    return a / b

# Demo usage
print("🧮 ABOV3 Genesis Calculator Demo")
print("=" * 30)

# Test operations
print(f"5 + 3 = {add(5, 3)}")
print(f"10 - 4 = {subtract(10, 4)}")
print(f"6 * 7 = {multiply(6, 7)}")
print(f"15 / 3 = {divide(15, 3)}")
print(f"10 / 0 = {divide(10, 0)}")

print("✅ Calculator demo completed!")
''')
    
    print("  📄 Created calculator demo file")
    
    # 3. Execute with integrated feedback
    async with integration.operation_context("Calculator Demo Execution"):
        result = await integration.execute_with_feedback(calc_file, auto_iterate=False)
        
        if result['success']:
            print(f"  ✅ Execution successful in {result['iterations']} iterations")
        else:
            print(f"  ❌ Execution failed after {result['iterations']} iterations")
    
    # 4. Retrieve relevant context
    context = await integration.retrieve_relevant_context("calculator", limit=2)
    print(f"  🔍 Retrieved {len(context)} relevant context items")
    
    # Final performance metrics
    perf = integration.get_performance_metrics()
    print(f"\n📈 Final Performance Metrics:")
    print(f"  Operations Completed: {perf['operations_completed']}")
    print(f"  Memory Operations: {perf['memory_operations']}")
    print(f"  Feedback Cycles: {perf['feedback_cycles']}")
    print(f"  Average Response Time: {perf['average_response_time']:.3f}s")
    print(f"  Uptime: {perf['uptime_seconds']:.1f}s")
    
    await integration.shutdown()
    print("✅ Complete integration demo finished!")

async def main():
    """Run complete demo"""
    print("🎯 ABOV3 GENESIS - CLAUDE CODER INTEGRATION DEMO")
    print("=" * 60)
    print("Demonstrating three critical features:")
    print("1. 🧠 Advanced Memory Management")  
    print("2. 🔄 Write-Run-Debug Feedback Loop")
    print("3. ⌨️  Real-time Keyboard Controls")
    print("4. 🎯 Complete Integration")
    print("=" * 60)
    
    try:
        await demo_memory_system()
        await demo_feedback_system()
        await demo_keyboard_integration()
        await demo_complete_integration()
        
        print("\n🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("🎯 ABOV3 Genesis now has Claude Coder-level capabilities!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
        
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup demo project
        import shutil
        demo_project = Path.cwd() / "demo_project"
        if demo_project.exists():
            shutil.rmtree(demo_project)
            print("🧹 Cleaned up demo files")

if __name__ == "__main__":
    asyncio.run(main())