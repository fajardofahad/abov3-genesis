"""
ABOV3 Enterprise Debugger - Demonstration
Shows Claude-level debugging capabilities
"""

import sys
import os
import time
import traceback

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abov3.core.enterprise_debugger import (
    EnterpriseDebugEngine,
    get_debug_engine,
    debug,
    analyze_error,
    ask_debug,
    profile
)
from abov3.core.debug_integration import (
    DebugIntegration,
    enable_abov3_debugging,
    debug_command,
    debug_response,
    debug_generated_code
)


def print_section(title: str):
    """Print section header"""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)


def demonstrate_error_analysis():
    """Demonstrate intelligent error analysis"""
    print_section("1. INTELLIGENT ERROR ANALYSIS")
    
    engine = get_debug_engine()
    
    # Example 1: AttributeError
    print("\n[Example: AttributeError on None]")
    try:
        user_data = None
        name = user_data.name
    except AttributeError as e:
        analysis = analyze_error(e)
        
        print(f"Error: {analysis['error_type']}")
        print(f"Severity: {analysis['severity']}/5")
        print(f"Root Cause: {analysis['root_cause']['description']}")
        print(f"Confidence: {analysis['root_cause']['confidence']:.1%}")
        print("\nSolutions:")
        for i, solution in enumerate(analysis['solutions'][:3], 1):
            print(f"  {i}. {solution}")
    
    # Example 2: KeyError
    print("\n[Example: KeyError in dictionary]")
    try:
        config = {'host': 'localhost', 'port': 8080}
        password = config['password']
    except KeyError as e:
        analysis = analyze_error(e)
        
        print(f"Error: {analysis['error_type']}")
        print("\nSuggested fixes:")
        for solution in analysis['solutions'][:2]:
            print(f"  - {solution}")
    
    # Example 3: Complex error with call chain
    print("\n[Example: Complex error with call chain]")
    def function_a():
        return function_b()
    
    def function_b():
        return function_c()
    
    def function_c():
        data = [1, 2, 3]
        return data[10]  # IndexError
    
    try:
        result = function_a()
    except IndexError as e:
        analysis = analyze_error(e)
        
        print("Call chain:")
        for frame in analysis['call_chain']:
            print(f"  {frame['function']} at line {frame['line']}")
        
        print(f"\nCode context:")
        for line in analysis['code_context']['code_snippet']:
            marker = ">>>" if line['is_error_line'] else "   "
            print(f"{marker} {line['line_number']:3}: {line['code']}")


def demonstrate_natural_language_debugging():
    """Demonstrate natural language debug interface"""
    print_section("2. NATURAL LANGUAGE DEBUGGING")
    
    engine = get_debug_engine()
    session = engine.create_debug_session("nl_demo")
    
    # Simulate some issues
    print("\n[Setting up context with performance and error data...]")
    
    # Add performance data
    @profile
    def slow_function():
        time.sleep(0.5)
        total = sum(range(1000000))
        return total
    
    @profile
    def fast_function():
        return 42
    
    # Execute functions to generate data
    slow_function()
    fast_function()
    for _ in range(100):
        fast_function()
    
    # Generate an error
    try:
        x = 1 / 0
    except ZeroDivisionError as e:
        engine.analyze_exception(e)
    
    # Natural language queries
    queries = [
        "Why is my code slow?",
        "What errors have occurred?",
        "How can I optimize performance?",
        "Is there a memory leak?",
        "What are the bottlenecks in my code?"
    ]
    
    print("\n[Natural Language Debugging Session]")
    for query in queries:
        print(f"\nQ: {query}")
        response = ask_debug(query)
        # Show first 300 chars of response
        if len(response) > 300:
            print(f"A: {response[:300]}...")
        else:
            print(f"A: {response}")


def demonstrate_code_debugging():
    """Demonstrate code analysis and debugging"""
    print_section("3. CODE ANALYSIS & DEBUGGING")
    
    # Example code with multiple issues
    problematic_code = """
import pickle
import os

def process_data(data):
    # Inefficient loop
    result = []
    for i in range(len(data)):
        if data[i] == None:  # Should use 'is None'
            continue
        result.append(data[i] ** 2)
    
    unused_variable = 100  # Unused
    
    # Security risk
    user_input = input("Enter command: ")
    eval(user_input)  # Dangerous!
    
    # Poor error handling
    try:
        risky_operation()
    except:  # Bare except
        pass
    
    return result

def calculate_average(numbers):
    total = sum(numbers)
    return total / len(numbers)  # Division by zero risk

class myClass:  # Wrong naming convention
    def __init__(self):
        self.data = []
    
    def MyMethod(self):  # Wrong naming convention
        pass
"""
    
    print("\n[Analyzing problematic code...]")
    analysis = debug(problematic_code)
    
    # Syntax check
    print(f"\nSyntax: {'✓ Valid' if analysis['syntax_check']['valid'] else '✗ Invalid'}")
    
    # Static analysis
    print("\nStatic Analysis:")
    if analysis['static_analysis']['unused_variables']:
        print(f"  Unused variables: {', '.join(analysis['static_analysis']['unused_variables'])}")
    if analysis['static_analysis']['undefined_variables']:
        print(f"  Undefined variables: {', '.join(analysis['static_analysis']['undefined_variables'])}")
    
    # Complexity
    print(f"\nComplexity Metrics:")
    print(f"  Lines of code: {analysis['complexity_analysis']['lines_of_code']}")
    print(f"  Cyclomatic complexity: {analysis['complexity_analysis']['cyclomatic_complexity']}")
    print(f"  Max nesting depth: {analysis['complexity_analysis']['nesting_depth']}")
    
    # Security issues
    print("\nSecurity Analysis:")
    for risk_level in ['high_risk', 'medium_risk', 'low_risk']:
        risks = analysis['security_analysis'][risk_level]
        if risks:
            print(f"  {risk_level.replace('_', ' ').title()}:")
            for risk in risks:
                print(f"    - {risk}")
    
    # Best practices
    if analysis['best_practices']:
        print("\nBest Practice Violations:")
        for violation in analysis['best_practices'][:5]:
            print(f"  - {violation}")
    
    # Potential issues
    if analysis['potential_issues']:
        print("\nPotential Issues:")
        for issue in analysis['potential_issues']:
            print(f"  - {issue['issue']} (severity: {issue['severity']})")
            print(f"    Fix: {issue['suggestion']}")
    
    # Suggestions
    if analysis['suggestions']:
        print("\nImprovement Suggestions:")
        for suggestion in analysis['suggestions'][:5]:
            print(f"  - {suggestion}")


def demonstrate_performance_profiling():
    """Demonstrate performance profiling"""
    print_section("4. PERFORMANCE PROFILING")
    
    engine = get_debug_engine()
    
    # Functions to profile
    @profile
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    
    @profile
    def optimized_fibonacci(n, memo={}):
        if n in memo:
            return memo[n]
        if n <= 1:
            return n
        memo[n] = optimized_fibonacci(n-1, memo) + optimized_fibonacci(n-2, memo)
        return memo[n]
    
    @profile
    def memory_intensive():
        data = []
        for i in range(100000):
            data.append({'id': i, 'value': i * 2})
        return len(data)
    
    print("\n[Profiling functions...]")
    
    # Run functions
    print("  Running fibonacci(20)...")
    fib_result = fibonacci(20)
    
    print("  Running optimized_fibonacci(20)...")
    opt_fib_result = optimized_fibonacci(20)
    
    print("  Running memory_intensive()...")
    mem_result = memory_intensive()
    
    # Get performance report
    report = engine.get_debug_report()
    perf_summary = report['performance_summary']
    
    print("\n[Performance Report]")
    print(f"Total execution time: {perf_summary['total_execution_time']:.3f}s")
    print(f"Total function calls: {perf_summary['total_function_calls']}")
    
    print("\nSlowest Functions:")
    for func in perf_summary['slowest_functions']:
        print(f"  - {func['name']}")
        print(f"    Total time: {func['total_time']:.3f}s")
        print(f"    Calls: {func['calls']}")
        print(f"    Avg time: {func['avg_time']:.6f}s")
    
    print("\nMost Called Functions:")
    for func in perf_summary['most_called_functions']:
        print(f"  - {func['name']}: {func['calls']} calls")
    
    # Ask for optimization suggestions
    print("\n[Optimization Analysis]")
    response = ask_debug("How can I optimize the fibonacci function?")
    print(response[:500] + "..." if len(response) > 500 else response)


def demonstrate_debug_integration():
    """Demonstrate ABOV3 integration features"""
    print_section("5. ABOV3 DEBUG INTEGRATION")
    
    # Enable debugging
    integration = enable_abov3_debugging()
    
    print("\n[Debug Integration Active]")
    print(debug_command("debug status"))
    
    # Debug generated code
    print("\n[Debugging AI-generated code...]")
    generated_code = """
def calculate_statistics(data):
    # TODO: Implement this function
    total = sum(data)
    average = total / len(data)
    
    # Find maximum
    maximum = data[0]
    for i in range(len(data)):  # Could use max()
        if data[i] > maximum:
            maximum = data[i]
    
    unused = 42  # This variable is not used
    
    return {
        'total': total,
        'average': average,
        'max': maximum
    }
"""
    
    code_analysis = debug_generated_code(generated_code)
    print(f"Code Quality Score: {code_analysis['generation_quality']:.2%}")
    
    if code_analysis['ai_specific_issues']:
        print("AI Generation Issues:")
        for issue in code_analysis['ai_specific_issues']:
            print(f"  - {issue}")
    
    # Debug assistant response
    print("\n[Debugging Assistant Response...]")
    prompt = "How do I implement a binary search algorithm?"
    response = "Here's a binary search implementation:\n```python\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n```"
    
    response_analysis = debug_response(prompt, response)
    print(f"Response Quality Score: {response_analysis['quality_score']:.2%}")
    print(f"Prompt Clarity: {response_analysis['prompt_analysis']['clarity_score']:.2%}")
    
    # Process debug commands
    print("\n[Interactive Debug Commands]")
    commands = [
        "debug analyze performance",
        "debug analyze errors",
        "debug report"
    ]
    
    for cmd in commands[:2]:  # Show first 2 for brevity
        print(f"\nCommand: {cmd}")
        result = debug_command(cmd)
        print(result[:300] + "..." if len(result) > 300 else result)
    
    # Export debug data
    print("\n[Exporting debug data...]")
    export_file = integration.export_debug_data()
    print(f"Debug data exported to: {export_file}")


def demonstrate_interactive_debugging():
    """Demonstrate interactive debugging (simplified demo)"""
    print_section("6. INTERACTIVE DEBUGGING CAPABILITIES")
    
    from abov3.core.enterprise_debugger import InteractiveDebugger
    
    debugger = InteractiveDebugger()
    
    print("\n[Setting up breakpoints...]")
    
    # Set breakpoints
    bp1 = debugger.set_breakpoint("example.py", 10, condition="x > 100")
    bp2 = debugger.set_breakpoint("example.py", 20)
    
    print(f"Breakpoint 1: {bp1.file}:{bp1.line} (condition: {bp1.condition})")
    print(f"Breakpoint 2: {bp2.file}:{bp2.line}")
    
    # Add watch expressions
    debugger.watch_expressions.append("x + y")
    debugger.watch_expressions.append("len(data)")
    
    print(f"\nWatch expressions: {debugger.watch_expressions}")
    
    print("\n[Interactive debugging commands available:]")
    commands = [
        "c/continue - Continue execution",
        "s/step - Step into function",
        "n/next - Step over function",
        "l/list - Show code context",
        "p <expr> - Print expression",
        "locals - Show local variables",
        "stack - Show call stack",
        "watch <expr> - Add watch expression",
        "h/help - Show help"
    ]
    
    for cmd in commands:
        print(f"  {cmd}")
    
    print("\n[Note: Full interactive debugging requires active debug session]")


def main():
    """Run all demonstrations"""
    print("\n" + "="*70)
    print(" ABOV3 ENTERPRISE DEBUGGER - COMPREHENSIVE DEMONSTRATION")
    print(" Claude-Level Intelligent Debugging System")
    print("="*70)
    
    try:
        # Run demonstrations
        demonstrate_error_analysis()
        demonstrate_natural_language_debugging()
        demonstrate_code_debugging()
        demonstrate_performance_profiling()
        demonstrate_debug_integration()
        demonstrate_interactive_debugging()
        
        # Summary
        print_section("DEMONSTRATION COMPLETE")
        print("""
The ABOV3 Enterprise Debugger provides:

✓ Intelligent Error Analysis
  - Root cause identification with confidence scoring
  - Context-aware solution generation
  - Error pattern learning and recognition

✓ Natural Language Debugging
  - Ask questions in plain English
  - Get Claude-level explanations
  - Intelligent performance and error analysis

✓ Comprehensive Code Analysis
  - Syntax and static analysis
  - Security vulnerability scanning
  - Complexity metrics and best practices
  - AI-specific issue detection

✓ Performance Profiling
  - Function-level profiling
  - Memory usage tracking
  - Bottleneck identification
  - Optimization suggestions

✓ ABOV3 Integration
  - Seamless integration with assistant
  - Debug command processing
  - Response quality analysis
  - Code generation debugging

✓ Interactive Debugging
  - Breakpoints with conditions
  - Step-through debugging
  - Watch expressions
  - Call stack inspection

This enterprise-grade debugging system matches and exceeds
Claude's analytical capabilities for code debugging.
        """)
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        traceback.print_exc()
        
        # Use our debugger to analyze the error
        print("\n[Using debugger to analyze demonstration error...]")
        analysis = analyze_error(e)
        print(f"Root cause: {analysis['root_cause']['description']}")
        if analysis['solutions']:
            print("Solutions:")
            for solution in analysis['solutions'][:3]:
                print(f"  - {solution}")


if __name__ == "__main__":
    main()