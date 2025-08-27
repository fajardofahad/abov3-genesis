"""
Demo script for ABOV3 Bug Diagnosis & Fixes Module
Shows comprehensive debugging capabilities
"""

import asyncio
import json
from typing import Dict, Any
from colorama import init, Fore, Style

from abov3.modules.bug_diagnosis import (
    BugDiagnosisEngine,
    DiagnosisRequest,
    DiagnosisMode,
    ErrorType,
    FixStrategy
)

# Initialize colorama for cross-platform colored output
init()

class BugDiagnosisDemo:
    """Demonstration of bug diagnosis capabilities"""
    
    def __init__(self):
        self.engine = BugDiagnosisEngine()
        self.test_cases = self._create_test_cases()
    
    def _create_test_cases(self) -> list:
        """Create various bug scenarios for demonstration"""
        return [
            {
                "name": "Python NullPointerException",
                "request": DiagnosisRequest(
                    error_message="AttributeError: 'NoneType' object has no attribute 'split'",
                    stack_trace="""Traceback (most recent call last):
  File "app.py", line 45, in process_data
    result = data.split(',')
AttributeError: 'NoneType' object has no attribute 'split'""",
                    symptom_description="Application crashes when processing empty API response",
                    code_snippet="""def process_data(data):
    result = data.split(',')  # Line 45
    return [item.strip() for item in result]""",
                    file_path="app.py",
                    line_number=45,
                    mode=DiagnosisMode.RUNTIME_ERROR,
                    language="python",
                    fix_strategy=FixStrategy.OPTIMAL
                )
            },
            {
                "name": "JavaScript Array Index Out of Bounds",
                "request": DiagnosisRequest(
                    error_message="TypeError: Cannot read property 'name' of undefined",
                    stack_trace="""TypeError: Cannot read property 'name' of undefined
    at getUserName (users.js:23:28)
    at processUsers (main.js:45:15)
    at Object.<anonymous> (main.js:67:1)""",
                    symptom_description="Error occurs when accessing user array element",
                    code_snippet="""function getUserName(users, index) {
    return users[index].name;  // Line 23
}""",
                    file_path="users.js",
                    line_number=23,
                    mode=DiagnosisMode.RUNTIME_ERROR,
                    language="javascript",
                    fix_strategy=FixStrategy.SAFE
                )
            },
            {
                "name": "TypeScript Type Mismatch",
                "request": DiagnosisRequest(
                    error_message="error TS2345: Argument of type 'string' is not assignable to parameter of type 'number'.",
                    stack_trace="",
                    symptom_description="Compilation error when building TypeScript project",
                    code_snippet="""function calculateTotal(price: number, quantity: number): number {
    return price * quantity;
}

const total = calculateTotal("10", 5);  // Error here""",
                    file_path="calculator.ts",
                    line_number=5,
                    mode=DiagnosisMode.COMPILATION_ERROR,
                    language="typescript",
                    fix_strategy=FixStrategy.OPTIMAL
                )
            },
            {
                "name": "Performance Issue - Memory Leak",
                "request": DiagnosisRequest(
                    error_message="",
                    symptom_description="Application memory usage grows continuously, eventually causing OutOfMemoryError",
                    code_snippet="""class EventManager:
    def __init__(self):
        self.listeners = []
    
    def add_listener(self, listener):
        self.listeners.append(listener)  # Never removed
    
    def process_events(self):
        for listener in self.listeners:
            listener.handle()""",
                    file_path="event_manager.py",
                    mode=DiagnosisMode.MEMORY_LEAK,
                    language="python",
                    fix_strategy=FixStrategy.COMPREHENSIVE
                )
            },
            {
                "name": "Race Condition in Async Code",
                "request": DiagnosisRequest(
                    error_message="",
                    symptom_description="Intermittent data corruption when multiple async operations access shared state",
                    code_snippet="""let sharedCounter = 0;

async function incrementCounter() {
    const current = sharedCounter;
    await delay(Math.random() * 100);
    sharedCounter = current + 1;
}

Promise.all([incrementCounter(), incrementCounter(), incrementCounter()]);""",
                    file_path="async_counter.js",
                    mode=DiagnosisMode.RACE_CONDITION,
                    language="javascript",
                    fix_strategy=FixStrategy.OPTIMAL
                )
            },
            {
                "name": "Java NullPointerException",
                "request": DiagnosisRequest(
                    error_message="java.lang.NullPointerException",
                    stack_trace="""Exception in thread "main" java.lang.NullPointerException
    at com.example.UserService.getUserEmail(UserService.java:42)
    at com.example.Main.main(Main.java:15)""",
                    symptom_description="Null pointer when accessing user object",
                    code_snippet="""public String getUserEmail(User user) {
    return user.getEmail().toLowerCase();  // Line 42
}""",
                    file_path="UserService.java",
                    line_number=42,
                    mode=DiagnosisMode.RUNTIME_ERROR,
                    language="java",
                    fix_strategy=FixStrategy.SAFE
                )
            }
        ]
    
    def print_header(self, text: str):
        """Print a formatted header"""
        print(f"\n{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{text:^80}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")
    
    def print_section(self, title: str):
        """Print a section title"""
        print(f"\n{Fore.YELLOW}▶ {title}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'─' * 40}{Style.RESET_ALL}")
    
    def print_step(self, step_num: int, description: str, confidence: str):
        """Print a diagnosis step"""
        confidence_color = {
            "high": Fore.GREEN,
            "medium": Fore.YELLOW,
            "low": Fore.RED,
            "uncertain": Fore.MAGENTA
        }.get(confidence, Fore.WHITE)
        
        print(f"{Fore.BLUE}Step {step_num}:{Style.RESET_ALL} {description}")
        print(f"  Confidence: {confidence_color}{confidence}{Style.RESET_ALL}")
    
    def print_fix(self, fix: Dict[str, Any]):
        """Print a fix suggestion"""
        print(f"\n{Fore.GREEN}✓ Fix Strategy: {fix['strategy']}{Style.RESET_ALL}")
        print(f"  Description: {fix['description']}")
        print(f"  Confidence: {fix['confidence']:.0%}")
        
        if fix.get('code_changes'):
            print(f"\n  {Fore.CYAN}Code Changes:{Style.RESET_ALL}")
            for change in fix['code_changes']:
                print(f"    File: {change['file']}, Line: {change['line']}")
                print(f"    {Fore.RED}- {change['original']}{Style.RESET_ALL}")
                print(f"    {Fore.GREEN}+ {change['fixed']}{Style.RESET_ALL}")
    
    async def run_diagnosis(self, test_case: Dict[str, Any]):
        """Run diagnosis for a single test case"""
        self.print_header(f"Bug Diagnosis: {test_case['name']}")
        
        # Display the problem
        self.print_section("Problem Description")
        request = test_case['request']
        
        if request.error_message:
            print(f"{Fore.RED}Error: {request.error_message}{Style.RESET_ALL}")
        
        if request.symptom_description:
            print(f"Symptom: {request.symptom_description}")
        
        if request.code_snippet:
            print(f"\nCode Snippet:")
            print(f"{Fore.MAGENTA}{request.code_snippet}{Style.RESET_ALL}")
        
        # Run diagnosis
        self.print_section("Running Diagnosis...")
        
        try:
            result = await self.engine.diagnose(request)
            
            if result.success:
                # Display diagnosis steps
                self.print_section("Debugging Process")
                for step in result.diagnosis_steps:
                    self.print_step(
                        step.step_number,
                        step.description,
                        step.confidence.value
                    )
                    print(f"  Action: {step.action_taken}")
                    print(f"  Duration: {step.duration_ms:.0f}ms")
                
                # Display root cause
                self.print_section("Root Cause Analysis")
                print(f"{Fore.RED}Root Cause:{Style.RESET_ALL} {result.root_cause}")
                print(f"Confidence: {result.confidence.value}")
                print(f"Error Type: {result.error_type.value}")
                
                # Display suggested fixes
                if result.suggested_fixes:
                    self.print_section("Suggested Fixes")
                    for i, fix in enumerate(result.suggested_fixes, 1):
                        print(f"\n{Fore.GREEN}Fix #{i}:{Style.RESET_ALL}")
                        print(f"  Strategy: {fix.strategy.value}")
                        print(f"  Description: {fix.description}")
                        print(f"  Confidence: {fix.confidence_score:.0%}")
                        print(f"  Impact: {fix.impact_assessment}")
                        print(f"  Time Estimate: {fix.estimated_time}")
                        
                        if fix.implementation_steps:
                            print(f"\n  Implementation Steps:")
                            for j, step in enumerate(fix.implementation_steps, 1):
                                print(f"    {j}. {step}")
                        
                        if fix.warnings:
                            print(f"\n  {Fore.YELLOW}Warnings:{Style.RESET_ALL}")
                            for warning in fix.warnings:
                                print(f"    ⚠ {warning}")
                
                # Display related issues
                if result.related_issues:
                    self.print_section("Related Issues Found")
                    for issue in result.related_issues[:3]:
                        print(f"  • {issue['pattern']} in {issue['file']}:{issue['line']}")
                        print(f"    Similarity: {issue['similarity']:.0%}")
                
                # Display performance impact
                if result.performance_impact:
                    self.print_section("Performance Impact")
                    print(f"  Severity: {result.performance_impact['severity']}")
                    print(f"  Impact: {result.performance_impact.get('estimated_impact', 'N/A')}")
                
                # Display security implications
                if result.security_implications:
                    self.print_section("Security Implications")
                    print(f"  {Fore.RED}Severity: {result.security_implications['severity']}{Style.RESET_ALL}")
                    print(f"  Type: {result.security_implications['vulnerability_type']}")
                    print(f"  Recommendation: {result.security_implications['recommendation']}")
                
                # Summary
                self.print_section("Summary")
                print(f"Total diagnosis time: {result.total_duration_ms:.0f}ms")
                print(f"Confidence in diagnosis: {result.confidence.value}")
                print(f"Number of fixes generated: {len(result.suggested_fixes)}")
                
            else:
                print(f"{Fore.RED}Diagnosis failed: {result.root_cause}{Style.RESET_ALL}")
        
        except Exception as e:
            print(f"{Fore.RED}Error during diagnosis: {str(e)}{Style.RESET_ALL}")
    
    async def run_all_demos(self):
        """Run all demonstration test cases"""
        self.print_header("ABOV3 Bug Diagnosis & Fixes Module Demo")
        
        print("This demo showcases the advanced debugging capabilities of ABOV3:")
        print("• Error parsing and classification")
        print("• Root cause analysis")
        print("• Execution path tracing")
        print("• Automated fix generation")
        print("• Performance and security impact assessment")
        
        for test_case in self.test_cases:
            await self.run_diagnosis(test_case)
            print(f"\n{Fore.CYAN}{'─' * 80}{Style.RESET_ALL}")
            await asyncio.sleep(1)  # Brief pause between demos
        
        self.print_header("Demo Complete!")
        print("The Bug Diagnosis module provides:")
        print("✓ Multi-language support (Python, JavaScript, TypeScript, Java, Go, etc.)")
        print("✓ Comprehensive error analysis across different error types")
        print("✓ Step-by-step debugging process with confidence levels")
        print("✓ Multiple fix strategies (Safe, Optimal, Quick, Comprehensive)")
        print("✓ Performance and security impact assessment")
        print("✓ Related issue detection")
        print("✓ Integration with Context-Aware module for deep code understanding")

async def main():
    """Main entry point for the demo"""
    demo = BugDiagnosisDemo()
    await demo.run_all_demos()

if __name__ == "__main__":
    print(f"{Fore.GREEN}Starting ABOV3 Bug Diagnosis Demo...{Style.RESET_ALL}\n")
    asyncio.run(main())