"""
ABOV3 Genesis - Advanced Feedback Loop System
Claude Coder-level write → run → debug cycle with intelligent iteration
"""

import asyncio
import subprocess
import sys
import os
import time
import json
import traceback
import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import threading
import signal
from contextlib import asynccontextmanager

from .memory_manager import MemoryManager, MemoryType, Priority

class ExecutionResult(Enum):
    """Execution result types"""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    INTERRUPTED = "interrupted"
    PARTIAL_SUCCESS = "partial_success"

class FeedbackType(Enum):
    """Types of feedback analysis"""
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    LOGIC_ERROR = "logic_error"
    PERFORMANCE_ISSUE = "performance_issue"
    DEPENDENCY_ERROR = "dependency_error"
    TEST_FAILURE = "test_failure"
    SUCCESS_VALIDATION = "success_validation"

@dataclass
class ExecutionMetrics:
    """Metrics from code execution"""
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    exit_code: int
    stdout: str
    stderr: str
    result: ExecutionResult
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    files_modified: List[str] = field(default_factory=list)
    dependencies_checked: bool = False

@dataclass
class FeedbackAnalysis:
    """Analysis of execution feedback"""
    feedback_type: FeedbackType
    severity: int  # 1-10 scale
    message: str
    suggestions: List[str] = field(default_factory=list)
    code_fixes: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0  # 0-1 scale
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IterationPlan:
    """Plan for the next iteration"""
    changes_needed: List[str]
    files_to_modify: List[str]
    priority: int  # 1-10
    estimated_effort: int  # 1-5 scale
    success_criteria: List[str]
    approach: str
    
class FeedbackLoop:
    """
    Advanced feedback loop system that automates the write → run → debug cycle.
    Provides Claude Coder-level intelligence for iterative improvement.
    """
    
    def __init__(self, project_path: Path, memory_manager: MemoryManager = None):
        self.project_path = project_path
        self.memory_manager = memory_manager
        
        # Execution configuration
        self.max_execution_time = 30.0  # seconds
        self.max_iterations = 10
        self.success_threshold = 0.8  # confidence threshold
        
        # State tracking
        self.current_iteration = 0
        self.execution_history: List[ExecutionMetrics] = []
        self.feedback_history: List[FeedbackAnalysis] = []
        self.improvements_made: List[str] = []
        
        # Pattern recognition for common issues
        self.error_patterns = self._load_error_patterns()
        
        # Performance tracking
        self.metrics = {
            'total_executions': 0,
            'successful_fixes': 0,
            'average_iterations_to_success': 0.0,
            'common_errors': {},
            'time_saved': 0.0
        }
        
        # Interrupt handling
        self.interrupt_requested = False
        self.current_process: Optional[subprocess.Popen] = None
        
        # Callback hooks
        self.on_execution_start: Optional[Callable] = None
        self.on_execution_complete: Optional[Callable] = None
        self.on_feedback_analysis: Optional[Callable] = None
        self.on_iteration_complete: Optional[Callable] = None
    
    def _load_error_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load common error patterns and their solutions"""
        return {
            'syntax_error': {
                'patterns': [
                    r"SyntaxError: (.+)",
                    r"IndentationError: (.+)",
                    r"TabError: (.+)"
                ],
                'solutions': [
                    "Check for missing colons, parentheses, or brackets",
                    "Verify indentation consistency (tabs vs spaces)",
                    "Look for unclosed strings or comments"
                ]
            },
            'import_error': {
                'patterns': [
                    r"ModuleNotFoundError: No module named '(.+)'",
                    r"ImportError: (.+)",
                    r"from (.+) import .+ # ImportError"
                ],
                'solutions': [
                    "Install missing package with pip",
                    "Check if module name is correct",
                    "Verify package is in requirements.txt"
                ]
            },
            'name_error': {
                'patterns': [
                    r"NameError: name '(.+)' is not defined",
                    r"UnboundLocalError: (.+)"
                ],
                'solutions': [
                    "Define the variable before using it",
                    "Check for typos in variable names",
                    "Import required modules/functions"
                ]
            },
            'type_error': {
                'patterns': [
                    r"TypeError: (.+)",
                    r"AttributeError: '(.+)' object has no attribute '(.+)'"
                ],
                'solutions': [
                    "Check data types being passed to functions",
                    "Verify object has the expected attributes/methods",
                    "Add type checking and conversion"
                ]
            },
            'file_error': {
                'patterns': [
                    r"FileNotFoundError: (.+)",
                    r"PermissionError: (.+)",
                    r"IsADirectoryError: (.+)"
                ],
                'solutions': [
                    "Check if file path exists and is correct",
                    "Verify file permissions",
                    "Create directory structure if needed"
                ]
            }
        }
    
    async def execute_code(
        self, 
        file_path: Path, 
        args: List[str] = None,
        env_vars: Dict[str, str] = None,
        timeout: float = None
    ) -> ExecutionMetrics:
        """Execute code file and capture detailed metrics"""
        
        start_time = datetime.now()
        timeout = timeout or self.max_execution_time
        args = args or []
        env_vars = env_vars or {}
        
        # Prepare environment
        env = os.environ.copy()
        env.update(env_vars)
        
        # Determine execution command
        if file_path.suffix == '.py':
            cmd = [sys.executable, str(file_path)] + args
        elif file_path.suffix == '.js':
            cmd = ['node', str(file_path)] + args
        elif file_path.suffix in ['.sh', '.bash']:
            cmd = ['bash', str(file_path)] + args
        else:
            cmd = [str(file_path)] + args
        
        stdout = ""
        stderr = ""
        exit_code = 0
        result = ExecutionResult.SUCCESS
        
        try:
            # Notify execution start
            if self.on_execution_start:
                await self._safe_callback(self.on_execution_start, file_path, cmd)
            
            # Execute with timeout
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self.project_path
            )
            
            self.current_process = process
            
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
                
                stdout = stdout_bytes.decode('utf-8', errors='ignore')
                stderr = stderr_bytes.decode('utf-8', errors='ignore')
                exit_code = process.returncode
                
                if exit_code == 0:
                    result = ExecutionResult.SUCCESS
                else:
                    result = ExecutionResult.ERROR
                    
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                result = ExecutionResult.TIMEOUT
                stderr = f"Execution timed out after {timeout} seconds"
                exit_code = -1
                
        except Exception as e:
            result = ExecutionResult.ERROR
            stderr = f"Execution failed: {str(e)}"
            exit_code = -1
        
        finally:
            self.current_process = None
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
        
        # Create execution metrics
        metrics = ExecutionMetrics(
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            result=result
        )
        
        # Store execution in memory
        if self.memory_manager:
            await self.memory_manager.store(
                {
                    'file_path': str(file_path),
                    'command': ' '.join(cmd),
                    'metrics': {
                        'duration': duration,
                        'exit_code': exit_code,
                        'result': result.value,
                        'stdout_length': len(stdout),
                        'stderr_length': len(stderr)
                    },
                    'timestamp': start_time.isoformat()
                },
                MemoryType.SYSTEM,
                Priority.MEDIUM,
                tags={'execution', 'metrics'}
            )
        
        # Update metrics
        self.metrics['total_executions'] += 1
        self.execution_history.append(metrics)
        
        # Keep history manageable
        if len(self.execution_history) > 50:
            self.execution_history = self.execution_history[-25:]
        
        # Notify execution complete
        if self.on_execution_complete:
            await self._safe_callback(self.on_execution_complete, metrics)
        
        return metrics
    
    async def analyze_feedback(self, metrics: ExecutionMetrics) -> List[FeedbackAnalysis]:
        """Analyze execution results and provide intelligent feedback"""
        
        analyses = []
        
        # Analyze stderr for errors
        if metrics.stderr:
            error_analyses = await self._analyze_errors(metrics.stderr)
            analyses.extend(error_analyses)
        
        # Analyze stdout for warnings or partial success
        if metrics.stdout:
            output_analyses = await self._analyze_output(metrics.stdout)
            analyses.extend(output_analyses)
        
        # Analyze performance
        performance_analyses = await self._analyze_performance(metrics)
        analyses.extend(performance_analyses)
        
        # Check for success patterns
        if metrics.result == ExecutionResult.SUCCESS:
            success_analysis = FeedbackAnalysis(
                feedback_type=FeedbackType.SUCCESS_VALIDATION,
                severity=1,
                message="Code executed successfully",
                confidence=0.9,
                context={'exit_code': metrics.exit_code, 'duration': metrics.duration_seconds}
            )
            analyses.append(success_analysis)
        
        # Store feedback analyses
        self.feedback_history.extend(analyses)
        
        # Notify feedback analysis complete
        if self.on_feedback_analysis:
            await self._safe_callback(self.on_feedback_analysis, analyses)
        
        return analyses
    
    async def _analyze_errors(self, stderr: str) -> List[FeedbackAnalysis]:
        """Analyze error output for common patterns"""
        analyses = []
        
        for error_type, config in self.error_patterns.items():
            for pattern in config['patterns']:
                matches = re.findall(pattern, stderr, re.MULTILINE | re.IGNORECASE)
                
                if matches:
                    # Determine severity based on error type
                    severity = {
                        'syntax_error': 8,
                        'import_error': 7,
                        'name_error': 6,
                        'type_error': 5,
                        'file_error': 6
                    }.get(error_type, 5)
                    
                    analysis = FeedbackAnalysis(
                        feedback_type=getattr(FeedbackType, error_type.upper(), FeedbackType.RUNTIME_ERROR),
                        severity=severity,
                        message=f"{error_type.replace('_', ' ').title()}: {matches[0] if matches else 'Unknown error'}",
                        suggestions=config['solutions'].copy(),
                        confidence=0.8,
                        context={
                            'error_type': error_type,
                            'matches': matches[:3],  # Limit matches
                            'full_stderr': stderr[:500]  # Limit context
                        }
                    )
                    
                    # Generate specific code fixes
                    analysis.code_fixes = await self._generate_code_fixes(error_type, matches, stderr)
                    
                    analyses.append(analysis)
                    break  # Only match first pattern per type
        
        # Update error tracking
        for analysis in analyses:
            error_key = analysis.context.get('error_type', 'unknown')
            self.metrics['common_errors'][error_key] = self.metrics['common_errors'].get(error_key, 0) + 1
        
        return analyses
    
    async def _analyze_output(self, stdout: str) -> List[FeedbackAnalysis]:
        """Analyze stdout for warnings and success indicators"""
        analyses = []
        
        # Look for warning patterns
        warning_patterns = [
            r"WARNING: (.+)",
            r"UserWarning: (.+)",
            r"DeprecationWarning: (.+)",
            r"FutureWarning: (.+)"
        ]
        
        for pattern in warning_patterns:
            matches = re.findall(pattern, stdout, re.MULTILINE)
            if matches:
                analysis = FeedbackAnalysis(
                    feedback_type=FeedbackType.LOGIC_ERROR,
                    severity=3,
                    message=f"Warning detected: {matches[0][:100]}...",
                    suggestions=[
                        "Review warning message and consider fixing",
                        "Warnings may become errors in future versions",
                        "Check documentation for recommended alternatives"
                    ],
                    confidence=0.6,
                    context={'warnings': matches[:5]}
                )
                analyses.append(analysis)
        
        # Look for success indicators
        success_patterns = [
            r"✅|SUCCESS|PASSED|OK|COMPLETE",
            r"All tests passed",
            r"Build successful",
            r"No errors found"
        ]
        
        for pattern in success_patterns:
            if re.search(pattern, stdout, re.IGNORECASE):
                analysis = FeedbackAnalysis(
                    feedback_type=FeedbackType.SUCCESS_VALIDATION,
                    severity=1,
                    message="Success indicators found in output",
                    confidence=0.7,
                    context={'success_pattern': pattern}
                )
                analyses.append(analysis)
                break
        
        return analyses
    
    async def _analyze_performance(self, metrics: ExecutionMetrics) -> List[FeedbackAnalysis]:
        """Analyze execution performance"""
        analyses = []
        
        # Check execution time
        if metrics.duration_seconds > 5.0:  # Slow execution
            severity = min(int(metrics.duration_seconds / 2), 7)
            analysis = FeedbackAnalysis(
                feedback_type=FeedbackType.PERFORMANCE_ISSUE,
                severity=severity,
                message=f"Slow execution: {metrics.duration_seconds:.2f} seconds",
                suggestions=[
                    "Profile code to identify bottlenecks",
                    "Optimize algorithms and data structures",
                    "Consider asynchronous processing for I/O operations",
                    "Add progress indicators for long-running tasks"
                ],
                confidence=0.7,
                context={'duration': metrics.duration_seconds}
            )
            analyses.append(analysis)
        
        # Check for timeout
        if metrics.result == ExecutionResult.TIMEOUT:
            analysis = FeedbackAnalysis(
                feedback_type=FeedbackType.PERFORMANCE_ISSUE,
                severity=9,
                message=f"Execution timed out after {self.max_execution_time} seconds",
                suggestions=[
                    "Increase timeout limit if appropriate",
                    "Optimize code for better performance",
                    "Break down into smaller, faster operations",
                    "Add early termination conditions"
                ],
                confidence=0.9,
                context={'timeout_seconds': self.max_execution_time}
            )
            analyses.append(analysis)
        
        return analyses
    
    async def _generate_code_fixes(
        self, 
        error_type: str, 
        matches: List[str], 
        stderr: str
    ) -> List[Dict[str, Any]]:
        """Generate specific code fixes based on error analysis"""
        fixes = []
        
        if error_type == 'import_error' and matches:
            module_name = matches[0].strip("'\"")
            fixes.append({
                'type': 'install_package',
                'description': f'Install missing package: {module_name}',
                'command': f'pip install {module_name}',
                'confidence': 0.8
            })
            
            fixes.append({
                'type': 'add_import',
                'description': f'Add import statement for {module_name}',
                'code': f'import {module_name}',
                'confidence': 0.7
            })
        
        elif error_type == 'name_error' and matches:
            var_name = matches[0].strip("'\"")
            fixes.append({
                'type': 'define_variable',
                'description': f'Define variable: {var_name}',
                'suggestion': f'Make sure {var_name} is defined before use',
                'confidence': 0.6
            })
        
        elif error_type == 'syntax_error':
            fixes.append({
                'type': 'syntax_fix',
                'description': 'Fix syntax error',
                'suggestions': [
                    'Check for missing colons after if/for/while/def statements',
                    'Verify all parentheses and brackets are properly closed',
                    'Check for consistent indentation'
                ],
                'confidence': 0.5
            })
        
        return fixes
    
    async def create_iteration_plan(self, analyses: List[FeedbackAnalysis]) -> IterationPlan:
        """Create a plan for the next iteration based on feedback"""
        
        if not analyses:
            return IterationPlan(
                changes_needed=[],
                files_to_modify=[],
                priority=1,
                estimated_effort=1,
                success_criteria=["No changes needed"],
                approach="No action required"
            )
        
        # Sort analyses by severity
        analyses.sort(key=lambda x: x.severity, reverse=True)
        
        # Determine changes needed
        changes = []
        files_to_modify = []
        priority = 1
        effort = 1
        
        for analysis in analyses[:3]:  # Focus on top 3 issues
            if analysis.feedback_type == FeedbackType.SYNTAX_ERROR:
                changes.append(f"Fix syntax error: {analysis.message}")
                priority = max(priority, 8)
                effort = max(effort, 2)
            
            elif analysis.feedback_type == FeedbackType.DEPENDENCY_ERROR:
                changes.append(f"Install missing dependencies")
                priority = max(priority, 7)
                effort = max(effort, 1)
            
            elif analysis.feedback_type == FeedbackType.RUNTIME_ERROR:
                changes.append(f"Fix runtime error: {analysis.message}")
                priority = max(priority, 6)
                effort = max(effort, 3)
            
            elif analysis.feedback_type == FeedbackType.PERFORMANCE_ISSUE:
                changes.append(f"Optimize performance: {analysis.message}")
                priority = max(priority, 4)
                effort = max(effort, 4)
            
            # Add specific code fixes
            for fix in analysis.code_fixes:
                if fix.get('type') == 'install_package':
                    changes.append(f"Run: {fix.get('command')}")
        
        # Success criteria
        success_criteria = [
            "Code executes without errors",
            "All critical issues resolved",
            "Performance is acceptable"
        ]
        
        # Determine approach
        approach = "Sequential fix approach: address highest severity issues first"
        if len(analyses) > 5:
            approach = "Batch fix approach: group related issues and fix together"
        
        return IterationPlan(
            changes_needed=changes,
            files_to_modify=files_to_modify,
            priority=priority,
            estimated_effort=effort,
            success_criteria=success_criteria,
            approach=approach
        )
    
    async def execute_feedback_cycle(
        self, 
        file_path: Path,
        max_iterations: int = None,
        auto_fix: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Execute complete feedback cycle with automatic iteration"""
        
        max_iterations = max_iterations or self.max_iterations
        self.current_iteration = 0
        self.interrupt_requested = False
        
        cycle_start = datetime.now()
        results = {
            'success': False,
            'iterations': 0,
            'total_time': 0.0,
            'final_result': None,
            'improvements': [],
            'analyses': [],
            'error_log': []
        }
        
        try:
            for iteration in range(max_iterations):
                if self.interrupt_requested:
                    results['error_log'].append("Cycle interrupted by user")
                    break
                
                self.current_iteration = iteration + 1
                
                if progress_callback:
                    await self._safe_callback(
                        progress_callback, 
                        f"Iteration {self.current_iteration}/{max_iterations}",
                        iteration / max_iterations
                    )
                
                print(f"\n🔄 Feedback Cycle - Iteration {self.current_iteration}/{max_iterations}")
                
                # Execute code
                execution_metrics = await self.execute_code(file_path)
                
                # Analyze feedback
                analyses = await self.analyze_feedback(execution_metrics)
                results['analyses'].extend(analyses)
                
                # Check for success
                if execution_metrics.result == ExecutionResult.SUCCESS:
                    success_analyses = [a for a in analyses if a.feedback_type == FeedbackType.SUCCESS_VALIDATION]
                    if success_analyses and max(a.confidence for a in success_analyses) >= self.success_threshold:
                        results['success'] = True
                        results['final_result'] = execution_metrics
                        print("✅ Success! Code executed successfully.")
                        break
                
                # Create iteration plan
                plan = await self.create_iteration_plan(analyses)
                
                if not plan.changes_needed:
                    print("ℹ️  No changes needed, cycle complete.")
                    break
                
                print(f"📋 Iteration Plan:")
                for i, change in enumerate(plan.changes_needed[:3], 1):
                    print(f"   {i}. {change}")
                
                # Auto-fix if enabled
                if auto_fix:
                    fixes_applied = await self._apply_automatic_fixes(analyses, file_path)
                    results['improvements'].extend(fixes_applied)
                    
                    if fixes_applied:
                        print(f"🔧 Applied {len(fixes_applied)} automatic fixes")
                    else:
                        print("⚠️  No automatic fixes available, manual intervention needed")
                        break
                else:
                    print("🔧 Auto-fix disabled, manual fixes needed")
                    break
                
                # Notify iteration complete
                if self.on_iteration_complete:
                    await self._safe_callback(self.on_iteration_complete, self.current_iteration, plan)
                
                # Brief pause between iterations
                await asyncio.sleep(0.5)
            
            cycle_end = datetime.now()
            results['iterations'] = self.current_iteration
            results['total_time'] = (cycle_end - cycle_start).total_seconds()
            
            # Update metrics
            if results['success']:
                self.metrics['successful_fixes'] += 1
            
            # Update average iterations
            total_cycles = self.metrics['successful_fixes'] + 1
            self.metrics['average_iterations_to_success'] = (
                (self.metrics['average_iterations_to_success'] * (total_cycles - 1) + 
                 self.current_iteration) / total_cycles if total_cycles > 0 else 0
            )
            
            # Store cycle results in memory
            if self.memory_manager:
                await self.memory_manager.store(
                    results,
                    MemoryType.SYSTEM,
                    Priority.HIGH,
                    tags={'feedback_cycle', 'results'},
                    metadata={'file_path': str(file_path)}
                )
            
            return results
            
        except Exception as e:
            results['error_log'].append(f"Cycle error: {str(e)}")
            results['error_log'].append(traceback.format_exc())
            return results
    
    async def _apply_automatic_fixes(
        self, 
        analyses: List[FeedbackAnalysis], 
        file_path: Path
    ) -> List[str]:
        """Apply automatic fixes based on analysis"""
        fixes_applied = []
        
        for analysis in analyses:
            for fix in analysis.code_fixes:
                if fix.get('confidence', 0) >= 0.7:  # High confidence fixes only
                    try:
                        if fix.get('type') == 'install_package':
                            command = fix.get('command')
                            if command:
                                result = await asyncio.create_subprocess_shell(
                                    command,
                                    stdout=asyncio.subprocess.PIPE,
                                    stderr=asyncio.subprocess.PIPE
                                )
                                await result.communicate()
                                if result.returncode == 0:
                                    fixes_applied.append(f"Installed package: {command}")
                        
                        elif fix.get('type') == 'add_import':
                            # Simple import addition (would need more sophisticated implementation)
                            fixes_applied.append(f"Would add import: {fix.get('code')}")
                        
                    except Exception as e:
                        print(f"Warning: Could not apply fix {fix.get('type')}: {e}")
        
        return fixes_applied
    
    async def _safe_callback(self, callback: Callable, *args, **kwargs):
        """Safely execute callback with error handling"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args, **kwargs)
            else:
                callback(*args, **kwargs)
        except Exception as e:
            print(f"Warning: Callback error: {e}")
    
    def interrupt_cycle(self):
        """Interrupt the current feedback cycle"""
        self.interrupt_requested = True
        if self.current_process:
            try:
                self.current_process.terminate()
            except Exception:
                pass
    
    def get_cycle_metrics(self) -> Dict[str, Any]:
        """Get feedback cycle performance metrics"""
        return {
            'total_executions': self.metrics['total_executions'],
            'successful_fixes': self.metrics['successful_fixes'],
            'average_iterations_to_success': self.metrics['average_iterations_to_success'],
            'common_errors': dict(self.metrics['common_errors']),
            'current_iteration': self.current_iteration,
            'execution_history_count': len(self.execution_history),
            'feedback_history_count': len(self.feedback_history),
            'improvements_made_count': len(self.improvements_made)
        }
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights for learning and improvement"""
        insights = {
            'most_common_errors': {},
            'fastest_fixes': [],
            'patterns_identified': [],
            'success_rate': 0.0
        }
        
        # Analyze common errors
        if self.metrics['common_errors']:
            total_errors = sum(self.metrics['common_errors'].values())
            if total_errors > 0:
                insights['most_common_errors'] = {
                    error: (count / total_errors) * 100
                    for error, count in self.metrics['common_errors'].items()
                }
            else:
                insights['most_common_errors'] = {}
        
        # Calculate success rate
        if self.metrics['total_executions'] > 0:
            insights['success_rate'] = (self.metrics['successful_fixes'] / self.metrics['total_executions']) * 100
        
        return insights

# Global feedback loop instance
_global_feedback_loop: Optional[FeedbackLoop] = None

def get_feedback_loop(project_path: Path = None, memory_manager: MemoryManager = None) -> FeedbackLoop:
    """Get the global feedback loop instance"""
    global _global_feedback_loop
    if _global_feedback_loop is None:
        if project_path is None:
            project_path = Path.cwd()
        _global_feedback_loop = FeedbackLoop(project_path, memory_manager)
    return _global_feedback_loop

async def initialize_feedback_system(project_path: Path, memory_manager: MemoryManager = None) -> FeedbackLoop:
    """Initialize the global feedback system"""
    global _global_feedback_loop
    _global_feedback_loop = FeedbackLoop(project_path, memory_manager)
    return _global_feedback_loop