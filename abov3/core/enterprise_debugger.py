"""
ABOV3 Genesis - Enterprise Debugging Engine
Claude-level intelligent debugging system with natural language interface
"""

import sys
import os
import ast
import dis
import time
import json
import inspect
import traceback
import linecache
import threading
import asyncio
import re
import gc
import weakref
import pickle
import hashlib
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, Set
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from functools import wraps, lru_cache
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import logging

# Third-party imports
try:
    import psutil
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    psutil = None
    np = None


class DebugLevel(Enum):
    """Debug levels with intelligence"""
    MINIMAL = 1
    STANDARD = 2
    DETAILED = 3
    EXHAUSTIVE = 4
    CLAUDE_LEVEL = 5  # Maximum intelligence


@dataclass
class Breakpoint:
    """Advanced breakpoint with conditions and actions"""
    file: str
    line: int
    condition: Optional[str] = None
    action: Optional[Callable] = None
    hit_count: int = 0
    ignore_count: int = 0
    enabled: bool = True
    temporary: bool = False
    log_expression: Optional[str] = None
    
    def should_break(self, frame: Any) -> bool:
        """Evaluate if breakpoint should trigger"""
        if not self.enabled:
            return False
        
        if self.ignore_count > 0:
            self.ignore_count -= 1
            return False
        
        if self.condition:
            try:
                return eval(self.condition, frame.f_globals, frame.f_locals)
            except:
                return False
        
        return True


@dataclass
class ErrorPattern:
    """Pattern for error recognition and solutions"""
    pattern: str
    category: str
    severity: int
    solutions: List[str]
    confidence: float = 0.0
    occurrences: int = 0
    last_seen: Optional[datetime] = None


@dataclass
class PerformanceProfile:
    """Performance profiling data"""
    function_name: str
    file_path: str
    line_number: int
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    memory_allocated: int = 0
    memory_freed: int = 0
    memory_peak: int = 0
    
    def update(self, execution_time: float, memory_delta: int):
        """Update profile statistics"""
        self.call_count += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.avg_time = self.total_time / self.call_count
        
        if memory_delta > 0:
            self.memory_allocated += memory_delta
            self.memory_peak = max(self.memory_peak, memory_delta)
        else:
            self.memory_freed += abs(memory_delta)


class IntelligentErrorAnalyzer:
    """Claude-level error analysis with root cause detection"""
    
    def __init__(self):
        self.error_patterns = self._load_error_patterns()
        self.error_history = deque(maxlen=1000)
        self.solution_cache = {}
        self.learning_enabled = True
        
    def _load_error_patterns(self) -> List[ErrorPattern]:
        """Load known error patterns with solutions"""
        patterns = [
            ErrorPattern(
                pattern=r"AttributeError.*'NoneType'.*has no attribute",
                category="null_reference",
                severity=3,
                solutions=[
                    "Check if object is None before accessing attributes",
                    "Add null check: if obj is not None:",
                    "Use optional chaining or getattr() with default",
                    "Verify object initialization in __init__ method",
                    "Check return values from functions that might return None"
                ]
            ),
            ErrorPattern(
                pattern=r"KeyError:.*",
                category="missing_key",
                severity=2,
                solutions=[
                    "Use dict.get(key, default) instead of dict[key]",
                    "Check if key exists: if key in dict:",
                    "Use try-except block to handle missing keys",
                    "Verify data structure before accessing",
                    "Use defaultdict for automatic key creation"
                ]
            ),
            ErrorPattern(
                pattern=r"IndexError.*list index out of range",
                category="index_bounds",
                severity=2,
                solutions=[
                    "Check list length before accessing: if len(list) > index:",
                    "Use try-except to handle index errors",
                    "Use list slicing with bounds checking",
                    "Verify loop conditions and boundaries",
                    "Consider using enumerate() for safer iteration"
                ]
            ),
            ErrorPattern(
                pattern=r"RecursionError.*maximum recursion depth exceeded",
                category="infinite_recursion",
                severity=4,
                solutions=[
                    "Add base case to recursive function",
                    "Check for circular references in data structures",
                    "Use iterative approach instead of recursion",
                    "Increase recursion limit with sys.setrecursionlimit()",
                    "Add memoization to avoid repeated calculations"
                ]
            ),
            ErrorPattern(
                pattern=r"MemoryError",
                category="memory_exhaustion",
                severity=5,
                solutions=[
                    "Process data in smaller chunks",
                    "Use generators instead of lists for large datasets",
                    "Clear unused variables with del statement",
                    "Use memory-efficient data structures",
                    "Enable garbage collection with gc.collect()"
                ]
            ),
            ErrorPattern(
                pattern=r"TypeError.*unexpected keyword argument",
                category="function_signature",
                severity=2,
                solutions=[
                    "Check function signature and parameters",
                    "Remove unexpected keyword arguments",
                    "Use **kwargs to accept arbitrary keywords",
                    "Verify API documentation for correct parameters",
                    "Check for version compatibility issues"
                ]
            ),
            ErrorPattern(
                pattern=r"ImportError.*No module named",
                category="missing_dependency",
                severity=3,
                solutions=[
                    "Install missing package: pip install <package>",
                    "Check virtual environment activation",
                    "Verify PYTHONPATH includes module location",
                    "Check for typos in import statement",
                    "Ensure package compatibility with Python version"
                ]
            ),
            ErrorPattern(
                pattern=r"ZeroDivisionError",
                category="division_by_zero",
                severity=2,
                solutions=[
                    "Check divisor before division: if divisor != 0:",
                    "Use try-except to handle division errors",
                    "Return special value for zero division cases",
                    "Use epsilon comparison for floating point",
                    "Consider using numpy.divide with where parameter"
                ]
            )
        ]
        
        return patterns
    
    def analyze_error(self, exception: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Claude-level error analysis"""
        analysis = {
            'error_type': type(exception).__name__,
            'message': str(exception),
            'severity': self._calculate_severity(exception),
            'confidence': 0.0,
            'root_cause': None,
            'immediate_cause': None,
            'contributing_factors': [],
            'solutions': [],
            'code_context': self._extract_code_context(exception),
            'variable_state': self._capture_variable_state(exception),
            'call_chain': self._analyze_call_chain(exception),
            'similar_errors': self._find_similar_errors(exception),
            'prevention_suggestions': [],
            'learning_points': []
        }
        
        # Pattern matching for known errors
        matched_pattern = self._match_error_pattern(exception)
        if matched_pattern:
            analysis['confidence'] = matched_pattern.confidence
            analysis['solutions'].extend(matched_pattern.solutions)
            matched_pattern.occurrences += 1
            matched_pattern.last_seen = datetime.now()
        
        # Root cause analysis
        analysis['root_cause'] = self._identify_root_cause(exception, context)
        
        # Generate intelligent solutions
        analysis['solutions'].extend(self._generate_solutions(exception, context))
        
        # Prevention suggestions
        analysis['prevention_suggestions'] = self._generate_prevention_suggestions(exception)
        
        # Store in history for learning
        if self.learning_enabled:
            self._learn_from_error(exception, analysis)
        
        return analysis
    
    def _calculate_severity(self, exception: Exception) -> int:
        """Calculate error severity (1-5 scale)"""
        severity_map = {
            'SystemExit': 5,
            'KeyboardInterrupt': 5,
            'MemoryError': 5,
            'RecursionError': 4,
            'SystemError': 4,
            'RuntimeError': 3,
            'ImportError': 3,
            'AttributeError': 2,
            'KeyError': 2,
            'IndexError': 2,
            'TypeError': 2,
            'ValueError': 2,
            'ZeroDivisionError': 2,
            'NameError': 1,
            'SyntaxError': 1
        }
        
        error_type = type(exception).__name__
        return severity_map.get(error_type, 3)
    
    def _extract_code_context(self, exception: Exception, lines_before: int = 5, lines_after: int = 5) -> Dict[str, Any]:
        """Extract code context around error"""
        context = {
            'file': None,
            'line': None,
            'function': None,
            'code_snippet': [],
            'error_line': None
        }
        
        if exception.__traceback__:
            tb = traceback.extract_tb(exception.__traceback__)
            if tb:
                frame = tb[-1]
                context['file'] = frame.filename
                context['line'] = frame.lineno
                context['function'] = frame.name
                
                # Get code snippet
                try:
                    start_line = max(1, frame.lineno - lines_before)
                    end_line = frame.lineno + lines_after + 1
                    
                    for line_no in range(start_line, end_line):
                        line = linecache.getline(frame.filename, line_no)
                        if line:
                            is_error_line = (line_no == frame.lineno)
                            context['code_snippet'].append({
                                'line_number': line_no,
                                'code': line.rstrip(),
                                'is_error_line': is_error_line
                            })
                            if is_error_line:
                                context['error_line'] = line.rstrip()
                except:
                    pass
        
        return context
    
    def _capture_variable_state(self, exception: Exception) -> Dict[str, Any]:
        """Capture variable state at error point"""
        state = {}
        
        if exception.__traceback__:
            frame = exception.__traceback__.tb_frame
            
            # Local variables
            state['locals'] = {}
            for name, value in frame.f_locals.items():
                if not name.startswith('__'):
                    try:
                        state['locals'][name] = {
                            'type': type(value).__name__,
                            'value': str(value)[:200],
                            'repr': repr(value)[:200]
                        }
                    except:
                        state['locals'][name] = {'error': 'Could not serialize'}
            
            # Global variables (filtered)
            state['globals'] = {}
            for name, value in frame.f_globals.items():
                if not name.startswith('__') and not callable(value):
                    try:
                        state['globals'][name] = {
                            'type': type(value).__name__,
                            'value': str(value)[:100]
                        }
                    except:
                        pass
        
        return state
    
    def _analyze_call_chain(self, exception: Exception) -> List[Dict[str, Any]]:
        """Analyze the call chain leading to error"""
        chain = []
        
        if exception.__traceback__:
            for frame_info in traceback.extract_tb(exception.__traceback__):
                chain.append({
                    'file': frame_info.filename,
                    'line': frame_info.lineno,
                    'function': frame_info.name,
                    'code': frame_info.line
                })
        
        return chain
    
    def _match_error_pattern(self, exception: Exception) -> Optional[ErrorPattern]:
        """Match error against known patterns"""
        error_str = f"{type(exception).__name__}: {str(exception)}"
        
        best_match = None
        best_confidence = 0.0
        
        for pattern in self.error_patterns:
            if re.search(pattern.pattern, error_str):
                # Calculate confidence based on pattern specificity
                confidence = len(pattern.pattern) / max(len(error_str), 1)
                confidence = min(confidence * 1.5, 1.0)  # Boost and cap at 1.0
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = pattern
                    best_match.confidence = confidence
        
        return best_match
    
    def _identify_root_cause(self, exception: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Identify root cause using intelligent analysis"""
        root_cause = {
            'identified': False,
            'description': None,
            'confidence': 0.0,
            'evidence': []
        }
        
        # Analyze based on exception type
        if isinstance(exception, AttributeError):
            if "'NoneType'" in str(exception):
                root_cause['identified'] = True
                root_cause['description'] = "Attempting to access attribute on None object"
                root_cause['confidence'] = 0.9
                root_cause['evidence'].append("NoneType detected in error message")
                
                # Find where None originated
                if exception.__traceback__:
                    frame = exception.__traceback__.tb_frame
                    for var_name, var_value in frame.f_locals.items():
                        if var_value is None:
                            root_cause['evidence'].append(f"Variable '{var_name}' is None")
        
        elif isinstance(exception, KeyError):
            key = str(exception).strip("'\"")
            root_cause['identified'] = True
            root_cause['description'] = f"Dictionary missing key: {key}"
            root_cause['confidence'] = 0.95
            root_cause['evidence'].append(f"Key '{key}' not found in dictionary")
        
        elif isinstance(exception, IndexError):
            root_cause['identified'] = True
            root_cause['description'] = "Accessing index beyond list bounds"
            root_cause['confidence'] = 0.9
            
            # Try to find the problematic list
            if exception.__traceback__:
                frame = exception.__traceback__.tb_frame
                for var_name, var_value in frame.f_locals.items():
                    if isinstance(var_value, (list, tuple)):
                        root_cause['evidence'].append(f"List '{var_name}' has length {len(var_value)}")
        
        return root_cause
    
    def _generate_solutions(self, exception: Exception, context: Dict[str, Any]) -> List[str]:
        """Generate intelligent solutions using Claude-level reasoning"""
        solutions = []
        
        # Context-aware solutions
        if isinstance(exception, AttributeError):
            obj_name = self._extract_object_name(str(exception))
            if obj_name:
                solutions.append(f"Add null check: if {obj_name} is not None:")
                solutions.append(f"Use getattr({obj_name}, 'attribute', default_value)")
                solutions.append(f"Initialize {obj_name} properly before use")
        
        elif isinstance(exception, KeyError):
            key = str(exception).strip("'\"")
            solutions.append(f"Use dict.get('{key}', default_value)")
            solutions.append(f"Check existence: if '{key}' in dict:")
            solutions.append(f"Use collections.defaultdict to handle missing keys")
        
        elif isinstance(exception, TypeError):
            if "takes" in str(exception) and "positional argument" in str(exception):
                solutions.append("Check function signature and argument count")
                solutions.append("Use *args to accept variable arguments")
                solutions.append("Verify function call matches definition")
        
        return solutions
    
    def _extract_object_name(self, error_message: str) -> Optional[str]:
        """Extract object name from error message"""
        match = re.search(r"'(\w+)' object has no attribute", error_message)
        if match:
            return match.group(1)
        return None
    
    def _find_similar_errors(self, exception: Exception) -> List[Dict[str, Any]]:
        """Find similar errors from history"""
        similar = []
        current_error = str(exception)
        
        for historic_error in self.error_history:
            similarity = self._calculate_similarity(current_error, historic_error['message'])
            if similarity > 0.7:
                similar.append({
                    'error': historic_error['message'],
                    'similarity': similarity,
                    'timestamp': historic_error['timestamp'],
                    'solution_used': historic_error.get('solution')
                })
        
        return sorted(similar, key=lambda x: x['similarity'], reverse=True)[:5]
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using TF-IDF"""
        if not text1 or not text2:
            return 0.0
        
        # Simple Jaccard similarity as fallback
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _generate_prevention_suggestions(self, exception: Exception) -> List[str]:
        """Generate suggestions to prevent future occurrences"""
        suggestions = []
        
        error_type = type(exception).__name__
        
        if error_type in ['AttributeError', 'KeyError', 'IndexError']:
            suggestions.append("Add input validation at function entry points")
            suggestions.append("Use type hints and static type checking (mypy)")
            suggestions.append("Implement defensive programming patterns")
        
        if error_type in ['MemoryError', 'RecursionError']:
            suggestions.append("Add resource limits and monitoring")
            suggestions.append("Implement circuit breaker pattern")
            suggestions.append("Use streaming/chunking for large data")
        
        if error_type in ['ImportError', 'ModuleNotFoundError']:
            suggestions.append("Use requirements.txt for dependency management")
            suggestions.append("Implement dependency checking at startup")
            suggestions.append("Use virtual environments for isolation")
        
        return suggestions
    
    def _learn_from_error(self, exception: Exception, analysis: Dict[str, Any]):
        """Learn from error for future improvements"""
        error_entry = {
            'message': str(exception),
            'type': type(exception).__name__,
            'timestamp': datetime.now(),
            'analysis': analysis,
            'solution': None  # To be filled when solution is applied
        }
        
        self.error_history.append(error_entry)
        
        # Update pattern confidence based on success
        for pattern in self.error_patterns:
            if pattern.last_seen and (datetime.now() - pattern.last_seen).seconds < 60:
                # Recent pattern match - adjust confidence
                if pattern.occurrences > 10:
                    pattern.confidence = min(pattern.confidence * 1.1, 1.0)


class InteractiveDebugger:
    """Interactive debugging with step-through capabilities"""
    
    def __init__(self):
        self.breakpoints: Dict[str, Breakpoint] = {}
        self.watch_expressions: List[str] = []
        self.call_stack: List[Dict[str, Any]] = []
        self.current_frame = None
        self.stepping_mode = None
        self.debug_session_active = False
        self.command_history = deque(maxlen=100)
        
    def set_breakpoint(self, file: str, line: int, **kwargs) -> Breakpoint:
        """Set a breakpoint with optional conditions"""
        bp = Breakpoint(file=file, line=line, **kwargs)
        key = f"{file}:{line}"
        self.breakpoints[key] = bp
        return bp
    
    def remove_breakpoint(self, file: str, line: int):
        """Remove a breakpoint"""
        key = f"{file}:{line}"
        if key in self.breakpoints:
            del self.breakpoints[key]
    
    def start_debugging(self, target: Callable, *args, **kwargs):
        """Start interactive debugging session"""
        self.debug_session_active = True
        self.call_stack = []
        
        # Set trace function
        sys.settrace(self._trace_dispatch)
        
        try:
            result = target(*args, **kwargs)
            return result
        finally:
            sys.settrace(None)
            self.debug_session_active = False
    
    def _trace_dispatch(self, frame, event, arg):
        """Trace function for debugging"""
        if not self.debug_session_active:
            return None
        
        if event == 'call':
            self._handle_call(frame)
        elif event == 'line':
            return self._handle_line(frame)
        elif event == 'return':
            self._handle_return(frame, arg)
        elif event == 'exception':
            self._handle_exception(frame, arg)
        
        return self._trace_dispatch
    
    def _handle_call(self, frame):
        """Handle function call"""
        self.call_stack.append({
            'function': frame.f_code.co_name,
            'file': frame.f_code.co_filename,
            'line': frame.f_lineno,
            'locals': dict(frame.f_locals)
        })
    
    def _handle_line(self, frame):
        """Handle line execution"""
        # Check breakpoints
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        key = f"{filename}:{lineno}"
        
        if key in self.breakpoints:
            bp = self.breakpoints[key]
            if bp.should_break(frame):
                bp.hit_count += 1
                self._enter_interactive_mode(frame, f"Breakpoint {key} hit")
                
                if bp.temporary:
                    del self.breakpoints[key]
        
        # Handle stepping
        if self.stepping_mode == 'step_into':
            self._enter_interactive_mode(frame, "Step")
        elif self.stepping_mode == 'step_over' and len(self.call_stack) <= 1:
            self._enter_interactive_mode(frame, "Step over")
        
        return self._trace_dispatch
    
    def _handle_return(self, frame, return_value):
        """Handle function return"""
        if self.call_stack:
            self.call_stack.pop()
    
    def _handle_exception(self, frame, exc_info):
        """Handle exception"""
        exc_type, exc_value, exc_tb = exc_info
        self._enter_interactive_mode(frame, f"Exception: {exc_type.__name__}: {exc_value}")
    
    def _enter_interactive_mode(self, frame, message: str):
        """Enter interactive debugging mode"""
        self.current_frame = frame
        print(f"\n[DEBUG] {message}")
        self._show_context(frame)
        
        while True:
            try:
                command = input("(abov3-debug) ").strip()
                if not command:
                    continue
                
                self.command_history.append(command)
                
                if self._execute_command(command, frame):
                    break
            except KeyboardInterrupt:
                print("\nUse 'quit' to exit debugger")
            except Exception as e:
                print(f"Error: {e}")
    
    def _show_context(self, frame, lines: int = 5):
        """Show code context around current line"""
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        
        print(f"\nFile: {filename}")
        print(f"Function: {frame.f_code.co_name}")
        print(f"Line: {lineno}\n")
        
        start = max(1, lineno - lines)
        end = lineno + lines + 1
        
        for i in range(start, end):
            line = linecache.getline(filename, i)
            if line:
                marker = "=>" if i == lineno else "  "
                print(f"{marker} {i:4d}: {line.rstrip()}")
    
    def _execute_command(self, command: str, frame) -> bool:
        """Execute debug command"""
        parts = command.split()
        if not parts:
            return False
        
        cmd = parts[0].lower()
        
        if cmd in ['c', 'continue']:
            self.stepping_mode = None
            return True
        
        elif cmd in ['s', 'step', 'step_into']:
            self.stepping_mode = 'step_into'
            return True
        
        elif cmd in ['n', 'next', 'step_over']:
            self.stepping_mode = 'step_over'
            return True
        
        elif cmd in ['l', 'list']:
            self._show_context(frame, lines=10)
        
        elif cmd in ['p', 'print']:
            if len(parts) > 1:
                expr = ' '.join(parts[1:])
                self._evaluate_expression(expr, frame)
        
        elif cmd in ['pp', 'pprint']:
            if len(parts) > 1:
                expr = ' '.join(parts[1:])
                self._pretty_print(expr, frame)
        
        elif cmd == 'locals':
            self._show_locals(frame)
        
        elif cmd == 'globals':
            self._show_globals(frame)
        
        elif cmd == 'stack':
            self._show_stack()
        
        elif cmd == 'watch':
            if len(parts) > 1:
                expr = ' '.join(parts[1:])
                self.watch_expressions.append(expr)
                print(f"Watching: {expr}")
        
        elif cmd == 'watches':
            self._show_watches(frame)
        
        elif cmd in ['h', 'help']:
            self._show_help()
        
        elif cmd in ['q', 'quit']:
            self.debug_session_active = False
            sys.settrace(None)
            return True
        
        else:
            # Try to evaluate as expression
            self._evaluate_expression(command, frame)
        
        return False
    
    def _evaluate_expression(self, expr: str, frame):
        """Evaluate expression in current context"""
        try:
            result = eval(expr, frame.f_globals, frame.f_locals)
            print(f"{expr} = {result}")
        except Exception as e:
            print(f"Error evaluating '{expr}': {e}")
    
    def _pretty_print(self, expr: str, frame):
        """Pretty print expression"""
        try:
            import pprint
            result = eval(expr, frame.f_globals, frame.f_locals)
            pprint.pprint(result)
        except Exception as e:
            print(f"Error evaluating '{expr}': {e}")
    
    def _show_locals(self, frame):
        """Show local variables"""
        print("\nLocal variables:")
        for name, value in sorted(frame.f_locals.items()):
            if not name.startswith('__'):
                print(f"  {name} = {repr(value)[:100]}")
    
    def _show_globals(self, frame):
        """Show global variables"""
        print("\nGlobal variables (non-callable):")
        for name, value in sorted(frame.f_globals.items()):
            if not name.startswith('__') and not callable(value):
                print(f"  {name} = {repr(value)[:100]}")
    
    def _show_stack(self):
        """Show call stack"""
        print("\nCall stack:")
        for i, frame_info in enumerate(self.call_stack):
            print(f"  {i}: {frame_info['function']} at {frame_info['file']}:{frame_info['line']}")
    
    def _show_watches(self, frame):
        """Show watch expressions"""
        print("\nWatch expressions:")
        for expr in self.watch_expressions:
            try:
                result = eval(expr, frame.f_globals, frame.f_locals)
                print(f"  {expr} = {result}")
            except Exception as e:
                print(f"  {expr} = <Error: {e}>")
    
    def _show_help(self):
        """Show help for debug commands"""
        help_text = """
Debug Commands:
  c, continue     - Continue execution
  s, step         - Step into function
  n, next         - Step over function
  l, list         - Show code context
  p <expr>        - Print expression
  pp <expr>       - Pretty print expression
  locals          - Show local variables
  globals         - Show global variables
  stack           - Show call stack
  watch <expr>    - Add watch expression
  watches         - Show all watches
  h, help         - Show this help
  q, quit         - Quit debugger
        """
        print(help_text)


class NaturalLanguageDebugger:
    """Natural language interface for debugging"""
    
    def __init__(self, error_analyzer: IntelligentErrorAnalyzer):
        self.error_analyzer = error_analyzer
        self.query_patterns = self._load_query_patterns()
        self.context = {}
        
    def _load_query_patterns(self) -> List[Tuple[str, Callable]]:
        """Load natural language query patterns"""
        patterns = [
            (r"why.*slow|performance.*issue", self._analyze_performance),
            (r"why.*error|why.*fail|why.*crash", self._analyze_error),
            (r"what.*wrong|what.*problem", self._identify_problem),
            (r"how.*fix|how.*solve", self._suggest_fix),
            (r"memory.*leak|memory.*issue", self._analyze_memory),
            (r"infinite.*loop|stuck|hang", self._detect_infinite_loop),
            (r"variable.*value|what.*value", self._check_variable),
            (r"trace|call.*stack", self._show_trace),
            (r"bottleneck|hotspot", self._find_bottleneck),
            (r"optimize|improve.*performance", self._suggest_optimization)
        ]
        return patterns
    
    def process_query(self, query: str, context: Dict[str, Any]) -> str:
        """Process natural language debug query"""
        self.context = context
        query_lower = query.lower()
        
        # Match query pattern
        for pattern, handler in self.query_patterns:
            if re.search(pattern, query_lower):
                return handler(query, context)
        
        # Default response
        return self._general_analysis(query, context)
    
    def _analyze_performance(self, query: str, context: Dict[str, Any]) -> str:
        """Analyze performance issues"""
        response = "Performance Analysis:\n\n"
        
        if 'profile_data' in context:
            profile = context['profile_data']
            
            # Find slow functions
            slow_functions = []
            for func_name, times in profile.get('execution_times', {}).items():
                if times:
                    avg_time = sum(times) / len(times)
                    if avg_time > 0.1:  # Functions taking >100ms
                        slow_functions.append((func_name, avg_time))
            
            if slow_functions:
                response += "Slow Functions Detected:\n"
                for func, time in sorted(slow_functions, key=lambda x: x[1], reverse=True)[:5]:
                    response += f"  - {func}: {time:.3f}s average\n"
                
                response += "\nRecommendations:\n"
                response += "  1. Consider caching frequently called functions\n"
                response += "  2. Use profiling to identify exact bottlenecks\n"
                response += "  3. Optimize algorithms in hot paths\n"
                response += "  4. Consider async/parallel processing\n"
            else:
                response += "No significant performance issues detected.\n"
        else:
            response += "No performance data available. Run profiler first.\n"
        
        return response
    
    def _analyze_error(self, query: str, context: Dict[str, Any]) -> str:
        """Analyze error causes"""
        response = "Error Analysis:\n\n"
        
        if 'exception' in context:
            exception = context['exception']
            analysis = self.error_analyzer.analyze_error(exception, context)
            
            response += f"Error Type: {analysis['error_type']}\n"
            response += f"Message: {analysis['message']}\n"
            response += f"Severity: {analysis['severity']}/5\n\n"
            
            if analysis['root_cause'] and analysis['root_cause']['identified']:
                response += f"Root Cause: {analysis['root_cause']['description']}\n"
                response += f"Confidence: {analysis['root_cause']['confidence']:.1%}\n\n"
            
            if analysis['solutions']:
                response += "Suggested Solutions:\n"
                for i, solution in enumerate(analysis['solutions'][:5], 1):
                    response += f"  {i}. {solution}\n"
            
            if analysis['prevention_suggestions']:
                response += "\nPrevention Tips:\n"
                for tip in analysis['prevention_suggestions'][:3]:
                    response += f"  - {tip}\n"
        else:
            response += "No error context provided.\n"
        
        return response
    
    def _identify_problem(self, query: str, context: Dict[str, Any]) -> str:
        """Identify what's wrong with the code"""
        response = "Problem Identification:\n\n"
        
        problems = []
        
        # Check for errors
        if 'exception' in context:
            problems.append(f"Exception occurred: {context['exception']}")
        
        # Check for performance issues
        if 'profile_data' in context:
            profile = context['profile_data']
            if profile.get('bottlenecks'):
                problems.append(f"Found {len(profile['bottlenecks'])} performance bottlenecks")
        
        # Check for memory issues
        if 'memory_usage' in context:
            memory = context['memory_usage']
            if memory > 1000:  # Over 1GB
                problems.append(f"High memory usage: {memory/1024:.1f}GB")
        
        if problems:
            response += "Identified Problems:\n"
            for problem in problems:
                response += f"  - {problem}\n"
        else:
            response += "No obvious problems detected. Consider:\n"
            response += "  - Running comprehensive tests\n"
            response += "  - Checking edge cases\n"
            response += "  - Validating input data\n"
        
        return response
    
    def _suggest_fix(self, query: str, context: Dict[str, Any]) -> str:
        """Suggest how to fix issues"""
        response = "Fix Suggestions:\n\n"
        
        if 'exception' in context:
            exception = context['exception']
            analysis = self.error_analyzer.analyze_error(exception, context)
            
            response += f"For {analysis['error_type']}:\n\n"
            
            if analysis['solutions']:
                response += "Step-by-step fix:\n"
                for i, solution in enumerate(analysis['solutions'][:5], 1):
                    response += f"  {i}. {solution}\n"
                
                response += "\nCode example:\n"
                response += self._generate_fix_code(exception)
        else:
            response += "Please provide error context for specific fix suggestions.\n"
        
        return response
    
    def _generate_fix_code(self, exception: Exception) -> str:
        """Generate example fix code"""
        code = "```python\n"
        
        if isinstance(exception, AttributeError):
            code += "# Fix for AttributeError\n"
            code += "if obj is not None:\n"
            code += "    value = obj.attribute\n"
            code += "else:\n"
            code += "    value = default_value\n"
        
        elif isinstance(exception, KeyError):
            code += "# Fix for KeyError\n"
            code += "# Option 1: Using get()\n"
            code += "value = dictionary.get('key', default_value)\n\n"
            code += "# Option 2: Check existence\n"
            code += "if 'key' in dictionary:\n"
            code += "    value = dictionary['key']\n"
        
        elif isinstance(exception, IndexError):
            code += "# Fix for IndexError\n"
            code += "if index < len(list_data):\n"
            code += "    value = list_data[index]\n"
            code += "else:\n"
            code += "    value = None  # or handle appropriately\n"
        
        else:
            code += "# Generic error handling\n"
            code += "try:\n"
            code += "    # Your code here\n"
            code += "    result = risky_operation()\n"
            code += f"except {type(exception).__name__} as e:\n"
            code += "    # Handle the error\n"
            code += "    print(f'Error: {e}')\n"
            code += "    result = default_value\n"
        
        code += "```"
        return code
    
    def _analyze_memory(self, query: str, context: Dict[str, Any]) -> str:
        """Analyze memory issues"""
        response = "Memory Analysis:\n\n"
        
        if 'memory_profile' in context:
            profile = context['memory_profile']
            
            response += f"Current Memory Usage: {profile.get('current', 0):.1f}MB\n"
            response += f"Peak Memory Usage: {profile.get('peak', 0):.1f}MB\n\n"
            
            # Check for memory leaks
            if profile.get('growth_rate', 0) > 10:  # MB per minute
                response += "WARNING: Possible memory leak detected!\n"
                response += f"Memory growing at {profile['growth_rate']:.1f}MB/min\n\n"
                response += "Common causes:\n"
                response += "  - Circular references\n"
                response += "  - Large objects in global scope\n"
                response += "  - Unclosed file handles or connections\n"
                response += "  - Growing lists/dicts without cleanup\n\n"
                response += "Solutions:\n"
                response += "  1. Use weak references for circular dependencies\n"
                response += "  2. Implement __del__ methods for cleanup\n"
                response += "  3. Use context managers (with statement)\n"
                response += "  4. Call gc.collect() periodically\n"
        else:
            response += "No memory profile data available.\n"
        
        return response
    
    def _detect_infinite_loop(self, query: str, context: Dict[str, Any]) -> str:
        """Detect infinite loops"""
        response = "Infinite Loop Detection:\n\n"
        
        if 'execution_trace' in context:
            trace = context['execution_trace']
            
            # Look for repeated patterns
            pattern_count = defaultdict(int)
            for i in range(len(trace) - 10):
                pattern = tuple(trace[i:i+5])  # Look for 5-call patterns
                pattern_count[pattern] += 1
            
            repeated_patterns = [(p, c) for p, c in pattern_count.items() if c > 10]
            
            if repeated_patterns:
                response += "Detected repeated execution patterns:\n"
                for pattern, count in repeated_patterns[:3]:
                    response += f"  Pattern repeated {count} times\n"
                
                response += "\nPossible infinite loop locations:\n"
                # Extract function names from patterns
                for pattern, _ in repeated_patterns[:3]:
                    if pattern:
                        func_name = pattern[0].get('function', 'unknown')
                        response += f"  - Function: {func_name}\n"
                
                response += "\nSolutions:\n"
                response += "  1. Add loop termination condition\n"
                response += "  2. Check loop variable updates\n"
                response += "  3. Add maximum iteration limit\n"
                response += "  4. Verify recursion base case\n"
            else:
                response += "No infinite loop patterns detected.\n"
        else:
            response += "No execution trace available.\n"
        
        return response
    
    def _check_variable(self, query: str, context: Dict[str, Any]) -> str:
        """Check variable values"""
        response = "Variable Inspection:\n\n"
        
        # Extract variable name from query
        var_match = re.search(r"variable\s+(\w+)|value\s+of\s+(\w+)", query.lower())
        if var_match:
            var_name = var_match.group(1) or var_match.group(2)
            
            if 'variables' in context and var_name in context['variables']:
                var_info = context['variables'][var_name]
                response += f"Variable: {var_name}\n"
                response += f"  Type: {var_info.get('type', 'unknown')}\n"
                response += f"  Value: {var_info.get('value', 'unknown')}\n"
                
                if 'history' in var_info:
                    response += f"  History: {var_info['history'][-5:]}\n"
            else:
                response += f"Variable '{var_name}' not found in current context.\n"
        else:
            response += "Please specify a variable name to inspect.\n"
        
        return response
    
    def _show_trace(self, query: str, context: Dict[str, Any]) -> str:
        """Show execution trace"""
        response = "Execution Trace:\n\n"
        
        if 'call_stack' in context:
            stack = context['call_stack']
            
            response += "Call Stack (most recent first):\n"
            for i, frame in enumerate(reversed(stack[-10:]), 1):
                response += f"  {i}. {frame.get('function', 'unknown')} "
                response += f"at {frame.get('file', 'unknown')}:"
                response += f"{frame.get('line', '?')}\n"
        else:
            response += "No call stack available.\n"
        
        return response
    
    def _find_bottleneck(self, query: str, context: Dict[str, Any]) -> str:
        """Find performance bottlenecks"""
        response = "Bottleneck Analysis:\n\n"
        
        if 'profile_data' in context:
            profile = context['profile_data']
            
            bottlenecks = profile.get('bottlenecks', [])
            if bottlenecks:
                response += "Top Performance Bottlenecks:\n"
                for i, bottleneck in enumerate(bottlenecks[:5], 1):
                    response += f"  {i}. {bottleneck['name']}\n"
                    response += f"     Duration: {bottleneck['duration']:.3f}s\n"
                    response += f"     Memory: {bottleneck.get('memory', 0):.1f}MB\n"
                
                response += "\nOptimization Suggestions:\n"
                response += "  1. Cache expensive computations\n"
                response += "  2. Use vectorized operations (numpy)\n"
                response += "  3. Implement lazy loading\n"
                response += "  4. Consider parallel processing\n"
                response += "  5. Optimize database queries\n"
            else:
                response += "No significant bottlenecks detected.\n"
        else:
            response += "No profiling data available.\n"
        
        return response
    
    def _suggest_optimization(self, query: str, context: Dict[str, Any]) -> str:
        """Suggest optimizations"""
        response = "Optimization Suggestions:\n\n"
        
        suggestions = []
        
        # Check for common optimization opportunities
        if 'code' in context:
            code = context['code']
            
            # Check for inefficient patterns
            if 'for' in code and 'append' in code:
                suggestions.append("Use list comprehensions instead of loops with append")
            
            if 'open(' in code and 'with' not in code:
                suggestions.append("Use context managers (with statement) for file operations")
            
            if 'global' in code:
                suggestions.append("Minimize use of global variables")
            
            if code.count('for') > 3:
                suggestions.append("Consider reducing nested loops or using vectorization")
        
        # Performance-based suggestions
        if 'profile_data' in context:
            profile = context['profile_data']
            
            high_call_functions = [f for f, c in profile.get('function_calls', {}).items() if c > 1000]
            if high_call_functions:
                suggestions.append(f"Cache results for frequently called functions: {', '.join(high_call_functions[:3])}")
            
            if profile.get('memory_usage', {}).get('peak', 0) > 500:  # MB
                suggestions.append("Optimize memory usage with generators or streaming")
        
        if suggestions:
            response += "Specific Optimizations:\n"
            for i, suggestion in enumerate(suggestions, 1):
                response += f"  {i}. {suggestion}\n"
        else:
            response += "General Optimization Tips:\n"
            response += "  1. Profile before optimizing\n"
            response += "  2. Focus on algorithmic improvements\n"
            response += "  3. Use appropriate data structures\n"
            response += "  4. Minimize I/O operations\n"
            response += "  5. Consider caching and memoization\n"
        
        return response
    
    def _general_analysis(self, query: str, context: Dict[str, Any]) -> str:
        """General analysis for unmatched queries"""
        response = "Debug Analysis:\n\n"
        response += "I understand you're asking about: " + query + "\n\n"
        
        response += "Available debug information:\n"
        if 'exception' in context:
            response += "  - Exception data available\n"
        if 'profile_data' in context:
            response += "  - Performance profile available\n"
        if 'memory_profile' in context:
            response += "  - Memory profile available\n"
        if 'call_stack' in context:
            response += "  - Call stack available\n"
        
        response += "\nTry asking:\n"
        response += "  - 'Why is this slow?'\n"
        response += "  - 'What's causing the error?'\n"
        response += "  - 'How do I fix this?'\n"
        response += "  - 'Is there a memory leak?'\n"
        response += "  - 'Where is the bottleneck?'\n"
        
        return response


class EnterpriseDebugEngine:
    """Main enterprise debugging engine with Claude-level capabilities"""
    
    def __init__(self):
        self.error_analyzer = IntelligentErrorAnalyzer()
        self.interactive_debugger = InteractiveDebugger()
        self.nl_debugger = NaturalLanguageDebugger(self.error_analyzer)
        self.performance_profiles: Dict[str, PerformanceProfile] = {}
        self.debug_sessions: Dict[str, Dict[str, Any]] = {}
        self.current_session_id = None
        
    def create_debug_session(self, name: str = None) -> str:
        """Create a new debug session"""
        import uuid
        session_id = str(uuid.uuid4())
        
        self.debug_sessions[session_id] = {
            'id': session_id,
            'name': name or f"Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'created': datetime.now(),
            'error_count': 0,
            'performance_data': {},
            'memory_data': {},
            'execution_trace': [],
            'insights': []
        }
        
        self.current_session_id = session_id
        return session_id
    
    def debug_code(self, code: str, language: str = 'python') -> Dict[str, Any]:
        """Debug code with comprehensive analysis"""
        analysis = {
            'syntax_check': self._check_syntax(code, language),
            'static_analysis': self._static_analysis(code, language),
            'complexity_analysis': self._analyze_complexity(code),
            'security_analysis': self._security_scan(code),
            'best_practices': self._check_best_practices(code, language),
            'potential_issues': [],
            'suggestions': []
        }
        
        # Identify potential issues
        issues = self._identify_potential_issues(code, language)
        analysis['potential_issues'] = issues
        
        # Generate suggestions
        suggestions = self._generate_suggestions(code, issues, language)
        analysis['suggestions'] = suggestions
        
        return analysis
    
    def _check_syntax(self, code: str, language: str) -> Dict[str, Any]:
        """Check code syntax"""
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        if language == 'python':
            try:
                compile(code, '<string>', 'exec')
            except SyntaxError as e:
                result['valid'] = False
                result['errors'].append({
                    'line': e.lineno,
                    'column': e.offset,
                    'message': str(e)
                })
        
        return result
    
    def _static_analysis(self, code: str, language: str) -> Dict[str, Any]:
        """Perform static code analysis"""
        analysis = {
            'unused_variables': [],
            'undefined_variables': [],
            'type_issues': [],
            'code_smells': []
        }
        
        if language == 'python':
            try:
                tree = ast.parse(code)
                
                # Find unused variables
                class VariableAnalyzer(ast.NodeVisitor):
                    def __init__(self):
                        self.defined = set()
                        self.used = set()
                    
                    def visit_Name(self, node):
                        if isinstance(node.ctx, ast.Store):
                            self.defined.add(node.id)
                        elif isinstance(node.ctx, ast.Load):
                            self.used.add(node.id)
                        self.generic_visit(node)
                
                analyzer = VariableAnalyzer()
                analyzer.visit(tree)
                
                unused = analyzer.defined - analyzer.used
                analysis['unused_variables'] = list(unused)
                
                undefined = analyzer.used - analyzer.defined
                # Filter out built-ins
                import builtins
                undefined = [v for v in undefined if not hasattr(builtins, v)]
                analysis['undefined_variables'] = undefined
                
            except:
                pass
        
        return analysis
    
    def _analyze_complexity(self, code: str) -> Dict[str, Any]:
        """Analyze code complexity"""
        complexity = {
            'cyclomatic_complexity': 0,
            'cognitive_complexity': 0,
            'lines_of_code': len(code.splitlines()),
            'nesting_depth': 0,
            'function_count': 0,
            'class_count': 0
        }
        
        try:
            tree = ast.parse(code)
            
            # Count functions and classes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    complexity['function_count'] += 1
                elif isinstance(node, ast.ClassDef):
                    complexity['class_count'] += 1
            
            # Calculate cyclomatic complexity (simplified)
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For)):
                    complexity['cyclomatic_complexity'] += 1
                elif isinstance(node, ast.ExceptHandler):
                    complexity['cyclomatic_complexity'] += 1
            
            # Calculate max nesting depth
            class NestingAnalyzer(ast.NodeVisitor):
                def __init__(self):
                    self.max_depth = 0
                    self.current_depth = 0
                
                def visit_If(self, node):
                    self.current_depth += 1
                    self.max_depth = max(self.max_depth, self.current_depth)
                    self.generic_visit(node)
                    self.current_depth -= 1
                
                def visit_For(self, node):
                    self.current_depth += 1
                    self.max_depth = max(self.max_depth, self.current_depth)
                    self.generic_visit(node)
                    self.current_depth -= 1
                
                def visit_While(self, node):
                    self.current_depth += 1
                    self.max_depth = max(self.max_depth, self.current_depth)
                    self.generic_visit(node)
                    self.current_depth -= 1
            
            nesting_analyzer = NestingAnalyzer()
            nesting_analyzer.visit(tree)
            complexity['nesting_depth'] = nesting_analyzer.max_depth
            
        except:
            pass
        
        return complexity
    
    def _security_scan(self, code: str) -> Dict[str, Any]:
        """Scan for security vulnerabilities"""
        vulnerabilities = {
            'high_risk': [],
            'medium_risk': [],
            'low_risk': []
        }
        
        # Check for common security issues
        security_patterns = [
            (r'eval\s*\(', 'high_risk', 'Use of eval() is dangerous'),
            (r'exec\s*\(', 'high_risk', 'Use of exec() is dangerous'),
            (r'__import__\s*\(', 'medium_risk', 'Dynamic imports can be risky'),
            (r'pickle\.loads?\s*\(', 'high_risk', 'Pickle can execute arbitrary code'),
            (r'input\s*\(', 'low_risk', 'User input should be validated'),
            (r'os\.system\s*\(', 'high_risk', 'Command injection risk'),
            (r'subprocess\.\w+\s*\(.*shell\s*=\s*True', 'high_risk', 'Shell injection risk'),
            (r'open\s*\([^)]*[\'"]w[\'"]', 'low_risk', 'File write operation detected'),
            (r'requests\.get\s*\([^)]*verify\s*=\s*False', 'medium_risk', 'SSL verification disabled')
        ]
        
        for pattern, risk_level, message in security_patterns:
            if re.search(pattern, code):
                vulnerabilities[risk_level].append(message)
        
        return vulnerabilities
    
    def _check_best_practices(self, code: str, language: str) -> List[str]:
        """Check code against best practices"""
        violations = []
        
        if language == 'python':
            # PEP 8 style checks (simplified)
            lines = code.splitlines()
            
            for i, line in enumerate(lines, 1):
                # Line length
                if len(line) > 79:
                    violations.append(f"Line {i}: Line too long ({len(line)} > 79 characters)")
                
                # Trailing whitespace
                if line.rstrip() != line:
                    violations.append(f"Line {i}: Trailing whitespace")
                
                # Multiple statements on one line
                if ';' in line and not line.strip().startswith('#'):
                    violations.append(f"Line {i}: Multiple statements on one line")
            
            # Function naming
            if re.search(r'def [A-Z]', code):
                violations.append("Function names should be lowercase with underscores")
            
            # Class naming
            if re.search(r'class [a-z]', code):
                violations.append("Class names should use CapWords convention")
        
        return violations
    
    def _identify_potential_issues(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Identify potential issues in code"""
        issues = []
        
        # Common anti-patterns
        anti_patterns = [
            {
                'pattern': r'except:\s*$',
                'issue': 'Bare except clause catches all exceptions',
                'severity': 'medium',
                'suggestion': 'Specify exception types to catch'
            },
            {
                'pattern': r'if\s+\w+\s*==\s*True',
                'issue': 'Redundant comparison to True',
                'severity': 'low',
                'suggestion': 'Use "if variable:" instead'
            },
            {
                'pattern': r'if\s+\w+\s*==\s*None',
                'issue': 'Use "is None" for None comparison',
                'severity': 'low',
                'suggestion': 'Use "if variable is None:" instead'
            },
            {
                'pattern': r'for\s+\w+\s+in\s+range\(len\(',
                'issue': 'Iterating with range(len()) is unpythonic',
                'severity': 'low',
                'suggestion': 'Use enumerate() or iterate directly'
            }
        ]
        
        for anti_pattern in anti_patterns:
            if re.search(anti_pattern['pattern'], code):
                issues.append(anti_pattern)
        
        return issues
    
    def _generate_suggestions(self, code: str, issues: List[Dict[str, Any]], language: str) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []
        
        # Based on identified issues
        for issue in issues:
            suggestions.append(issue['suggestion'])
        
        # General suggestions based on code analysis
        if 'print(' in code and 'logging' not in code:
            suggestions.append("Consider using logging module instead of print statements")
        
        if not re.search(r'""".*"""', code, re.DOTALL) and not re.search(r"'''.*'''", code, re.DOTALL):
            suggestions.append("Add docstrings to functions and classes")
        
        if 'TODO' in code or 'FIXME' in code:
            suggestions.append("Address TODO/FIXME comments in code")
        
        return suggestions
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile function performance"""
        func_id = f"{func.__module__}.{func.__name__}"
        
        if func_id not in self.performance_profiles:
            self.performance_profiles[func_id] = PerformanceProfile(
                function_name=func.__name__,
                file_path=func.__code__.co_filename,
                line_number=func.__code__.co_firstlineno
            )
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            profile = self.performance_profiles[func_id]
            
            # Measure execution time
            start_time = time.perf_counter()
            start_memory = self._get_memory_usage()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.perf_counter()
                end_memory = self._get_memory_usage()
                
                execution_time = end_time - start_time
                memory_delta = end_memory - start_memory
                
                profile.update(execution_time, memory_delta)
                
                # Store in current session
                if self.current_session_id:
                    session = self.debug_sessions[self.current_session_id]
                    if 'performance_data' not in session:
                        session['performance_data'] = {}
                    session['performance_data'][func_id] = profile
        
        return wrapper
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes"""
        if psutil:
            process = psutil.Process()
            return process.memory_info().rss
        return 0
    
    def analyze_exception(self, exception: Exception, **kwargs) -> Dict[str, Any]:
        """Analyze exception with Claude-level intelligence"""
        context = {
            'exception': exception,
            'timestamp': datetime.now(),
            **kwargs
        }
        
        analysis = self.error_analyzer.analyze_error(exception, context)
        
        # Store in current session
        if self.current_session_id:
            session = self.debug_sessions[self.current_session_id]
            session['error_count'] += 1
            
            if 'errors' not in session:
                session['errors'] = []
            session['errors'].append(analysis)
        
        return analysis
    
    def query(self, natural_language_query: str, **context) -> str:
        """Process natural language debug query"""
        # Add session context
        if self.current_session_id:
            session = self.debug_sessions[self.current_session_id]
            context.update({
                'session': session,
                'performance_data': session.get('performance_data', {}),
                'errors': session.get('errors', [])
            })
        
        return self.nl_debugger.process_query(natural_language_query, context)
    
    def get_debug_report(self, session_id: str = None) -> Dict[str, Any]:
        """Generate comprehensive debug report"""
        session_id = session_id or self.current_session_id
        
        if not session_id or session_id not in self.debug_sessions:
            return {'error': 'No debug session found'}
        
        session = self.debug_sessions[session_id]
        
        report = {
            'session_id': session_id,
            'session_name': session['name'],
            'duration': (datetime.now() - session['created']).total_seconds(),
            'error_summary': {
                'total_errors': session['error_count'],
                'error_types': self._summarize_error_types(session.get('errors', []))
            },
            'performance_summary': self._generate_performance_summary(session.get('performance_data', {})),
            'recommendations': self._generate_recommendations(session),
            'insights': session.get('insights', [])
        }
        
        return report
    
    def _summarize_error_types(self, errors: List[Dict[str, Any]]) -> Dict[str, int]:
        """Summarize error types"""
        summary = defaultdict(int)
        for error in errors:
            summary[error.get('error_type', 'Unknown')] += 1
        return dict(summary)
    
    def _generate_performance_summary(self, performance_data: Dict[str, PerformanceProfile]) -> Dict[str, Any]:
        """Generate performance summary"""
        if not performance_data:
            return {}
        
        total_time = sum(p.total_time for p in performance_data.values())
        total_calls = sum(p.call_count for p in performance_data.values())
        
        # Find slowest functions
        slowest = sorted(performance_data.items(), key=lambda x: x[1].total_time, reverse=True)[:5]
        
        # Find most called functions
        most_called = sorted(performance_data.items(), key=lambda x: x[1].call_count, reverse=True)[:5]
        
        return {
            'total_execution_time': total_time,
            'total_function_calls': total_calls,
            'slowest_functions': [
                {
                    'name': name,
                    'total_time': profile.total_time,
                    'avg_time': profile.avg_time,
                    'calls': profile.call_count
                }
                for name, profile in slowest
            ],
            'most_called_functions': [
                {
                    'name': name,
                    'calls': profile.call_count,
                    'total_time': profile.total_time
                }
                for name, profile in most_called
            ]
        }
    
    def _generate_recommendations(self, session: Dict[str, Any]) -> List[str]:
        """Generate session-based recommendations"""
        recommendations = []
        
        # Error-based recommendations
        if session['error_count'] > 10:
            recommendations.append("High error rate detected - consider adding more error handling")
        
        errors = session.get('errors', [])
        if any(e.get('severity', 0) >= 4 for e in errors):
            recommendations.append("Critical errors detected - prioritize fixing high-severity issues")
        
        # Performance-based recommendations
        perf_data = session.get('performance_data', {})
        if perf_data:
            slow_functions = [k for k, v in perf_data.items() if v.avg_time > 1.0]
            if slow_functions:
                recommendations.append(f"Optimize slow functions: {', '.join(slow_functions[:3])}")
        
        return recommendations


# Global instance
_debug_engine = None

def get_debug_engine() -> EnterpriseDebugEngine:
    """Get global debug engine instance"""
    global _debug_engine
    if _debug_engine is None:
        _debug_engine = EnterpriseDebugEngine()
    return _debug_engine

def get_enhanced_debug_engine():
    """Get enhanced ML-powered debug engine"""
    try:
        from .enhanced_ml_debugger import get_enhanced_debugger
        return get_enhanced_debugger()
    except ImportError:
        # Fallback to basic debug engine
        return get_debug_engine()


# Convenience functions
def debug(code: str, language: str = 'python') -> Dict[str, Any]:
    """Quick debug function"""
    engine = get_debug_engine()
    return engine.debug_code(code, language)

def analyze_error(exception: Exception, **context) -> Dict[str, Any]:
    """Analyze an exception"""
    engine = get_debug_engine()
    return engine.analyze_exception(exception, **context)

def ask_debug(query: str, **context) -> str:
    """Ask a natural language debug question"""
    engine = get_debug_engine()
    return engine.query(query, **context)

def profile(func: Callable) -> Callable:
    """Profile decorator"""
    engine = get_debug_engine()
    return engine.profile_function(func)