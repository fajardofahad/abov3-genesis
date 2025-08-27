"""
Execution tracing system for tracking code flow and state
"""

import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
import ast
import re

from ..parsers.error_parser import ErrorType

logger = logging.getLogger(__name__)

@dataclass
class TracePoint:
    """A point in the execution trace"""
    file: str
    line: int
    function: str
    variables: Dict[str, Any]
    timestamp: float
    
@dataclass
class TraceResult:
    """Result of execution tracing"""
    path: List[TracePoint]
    failure_point: Optional[str]
    state_changes: Dict[str, List[Any]]
    loops_detected: List[Dict[str, Any]]
    recursion_depth: int
    branches_taken: List[str]
    exceptions_caught: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]

class ExecutionTracer:
    """
    Advanced execution tracer for understanding code flow
    """
    
    def __init__(self):
        self.trace_cache = {}
        self.language_tracers = self._init_language_tracers()
        
    def _init_language_tracers(self) -> Dict[str, Any]:
        """Initialize language-specific tracing strategies"""
        return {
            "python": PythonTracer(),
            "javascript": JavaScriptTracer(),
            "java": JavaTracer(),
            "go": GoTracer()
        }
    
    async def trace(
        self,
        file_path: str,
        line_number: Optional[int],
        error_type: ErrorType,
        max_depth: int = 10
    ) -> Optional[TraceResult]:
        """
        Trace execution path to the error point
        """
        # Detect language from file extension
        language = self._detect_language(file_path)
        
        if language not in self.language_tracers:
            return self._generic_trace(file_path, line_number, error_type)
        
        tracer = self.language_tracers[language]
        return await tracer.trace(file_path, line_number, error_type, max_depth)
    
    def _detect_language(self, file_path: str) -> str:
        """Detect language from file extension"""
        extensions = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "javascript",
            ".java": "java",
            ".go": "go",
            ".rb": "ruby",
            ".php": "php",
            ".cpp": "cpp",
            ".c": "cpp",
            ".cs": "csharp",
            ".rs": "rust"
        }
        
        for ext, lang in extensions.items():
            if file_path.endswith(ext):
                return lang
        
        return "unknown"
    
    def _generic_trace(
        self,
        file_path: str,
        line_number: Optional[int],
        error_type: ErrorType
    ) -> TraceResult:
        """Generic tracing for unsupported languages"""
        return TraceResult(
            path=[],
            failure_point=f"{file_path}:{line_number}" if line_number else file_path,
            state_changes={},
            loops_detected=[],
            recursion_depth=0,
            branches_taken=[],
            exceptions_caught=[],
            performance_metrics={}
        )

class PythonTracer:
    """Python-specific execution tracer"""
    
    async def trace(
        self,
        file_path: str,
        line_number: Optional[int],
        error_type: ErrorType,
        max_depth: int
    ) -> TraceResult:
        """Trace Python code execution"""
        try:
            with open(file_path, 'r') as f:
                code = f.read()
            
            tree = ast.parse(code)
            
            # Analyze AST to understand code structure
            analyzer = PythonASTAnalyzer()
            analysis = analyzer.analyze(tree, line_number)
            
            # Build execution path
            path = self._build_execution_path(analysis, file_path, line_number)
            
            # Detect patterns
            loops = self._detect_loops(tree, line_number)
            recursion_depth = self._detect_recursion(analysis)
            branches = self._analyze_branches(tree, line_number)
            
            # Analyze state changes
            state_changes = self._analyze_state_changes(analysis)
            
            return TraceResult(
                path=path,
                failure_point=f"{file_path}:{line_number}" if line_number else None,
                state_changes=state_changes,
                loops_detected=loops,
                recursion_depth=recursion_depth,
                branches_taken=branches,
                exceptions_caught=self._find_exception_handlers(tree, line_number),
                performance_metrics={}
            )
            
        except Exception as e:
            logger.error(f"Python tracing failed: {str(e)}")
            return TraceResult(
                path=[],
                failure_point=f"{file_path}:{line_number}" if line_number else None,
                state_changes={},
                loops_detected=[],
                recursion_depth=0,
                branches_taken=[],
                exceptions_caught=[],
                performance_metrics={}
            )
    
    def _build_execution_path(
        self, analysis: Dict[str, Any], file_path: str, line_number: Optional[int]
    ) -> List[TracePoint]:
        """Build the execution path from analysis"""
        path = []
        
        # Add entry point
        if "entry_function" in analysis:
            path.append(TracePoint(
                file=file_path,
                line=analysis["entry_function"]["line"],
                function=analysis["entry_function"]["name"],
                variables={},
                timestamp=0.0
            ))
        
        # Add intermediate points
        if "call_chain" in analysis:
            for i, call in enumerate(analysis["call_chain"]):
                path.append(TracePoint(
                    file=file_path,
                    line=call["line"],
                    function=call["function"],
                    variables=call.get("variables", {}),
                    timestamp=float(i + 1)
                ))
        
        # Add failure point
        if line_number:
            path.append(TracePoint(
                file=file_path,
                line=line_number,
                function=analysis.get("containing_function", "unknown"),
                variables=analysis.get("local_variables", {}),
                timestamp=float(len(path))
            ))
        
        return path
    
    def _detect_loops(self, tree: ast.AST, target_line: Optional[int]) -> List[Dict[str, Any]]:
        """Detect loops in the code"""
        loops = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                loop_info = {
                    "type": "for" if isinstance(node, ast.For) else "while",
                    "line": node.lineno,
                    "nested": False
                }
                
                # Check if target line is within this loop
                if target_line and hasattr(node, 'end_lineno'):
                    if node.lineno <= target_line <= node.end_lineno:
                        loop_info["contains_error"] = True
                
                loops.append(loop_info)
        
        return loops
    
    def _detect_recursion(self, analysis: Dict[str, Any]) -> int:
        """Detect recursion depth"""
        if "call_chain" not in analysis:
            return 0
        
        function_counts = {}
        max_recursion = 0
        
        for call in analysis["call_chain"]:
            func_name = call["function"]
            function_counts[func_name] = function_counts.get(func_name, 0) + 1
            max_recursion = max(max_recursion, function_counts[func_name])
        
        return max_recursion
    
    def _analyze_branches(self, tree: ast.AST, target_line: Optional[int]) -> List[str]:
        """Analyze conditional branches"""
        branches = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                if target_line and hasattr(node, 'lineno'):
                    if node.lineno <= target_line:
                        branches.append(f"if at line {node.lineno}")
            elif isinstance(node, ast.Match):  # Python 3.10+
                if target_line and hasattr(node, 'lineno'):
                    if node.lineno <= target_line:
                        branches.append(f"match at line {node.lineno}")
        
        return branches
    
    def _analyze_state_changes(self, analysis: Dict[str, Any]) -> Dict[str, List[Any]]:
        """Analyze variable state changes"""
        state_changes = {}
        
        if "variable_assignments" in analysis:
            for var_name, assignments in analysis["variable_assignments"].items():
                state_changes[var_name] = assignments
        
        return state_changes
    
    def _find_exception_handlers(self, tree: ast.AST, target_line: Optional[int]) -> List[Dict[str, Any]]:
        """Find exception handlers in the code"""
        handlers = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                handler_info = {
                    "line": node.lineno,
                    "exceptions": []
                }
                
                for handler in node.handlers:
                    if handler.type:
                        if isinstance(handler.type, ast.Name):
                            handler_info["exceptions"].append(handler.type.id)
                
                # Check if target line is in try block
                if target_line and hasattr(node, 'end_lineno'):
                    if node.lineno <= target_line <= node.end_lineno:
                        handler_info["contains_error"] = True
                
                handlers.append(handler_info)
        
        return handlers

class PythonASTAnalyzer:
    """Analyzer for Python AST"""
    
    def analyze(self, tree: ast.AST, target_line: Optional[int]) -> Dict[str, Any]:
        """Analyze Python AST to understand code structure"""
        analysis = {
            "functions": [],
            "classes": [],
            "imports": [],
            "variable_assignments": {},
            "call_chain": []
        }
        
        # Find all functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    "name": node.name,
                    "line": node.lineno,
                    "args": [arg.arg for arg in node.args.args]
                }
                analysis["functions"].append(func_info)
                
                # Check if target line is in this function
                if target_line and hasattr(node, 'end_lineno'):
                    if node.lineno <= target_line <= node.end_lineno:
                        analysis["containing_function"] = node.name
            
            elif isinstance(node, ast.ClassDef):
                analysis["classes"].append({
                    "name": node.name,
                    "line": node.lineno
                })
            
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    analysis["imports"].append(alias.name)
            
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        if var_name not in analysis["variable_assignments"]:
                            analysis["variable_assignments"][var_name] = []
                        analysis["variable_assignments"][var_name].append({
                            "line": node.lineno,
                            "value": ast.unparse(node.value) if hasattr(ast, 'unparse') else str(node.value)
                        })
        
        return analysis

class JavaScriptTracer:
    """JavaScript-specific execution tracer"""
    
    async def trace(
        self,
        file_path: str,
        line_number: Optional[int],
        error_type: ErrorType,
        max_depth: int
    ) -> TraceResult:
        """Trace JavaScript code execution"""
        # Simplified JavaScript tracing
        return TraceResult(
            path=[],
            failure_point=f"{file_path}:{line_number}" if line_number else None,
            state_changes={},
            loops_detected=[],
            recursion_depth=0,
            branches_taken=[],
            exceptions_caught=[],
            performance_metrics={}
        )

class JavaTracer:
    """Java-specific execution tracer"""
    
    async def trace(
        self,
        file_path: str,
        line_number: Optional[int],
        error_type: ErrorType,
        max_depth: int
    ) -> TraceResult:
        """Trace Java code execution"""
        # Simplified Java tracing
        return TraceResult(
            path=[],
            failure_point=f"{file_path}:{line_number}" if line_number else None,
            state_changes={},
            loops_detected=[],
            recursion_depth=0,
            branches_taken=[],
            exceptions_caught=[],
            performance_metrics={}
        )

class GoTracer:
    """Go-specific execution tracer"""
    
    async def trace(
        self,
        file_path: str,
        line_number: Optional[int],
        error_type: ErrorType,
        max_depth: int
    ) -> TraceResult:
        """Trace Go code execution"""
        # Simplified Go tracing
        return TraceResult(
            path=[],
            failure_point=f"{file_path}:{line_number}" if line_number else None,
            state_changes={},
            loops_detected=[],
            recursion_depth=0,
            branches_taken=[],
            exceptions_caught=[],
            performance_metrics={}
        )