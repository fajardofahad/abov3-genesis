"""
ABOV3 Genesis - Refactoring Suggestion Engine
Intelligent suggestions for code improvements and refactoring
"""

import ast
import logging
import re
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum
import json

logger = logging.getLogger(__name__)

class RefactoringType(Enum):
    """Types of refactoring suggestions"""
    EXTRACT_METHOD = "extract_method"
    EXTRACT_CLASS = "extract_class"
    EXTRACT_VARIABLE = "extract_variable"
    INLINE_METHOD = "inline_method"
    INLINE_VARIABLE = "inline_variable"
    RENAME_SYMBOL = "rename_symbol"
    MOVE_METHOD = "move_method"
    REMOVE_DUPLICATION = "remove_duplication"
    SIMPLIFY_CONDITIONAL = "simplify_conditional"
    REPLACE_MAGIC_NUMBER = "replace_magic_number"
    ADD_PARAMETER = "add_parameter"
    REMOVE_PARAMETER = "remove_parameter"
    SPLIT_LARGE_CLASS = "split_large_class"
    MERGE_SIMILAR_METHODS = "merge_similar_methods"
    IMPROVE_NAMING = "improve_naming"
    ADD_ERROR_HANDLING = "add_error_handling"
    OPTIMIZE_PERFORMANCE = "optimize_performance"
    IMPROVE_READABILITY = "improve_readability"

class Priority(Enum):
    """Priority levels for refactoring suggestions"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RefactoringSuggestion:
    """A refactoring suggestion"""
    type: RefactoringType
    priority: Priority
    title: str
    description: str
    file_path: str
    line_start: int
    line_end: Optional[int] = None
    column_start: Optional[int] = None
    column_end: Optional[int] = None
    before_code: Optional[str] = None
    after_code: Optional[str] = None
    reasoning: str = ""
    effort_estimate: str = "medium"  # low, medium, high
    impact: str = "medium"  # low, medium, high
    tags: List[str] = field(default_factory=list)
    related_suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def id(self) -> str:
        """Generate unique ID for the suggestion"""
        import hashlib
        id_string = f"{self.type.value}:{self.file_path}:{self.line_start}:{hash(self.title)}"
        return hashlib.md5(id_string.encode()).hexdigest()[:12]

class RefactoringSuggestionEngine:
    """
    Engine for generating intelligent refactoring suggestions
    Analyzes code patterns and suggests improvements
    """
    
    def __init__(self):
        # Thresholds for various refactoring triggers
        self.thresholds = {
            'method_length': 50,
            'class_length': 200,
            'parameter_count': 5,
            'cyclomatic_complexity': 10,
            'nesting_depth': 4,
            'duplicate_lines': 5,
            'magic_number_usage': 3
        }
        
        # Pattern matchers for different refactoring opportunities
        self.refactoring_patterns = self._initialize_refactoring_patterns()
        
        # Code smell detectors
        self.code_smell_detectors = self._initialize_code_smell_detectors()
        
        # Performance improvement patterns
        self.performance_patterns = self._initialize_performance_patterns()
        
        logger.info("RefactoringSuggestionEngine initialized")
    
    def _initialize_refactoring_patterns(self) -> Dict[str, List[Dict]]:
        """Initialize patterns that trigger refactoring suggestions"""
        return {
            'extract_method': [
                {
                    'pattern': r'(.*\n.*\n.*\n.*\n.*\n.*\n.*\n.*\n.*\n.*\n)',  # Long sequences
                    'context': 'long_code_block',
                    'priority': Priority.MEDIUM,
                    'description': 'Long code block that could be extracted into a method'
                }
            ],
            'extract_variable': [
                {
                    'pattern': r'(\w+\.\w+\.\w+\.\w+)',  # Long method chains
                    'context': 'method_chaining',
                    'priority': Priority.LOW,
                    'description': 'Long method chain could be extracted to variable'
                }
            ],
            'replace_magic_number': [
                {
                    'pattern': r'\b(\d{2,})\b',  # Numbers with 2+ digits
                    'context': 'numeric_literal',
                    'priority': Priority.MEDIUM,
                    'description': 'Magic number should be replaced with named constant'
                }
            ],
            'simplify_conditional': [
                {
                    'pattern': r'if.*and.*and.*:',
                    'context': 'complex_condition',
                    'priority': Priority.MEDIUM,
                    'description': 'Complex conditional could be simplified'
                }
            ]
        }
    
    def _initialize_code_smell_detectors(self) -> Dict[str, Dict]:
        """Initialize code smell detection rules"""
        return {
            'long_method': {
                'threshold': self.thresholds['method_length'],
                'priority': Priority.HIGH,
                'refactoring': RefactoringType.EXTRACT_METHOD
            },
            'large_class': {
                'threshold': self.thresholds['class_length'],
                'priority': Priority.HIGH,
                'refactoring': RefactoringType.SPLIT_LARGE_CLASS
            },
            'too_many_parameters': {
                'threshold': self.thresholds['parameter_count'],
                'priority': Priority.MEDIUM,
                'refactoring': RefactoringType.EXTRACT_CLASS
            },
            'complex_method': {
                'threshold': self.thresholds['cyclomatic_complexity'],
                'priority': Priority.HIGH,
                'refactoring': RefactoringType.EXTRACT_METHOD
            },
            'deep_nesting': {
                'threshold': self.thresholds['nesting_depth'],
                'priority': Priority.MEDIUM,
                'refactoring': RefactoringType.EXTRACT_METHOD
            }
        }
    
    def _initialize_performance_patterns(self) -> Dict[str, List[Dict]]:
        """Initialize performance improvement patterns"""
        return {
            'python': [
                {
                    'pattern': r'for\s+\w+\s+in\s+range\(len\(',
                    'suggestion': 'Use enumerate() instead of range(len())',
                    'refactoring': RefactoringType.OPTIMIZE_PERFORMANCE,
                    'priority': Priority.LOW
                },
                {
                    'pattern': r'\w+\s*\+=\s*\[.*\]',
                    'suggestion': 'Use list.extend() instead of += for lists',
                    'refactoring': RefactoringType.OPTIMIZE_PERFORMANCE,
                    'priority': Priority.MEDIUM
                },
                {
                    'pattern': r'\.append\([^)]*\)\s*$',
                    'suggestion': 'Consider using list comprehension',
                    'refactoring': RefactoringType.OPTIMIZE_PERFORMANCE,
                    'priority': Priority.LOW
                }
            ],
            'javascript': [
                {
                    'pattern': r'document\.getElementById',
                    'suggestion': 'Cache DOM queries to improve performance',
                    'refactoring': RefactoringType.OPTIMIZE_PERFORMANCE,
                    'priority': Priority.LOW
                },
                {
                    'pattern': r'for\s*\(\s*var\s+\w+\s*=\s*0',
                    'suggestion': 'Use modern iteration methods (forEach, map, etc.)',
                    'refactoring': RefactoringType.IMPROVE_READABILITY,
                    'priority': Priority.LOW
                }
            ]
        }
    
    async def analyze_file(self, file_path: str, content: str = None) -> List[RefactoringSuggestion]:
        """Analyze a file and generate refactoring suggestions"""
        try:
            file_path = Path(file_path)
            
            if content is None:
                if not file_path.exists():
                    return []
                
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            
            # Detect language
            language = self._detect_language(file_path)
            
            suggestions = []
            
            # Analyze based on language
            if language == 'python':
                suggestions.extend(await self._analyze_python_file(str(file_path), content))
            else:
                suggestions.extend(await self._analyze_generic_file(str(file_path), content, language))
            
            # Apply pattern-based suggestions
            suggestions.extend(await self._apply_pattern_suggestions(str(file_path), content, language))
            
            # Apply performance suggestions
            suggestions.extend(await self._apply_performance_suggestions(str(file_path), content, language))
            
            # Remove duplicates and rank suggestions
            suggestions = self._deduplicate_suggestions(suggestions)
            suggestions = self._rank_suggestions(suggestions)
            
            logger.debug(f"Generated {len(suggestions)} refactoring suggestions for {file_path}")
            return suggestions
            
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            return []
    
    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension"""
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'jsx',
            '.tsx': 'tsx',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby'
        }
        return extension_map.get(file_path.suffix.lower(), 'unknown')
    
    async def _analyze_python_file(self, file_path: str, content: str) -> List[RefactoringSuggestion]:
        """Analyze Python file using AST for detailed suggestions"""
        suggestions = []
        
        try:
            tree = ast.parse(content)
            lines = content.splitlines()
            
            class RefactoringVisitor(ast.NodeVisitor):
                def __init__(self, suggestions_list, lines, file_path):
                    self.suggestions = suggestions_list
                    self.lines = lines
                    self.file_path = file_path
                    self.current_class = None
                
                def visit_FunctionDef(self, node):
                    self._analyze_function(node)
                    self.generic_visit(node)
                
                def visit_AsyncFunctionDef(self, node):
                    self._analyze_function(node)
                    self.generic_visit(node)
                
                def visit_ClassDef(self, node):
                    self._analyze_class(node)
                    old_class = self.current_class
                    self.current_class = node.name
                    self.generic_visit(node)
                    self.current_class = old_class
                
                def visit_If(self, node):
                    self._analyze_conditional(node)
                    self.generic_visit(node)
                
                def visit_For(self, node):
                    self._analyze_loop(node)
                    self.generic_visit(node)
                
                def _analyze_function(self, node):
                    # Check function length
                    func_length = (getattr(node, 'end_lineno', node.lineno) - node.lineno)
                    if func_length > self.thresholds['method_length']:
                        self.suggestions.append(RefactoringSuggestion(
                            type=RefactoringType.EXTRACT_METHOD,
                            priority=Priority.HIGH,
                            title=f"Extract parts of long function '{node.name}'",
                            description=f"Function '{node.name}' is {func_length} lines long, consider breaking it into smaller functions",
                            file_path=self.file_path,
                            line_start=node.lineno,
                            line_end=getattr(node, 'end_lineno', node.lineno),
                            reasoning="Long functions are harder to understand, test, and maintain",
                            effort_estimate="medium",
                            impact="high",
                            tags=["complexity", "maintainability"]
                        ))
                    
                    # Check parameter count
                    param_count = len(node.args.args)
                    if param_count > self.thresholds['parameter_count']:
                        self.suggestions.append(RefactoringSuggestion(
                            type=RefactoringType.EXTRACT_CLASS,
                            priority=Priority.MEDIUM,
                            title=f"Reduce parameters in function '{node.name}'",
                            description=f"Function '{node.name}' has {param_count} parameters, consider using a parameter object",
                            file_path=self.file_path,
                            line_start=node.lineno,
                            reasoning="Functions with many parameters are hard to understand and use",
                            effort_estimate="medium",
                            impact="medium",
                            tags=["parameters", "design"]
                        ))
                    
                    # Check cyclomatic complexity
                    complexity = self._calculate_complexity(node)
                    if complexity > self.thresholds['cyclomatic_complexity']:
                        self.suggestions.append(RefactoringSuggestion(
                            type=RefactoringType.EXTRACT_METHOD,
                            priority=Priority.HIGH,
                            title=f"Reduce complexity of function '{node.name}'",
                            description=f"Function '{node.name}' has complexity {complexity}, consider breaking it down",
                            file_path=self.file_path,
                            line_start=node.lineno,
                            reasoning="High complexity makes code harder to understand and test",
                            effort_estimate="high",
                            impact="high",
                            tags=["complexity", "testing"]
                        ))
                    
                    # Check for missing docstring
                    if not ast.get_docstring(node) and not node.name.startswith('_'):
                        self.suggestions.append(RefactoringSuggestion(
                            type=RefactoringType.IMPROVE_READABILITY,
                            priority=Priority.LOW,
                            title=f"Add docstring to function '{node.name}'",
                            description=f"Function '{node.name}' lacks documentation",
                            file_path=self.file_path,
                            line_start=node.lineno,
                            reasoning="Docstrings improve code understanding and documentation",
                            effort_estimate="low",
                            impact="medium",
                            tags=["documentation", "readability"]
                        ))
                
                def _analyze_class(self, node):
                    # Calculate class length
                    class_length = (getattr(node, 'end_lineno', node.lineno) - node.lineno)
                    method_count = len([n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))])
                    
                    if class_length > self.thresholds['class_length'] or method_count > 20:
                        self.suggestions.append(RefactoringSuggestion(
                            type=RefactoringType.SPLIT_LARGE_CLASS,
                            priority=Priority.HIGH,
                            title=f"Split large class '{node.name}'",
                            description=f"Class '{node.name}' has {class_length} lines and {method_count} methods",
                            file_path=self.file_path,
                            line_start=node.lineno,
                            line_end=getattr(node, 'end_lineno', node.lineno),
                            reasoning="Large classes violate single responsibility principle",
                            effort_estimate="high",
                            impact="high",
                            tags=["design", "single_responsibility"]
                        ))
                    
                    # Check for missing docstring
                    if not ast.get_docstring(node):
                        self.suggestions.append(RefactoringSuggestion(
                            type=RefactoringType.IMPROVE_READABILITY,
                            priority=Priority.LOW,
                            title=f"Add docstring to class '{node.name}'",
                            description=f"Class '{node.name}' lacks documentation",
                            file_path=self.file_path,
                            line_start=node.lineno,
                            reasoning="Class docstrings help understand the purpose and usage",
                            effort_estimate="low",
                            impact="medium",
                            tags=["documentation", "readability"]
                        ))
                
                def _analyze_conditional(self, node):
                    # Check for complex conditionals
                    condition_complexity = self._count_boolean_operators(node.test)
                    if condition_complexity > 3:
                        self.suggestions.append(RefactoringSuggestion(
                            type=RefactoringType.SIMPLIFY_CONDITIONAL,
                            priority=Priority.MEDIUM,
                            title="Simplify complex conditional",
                            description="Complex conditional with multiple boolean operators",
                            file_path=self.file_path,
                            line_start=node.lineno,
                            reasoning="Complex conditionals are harder to understand and debug",
                            effort_estimate="low",
                            impact="medium",
                            tags=["readability", "complexity"]
                        ))
                
                def _analyze_loop(self, node):
                    # Check for range(len()) pattern
                    if (isinstance(node.iter, ast.Call) and 
                        isinstance(node.iter.func, ast.Name) and 
                        node.iter.func.id == 'range' and 
                        node.iter.args and
                        isinstance(node.iter.args[0], ast.Call) and
                        isinstance(node.iter.args[0].func, ast.Name) and
                        node.iter.args[0].func.id == 'len'):
                        
                        self.suggestions.append(RefactoringSuggestion(
                            type=RefactoringType.OPTIMIZE_PERFORMANCE,
                            priority=Priority.LOW,
                            title="Use enumerate instead of range(len())",
                            description="Replace range(len()) pattern with enumerate()",
                            file_path=self.file_path,
                            line_start=node.lineno,
                            before_code=f"for {ast.unparse(node.target)} in range(len(...)):",
                            after_code=f"for {ast.unparse(node.target)}, item in enumerate(...):",
                            reasoning="enumerate() is more Pythonic and readable",
                            effort_estimate="low",
                            impact="low",
                            tags=["pythonic", "readability"]
                        ))
                
                def _calculate_complexity(self, node):
                    """Calculate cyclomatic complexity"""
                    complexity = 1
                    for child in ast.walk(node):
                        if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                            complexity += 1
                        elif isinstance(child, ast.ExceptHandler):
                            complexity += 1
                        elif isinstance(child, ast.BoolOp):
                            complexity += len(child.values) - 1
                    return complexity
                
                def _count_boolean_operators(self, node):
                    """Count boolean operators in an expression"""
                    count = 0
                    for child in ast.walk(node):
                        if isinstance(child, ast.BoolOp):
                            count += len(child.values) - 1
                    return count
            
            visitor = RefactoringVisitor(suggestions, lines, file_path)
            visitor.visit(tree)
            
        except SyntaxError:
            # Handle syntax errors gracefully
            logger.debug(f"Syntax error in file {file_path}, skipping AST analysis")
        
        return suggestions
    
    async def _analyze_generic_file(self, file_path: str, content: str, language: str) -> List[RefactoringSuggestion]:
        """Analyze non-Python files using pattern matching"""
        suggestions = []
        lines = content.splitlines()
        
        # Check line length
        for i, line in enumerate(lines, 1):
            if len(line) > 120:
                suggestions.append(RefactoringSuggestion(
                    type=RefactoringType.IMPROVE_READABILITY,
                    priority=Priority.LOW,
                    title="Break long line",
                    description=f"Line {i} is {len(line)} characters long",
                    file_path=file_path,
                    line_start=i,
                    reasoning="Long lines are harder to read and review",
                    effort_estimate="low",
                    impact="low",
                    tags=["formatting", "readability"]
                ))
        
        # Language-specific analysis
        if language == 'javascript':
            suggestions.extend(await self._analyze_javascript_patterns(file_path, content, lines))
        elif language == 'java':
            suggestions.extend(await self._analyze_java_patterns(file_path, content, lines))
        
        return suggestions
    
    async def _analyze_javascript_patterns(self, file_path: str, content: str, lines: List[str]) -> List[RefactoringSuggestion]:
        """Analyze JavaScript-specific patterns"""
        suggestions = []
        
        # Check for var usage
        for i, line in enumerate(lines, 1):
            if re.search(r'\bvar\s+\w+', line):
                suggestions.append(RefactoringSuggestion(
                    type=RefactoringType.IMPROVE_READABILITY,
                    priority=Priority.LOW,
                    title="Use const/let instead of var",
                    description=f"Line {i} uses var declaration",
                    file_path=file_path,
                    line_start=i,
                    before_code=line.strip(),
                    after_code=line.strip().replace('var ', 'const '),
                    reasoning="const/let have better scoping rules than var",
                    effort_estimate="low",
                    impact="low",
                    tags=["modern_js", "scoping"]
                ))
            
            # Check for == usage
            if '==' in line and '===' not in line:
                suggestions.append(RefactoringSuggestion(
                    type=RefactoringType.IMPROVE_READABILITY,
                    priority=Priority.MEDIUM,
                    title="Use strict equality (===)",
                    description=f"Line {i} uses loose equality (==)",
                    file_path=file_path,
                    line_start=i,
                    reasoning="Strict equality prevents type coercion issues",
                    effort_estimate="low",
                    impact="medium",
                    tags=["best_practices", "type_safety"]
                ))
        
        return suggestions
    
    async def _analyze_java_patterns(self, file_path: str, content: str, lines: List[str]) -> List[RefactoringSuggestion]:
        """Analyze Java-specific patterns"""
        suggestions = []
        
        # Check for System.out.println in non-test files
        if 'test' not in file_path.lower():
            for i, line in enumerate(lines, 1):
                if 'System.out.println' in line:
                    suggestions.append(RefactoringSuggestion(
                        type=RefactoringType.IMPROVE_READABILITY,
                        priority=Priority.LOW,
                        title="Replace System.out.println with proper logging",
                        description=f"Line {i} uses System.out.println",
                        file_path=file_path,
                        line_start=i,
                        reasoning="Proper logging is better than console output",
                        effort_estimate="low",
                        impact="medium",
                        tags=["logging", "best_practices"]
                    ))
        
        return suggestions
    
    async def _apply_pattern_suggestions(self, file_path: str, content: str, language: str) -> List[RefactoringSuggestion]:
        """Apply pattern-based refactoring suggestions"""
        suggestions = []
        lines = content.splitlines()
        
        # Check for magic numbers
        for i, line in enumerate(lines, 1):
            # Find numeric literals (excluding 0, 1, -1)
            numbers = re.findall(r'\b(\d{2,})\b', line)
            for number in numbers:
                if int(number) not in [0, 1, 10, 100, 1000]:  # Common acceptable numbers
                    suggestions.append(RefactoringSuggestion(
                        type=RefactoringType.REPLACE_MAGIC_NUMBER,
                        priority=Priority.LOW,
                        title=f"Replace magic number {number}",
                        description=f"Magic number {number} found on line {i}",
                        file_path=file_path,
                        line_start=i,
                        reasoning="Magic numbers make code less maintainable",
                        effort_estimate="low",
                        impact="medium",
                        tags=["maintainability", "constants"]
                    ))
        
        # Check for code duplication (simplified)
        line_counts = Counter(line.strip() for line in lines if line.strip())
        for line_content, count in line_counts.items():
            if count >= 3 and len(line_content) > 20:  # Significant duplication
                # Find line numbers of duplicates
                duplicate_lines = [i+1 for i, line in enumerate(lines) if line.strip() == line_content]
                suggestions.append(RefactoringSuggestion(
                    type=RefactoringType.REMOVE_DUPLICATION,
                    priority=Priority.MEDIUM,
                    title=f"Remove duplicated code",
                    description=f"Code '{line_content[:50]}...' appears {count} times",
                    file_path=file_path,
                    line_start=duplicate_lines[0],
                    reasoning="Code duplication increases maintenance burden",
                    effort_estimate="medium",
                    impact="high",
                    tags=["duplication", "maintainability"],
                    metadata={"duplicate_lines": duplicate_lines}
                ))
        
        return suggestions
    
    async def _apply_performance_suggestions(self, file_path: str, content: str, language: str) -> List[RefactoringSuggestion]:
        """Apply performance-related suggestions"""
        suggestions = []
        
        performance_patterns = self.performance_patterns.get(language, [])
        lines = content.splitlines()
        
        for pattern_info in performance_patterns:
            pattern = pattern_info['pattern']
            suggestion_text = pattern_info['suggestion']
            refactoring_type = pattern_info['refactoring']
            priority = pattern_info['priority']
            
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line):
                    suggestions.append(RefactoringSuggestion(
                        type=refactoring_type,
                        priority=priority,
                        title=suggestion_text,
                        description=f"Performance improvement opportunity on line {i}",
                        file_path=file_path,
                        line_start=i,
                        reasoning="This pattern can be optimized for better performance",
                        effort_estimate="low",
                        impact="medium",
                        tags=["performance", "optimization"]
                    ))
        
        return suggestions
    
    def _deduplicate_suggestions(self, suggestions: List[RefactoringSuggestion]) -> List[RefactoringSuggestion]:
        """Remove duplicate suggestions"""
        seen_ids = set()
        unique_suggestions = []
        
        for suggestion in suggestions:
            suggestion_id = suggestion.id
            if suggestion_id not in seen_ids:
                seen_ids.add(suggestion_id)
                unique_suggestions.append(suggestion)
        
        return unique_suggestions
    
    def _rank_suggestions(self, suggestions: List[RefactoringSuggestion]) -> List[RefactoringSuggestion]:
        """Rank suggestions by priority and impact"""
        priority_order = {
            Priority.CRITICAL: 4,
            Priority.HIGH: 3,
            Priority.MEDIUM: 2,
            Priority.LOW: 1
        }
        
        impact_order = {
            'high': 3,
            'medium': 2,
            'low': 1
        }
        
        def sort_key(suggestion):
            priority_score = priority_order.get(suggestion.priority, 0)
            impact_score = impact_order.get(suggestion.impact, 0)
            return (priority_score, impact_score)
        
        return sorted(suggestions, key=sort_key, reverse=True)
    
    def prioritize_suggestions(self, suggestions: List[RefactoringSuggestion]) -> List[RefactoringSuggestion]:
        """Prioritize suggestions based on impact and effort"""
        # Group suggestions by priority
        priority_groups = defaultdict(list)
        for suggestion in suggestions:
            priority_groups[suggestion.priority].append(suggestion)
        
        # Re-order within each priority group by impact/effort ratio
        prioritized = []
        for priority in [Priority.CRITICAL, Priority.HIGH, Priority.MEDIUM, Priority.LOW]:
            group = priority_groups.get(priority, [])
            
            # Sort by impact/effort ratio
            def effort_impact_score(suggestion):
                effort_score = {'low': 1, 'medium': 2, 'high': 3}.get(suggestion.effort_estimate, 2)
                impact_score = {'low': 1, 'medium': 2, 'high': 3}.get(suggestion.impact, 2)
                return impact_score / effort_score
            
            group.sort(key=effort_impact_score, reverse=True)
            prioritized.extend(group)
        
        return prioritized
    
    def group_suggestions_by_type(self, suggestions: List[RefactoringSuggestion]) -> Dict[RefactoringType, List[RefactoringSuggestion]]:
        """Group suggestions by refactoring type"""
        groups = defaultdict(list)
        for suggestion in suggestions:
            groups[suggestion.type].append(suggestion)
        return dict(groups)
    
    def filter_suggestions(
        self,
        suggestions: List[RefactoringSuggestion],
        min_priority: Priority = Priority.LOW,
        types: List[RefactoringType] = None,
        tags: List[str] = None
    ) -> List[RefactoringSuggestion]:
        """Filter suggestions by criteria"""
        priority_order = {
            Priority.LOW: 1,
            Priority.MEDIUM: 2,
            Priority.HIGH: 3,
            Priority.CRITICAL: 4
        }
        
        min_priority_value = priority_order[min_priority]
        
        filtered = []
        for suggestion in suggestions:
            # Check priority
            if priority_order[suggestion.priority] < min_priority_value:
                continue
            
            # Check types
            if types and suggestion.type not in types:
                continue
            
            # Check tags
            if tags and not any(tag in suggestion.tags for tag in tags):
                continue
            
            filtered.append(suggestion)
        
        return filtered
    
    def get_suggestion_summary(self, suggestions: List[RefactoringSuggestion]) -> Dict[str, Any]:
        """Get summary statistics of suggestions"""
        if not suggestions:
            return {'total': 0}
        
        by_priority = Counter(s.priority.value for s in suggestions)
        by_type = Counter(s.type.value for s in suggestions)
        by_effort = Counter(s.effort_estimate for s in suggestions)
        by_impact = Counter(s.impact for s in suggestions)
        
        all_tags = []
        for s in suggestions:
            all_tags.extend(s.tags)
        common_tags = Counter(all_tags)
        
        return {
            'total': len(suggestions),
            'by_priority': dict(by_priority),
            'by_type': dict(by_type),
            'by_effort': dict(by_effort),
            'by_impact': dict(by_impact),
            'common_tags': dict(common_tags.most_common(10)),
            'high_priority_count': by_priority.get('high', 0) + by_priority.get('critical', 0),
            'quick_wins': len([s for s in suggestions if s.effort_estimate == 'low' and s.impact in ['medium', 'high']])
        }
    
    def export_suggestions(self, suggestions: List[RefactoringSuggestion], format: str = 'json') -> str:
        """Export suggestions in various formats"""
        if format == 'json':
            data = []
            for suggestion in suggestions:
                data.append({
                    'id': suggestion.id,
                    'type': suggestion.type.value,
                    'priority': suggestion.priority.value,
                    'title': suggestion.title,
                    'description': suggestion.description,
                    'file_path': suggestion.file_path,
                    'line_start': suggestion.line_start,
                    'line_end': suggestion.line_end,
                    'reasoning': suggestion.reasoning,
                    'effort_estimate': suggestion.effort_estimate,
                    'impact': suggestion.impact,
                    'tags': suggestion.tags,
                    'before_code': suggestion.before_code,
                    'after_code': suggestion.after_code
                })
            return json.dumps(data, indent=2)
        
        elif format == 'markdown':
            md_lines = ["# Refactoring Suggestions\n"]
            
            # Group by priority
            by_priority = self.group_suggestions_by_priority(suggestions)
            
            for priority in [Priority.CRITICAL, Priority.HIGH, Priority.MEDIUM, Priority.LOW]:
                priority_suggestions = by_priority.get(priority, [])
                if not priority_suggestions:
                    continue
                
                md_lines.append(f"## {priority.value.title()} Priority ({len(priority_suggestions)} suggestions)\n")
                
                for suggestion in priority_suggestions:
                    md_lines.extend([
                        f"### {suggestion.title}",
                        f"**File:** {suggestion.file_path}:{suggestion.line_start}",
                        f"**Type:** {suggestion.type.value}",
                        f"**Impact:** {suggestion.impact} | **Effort:** {suggestion.effort_estimate}",
                        f"**Description:** {suggestion.description}",
                        f"**Reasoning:** {suggestion.reasoning}",
                        ""
                    ])
                    
                    if suggestion.before_code and suggestion.after_code:
                        md_lines.extend([
                            "**Before:**",
                            f"```",
                            suggestion.before_code,
                            "```",
                            "**After:**",
                            f"```", 
                            suggestion.after_code,
                            "```",
                            ""
                        ])
                
                md_lines.append("")
            
            return '\n'.join(md_lines)
        
        elif format == 'csv':
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Header
            writer.writerow([
                'ID', 'Type', 'Priority', 'Title', 'File', 'Line', 
                'Effort', 'Impact', 'Tags', 'Description'
            ])
            
            # Data
            for suggestion in suggestions:
                writer.writerow([
                    suggestion.id,
                    suggestion.type.value,
                    suggestion.priority.value,
                    suggestion.title,
                    suggestion.file_path,
                    suggestion.line_start,
                    suggestion.effort_estimate,
                    suggestion.impact,
                    ', '.join(suggestion.tags),
                    suggestion.description
                ])
            
            return output.getvalue()
        
        return "Unsupported format"
    
    def group_suggestions_by_priority(self, suggestions: List[RefactoringSuggestion]) -> Dict[Priority, List[RefactoringSuggestion]]:
        """Group suggestions by priority"""
        groups = defaultdict(list)
        for suggestion in suggestions:
            groups[suggestion.priority].append(suggestion)
        return dict(groups)
    
    def find_related_suggestions(self, suggestions: List[RefactoringSuggestion]) -> List[RefactoringSuggestion]:
        """Find and link related suggestions"""
        # Simple implementation - could be enhanced with more sophisticated analysis
        for i, suggestion in enumerate(suggestions):
            related = []
            
            # Find suggestions in the same file
            same_file = [s for j, s in enumerate(suggestions) 
                        if i != j and s.file_path == suggestion.file_path]
            
            # Find suggestions of complementary types
            if suggestion.type == RefactoringType.EXTRACT_METHOD:
                related.extend([s.id for s in same_file 
                              if s.type in [RefactoringType.EXTRACT_VARIABLE, RefactoringType.SIMPLIFY_CONDITIONAL]])
            
            suggestion.related_suggestions = related[:3]  # Limit to 3 related suggestions
        
        return suggestions
    
    async def generate_refactoring_plan(self, suggestions: List[RefactoringSuggestion]) -> Dict[str, Any]:
        """Generate a refactoring execution plan"""
        # Prioritize suggestions
        prioritized = self.prioritize_suggestions(suggestions)
        
        # Group into phases
        phases = {
            'quick_wins': [],  # Low effort, medium/high impact
            'high_priority': [],  # High/critical priority
            'technical_debt': [],  # Maintainability improvements
            'optimization': []  # Performance improvements
        }
        
        for suggestion in prioritized:
            if (suggestion.effort_estimate == 'low' and 
                suggestion.impact in ['medium', 'high']):
                phases['quick_wins'].append(suggestion)
            elif suggestion.priority in [Priority.CRITICAL, Priority.HIGH]:
                phases['high_priority'].append(suggestion)
            elif 'maintainability' in suggestion.tags:
                phases['technical_debt'].append(suggestion)
            elif 'performance' in suggestion.tags:
                phases['optimization'].append(suggestion)
        
        # Estimate effort for each phase
        effort_estimates = {}
        for phase, phase_suggestions in phases.items():
            effort_map = {'low': 1, 'medium': 3, 'high': 8}
            total_effort = sum(effort_map.get(s.effort_estimate, 3) for s in phase_suggestions)
            effort_estimates[phase] = f"{total_effort} story points"
        
        return {
            'phases': {k: [s.id for s in v] for k, v in phases.items()},
            'effort_estimates': effort_estimates,
            'total_suggestions': len(suggestions),
            'estimated_total_effort': sum(len(v) * 2 for v in phases.values()),  # Rough estimate
            'recommended_order': ['quick_wins', 'high_priority', 'technical_debt', 'optimization']
        }