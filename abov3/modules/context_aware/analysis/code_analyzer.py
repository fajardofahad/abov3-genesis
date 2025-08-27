"""
ABOV3 Genesis - Code Analyzer
Advanced code analysis for intelligent comprehension and insights
"""

import ast
import logging
import time
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum
import json

logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    """Types of code analysis"""
    STRUCTURE = "structure"
    COMPLEXITY = "complexity"
    QUALITY = "quality"
    DEPENDENCIES = "dependencies"
    PATTERNS = "patterns"
    ISSUES = "issues"
    METRICS = "metrics"
    SECURITY = "security"

class IssueLevel(Enum):
    """Severity levels for code issues"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class CodeIssue:
    """Represents a code issue or problem"""
    issue_type: str
    level: IssueLevel
    message: str
    file_path: str
    line_number: int
    column: Optional[int] = None
    suggestion: Optional[str] = None
    rule_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CodeMetrics:
    """Code quality metrics"""
    lines_of_code: int
    cyclomatic_complexity: int
    cognitive_complexity: int
    maintainability_index: float
    technical_debt_ratio: float
    code_duplication: float
    test_coverage: Optional[float] = None
    dependencies_count: int = 0
    security_issues: int = 0
    performance_issues: int = 0
    
@dataclass
class StructureInfo:
    """Code structure information"""
    classes: List[Dict[str, Any]]
    functions: List[Dict[str, Any]]
    imports: List[str]
    global_variables: List[str]
    constants: List[str]
    decorators: List[str]
    inheritance_tree: Dict[str, List[str]]
    call_graph: Dict[str, List[str]]

@dataclass
class AnalysisResult:
    """Complete analysis result for a file"""
    file_path: str
    language: str
    analysis_time: float
    metrics: CodeMetrics
    structure: StructureInfo
    issues: List[CodeIssue]
    patterns: List[Dict[str, Any]]
    dependencies: List[str]
    summary: str
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

class CodeAnalyzer:
    """
    Advanced code analyzer for intelligent code comprehension
    Analyzes structure, quality, patterns, and issues
    """
    
    def __init__(self):
        # Analysis rules and patterns
        self.quality_rules = self._initialize_quality_rules()
        self.security_patterns = self._initialize_security_patterns()
        self.performance_patterns = self._initialize_performance_patterns()
        self.code_smells = self._initialize_code_smells()
        
        # Metrics thresholds
        self.complexity_thresholds = {
            'low': 5,
            'medium': 10,
            'high': 20,
            'critical': 50
        }
        
        self.maintainability_thresholds = {
            'excellent': 85,
            'good': 70,
            'acceptable': 50,
            'poor': 30
        }
        
        logger.info("CodeAnalyzer initialized")
    
    def _initialize_quality_rules(self) -> Dict[str, Dict]:
        """Initialize code quality rules"""
        return {
            'python': {
                'line_length': {'max': 88, 'severity': IssueLevel.LOW},
                'function_length': {'max': 50, 'severity': IssueLevel.MEDIUM},
                'class_length': {'max': 200, 'severity': IssueLevel.MEDIUM},
                'parameter_count': {'max': 5, 'severity': IssueLevel.MEDIUM},
                'nesting_depth': {'max': 4, 'severity': IssueLevel.HIGH},
                'variable_naming': {'pattern': r'^[a-z_][a-z0-9_]*$', 'severity': IssueLevel.LOW},
                'function_naming': {'pattern': r'^[a-z_][a-z0-9_]*$', 'severity': IssueLevel.LOW},
                'class_naming': {'pattern': r'^[A-Z][a-zA-Z0-9]*$', 'severity': IssueLevel.LOW},
                'constant_naming': {'pattern': r'^[A-Z_][A-Z0-9_]*$', 'severity': IssueLevel.LOW}
            },
            'javascript': {
                'line_length': {'max': 100, 'severity': IssueLevel.LOW},
                'function_length': {'max': 50, 'severity': IssueLevel.MEDIUM},
                'parameter_count': {'max': 4, 'severity': IssueLevel.MEDIUM},
                'nesting_depth': {'max': 4, 'severity': IssueLevel.HIGH},
                'variable_naming': {'pattern': r'^[a-zA-Z_$][a-zA-Z0-9_$]*$', 'severity': IssueLevel.LOW},
                'function_naming': {'pattern': r'^[a-z][a-zA-Z0-9]*$', 'severity': IssueLevel.LOW},
                'class_naming': {'pattern': r'^[A-Z][a-zA-Z0-9]*$', 'severity': IssueLevel.LOW}
            }
        }
    
    def _initialize_security_patterns(self) -> Dict[str, List[Dict]]:
        """Initialize security vulnerability patterns"""
        return {
            'python': [
                {'pattern': r'eval\s*\(', 'issue': 'Code injection risk', 'severity': IssueLevel.CRITICAL},
                {'pattern': r'exec\s*\(', 'issue': 'Code injection risk', 'severity': IssueLevel.CRITICAL},
                {'pattern': r'os\.system\s*\(', 'issue': 'Command injection risk', 'severity': IssueLevel.HIGH},
                {'pattern': r'subprocess\.call\s*\(.*shell=True', 'issue': 'Shell injection risk', 'severity': IssueLevel.HIGH},
                {'pattern': r'pickle\.loads?\s*\(', 'issue': 'Deserialization risk', 'severity': IssueLevel.HIGH},
                {'pattern': r'input\s*\(', 'issue': 'User input without validation', 'severity': IssueLevel.MEDIUM},
                {'pattern': r'random\.random\s*\(', 'issue': 'Weak random number generator', 'severity': IssueLevel.LOW},
                {'pattern': r'md5\s*\(', 'issue': 'Weak hash function', 'severity': IssueLevel.MEDIUM},
                {'pattern': r'sha1\s*\(', 'issue': 'Weak hash function', 'severity': IssueLevel.MEDIUM}
            ],
            'javascript': [
                {'pattern': r'eval\s*\(', 'issue': 'Code injection risk', 'severity': IssueLevel.CRITICAL},
                {'pattern': r'setTimeout\s*\(\s*["\']', 'issue': 'Code injection via setTimeout', 'severity': IssueLevel.HIGH},
                {'pattern': r'setInterval\s*\(\s*["\']', 'issue': 'Code injection via setInterval', 'severity': IssueLevel.HIGH},
                {'pattern': r'document\.write\s*\(', 'issue': 'XSS vulnerability', 'severity': IssueLevel.HIGH},
                {'pattern': r'innerHTML\s*=', 'issue': 'Potential XSS', 'severity': IssueLevel.MEDIUM},
                {'pattern': r'Math\.random\s*\(', 'issue': 'Weak random number generator', 'severity': IssueLevel.LOW}
            ]
        }
    
    def _initialize_performance_patterns(self) -> Dict[str, List[Dict]]:
        """Initialize performance anti-patterns"""
        return {
            'python': [
                {'pattern': r'for\s+\w+\s+in\s+range\s*\(\s*len\s*\(', 'issue': 'Inefficient iteration', 'severity': IssueLevel.LOW},
                {'pattern': r'\+\s*=.*\[.*\]', 'issue': 'Inefficient list concatenation', 'severity': IssueLevel.MEDIUM},
                {'pattern': r'\.append\s*\(.*\)\s*$', 'issue': 'Consider list comprehension', 'severity': IssueLevel.LOW},
                {'pattern': r'global\s+\w+', 'issue': 'Global variable usage', 'severity': IssueLevel.MEDIUM}
            ],
            'javascript': [
                {'pattern': r'document\.getElementById', 'issue': 'Consider caching DOM queries', 'severity': IssueLevel.LOW},
                {'pattern': r'\.innerHTML\s*\+=', 'issue': 'Inefficient DOM manipulation', 'severity': IssueLevel.MEDIUM},
                {'pattern': r'for\s*\(\s*var\s+\w+\s*=\s*0', 'issue': 'Consider modern iteration', 'severity': IssueLevel.LOW}
            ]
        }
    
    def _initialize_code_smells(self) -> Dict[str, List[Dict]]:
        """Initialize code smell patterns"""
        return {
            'common': [
                {'pattern': r'^.{200,}$', 'issue': 'Long line', 'severity': IssueLevel.LOW},
                {'pattern': r'TODO|FIXME|HACK|BUG', 'issue': 'TODO/FIXME comment', 'severity': IssueLevel.INFO},
                {'pattern': r'^\s*#.*', 'issue': 'Commented code', 'severity': IssueLevel.LOW, 'language': 'python'},
                {'pattern': r'^\s*//.*', 'issue': 'Commented code', 'severity': IssueLevel.LOW, 'language': 'javascript'}
            ],
            'python': [
                {'pattern': r'except\s*:', 'issue': 'Bare except clause', 'severity': IssueLevel.HIGH},
                {'pattern': r'pass\s*$', 'issue': 'Empty implementation', 'severity': IssueLevel.LOW},
                {'pattern': r'import\s+\*', 'issue': 'Wildcard import', 'severity': IssueLevel.MEDIUM},
                {'pattern': r'lambda.*:', 'issue': 'Complex lambda', 'severity': IssueLevel.LOW}
            ],
            'javascript': [
                {'pattern': r'==\s*', 'issue': 'Use strict equality (===)', 'severity': IssueLevel.MEDIUM},
                {'pattern': r'!=\s*', 'issue': 'Use strict inequality (!==)', 'severity': IssueLevel.MEDIUM},
                {'pattern': r'var\s+', 'issue': 'Use let/const instead of var', 'severity': IssueLevel.LOW},
                {'pattern': r'function\s*\(\s*\)\s*{\s*}', 'issue': 'Empty function', 'severity': IssueLevel.LOW}
            ]
        }
    
    async def analyze_file(
        self, 
        file_path: str, 
        include_ast: bool = True,
        include_metrics: bool = True,
        include_issues: bool = True,
        include_patterns: bool = True,
        include_dependencies: bool = True
    ) -> Optional[AnalysisResult]:
        """Perform comprehensive analysis of a code file"""
        start_time = time.time()
        
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return None
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Determine language
            language = self._detect_language(file_path)
            
            # Initialize analysis components
            metrics = None
            structure = None
            issues = []
            patterns = []
            dependencies = []
            
            # Perform different types of analysis based on parameters
            if include_ast and language == 'python':
                structure = await self._analyze_python_structure(content, str(file_path))
                
            if include_metrics:
                metrics = await self._calculate_metrics(content, language, structure)
            
            if include_issues:
                issues = await self._find_code_issues(content, language, str(file_path))
            
            if include_patterns:
                patterns = await self._identify_patterns(content, language)
            
            if include_dependencies:
                dependencies = await self._extract_dependencies(content, language)
            
            # Generate summary and recommendations
            summary = self._generate_analysis_summary(metrics, issues, patterns, language)
            recommendations = self._generate_recommendations(metrics, issues, patterns, language)
            
            analysis_time = time.time() - start_time
            
            return AnalysisResult(
                file_path=str(file_path),
                language=language,
                analysis_time=analysis_time,
                metrics=metrics,
                structure=structure,
                issues=issues,
                patterns=patterns,
                dependencies=dependencies,
                summary=summary,
                recommendations=recommendations,
                metadata={
                    'file_size': file_path.stat().st_size,
                    'lines_count': len(content.splitlines()),
                    'analysis_timestamp': time.time()
                }
            )
            
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            return None
    
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
            '.rb': 'ruby',
            '.swift': 'swift',
            '.kt': 'kotlin'
        }
        
        return extension_map.get(file_path.suffix.lower(), 'unknown')
    
    async def _analyze_python_structure(self, content: str, file_path: str) -> StructureInfo:
        """Analyze Python code structure using AST"""
        try:
            tree = ast.parse(content)
            
            classes = []
            functions = []
            imports = []
            global_variables = []
            constants = []
            decorators = []
            inheritance_tree = defaultdict(list)
            call_graph = defaultdict(list)
            
            class StructureVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.current_class = None
                    self.current_function = None
                
                def visit_ClassDef(self, node):
                    class_info = {
                        'name': node.name,
                        'line_start': node.lineno,
                        'line_end': getattr(node, 'end_lineno', node.lineno),
                        'methods': [],
                        'properties': [],
                        'base_classes': [self._get_name(base) for base in node.bases],
                        'decorators': [self._get_name(dec) for dec in node.decorator_list],
                        'docstring': ast.get_docstring(node),
                        'complexity': self._calculate_class_complexity(node)
                    }
                    
                    classes.append(class_info)
                    
                    # Record inheritance
                    for base in node.bases:
                        base_name = self._get_name(base)
                        inheritance_tree[base_name].append(node.name)
                    
                    old_class = self.current_class
                    self.current_class = node.name
                    self.generic_visit(node)
                    self.current_class = old_class
                
                def visit_FunctionDef(self, node):
                    func_info = {
                        'name': node.name,
                        'line_start': node.lineno,
                        'line_end': getattr(node, 'end_lineno', node.lineno),
                        'parameters': [arg.arg for arg in node.args.args],
                        'decorators': [self._get_name(dec) for dec in node.decorator_list],
                        'docstring': ast.get_docstring(node),
                        'complexity': self._calculate_cyclomatic_complexity(node),
                        'cognitive_complexity': self._calculate_cognitive_complexity(node),
                        'is_method': self.current_class is not None,
                        'is_async': isinstance(node, ast.AsyncFunctionDef),
                        'class_name': self.current_class
                    }
                    
                    functions.append(func_info)
                    
                    # Extract function calls for call graph
                    calls = self._extract_function_calls(node)
                    call_graph[node.name].extend(calls)
                    
                    # Record decorators
                    for dec in node.decorator_list:
                        dec_name = self._get_name(dec)
                        if dec_name not in decorators:
                            decorators.append(dec_name)
                    
                    old_function = self.current_function
                    self.current_function = node.name
                    self.generic_visit(node)
                    self.current_function = old_function
                
                def visit_AsyncFunctionDef(self, node):
                    self.visit_FunctionDef(node)
                
                def visit_Import(self, node):
                    for alias in node.names:
                        imports.append(alias.name)
                
                def visit_ImportFrom(self, node):
                    module = node.module or ''
                    for alias in node.names:
                        full_name = f"{module}.{alias.name}" if module else alias.name
                        imports.append(full_name)
                
                def visit_Assign(self, node):
                    # Global variables and constants
                    if self.current_class is None and self.current_function is None:
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                if target.id.isupper():
                                    constants.append(target.id)
                                else:
                                    global_variables.append(target.id)
                    
                    self.generic_visit(node)
                
                def _get_name(self, node):
                    """Get name from AST node"""
                    if isinstance(node, ast.Name):
                        return node.id
                    elif isinstance(node, ast.Attribute):
                        return f"{self._get_name(node.value)}.{node.attr}"
                    elif isinstance(node, ast.Call):
                        return self._get_name(node.func)
                    else:
                        return str(node)
                
                def _calculate_class_complexity(self, node):
                    """Calculate complexity of a class"""
                    complexity = 1  # Base complexity
                    for child in ast.walk(node):
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            complexity += self._calculate_cyclomatic_complexity(child)
                    return complexity
                
                def _calculate_cyclomatic_complexity(self, node):
                    """Calculate cyclomatic complexity"""
                    complexity = 1
                    for child in ast.walk(node):
                        if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                            complexity += 1
                        elif isinstance(child, ast.ExceptHandler):
                            complexity += 1
                        elif isinstance(child, (ast.With, ast.AsyncWith)):
                            complexity += 1
                        elif isinstance(child, ast.BoolOp):
                            complexity += len(child.values) - 1
                    return complexity
                
                def _calculate_cognitive_complexity(self, node):
                    """Calculate cognitive complexity (simplified)"""
                    cognitive = 0
                    nesting_level = 0
                    
                    for child in ast.walk(node):
                        if isinstance(child, (ast.If, ast.While, ast.For)):
                            cognitive += 1 + nesting_level
                            nesting_level += 1
                        elif isinstance(child, ast.ExceptHandler):
                            cognitive += 1 + nesting_level
                        elif isinstance(child, ast.BoolOp):
                            cognitive += 1
                    
                    return cognitive
                
                def _extract_function_calls(self, node):
                    """Extract function calls from a node"""
                    calls = []
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            if isinstance(child.func, ast.Name):
                                calls.append(child.func.id)
                            elif isinstance(child.func, ast.Attribute):
                                calls.append(child.func.attr)
                    return calls
            
            visitor = StructureVisitor()
            visitor.visit(tree)
            
            return StructureInfo(
                classes=classes,
                functions=functions,
                imports=imports,
                global_variables=global_variables,
                constants=constants,
                decorators=decorators,
                inheritance_tree=dict(inheritance_tree),
                call_graph=dict(call_graph)
            )
            
        except SyntaxError as e:
            logger.debug(f"Syntax error parsing Python file: {e}")
            return StructureInfo([], [], [], [], [], [], {}, {})
        except Exception as e:
            logger.error(f"Error analyzing Python structure: {e}")
            return StructureInfo([], [], [], [], [], [], {}, {})
    
    async def _calculate_metrics(self, content: str, language: str, structure: StructureInfo = None) -> CodeMetrics:
        """Calculate code quality metrics"""
        lines = content.splitlines()
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Basic metrics
        lines_of_code = len(non_empty_lines)
        
        # Complexity metrics
        if structure and structure.functions:
            total_complexity = sum(func.get('complexity', 1) for func in structure.functions)
            avg_complexity = total_complexity / len(structure.functions)
            cyclomatic_complexity = max(func.get('complexity', 1) for func in structure.functions)
            
            total_cognitive = sum(func.get('cognitive_complexity', 1) for func in structure.functions)
            cognitive_complexity = max(func.get('cognitive_complexity', 1) for func in structure.functions)
        else:
            # Simplified complexity calculation for non-Python files
            cyclomatic_complexity = self._estimate_complexity(content, language)
            cognitive_complexity = cyclomatic_complexity
        
        # Maintainability Index (simplified calculation)
        volume = len(content) * 0.1  # Simplified Halstead volume
        maintainability_index = max(0, min(100, 
            171 - 5.2 * (cyclomatic_complexity / 10) - 0.23 * volume - 16.2 * (lines_of_code / 1000)
        ))
        
        # Technical debt estimation
        technical_debt_ratio = self._calculate_technical_debt_ratio(content, language)
        
        # Code duplication (simplified)
        code_duplication = self._estimate_code_duplication(content)
        
        # Dependencies count
        dependencies_count = len(structure.imports) if structure else len(self._extract_imports(content, language))
        
        return CodeMetrics(
            lines_of_code=lines_of_code,
            cyclomatic_complexity=cyclomatic_complexity,
            cognitive_complexity=cognitive_complexity,
            maintainability_index=maintainability_index,
            technical_debt_ratio=technical_debt_ratio,
            code_duplication=code_duplication,
            dependencies_count=dependencies_count,
            security_issues=0,  # Will be set by issue analysis
            performance_issues=0  # Will be set by issue analysis
        )
    
    def _estimate_complexity(self, content: str, language: str) -> int:
        """Estimate complexity for non-Python languages"""
        complexity = 1
        
        # Count control structures
        if_patterns = r'\b(if|while|for|switch|case)\b'
        complexity += len(re.findall(if_patterns, content, re.IGNORECASE))
        
        # Count exception handlers
        exception_patterns = r'\b(try|catch|except|finally)\b'
        complexity += len(re.findall(exception_patterns, content, re.IGNORECASE))
        
        return complexity
    
    def _calculate_technical_debt_ratio(self, content: str, language: str) -> float:
        """Calculate technical debt ratio (0.0 to 1.0)"""
        debt_indicators = [
            r'TODO',
            r'FIXME',
            r'HACK',
            r'XXX',
            r'KLUDGE',
            r'BUG',
            r'WORKAROUND'
        ]
        
        total_debt = 0
        for pattern in debt_indicators:
            total_debt += len(re.findall(pattern, content, re.IGNORECASE))
        
        lines_count = len(content.splitlines())
        return min(1.0, total_debt / max(1, lines_count / 10))
    
    def _estimate_code_duplication(self, content: str) -> float:
        """Estimate code duplication percentage"""
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        
        if not lines:
            return 0.0
        
        # Simple line-based duplication detection
        line_counts = Counter(lines)
        duplicated_lines = sum(count - 1 for count in line_counts.values() if count > 1)
        
        return min(1.0, duplicated_lines / len(lines))
    
    def _extract_imports(self, content: str, language: str) -> List[str]:
        """Extract imports for non-Python languages"""
        imports = []
        
        if language in ['javascript', 'typescript']:
            # ES6 imports
            import_pattern = r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]'
            imports.extend(re.findall(import_pattern, content))
            
            # Require statements
            require_pattern = r'require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)'
            imports.extend(re.findall(require_pattern, content))
        
        elif language == 'java':
            import_pattern = r'import\s+([\w\.]+);'
            imports.extend(re.findall(import_pattern, content))
        
        elif language in ['cpp', 'c']:
            include_pattern = r'#include\s*[<"](.*?)[>"]'
            imports.extend(re.findall(include_pattern, content))
        
        return imports
    
    async def _find_code_issues(self, content: str, language: str, file_path: str) -> List[CodeIssue]:
        """Find code issues and problems"""
        issues = []
        lines = content.splitlines()
        
        # Apply quality rules
        quality_rules = self.quality_rules.get(language, {})
        issues.extend(self._apply_quality_rules(content, lines, quality_rules, file_path))
        
        # Apply security checks
        security_patterns = self.security_patterns.get(language, [])
        issues.extend(self._apply_security_patterns(content, security_patterns, file_path))
        
        # Apply performance checks
        performance_patterns = self.performance_patterns.get(language, [])
        issues.extend(self._apply_performance_patterns(content, performance_patterns, file_path))
        
        # Apply code smell detection
        code_smells = self.code_smells.get('common', []) + self.code_smells.get(language, [])
        issues.extend(self._apply_code_smell_patterns(content, lines, code_smells, file_path))
        
        return issues
    
    def _apply_quality_rules(self, content: str, lines: List[str], rules: Dict, file_path: str) -> List[CodeIssue]:
        """Apply quality rules"""
        issues = []
        
        # Line length check
        if 'line_length' in rules:
            max_length = rules['line_length']['max']
            severity = rules['line_length']['severity']
            
            for i, line in enumerate(lines, 1):
                if len(line) > max_length:
                    issues.append(CodeIssue(
                        issue_type='line_length',
                        level=severity,
                        message=f'Line too long ({len(line)} > {max_length})',
                        file_path=file_path,
                        line_number=i,
                        suggestion=f'Break line into multiple lines or refactor'
                    ))
        
        # Function length check (simplified for non-AST analysis)
        if 'function_length' in rules:
            max_length = rules['function_length']['max']
            severity = rules['function_length']['severity']
            
            # Simple function detection
            func_pattern = r'^(\s*)(def|function|async\s+function)\s+(\w+)'
            current_func = None
            func_start = 0
            
            for i, line in enumerate(lines):
                match = re.match(func_pattern, line)
                if match:
                    if current_func:
                        func_length = i - func_start
                        if func_length > max_length:
                            issues.append(CodeIssue(
                                issue_type='function_length',
                                level=severity,
                                message=f'Function too long ({func_length} lines > {max_length})',
                                file_path=file_path,
                                line_number=func_start + 1,
                                suggestion='Break function into smaller functions'
                            ))
                    
                    current_func = match.group(3)
                    func_start = i
        
        return issues
    
    def _apply_security_patterns(self, content: str, patterns: List[Dict], file_path: str) -> List[CodeIssue]:
        """Apply security vulnerability patterns"""
        issues = []
        lines = content.splitlines()
        
        for pattern_info in patterns:
            pattern = pattern_info['pattern']
            issue_msg = pattern_info['issue']
            severity = pattern_info['severity']
            
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(CodeIssue(
                        issue_type='security',
                        level=severity,
                        message=issue_msg,
                        file_path=file_path,
                        line_number=i,
                        rule_name=f'security_{pattern_info.get("rule", "unknown")}',
                        suggestion='Review security implications and use safer alternatives'
                    ))
        
        return issues
    
    def _apply_performance_patterns(self, content: str, patterns: List[Dict], file_path: str) -> List[CodeIssue]:
        """Apply performance anti-patterns"""
        issues = []
        lines = content.splitlines()
        
        for pattern_info in patterns:
            pattern = pattern_info['pattern']
            issue_msg = pattern_info['issue']
            severity = pattern_info['severity']
            
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line):
                    issues.append(CodeIssue(
                        issue_type='performance',
                        level=severity,
                        message=issue_msg,
                        file_path=file_path,
                        line_number=i,
                        suggestion='Consider performance optimization'
                    ))
        
        return issues
    
    def _apply_code_smell_patterns(self, content: str, lines: List[str], patterns: List[Dict], file_path: str) -> List[CodeIssue]:
        """Apply code smell detection"""
        issues = []
        
        for pattern_info in patterns:
            pattern = pattern_info['pattern']
            issue_msg = pattern_info['issue']
            severity = pattern_info['severity']
            
            # Skip if language-specific and doesn't match
            if 'language' in pattern_info:
                # This would need proper language detection
                continue
            
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(CodeIssue(
                        issue_type='code_smell',
                        level=severity,
                        message=issue_msg,
                        file_path=file_path,
                        line_number=i,
                        suggestion='Consider refactoring'
                    ))
        
        return issues
    
    async def _identify_patterns(self, content: str, language: str) -> List[Dict[str, Any]]:
        """Identify code patterns and architectural patterns"""
        patterns = []
        
        # Design patterns detection
        patterns.extend(self._detect_design_patterns(content, language))
        
        # Architectural patterns
        patterns.extend(self._detect_architectural_patterns(content, language))
        
        # Common code patterns
        patterns.extend(self._detect_common_patterns(content, language))
        
        return patterns
    
    def _detect_design_patterns(self, content: str, language: str) -> List[Dict[str, Any]]:
        """Detect common design patterns"""
        patterns = []
        
        if language == 'python':
            # Singleton pattern
            if re.search(r'class.*Singleton|def __new__\(.*\):', content, re.IGNORECASE):
                patterns.append({
                    'type': 'design_pattern',
                    'name': 'Singleton',
                    'confidence': 0.8,
                    'description': 'Singleton pattern implementation detected'
                })
            
            # Factory pattern
            if re.search(r'def create|def factory|Factory.*class', content, re.IGNORECASE):
                patterns.append({
                    'type': 'design_pattern',
                    'name': 'Factory',
                    'confidence': 0.7,
                    'description': 'Factory pattern implementation detected'
                })
            
            # Observer pattern
            if re.search(r'def notify|def subscribe|def unsubscribe', content, re.IGNORECASE):
                patterns.append({
                    'type': 'design_pattern',
                    'name': 'Observer',
                    'confidence': 0.6,
                    'description': 'Observer pattern implementation detected'
                })
            
            # Decorator pattern
            if re.search(r'@\w+|def decorator|def wrapper', content):
                patterns.append({
                    'type': 'design_pattern',
                    'name': 'Decorator',
                    'confidence': 0.9,
                    'description': 'Decorator pattern usage detected'
                })
        
        return patterns
    
    def _detect_architectural_patterns(self, content: str, language: str) -> List[Dict[str, Any]]:
        """Detect architectural patterns"""
        patterns = []
        
        # MVC pattern
        if re.search(r'(Model|View|Controller|models|views|controllers)', content, re.IGNORECASE):
            patterns.append({
                'type': 'architectural_pattern',
                'name': 'MVC',
                'confidence': 0.6,
                'description': 'MVC architectural pattern detected'
            })
        
        # API/REST patterns
        if re.search(r'@app\.route|@router\.|GET|POST|PUT|DELETE|api|endpoint', content, re.IGNORECASE):
            patterns.append({
                'type': 'architectural_pattern',
                'name': 'REST_API',
                'confidence': 0.8,
                'description': 'REST API pattern detected'
            })
        
        # Database patterns
        if re.search(r'SELECT|INSERT|UPDATE|DELETE|query|session|transaction', content, re.IGNORECASE):
            patterns.append({
                'type': 'architectural_pattern',
                'name': 'Database_Access',
                'confidence': 0.7,
                'description': 'Database access pattern detected'
            })
        
        # Async patterns
        if re.search(r'async|await|asyncio|Future|Task', content):
            patterns.append({
                'type': 'architectural_pattern',
                'name': 'Async_Programming',
                'confidence': 0.9,
                'description': 'Asynchronous programming pattern detected'
            })
        
        return patterns
    
    def _detect_common_patterns(self, content: str, language: str) -> List[Dict[str, Any]]:
        """Detect common coding patterns"""
        patterns = []
        
        # Error handling patterns
        error_handling = len(re.findall(r'try|except|catch|throw|raise', content, re.IGNORECASE))
        if error_handling > 0:
            patterns.append({
                'type': 'coding_pattern',
                'name': 'Error_Handling',
                'confidence': min(1.0, error_handling * 0.2),
                'description': f'Error handling patterns found ({error_handling} instances)'
            })
        
        # Logging patterns
        logging_patterns = len(re.findall(r'log\.|logger\.|print\(|console\.log', content, re.IGNORECASE))
        if logging_patterns > 0:
            patterns.append({
                'type': 'coding_pattern',
                'name': 'Logging',
                'confidence': min(1.0, logging_patterns * 0.1),
                'description': f'Logging patterns found ({logging_patterns} instances)'
            })
        
        # Testing patterns
        testing_patterns = len(re.findall(r'test_|Test|assert|expect|describe|it\(', content))
        if testing_patterns > 0:
            patterns.append({
                'type': 'coding_pattern',
                'name': 'Testing',
                'confidence': min(1.0, testing_patterns * 0.15),
                'description': f'Testing patterns found ({testing_patterns} instances)'
            })
        
        return patterns
    
    async def _extract_dependencies(self, content: str, language: str) -> List[str]:
        """Extract dependencies from code"""
        return self._extract_imports(content, language)
    
    def _generate_analysis_summary(
        self, 
        metrics: CodeMetrics, 
        issues: List[CodeIssue], 
        patterns: List[Dict[str, Any]], 
        language: str
    ) -> str:
        """Generate a summary of the analysis"""
        summary_parts = []
        
        if metrics:
            # Code size summary
            summary_parts.append(f"{language.capitalize()} file with {metrics.lines_of_code} lines of code")
            
            # Complexity summary
            complexity_level = "low"
            if metrics.cyclomatic_complexity > self.complexity_thresholds['critical']:
                complexity_level = "very high"
            elif metrics.cyclomatic_complexity > self.complexity_thresholds['high']:
                complexity_level = "high"
            elif metrics.cyclomatic_complexity > self.complexity_thresholds['medium']:
                complexity_level = "medium"
            
            summary_parts.append(f"Complexity: {complexity_level} (CC: {metrics.cyclomatic_complexity})")
            
            # Maintainability summary
            maintainability_level = "poor"
            if metrics.maintainability_index >= self.maintainability_thresholds['excellent']:
                maintainability_level = "excellent"
            elif metrics.maintainability_index >= self.maintainability_thresholds['good']:
                maintainability_level = "good"
            elif metrics.maintainability_index >= self.maintainability_thresholds['acceptable']:
                maintainability_level = "acceptable"
            
            summary_parts.append(f"Maintainability: {maintainability_level} ({metrics.maintainability_index:.1f})")
        
        # Issues summary
        if issues:
            issue_counts = Counter(issue.level.value for issue in issues)
            critical_high = issue_counts.get('critical', 0) + issue_counts.get('high', 0)
            if critical_high > 0:
                summary_parts.append(f"{critical_high} critical/high severity issues found")
            else:
                summary_parts.append(f"{len(issues)} issues found")
        
        # Patterns summary
        if patterns:
            pattern_types = Counter(pattern['type'] for pattern in patterns)
            if 'design_pattern' in pattern_types:
                summary_parts.append(f"Contains {pattern_types['design_pattern']} design patterns")
        
        return "; ".join(summary_parts) if summary_parts else f"Basic {language} code file"
    
    def _generate_recommendations(
        self, 
        metrics: CodeMetrics, 
        issues: List[CodeIssue], 
        patterns: List[Dict[str, Any]], 
        language: str
    ) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if metrics:
            # Complexity recommendations
            if metrics.cyclomatic_complexity > self.complexity_thresholds['high']:
                recommendations.append("Break down complex functions into smaller, simpler functions")
            
            if metrics.cognitive_complexity > 15:
                recommendations.append("Reduce cognitive complexity by simplifying conditional logic")
            
            # Maintainability recommendations
            if metrics.maintainability_index < self.maintainability_thresholds['good']:
                recommendations.append("Improve maintainability by reducing complexity and adding documentation")
            
            # Technical debt recommendations
            if metrics.technical_debt_ratio > 0.3:
                recommendations.append("Address TODO/FIXME comments to reduce technical debt")
            
            # Duplication recommendations
            if metrics.code_duplication > 0.2:
                recommendations.append("Refactor duplicated code into reusable functions")
        
        # Issue-based recommendations
        if issues:
            critical_issues = [i for i in issues if i.level == IssueLevel.CRITICAL]
            if critical_issues:
                recommendations.append("Address critical security vulnerabilities immediately")
            
            security_issues = [i for i in issues if i.issue_type == 'security']
            if security_issues:
                recommendations.append("Review and fix security vulnerabilities")
            
            performance_issues = [i for i in issues if i.issue_type == 'performance']
            if performance_issues:
                recommendations.append("Optimize performance bottlenecks")
        
        # Pattern-based recommendations
        design_patterns = [p for p in patterns if p['type'] == 'design_pattern']
        if not design_patterns and metrics and metrics.cyclomatic_complexity > 10:
            recommendations.append("Consider using design patterns to improve code structure")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Code quality is good - consider adding more tests and documentation")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def get_issue_summary(self, issues: List[CodeIssue]) -> Dict[str, Any]:
        """Get summary of issues by type and severity"""
        if not issues:
            return {'total': 0, 'by_level': {}, 'by_type': {}}
        
        by_level = Counter(issue.level.value for issue in issues)
        by_type = Counter(issue.issue_type for issue in issues)
        
        return {
            'total': len(issues),
            'by_level': dict(by_level),
            'by_type': dict(by_type),
            'critical_count': by_level.get('critical', 0),
            'high_count': by_level.get('high', 0),
            'security_count': by_type.get('security', 0),
            'performance_count': by_type.get('performance', 0)
        }
    
    def filter_issues(
        self, 
        issues: List[CodeIssue], 
        min_level: IssueLevel = IssueLevel.LOW,
        issue_types: List[str] = None
    ) -> List[CodeIssue]:
        """Filter issues by severity level and type"""
        level_order = {
            IssueLevel.INFO: 0,
            IssueLevel.LOW: 1,
            IssueLevel.MEDIUM: 2,
            IssueLevel.HIGH: 3,
            IssueLevel.CRITICAL: 4
        }
        
        min_level_value = level_order[min_level]
        
        filtered = []
        for issue in issues:
            if level_order[issue.level] >= min_level_value:
                if issue_types is None or issue.issue_type in issue_types:
                    filtered.append(issue)
        
        return filtered
    
    async def analyze_project_health(self, analysis_results: List[AnalysisResult]) -> Dict[str, Any]:
        """Analyze overall project health from multiple file analyses"""
        if not analysis_results:
            return {}
        
        # Aggregate metrics
        total_loc = sum(r.metrics.lines_of_code for r in analysis_results if r.metrics)
        total_issues = sum(len(r.issues) for r in analysis_results)
        avg_complexity = np.mean([r.metrics.cyclomatic_complexity for r in analysis_results if r.metrics])
        avg_maintainability = np.mean([r.metrics.maintainability_index for r in analysis_results if r.metrics])
        
        # Issue distribution
        all_issues = []
        for result in analysis_results:
            all_issues.extend(result.issues)
        
        issue_summary = self.get_issue_summary(all_issues)
        
        # Language distribution
        languages = Counter(r.language for r in analysis_results)
        
        # Pattern analysis
        all_patterns = []
        for result in analysis_results:
            all_patterns.extend(result.patterns)
        
        pattern_summary = Counter(p['name'] for p in all_patterns)
        
        # Health score calculation
        health_score = self._calculate_health_score(
            avg_complexity, avg_maintainability, issue_summary, total_loc
        )
        
        return {
            'total_files': len(analysis_results),
            'total_lines_of_code': total_loc,
            'total_issues': total_issues,
            'average_complexity': avg_complexity,
            'average_maintainability': avg_maintainability,
            'health_score': health_score,
            'issue_summary': issue_summary,
            'language_distribution': dict(languages),
            'common_patterns': dict(pattern_summary.most_common(10)),
            'recommendations': self._generate_project_recommendations(
                avg_complexity, avg_maintainability, issue_summary
            )
        }
    
    def _calculate_health_score(
        self, 
        avg_complexity: float, 
        avg_maintainability: float, 
        issue_summary: Dict, 
        total_loc: int
    ) -> float:
        """Calculate overall project health score (0-100)"""
        score = 100.0
        
        # Complexity penalty
        if avg_complexity > 20:
            score -= 30
        elif avg_complexity > 10:
            score -= 15
        elif avg_complexity > 5:
            score -= 5
        
        # Maintainability bonus/penalty
        if avg_maintainability < 30:
            score -= 25
        elif avg_maintainability < 50:
            score -= 10
        elif avg_maintainability > 80:
            score += 5
        
        # Issues penalty
        critical_high = issue_summary.get('critical_count', 0) + issue_summary.get('high_count', 0)
        if critical_high > 0:
            score -= critical_high * 10
        
        total_issues = issue_summary.get('total', 0)
        issue_density = total_issues / max(1, total_loc / 100)  # Issues per 100 LOC
        if issue_density > 5:
            score -= 15
        elif issue_density > 2:
            score -= 5
        
        return max(0.0, min(100.0, score))
    
    def _generate_project_recommendations(
        self, 
        avg_complexity: float, 
        avg_maintainability: float, 
        issue_summary: Dict
    ) -> List[str]:
        """Generate project-level recommendations"""
        recommendations = []
        
        if avg_complexity > 15:
            recommendations.append("Focus on reducing code complexity across the project")
        
        if avg_maintainability < 50:
            recommendations.append("Improve overall code maintainability through refactoring")
        
        critical_count = issue_summary.get('critical_count', 0)
        if critical_count > 0:
            recommendations.append(f"Address {critical_count} critical security/quality issues")
        
        security_count = issue_summary.get('security_count', 0)
        if security_count > 0:
            recommendations.append("Conduct security review and fix vulnerabilities")
        
        if not recommendations:
            recommendations.append("Project health is good - focus on maintaining quality")
        
        return recommendations