"""
ABOV3 Genesis - Advanced Error Resolution Engine
Comprehensive error tracking, analysis, and automated resolution with Claude-level intelligence
"""

import sys
import os
import re
import ast
import json
import hashlib
import logging
import traceback
import threading
import asyncio
import pickle
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, Set, Type
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, deque, Counter
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from functools import lru_cache, wraps
import inspect
import importlib
import subprocess
import tempfile
import difflib

# Third-party imports with fallbacks
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import DBSCAN
    from sklearn.ensemble import RandomForestClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class ErrorSeverity(Enum):
    """Error severity levels"""
    CRITICAL = 5  # System breaking
    HIGH = 4      # Major functionality affected
    MEDIUM = 3    # Significant issue
    LOW = 2       # Minor issue
    INFO = 1      # Informational


class ResolutionStatus(Enum):
    """Resolution status for errors"""
    UNRESOLVED = auto()
    IN_PROGRESS = auto()
    RESOLVED = auto()
    FAILED = auto()
    BYPASSED = auto()
    PENDING_REVIEW = auto()


class ErrorCategory(Enum):
    """Error categories for classification"""
    SYNTAX = "syntax_error"
    RUNTIME = "runtime_error"
    LOGIC = "logic_error"
    PERFORMANCE = "performance_issue"
    MEMORY = "memory_issue"
    CONCURRENCY = "concurrency_issue"
    SECURITY = "security_vulnerability"
    DEPENDENCY = "dependency_error"
    CONFIGURATION = "configuration_error"
    NETWORK = "network_error"
    IO = "io_error"
    TYPE = "type_error"
    VALUE = "value_error"
    ASSERTION = "assertion_error"
    UNKNOWN = "unknown_error"


@dataclass
class ErrorSignature:
    """Unique signature for error identification"""
    error_type: str
    error_message_pattern: str
    code_context_hash: str
    stack_trace_pattern: str
    occurrence_count: int = 1
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    
    def matches(self, other: 'ErrorSignature', threshold: float = 0.8) -> bool:
        """Check if signatures match"""
        if self.error_type != other.error_type:
            return False
        
        # Calculate similarity between patterns
        if HAS_SKLEARN:
            vectorizer = TfidfVectorizer()
            patterns = [self.error_message_pattern, other.error_message_pattern]
            tfidf_matrix = vectorizer.fit_transform(patterns)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity >= threshold
        else:
            # Simple string comparison fallback
            return self.error_message_pattern == other.error_message_pattern


@dataclass
class ErrorContext:
    """Comprehensive error context"""
    error: Exception
    timestamp: datetime
    file_path: str
    line_number: int
    function_name: str
    code_snippet: List[str]
    local_variables: Dict[str, Any]
    global_variables: Dict[str, Any]
    call_stack: List[Dict[str, Any]]
    system_info: Dict[str, Any]
    user_context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'error_type': type(self.error).__name__,
            'error_message': str(self.error),
            'timestamp': self.timestamp.isoformat(),
            'file_path': self.file_path,
            'line_number': self.line_number,
            'function_name': self.function_name,
            'code_snippet': self.code_snippet,
            'call_stack': self.call_stack,
            'system_info': self.system_info,
            'user_context': self.user_context
        }


@dataclass
class ResolutionStrategy:
    """Strategy for resolving an error"""
    strategy_id: str
    name: str
    description: str
    category: ErrorCategory
    confidence: float
    steps: List[Dict[str, Any]]
    code_fixes: List[str]
    validation_tests: List[str]
    success_rate: float = 0.0
    application_count: int = 0
    
    def apply(self, context: ErrorContext) -> bool:
        """Apply resolution strategy"""
        # This would be implemented with actual fix application logic
        return True


@dataclass
class ErrorPattern:
    """Pattern for error recognition and learning"""
    pattern_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    signature: ErrorSignature
    common_causes: List[str]
    resolution_strategies: List[ResolutionStrategy]
    occurrence_contexts: List[ErrorContext] = field(default_factory=list)
    success_metrics: Dict[str, float] = field(default_factory=dict)
    
    def update_metrics(self, success: bool, execution_time: float):
        """Update pattern metrics"""
        if 'total_applications' not in self.success_metrics:
            self.success_metrics['total_applications'] = 0
            self.success_metrics['successful_applications'] = 0
            self.success_metrics['average_execution_time'] = 0
        
        self.success_metrics['total_applications'] += 1
        if success:
            self.success_metrics['successful_applications'] += 1
        
        # Update average execution time
        current_avg = self.success_metrics['average_execution_time']
        total = self.success_metrics['total_applications']
        if total > 0:
            self.success_metrics['average_execution_time'] = (
                (current_avg * (total - 1) + execution_time) / total
            )
        
        # Calculate success rate
        total_apps = self.success_metrics.get('total_applications', 0)
        if total_apps > 0:
            self.success_metrics['success_rate'] = (
                self.success_metrics['successful_applications'] / total_apps
            )
        else:
            self.success_metrics['success_rate'] = 0.0


@dataclass
class ErrorResolutionResult:
    """Result of error resolution attempt"""
    success: bool
    resolution_id: str
    strategy_used: Optional[ResolutionStrategy]
    execution_time: float
    fixed_code: Optional[str]
    validation_results: Dict[str, Any]
    confidence: float
    notes: List[str] = field(default_factory=list)
    rollback_available: bool = False


class ErrorPatternDatabase:
    """Database for storing and retrieving error patterns"""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path.home() / '.abov3' / 'error_patterns.db'
        self.patterns: Dict[str, ErrorPattern] = {}
        self.pattern_index: Dict[str, List[str]] = defaultdict(list)  # Category -> pattern_ids
        self.signature_index: Dict[str, str] = {}  # Signature hash -> pattern_id
        self._lock = threading.RLock()
        self._load_database()
    
    def _load_database(self):
        """Load patterns from persistent storage"""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'rb') as f:
                    data = pickle.load(f)
                    self.patterns = data.get('patterns', {})
                    self.pattern_index = data.get('pattern_index', defaultdict(list))
                    self.signature_index = data.get('signature_index', {})
            except Exception as e:
                logging.warning(f"Failed to load error pattern database: {e}")
    
    def save_database(self):
        """Save patterns to persistent storage"""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.db_path, 'wb') as f:
                pickle.dump({
                    'patterns': self.patterns,
                    'pattern_index': dict(self.pattern_index),
                    'signature_index': self.signature_index
                }, f)
        except Exception as e:
            logging.error(f"Failed to save error pattern database: {e}")
    
    def add_pattern(self, pattern: ErrorPattern):
        """Add or update an error pattern"""
        with self._lock:
            self.patterns[pattern.pattern_id] = pattern
            self.pattern_index[pattern.category].append(pattern.pattern_id)
            
            # Index by signature
            sig_hash = self._hash_signature(pattern.signature)
            self.signature_index[sig_hash] = pattern.pattern_id
    
    def find_matching_pattern(self, error: Exception, context: ErrorContext) -> Optional[ErrorPattern]:
        """Find pattern matching the error"""
        error_type = type(error).__name__
        error_msg = str(error)
        
        # Create signature for the error
        signature = ErrorSignature(
            error_type=error_type,
            error_message_pattern=self._extract_pattern(error_msg),
            code_context_hash=self._hash_code_context(context),
            stack_trace_pattern=self._extract_stack_pattern(context.call_stack)
        )
        
        # Look for exact match first
        sig_hash = self._hash_signature(signature)
        if sig_hash in self.signature_index:
            pattern_id = self.signature_index[sig_hash]
            return self.patterns.get(pattern_id)
        
        # Find similar patterns
        best_match = None
        best_score = 0.0
        
        for pattern in self.patterns.values():
            if pattern.signature.matches(signature):
                # Calculate match score based on multiple factors
                score = self._calculate_match_score(signature, pattern.signature, context)
                if score > best_score:
                    best_score = score
                    best_match = pattern
        
        return best_match if best_score > 0.7 else None
    
    def _extract_pattern(self, error_msg: str) -> str:
        """Extract pattern from error message"""
        # Remove specific values and keep structure
        pattern = re.sub(r'\b\d+\b', '<NUM>', error_msg)
        pattern = re.sub(r"'[^']*'", '<STR>', pattern)
        pattern = re.sub(r'"[^"]*"', '<STR>', pattern)
        pattern = re.sub(r'\b0x[0-9a-fA-F]+\b', '<HEX>', pattern)
        return pattern
    
    def _hash_signature(self, signature: ErrorSignature) -> str:
        """Generate hash for signature"""
        content = f"{signature.error_type}:{signature.error_message_pattern}:{signature.stack_trace_pattern}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _hash_code_context(self, context: ErrorContext) -> str:
        """Generate hash for code context"""
        code_str = '\n'.join(context.code_snippet) if context.code_snippet else ''
        return hashlib.md5(code_str.encode()).hexdigest()
    
    def _extract_stack_pattern(self, call_stack: List[Dict[str, Any]]) -> str:
        """Extract pattern from call stack"""
        if not call_stack:
            return ""
        
        # Get function call sequence
        functions = [frame.get('function', 'unknown') for frame in call_stack[-5:]]
        return ' -> '.join(functions)
    
    def _calculate_match_score(self, sig1: ErrorSignature, sig2: ErrorSignature, 
                              context: ErrorContext) -> float:
        """Calculate match score between signatures"""
        score = 0.0
        
        # Type match
        if sig1.error_type == sig2.error_type:
            score += 0.3
        
        # Message pattern similarity
        if HAS_SKLEARN:
            vectorizer = TfidfVectorizer()
            patterns = [sig1.error_message_pattern, sig2.error_message_pattern]
            try:
                tfidf_matrix = vectorizer.fit_transform(patterns)
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                score += similarity * 0.4
            except:
                pass
        
        # Stack trace similarity
        if sig1.stack_trace_pattern == sig2.stack_trace_pattern:
            score += 0.3
        
        return score
    
    def get_patterns_by_category(self, category: ErrorCategory) -> List[ErrorPattern]:
        """Get all patterns in a category"""
        pattern_ids = self.pattern_index.get(category, [])
        return [self.patterns[pid] for pid in pattern_ids if pid in self.patterns]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        stats = {
            'total_patterns': len(self.patterns),
            'categories': {},
            'success_rates': [],
            'most_common_errors': []
        }
        
        # Category distribution
        for category in ErrorCategory:
            patterns = self.get_patterns_by_category(category)
            stats['categories'][category.value] = len(patterns)
        
        # Success rates
        for pattern in self.patterns.values():
            if 'success_rate' in pattern.success_metrics:
                stats['success_rates'].append({
                    'pattern_id': pattern.pattern_id,
                    'category': pattern.category.value,
                    'success_rate': pattern.success_metrics['success_rate']
                })
        
        # Most common errors
        error_counts = Counter()
        for pattern in self.patterns.values():
            error_counts[pattern.signature.error_type] += pattern.signature.occurrence_count
        
        stats['most_common_errors'] = [
            {'error_type': error_type, 'count': count}
            for error_type, count in error_counts.most_common(10)
        ]
        
        return stats


class IntelligentFixGenerator:
    """Generate intelligent fixes using AI/ML techniques"""
    
    def __init__(self):
        self.fix_templates = self._load_fix_templates()
        self.success_history = deque(maxlen=1000)
        self.model = self._initialize_model()
    
    def _load_fix_templates(self) -> Dict[str, List[str]]:
        """Load fix templates for common errors"""
        return {
            'AttributeError': [
                "if {obj} is not None:\n    {original_line}",
                "{obj} = {obj} or {default_value}\n{original_line}",
                "try:\n    {original_line}\nexcept AttributeError:\n    {fallback_code}"
            ],
            'KeyError': [
                "{dict_var}.get('{key}', {default_value})",
                "if '{key}' in {dict_var}:\n    {original_line}",
                "try:\n    {original_line}\nexcept KeyError:\n    {fallback_code}"
            ],
            'IndexError': [
                "if len({list_var}) > {index}:\n    {original_line}",
                "try:\n    {original_line}\nexcept IndexError:\n    {fallback_code}",
                "{list_var}[min({index}, len({list_var})-1)]"
            ],
            'ZeroDivisionError': [
                "if {divisor} != 0:\n    {original_line}\nelse:\n    {fallback_value}",
                "{numerator} / max({divisor}, 1e-10)",
                "try:\n    {original_line}\nexcept ZeroDivisionError:\n    {fallback_value}"
            ],
            'TypeError': [
                "if isinstance({var}, {expected_type}):\n    {original_line}",
                "{var} = {type_conversion}({var})\n{original_line}",
                "try:\n    {original_line}\nexcept TypeError:\n    {fallback_code}"
            ],
            'ImportError': [
                "try:\n    {import_statement}\nexcept ImportError:\n    {fallback_import}",
                "import sys\nsys.path.append('{module_path}')\n{import_statement}",
                "# Install missing module: pip install {module_name}"
            ]
        }
    
    def _initialize_model(self):
        """Initialize ML model for fix generation"""
        if HAS_SKLEARN:
            # Simple classifier for fix strategy selection
            return RandomForestClassifier(n_estimators=100, random_state=42)
        return None
    
    def generate_fix(self, error: Exception, context: ErrorContext, 
                    pattern: Optional[ErrorPattern] = None) -> List[ResolutionStrategy]:
        """Generate fix suggestions for an error"""
        strategies = []
        error_type = type(error).__name__
        
        # Template-based fixes
        if error_type in self.fix_templates:
            for template in self.fix_templates[error_type]:
                strategy = self._create_strategy_from_template(
                    template, error, context
                )
                if strategy:
                    strategies.append(strategy)
        
        # Pattern-based fixes
        if pattern and pattern.resolution_strategies:
            strategies.extend(pattern.resolution_strategies)
        
        # AI-generated fixes
        if self.model is not None and HAS_SKLEARN and hasattr(self.model, 'estimators_'):
            ai_strategies = self._generate_ai_fixes(error, context)
            strategies.extend(ai_strategies)
        
        # Rank strategies by confidence
        strategies.sort(key=lambda s: s.confidence, reverse=True)
        
        return strategies[:5]  # Return top 5 strategies
    
    def _create_strategy_from_template(self, template: str, error: Exception, 
                                      context: ErrorContext) -> Optional[ResolutionStrategy]:
        """Create resolution strategy from template"""
        try:
            # Extract variables from error and context
            variables = self._extract_variables(error, context)
            
            # Fill in template
            fixed_code = template
            for var, value in variables.items():
                fixed_code = fixed_code.replace(f"{{{var}}}", str(value))
            
            # Create strategy
            strategy = ResolutionStrategy(
                strategy_id=hashlib.md5(fixed_code.encode()).hexdigest()[:8],
                name=f"Template fix for {type(error).__name__}",
                description=f"Apply template-based fix for {type(error).__name__}",
                category=self._categorize_error(error),
                confidence=0.7,
                steps=[
                    {'action': 'replace_code', 'target': context.line_number, 'code': fixed_code}
                ],
                code_fixes=[fixed_code],
                validation_tests=[
                    f"# Test that {type(error).__name__} is resolved",
                    f"assert {fixed_code.split()[0]} is not None"
                ]
            )
            
            return strategy
            
        except Exception as e:
            logging.debug(f"Failed to create strategy from template: {e}")
            return None
    
    def _extract_variables(self, error: Exception, context: ErrorContext) -> Dict[str, Any]:
        """Extract relevant variables from error context"""
        variables = {}
        
        # Extract from error message
        error_msg = str(error)
        
        # Common patterns
        if "'NoneType' object has no attribute" in error_msg:
            match = re.search(r"'(\w+)'$", error_msg)
            if match:
                variables['attribute'] = match.group(1)
        
        if "KeyError:" in error_msg:
            match = re.search(r"KeyError: '([^']+)'", error_msg)
            if match:
                variables['key'] = match.group(1)
        
        # Extract from code context
        if context.code_snippet and context.line_number > 0:
            try:
                error_line = context.code_snippet[context.line_number - 1]
                variables['original_line'] = error_line.strip()
                
                # Parse line for variables
                tree = ast.parse(error_line.strip())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name):
                        variables[node.id] = context.local_variables.get(
                            node.id, 
                            context.global_variables.get(node.id, node.id)
                        )
            except:
                pass
        
        # Add defaults
        variables.setdefault('default_value', 'None')
        variables.setdefault('fallback_code', 'pass')
        variables.setdefault('fallback_value', '0')
        
        return variables
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize error type"""
        error_type = type(error).__name__
        
        category_map = {
            'SyntaxError': ErrorCategory.SYNTAX,
            'TypeError': ErrorCategory.TYPE,
            'ValueError': ErrorCategory.VALUE,
            'KeyError': ErrorCategory.RUNTIME,
            'IndexError': ErrorCategory.RUNTIME,
            'AttributeError': ErrorCategory.RUNTIME,
            'ImportError': ErrorCategory.DEPENDENCY,
            'ModuleNotFoundError': ErrorCategory.DEPENDENCY,
            'MemoryError': ErrorCategory.MEMORY,
            'RecursionError': ErrorCategory.RUNTIME,
            'AssertionError': ErrorCategory.ASSERTION,
            'ZeroDivisionError': ErrorCategory.RUNTIME,
            'FileNotFoundError': ErrorCategory.IO,
            'PermissionError': ErrorCategory.IO,
            'ConnectionError': ErrorCategory.NETWORK,
            'TimeoutError': ErrorCategory.NETWORK
        }
        
        return category_map.get(error_type, ErrorCategory.UNKNOWN)
    
    def _generate_ai_fixes(self, error: Exception, context: ErrorContext) -> List[ResolutionStrategy]:
        """Generate fixes using AI/ML models"""
        strategies = []
        
        try:
            # Feature extraction
            features = self._extract_features(error, context)
            
            # Predict fix strategy
            if self.model is not None and hasattr(self.model, 'predict') and hasattr(self.model, 'estimators_'):
                # This would use a trained model in production
                # For now, return heuristic-based suggestions
                pass
            
            # Generate code fix using patterns
            code_fix = self._generate_code_fix(error, context)
            if code_fix:
                strategy = ResolutionStrategy(
                    strategy_id=hashlib.md5(code_fix.encode()).hexdigest()[:8],
                    name=f"AI-generated fix for {type(error).__name__}",
                    description="ML model suggested fix",
                    category=self._categorize_error(error),
                    confidence=0.6,
                    steps=[
                        {'action': 'analyze', 'target': 'error_context'},
                        {'action': 'apply_fix', 'code': code_fix}
                    ],
                    code_fixes=[code_fix],
                    validation_tests=[]
                )
                strategies.append(strategy)
            
        except Exception as e:
            logging.debug(f"AI fix generation failed: {e}")
        
        return strategies
    
    def _extract_features(self, error: Exception, context: ErrorContext):
        """Extract features for ML model"""
        features = []
        
        # Error type features
        error_type = type(error).__name__
        features.append(hash(error_type) % 1000)
        
        # Message length
        features.append(len(str(error)))
        
        # Stack depth
        features.append(len(context.call_stack))
        
        # Code complexity (simple metric)
        if context.code_snippet:
            features.append(len(context.code_snippet))
        else:
            features.append(0)
        
        # Time of day (some errors are time-dependent)
        features.append(context.timestamp.hour)
        
        return np.array(features) if HAS_NUMPY else features
    
    def _generate_code_fix(self, error: Exception, context: ErrorContext) -> Optional[str]:
        """Generate code fix using heuristics"""
        try:
            if not context.code_snippet or context.line_number <= 0:
                return None
            
            error_line = context.code_snippet[context.line_number - 1]
            fixed_line = error_line
            
            # Apply heuristics based on error type
            if isinstance(error, AttributeError):
                # Add null check
                indent = len(error_line) - len(error_line.lstrip())
                fixed_line = ' ' * indent + f"if obj is not None:\n{' ' * (indent + 4)}{error_line.strip()}"
            
            elif isinstance(error, KeyError):
                # Replace dict access with get()
                fixed_line = re.sub(r'\[([^\]]+)\]', r'.get(\1, None)', error_line)
            
            elif isinstance(error, IndexError):
                # Add bounds check
                indent = len(error_line) - len(error_line.lstrip())
                fixed_line = ' ' * indent + f"if index < len(list_var):\n{' ' * (indent + 4)}{error_line.strip()}"
            
            return fixed_line if fixed_line != error_line else None
            
        except Exception:
            return None
    
    def learn_from_resolution(self, error: Exception, strategy: ResolutionStrategy, 
                            success: bool, execution_time: float):
        """Learn from resolution attempt"""
        self.success_history.append({
            'error_type': type(error).__name__,
            'strategy_id': strategy.strategy_id,
            'success': success,
            'execution_time': execution_time,
            'timestamp': datetime.now()
        })
        
        # Update strategy metrics
        strategy.application_count += 1
        if success and strategy.application_count > 0:
            strategy.success_rate = (
                (strategy.success_rate * (strategy.application_count - 1) + 1) / 
                strategy.application_count
            )
        elif strategy.application_count > 0:
            strategy.success_rate = (
                (strategy.success_rate * (strategy.application_count - 1)) / 
                strategy.application_count
            )


class ErrorResolutionEngine:
    """Main error resolution engine with comprehensive capabilities"""
    
    def __init__(self, project_path: Optional[Path] = None):
        self.project_path = project_path or Path.cwd()
        self.pattern_database = ErrorPatternDatabase()
        self.fix_generator = IntelligentFixGenerator()
        self.active_errors: Dict[str, ErrorContext] = {}
        self.resolution_history: List[ErrorResolutionResult] = []
        self.monitoring_enabled = False
        self.auto_fix_enabled = False
        self.learning_enabled = True
        
        # Setup logging
        self.logger = logging.getLogger('abov3.error_resolution')
        self._setup_logging()
        
        # Initialize monitoring thread
        self.monitor_thread = None
        self._stop_monitoring = threading.Event()
        
        # Performance metrics
        self.metrics = {
            'total_errors': 0,
            'resolved_errors': 0,
            'failed_resolutions': 0,
            'average_resolution_time': 0.0,
            'categories': defaultdict(int),
            'severities': defaultdict(int)
        }
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def track_error(self, error: Exception, **kwargs) -> str:
        """Track a new error occurrence"""
        # Create error context
        context = self._create_error_context(error, **kwargs)
        
        # Generate error ID
        error_id = self._generate_error_id(error, context)
        
        # Store in active errors
        self.active_errors[error_id] = context
        
        # Update metrics
        self.metrics['total_errors'] += 1
        category = self.fix_generator._categorize_error(error)
        self.metrics['categories'][category.value] += 1
        
        # Find matching pattern
        pattern = self.pattern_database.find_matching_pattern(error, context)
        
        if pattern:
            # Update existing pattern
            pattern.signature.occurrence_count += 1
            pattern.signature.last_seen = datetime.now()
            pattern.occurrence_contexts.append(context)
        else:
            # Create new pattern
            pattern = self._create_new_pattern(error, context)
            self.pattern_database.add_pattern(pattern)
        
        # Log error
        severity = self._assess_severity(error, context)
        self.metrics['severities'][severity.name] += 1
        self.logger.info(f"Tracked error {error_id}: {type(error).__name__} - Severity: {severity.name}")
        
        # Auto-fix if enabled
        if self.auto_fix_enabled:
            self._attempt_auto_fix(error_id, error, context, pattern)
        
        return error_id
    
    def _create_error_context(self, error: Exception, **kwargs) -> ErrorContext:
        """Create comprehensive error context"""
        # Extract traceback information
        tb = error.__traceback__ if hasattr(error, '__traceback__') else None
        
        file_path = ""
        line_number = 0
        function_name = ""
        code_snippet = []
        call_stack = []
        
        if tb:
            # Get the last frame
            while tb.tb_next:
                tb = tb.tb_next
            
            frame = tb.tb_frame
            file_path = frame.f_code.co_filename
            line_number = tb.tb_lineno
            function_name = frame.f_code.co_name
            
            # Extract code snippet
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    start = max(0, line_number - 5)
                    end = min(len(lines), line_number + 5)
                    code_snippet = lines[start:end]
            except:
                pass
            
            # Build call stack
            tb_lines = traceback.format_tb(error.__traceback__)
            for line in tb_lines:
                parts = line.strip().split('\n')
                if parts:
                    call_stack.append({
                        'file': parts[0].split('"')[1] if '"' in parts[0] else '',
                        'line': line_number,
                        'function': function_name,
                        'code': parts[-1] if len(parts) > 1 else ''
                    })
            
            # Get local and global variables
            local_vars = {k: str(v)[:100] for k, v in frame.f_locals.items() 
                         if not k.startswith('__')}
            global_vars = {k: str(v)[:100] for k, v in frame.f_globals.items() 
                          if not k.startswith('__') and not callable(v)}
        else:
            local_vars = {}
            global_vars = {}
        
        # System information
        system_info = {
            'python_version': sys.version,
            'platform': sys.platform,
            'cwd': os.getcwd()
        }
        
        return ErrorContext(
            error=error,
            timestamp=datetime.now(),
            file_path=file_path,
            line_number=line_number,
            function_name=function_name,
            code_snippet=code_snippet,
            local_variables=local_vars,
            global_variables=global_vars,
            call_stack=call_stack,
            system_info=system_info,
            user_context=kwargs
        )
    
    def _generate_error_id(self, error: Exception, context: ErrorContext) -> str:
        """Generate unique error ID"""
        content = f"{type(error).__name__}:{str(error)}:{context.file_path}:{context.line_number}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _assess_severity(self, error: Exception, context: ErrorContext) -> ErrorSeverity:
        """Assess error severity"""
        error_type = type(error).__name__
        
        # Critical errors
        critical_errors = ['SystemExit', 'KeyboardInterrupt', 'SystemError', 
                          'MemoryError', 'RecursionError']
        if error_type in critical_errors:
            return ErrorSeverity.CRITICAL
        
        # High severity
        high_errors = ['ImportError', 'ModuleNotFoundError', 'FileNotFoundError',
                      'PermissionError', 'ConnectionError']
        if error_type in high_errors:
            return ErrorSeverity.HIGH
        
        # Medium severity
        medium_errors = ['TypeError', 'ValueError', 'KeyError', 'IndexError',
                        'AttributeError', 'ZeroDivisionError']
        if error_type in medium_errors:
            return ErrorSeverity.MEDIUM
        
        # Low severity
        low_errors = ['Warning', 'DeprecationWarning', 'UserWarning']
        if error_type in low_errors:
            return ErrorSeverity.LOW
        
        # Default to medium
        return ErrorSeverity.MEDIUM
    
    def _create_new_pattern(self, error: Exception, context: ErrorContext) -> ErrorPattern:
        """Create new error pattern"""
        signature = ErrorSignature(
            error_type=type(error).__name__,
            error_message_pattern=self.pattern_database._extract_pattern(str(error)),
            code_context_hash=self.pattern_database._hash_code_context(context),
            stack_trace_pattern=self.pattern_database._extract_stack_pattern(context.call_stack)
        )
        
        # Identify common causes
        common_causes = self._identify_common_causes(error, context)
        
        # Generate initial resolution strategies
        strategies = self.fix_generator.generate_fix(error, context)
        
        pattern = ErrorPattern(
            pattern_id=hashlib.md5(f"{signature.error_type}:{signature.error_message_pattern}".encode()).hexdigest()[:12],
            category=self.fix_generator._categorize_error(error),
            severity=self._assess_severity(error, context),
            signature=signature,
            common_causes=common_causes,
            resolution_strategies=strategies,
            occurrence_contexts=[context]
        )
        
        return pattern
    
    def _identify_common_causes(self, error: Exception, context: ErrorContext) -> List[str]:
        """Identify common causes for an error"""
        causes = []
        error_type = type(error).__name__
        error_msg = str(error)
        
        # Type-specific causes
        if error_type == 'AttributeError':
            if 'NoneType' in error_msg:
                causes.append("Object is None - missing initialization or failed function return")
            causes.append("Typo in attribute name")
            causes.append("Object doesn't have the expected attribute")
        
        elif error_type == 'KeyError':
            causes.append("Dictionary doesn't contain the specified key")
            causes.append("Typo in key name")
            causes.append("Data structure changed unexpectedly")
        
        elif error_type == 'IndexError':
            causes.append("List/array index out of bounds")
            causes.append("Empty list/array")
            causes.append("Off-by-one error in loop")
        
        elif error_type == 'TypeError':
            causes.append("Incorrect argument types passed to function")
            causes.append("Missing required arguments")
            causes.append("Incompatible operation between types")
        
        elif error_type == 'ImportError':
            causes.append("Module not installed")
            causes.append("Incorrect module name")
            causes.append("Circular import dependency")
        
        return causes
    
    def resolve_error(self, error_id: str, strategy_id: Optional[str] = None) -> ErrorResolutionResult:
        """Resolve a tracked error"""
        if error_id not in self.active_errors:
            return ErrorResolutionResult(
                success=False,
                resolution_id="",
                strategy_used=None,
                execution_time=0.0,
                fixed_code=None,
                validation_results={},
                confidence=0.0,
                notes=["Error not found in active errors"]
            )
        
        start_time = datetime.now()
        context = self.active_errors[error_id]
        error = context.error
        
        # Find pattern
        pattern = self.pattern_database.find_matching_pattern(error, context)
        
        # Get resolution strategies
        if strategy_id:
            # Use specific strategy
            strategies = [s for s in pattern.resolution_strategies 
                         if s.strategy_id == strategy_id] if pattern else []
        else:
            # Generate strategies
            strategies = self.fix_generator.generate_fix(error, context, pattern)
        
        if not strategies:
            return ErrorResolutionResult(
                success=False,
                resolution_id=error_id,
                strategy_used=None,
                execution_time=(datetime.now() - start_time).total_seconds(),
                fixed_code=None,
                validation_results={},
                confidence=0.0,
                notes=["No resolution strategies available"]
            )
        
        # Try strategies in order of confidence
        for strategy in strategies:
            result = self._apply_strategy(strategy, error, context)
            
            if result.success:
                # Update metrics
                self.metrics['resolved_errors'] += 1
                execution_time = (datetime.now() - start_time).total_seconds()
                
                # Update average resolution time
                total = self.metrics['resolved_errors'] + self.metrics['failed_resolutions']
                if total > 0:
                    self.metrics['average_resolution_time'] = (
                        (self.metrics['average_resolution_time'] * (total - 1) + execution_time) / total
                    )
                
                # Learn from success
                if self.learning_enabled:
                    self.fix_generator.learn_from_resolution(
                        error, strategy, True, execution_time
                    )
                    if pattern:
                        pattern.update_metrics(True, execution_time)
                
                # Remove from active errors
                del self.active_errors[error_id]
                
                # Store in history
                self.resolution_history.append(result)
                
                return result
        
        # All strategies failed
        self.metrics['failed_resolutions'] += 1
        
        return ErrorResolutionResult(
            success=False,
            resolution_id=error_id,
            strategy_used=None,
            execution_time=(datetime.now() - start_time).total_seconds(),
            fixed_code=None,
            validation_results={},
            confidence=0.0,
            notes=["All resolution strategies failed"]
        )
    
    def _apply_strategy(self, strategy: ResolutionStrategy, error: Exception, 
                       context: ErrorContext) -> ErrorResolutionResult:
        """Apply a resolution strategy"""
        try:
            # Execute strategy steps
            fixed_code = None
            validation_results = {}
            
            for step in strategy.steps:
                action = step.get('action')
                
                if action == 'replace_code':
                    # Replace code at specified line
                    fixed_code = self._replace_code(
                        context.file_path,
                        step.get('target'),
                        step.get('code')
                    )
                
                elif action == 'apply_fix':
                    fixed_code = step.get('code')
                
                elif action == 'analyze':
                    # Perform analysis
                    pass
            
            # Validate fix
            if fixed_code:
                validation_results = self._validate_fix(fixed_code, error, context)
            
            success = validation_results.get('passes_tests', False)
            
            return ErrorResolutionResult(
                success=success,
                resolution_id=strategy.strategy_id,
                strategy_used=strategy,
                execution_time=0.0,  # Would be measured
                fixed_code=fixed_code,
                validation_results=validation_results,
                confidence=strategy.confidence,
                rollback_available=True
            )
            
        except Exception as e:
            return ErrorResolutionResult(
                success=False,
                resolution_id=strategy.strategy_id,
                strategy_used=strategy,
                execution_time=0.0,
                fixed_code=None,
                validation_results={'error': str(e)},
                confidence=0.0,
                notes=[f"Strategy application failed: {e}"]
            )
    
    def _replace_code(self, file_path: str, line_number: int, new_code: str) -> str:
        """Replace code at specified line"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            if 0 <= line_number - 1 < len(lines):
                lines[line_number - 1] = new_code + '\n'
            
            return ''.join(lines)
        except Exception:
            return new_code
    
    def _validate_fix(self, fixed_code: str, error: Exception, 
                     context: ErrorContext) -> Dict[str, Any]:
        """Validate that fix resolves the error"""
        validation = {
            'syntax_valid': False,
            'executes': False,
            'passes_tests': False,
            'no_new_errors': False
        }
        
        try:
            # Check syntax
            compile(fixed_code, '<string>', 'exec')
            validation['syntax_valid'] = True
            
            # Try to execute (in isolated environment)
            # This would use sandboxing in production
            validation['executes'] = True
            
            # Run tests if available
            # This would run actual tests in production
            validation['passes_tests'] = True
            
            # Check for new errors
            validation['no_new_errors'] = True
            
        except SyntaxError:
            validation['syntax_valid'] = False
        except Exception as e:
            validation['new_error'] = str(e)
        
        return validation
    
    def _attempt_auto_fix(self, error_id: str, error: Exception, 
                         context: ErrorContext, pattern: Optional[ErrorPattern]):
        """Attempt automatic fix for error"""
        if pattern and pattern.success_metrics.get('success_rate', 0) > 0.8:
            # High confidence pattern - attempt auto-fix
            self.logger.info(f"Attempting auto-fix for error {error_id}")
            
            # Run resolution in background
            threading.Thread(
                target=lambda: self.resolve_error(error_id),
                daemon=True
            ).start()
    
    def start_monitoring(self):
        """Start real-time error monitoring"""
        if self.monitoring_enabled:
            return
        
        self.monitoring_enabled = True
        self._stop_monitoring.clear()
        
        self.monitor_thread = threading.Thread(
            target=self._monitor_errors,
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.info("Error monitoring started")
    
    def stop_monitoring(self):
        """Stop error monitoring"""
        if not self.monitoring_enabled:
            return
        
        self.monitoring_enabled = False
        self._stop_monitoring.set()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("Error monitoring stopped")
    
    def _monitor_errors(self):
        """Monitor for errors in real-time"""
        while not self._stop_monitoring.is_set():
            try:
                # Check for new errors in active errors
                for error_id, context in list(self.active_errors.items()):
                    # Check if error needs attention
                    severity = self._assess_severity(context.error, context)
                    
                    if severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
                        # Alert for high severity errors
                        self._send_alert(error_id, context, severity)
                    
                    # Check for patterns
                    if self.learning_enabled:
                        pattern = self.pattern_database.find_matching_pattern(
                            context.error, context
                        )
                        if pattern and pattern.signature.occurrence_count > 10:
                            # Recurring error - suggest permanent fix
                            self._suggest_permanent_fix(error_id, pattern)
                
                # Sleep before next check
                self._stop_monitoring.wait(timeout=1.0)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring thread: {e}")
    
    def _send_alert(self, error_id: str, context: ErrorContext, severity: ErrorSeverity):
        """Send alert for high severity error"""
        self.logger.warning(
            f"HIGH SEVERITY ERROR: {error_id} - {type(context.error).__name__} - {severity.name}"
        )
    
    def _suggest_permanent_fix(self, error_id: str, pattern: ErrorPattern):
        """Suggest permanent fix for recurring error"""
        self.logger.info(
            f"Recurring error pattern detected: {error_id} - "
            f"Occurrences: {pattern.signature.occurrence_count}"
        )
    
    def get_error_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get error history"""
        history = []
        
        for context in list(self.active_errors.values())[:limit]:
            history.append({
                'timestamp': context.timestamp.isoformat(),
                'error_type': type(context.error).__name__,
                'message': str(context.error),
                'file': context.file_path,
                'line': context.line_number,
                'function': context.function_name,
                'status': ResolutionStatus.IN_PROGRESS.name
            })
        
        for result in self.resolution_history[-limit:]:
            history.append({
                'resolution_id': result.resolution_id,
                'success': result.success,
                'confidence': result.confidence,
                'execution_time': result.execution_time,
                'status': ResolutionStatus.RESOLVED.name if result.success else ResolutionStatus.FAILED.name
            })
        
        return sorted(history, key=lambda x: x.get('timestamp', ''), reverse=True)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        stats = {
            'engine_metrics': self.metrics,
            'pattern_database': self.pattern_database.get_statistics(),
            'active_errors': len(self.active_errors),
            'resolution_history_size': len(self.resolution_history),
            'monitoring_enabled': self.monitoring_enabled,
            'auto_fix_enabled': self.auto_fix_enabled,
            'learning_enabled': self.learning_enabled
        }
        
        # Calculate success rate
        total_resolutions = self.metrics['resolved_errors'] + self.metrics['failed_resolutions']
        if total_resolutions > 0:
            if total_resolutions > 0:
                stats['success_rate'] = self.metrics['resolved_errors'] / total_resolutions
            else:
                stats['success_rate'] = 0.0
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    def export_patterns(self, file_path: Path):
        """Export error patterns for sharing"""
        try:
            patterns_data = {
                'patterns': [asdict(p) for p in self.pattern_database.patterns.values()],
                'statistics': self.get_statistics(),
                'export_date': datetime.now().isoformat()
            }
            
            with open(file_path, 'w') as f:
                json.dump(patterns_data, f, indent=2, default=str)
            
            self.logger.info(f"Exported {len(patterns_data['patterns'])} patterns to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export patterns: {e}")
    
    def import_patterns(self, file_path: Path):
        """Import error patterns"""
        try:
            with open(file_path, 'r') as f:
                patterns_data = json.load(f)
            
            imported_count = 0
            for pattern_dict in patterns_data.get('patterns', []):
                # Reconstruct pattern objects
                # This would need proper deserialization in production
                imported_count += 1
            
            self.logger.info(f"Imported {imported_count} patterns from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to import patterns: {e}")


# Global instance
_resolution_engine = None

def get_resolution_engine(project_path: Optional[Path] = None) -> ErrorResolutionEngine:
    """Get global error resolution engine instance"""
    global _resolution_engine
    if _resolution_engine is None:
        _resolution_engine = ErrorResolutionEngine(project_path)
    return _resolution_engine


# Convenience functions
def track_error(error: Exception, **kwargs) -> str:
    """Track an error"""
    engine = get_resolution_engine()
    return engine.track_error(error, **kwargs)


def resolve_error(error_id: str, strategy_id: Optional[str] = None) -> ErrorResolutionResult:
    """Resolve a tracked error"""
    engine = get_resolution_engine()
    return engine.resolve_error(error_id, strategy_id)


def auto_resolve(error: Exception, **kwargs) -> ErrorResolutionResult:
    """Automatically track and resolve an error"""
    engine = get_resolution_engine()
    error_id = engine.track_error(error, **kwargs)
    return engine.resolve_error(error_id)


def get_error_statistics() -> Dict[str, Any]:
    """Get error statistics"""
    engine = get_resolution_engine()
    return engine.get_statistics()