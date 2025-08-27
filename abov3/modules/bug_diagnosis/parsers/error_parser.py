"""
Advanced error parsing and classification system
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import traceback

logger = logging.getLogger(__name__)

class ErrorType(Enum):
    """Classification of error types"""
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    TYPE_ERROR = "type_error"
    VALUE_ERROR = "value_error"
    INDEX_ERROR = "index_error"
    KEY_ERROR = "key_error"
    ATTRIBUTE_ERROR = "attribute_error"
    IMPORT_ERROR = "import_error"
    NAME_ERROR = "name_error"
    MEMORY_ERROR = "memory_error"
    RECURSION_ERROR = "recursion_error"
    TIMEOUT_ERROR = "timeout_error"
    CONNECTION_ERROR = "connection_error"
    PERMISSION_ERROR = "permission_error"
    FILE_NOT_FOUND = "file_not_found"
    NULL_REFERENCE = "null_reference"
    ASSERTION_ERROR = "assertion_error"
    COMPILATION_ERROR = "compilation_error"
    LINKER_ERROR = "linker_error"
    PERFORMANCE = "performance"
    MEMORY_LEAK = "memory_leak"
    RACE_CONDITION = "race_condition"
    DEADLOCK = "deadlock"
    SECURITY = "security"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    UNKNOWN = "unknown"

@dataclass
class ParsedError:
    """Parsed and classified error information"""
    error_type: ErrorType
    primary_message: str
    secondary_messages: List[str]
    stack_frames: List[Dict[str, Any]]
    language: Optional[str]
    framework: Optional[str]
    error_code: Optional[str]
    severity: str  # critical, high, medium, low
    confidence_score: float
    extracted_values: Dict[str, Any]
    patterns_matched: List[str]
    raw_error: str

class ErrorParser:
    """
    Sophisticated error parser supporting multiple languages and frameworks
    """
    
    def __init__(self):
        self.language_patterns = self._init_language_patterns()
        self.framework_patterns = self._init_framework_patterns()
        self.error_patterns = self._init_error_patterns()
        
    def _init_language_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize language-specific error patterns"""
        return {
            "python": {
                "stack_pattern": r'File "([^"]+)", line (\d+), in ([^\n]+)',
                "error_pattern": r'^(\w+Error): (.+)$',
                "traceback_start": "Traceback (most recent call last):",
                "common_errors": {
                    "AttributeError": ErrorType.ATTRIBUTE_ERROR,
                    "ImportError": ErrorType.IMPORT_ERROR,
                    "IndexError": ErrorType.INDEX_ERROR,
                    "KeyError": ErrorType.KEY_ERROR,
                    "NameError": ErrorType.NAME_ERROR,
                    "TypeError": ErrorType.TYPE_ERROR,
                    "ValueError": ErrorType.VALUE_ERROR,
                    "SyntaxError": ErrorType.SYNTAX_ERROR,
                    "IndentationError": ErrorType.SYNTAX_ERROR,
                    "RecursionError": ErrorType.RECURSION_ERROR,
                    "MemoryError": ErrorType.MEMORY_ERROR,
                    "FileNotFoundError": ErrorType.FILE_NOT_FOUND,
                    "PermissionError": ErrorType.PERMISSION_ERROR,
                    "ConnectionError": ErrorType.CONNECTION_ERROR,
                    "TimeoutError": ErrorType.TIMEOUT_ERROR,
                    "AssertionError": ErrorType.ASSERTION_ERROR
                }
            },
            "javascript": {
                "stack_pattern": r'at\s+([^\s]+)\s+\(([^:]+):(\d+):(\d+)\)',
                "error_pattern": r'^(\w+Error): (.+)$',
                "common_errors": {
                    "TypeError": ErrorType.TYPE_ERROR,
                    "ReferenceError": ErrorType.NAME_ERROR,
                    "SyntaxError": ErrorType.SYNTAX_ERROR,
                    "RangeError": ErrorType.VALUE_ERROR,
                    "URIError": ErrorType.VALUE_ERROR,
                    "EvalError": ErrorType.RUNTIME_ERROR
                }
            },
            "typescript": {
                "stack_pattern": r'at\s+([^\s]+)\s+\(([^:]+):(\d+):(\d+)\)',
                "error_pattern": r'^(TS\d+): (.+)$',
                "compilation_pattern": r'error TS(\d+): (.+)',
                "common_errors": {
                    "TS2304": ErrorType.NAME_ERROR,  # Cannot find name
                    "TS2339": ErrorType.ATTRIBUTE_ERROR,  # Property does not exist
                    "TS2345": ErrorType.TYPE_ERROR,  # Argument type mismatch
                    "TS2322": ErrorType.TYPE_ERROR,  # Type not assignable
                }
            },
            "java": {
                "stack_pattern": r'at\s+([\w\.]+)\(([\w\.]+):(\d+)\)',
                "error_pattern": r'^([\w\.]+Exception): (.+)$',
                "common_errors": {
                    "NullPointerException": ErrorType.NULL_REFERENCE,
                    "ArrayIndexOutOfBoundsException": ErrorType.INDEX_ERROR,
                    "ClassCastException": ErrorType.TYPE_ERROR,
                    "IllegalArgumentException": ErrorType.VALUE_ERROR,
                    "IOException": ErrorType.RUNTIME_ERROR,
                    "SQLException": ErrorType.CONNECTION_ERROR,
                    "OutOfMemoryError": ErrorType.MEMORY_ERROR,
                    "StackOverflowError": ErrorType.RECURSION_ERROR,
                    "SecurityException": ErrorType.PERMISSION_ERROR
                }
            },
            "csharp": {
                "stack_pattern": r'at\s+([\w\.]+)\s+in\s+([^:]+):line\s+(\d+)',
                "error_pattern": r'^System\.([\w\.]+Exception): (.+)$',
                "common_errors": {
                    "NullReferenceException": ErrorType.NULL_REFERENCE,
                    "IndexOutOfRangeException": ErrorType.INDEX_ERROR,
                    "InvalidCastException": ErrorType.TYPE_ERROR,
                    "ArgumentException": ErrorType.VALUE_ERROR,
                    "FileNotFoundException": ErrorType.FILE_NOT_FOUND,
                    "UnauthorizedAccessException": ErrorType.PERMISSION_ERROR,
                    "OutOfMemoryException": ErrorType.MEMORY_ERROR,
                    "StackOverflowException": ErrorType.RECURSION_ERROR
                }
            },
            "go": {
                "stack_pattern": r'([^:]+):(\d+)',
                "error_pattern": r'^(panic|error): (.+)$',
                "common_errors": {
                    "panic": ErrorType.RUNTIME_ERROR,
                    "nil pointer dereference": ErrorType.NULL_REFERENCE,
                    "index out of range": ErrorType.INDEX_ERROR,
                    "type assertion": ErrorType.TYPE_ERROR,
                    "deadlock": ErrorType.DEADLOCK
                }
            },
            "rust": {
                "stack_pattern": r'at\s+([^:]+):(\d+):(\d+)',
                "error_pattern": r'^error\[E(\d+)\]: (.+)$',
                "common_errors": {
                    "E0308": ErrorType.TYPE_ERROR,  # Mismatched types
                    "E0382": ErrorType.RUNTIME_ERROR,  # Use after move
                    "E0499": ErrorType.RUNTIME_ERROR,  # Cannot borrow as mutable
                    "E0507": ErrorType.RUNTIME_ERROR,  # Cannot move out
                }
            },
            "cpp": {
                "stack_pattern": r'#\d+\s+[0-9a-fx]+\s+in\s+([^\s]+)',
                "error_pattern": r'^(.+):\s*(.+)$',
                "common_errors": {
                    "segmentation fault": ErrorType.MEMORY_ERROR,
                    "assertion failed": ErrorType.ASSERTION_ERROR,
                    "undefined reference": ErrorType.LINKER_ERROR,
                    "no matching function": ErrorType.COMPILATION_ERROR
                }
            },
            "ruby": {
                "stack_pattern": r'from\s+([^:]+):(\d+):in\s+`([^\']+)\'',
                "error_pattern": r'^(\w+Error): (.+)$',
                "common_errors": {
                    "NoMethodError": ErrorType.ATTRIBUTE_ERROR,
                    "NameError": ErrorType.NAME_ERROR,
                    "TypeError": ErrorType.TYPE_ERROR,
                    "ArgumentError": ErrorType.VALUE_ERROR,
                    "SyntaxError": ErrorType.SYNTAX_ERROR,
                    "LoadError": ErrorType.IMPORT_ERROR
                }
            },
            "php": {
                "stack_pattern": r'#\d+\s+([^(]+)\((\d+)\)',
                "error_pattern": r'^(Fatal error|Parse error|Warning|Notice): (.+) in (.+) on line (\d+)',
                "common_errors": {
                    "Fatal error": ErrorType.RUNTIME_ERROR,
                    "Parse error": ErrorType.SYNTAX_ERROR,
                    "Undefined variable": ErrorType.NAME_ERROR,
                    "Call to undefined function": ErrorType.ATTRIBUTE_ERROR
                }
            }
        }
    
    def _init_framework_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize framework-specific patterns"""
        return {
            "django": {
                "indicators": ["django", "manage.py", "wsgi"],
                "error_patterns": {
                    "DoesNotExist": ErrorType.VALUE_ERROR,
                    "ValidationError": ErrorType.VALUE_ERROR,
                    "PermissionDenied": ErrorType.PERMISSION_ERROR,
                    "ImproperlyConfigured": ErrorType.CONFIGURATION
                }
            },
            "flask": {
                "indicators": ["flask", "werkzeug", "app.py"],
                "error_patterns": {
                    "BadRequest": ErrorType.VALUE_ERROR,
                    "NotFound": ErrorType.FILE_NOT_FOUND,
                    "MethodNotAllowed": ErrorType.RUNTIME_ERROR
                }
            },
            "react": {
                "indicators": ["react", "jsx", "webpack"],
                "error_patterns": {
                    "Cannot read property": ErrorType.NULL_REFERENCE,
                    "is not a function": ErrorType.TYPE_ERROR,
                    "Maximum update depth exceeded": ErrorType.RECURSION_ERROR
                }
            },
            "angular": {
                "indicators": ["@angular", "ng", "zone.js"],
                "error_patterns": {
                    "ExpressionChangedAfterItHasBeenCheckedError": ErrorType.RUNTIME_ERROR,
                    "NullInjectorError": ErrorType.DEPENDENCY
                }
            },
            "spring": {
                "indicators": ["springframework", "boot", "@Component"],
                "error_patterns": {
                    "BeanCreationException": ErrorType.CONFIGURATION,
                    "DataAccessException": ErrorType.CONNECTION_ERROR,
                    "NoSuchBeanDefinitionException": ErrorType.DEPENDENCY
                }
            },
            "express": {
                "indicators": ["express", "router", "middleware"],
                "error_patterns": {
                    "Cannot GET": ErrorType.FILE_NOT_FOUND,
                    "PayloadTooLargeError": ErrorType.VALUE_ERROR
                }
            },
            "fastapi": {
                "indicators": ["fastapi", "uvicorn", "starlette"],
                "error_patterns": {
                    "ValidationError": ErrorType.VALUE_ERROR,
                    "HTTPException": ErrorType.RUNTIME_ERROR
                }
            }
        }
    
    def _init_error_patterns(self) -> List[Dict[str, Any]]:
        """Initialize generic error patterns"""
        return [
            {
                "pattern": r"null|nil|undefined|None",
                "type": ErrorType.NULL_REFERENCE,
                "confidence": 0.7
            },
            {
                "pattern": r"index|bounds|range|length",
                "type": ErrorType.INDEX_ERROR,
                "confidence": 0.6
            },
            {
                "pattern": r"type|cast|convert|mismatch",
                "type": ErrorType.TYPE_ERROR,
                "confidence": 0.6
            },
            {
                "pattern": r"memory|heap|stack|leak|gc",
                "type": ErrorType.MEMORY_ERROR,
                "confidence": 0.7
            },
            {
                "pattern": r"timeout|deadline|expired",
                "type": ErrorType.TIMEOUT_ERROR,
                "confidence": 0.8
            },
            {
                "pattern": r"connection|network|socket|refused",
                "type": ErrorType.CONNECTION_ERROR,
                "confidence": 0.8
            },
            {
                "pattern": r"permission|denied|unauthorized|forbidden",
                "type": ErrorType.PERMISSION_ERROR,
                "confidence": 0.9
            },
            {
                "pattern": r"file.*not.*found|no such file",
                "type": ErrorType.FILE_NOT_FOUND,
                "confidence": 0.9
            },
            {
                "pattern": r"deadlock|lock|mutex|concurrent",
                "type": ErrorType.DEADLOCK,
                "confidence": 0.7
            },
            {
                "pattern": r"race|concurrent.*modification|thread.*safe",
                "type": ErrorType.RACE_CONDITION,
                "confidence": 0.7
            }
        ]
    
    async def parse(
        self,
        error_message: Optional[str] = None,
        stack_trace: Optional[str] = None,
        symptom_description: Optional[str] = None,
        code_snippet: Optional[str] = None,
        language: Optional[str] = None,
        framework: Optional[str] = None
    ) -> ParsedError:
        """
        Parse and classify the error from various inputs
        """
        # Combine all inputs for analysis
        raw_error = self._combine_inputs(
            error_message, stack_trace, symptom_description
        )
        
        # Detect language if not provided
        if not language:
            language = self._detect_language(raw_error, code_snippet)
        
        # Detect framework if not provided
        if not framework:
            framework = self._detect_framework(raw_error, code_snippet)
        
        # Parse stack trace
        stack_frames = self._parse_stack_trace(stack_trace, language)
        
        # Extract primary error message
        primary_message, error_type_detected = self._extract_error_message(
            error_message or raw_error, language
        )
        
        # Classify error type
        error_type = self._classify_error_type(
            primary_message, raw_error, language, error_type_detected
        )
        
        # Extract values and patterns
        extracted_values = self._extract_values(raw_error)
        patterns_matched = self._match_patterns(raw_error)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(
            error_type, patterns_matched, stack_frames, language
        )
        
        # Determine severity
        severity = self._determine_severity(error_type, raw_error)
        
        return ParsedError(
            error_type=error_type,
            primary_message=primary_message,
            secondary_messages=self._extract_secondary_messages(raw_error),
            stack_frames=stack_frames,
            language=language,
            framework=framework,
            error_code=self._extract_error_code(raw_error, language),
            severity=severity,
            confidence_score=confidence_score,
            extracted_values=extracted_values,
            patterns_matched=patterns_matched,
            raw_error=raw_error
        )
    
    def _combine_inputs(
        self,
        error_message: Optional[str],
        stack_trace: Optional[str],
        symptom: Optional[str]
    ) -> str:
        """Combine all input sources"""
        parts = []
        if error_message:
            parts.append(error_message)
        if stack_trace:
            parts.append(stack_trace)
        if symptom:
            parts.append(f"Symptom: {symptom}")
        return "\n".join(parts)
    
    def _detect_language(
        self, raw_error: str, code_snippet: Optional[str]
    ) -> Optional[str]:
        """Detect programming language from error and code"""
        # Check error patterns
        for lang, patterns in self.language_patterns.items():
            if any(indicator in raw_error.lower() for indicator in [lang, f".{lang[:2]}"]):
                return lang
        
        # Check code snippet
        if code_snippet:
            language_indicators = {
                "python": ["def ", "import ", "from ", "class ", "self."],
                "javascript": ["const ", "let ", "var ", "function ", "=>"],
                "typescript": ["interface ", "type ", ": string", ": number"],
                "java": ["public class", "private ", "void ", "static "],
                "csharp": ["namespace ", "using ", "public class"],
                "go": ["func ", "package ", "import ("],
                "rust": ["fn ", "let mut", "impl ", "pub "],
                "cpp": ["#include", "std::", "int main"],
                "ruby": ["def ", "end\n", "require ", "class "],
                "php": ["<?php", "$", "function ", "echo "]
            }
            
            for lang, indicators in language_indicators.items():
                if any(ind in code_snippet for ind in indicators):
                    return lang
        
        return None
    
    def _detect_framework(
        self, raw_error: str, code_snippet: Optional[str]
    ) -> Optional[str]:
        """Detect framework from error and code"""
        combined = raw_error + (code_snippet or "")
        
        for framework, info in self.framework_patterns.items():
            if any(indicator in combined.lower() for indicator in info["indicators"]):
                return framework
        
        return None
    
    def _parse_stack_trace(
        self, stack_trace: Optional[str], language: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Parse stack trace into structured format"""
        if not stack_trace:
            return []
        
        frames = []
        
        if language and language in self.language_patterns:
            pattern = self.language_patterns[language].get("stack_pattern")
            if pattern:
                matches = re.findall(pattern, stack_trace)
                for match in matches:
                    if language == "python":
                        frames.append({
                            "file": match[0],
                            "line": int(match[1]),
                            "function": match[2]
                        })
                    elif language in ["javascript", "typescript"]:
                        frames.append({
                            "function": match[0],
                            "file": match[1],
                            "line": int(match[2]),
                            "column": int(match[3])
                        })
                    elif language == "java":
                        frames.append({
                            "class": match[0],
                            "file": match[1],
                            "line": int(match[2])
                        })
        
        return frames
    
    def _extract_error_message(
        self, text: str, language: Optional[str]
    ) -> Tuple[str, Optional[str]]:
        """Extract the primary error message"""
        if language and language in self.language_patterns:
            pattern = self.language_patterns[language].get("error_pattern")
            if pattern:
                match = re.search(pattern, text, re.MULTILINE)
                if match:
                    return match.group(2) if len(match.groups()) > 1 else match.group(0), match.group(1)
        
        # Fallback: extract first line that looks like an error
        lines = text.split('\n')
        for line in lines:
            if any(word in line.lower() for word in ["error", "exception", "failed", "fatal"]):
                return line.strip(), None
        
        return text.split('\n')[0].strip() if text else "", None
    
    def _classify_error_type(
        self,
        message: str,
        raw_error: str,
        language: Optional[str],
        detected_type: Optional[str]
    ) -> ErrorType:
        """Classify the error type"""
        # Check language-specific error types
        if language and detected_type:
            lang_errors = self.language_patterns.get(language, {}).get("common_errors", {})
            if detected_type in lang_errors:
                return lang_errors[detected_type]
        
        # Check framework-specific patterns
        for framework, info in self.framework_patterns.items():
            for pattern, error_type in info.get("error_patterns", {}).items():
                if pattern.lower() in raw_error.lower():
                    return error_type
        
        # Check generic patterns
        for pattern_info in self.error_patterns:
            if re.search(pattern_info["pattern"], raw_error.lower()):
                return pattern_info["type"]
        
        return ErrorType.UNKNOWN
    
    def _extract_secondary_messages(self, raw_error: str) -> List[str]:
        """Extract secondary error messages"""
        messages = []
        lines = raw_error.split('\n')
        
        for line in lines[1:]:  # Skip first line (primary)
            if any(word in line.lower() for word in ["error", "warning", "failed", "caused by"]):
                messages.append(line.strip())
        
        return messages[:5]  # Limit to 5 secondary messages
    
    def _extract_error_code(
        self, raw_error: str, language: Optional[str]
    ) -> Optional[str]:
        """Extract error code if present"""
        # TypeScript error codes
        if language == "typescript":
            match = re.search(r'TS(\d+)', raw_error)
            if match:
                return f"TS{match.group(1)}"
        
        # Rust error codes
        if language == "rust":
            match = re.search(r'E(\d{4})', raw_error)
            if match:
                return f"E{match.group(1)}"
        
        # Generic error code patterns
        match = re.search(r'([A-Z]+[-_]?\d+)', raw_error)
        if match:
            return match.group(1)
        
        return None
    
    def _extract_values(self, raw_error: str) -> Dict[str, Any]:
        """Extract specific values from error message"""
        values = {}
        
        # Extract file paths
        file_matches = re.findall(r'["\']([^"\']+\.[a-zA-Z]+)["\']', raw_error)
        if file_matches:
            values["files"] = file_matches
        
        # Extract line numbers
        line_matches = re.findall(r'line (\d+)', raw_error)
        if line_matches:
            values["lines"] = [int(l) for l in line_matches]
        
        # Extract variable names
        var_matches = re.findall(r"'([a-zA-Z_]\w+)'", raw_error)
        if var_matches:
            values["variables"] = var_matches
        
        # Extract numeric values
        num_matches = re.findall(r'\b(\d+)\b', raw_error)
        if num_matches:
            values["numbers"] = [int(n) for n in num_matches]
        
        return values
    
    def _match_patterns(self, raw_error: str) -> List[str]:
        """Match known error patterns"""
        matched = []
        
        for pattern_info in self.error_patterns:
            if re.search(pattern_info["pattern"], raw_error.lower()):
                matched.append(pattern_info["pattern"])
        
        return matched
    
    def _calculate_confidence(
        self,
        error_type: ErrorType,
        patterns_matched: List[str],
        stack_frames: List[Dict[str, Any]],
        language: Optional[str]
    ) -> float:
        """Calculate confidence score for the parsing"""
        score = 0.0
        
        # Base score based on error type
        if error_type != ErrorType.UNKNOWN:
            score += 0.4
        
        # Pattern matching
        if patterns_matched:
            score += min(0.3, len(patterns_matched) * 0.1)
        
        # Stack trace quality
        if stack_frames:
            score += min(0.2, len(stack_frames) * 0.05)
        
        # Language detection
        if language:
            score += 0.1
        
        return min(1.0, score)
    
    def _determine_severity(self, error_type: ErrorType, raw_error: str) -> str:
        """Determine error severity"""
        critical_types = [
            ErrorType.MEMORY_ERROR,
            ErrorType.SECURITY,
            ErrorType.DEADLOCK,
            ErrorType.RACE_CONDITION
        ]
        
        high_types = [
            ErrorType.NULL_REFERENCE,
            ErrorType.RUNTIME_ERROR,
            ErrorType.PERMISSION_ERROR,
            ErrorType.CONNECTION_ERROR
        ]
        
        if error_type in critical_types:
            return "critical"
        elif error_type in high_types:
            return "high"
        elif "fatal" in raw_error.lower() or "critical" in raw_error.lower():
            return "critical"
        elif error_type == ErrorType.UNKNOWN:
            return "low"
        else:
            return "medium"