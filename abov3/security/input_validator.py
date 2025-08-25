"""
ABOV3 Genesis - Input Validation and Sanitization
Comprehensive input validation system to prevent injection attacks and malicious input
"""

import re
import html
import json
import base64
import hashlib
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Set
from enum import Enum
from pathlib import Path
import urllib.parse
import bleach
from dataclasses import dataclass


class SecurityLevel(Enum):
    """Security validation levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"


class InputType(Enum):
    """Types of input for specialized validation"""
    TEXT = "text"
    CODE = "code"
    FILENAME = "filename"
    URL = "url"
    EMAIL = "email"
    JSON = "json"
    COMMAND = "command"
    SQL = "sql"
    HTML = "html"
    JAVASCRIPT = "javascript"
    PYTHON = "python"
    SHELL = "shell"


@dataclass
class ValidationRule:
    """Individual validation rule"""
    name: str
    pattern: str
    replacement: str = ""
    severity: str = "medium"
    description: str = ""


class InputValidator:
    """
    Comprehensive input validation and sanitization system
    Protects against injection attacks, XSS, command injection, and other threats
    """
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.HIGH, 
                 crypto_manager=None, audit_logger=None):
        self.security_level = security_level
        self.crypto_manager = crypto_manager
        self.audit_logger = audit_logger
        
        # Validation statistics
        self.validation_stats = {
            'total_validations': 0,
            'blocked_inputs': 0,
            'sanitized_inputs': 0,
            'high_risk_inputs': 0
        }
        
        # Initialize validation rules
        self._initialize_validation_rules()
        
        # Initialize allowed patterns
        self._initialize_allowed_patterns()
        
        # Initialize content filters
        self._initialize_content_filters()
    
    def _initialize_validation_rules(self):
        """Initialize comprehensive validation rules"""
        self.validation_rules = {
            # SQL Injection patterns
            InputType.SQL: [
                ValidationRule(
                    name="sql_injection_union",
                    pattern=r"(?i)(union\s+select|union\s+all\s+select)",
                    severity="high",
                    description="SQL UNION injection attempt"
                ),
                ValidationRule(
                    name="sql_injection_drop",
                    pattern=r"(?i)(drop\s+table|drop\s+database|drop\s+schema)",
                    severity="critical",
                    description="SQL DROP statement detected"
                ),
                ValidationRule(
                    name="sql_injection_delete",
                    pattern=r"(?i)(delete\s+from|truncate\s+table)",
                    severity="high", 
                    description="SQL DELETE/TRUNCATE detected"
                ),
                ValidationRule(
                    name="sql_injection_exec",
                    pattern=r"(?i)(exec\s*\(|execute\s*\(|sp_executesql)",
                    severity="critical",
                    description="SQL execution command detected"
                )
            ],
            
            # XSS patterns
            InputType.HTML: [
                ValidationRule(
                    name="xss_script_tags",
                    pattern=r"(?i)<script[^>]*>.*?</script>",
                    replacement="[SCRIPT_BLOCKED]",
                    severity="high",
                    description="Script tag detected"
                ),
                ValidationRule(
                    name="xss_javascript_protocol",
                    pattern=r"(?i)javascript:",
                    replacement="blocked:",
                    severity="medium",
                    description="JavaScript protocol detected"
                ),
                ValidationRule(
                    name="xss_on_events",
                    pattern=r"(?i)\bon\w+\s*=",
                    replacement="",
                    severity="medium",
                    description="HTML event handler detected"
                ),
                ValidationRule(
                    name="xss_iframe_injection",
                    pattern=r"(?i)<iframe[^>]*>.*?</iframe>",
                    replacement="[IFRAME_BLOCKED]",
                    severity="high",
                    description="IFrame injection detected"
                )
            ],
            
            # Command injection patterns
            InputType.COMMAND: [
                ValidationRule(
                    name="cmd_injection_pipe",
                    pattern=r"[|&;`$(){}[\]\\]",
                    replacement="",
                    severity="high",
                    description="Command injection characters detected"
                ),
                ValidationRule(
                    name="cmd_injection_backticks",
                    pattern=r"`[^`]*`",
                    replacement="",
                    severity="high",
                    description="Backtick command execution detected"
                ),
                ValidationRule(
                    name="cmd_injection_eval",
                    pattern=r"(?i)(eval\s*\(|exec\s*\()",
                    replacement="",
                    severity="critical",
                    description="Code evaluation detected"
                )
            ],
            
            # Path traversal patterns
            InputType.FILENAME: [
                ValidationRule(
                    name="path_traversal_dotdot",
                    pattern=r"\.\./",
                    replacement="",
                    severity="high",
                    description="Path traversal attempt detected"
                ),
                ValidationRule(
                    name="path_traversal_absolute",
                    pattern=r"^(/|\\|[a-zA-Z]:)",
                    replacement="",
                    severity="medium",
                    description="Absolute path detected"
                ),
                ValidationRule(
                    name="path_null_bytes",
                    pattern=r"\x00",
                    replacement="",
                    severity="high",
                    description="Null byte injection detected"
                )
            ],
            
            # Code injection patterns
            InputType.PYTHON: [
                ValidationRule(
                    name="python_exec_eval",
                    pattern=r"(?i)\b(exec|eval|compile|__import__)\s*\(",
                    severity="critical",
                    description="Dangerous Python function detected"
                ),
                ValidationRule(
                    name="python_os_system",
                    pattern=r"(?i)\b(os\.system|subprocess|popen)\s*\(",
                    severity="high",
                    description="System command execution detected"
                ),
                ValidationRule(
                    name="python_file_operations",
                    pattern=r"(?i)\b(open|file)\s*\([^)]*['\"][^'\"]*\.\./",
                    severity="medium",
                    description="Suspicious file operation detected"
                )
            ],
            
            # JavaScript injection patterns
            InputType.JAVASCRIPT: [
                ValidationRule(
                    name="js_eval_function",
                    pattern=r"(?i)\beval\s*\(",
                    severity="high",
                    description="JavaScript eval() detected"
                ),
                ValidationRule(
                    name="js_function_constructor",
                    pattern=r"(?i)new\s+function\s*\(",
                    severity="high",
                    description="Function constructor detected"
                ),
                ValidationRule(
                    name="js_document_write",
                    pattern=r"(?i)document\.write\s*\(",
                    severity="medium",
                    description="document.write detected"
                )
            ],
            
            # Shell injection patterns
            InputType.SHELL: [
                ValidationRule(
                    name="shell_command_substitution",
                    pattern=r"\$\([^)]*\)",
                    severity="high",
                    description="Shell command substitution detected"
                ),
                ValidationRule(
                    name="shell_redirection",
                    pattern=r"[<>]|>>|<<",
                    severity="medium",
                    description="Shell redirection detected"
                ),
                ValidationRule(
                    name="shell_dangerous_commands",
                    pattern=r"(?i)\b(rm\s+-rf|chmod\s+777|sudo|su\s)",
                    severity="critical",
                    description="Dangerous shell command detected"
                )
            ]
        }
    
    def _initialize_allowed_patterns(self):
        """Initialize patterns for allowed content"""
        self.allowed_patterns = {
            InputType.EMAIL: re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            InputType.URL: re.compile(r'^https?://[a-zA-Z0-9.-]+(/.*)?$'),
            InputType.FILENAME: re.compile(r'^[a-zA-Z0-9._-]+\.[a-zA-Z0-9]+$'),
            InputType.JSON: re.compile(r'^\s*[\{\[].*[\}\]]\s*$', re.DOTALL)
        }
    
    def _initialize_content_filters(self):
        """Initialize content filtering rules"""
        self.content_filters = {
            # Malicious keywords that should be blocked
            'malicious_keywords': {
                'virus', 'malware', 'backdoor', 'trojan', 'keylogger',
                'rootkit', 'botnet', 'exploit', 'payload', 'shellcode',
                'ransomware', 'cryptocurrency miner', 'cryptojacker'
            },
            
            # Dangerous file extensions
            'dangerous_extensions': {
                '.exe', '.bat', '.cmd', '.com', '.scr', '.pif', '.vbs',
                '.js', '.jar', '.msi', '.dll', '.so', '.dylib'
            },
            
            # Sensitive system paths
            'sensitive_paths': {
                '/etc/passwd', '/etc/shadow', 'C:\\Windows\\System32',
                '/root/', '/home/', '~/', '%USERPROFILE%', '%APPDATA%'
            },
            
            # Blocked protocols
            'blocked_protocols': {
                'file://', 'ftp://', 'ldap://', 'gopher://', 'dict://'
            }
        }
        
        # Configure bleach for HTML sanitization
        self.bleach_config = {
            'tags': ['p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li'],
            'attributes': {},
            'protocols': ['http', 'https'],
            'strip': True,
            'strip_comments': True
        }
    
    async def validate_input(self, input_data: Any, input_type: InputType = InputType.TEXT,
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Comprehensive input validation and sanitization
        
        Args:
            input_data: The input to validate
            input_type: Type of input for specialized validation
            context: Optional context for validation decisions
            
        Returns:
            Dict containing validation results
        """
        self.validation_stats['total_validations'] += 1
        
        result = {
            'valid': True,
            'sanitized_input': input_data,
            'errors': [],
            'warnings': [],
            'risk_score': 0,
            'blocked_patterns': [],
            'sanitization_applied': []
        }
        
        try:
            # Convert input to string for analysis
            if input_data is None:
                input_str = ""
            elif isinstance(input_data, (dict, list)):
                input_str = json.dumps(input_data)
            else:
                input_str = str(input_data)
            
            # Basic length validation
            max_length = self._get_max_length(input_type)
            if len(input_str) > max_length:
                result['valid'] = False
                result['errors'].append(f"Input exceeds maximum length of {max_length}")
                result['risk_score'] += 20
                
            # Character encoding validation
            encoding_result = await self._validate_encoding(input_str)
            if not encoding_result['valid']:
                result['valid'] = False
                result['errors'].extend(encoding_result['errors'])
                result['risk_score'] += encoding_result['risk_score']
            
            # Pattern-based validation
            pattern_result = await self._validate_patterns(input_str, input_type)
            if not pattern_result['valid']:
                result['valid'] = False
            result['errors'].extend(pattern_result['errors'])
            result['warnings'].extend(pattern_result['warnings'])
            result['blocked_patterns'].extend(pattern_result['blocked_patterns'])
            result['risk_score'] += pattern_result['risk_score']
            
            # Content filtering
            content_result = await self._filter_content(input_str, input_type)
            if not content_result['valid']:
                result['valid'] = False
            result['errors'].extend(content_result['errors'])
            result['warnings'].extend(content_result['warnings'])
            result['risk_score'] += content_result['risk_score']
            
            # Sanitization
            if result['valid'] or self.security_level in [SecurityLevel.LOW, SecurityLevel.MEDIUM]:
                sanitized_result = await self._sanitize_input(input_data, input_type)
                result['sanitized_input'] = sanitized_result['sanitized']
                result['sanitization_applied'].extend(sanitized_result['methods_applied'])
                if sanitized_result['sanitized'] != input_data:
                    self.validation_stats['sanitized_inputs'] += 1
            
            # Update statistics
            if not result['valid']:
                self.validation_stats['blocked_inputs'] += 1
            if result['risk_score'] > 70:
                self.validation_stats['high_risk_inputs'] += 1
            
            # Audit logging
            if self.audit_logger:
                await self.audit_logger.log_event("input_validation", {
                    "input_type": input_type.value,
                    "valid": result['valid'],
                    "risk_score": result['risk_score'],
                    "errors": len(result['errors']),
                    "warnings": len(result['warnings']),
                    "blocked_patterns": result['blocked_patterns'],
                    "sanitization_applied": result['sanitization_applied'],
                    "input_length": len(input_str),
                    "context": context
                })
            
            return result
            
        except Exception as e:
            result = {
                'valid': False,
                'sanitized_input': input_data,
                'errors': [f"Validation error: {str(e)}"],
                'warnings': [],
                'risk_score': 100,
                'blocked_patterns': [],
                'sanitization_applied': []
            }
            return result
    
    async def _validate_encoding(self, input_str: str) -> Dict[str, Any]:
        """Validate character encoding and detect suspicious characters"""
        result = {'valid': True, 'errors': [], 'risk_score': 0}
        
        try:
            # Check for null bytes
            if '\x00' in input_str:
                result['valid'] = False
                result['errors'].append("Null bytes detected")
                result['risk_score'] += 30
            
            # Check for control characters (except common ones)
            control_chars = [c for c in input_str if ord(c) < 32 and c not in '\t\n\r']
            if control_chars:
                result['valid'] = False
                result['errors'].append(f"Control characters detected: {control_chars}")
                result['risk_score'] += 20
            
            # Check for Unicode homograph attacks
            suspicious_unicode = self._detect_homograph_attack(input_str)
            if suspicious_unicode:
                result['errors'].append("Suspicious Unicode characters detected")
                result['risk_score'] += 15
            
            # Check for base64 encoded payloads
            base64_payloads = re.findall(r'[A-Za-z0-9+/]{20,}={0,2}', input_str)
            for payload in base64_payloads:
                try:
                    decoded = base64.b64decode(payload).decode('utf-8', 'ignore')
                    if any(keyword in decoded.lower() for keyword in ['script', 'eval', 'exec', 'system']):
                        result['errors'].append("Suspicious base64 encoded content")
                        result['risk_score'] += 25
                        break
                except:
                    pass
            
        except Exception as e:
            result['errors'].append(f"Encoding validation error: {str(e)}")
            result['risk_score'] += 10
        
        return result
    
    async def _validate_patterns(self, input_str: str, input_type: InputType) -> Dict[str, Any]:
        """Validate input against malicious patterns"""
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'blocked_patterns': [],
            'risk_score': 0
        }
        
        # Get relevant validation rules
        rules_to_check = []
        
        # Always check general patterns
        rules_to_check.extend(self.validation_rules.get(InputType.TEXT, []))
        
        # Add type-specific rules
        if input_type in self.validation_rules:
            rules_to_check.extend(self.validation_rules[input_type])
        
        # Check for code patterns in text input
        if input_type == InputType.TEXT:
            # Check if text looks like code
            code_indicators = ['import ', 'function ', 'var ', 'const ', 'def ', 'class ', '#!/']
            if any(indicator in input_str.lower() for indicator in code_indicators):
                rules_to_check.extend(self.validation_rules.get(InputType.CODE, []))
        
        # Apply validation rules
        for rule in rules_to_check:
            matches = re.findall(rule.pattern, input_str)
            if matches:
                severity_scores = {'low': 5, 'medium': 15, 'high': 30, 'critical': 50}
                score = severity_scores.get(rule.severity, 15)
                result['risk_score'] += score
                result['blocked_patterns'].append(rule.name)
                
                if rule.severity in ['high', 'critical']:
                    result['valid'] = False
                    result['errors'].append(f"{rule.description}: {rule.name}")
                else:
                    result['warnings'].append(f"{rule.description}: {rule.name}")
        
        return result
    
    async def _filter_content(self, input_str: str, input_type: InputType) -> Dict[str, Any]:
        """Filter content for malicious or sensitive information"""
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'risk_score': 0
        }
        
        input_lower = input_str.lower()
        
        # Check for malicious keywords
        found_keywords = [kw for kw in self.content_filters['malicious_keywords'] if kw in input_lower]
        if found_keywords:
            result['errors'].append(f"Malicious keywords detected: {found_keywords}")
            result['risk_score'] += len(found_keywords) * 10
            if self.security_level in [SecurityLevel.HIGH, SecurityLevel.MAXIMUM]:
                result['valid'] = False
        
        # Check for dangerous file extensions
        found_extensions = [ext for ext in self.content_filters['dangerous_extensions'] if ext in input_lower]
        if found_extensions:
            result['warnings'].append(f"Dangerous file extensions detected: {found_extensions}")
            result['risk_score'] += len(found_extensions) * 5
        
        # Check for sensitive system paths
        found_paths = [path for path in self.content_filters['sensitive_paths'] if path in input_str]
        if found_paths:
            result['warnings'].append(f"Sensitive system paths detected: {found_paths}")
            result['risk_score'] += len(found_paths) * 8
        
        # Check for blocked protocols
        found_protocols = [proto for proto in self.content_filters['blocked_protocols'] if proto in input_lower]
        if found_protocols:
            result['errors'].append(f"Blocked protocols detected: {found_protocols}")
            result['risk_score'] += len(found_protocols) * 15
            if self.security_level in [SecurityLevel.HIGH, SecurityLevel.MAXIMUM]:
                result['valid'] = False
        
        # Check for credential patterns
        credential_patterns = [
            r'password\s*[:=]\s*[^\s]+',
            r'api_key\s*[:=]\s*[^\s]+',
            r'secret\s*[:=]\s*[^\s]+',
            r'token\s*[:=]\s*[^\s]+'
        ]
        
        for pattern in credential_patterns:
            if re.search(pattern, input_lower):
                result['warnings'].append("Potential credential information detected")
                result['risk_score'] += 20
                break
        
        return result
    
    async def _sanitize_input(self, input_data: Any, input_type: InputType) -> Dict[str, Any]:
        """Sanitize input based on type and security level"""
        result = {
            'sanitized': input_data,
            'methods_applied': []
        }
        
        try:
            if isinstance(input_data, str):
                sanitized = input_data
                
                # HTML sanitization
                if input_type == InputType.HTML or '<' in sanitized:
                    sanitized = bleach.clean(sanitized, **self.bleach_config)
                    if sanitized != input_data:
                        result['methods_applied'].append('html_sanitization')
                
                # URL encoding for URLs
                elif input_type == InputType.URL:
                    sanitized = urllib.parse.quote(sanitized, safe=':/?#[]@!$&\'()*+,;=')
                    if sanitized != input_data:
                        result['methods_applied'].append('url_encoding')
                
                # Escape special characters for filenames
                elif input_type == InputType.FILENAME:
                    sanitized = re.sub(r'[<>:"/\\|?*]', '_', sanitized)
                    sanitized = sanitized[:255]  # Limit filename length
                    if sanitized != input_data:
                        result['methods_applied'].append('filename_sanitization')
                
                # Remove dangerous patterns from code
                elif input_type in [InputType.CODE, InputType.PYTHON, InputType.JAVASCRIPT]:
                    # Apply code-specific sanitization based on security level
                    if self.security_level == SecurityLevel.MAXIMUM:
                        # Very strict sanitization
                        sanitized = self._sanitize_code_maximum(sanitized, input_type)
                        result['methods_applied'].append('maximum_code_sanitization')
                    elif self.security_level == SecurityLevel.HIGH:
                        # High-level sanitization
                        sanitized = self._sanitize_code_high(sanitized, input_type)
                        result['methods_applied'].append('high_code_sanitization')
                
                # General text sanitization
                else:
                    # Remove null bytes
                    if '\x00' in sanitized:
                        sanitized = sanitized.replace('\x00', '')
                        result['methods_applied'].append('null_byte_removal')
                    
                    # HTML escape if needed
                    if '<' in sanitized or '>' in sanitized:
                        sanitized = html.escape(sanitized)
                        result['methods_applied'].append('html_escaping')
                
                result['sanitized'] = sanitized
                
            elif isinstance(input_data, dict):
                # Recursively sanitize dictionary values
                sanitized_dict = {}
                for key, value in input_data.items():
                    # Sanitize key
                    sanitized_key_result = await self._sanitize_input(key, InputType.TEXT)
                    sanitized_key = sanitized_key_result['sanitized']
                    
                    # Sanitize value
                    sanitized_value_result = await self._sanitize_input(value, input_type)
                    sanitized_value = sanitized_value_result['sanitized']
                    
                    sanitized_dict[sanitized_key] = sanitized_value
                    
                    # Track sanitization methods
                    result['methods_applied'].extend(sanitized_key_result['methods_applied'])
                    result['methods_applied'].extend(sanitized_value_result['methods_applied'])
                
                result['sanitized'] = sanitized_dict
                result['methods_applied'].append('recursive_dict_sanitization')
                
            elif isinstance(input_data, list):
                # Recursively sanitize list items
                sanitized_list = []
                for item in input_data:
                    sanitized_item_result = await self._sanitize_input(item, input_type)
                    sanitized_list.append(sanitized_item_result['sanitized'])
                    result['methods_applied'].extend(sanitized_item_result['methods_applied'])
                
                result['sanitized'] = sanitized_list
                result['methods_applied'].append('recursive_list_sanitization')
        
        except Exception as e:
            # If sanitization fails, return original data with error logged
            result['sanitized'] = input_data
            result['methods_applied'].append(f'sanitization_error: {str(e)}')
        
        return result
    
    def _sanitize_code_maximum(self, code: str, input_type: InputType) -> str:
        """Maximum security code sanitization"""
        # Remove extremely dangerous patterns
        dangerous_functions = [
            r'\beval\s*\(',
            r'\bexec\s*\(',
            r'\b__import__\s*\(',
            r'\bcompile\s*\(',
            r'\bgetattr\s*\(',
            r'\bsetattr\s*\(',
            r'\bdelattr\s*\(',
            r'\bglobals\s*\(',
            r'\blocals\s*\(',
            r'\bvars\s*\(',
            r'\bdir\s*\(',
            r'\binput\s*\(',
            r'\braw_input\s*\(',
            r'os\.system\s*\(',
            r'subprocess\.',
            r'commands\.',
            r'popen\s*\(',
        ]
        
        sanitized = code
        for pattern in dangerous_functions:
            sanitized = re.sub(pattern, '# [BLOCKED_FUNCTION]', sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    def _sanitize_code_high(self, code: str, input_type: InputType) -> str:
        """High security code sanitization"""
        # Remove moderately dangerous patterns but allow more functionality
        dangerous_patterns = [
            r'\beval\s*\(',
            r'\bexec\s*\(',
            r'os\.system\s*\(',
            r'subprocess\.call\s*\(',
            r'subprocess\.run\s*\(',
        ]
        
        sanitized = code
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, '# [BLOCKED_FUNCTION]', sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    def _detect_homograph_attack(self, text: str) -> bool:
        """Detect potential Unicode homograph attacks"""
        # Check for mixing of different scripts that could be confusing
        scripts = set()
        for char in text:
            if char.isalpha():
                code = ord(char)
                if 0x0041 <= code <= 0x007A:  # Latin
                    scripts.add('latin')
                elif 0x0400 <= code <= 0x04FF:  # Cyrillic
                    scripts.add('cyrillic')
                elif 0x0370 <= code <= 0x03FF:  # Greek
                    scripts.add('greek')
        
        # If mixing scripts, it could be a homograph attack
        return len(scripts) > 1
    
    def _get_max_length(self, input_type: InputType) -> int:
        """Get maximum allowed length for input type"""
        max_lengths = {
            InputType.TEXT: 10000,
            InputType.CODE: 100000,
            InputType.FILENAME: 255,
            InputType.URL: 2048,
            InputType.EMAIL: 254,
            InputType.JSON: 50000,
            InputType.COMMAND: 1000,
            InputType.SQL: 5000,
            InputType.HTML: 50000,
            InputType.JAVASCRIPT: 50000,
            InputType.PYTHON: 100000,
            InputType.SHELL: 1000
        }
        
        base_length = max_lengths.get(input_type, 10000)
        
        # Adjust based on security level
        if self.security_level == SecurityLevel.MAXIMUM:
            return int(base_length * 0.5)
        elif self.security_level == SecurityLevel.HIGH:
            return int(base_length * 0.8)
        elif self.security_level == SecurityLevel.LOW:
            return int(base_length * 2)
        
        return base_length
    
    async def validate_batch(self, inputs: List[Any], input_types: List[InputType] = None) -> List[Dict[str, Any]]:
        """Validate multiple inputs in batch"""
        if input_types is None:
            input_types = [InputType.TEXT] * len(inputs)
        
        if len(inputs) != len(input_types):
            raise ValueError("Number of inputs must match number of input types")
        
        tasks = []
        for i, (input_data, input_type) in enumerate(zip(inputs, input_types)):
            task = self.validate_input(input_data, input_type, context={'batch_index': i})
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return self.validation_stats.copy()
    
    def reset_statistics(self):
        """Reset validation statistics"""
        self.validation_stats = {
            'total_validations': 0,
            'blocked_inputs': 0,
            'sanitized_inputs': 0,
            'high_risk_inputs': 0
        }