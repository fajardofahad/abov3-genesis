"""
ABOV3 Genesis - Code and Input Validator
Ensures code quality, correctness, and input validation
"""

import ast
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import subprocess
import tempfile
import os


class CodeValidator:
    """Validates generated code for syntax, security, and quality"""
    
    def __init__(self):
        self.security_patterns = self._initialize_security_patterns()
        self.quality_rules = self._initialize_quality_rules()
        self.language_validators = {
            'python': self.validate_python,
            'javascript': self.validate_javascript,
            'html': self.validate_html,
            'css': self.validate_css,
            'json': self.validate_json,
            'yaml': self.validate_yaml,
            'sql': self.validate_sql
        }
    
    def _initialize_security_patterns(self) -> Dict[str, List[str]]:
        """Initialize security vulnerability patterns"""
        return {
            'python': [
                r'eval\s*\(',  # Dangerous eval
                r'exec\s*\(',  # Dangerous exec
                r'__import__\s*\(',  # Dynamic imports
                r'os\.system\s*\(',  # Command injection risk
                r'subprocess\.call\s*\([^,]*shell\s*=\s*True',  # Shell injection
                r'pickle\.loads?\s*\(',  # Pickle deserialization
                r'input\s*\(\s*\).*eval',  # Eval with user input
            ],
            'javascript': [
                r'eval\s*\(',  # Dangerous eval
                r'innerHTML\s*=',  # XSS risk
                r'document\.write\s*\(',  # XSS risk
                r'setTimeout\s*\(["\']',  # String eval risk
                r'setInterval\s*\(["\']',  # String eval risk
                r'new\s+Function\s*\(',  # Dynamic function creation
            ],
            'sql': [
                r';\s*DROP\s+',  # SQL injection attempt
                r';\s*DELETE\s+FROM',  # SQL injection attempt
                r';\s*UPDATE\s+.*SET',  # SQL injection attempt
                r'OR\s+1\s*=\s*1',  # SQL injection pattern
                r'UNION\s+SELECT',  # SQL injection pattern
            ]
        }
    
    def _initialize_quality_rules(self) -> Dict[str, Any]:
        """Initialize code quality rules"""
        return {
            'max_line_length': 120,
            'max_function_length': 50,
            'max_file_length': 1000,
            'max_complexity': 10,
            'min_documentation': True,
            'naming_conventions': {
                'python': {
                    'function': r'^[a-z_][a-z0-9_]*$',
                    'class': r'^[A-Z][a-zA-Z0-9]*$',
                    'constant': r'^[A-Z_][A-Z0-9_]*$'
                },
                'javascript': {
                    'function': r'^[a-z][a-zA-Z0-9]*$',
                    'class': r'^[A-Z][a-zA-Z0-9]*$',
                    'constant': r'^[A-Z_][A-Z0-9_]*$'
                }
            }
        }
    
    def validate_code(self, code: str, language: str) -> Dict[str, Any]:
        """Main validation method"""
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'security_issues': [],
            'quality_issues': [],
            'metrics': {}
        }
        
        # Language-specific validation
        validator = self.language_validators.get(language.lower())
        if validator:
            lang_result = validator(code)
            result.update(lang_result)
        
        # Security validation
        security_result = self.validate_security(code, language)
        result['security_issues'] = security_result['issues']
        if security_result['issues']:
            result['valid'] = False
        
        # Quality validation
        quality_result = self.validate_quality(code, language)
        result['quality_issues'] = quality_result['issues']
        result['metrics'] = quality_result['metrics']
        
        return result
    
    def validate_python(self, code: str) -> Dict[str, Any]:
        """Validate Python code"""
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Parse Python code
            tree = ast.parse(code)
            
            # Check for syntax errors
            compile(code, '<string>', 'exec')
            
            # Analyze AST
            analyzer = PythonAnalyzer()
            analyzer.visit(tree)
            
            # Add analysis results
            result['warnings'].extend(analyzer.warnings)
            result['metrics'] = analyzer.metrics
            
        except SyntaxError as e:
            result['valid'] = False
            result['errors'].append(f"Syntax error at line {e.lineno}: {e.msg}")
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        # Check with pylint if available
        try:
            pylint_result = self._run_pylint(code)
            if pylint_result:
                result['warnings'].extend(pylint_result['warnings'])
        except:
            pass  # Pylint not available
        
        return result
    
    def validate_javascript(self, code: str) -> Dict[str, Any]:
        """Validate JavaScript code"""
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Basic syntax checks
        bracket_count = code.count('{') - code.count('}')
        paren_count = code.count('(') - code.count(')')
        bracket_sq_count = code.count('[') - code.count(']')
        
        if bracket_count != 0:
            result['errors'].append(f"Mismatched curly braces: {bracket_count} unclosed")
            result['valid'] = False
        
        if paren_count != 0:
            result['errors'].append(f"Mismatched parentheses: {paren_count} unclosed")
            result['valid'] = False
        
        if bracket_sq_count != 0:
            result['errors'].append(f"Mismatched square brackets: {bracket_sq_count} unclosed")
            result['valid'] = False
        
        # Check for common issues
        if 'var ' in code:
            result['warnings'].append("Consider using 'let' or 'const' instead of 'var'")
        
        if '==' in code and '===' not in code:
            result['warnings'].append("Consider using '===' for strict equality")
        
        # Check with ESLint if available
        try:
            eslint_result = self._run_eslint(code)
            if eslint_result:
                result['warnings'].extend(eslint_result['warnings'])
        except:
            pass  # ESLint not available
        
        return result
    
    def validate_html(self, code: str) -> Dict[str, Any]:
        """Validate HTML code"""
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check for basic structure
        if '<html' not in code.lower():
            result['warnings'].append("Missing <html> tag")
        
        if '<head' not in code.lower():
            result['warnings'].append("Missing <head> tag")
        
        if '<body' not in code.lower():
            result['warnings'].append("Missing <body> tag")
        
        # Check for unclosed tags
        tags = re.findall(r'<([^/>]+)>', code)
        tag_stack = []
        
        for tag in tags:
            tag_name = tag.split()[0].lower()
            if not tag_name.startswith('/'):
                # Self-closing tags
                if tag_name not in ['br', 'hr', 'img', 'input', 'meta', 'link']:
                    tag_stack.append(tag_name)
            else:
                closing_tag = tag_name[1:]
                if tag_stack and tag_stack[-1] == closing_tag:
                    tag_stack.pop()
                else:
                    result['errors'].append(f"Mismatched closing tag: </{closing_tag}>")
        
        if tag_stack:
            result['errors'].append(f"Unclosed tags: {', '.join(tag_stack)}")
            result['valid'] = False
        
        return result
    
    def validate_css(self, code: str) -> Dict[str, Any]:
        """Validate CSS code"""
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check for balanced braces
        if code.count('{') != code.count('}'):
            result['errors'].append("Mismatched curly braces in CSS")
            result['valid'] = False
        
        # Check for common issues
        if '!important' in code:
            count = code.count('!important')
            result['warnings'].append(f"Found {count} uses of !important - consider specificity instead")
        
        # Check for vendor prefixes
        if any(prefix in code for prefix in ['-webkit-', '-moz-', '-ms-', '-o-']):
            result['warnings'].append("Consider using autoprefixer for vendor prefixes")
        
        return result
    
    def validate_json(self, code: str) -> Dict[str, Any]:
        """Validate JSON code"""
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            json.loads(code)
        except json.JSONDecodeError as e:
            result['valid'] = False
            result['errors'].append(f"Invalid JSON: {str(e)}")
        
        return result
    
    def validate_yaml(self, code: str) -> Dict[str, Any]:
        """Validate YAML code"""
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            import yaml
            yaml.safe_load(code)
        except yaml.YAMLError as e:
            result['valid'] = False
            result['errors'].append(f"Invalid YAML: {str(e)}")
        except ImportError:
            result['warnings'].append("YAML validation skipped (pyyaml not installed)")
        
        return result
    
    def validate_sql(self, code: str) -> Dict[str, Any]:
        """Validate SQL code"""
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check for dangerous operations
        dangerous_keywords = ['DROP', 'TRUNCATE', 'DELETE FROM', 'UPDATE']
        for keyword in dangerous_keywords:
            if keyword in code.upper():
                result['warnings'].append(f"Found potentially dangerous operation: {keyword}")
        
        # Check for missing WHERE clause in UPDATE/DELETE
        if 'DELETE FROM' in code.upper() and 'WHERE' not in code.upper():
            result['warnings'].append("DELETE without WHERE clause - will delete all rows!")
        
        if 'UPDATE' in code.upper() and 'WHERE' not in code.upper():
            result['warnings'].append("UPDATE without WHERE clause - will update all rows!")
        
        return result
    
    def validate_security(self, code: str, language: str) -> Dict[str, Any]:
        """Check for security vulnerabilities"""
        result = {
            'secure': True,
            'issues': []
        }
        
        patterns = self.security_patterns.get(language.lower(), [])
        
        for pattern in patterns:
            matches = re.findall(pattern, code, re.IGNORECASE)
            if matches:
                result['secure'] = False
                result['issues'].append(f"Security risk: Found pattern '{pattern}' ({len(matches)} occurrences)")
        
        return result
    
    def validate_quality(self, code: str, language: str) -> Dict[str, Any]:
        """Check code quality"""
        result = {
            'issues': [],
            'metrics': {}
        }
        
        lines = code.split('\n')
        
        # Line length check
        long_lines = [i+1 for i, line in enumerate(lines) 
                     if len(line) > self.quality_rules['max_line_length']]
        if long_lines:
            result['issues'].append(f"Lines exceeding {self.quality_rules['max_line_length']} characters: {long_lines[:5]}")
        
        # File length check
        if len(lines) > self.quality_rules['max_file_length']:
            result['issues'].append(f"File too long: {len(lines)} lines (max: {self.quality_rules['max_file_length']})")
        
        # Calculate metrics
        result['metrics'] = {
            'lines_of_code': len(lines),
            'blank_lines': sum(1 for line in lines if not line.strip()),
            'comment_lines': sum(1 for line in lines if line.strip().startswith(('#', '//', '/*', '*'))),
            'average_line_length': sum(len(line) for line in lines) / len(lines) if lines else 0
        }
        
        return result
    
    def _run_pylint(self, code: str) -> Optional[Dict[str, Any]]:
        """Run pylint on Python code"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            result = subprocess.run(
                ['pylint', '--output-format=json', temp_file],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            os.unlink(temp_file)
            
            if result.stdout:
                messages = json.loads(result.stdout)
                warnings = [f"{msg['type']}: {msg['message']}" for msg in messages]
                return {'warnings': warnings}
            
        except Exception:
            pass
        
        return None
    
    def _run_eslint(self, code: str) -> Optional[Dict[str, Any]]:
        """Run ESLint on JavaScript code"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            result = subprocess.run(
                ['eslint', '--format=json', temp_file],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            os.unlink(temp_file)
            
            if result.stdout:
                data = json.loads(result.stdout)
                if data and data[0].get('messages'):
                    warnings = [f"{msg['severity']}: {msg['message']}" for msg in data[0]['messages']]
                    return {'warnings': warnings}
            
        except Exception:
            pass
        
        return None


class PythonAnalyzer(ast.NodeVisitor):
    """AST analyzer for Python code"""
    
    def __init__(self):
        self.warnings = []
        self.metrics = {
            'functions': 0,
            'classes': 0,
            'imports': 0,
            'complexity': 0
        }
        self.current_function = None
        self.function_lines = {}
    
    def visit_FunctionDef(self, node):
        self.metrics['functions'] += 1
        self.current_function = node.name
        
        # Check function length
        function_lines = node.end_lineno - node.lineno
        if function_lines > 50:
            self.warnings.append(f"Function '{node.name}' is too long ({function_lines} lines)")
        
        # Check for docstring
        if not ast.get_docstring(node):
            self.warnings.append(f"Function '{node.name}' lacks a docstring")
        
        # Calculate complexity
        self.metrics['complexity'] += self._calculate_complexity(node)
        
        self.generic_visit(node)
        self.current_function = None
    
    def visit_ClassDef(self, node):
        self.metrics['classes'] += 1
        
        # Check for docstring
        if not ast.get_docstring(node):
            self.warnings.append(f"Class '{node.name}' lacks a docstring")
        
        self.generic_visit(node)
    
    def visit_Import(self, node):
        self.metrics['imports'] += 1
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        self.metrics['imports'] += 1
        self.generic_visit(node)
    
    def _calculate_complexity(self, node) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
        return complexity


class InputValidator:
    """Validates user input for safety and correctness"""
    
    def __init__(self):
        self.max_input_length = 10000
        self.forbidden_patterns = [
            r'rm\s+-rf\s+/',  # Dangerous system commands
            r'format\s+c:',
            r'del\s+/f\s+/s\s+/q',
            r':(){ :|:& };:',  # Fork bomb
        ]
    
    def validate_user_input(self, input_text: str) -> Dict[str, Any]:
        """Validate user input"""
        result = {
            'valid': True,
            'sanitized_input': input_text,
            'warnings': [],
            'errors': []
        }
        
        # Length check
        if len(input_text) > self.max_input_length:
            result['errors'].append(f"Input too long: {len(input_text)} characters (max: {self.max_input_length})")
            result['valid'] = False
            result['sanitized_input'] = input_text[:self.max_input_length]
        
        # Check for forbidden patterns
        for pattern in self.forbidden_patterns:
            if re.search(pattern, input_text, re.IGNORECASE):
                result['errors'].append(f"Forbidden pattern detected: {pattern}")
                result['valid'] = False
                result['sanitized_input'] = re.sub(pattern, '[REMOVED]', input_text, flags=re.IGNORECASE)
        
        # Check for potential code injection
        if any(char in input_text for char in ['<script', '<?php', '<%', '${', '#{', '{{']):
            result['warnings'].append("Potential code injection attempt detected")
        
        # Sanitize special characters for file operations
        if any(char in input_text for char in ['../', '..\\', '~/', '\\\\', '|', '>', '<', '&']):
            result['warnings'].append("Special characters detected - sanitizing for safety")
            for char in ['../', '..\\', '~/', '\\\\', '|', '>', '<', '&']:
                result['sanitized_input'] = result['sanitized_input'].replace(char, '')
        
        return result
    
    def validate_file_path(self, path: str, base_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Validate file path for safety"""
        result = {
            'valid': True,
            'safe_path': path,
            'errors': []
        }
        
        try:
            path_obj = Path(path)
            
            # Check for path traversal
            if '..' in path_obj.parts:
                result['errors'].append("Path traversal detected")
                result['valid'] = False
            
            # Check if path is within base directory
            if base_dir:
                base_dir = Path(base_dir).resolve()
                try:
                    resolved_path = (base_dir / path_obj).resolve()
                    if not str(resolved_path).startswith(str(base_dir)):
                        result['errors'].append("Path outside of project directory")
                        result['valid'] = False
                    else:
                        result['safe_path'] = str(resolved_path.relative_to(base_dir))
                except:
                    result['errors'].append("Invalid path")
                    result['valid'] = False
            
        except Exception as e:
            result['errors'].append(f"Path validation error: {str(e)}")
            result['valid'] = False
        
        return result