"""
Validation Utility Functions

Helper functions for validating patch sets, file changes, and other operations.
"""

import os
import re
import ast
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ValidationUtils:
    """Utility class for validation operations"""
    
    # File path validation patterns
    INVALID_PATH_CHARS = r'[<>:"|?*\x00-\x1f]'
    RESERVED_NAMES = {
        'CON', 'PRN', 'AUX', 'NUL',
        'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
        'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    }
    
    @staticmethod
    def validate_file_path(file_path: str, project_root: str = None) -> Tuple[bool, str]:
        """
        Validate file path for safety and correctness
        
        Args:
            file_path: File path to validate
            project_root: Project root directory
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        
        if not file_path:
            return False, "File path cannot be empty"
        
        # Check for invalid characters
        if re.search(ValidationUtils.INVALID_PATH_CHARS, file_path):
            return False, "File path contains invalid characters"
        
        # Check for reserved names
        path_parts = Path(file_path).parts
        for part in path_parts:
            if part.upper() in ValidationUtils.RESERVED_NAMES:
                return False, f"File path contains reserved name: {part}"
        
        # Check path length (Windows limitation)
        if len(file_path) > 260:
            return False, "File path too long (exceeds 260 characters)"
        
        # Check for relative path traversal
        if '..' in path_parts or file_path.startswith('/'):
            return False, "File path contains directory traversal or absolute path"
        
        # Validate against project root if provided
        if project_root:
            try:
                full_path = Path(project_root) / file_path
                resolved_path = full_path.resolve()
                project_root_resolved = Path(project_root).resolve()
                
                # Ensure file is within project root
                try:
                    resolved_path.relative_to(project_root_resolved)
                except ValueError:
                    return False, "File path is outside project root"
                    
            except Exception as e:
                return False, f"Invalid file path: {e}"
        
        return True, ""
    
    @staticmethod
    def validate_patch_set_data(patch_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate patch set data structure
        
        Args:
            patch_data: Patch set data to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        
        errors = []
        
        # Required fields
        required_fields = ['id', 'description', 'files', 'created_at', 'created_by', 'status']
        
        for field in required_fields:
            if field not in patch_data:
                errors.append(f"Missing required field: {field}")
        
        # Validate files array
        if 'files' in patch_data:
            if not isinstance(patch_data['files'], list):
                errors.append("'files' must be a list")
            else:
                for i, file_change in enumerate(patch_data['files']):
                    file_errors = ValidationUtils.validate_file_change_data(file_change)
                    for error in file_errors[1]:
                        errors.append(f"File {i}: {error}")
        
        # Validate status
        if 'status' in patch_data:
            valid_statuses = ['draft', 'reviewing', 'approved', 'applied', 'rejected', 'rolled_back']
            if patch_data['status'] not in valid_statuses:
                errors.append(f"Invalid status: {patch_data['status']}")
        
        # Validate ID format
        if 'id' in patch_data:
            if not isinstance(patch_data['id'], str) or not patch_data['id'].strip():
                errors.append("ID must be a non-empty string")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_file_change_data(file_change: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate file change data structure
        
        Args:
            file_change: File change data to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        
        errors = []
        
        # Required fields
        required_fields = ['file_path', 'change_type']
        
        for field in required_fields:
            if field not in file_change:
                errors.append(f"Missing required field: {field}")
        
        # Validate change type
        if 'change_type' in file_change:
            valid_types = ['create', 'modify', 'delete', 'rename']
            if file_change['change_type'] not in valid_types:
                errors.append(f"Invalid change_type: {file_change['change_type']}")
        
        # Validate file path
        if 'file_path' in file_change:
            path_valid, path_error = ValidationUtils.validate_file_path(file_change['file_path'])
            if not path_valid:
                errors.append(f"Invalid file_path: {path_error}")
        
        # Validate rename operation
        if file_change.get('change_type') == 'rename':
            if 'old_path' not in file_change:
                errors.append("Rename operation requires 'old_path'")
            else:
                old_path_valid, old_path_error = ValidationUtils.validate_file_path(file_change['old_path'])
                if not old_path_valid:
                    errors.append(f"Invalid old_path: {old_path_error}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_code_syntax(content: str, file_extension: str) -> Tuple[bool, str]:
        """
        Validate code syntax for supported languages
        
        Args:
            content: Code content to validate
            file_extension: File extension (e.g., '.py', '.js')
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        
        if not content or not file_extension:
            return True, ""  # Empty content is valid
        
        try:
            if file_extension == '.py':
                return ValidationUtils._validate_python_syntax(content)
            elif file_extension in ['.js', '.ts']:
                return ValidationUtils._validate_javascript_syntax(content)
            elif file_extension == '.json':
                return ValidationUtils._validate_json_syntax(content)
            else:
                return True, ""  # Unsupported language, assume valid
                
        except Exception as e:
            return False, f"Syntax validation error: {e}"
    
    @staticmethod
    def _validate_python_syntax(content: str) -> Tuple[bool, str]:
        """Validate Python syntax"""
        try:
            ast.parse(content)
            return True, ""
        except SyntaxError as e:
            return False, f"Python syntax error: {e.msg} at line {e.lineno}"
        except Exception as e:
            return False, f"Python validation error: {e}"
    
    @staticmethod
    def _validate_javascript_syntax(content: str) -> Tuple[bool, str]:
        """Basic JavaScript syntax validation"""
        # Basic validation - check for balanced brackets
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        
        in_string = False
        in_comment = False
        escape_next = False
        
        for i, char in enumerate(content):
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            # Handle strings
            if char in ['"', "'"]:
                if not in_comment:
                    in_string = not in_string
                continue
            
            if in_string:
                continue
            
            # Handle comments
            if i < len(content) - 1:
                if content[i:i+2] == '//':
                    # Rest of line is comment
                    while i < len(content) and content[i] != '\n':
                        i += 1
                    continue
                elif content[i:i+2] == '/*':
                    in_comment = True
                    continue
            
            if in_comment:
                if i < len(content) - 1 and content[i:i+2] == '*/':
                    in_comment = False
                continue
            
            # Check brackets
            if char in brackets:
                stack.append(char)
            elif char in brackets.values():
                if not stack:
                    return False, f"Unmatched closing bracket '{char}' at position {i}"
                
                expected_closing = brackets[stack[-1]]
                if char != expected_closing:
                    return False, f"Mismatched bracket: expected '{expected_closing}', got '{char}' at position {i}"
                
                stack.pop()
        
        if stack:
            return False, f"Unmatched opening bracket '{stack[-1]}'"
        
        return True, ""
    
    @staticmethod
    def _validate_json_syntax(content: str) -> Tuple[bool, str]:
        """Validate JSON syntax"""
        try:
            json.loads(content)
            return True, ""
        except json.JSONDecodeError as e:
            return False, f"JSON syntax error: {e.msg} at line {e.lineno}, column {e.colno}"
        except Exception as e:
            return False, f"JSON validation error: {e}"
    
    @staticmethod
    def validate_transaction_data(transaction_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate transaction data structure
        
        Args:
            transaction_data: Transaction data to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        
        errors = []
        
        # Required fields
        required_fields = ['id', 'state', 'operations']
        
        for field in required_fields:
            if field not in transaction_data:
                errors.append(f"Missing required field: {field}")
        
        # Validate state
        if 'state' in transaction_data:
            valid_states = [
                'created', 'active', 'preparing', 'prepared', 
                'committing', 'committed', 'aborting', 'aborted', 'rolled_back'
            ]
            if transaction_data['state'] not in valid_states:
                errors.append(f"Invalid state: {transaction_data['state']}")
        
        # Validate operations
        if 'operations' in transaction_data:
            if not isinstance(transaction_data['operations'], list):
                errors.append("'operations' must be a list")
            else:
                for i, operation in enumerate(transaction_data['operations']):
                    op_errors = ValidationUtils.validate_operation_data(operation)
                    for error in op_errors[1]:
                        errors.append(f"Operation {i}: {error}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_operation_data(operation_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate operation data structure
        
        Args:
            operation_data: Operation data to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        
        errors = []
        
        # Required fields
        required_fields = ['id', 'operation_type', 'file_path']
        
        for field in required_fields:
            if field not in operation_data:
                errors.append(f"Missing required field: {field}")
        
        # Validate operation type
        if 'operation_type' in operation_data:
            valid_types = ['create', 'modify', 'delete', 'rename', 'copy', 'chmod']
            if operation_data['operation_type'] not in valid_types:
                errors.append(f"Invalid operation_type: {operation_data['operation_type']}")
        
        # Validate file path
        if 'file_path' in operation_data:
            path_valid, path_error = ValidationUtils.validate_file_path(operation_data['file_path'])
            if not path_valid:
                errors.append(f"Invalid file_path: {path_error}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_content_safety(content: str) -> Tuple[bool, List[str]]:
        """
        Validate content for safety (detect potentially malicious content)
        
        Args:
            content: Content to validate
            
        Returns:
            Tuple of (is_safe, list_of_warnings)
        """
        
        warnings = []
        
        # Check for potentially dangerous patterns
        dangerous_patterns = [
            (r'eval\s*\(', 'Potential code injection: eval()'),
            (r'exec\s*\(', 'Potential code injection: exec()'),
            (r'__import__\s*\(', 'Dynamic import detected'),
            (r'subprocess\s*\.', 'Subprocess execution detected'),
            (r'os\.system\s*\(', 'System command execution detected'),
            (r'shell\s*=\s*True', 'Shell execution enabled'),
            (r'rm\s+-rf\s+/', 'Potentially destructive command'),
            (r'del\s+/\w+', 'File deletion command'),
            (r'DROP\s+TABLE', 'SQL DROP statement'),
            (r'DELETE\s+FROM', 'SQL DELETE statement'),
            (r'<script[^>]*>', 'JavaScript injection potential'),
            (r'javascript\s*:', 'JavaScript protocol'),
        ]
        
        for pattern, warning in dangerous_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                warnings.append(warning)
        
        # Check for suspicious file operations
        file_patterns = [
            (r'open\s*\(\s*[\'"][^\'"]*/etc/', 'Access to system files'),
            (r'open\s*\(\s*[\'"][^\'"]*/proc/', 'Access to process files'),
            (r'open\s*\(\s*[\'"][^\'"]*/sys/', 'Access to system files'),
        ]
        
        for pattern, warning in file_patterns:
            if re.search(pattern, content):
                warnings.append(warning)
        
        # Content is considered safe if no high-risk patterns found
        high_risk_warnings = [w for w in warnings if 'injection' in w.lower() or 'execution' in w.lower()]
        is_safe = len(high_risk_warnings) == 0
        
        return is_safe, warnings
    
    @staticmethod
    def validate_diff_consistency(old_content: str, new_content: str, diff_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate that diff data is consistent with content changes
        
        Args:
            old_content: Original content
            new_content: Modified content
            diff_data: Diff data to validate
            
        Returns:
            Tuple of (is_consistent, error_message)
        """
        
        try:
            # Basic validation - check line counts
            old_lines = old_content.split('\n') if old_content else []
            new_lines = new_content.split('\n') if new_content else []
            
            if 'statistics' in diff_data:
                stats = diff_data['statistics']
                
                expected_old_lines = len(old_lines)
                expected_new_lines = len(new_lines)
                
                # Check if statistics make sense
                lines_added = stats.get('lines_added', 0)
                lines_removed = stats.get('lines_removed', 0)
                
                if expected_new_lines != expected_old_lines + lines_added - lines_removed:
                    return False, "Diff statistics don't match actual content changes"
            
            return True, ""
            
        except Exception as e:
            return False, f"Error validating diff consistency: {e}"
    
    @staticmethod
    def validate_permissions(permissions: int) -> Tuple[bool, str]:
        """
        Validate file permissions value
        
        Args:
            permissions: Permission value to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        
        # Check if permissions value is in valid range (0-0777 octal)
        if permissions < 0 or permissions > 0o777:
            return False, "Permissions must be between 0 and 0777 (octal)"
        
        # Check for common permission patterns
        if permissions & 0o111:  # Executable bits
            if not (permissions & 0o444):  # Should be readable if executable
                return False, "Executable files should typically be readable"
        
        return True, ""
    
    @staticmethod
    def sanitize_input(input_string: str, max_length: int = 1000) -> str:
        """
        Sanitize user input string
        
        Args:
            input_string: Input to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
        """
        
        if not isinstance(input_string, str):
            return ""
        
        # Remove control characters
        sanitized = ''.join(char for char in input_string if ord(char) >= 32 or char in '\n\r\t')
        
        # Limit length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        # Strip whitespace
        sanitized = sanitized.strip()
        
        return sanitized