"""
ABOV3 Genesis - File System Security Guard
Comprehensive file system access controls and sandboxing for secure file operations
"""

import os
import stat
import hashlib
import asyncio
import mimetypes
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Union
from enum import Enum
from dataclasses import dataclass
import shutil
import json
import re

from .input_validator import SecurityLevel


class FileOperation(Enum):
    """Types of file operations"""
    READ = "read"
    WRITE = "write"
    CREATE = "create"
    DELETE = "delete"
    MOVE = "move"
    COPY = "copy"
    EXECUTE = "execute"
    MODIFY_PERMISSIONS = "modify_permissions"
    LIST_DIRECTORY = "list_directory"
    CREATE_DIRECTORY = "create_directory"


class AccessLevel(Enum):
    """File access levels"""
    NONE = "none"
    READ_ONLY = "read_only"
    WRITE_ONLY = "write_only"
    READ_WRITE = "read_write"
    FULL_ACCESS = "full_access"


@dataclass
class FileRule:
    """File access rule definition"""
    path_pattern: str
    allowed_operations: Set[FileOperation]
    access_level: AccessLevel
    max_file_size: int
    allowed_extensions: Set[str]
    description: str
    priority: int = 0


@dataclass
class SandboxConfig:
    """Sandbox configuration"""
    sandbox_root: Path
    max_total_size: int
    max_files: int
    allowed_extensions: Set[str]
    blocked_extensions: Set[str]
    enable_quarantine: bool
    quarantine_path: Optional[Path]


class FileSystemGuard:
    """
    Comprehensive File System Security Guard
    Provides sandboxing, access controls, and security monitoring for file operations
    """
    
    def __init__(self, project_path: Path, security_level: SecurityLevel = SecurityLevel.HIGH,
                 max_file_size: int = 100 * 1024 * 1024, audit_logger=None):
        self.project_path = project_path
        self.security_level = security_level
        self.max_file_size = max_file_size
        self.audit_logger = audit_logger
        
        # Security statistics
        self.security_stats = {
            'total_operations': 0,
            'blocked_operations': 0,
            'quarantined_files': 0,
            'suspicious_files': 0,
            'by_operation_type': {}
        }
        
        # Initialize sandbox
        self.sandbox_root = project_path / '.abov3' / 'sandbox'
        self.sandbox_root.mkdir(parents=True, exist_ok=True)
        
        # Initialize quarantine
        self.quarantine_path = project_path / '.abov3' / 'quarantine'
        self.quarantine_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize file rules
        self._initialize_file_rules()
        
        # Initialize allowed/blocked patterns
        self._initialize_file_patterns()
        
        # File tracking
        self.tracked_files: Dict[str, Dict[str, Any]] = {}
        self.file_hashes: Dict[str, str] = {}
        
        # Sandbox configuration
        self.sandbox_config = SandboxConfig(
            sandbox_root=self.sandbox_root,
            max_total_size=500 * 1024 * 1024,  # 500MB
            max_files=1000,
            allowed_extensions=self._get_allowed_extensions(),
            blocked_extensions=self._get_blocked_extensions(),
            enable_quarantine=True,
            quarantine_path=self.quarantine_path
        )
    
    def _initialize_file_rules(self):
        """Initialize file access rules based on security level"""
        self.file_rules = []
        
        # Project root access rules
        self.file_rules.extend([
            FileRule(
                path_pattern=str(self.project_path / "**" / "*.py"),
                allowed_operations={FileOperation.READ, FileOperation.WRITE, FileOperation.CREATE},
                access_level=AccessLevel.READ_WRITE,
                max_file_size=1024 * 1024,  # 1MB for Python files
                allowed_extensions={'.py'},
                description="Python source files",
                priority=10
            ),
            FileRule(
                path_pattern=str(self.project_path / "**" / "*.js"),
                allowed_operations={FileOperation.READ, FileOperation.WRITE, FileOperation.CREATE},
                access_level=AccessLevel.READ_WRITE,
                max_file_size=512 * 1024,  # 512KB for JS files
                allowed_extensions={'.js', '.jsx'},
                description="JavaScript source files",
                priority=10
            ),
            FileRule(
                path_pattern=str(self.project_path / "**" / "*.html"),
                allowed_operations={FileOperation.READ, FileOperation.WRITE, FileOperation.CREATE},
                access_level=AccessLevel.READ_WRITE,
                max_file_size=512 * 1024,  # 512KB for HTML files
                allowed_extensions={'.html', '.htm'},
                description="HTML files",
                priority=10
            ),
            FileRule(
                path_pattern=str(self.project_path / "**" / "*.css"),
                allowed_operations={FileOperation.READ, FileOperation.WRITE, FileOperation.CREATE},
                access_level=AccessLevel.READ_WRITE,
                max_file_size=256 * 1024,  # 256KB for CSS files
                allowed_extensions={'.css', '.scss', '.sass'},
                description="CSS stylesheets",
                priority=10
            ),
        ])
        
        # Documentation and config files
        self.file_rules.extend([
            FileRule(
                path_pattern=str(self.project_path / "**" / "*.md"),
                allowed_operations={FileOperation.READ, FileOperation.WRITE, FileOperation.CREATE},
                access_level=AccessLevel.READ_WRITE,
                max_file_size=1024 * 1024,  # 1MB for markdown
                allowed_extensions={'.md'},
                description="Markdown documentation",
                priority=8
            ),
            FileRule(
                path_pattern=str(self.project_path / "**" / "*.json"),
                allowed_operations={FileOperation.READ, FileOperation.WRITE, FileOperation.CREATE},
                access_level=AccessLevel.READ_WRITE,
                max_file_size=256 * 1024,  # 256KB for JSON
                allowed_extensions={'.json'},
                description="JSON configuration",
                priority=8
            ),
            FileRule(
                path_pattern=str(self.project_path / "**" / "*.yaml"),
                allowed_operations={FileOperation.READ, FileOperation.WRITE, FileOperation.CREATE},
                access_level=AccessLevel.READ_WRITE,
                max_file_size=256 * 1024,  # 256KB for YAML
                allowed_extensions={'.yaml', '.yml'},
                description="YAML configuration",
                priority=8
            ),
        ])
        
        # System and sensitive files - restricted access
        system_patterns = [
            "/etc/**",
            "/sys/**", 
            "/proc/**",
            "/dev/**",
            "C:/Windows/**",
            "C:/System32/**",
            "/root/**",
            "/home/*/.ssh/**",
            "**/.env",
            "**/id_rsa*",
            "**/id_ed25519*",
            "**/.aws/**",
            "**/.ssh/**"
        ]
        
        for pattern in system_patterns:
            self.file_rules.append(
                FileRule(
                    path_pattern=pattern,
                    allowed_operations=set(),  # No operations allowed
                    access_level=AccessLevel.NONE,
                    max_file_size=0,
                    allowed_extensions=set(),
                    description="System/sensitive file - blocked",
                    priority=100  # Highest priority
                )
            )
        
        # Executable files - special handling
        executable_extensions = {'.exe', '.bat', '.cmd', '.sh', '.ps1', '.msi', '.dmg', '.pkg'}
        if self.security_level in [SecurityLevel.HIGH, SecurityLevel.MAXIMUM]:
            self.file_rules.append(
                FileRule(
                    path_pattern="**/*",
                    allowed_operations={FileOperation.READ} if self.security_level == SecurityLevel.HIGH else set(),
                    access_level=AccessLevel.READ_ONLY if self.security_level == SecurityLevel.HIGH else AccessLevel.NONE,
                    max_file_size=0,
                    allowed_extensions=executable_extensions,
                    description="Executable files - restricted",
                    priority=90
                )
            )
        
        # Sandbox-specific rules
        self.file_rules.append(
            FileRule(
                path_pattern=str(self.sandbox_root / "**" / "*"),
                allowed_operations={FileOperation.READ, FileOperation.WRITE, FileOperation.CREATE, 
                                 FileOperation.DELETE, FileOperation.MOVE, FileOperation.COPY},
                access_level=AccessLevel.FULL_ACCESS,
                max_file_size=self.max_file_size,
                allowed_extensions=self._get_allowed_extensions(),
                description="Sandbox area - full access",
                priority=5
            )
        )
    
    def _initialize_file_patterns(self):
        """Initialize allowed and blocked file patterns"""
        
        # Dangerous file patterns that should be blocked
        self.dangerous_patterns = [
            r'.*\.(exe|bat|cmd|com|scr|pif|vbs|js|jar|msi)$',
            r'.*\.(dll|so|dylib)$',
            r'.*\.php\?.*',  # PHP with parameters
            r'.*\.jsp\?.*',  # JSP with parameters
            r'.*\.(asp|aspx)\?.*',  # ASP with parameters
            r'.*\/\.\.\/.*',  # Path traversal
            r'.*\\\.\.\\.*',  # Path traversal (Windows)
            r'.*\0.*',  # Null bytes
        ]
        
        # Suspicious filename patterns
        self.suspicious_patterns = [
            r'.*virus.*',
            r'.*malware.*',
            r'.*trojan.*',
            r'.*backdoor.*',
            r'.*keylog.*',
            r'.*rootkit.*',
            r'.*payload.*',
            r'.*exploit.*',
            r'.*shell.*\.php',
            r'.*cmd.*\.asp',
            r'.*passwd.*',
            r'.*shadow.*',
            r'.*\.htaccess',
            r'.*\.htpasswd',
            r'.*web\.config',
        ]
        
        # MIME types that are allowed
        self.allowed_mime_types = {
            'text/plain',
            'text/html',
            'text/css',
            'text/javascript',
            'text/markdown',
            'application/json',
            'application/xml',
            'text/xml',
            'application/yaml',
            'text/yaml',
            'application/python',
            'text/python',
            'image/png',
            'image/jpeg',
            'image/gif',
            'image/svg+xml'
        }
        
        # MIME types that are blocked
        self.blocked_mime_types = {
            'application/x-executable',
            'application/x-msdos-program',
            'application/x-ms-dos-executable',
            'application/x-winexe',
            'application/x-msdownload',
            'application/vnd.microsoft.portable-executable',
            'application/x-sharedlib',
            'application/x-shellscript'
        }
    
    def _get_allowed_extensions(self) -> Set[str]:
        """Get allowed file extensions based on security level"""
        base_extensions = {
            '.py', '.js', '.jsx', '.ts', '.tsx', '.html', '.htm', '.css', '.scss', '.sass',
            '.json', '.yaml', '.yml', '.md', '.txt', '.xml', '.sql', '.csv',
            '.java', '.cpp', '.c', '.h', '.hpp', '.rs', '.go', '.php', '.rb',
            '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.webp'
        }
        
        if self.security_level == SecurityLevel.LOW:
            # Allow more extensions in low security
            base_extensions.update({
                '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
                '.zip', '.tar', '.gz', '.bz2', '.rar', '.7z'
            })
        elif self.security_level == SecurityLevel.MAXIMUM:
            # Very restrictive in maximum security
            base_extensions = {
                '.py', '.js', '.html', '.css', '.json', '.yaml', '.md', '.txt'
            }
        
        return base_extensions
    
    def _get_blocked_extensions(self) -> Set[str]:
        """Get blocked file extensions"""
        blocked = {
            '.exe', '.bat', '.cmd', '.com', '.scr', '.pif', '.vbs', '.jar',
            '.msi', '.deb', '.rpm', '.dmg', '.pkg', '.app', '.dll', '.so', '.dylib'
        }
        
        if self.security_level in [SecurityLevel.HIGH, SecurityLevel.MAXIMUM]:
            # Block more extensions in high security
            blocked.update({
                '.sh', '.bash', '.zsh', '.fish', '.ps1', '.psm1',
                '.php', '.asp', '.aspx', '.jsp', '.cgi', '.pl', '.rb'
            })
        
        return blocked
    
    async def validate_file_operation(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a file operation request
        
        Args:
            operation: Dict containing operation details
                - type: FileOperation type
                - source_path: Source file path
                - target_path: Target file path (for move/copy)
                - content: File content (for write operations)
                - user_context: Optional user context
        
        Returns:
            Dict containing validation results
        """
        self.security_stats['total_operations'] += 1
        
        result = {
            'allowed': True,
            'reason': '',
            'warnings': [],
            'sanitized_operation': operation.copy(),
            'risk_score': 0,
            'quarantine_required': False
        }
        
        try:
            op_type = FileOperation(operation.get('type', 'read'))
            source_path = Path(operation.get('source_path', ''))
            target_path = Path(operation.get('target_path', '')) if operation.get('target_path') else None
            content = operation.get('content', '')
            user_context = operation.get('user_context', {})
            
            # Update statistics
            op_type_str = op_type.value
            self.security_stats['by_operation_type'][op_type_str] = \
                self.security_stats['by_operation_type'].get(op_type_str, 0) + 1
            
            # Basic path validation
            path_result = await self._validate_path(source_path, op_type)
            if not path_result['valid']:
                result['allowed'] = False
                result['reason'] = path_result['reason']
                result['risk_score'] += path_result['risk_score']
                return result
            
            result['warnings'].extend(path_result['warnings'])
            result['risk_score'] += path_result['risk_score']
            
            # Validate target path if present
            if target_path:
                target_result = await self._validate_path(target_path, op_type)
                if not target_result['valid']:
                    result['allowed'] = False
                    result['reason'] = f"Target path invalid: {target_result['reason']}"
                    result['risk_score'] += target_result['risk_score']
                    return result
                result['warnings'].extend(target_result['warnings'])
                result['risk_score'] += target_result['risk_score']
            
            # Check file rules
            rule_result = await self._check_file_rules(source_path, op_type)
            if not rule_result['allowed']:
                result['allowed'] = False
                result['reason'] = rule_result['reason']
                result['risk_score'] += rule_result['risk_score']
                return result
            
            result['warnings'].extend(rule_result['warnings'])
            result['risk_score'] += rule_result['risk_score']
            
            # Content validation for write operations
            if op_type in [FileOperation.WRITE, FileOperation.CREATE] and content:
                content_result = await self._validate_content(content, source_path)
                if not content_result['valid']:
                    result['allowed'] = False
                    result['reason'] = content_result['reason']
                    result['risk_score'] += content_result['risk_score']
                    return result
                
                result['warnings'].extend(content_result['warnings'])
                result['risk_score'] += content_result['risk_score']
                result['sanitized_operation']['content'] = content_result['sanitized_content']
                
                if content_result['quarantine_required']:
                    result['quarantine_required'] = True
            
            # File size validation
            if op_type in [FileOperation.WRITE, FileOperation.CREATE]:
                size_limit = self._get_size_limit_for_path(source_path)
                content_size = len(content.encode('utf-8')) if content else 0
                
                if content_size > size_limit:
                    result['allowed'] = False
                    result['reason'] = f"File size {content_size} exceeds limit {size_limit}"
                    result['risk_score'] += 30
                    return result
            
            # Sandbox enforcement
            if self.security_level == SecurityLevel.MAXIMUM and not self._is_in_sandbox(source_path):
                # Force operation into sandbox
                sandbox_path = self.sandbox_root / source_path.name
                result['sanitized_operation']['source_path'] = str(sandbox_path)
                result['warnings'].append("Operation redirected to sandbox")
            
            # Final risk assessment
            if result['risk_score'] > 50:
                result['allowed'] = False
                result['reason'] = "High risk score - operation blocked"
            elif result['risk_score'] > 30:
                result['quarantine_required'] = True
                result['warnings'].append("Medium risk - quarantine recommended")
            
            # Update statistics
            if not result['allowed']:
                self.security_stats['blocked_operations'] += 1
            if result['quarantine_required']:
                self.security_stats['quarantined_files'] += 1
            
            # Audit logging
            if self.audit_logger:
                await self.audit_logger.log_event("file_operation_validation", {
                    "operation_type": op_type.value,
                    "source_path": str(source_path),
                    "target_path": str(target_path) if target_path else None,
                    "allowed": result['allowed'],
                    "risk_score": result['risk_score'],
                    "reason": result['reason'],
                    "warnings": len(result['warnings']),
                    "quarantine_required": result['quarantine_required'],
                    "user_context": user_context
                })
            
            return result
            
        except Exception as e:
            result = {
                'allowed': False,
                'reason': f"Validation error: {str(e)}",
                'warnings': [],
                'sanitized_operation': operation,
                'risk_score': 100,
                'quarantine_required': False
            }
            self.security_stats['blocked_operations'] += 1
            return result
    
    async def _validate_path(self, file_path: Path, operation: FileOperation) -> Dict[str, Any]:
        """Validate file path for security issues"""
        result = {
            'valid': True,
            'reason': '',
            'warnings': [],
            'risk_score': 0
        }
        
        path_str = str(file_path)
        
        # Check for path traversal attacks
        if '..' in path_str:
            result['valid'] = False
            result['reason'] = "Path traversal attempt detected"
            result['risk_score'] += 50
            return result
        
        # Check for null bytes
        if '\x00' in path_str:
            result['valid'] = False
            result['reason'] = "Null byte in path"
            result['risk_score'] += 40
            return result
        
        # Check dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.match(pattern, path_str, re.IGNORECASE):
                result['valid'] = False
                result['reason'] = f"Dangerous file pattern: {pattern}"
                result['risk_score'] += 60
                return result
        
        # Check suspicious patterns
        for pattern in self.suspicious_patterns:
            if re.match(pattern, path_str, re.IGNORECASE):
                result['warnings'].append(f"Suspicious file pattern: {pattern}")
                result['risk_score'] += 20
        
        # Check file extension
        if file_path.suffix.lower() in self._get_blocked_extensions():
            result['valid'] = False
            result['reason'] = f"Blocked file extension: {file_path.suffix}"
            result['risk_score'] += 70
            return result
        
        # Check if path is outside project directory (unless in sandbox)
        try:
            resolved_path = file_path.resolve()
            project_resolved = self.project_path.resolve()
            sandbox_resolved = self.sandbox_root.resolve()
            
            if not (str(resolved_path).startswith(str(project_resolved)) or 
                    str(resolved_path).startswith(str(sandbox_resolved))):
                result['valid'] = False
                result['reason'] = "Path outside project directory"
                result['risk_score'] += 80
                return result
        except (OSError, ValueError) as e:
            result['valid'] = False
            result['reason'] = f"Invalid path: {str(e)}"
            result['risk_score'] += 30
            return result
        
        # Check path length
        if len(path_str) > 260:  # Windows MAX_PATH limitation
            result['warnings'].append("Very long file path")
            result['risk_score'] += 10
        
        return result
    
    async def _check_file_rules(self, file_path: Path, operation: FileOperation) -> Dict[str, Any]:
        """Check file operation against defined rules"""
        result = {
            'allowed': True,
            'reason': '',
            'warnings': [],
            'risk_score': 0
        }
        
        path_str = str(file_path)
        applicable_rules = []
        
        # Find applicable rules
        for rule in self.file_rules:
            if self._path_matches_pattern(path_str, rule.path_pattern):
                applicable_rules.append(rule)
        
        # Sort by priority (higher priority first)
        applicable_rules.sort(key=lambda r: r.priority, reverse=True)
        
        if not applicable_rules:
            # No specific rules - apply default based on security level
            if self.security_level == SecurityLevel.MAXIMUM:
                result['allowed'] = False
                result['reason'] = "No matching rule - maximum security mode"
                result['risk_score'] += 100
            elif self.security_level == SecurityLevel.HIGH:
                # Only allow read operations by default
                if operation not in [FileOperation.READ, FileOperation.LIST_DIRECTORY]:
                    result['allowed'] = False
                    result['reason'] = "No matching rule - high security mode"
                    result['risk_score'] += 50
        else:
            # Apply highest priority rule
            rule = applicable_rules[0]
            
            # Check if operation is allowed
            if operation not in rule.allowed_operations:
                result['allowed'] = False
                result['reason'] = f"Operation {operation.value} not allowed by rule: {rule.description}"
                result['risk_score'] += 40
            
            # Check access level
            if rule.access_level == AccessLevel.NONE:
                result['allowed'] = False
                result['reason'] = f"Access denied by rule: {rule.description}"
                result['risk_score'] += 60
            
            # Check file extension if specified
            if rule.allowed_extensions and file_path.suffix.lower() not in rule.allowed_extensions:
                result['allowed'] = False
                result['reason'] = f"File extension not allowed by rule: {rule.description}"
                result['risk_score'] += 30
        
        return result
    
    async def _validate_content(self, content: str, file_path: Path) -> Dict[str, Any]:
        """Validate file content for security issues"""
        result = {
            'valid': True,
            'reason': '',
            'warnings': [],
            'risk_score': 0,
            'sanitized_content': content,
            'quarantine_required': False
        }
        
        # Basic content validation
        if '\x00' in content:
            result['valid'] = False
            result['reason'] = "Null bytes in content"
            result['risk_score'] += 40
            return result
        
        # Malicious content patterns
        malicious_patterns = [
            (r'(?i)eval\s*\(', 'Code evaluation detected'),
            (r'(?i)exec\s*\(', 'Code execution detected'),
            (r'(?i)system\s*\(', 'System call detected'),
            (r'(?i)shell_exec', 'Shell execution detected'),
            (r'(?i)passthru\s*\(', 'Passthru execution detected'),
            (r'(?i)base64_decode\s*\(', 'Base64 decode detected'),
            (r'(?i)<script[^>]*>.*?</script>', 'Script tag detected'),
            (r'(?i)javascript:', 'JavaScript protocol detected'),
            (r'(?i)vbscript:', 'VBScript protocol detected'),
            (r'(?i)data:', 'Data URI detected'),
            (r'(?i)file://', 'File protocol detected'),
        ]
        
        for pattern, description in malicious_patterns:
            if re.search(pattern, content):
                result['warnings'].append(description)
                result['risk_score'] += 25
                if self.security_level in [SecurityLevel.HIGH, SecurityLevel.MAXIMUM]:
                    result['quarantine_required'] = True
        
        # Suspicious keywords
        suspicious_keywords = [
            'virus', 'malware', 'trojan', 'backdoor', 'keylogger', 'rootkit',
            'botnet', 'exploit', 'payload', 'shellcode', 'ransomware'
        ]
        
        content_lower = content.lower()
        found_keywords = [kw for kw in suspicious_keywords if kw in content_lower]
        if found_keywords:
            result['warnings'].append(f"Suspicious keywords: {found_keywords}")
            result['risk_score'] += len(found_keywords) * 10
            result['quarantine_required'] = True
        
        # Check for embedded executables (simplified)
        executable_signatures = [
            b'MZ',  # PE/DOS header
            b'\x7fELF',  # ELF header
            b'\xca\xfe\xba\xbe',  # Mach-O fat binary
            b'\xfe\xed\xfa\xce',  # Mach-O 32-bit
            b'\xfe\xed\xfa\xcf',  # Mach-O 64-bit
        ]
        
        content_bytes = content.encode('utf-8', errors='ignore')
        for sig in executable_signatures:
            if sig in content_bytes:
                result['valid'] = False
                result['reason'] = "Embedded executable content detected"
                result['risk_score'] += 100
                return result
        
        # File-specific content validation
        extension = file_path.suffix.lower()
        
        if extension in ['.py']:
            # Python-specific validation
            python_result = await self._validate_python_content(content)
            result['warnings'].extend(python_result['warnings'])
            result['risk_score'] += python_result['risk_score']
            if python_result['quarantine_required']:
                result['quarantine_required'] = True
        
        elif extension in ['.js', '.jsx']:
            # JavaScript-specific validation
            js_result = await self._validate_javascript_content(content)
            result['warnings'].extend(js_result['warnings'])
            result['risk_score'] += js_result['risk_score']
            if js_result['quarantine_required']:
                result['quarantine_required'] = True
        
        elif extension in ['.html', '.htm']:
            # HTML-specific validation
            html_result = await self._validate_html_content(content)
            result['warnings'].extend(html_result['warnings'])
            result['risk_score'] += html_result['risk_score']
            if html_result['quarantine_required']:
                result['quarantine_required'] = True
        
        # Content size validation
        if len(content_bytes) > 10 * 1024 * 1024:  # 10MB
            result['warnings'].append("Large file content")
            result['risk_score'] += 10
        
        # High risk assessment
        if result['risk_score'] > 75:
            result['valid'] = False
            result['reason'] = "High risk content detected"
        elif result['risk_score'] > 40:
            result['quarantine_required'] = True
        
        return result
    
    async def _validate_python_content(self, content: str) -> Dict[str, Any]:
        """Validate Python-specific content"""
        result = {
            'warnings': [],
            'risk_score': 0,
            'quarantine_required': False
        }
        
        # Dangerous Python patterns
        dangerous_patterns = [
            (r'(?i)\b__import__\s*\(', 'Dynamic import detected'),
            (r'(?i)\bgetattr\s*\(', 'Dynamic attribute access'),
            (r'(?i)\bsetattr\s*\(', 'Dynamic attribute setting'),
            (r'(?i)\bdelattr\s*\(', 'Dynamic attribute deletion'),
            (r'(?i)\bglobals\s*\(', 'Global namespace access'),
            (r'(?i)\blocals\s*\(', 'Local namespace access'),
            (r'(?i)\bvars\s*\(', 'Variable namespace access'),
            (r'(?i)\bdir\s*\(', 'Directory function usage'),
            (r'(?i)os\.system\s*\(', 'OS system call'),
            (r'(?i)subprocess\s*\.', 'Subprocess usage'),
            (r'(?i)commands\s*\.', 'Commands module usage'),
            (r'(?i)\bcompile\s*\(', 'Code compilation'),
        ]
        
        for pattern, description in dangerous_patterns:
            if re.search(pattern, content):
                result['warnings'].append(f"Python: {description}")
                result['risk_score'] += 15
                if self.security_level == SecurityLevel.MAXIMUM:
                    result['quarantine_required'] = True
        
        return result
    
    async def _validate_javascript_content(self, content: str) -> Dict[str, Any]:
        """Validate JavaScript-specific content"""
        result = {
            'warnings': [],
            'risk_score': 0,
            'quarantine_required': False
        }
        
        # Dangerous JavaScript patterns
        dangerous_patterns = [
            (r'(?i)\beval\s*\(', 'JavaScript eval usage'),
            (r'(?i)new\s+Function\s*\(', 'Function constructor'),
            (r'(?i)document\.write\s*\(', 'Document.write usage'),
            (r'(?i)innerHTML\s*=', 'innerHTML assignment'),
            (r'(?i)outerHTML\s*=', 'outerHTML assignment'),
            (r'(?i)setTimeout\s*\(.*?["\']', 'setTimeout with string'),
            (r'(?i)setInterval\s*\(.*?["\']', 'setInterval with string'),
            (r'(?i)window\s*\[\s*["\']', 'Dynamic window access'),
            (r'(?i)location\s*\.', 'Location object access'),
        ]
        
        for pattern, description in dangerous_patterns:
            if re.search(pattern, content):
                result['warnings'].append(f"JavaScript: {description}")
                result['risk_score'] += 20
                if self.security_level in [SecurityLevel.HIGH, SecurityLevel.MAXIMUM]:
                    result['quarantine_required'] = True
        
        return result
    
    async def _validate_html_content(self, content: str) -> Dict[str, Any]:
        """Validate HTML-specific content"""
        result = {
            'warnings': [],
            'risk_score': 0,
            'quarantine_required': False
        }
        
        # Dangerous HTML patterns
        dangerous_patterns = [
            (r'(?i)<script[^>]*>.*?</script>', 'Script tag found'),
            (r'(?i)<iframe[^>]*>.*?</iframe>', 'IFrame tag found'),
            (r'(?i)<object[^>]*>.*?</object>', 'Object tag found'),
            (r'(?i)<embed[^>]*>', 'Embed tag found'),
            (r'(?i)<link[^>]*rel\s*=\s*["\']stylesheet["\']', 'External stylesheet'),
            (r'(?i)javascript:', 'JavaScript protocol in HTML'),
            (r'(?i)vbscript:', 'VBScript protocol in HTML'),
            (r'(?i)on\w+\s*=', 'Event handler attribute'),
        ]
        
        for pattern, description in dangerous_patterns:
            if re.search(pattern, content):
                result['warnings'].append(f"HTML: {description}")
                result['risk_score'] += 25
                if self.security_level in [SecurityLevel.HIGH, SecurityLevel.MAXIMUM]:
                    result['quarantine_required'] = True
        
        return result
    
    def _path_matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if path matches a glob-like pattern"""
        import fnmatch
        return fnmatch.fnmatch(path, pattern)
    
    def _get_size_limit_for_path(self, file_path: Path) -> int:
        """Get size limit for specific file path"""
        # Find applicable rule with size limit
        path_str = str(file_path)
        for rule in sorted(self.file_rules, key=lambda r: r.priority, reverse=True):
            if self._path_matches_pattern(path_str, rule.path_pattern):
                return rule.max_file_size
        
        # Default size limit
        return self.max_file_size
    
    def _is_in_sandbox(self, file_path: Path) -> bool:
        """Check if path is within sandbox"""
        try:
            file_path.resolve().relative_to(self.sandbox_root.resolve())
            return True
        except ValueError:
            return False
    
    async def quarantine_file(self, file_path: Path, reason: str) -> Dict[str, Any]:
        """Move file to quarantine area"""
        try:
            if not file_path.exists():
                return {'success': False, 'error': 'File does not exist'}
            
            # Create quarantine subdirectory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            quarantine_subdir = self.quarantine_path / timestamp
            quarantine_subdir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique quarantine filename
            quarantine_file = quarantine_subdir / f"{file_path.name}.quarantined"
            counter = 1
            while quarantine_file.exists():
                quarantine_file = quarantine_subdir / f"{file_path.name}.quarantined.{counter}"
                counter += 1
            
            # Move file to quarantine
            shutil.move(str(file_path), str(quarantine_file))
            
            # Create metadata file
            metadata = {
                'original_path': str(file_path),
                'quarantine_path': str(quarantine_file),
                'reason': reason,
                'timestamp': datetime.now().isoformat(),
                'file_hash': self._calculate_file_hash(quarantine_file),
                'file_size': quarantine_file.stat().st_size
            }
            
            metadata_file = quarantine_subdir / f"{file_path.name}.metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.security_stats['quarantined_files'] += 1
            
            # Audit log
            if self.audit_logger:
                await self.audit_logger.log_event("file_quarantined", metadata)
            
            return {
                'success': True,
                'quarantine_path': str(quarantine_file),
                'metadata_path': str(metadata_file)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except:
            return "unknown"
    
    async def create_sandbox_file(self, filename: str, content: str, 
                                user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a file safely in the sandbox"""
        try:
            # Validate filename
            if not filename or '..' in filename or '/' in filename or '\\' in filename:
                return {'success': False, 'error': 'Invalid filename'}
            
            # Create file in sandbox
            sandbox_file = self.sandbox_root / filename
            
            # Validate the operation
            operation = {
                'type': 'create',
                'source_path': str(sandbox_file),
                'content': content,
                'user_context': user_context or {}
            }
            
            validation_result = await self.validate_file_operation(operation)
            if not validation_result['allowed']:
                return {'success': False, 'error': validation_result['reason']}
            
            # Write the file
            with open(sandbox_file, 'w', encoding='utf-8') as f:
                f.write(validation_result['sanitized_operation']['content'])
            
            # Track the file
            file_hash = self._calculate_file_hash(sandbox_file)
            self.tracked_files[str(sandbox_file)] = {
                'created': datetime.now().isoformat(),
                'hash': file_hash,
                'size': sandbox_file.stat().st_size,
                'user_context': user_context
            }
            
            return {
                'success': True,
                'file_path': str(sandbox_file),
                'warnings': validation_result['warnings']
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def scan_directory_for_threats(self, directory_path: Path) -> Dict[str, Any]:
        """Scan directory for potential security threats"""
        threats = []
        scanned_files = 0
        
        try:
            for file_path in directory_path.rglob('*'):
                if file_path.is_file():
                    scanned_files += 1
                    
                    # Basic file validation
                    validation_result = await self._validate_path(file_path, FileOperation.READ)
                    if not validation_result['valid']:
                        threats.append({
                            'file': str(file_path),
                            'threat_type': 'path_validation',
                            'description': validation_result['reason'],
                            'risk_score': validation_result['risk_score']
                        })
                        continue
                    
                    # Content scanning for text files
                    if file_path.suffix.lower() in {'.py', '.js', '.html', '.php', '.txt'}:
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read(1024 * 1024)  # Read first 1MB
                            
                            content_result = await self._validate_content(content, file_path)
                            if content_result['risk_score'] > 30:
                                threats.append({
                                    'file': str(file_path),
                                    'threat_type': 'content_analysis',
                                    'description': '; '.join(content_result['warnings']),
                                    'risk_score': content_result['risk_score'],
                                    'quarantine_recommended': content_result['quarantine_required']
                                })
                        except:
                            pass  # Skip files that can't be read
            
            return {
                'success': True,
                'scanned_files': scanned_files,
                'threats_found': len(threats),
                'threats': threats,
                'high_risk_files': [t for t in threats if t['risk_score'] > 50]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'scanned_files': scanned_files,
                'threats_found': len(threats),
                'threats': threats
            }
    
    def get_sandbox_status(self) -> Dict[str, Any]:
        """Get current sandbox status"""
        try:
            total_size = 0
            file_count = 0
            
            for file_path in self.sandbox_root.rglob('*'):
                if file_path.is_file():
                    file_count += 1
                    total_size += file_path.stat().st_size
            
            return {
                'sandbox_root': str(self.sandbox_root),
                'total_files': file_count,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'max_size_mb': round(self.sandbox_config.max_total_size / (1024 * 1024), 2),
                'max_files': self.sandbox_config.max_files,
                'usage_percentage': (total_size / self.sandbox_config.max_total_size) * 100,
                'quarantine_path': str(self.quarantine_path),
                'security_level': self.security_level.value
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get file system guard statistics"""
        return self.security_stats.copy()
    
    def reset_statistics(self):
        """Reset security statistics"""
        self.security_stats = {
            'total_operations': 0,
            'blocked_operations': 0,
            'quarantined_files': 0,
            'suspicious_files': 0,
            'by_operation_type': {}
        }