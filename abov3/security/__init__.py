"""
ABOV3 Genesis - Comprehensive Cybersecurity Framework
Enterprise-grade security suite for AI coding platform

This module provides comprehensive security measures including:
- Input validation and sanitization
- Prompt injection protection
- File system sandboxing
- Authentication and authorization
- Data encryption and protection
- Security audit logging
- Vulnerability scanning
- Rate limiting and DDoS protection
- Secure AI model interactions
"""

from .core import SecurityCore
from .input_validator import InputValidator, SecurityLevel
from .prompt_guard import PromptInjectionGuard
from .file_guard import FileSystemGuard
from .auth_manager import AuthenticationManager, AuthorizationManager
from .crypto_manager import CryptographyManager
from .audit_logger import SecurityAuditLogger
from .vulnerability_scanner import VulnerabilityScanner
from .rate_limiter import RateLimiter, DDosProtection
from .secure_ai import SecureAIManager
from .security_monitor import SecurityMonitor
from .threat_detector import ThreatDetector

__all__ = [
    'SecurityCore',
    'InputValidator', 
    'SecurityLevel',
    'PromptInjectionGuard',
    'FileSystemGuard',
    'AuthenticationManager',
    'AuthorizationManager', 
    'CryptographyManager',
    'SecurityAuditLogger',
    'VulnerabilityScanner',
    'RateLimiter',
    'DDosProtection',
    'SecureAIManager',
    'SecurityMonitor',
    'ThreatDetector'
]

# Security configuration constants
SECURITY_VERSION = "1.0.0"
MINIMUM_ENCRYPTION_KEY_SIZE = 256
DEFAULT_RATE_LIMIT = 1000  # requests per minute
AUDIT_LOG_RETENTION_DAYS = 90
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_FILE_EXTENSIONS = {
    '.py', '.js', '.html', '.css', '.json', '.yaml', '.yml', '.md', '.txt',
    '.tsx', '.jsx', '.ts', '.java', '.cpp', '.c', '.h', '.rs', '.go'
}