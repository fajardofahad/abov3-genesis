"""
ABOV3 Genesis - Security Core
Central security management system for the ABOV3 Genesis platform
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import hashlib
import secrets
import json

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


class SecurityStatus(Enum):
    """Security system status levels"""
    SECURE = "secure"
    WARNING = "warning" 
    CRITICAL = "critical"
    COMPROMISED = "compromised"


@dataclass
class SecurityConfig:
    """Security configuration settings"""
    enable_input_validation: bool = True
    enable_prompt_injection_guard: bool = True
    enable_file_system_guard: bool = True
    enable_authentication: bool = True
    enable_encryption: bool = True
    enable_audit_logging: bool = True
    enable_vulnerability_scanning: bool = True
    enable_rate_limiting: bool = True
    enable_ddos_protection: bool = True
    enable_ai_security: bool = True
    enable_threat_detection: bool = True
    
    # Security levels
    default_security_level: SecurityLevel = SecurityLevel.HIGH
    ai_security_level: SecurityLevel = SecurityLevel.MAXIMUM
    file_security_level: SecurityLevel = SecurityLevel.HIGH
    
    # Limits and thresholds
    max_request_rate: int = 1000  # per minute
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    session_timeout: int = 3600  # 1 hour
    max_login_attempts: int = 5
    
    # Audit settings
    audit_retention_days: int = 90
    enable_realtime_monitoring: bool = True
    
    # AI-specific settings
    max_prompt_length: int = 10000
    max_response_length: int = 50000
    enable_content_filtering: bool = True


class SecurityCore:
    """
    Central security management system for ABOV3 Genesis
    Coordinates all security components and provides unified security interface
    """
    
    def __init__(self, project_path: Path, config: Optional[SecurityConfig] = None):
        self.project_path = project_path
        self.config = config or SecurityConfig()
        
        # Security components
        self.input_validator: Optional[InputValidator] = None
        self.prompt_guard: Optional[PromptInjectionGuard] = None
        self.file_guard: Optional[FileSystemGuard] = None
        self.auth_manager: Optional[AuthenticationManager] = None
        self.authz_manager: Optional[AuthorizationManager] = None
        self.crypto_manager: Optional[CryptographyManager] = None
        self.audit_logger: Optional[SecurityAuditLogger] = None
        self.vulnerability_scanner: Optional[VulnerabilityScanner] = None
        self.rate_limiter: Optional[RateLimiter] = None
        self.ddos_protection: Optional[DDosProtection] = None
        self.secure_ai: Optional[SecureAIManager] = None
        self.security_monitor: Optional[SecurityMonitor] = None
        self.threat_detector: Optional[ThreatDetector] = None
        
        # Security state
        self.security_status = SecurityStatus.SECURE
        self.security_events: List[Dict[str, Any]] = []
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.blocked_ips: set = set()
        self.security_metrics: Dict[str, Any] = {}
        
        # Initialize security directory
        self.security_dir = project_path / '.abov3' / 'security'
        self.security_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_security_logging()
        
    async def initialize(self) -> bool:
        """Initialize all security components"""
        try:
            self.logger.info("Initializing ABOV3 Genesis Security Core")
            
            # Initialize components based on configuration
            if self.config.enable_audit_logging:
                self.audit_logger = SecurityAuditLogger(self.security_dir)
                await self.audit_logger.initialize()
                
            if self.config.enable_encryption:
                self.crypto_manager = CryptographyManager(self.security_dir)
                await self.crypto_manager.initialize()
                
            if self.config.enable_input_validation:
                self.input_validator = InputValidator(
                    security_level=self.config.default_security_level,
                    crypto_manager=self.crypto_manager,
                    audit_logger=self.audit_logger
                )
                
            if self.config.enable_prompt_injection_guard:
                self.prompt_guard = PromptInjectionGuard(
                    security_level=self.config.ai_security_level,
                    audit_logger=self.audit_logger
                )
                
            if self.config.enable_file_system_guard:
                self.file_guard = FileSystemGuard(
                    project_path=self.project_path,
                    security_level=self.config.file_security_level,
                    max_file_size=self.config.max_file_size,
                    audit_logger=self.audit_logger
                )
                
            if self.config.enable_authentication:
                self.auth_manager = AuthenticationManager(
                    security_dir=self.security_dir,
                    crypto_manager=self.crypto_manager,
                    audit_logger=self.audit_logger,
                    max_attempts=self.config.max_login_attempts,
                    session_timeout=self.config.session_timeout
                )
                
                self.authz_manager = AuthorizationManager(
                    security_dir=self.security_dir,
                    audit_logger=self.audit_logger
                )
                
            if self.config.enable_rate_limiting:
                self.rate_limiter = RateLimiter(
                    max_requests=self.config.max_request_rate,
                    audit_logger=self.audit_logger
                )
                
            if self.config.enable_ddos_protection:
                self.ddos_protection = DDosProtection(
                    audit_logger=self.audit_logger
                )
                
            if self.config.enable_ai_security:
                self.secure_ai = SecureAIManager(
                    max_prompt_length=self.config.max_prompt_length,
                    max_response_length=self.config.max_response_length,
                    enable_content_filtering=self.config.enable_content_filtering,
                    prompt_guard=self.prompt_guard,
                    audit_logger=self.audit_logger
                )
                
            if self.config.enable_vulnerability_scanning:
                self.vulnerability_scanner = VulnerabilityScanner(
                    project_path=self.project_path,
                    audit_logger=self.audit_logger
                )
                
            if self.config.enable_threat_detection:
                self.threat_detector = ThreatDetector(
                    audit_logger=self.audit_logger
                )
                
            if self.config.enable_realtime_monitoring:
                self.security_monitor = SecurityMonitor(
                    security_core=self,
                    audit_logger=self.audit_logger
                )
                await self.security_monitor.start()
                
            # Log successful initialization
            await self._audit_security_event("security_core_initialized", {
                "components_enabled": self._get_enabled_components(),
                "security_level": self.config.default_security_level.value,
                "timestamp": datetime.now().isoformat()
            })
            
            self.logger.info("Security Core initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Security Core initialization failed: {e}")
            self.security_status = SecurityStatus.CRITICAL
            return False
    
    async def validate_request(self, request_data: Dict[str, Any], user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Comprehensive request validation through all security layers
        
        Args:
            request_data: The request to validate
            user_context: Optional user context for authorization
            
        Returns:
            Dict containing validation results and processed request
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'processed_request': request_data.copy(),
            'security_flags': [],
            'risk_score': 0
        }
        
        try:
            # Rate limiting check
            if self.rate_limiter:
                client_id = user_context.get('client_id', 'unknown') if user_context else 'unknown'
                if not await self.rate_limiter.check_rate_limit(client_id):
                    validation_result['valid'] = False
                    validation_result['errors'].append("Rate limit exceeded")
                    validation_result['risk_score'] += 30
                    
            # DDoS protection check
            if self.ddos_protection:
                client_ip = user_context.get('client_ip', 'unknown') if user_context else 'unknown'
                if await self.ddos_protection.is_attack_detected(client_ip):
                    validation_result['valid'] = False
                    validation_result['errors'].append("Potential DDoS attack detected")
                    validation_result['risk_score'] += 50
                    
            # Input validation
            if self.input_validator and 'input' in request_data:
                # Extract the actual input string from the request data
                input_str = request_data.get('input', '')
                input_type_str = request_data.get('input_type', 'text')
                
                # Map string input type to InputType enum if needed
                from abov3.security.input_validator import InputType
                input_type_map = {
                    'text': InputType.TEXT,
                    'code': InputType.CODE,
                    'command': InputType.COMMAND,
                    'python': InputType.PYTHON
                }
                input_type = input_type_map.get(input_type_str, InputType.TEXT)
                
                input_result = await self.input_validator.validate_input(input_str, input_type)
                if not input_result['valid']:
                    validation_result['valid'] = False
                    validation_result['errors'].extend(input_result['errors'])
                validation_result['warnings'].extend(input_result.get('warnings', []))
                
                # Update the processed request with sanitized input
                validation_result['processed_request']['input'] = input_result.get('sanitized_input', input_str)
                validation_result['risk_score'] += input_result.get('risk_score', 0)
                
            # AI prompt injection guard
            if self.prompt_guard and 'prompt' in request_data:
                prompt_result = await self.prompt_guard.analyze_prompt(request_data['prompt'])
                if prompt_result['is_injection']:
                    validation_result['valid'] = False
                    validation_result['errors'].append("Prompt injection attempt detected")
                    validation_result['security_flags'].append("prompt_injection")
                    validation_result['risk_score'] += 40
                elif prompt_result['risk_score'] > 50:
                    validation_result['warnings'].append("Suspicious prompt content detected")
                    validation_result['risk_score'] += prompt_result['risk_score']
                    
            # Authentication check
            if self.auth_manager and user_context:
                auth_token = user_context.get('auth_token')
                if auth_token:
                    auth_result = await self.auth_manager.validate_token(auth_token)
                    if not auth_result['valid']:
                        validation_result['valid'] = False
                        validation_result['errors'].append("Invalid authentication token")
                        validation_result['risk_score'] += 25
                else:
                    validation_result['warnings'].append("No authentication provided")
                    validation_result['risk_score'] += 10
                    
            # Authorization check
            if self.authz_manager and user_context and validation_result['valid']:
                action = request_data.get('action', 'unknown')
                resource = request_data.get('resource', 'unknown')
                if not await self.authz_manager.check_permission(user_context.get('user_id'), action, resource):
                    validation_result['valid'] = False
                    validation_result['errors'].append("Insufficient permissions")
                    validation_result['risk_score'] += 20
                    
            # File system guard
            if self.file_guard and 'file_operations' in request_data:
                for file_op in request_data['file_operations']:
                    file_result = await self.file_guard.validate_file_operation(file_op)
                    if not file_result['allowed']:
                        validation_result['valid'] = False
                        validation_result['errors'].append(f"File operation blocked: {file_result['reason']}")
                        validation_result['risk_score'] += 15
                        
            # Threat detection
            if self.threat_detector:
                threat_result = await self.threat_detector.analyze_request(request_data, user_context)
                if threat_result['threat_detected']:
                    validation_result['valid'] = False
                    validation_result['errors'].append("Threat detected in request")
                    validation_result['security_flags'].extend(threat_result['threat_types'])
                    validation_result['risk_score'] += threat_result['threat_score']
                    
            # Final risk assessment
            if validation_result['risk_score'] > 75:
                validation_result['valid'] = False
                validation_result['errors'].append("High risk score - request blocked")
                
            # Audit the validation
            await self._audit_security_event("request_validation", {
                "valid": validation_result['valid'],
                "risk_score": validation_result['risk_score'],
                "errors": len(validation_result['errors']),
                "warnings": len(validation_result['warnings']),
                "security_flags": validation_result['security_flags'],
                "user_context": user_context.get('user_id', 'anonymous') if user_context else 'anonymous'
            })
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Request validation error: {e}")
            return {
                'valid': False,
                'errors': [f"Validation system error: {str(e)}"],
                'warnings': [],
                'processed_request': request_data,
                'security_flags': ['validation_error'],
                'risk_score': 100
            }
    
    async def secure_ai_interaction(self, prompt: str, model: str, options: Optional[Dict[str, Any]] = None, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Secure AI model interaction with comprehensive protection
        
        Args:
            prompt: The prompt to send to AI
            model: The AI model to use
            options: Optional model parameters
            user_context: User context for security checks
            
        Returns:
            Dict containing secure AI response or error
        """
        if not self.secure_ai:
            return {
                'success': False,
                'error': 'AI security manager not initialized'
            }
            
        return await self.secure_ai.secure_interaction(prompt, model, options, user_context)
    
    async def scan_for_vulnerabilities(self, scan_type: str = "full") -> Dict[str, Any]:
        """
        Run vulnerability scan on the project
        
        Args:
            scan_type: Type of scan to perform (full, quick, dependencies)
            
        Returns:
            Dict containing scan results
        """
        if not self.vulnerability_scanner:
            return {
                'success': False,
                'error': 'Vulnerability scanner not initialized'
            }
            
        return await self.vulnerability_scanner.scan_project(scan_type)
    
    async def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        status = {
            'overall_status': self.security_status.value,
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'metrics': self.security_metrics.copy(),
            'recent_events': self.security_events[-10:] if self.security_events else [],
            'active_sessions': len(self.active_sessions),
            'blocked_ips': len(self.blocked_ips)
        }
        
        # Component status
        components = {
            'input_validator': self.input_validator is not None,
            'prompt_guard': self.prompt_guard is not None,
            'file_guard': self.file_guard is not None,
            'auth_manager': self.auth_manager is not None,
            'crypto_manager': self.crypto_manager is not None,
            'audit_logger': self.audit_logger is not None,
            'vulnerability_scanner': self.vulnerability_scanner is not None,
            'rate_limiter': self.rate_limiter is not None,
            'ddos_protection': self.ddos_protection is not None,
            'secure_ai': self.secure_ai is not None,
            'security_monitor': self.security_monitor is not None,
            'threat_detector': self.threat_detector is not None
        }
        
        status['components'] = components
        
        # Get component-specific status
        if self.security_monitor:
            status['monitoring'] = await self.security_monitor.get_status()
            
        if self.rate_limiter:
            status['rate_limiting'] = self.rate_limiter.get_statistics()
            
        return status
    
    async def emergency_shutdown(self, reason: str) -> bool:
        """
        Emergency security shutdown
        
        Args:
            reason: Reason for emergency shutdown
            
        Returns:
            bool indicating if shutdown was successful
        """
        try:
            self.logger.critical(f"Emergency security shutdown initiated: {reason}")
            self.security_status = SecurityStatus.COMPROMISED
            
            # Stop all active components
            if self.security_monitor:
                await self.security_monitor.stop()
                
            # Block all new requests
            if self.rate_limiter:
                await self.rate_limiter.emergency_block_all()
                
            # Audit the emergency shutdown
            await self._audit_security_event("emergency_shutdown", {
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
                "active_sessions": len(self.active_sessions),
                "components_active": self._get_enabled_components()
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Emergency shutdown failed: {e}")
            return False
    
    def _setup_security_logging(self):
        """Setup security-specific logging"""
        self.logger = logging.getLogger('abov3.security.core')
        self.logger.setLevel(logging.INFO)
        
        # Create security log file handler
        log_file = self.security_dir / 'security.log'
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def _get_enabled_components(self) -> List[str]:
        """Get list of enabled security components"""
        enabled = []
        if self.input_validator:
            enabled.append('input_validator')
        if self.prompt_guard:
            enabled.append('prompt_guard')
        if self.file_guard:
            enabled.append('file_guard')
        if self.auth_manager:
            enabled.append('auth_manager')
        if self.crypto_manager:
            enabled.append('crypto_manager')
        if self.audit_logger:
            enabled.append('audit_logger')
        if self.vulnerability_scanner:
            enabled.append('vulnerability_scanner')
        if self.rate_limiter:
            enabled.append('rate_limiter')
        if self.ddos_protection:
            enabled.append('ddos_protection')
        if self.secure_ai:
            enabled.append('secure_ai')
        if self.security_monitor:
            enabled.append('security_monitor')
        if self.threat_detector:
            enabled.append('threat_detector')
        return enabled
    
    async def _audit_security_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log security event to audit system"""
        if self.audit_logger:
            await self.audit_logger.log_event(event_type, event_data)
        
        # Also add to internal event list
        self.security_events.append({
            'type': event_type,
            'data': event_data,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only recent events in memory
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-500:]
    
    async def cleanup(self):
        """Cleanup security resources"""
        try:
            if self.security_monitor:
                await self.security_monitor.stop()
                
            if self.audit_logger:
                await self.audit_logger.cleanup()
                
            self.logger.info("Security Core cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Security cleanup error: {e}")