"""
ABOV3 Genesis - Security Integration
Integration layer for ABOV3 Genesis security framework
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from .core import SecurityCore, SecurityConfig, SecurityLevel
from ..core.assistant import Assistant


class SecurityIntegration:
    """
    Security Integration Layer for ABOV3 Genesis
    Provides seamless integration of security features with the main application
    """
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.security_core: Optional[SecurityCore] = None
        self.logger = logging.getLogger('abov3.security.integration')
        
    async def initialize_security(self, config: Optional[SecurityConfig] = None) -> bool:
        """Initialize security framework"""
        try:
            # Create security configuration with enterprise defaults
            if config is None:
                config = SecurityConfig(
                    enable_input_validation=True,
                    enable_prompt_injection_guard=True,
                    enable_file_system_guard=True,
                    enable_authentication=True,
                    enable_encryption=True,
                    enable_audit_logging=True,
                    enable_vulnerability_scanning=True,
                    enable_rate_limiting=True,
                    enable_ddos_protection=True,
                    enable_ai_security=True,
                    enable_threat_detection=True,
                    default_security_level=SecurityLevel.HIGH,
                    ai_security_level=SecurityLevel.MAXIMUM,
                    file_security_level=SecurityLevel.HIGH,
                    max_request_rate=1000,
                    max_file_size=100 * 1024 * 1024,  # 100MB
                    session_timeout=3600,
                    max_login_attempts=5,
                    audit_retention_days=90,
                    enable_realtime_monitoring=True,
                    max_prompt_length=10000,
                    max_response_length=50000,
                    enable_content_filtering=True
                )
            
            # Initialize security core
            self.security_core = SecurityCore(self.project_path, config)
            
            # Initialize security systems
            success = await self.security_core.initialize()
            
            if success:
                self.logger.info("ABOV3 Genesis Security Framework initialized successfully")
                
                # Log security configuration
                await self.security_core._audit_security_event("security_framework_initialized", {
                    "project_path": str(self.project_path),
                    "security_level": config.default_security_level.value,
                    "ai_security_level": config.ai_security_level.value,
                    "components_enabled": self.security_core._get_enabled_components(),
                    "enterprise_ready": True,
                    "version": "1.0.0"
                })
                
                return True
            else:
                self.logger.error("Failed to initialize security framework")
                return False
                
        except Exception as e:
            self.logger.error(f"Security initialization error: {e}")
            return False
    
    async def secure_user_input(self, user_input: str, input_type: str = "text", 
                              user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Secure validation of user input"""
        if not self.security_core:
            return {'valid': False, 'error': 'Security not initialized'}
        
        # Create request object for validation
        request_data = {
            'type': 'user_input',
            'input': user_input,
            'input_type': input_type,
            'timestamp': asyncio.get_event_loop().time()
        }
        
        # Validate through security core
        return await self.security_core.validate_request(request_data, user_context)
    
    async def secure_ai_interaction(self, prompt: str, model: str, 
                                  options: Optional[Dict[str, Any]] = None,
                                  user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Secure AI model interaction"""
        if not self.security_core:
            return {'success': False, 'error': 'Security not initialized'}
        
        return await self.security_core.secure_ai_interaction(prompt, model, options, user_context)
    
    async def secure_file_operation(self, operation_type: str, file_path: str, 
                                  content: str = None, target_path: str = None,
                                  user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Secure file operation"""
        if not self.security_core:
            return {'allowed': False, 'reason': 'Security not initialized'}
        
        operation = {
            'type': operation_type,
            'source_path': file_path,
            'target_path': target_path,
            'content': content,
            'user_context': user_context or {}
        }
        
        return await self.security_core.validate_file_operation(operation)
    
    async def authenticate_user(self, username: str, password: str, 
                              ip_address: str = None, user_agent: str = None,
                              mfa_token: str = None) -> Dict[str, Any]:
        """Authenticate user"""
        if not self.security_core or not self.security_core.auth_manager:
            return {'success': False, 'error': 'Authentication not available'}
        
        return await self.security_core.auth_manager.authenticate(
            username, password, ip_address, user_agent, mfa_token
        )
    
    async def validate_session(self, token: str) -> Dict[str, Any]:
        """Validate user session"""
        if not self.security_core or not self.security_core.auth_manager:
            return {'valid': False, 'error': 'Authentication not available'}
        
        return await self.security_core.auth_manager.validate_token(token)
    
    async def scan_for_vulnerabilities(self, scan_type: str = "full") -> Dict[str, Any]:
        """Run security scan"""
        if not self.security_core:
            return {'success': False, 'error': 'Security not initialized'}
        
        return await self.security_core.scan_for_vulnerabilities(scan_type)
    
    async def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        if not self.security_core:
            return {'status': 'not_initialized'}
        
        return await self.security_core.get_security_status()
    
    async def generate_security_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate security report"""
        if not self.security_core or not self.security_core.audit_logger:
            return {'error': 'Security audit not available'}
        
        return await self.security_core.audit_logger.generate_security_report(days)
    
    def is_security_enabled(self) -> bool:
        """Check if security is enabled and initialized"""
        return self.security_core is not None
    
    async def emergency_shutdown(self, reason: str) -> bool:
        """Emergency security shutdown"""
        if not self.security_core:
            return False
        
        return await self.security_core.emergency_shutdown(reason)
    
    async def cleanup(self):
        """Cleanup security resources"""
        if self.security_core:
            await self.security_core.cleanup()


# Helper function for easy integration
async def initialize_abov3_security(project_path: Path, 
                                  security_level: SecurityLevel = SecurityLevel.HIGH) -> SecurityIntegration:
    """
    Easy initialization of ABOV3 Genesis Security Framework
    
    Args:
        project_path: Path to the ABOV3 project
        security_level: Desired security level
        
    Returns:
        SecurityIntegration instance
    """
    
    # Create security configuration
    config = SecurityConfig(
        enable_input_validation=True,
        enable_prompt_injection_guard=True,
        enable_file_system_guard=True,
        enable_authentication=True,
        enable_encryption=True,
        enable_audit_logging=True,
        enable_vulnerability_scanning=True,
        enable_rate_limiting=True,
        enable_ddos_protection=True,
        enable_ai_security=True,
        enable_threat_detection=True,
        default_security_level=security_level,
        ai_security_level=SecurityLevel.MAXIMUM if security_level == SecurityLevel.MAXIMUM else SecurityLevel.HIGH,
        file_security_level=security_level,
        enable_realtime_monitoring=True
    )
    
    # Initialize security integration
    security_integration = SecurityIntegration(project_path)
    
    # Initialize security framework
    success = await security_integration.initialize_security(config)
    
    if not success:
        raise Exception("Failed to initialize ABOV3 Genesis Security Framework")
    
    return security_integration


# Decorator for securing functions
def secure_function(security_integration: SecurityIntegration, 
                   input_validation: bool = True,
                   rate_limiting: bool = True,
                   auth_required: bool = False):
    """
    Decorator to add security to functions
    
    Args:
        security_integration: SecurityIntegration instance
        input_validation: Enable input validation
        rate_limiting: Enable rate limiting
        auth_required: Require authentication
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract user context if available
            user_context = kwargs.get('user_context')
            
            # Authentication check
            if auth_required and user_context:
                if not user_context.get('authenticated'):
                    return {'success': False, 'error': 'Authentication required'}
            
            # Rate limiting check
            if rate_limiting and user_context:
                client_id = user_context.get('client_id', 'unknown')
                if security_integration.security_core and security_integration.security_core.rate_limiter:
                    allowed = await security_integration.security_core.rate_limiter.check_rate_limit(client_id)
                    if not allowed:
                        return {'success': False, 'error': 'Rate limit exceeded'}
            
            # Input validation
            if input_validation and args:
                for arg in args:
                    if isinstance(arg, str):
                        validation_result = await security_integration.secure_user_input(arg, 'text', user_context)
                        if not validation_result.get('valid', True):
                            return {'success': False, 'error': 'Input validation failed'}
            
            # Call original function
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator