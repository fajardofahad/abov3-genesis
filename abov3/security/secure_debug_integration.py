"""
ABOV3 Genesis - Secure Debug Integration
Enterprise-grade security integration for the debug module
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import threading

from .secure_debug_session import SecureDebugSessionManager, DebugPermissionLevel, DebugRole
from .secure_debug_storage import SecureDebugStorage, DataClassification, StorageEncryptionLevel
from .debug_audit_logger import DebugAuditLogger, AuditEventType, AuditSeverity
from .sandbox_executor import SecureSandboxExecutor, SandboxType, SecurityProfile, ExecutionMode
from .data_classifier import SensitiveDataClassifier, DataSensitivityLevel, RedactionMethod
from .crypto_manager import CryptographyManager
from .auth_manager import AuthenticationManager, AuthorizationManager
from .core import SecurityCore, SecurityConfig, SecurityStatus


class DebugSecurityLevel(Enum):
    """Debug security levels"""
    DEVELOPMENT = "development"      # Relaxed for development
    TESTING = "testing"             # Standard for testing
    STAGING = "staging"             # High security for staging
    PRODUCTION = "production"       # Maximum security for production


@dataclass
class SecureDebugConfig:
    """Configuration for secure debug operations"""
    # Security levels
    security_level: DebugSecurityLevel = DebugSecurityLevel.PRODUCTION
    enable_mfa: bool = True
    require_approval_for_advanced: bool = True
    
    # Session management
    max_session_duration: int = 3600  # 1 hour
    max_concurrent_sessions: int = 3
    session_idle_timeout: int = 1800  # 30 minutes
    
    # Data protection
    encrypt_debug_data: bool = True
    data_retention_days: int = 30
    enable_data_classification: bool = True
    auto_redact_sensitive: bool = True
    
    # Sandbox execution
    default_sandbox_type: SandboxType = SandboxType.CONTAINER
    default_security_profile: SecurityProfile = SecurityProfile.HIGH
    allow_code_execution: bool = True
    
    # Audit and compliance
    comprehensive_logging: bool = True
    real_time_monitoring: bool = True
    compliance_reporting: bool = True
    
    # Network and communication
    encrypt_communications: bool = True
    verify_client_certificates: bool = False
    allowed_ip_ranges: Optional[List[str]] = None


class SecureDebugIntegration:
    """
    Main integration class that combines all security components
    with the existing debug module for enterprise-grade secure debugging
    """
    
    def __init__(
        self,
        project_path: Path,
        config: Optional[SecureDebugConfig] = None,
        security_config: Optional[SecurityConfig] = None
    ):
        self.project_path = project_path
        self.config = config or SecureDebugConfig()
        self.security_config = security_config or SecurityConfig()
        
        # Security directory
        self.security_dir = project_path / '.abov3' / 'security'
        self.debug_security_dir = self.security_dir / 'debug'
        self.debug_security_dir.mkdir(parents=True, exist_ok=True)
        
        # Core security components
        self.security_core: Optional[SecurityCore] = None
        self.crypto_manager: Optional[CryptographyManager] = None
        self.auth_manager: Optional[AuthenticationManager] = None
        self.authz_manager: Optional[AuthorizationManager] = None
        
        # Debug-specific security components
        self.audit_logger: Optional[DebugAuditLogger] = None
        self.session_manager: Optional[SecureDebugSessionManager] = None
        self.storage_manager: Optional[SecureDebugStorage] = None
        self.sandbox_executor: Optional[SecureSandboxExecutor] = None
        self.data_classifier: Optional[SensitiveDataClassifier] = None
        
        # Security monitoring
        self.security_monitors: Dict[str, Any] = {}
        self.active_threats: Dict[str, Dict[str, Any]] = {}
        self.security_metrics: Dict[str, Any] = {}
        
        # Thread safety
        self._lock = threading.Lock()
        self._initialized = False
        
        # Setup logging
        self.logger = logging.getLogger('abov3.security.debug_integration')
        
        # Background tasks
        self._monitor_task: Optional[asyncio.Task] = None
        self._threat_task: Optional[asyncio.Task] = None
        
    async def initialize(self) -> bool:
        """
        Initialize the secure debug integration system
        
        Returns:
            bool indicating successful initialization
        """
        with self._lock:
            if self._initialized:
                return True
        
        try:
            self.logger.info("Initializing Secure Debug Integration...")
            
            # Initialize core security
            await self._initialize_core_security()
            
            # Initialize debug-specific security
            await self._initialize_debug_security()
            
            # Configure security levels
            await self._configure_security_levels()
            
            # Start monitoring
            await self._start_security_monitoring()
            
            with self._lock:
                self._initialized = True
            
            self.logger.info("Secure Debug Integration initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Secure Debug Integration: {e}")
            return False
    
    async def _initialize_core_security(self):
        """Initialize core security components"""
        # Initialize security core
        self.security_core = SecurityCore(self.project_path, self.security_config)
        await self.security_core.initialize()
        
        # Get references to core components
        self.crypto_manager = self.security_core.crypto_manager
        self.auth_manager = self.security_core.auth_manager
        self.authz_manager = self.security_core.authz_manager
    
    async def _initialize_debug_security(self):
        """Initialize debug-specific security components"""
        # Initialize audit logger
        self.audit_logger = DebugAuditLogger(
            audit_dir=self.debug_security_dir / 'audit',
            crypto_manager=self.crypto_manager
        )
        
        # Initialize data classifier
        self.data_classifier = SensitiveDataClassifier(
            audit_logger=self.audit_logger,
            crypto_manager=self.crypto_manager
        )
        
        # Initialize storage manager
        self.storage_manager = SecureDebugStorage(
            storage_dir=self.debug_security_dir / 'storage',
            crypto_manager=self.crypto_manager,
            audit_logger=self.audit_logger
        )
        
        # Initialize sandbox executor
        self.sandbox_executor = SecureSandboxExecutor(
            audit_logger=self.audit_logger,
            crypto_manager=self.crypto_manager,
            temp_dir=self.debug_security_dir / 'sandbox'
        )
        
        # Initialize session manager
        self.session_manager = SecureDebugSessionManager(
            crypto_manager=self.crypto_manager,
            audit_logger=self.audit_logger,
            auth_manager=self.auth_manager,
            authz_manager=self.authz_manager,
            security_dir=self.debug_security_dir
        )
    
    async def _configure_security_levels(self):
        """Configure security based on the specified security level"""
        if self.config.security_level == DebugSecurityLevel.DEVELOPMENT:
            # Relaxed security for development
            self.config.require_approval_for_advanced = False
            self.config.enable_mfa = False
            self.config.encrypt_debug_data = False
            self.config.auto_redact_sensitive = False
            
        elif self.config.security_level == DebugSecurityLevel.TESTING:
            # Standard security for testing
            self.config.require_approval_for_advanced = False
            self.config.enable_mfa = False
            self.config.encrypt_debug_data = True
            self.config.auto_redact_sensitive = True
            
        elif self.config.security_level == DebugSecurityLevel.STAGING:
            # High security for staging
            self.config.require_approval_for_advanced = True
            self.config.enable_mfa = True
            self.config.encrypt_debug_data = True
            self.config.auto_redact_sensitive = True
            
        elif self.config.security_level == DebugSecurityLevel.PRODUCTION:
            # Maximum security for production
            self.config.require_approval_for_advanced = True
            self.config.enable_mfa = True
            self.config.encrypt_debug_data = True
            self.config.auto_redact_sensitive = True
            self.config.verify_client_certificates = True
    
    async def create_secure_debug_session(
        self,
        user_id: str,
        role_id: str,
        client_ip: str,
        user_agent: str,
        mfa_token: Optional[str] = None,
        client_certificate: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a secure debug session with comprehensive security checks
        
        Args:
            user_id: User identifier
            role_id: Requested debug role
            client_ip: Client IP address
            user_agent: Client user agent
            mfa_token: Multi-factor authentication token
            client_certificate: Client certificate for verification
            
        Returns:
            Dict containing session creation result
        """
        try:
            # Comprehensive security validation
            validation_result = await self._validate_session_request(
                user_id, role_id, client_ip, user_agent, client_certificate
            )
            
            if not validation_result['valid']:
                return validation_result
            
            # Create session through session manager
            session_result = await self.session_manager.create_session(
                user_id=user_id,
                role_id=role_id,
                client_ip=client_ip,
                user_agent=user_agent,
                mfa_token=mfa_token,
                additional_context=validation_result.get('context', {})
            )
            
            if session_result['success']:
                # Initialize session security monitoring
                await self._initialize_session_monitoring(session_result['session_id'])
                
                # Log successful session creation
                await self.audit_logger.log_event(
                    event_type=AuditEventType.SESSION_CREATED,
                    user_id=user_id,
                    session_id=session_result['session_id'],
                    client_ip=client_ip,
                    user_agent=user_agent,
                    context={
                        'role_id': role_id,
                        'security_level': self.config.security_level.value,
                        'mfa_verified': mfa_token is not None
                    }
                )
            
            return session_result
            
        except Exception as e:
            self.logger.error(f"Failed to create secure debug session: {e}")
            return {
                'success': False,
                'error': 'Internal error during session creation',
                'error_code': 'INTERNAL_ERROR'
            }
    
    async def execute_debug_code(
        self,
        session_id: str,
        session_token: str,
        code: str,
        language: str,
        execution_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute debug code with comprehensive security controls
        
        Args:
            session_id: Debug session identifier
            session_token: Session authentication token
            code: Code to execute
            language: Programming language
            execution_context: Optional execution context
            
        Returns:
            Dict containing execution results
        """
        try:
            # Validate session
            session_validation = await self.session_manager.validate_session(session_id, session_token)
            if not session_validation['valid']:
                return {
                    'success': False,
                    'error': session_validation['error'],
                    'error_code': session_validation['error_code']
                }
            
            session_info = session_validation['session']
            user_id = session_info['user_id']
            
            # Check execution permission
            permission_result = await self.session_manager.check_operation_permission(
                session_id, 'code_execution', 'sandbox_environment'
            )
            
            if not permission_result['allowed']:
                return {
                    'success': False,
                    'error': permission_result['reason'],
                    'error_code': 'PERMISSION_DENIED'
                }
            
            # Classify and potentially redact sensitive data in code
            if self.config.enable_data_classification:
                classification_result = await self.data_classifier.classify_data(
                    data=code,
                    context=execution_context,
                    user_id=user_id,
                    session_id=session_id
                )
                
                if self.config.auto_redact_sensitive and classification_result.detections:
                    # Use redacted code for execution if sensitive data detected
                    code = classification_result.redacted_data
                    
                    await self.audit_logger.log_event(
                        event_type=AuditEventType.DATA_MODIFIED,
                        user_id=user_id,
                        session_id=session_id,
                        context={
                            'operation': 'auto_redaction',
                            'detections_count': len(classification_result.detections),
                            'sensitivity_level': classification_result.overall_sensitivity.value
                        }
                    )
            
            # Execute code in sandbox
            execution_result = await self.sandbox_executor.execute_code(
                code=code,
                language=language,
                user_id=user_id,
                session_id=session_id,
                sandbox_type=self.config.default_sandbox_type,
                security_profile=self.config.default_security_profile,
                execution_mode=ExecutionMode.INTERACTIVE
            )
            
            # Store execution results securely
            if execution_result.success and self.config.encrypt_debug_data:
                await self._store_execution_result(session_id, user_id, execution_result)
            
            # Classify and redact output if needed
            processed_result = await self._process_execution_output(
                execution_result, user_id, session_id
            )
            
            return {
                'success': True,
                'execution_id': execution_result.execution_id,
                'exit_code': processed_result.exit_code,
                'stdout': processed_result.stdout,
                'stderr': processed_result.stderr,
                'execution_time': processed_result.execution_time,
                'security_warnings': processed_result.warnings,
                'security_violations': processed_result.security_violations
            }
            
        except Exception as e:
            self.logger.error(f"Debug code execution failed: {e}")
            return {
                'success': False,
                'error': 'Code execution failed',
                'error_code': 'EXECUTION_ERROR'
            }
    
    async def access_debug_data(
        self,
        session_id: str,
        session_token: str,
        data_id: str,
        access_reason: str = "debug_operation"
    ) -> Dict[str, Any]:
        """
        Access stored debug data with security controls
        
        Args:
            session_id: Debug session identifier
            session_token: Session authentication token
            data_id: Data identifier to access
            access_reason: Reason for data access
            
        Returns:
            Dict containing data or error
        """
        try:
            # Validate session
            session_validation = await self.session_manager.validate_session(session_id, session_token)
            if not session_validation['valid']:
                return {
                    'success': False,
                    'error': session_validation['error'],
                    'error_code': session_validation['error_code']
                }
            
            session_info = session_validation['session']
            user_id = session_info['user_id']
            
            # Check data access permission
            permission_result = await self.session_manager.check_operation_permission(
                session_id, 'data_access', data_id
            )
            
            if not permission_result['allowed']:
                return {
                    'success': False,
                    'error': permission_result['reason'],
                    'error_code': 'PERMISSION_DENIED'
                }
            
            # Retrieve data from secure storage
            data_result = await self.storage_manager.retrieve_debug_data(
                data_id=data_id,
                session_id=session_id,
                user_id=user_id,
                access_reason=access_reason
            )
            
            if not data_result:
                return {
                    'success': False,
                    'error': 'Data not found or access denied',
                    'error_code': 'DATA_NOT_FOUND'
                }
            
            # Record data access
            await self.session_manager.record_data_access(
                session_id=session_id,
                data_size=len(str(data_result['data'])),
                data_type=data_result['metadata']['data_type'],
                sensitive=data_result['metadata']['classification'] in ['restricted', 'top_secret']
            )
            
            return {
                'success': True,
                'data': data_result['data'],
                'metadata': data_result['metadata']
            }
            
        except Exception as e:
            self.logger.error(f"Debug data access failed: {e}")
            return {
                'success': False,
                'error': 'Data access failed',
                'error_code': 'ACCESS_ERROR'
            }
    
    async def terminate_debug_session(
        self,
        session_id: str,
        session_token: str,
        reason: str = "User request"
    ) -> bool:
        """Terminate a debug session securely"""
        try:
            # Validate session
            session_validation = await self.session_manager.validate_session(session_id, session_token)
            if not session_validation['valid']:
                return False
            
            # Terminate any running executions
            active_executions = self.sandbox_executor.get_active_executions()
            for execution in active_executions:
                if execution['session_id'] == session_id:
                    await self.sandbox_executor.terminate_execution(
                        execution['execution_id'], 
                        "Session termination"
                    )
            
            # Terminate session
            success = await self.session_manager.terminate_session(session_id, reason)
            
            # Cleanup session monitoring
            if session_id in self.security_monitors:
                del self.security_monitors[session_id]
            
            return success
            
        except Exception as e:
            self.logger.error(f"Session termination failed: {e}")
            return False
    
    async def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        try:
            status = {
                'overall_security_level': self.config.security_level.value,
                'timestamp': datetime.now().isoformat(),
                'initialized': self._initialized
            }
            
            if self._initialized:
                # Core security status
                if self.security_core:
                    status['core_security'] = await self.security_core.get_security_status()
                
                # Session management status
                if self.session_manager:
                    status['session_management'] = self.session_manager.get_metrics()
                
                # Storage management status
                if self.storage_manager:
                    status['storage_management'] = await self.storage_manager.get_storage_statistics()
                
                # Sandbox execution status
                if self.sandbox_executor:
                    status['sandbox_execution'] = self.sandbox_executor.get_metrics()
                
                # Data classification status
                if self.data_classifier:
                    status['data_classification'] = self.data_classifier.get_statistics()
                
                # Audit logging status
                if self.audit_logger:
                    status['audit_logging'] = self.audit_logger.get_audit_statistics()
                
                # Threat monitoring
                status['threat_monitoring'] = {
                    'active_threats': len(self.active_threats),
                    'monitored_sessions': len(self.security_monitors)
                }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get security status: {e}")
            return {
                'error': str(e),
                'initialized': False
            }
    
    async def generate_compliance_report(
        self,
        report_type: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """Generate compliance report"""
        if not self.audit_logger:
            return None
        
        try:
            from .debug_audit_logger import ComplianceStandard
            
            # Map report type to compliance standard
            standard_map = {
                'soc2': ComplianceStandard.SOC2,
                'gdpr': ComplianceStandard.GDPR,
                'hipaa': ComplianceStandard.HIPAA,
                'nist': ComplianceStandard.NIST
            }
            
            standard = standard_map.get(report_type.lower())
            if not standard:
                return None
            
            # Set default date range if not provided
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                start_date = end_date - timedelta(days=30)  # Last 30 days
            
            # Generate report
            report = await self.audit_logger.generate_compliance_report(
                standard=standard,
                start_date=start_date,
                end_date=end_date
            )
            
            # Add additional security metrics
            report['debug_security_metrics'] = {
                'total_debug_sessions': self.session_manager.get_metrics()['total_sessions'] if self.session_manager else 0,
                'code_executions': self.sandbox_executor.get_metrics()['total_executions'] if self.sandbox_executor else 0,
                'sensitive_data_detections': self.data_classifier.get_statistics()['sensitive_data_found'] if self.data_classifier else 0,
                'security_violations': self.session_manager.get_metrics()['security_violations'] if self.session_manager else 0
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate compliance report: {e}")
            return None
    
    async def _validate_session_request(
        self,
        user_id: str,
        role_id: str,
        client_ip: str,
        user_agent: str,
        client_certificate: Optional[str]
    ) -> Dict[str, Any]:
        """Validate session creation request"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'context': {}
        }
        
        # IP address validation
        if self.config.allowed_ip_ranges:
            ip_allowed = False
            for ip_range in self.config.allowed_ip_ranges:
                # Simplified IP range check - in production, use proper CIDR matching
                if client_ip.startswith(ip_range.split('/')[0].rsplit('.', 1)[0]):
                    ip_allowed = True
                    break
            
            if not ip_allowed:
                validation_result['valid'] = False
                validation_result['errors'].append(f"IP address {client_ip} not in allowed ranges")
        
        # Client certificate validation
        if self.config.verify_client_certificates and not client_certificate:
            validation_result['valid'] = False
            validation_result['errors'].append("Client certificate required but not provided")
        
        # Rate limiting check (simplified)
        # In production, implement proper rate limiting
        validation_result['context']['validation_timestamp'] = datetime.now().isoformat()
        
        return validation_result
    
    async def _initialize_session_monitoring(self, session_id: str):
        """Initialize security monitoring for a session"""
        self.security_monitors[session_id] = {
            'created_at': datetime.now(),
            'operations_count': 0,
            'data_access_count': 0,
            'security_events': [],
            'last_activity': datetime.now()
        }
    
    async def _store_execution_result(
        self,
        session_id: str,
        user_id: str,
        execution_result
    ):
        """Store execution result securely"""
        try:
            # Determine data classification
            classification = DataClassification.INTERNAL
            if execution_result.security_violations:
                classification = DataClassification.CONFIDENTIAL
            
            # Store execution data
            execution_data = {
                'execution_id': execution_result.execution_id,
                'exit_code': execution_result.exit_code,
                'stdout': execution_result.stdout,
                'stderr': execution_result.stderr,
                'execution_time': execution_result.execution_time,
                'memory_used': execution_result.memory_used,
                'cpu_usage': execution_result.cpu_usage,
                'warnings': execution_result.warnings,
                'security_violations': execution_result.security_violations
            }
            
            await self.storage_manager.store_debug_data(
                session_id=session_id,
                user_id=user_id,
                data_type='execution_result',
                data=execution_data,
                classification=classification,
                encryption_level=StorageEncryptionLevel.HIGH,
                tags=['execution', 'result']
            )
            
        except Exception as e:
            self.logger.error(f"Failed to store execution result: {e}")
    
    async def _process_execution_output(self, execution_result, user_id: str, session_id: str):
        """Process execution output for sensitive data"""
        if not self.config.enable_data_classification:
            return execution_result
        
        try:
            # Classify stdout and stderr
            stdout_classification = await self.data_classifier.classify_data(
                execution_result.stdout, user_id=user_id, session_id=session_id
            )
            stderr_classification = await self.data_classifier.classify_data(
                execution_result.stderr, user_id=user_id, session_id=session_id
            )
            
            # Apply redaction if needed
            if self.config.auto_redact_sensitive:
                if stdout_classification.detections:
                    execution_result.stdout = stdout_classification.redacted_data
                if stderr_classification.detections:
                    execution_result.stderr = stderr_classification.redacted_data
            
            return execution_result
            
        except Exception as e:
            self.logger.error(f"Failed to process execution output: {e}")
            return execution_result
    
    async def _start_security_monitoring(self):
        """Start background security monitoring tasks"""
        async def monitor_sessions():
            while True:
                try:
                    await self._monitor_session_security()
                    await asyncio.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    self.logger.error(f"Session monitoring error: {e}")
                    await asyncio.sleep(60)
        
        async def detect_threats():
            while True:
                try:
                    await self._detect_security_threats()
                    await asyncio.sleep(60)  # Check every minute
                except Exception as e:
                    self.logger.error(f"Threat detection error: {e}")
                    await asyncio.sleep(120)
        
        self._monitor_task = asyncio.create_task(monitor_sessions())
        self._threat_task = asyncio.create_task(detect_threats())
    
    async def _monitor_session_security(self):
        """Monitor active sessions for security issues"""
        current_time = datetime.now()
        
        for session_id, monitor_info in list(self.security_monitors.items()):
            try:
                # Check for idle sessions
                idle_time = current_time - monitor_info['last_activity']
                if idle_time.seconds > self.config.session_idle_timeout:
                    await self.audit_logger.log_security_event(
                        security_event_type="suspicious_activity",
                        user_id=None,
                        session_id=session_id,
                        threat_level="low",
                        additional_context={'idle_time_seconds': idle_time.seconds}
                    )
                
                # Check for excessive operations
                if monitor_info['operations_count'] > 1000:  # Threshold
                    await self.audit_logger.log_security_event(
                        security_event_type="suspicious_activity",
                        user_id=None,
                        session_id=session_id,
                        threat_level="medium",
                        additional_context={
                            'excessive_operations': monitor_info['operations_count']
                        }
                    )
                
            except Exception as e:
                self.logger.error(f"Error monitoring session {session_id}: {e}")
    
    async def _detect_security_threats(self):
        """Detect and respond to security threats"""
        try:
            # Analyze audit logs for threat patterns
            # This is a simplified implementation
            if self.audit_logger:
                recent_events = await self.audit_logger.query_events(
                    start_time=datetime.now() - timedelta(hours=1),
                    limit=1000
                )
                
                # Look for suspicious patterns
                failed_auths = [e for e in recent_events if e.get('event_type') == 'auth_failure']
                if len(failed_auths) > 10:  # More than 10 failed attempts in 1 hour
                    threat_id = f"auth_brute_force_{datetime.now().isoformat()}"
                    self.active_threats[threat_id] = {
                        'type': 'brute_force_attack',
                        'severity': 'high',
                        'detected_at': datetime.now(),
                        'events': failed_auths[:10]  # Store first 10 events
                    }
                    
                    await self.audit_logger.log_security_event(
                        security_event_type="threat_detected",
                        user_id=None,
                        session_id=None,
                        threat_level="high",
                        threat_indicators=["brute_force", "authentication"],
                        additional_context={
                            'threat_id': threat_id,
                            'failed_attempts': len(failed_auths)
                        }
                    )
        
        except Exception as e:
            self.logger.error(f"Threat detection error: {e}")
    
    async def shutdown(self):
        """Shutdown the secure debug integration system"""
        try:
            self.logger.info("Shutting down Secure Debug Integration...")
            
            # Cancel monitoring tasks
            if self._monitor_task:
                self._monitor_task.cancel()
            if self._threat_task:
                self._threat_task.cancel()
            
            # Shutdown components in reverse order
            if self.session_manager:
                await self.session_manager.shutdown()
            
            if self.sandbox_executor:
                await self.sandbox_executor.shutdown()
            
            if self.storage_manager:
                await self.storage_manager.shutdown()
            
            if self.audit_logger:
                await self.audit_logger.shutdown()
            
            if self.security_core:
                await self.security_core.cleanup()
            
            with self._lock:
                self._initialized = False
            
            self.logger.info("Secure Debug Integration shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")


# Global instance for easy access
_secure_debug_integration: Optional[SecureDebugIntegration] = None

async def get_secure_debug_integration(
    project_path: Path,
    config: Optional[SecureDebugConfig] = None,
    security_config: Optional[SecurityConfig] = None
) -> SecureDebugIntegration:
    """Get or create global secure debug integration instance"""
    global _secure_debug_integration
    
    if _secure_debug_integration is None:
        _secure_debug_integration = SecureDebugIntegration(project_path, config, security_config)
        await _secure_debug_integration.initialize()
    
    return _secure_debug_integration

async def shutdown_secure_debug_integration():
    """Shutdown global secure debug integration instance"""
    global _secure_debug_integration
    
    if _secure_debug_integration:
        await _secure_debug_integration.shutdown()
        _secure_debug_integration = None