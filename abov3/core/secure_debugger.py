"""
ABOV3 Genesis - Secure Enterprise Debugger
Enhanced debugger with comprehensive enterprise-grade security integration
"""

import asyncio
import logging
import sys
import os
from typing import Any, Dict, List, Optional, Callable, Union
from pathlib import Path
from datetime import datetime
from functools import wraps

# Import original debugger components
from .debugger import PerformanceProfiler, CodeDebugger, SystemDebugger, ResourceMonitor

# Import security components
try:
    from ..security.secure_debug_integration import (
        SecureDebugIntegration, 
        SecureDebugConfig, 
        DebugSecurityLevel,
        get_secure_debug_integration
    )
    from ..security.debug_audit_logger import AuditEventType, AuditSeverity
    from ..security.data_classifier import DataSensitivityLevel
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False
    SecureDebugIntegration = None
    SecureDebugConfig = None
    DebugSecurityLevel = None


class SecureDebugSession:
    """Secure debug session wrapper"""
    
    def __init__(
        self,
        session_id: str,
        session_token: str,
        user_id: str,
        security_integration: Optional[SecureDebugIntegration] = None
    ):
        self.session_id = session_id
        self.session_token = session_token
        self.user_id = user_id
        self.security_integration = security_integration
        self.is_secure = security_integration is not None
        
        # Original debugger components
        self.performance_profiler = PerformanceProfiler()
        self.code_debugger = CodeDebugger()
        self.system_debugger = SystemDebugger()
        
        # Setup logging
        self.logger = logging.getLogger('abov3.core.secure_debugger')
    
    async def execute_code(
        self,
        code: str,
        language: str = 'python',
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute code with security controls"""
        if not self.is_secure:
            # Fallback to basic execution without security
            return await self._execute_code_basic(code, language, context)
        
        try:
            # Execute through security integration
            result = await self.security_integration.execute_debug_code(
                session_id=self.session_id,
                session_token=self.session_token,
                code=code,
                language=language,
                execution_context=context
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Secure code execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'error_code': 'SECURE_EXECUTION_ERROR'
            }
    
    async def _execute_code_basic(
        self,
        code: str,
        language: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Basic code execution without security (fallback)"""
        try:
            if language.lower() == 'python':
                # Simple Python execution
                import subprocess
                import tempfile
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(code)
                    temp_file = f.name
                
                try:
                    result = subprocess.run(
                        [sys.executable, temp_file],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    return {
                        'success': result.returncode == 0,
                        'exit_code': result.returncode,
                        'stdout': result.stdout,
                        'stderr': result.stderr,
                        'execution_time': 0.0,
                        'security_warnings': ['Security integration not available'],
                        'security_violations': []
                    }
                    
                finally:
                    os.unlink(temp_file)
            
            else:
                return {
                    'success': False,
                    'error': f'Language {language} not supported in basic mode',
                    'error_code': 'UNSUPPORTED_LANGUAGE'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_code': 'BASIC_EXECUTION_ERROR'
            }
    
    async def access_debug_data(
        self,
        data_id: str,
        access_reason: str = "debug_operation"
    ) -> Dict[str, Any]:
        """Access debug data with security controls"""
        if not self.is_secure:
            return {
                'success': False,
                'error': 'Security integration required for data access',
                'error_code': 'SECURITY_REQUIRED'
            }
        
        try:
            result = await self.security_integration.access_debug_data(
                session_id=self.session_id,
                session_token=self.session_token,
                data_id=data_id,
                access_reason=access_reason
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Secure data access failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'error_code': 'SECURE_ACCESS_ERROR'
            }
    
    def profile_function(self, func: Callable) -> Callable:
        """Profile function with security awareness"""
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if self.is_secure:
                # Log profiling operation
                # Note: In a full implementation, this would go through security integration
                pass
            
            # Use original profiler
            return await self.performance_profiler.profile_async_function(func)(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if self.is_secure:
                # Log profiling operation
                # Note: In a full implementation, this would go through security integration
                pass
            
            # Use original profiler
            return self.performance_profiler.profile_function(func)(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    def inspect_object(self, obj: Any, depth: int = 2) -> Dict[str, Any]:
        """Inspect object with security controls"""
        try:
            # Use original inspector
            inspection = self.code_debugger.inspect_object(obj, depth)
            
            if self.is_secure:
                # Add security metadata
                inspection['security'] = {
                    'session_id': self.session_id,
                    'inspection_time': datetime.now().isoformat(),
                    'secure_session': True
                }
            
            return inspection
            
        except Exception as e:
            self.logger.error(f"Object inspection failed: {e}")
            return {'error': str(e)}
    
    def analyze_exception(self, exception: Exception) -> Dict[str, Any]:
        """Analyze exception with security awareness"""
        try:
            # Use original analyzer
            analysis = self.code_debugger.analyze_stack_trace(exception)
            
            if self.is_secure:
                # Check if exception contains sensitive information
                # This is a simplified check - full implementation would use data classifier
                sensitive_keywords = ['password', 'key', 'secret', 'token', 'credential']
                error_str = str(exception).lower()
                
                if any(keyword in error_str for keyword in sensitive_keywords):
                    analysis['security_warning'] = 'Exception may contain sensitive information'
                    analysis['sanitized_message'] = '[REDACTED: Potentially sensitive exception]'
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Exception analysis failed: {e}")
            return {'error': str(e)}
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report with security context"""
        try:
            report = self.performance_profiler.get_performance_report()
            
            if self.is_secure:
                report['security_context'] = {
                    'session_id': self.session_id,
                    'user_id': self.user_id,
                    'secure_session': True,
                    'report_generated': datetime.now().isoformat()
                }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Performance report generation failed: {e}")
            return {'error': str(e)}
    
    def get_system_diagnostics(self) -> Dict[str, Any]:
        """Get system diagnostics with security filtering"""
        try:
            diagnostics = self.system_debugger._collect_system_info()
            
            if self.is_secure:
                # Filter sensitive system information
                sensitive_keys = ['environment_variables', 'python_executable']
                for key in sensitive_keys:
                    if key in diagnostics:
                        diagnostics[key] = '[REDACTED: Sensitive system information]'
                
                # Add security metadata
                diagnostics['security_filtered'] = True
                diagnostics['collection_time'] = datetime.now().isoformat()
            
            return diagnostics
            
        except Exception as e:
            self.logger.error(f"System diagnostics failed: {e}")
            return {'error': str(e)}
    
    async def terminate(self, reason: str = "User request") -> bool:
        """Terminate debug session securely"""
        if not self.is_secure:
            return True  # Nothing to terminate in basic mode
        
        try:
            success = await self.security_integration.terminate_debug_session(
                session_id=self.session_id,
                session_token=self.session_token,
                reason=reason
            )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Session termination failed: {e}")
            return False


class SecureEnterpriseDebugger:
    """
    Enterprise-grade secure debugger with comprehensive security integration
    Provides all debugging capabilities with enterprise security controls
    """
    
    def __init__(
        self,
        project_path: Path,
        security_level: DebugSecurityLevel = DebugSecurityLevel.PRODUCTION if SECURITY_AVAILABLE else None,
        enable_security: bool = True
    ):
        self.project_path = project_path
        self.security_level = security_level
        self.enable_security = enable_security and SECURITY_AVAILABLE
        
        # Security integration
        self.security_integration: Optional[SecureDebugIntegration] = None
        self.security_config: Optional[SecureDebugConfig] = None
        
        # Active sessions
        self.active_sessions: Dict[str, SecureDebugSession] = {}
        
        # Fallback components for non-secure mode
        self.fallback_profiler = PerformanceProfiler()
        self.fallback_debugger = CodeDebugger()
        self.fallback_system = SystemDebugger()
        
        # Setup logging
        self.logger = logging.getLogger('abov3.core.secure_debugger')
        
        # Initialize
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the secure debugger"""
        if self._initialized:
            return True
        
        try:
            if self.enable_security and SECURITY_AVAILABLE:
                # Initialize security components
                self.security_config = SecureDebugConfig(
                    security_level=self.security_level,
                    enable_mfa=True,
                    require_approval_for_advanced=True,
                    encrypt_debug_data=True,
                    comprehensive_logging=True
                )
                
                self.security_integration = await get_secure_debug_integration(
                    project_path=self.project_path,
                    config=self.security_config
                )
                
                self.logger.info(f"Secure debugger initialized with {self.security_level.value} security level")
            else:
                self.logger.warning("Security integration not available - running in basic mode")
            
            self._initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize secure debugger: {e}")
            return False
    
    async def create_debug_session(
        self,
        user_id: str,
        role_id: str = "developer",
        client_ip: str = "127.0.0.1",
        user_agent: str = "ABOV3-Debugger",
        mfa_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a secure debug session"""
        try:
            if not self._initialized:
                await self.initialize()
            
            if self.enable_security and self.security_integration:
                # Create secure session
                session_result = await self.security_integration.create_secure_debug_session(
                    user_id=user_id,
                    role_id=role_id,
                    client_ip=client_ip,
                    user_agent=user_agent,
                    mfa_token=mfa_token
                )
                
                if session_result['success']:
                    # Create secure session wrapper
                    session = SecureDebugSession(
                        session_id=session_result['session_id'],
                        session_token=session_result['session_token'],
                        user_id=user_id,
                        security_integration=self.security_integration
                    )
                    
                    self.active_sessions[session_result['session_id']] = session
                    
                    return {
                        'success': True,
                        'session_id': session_result['session_id'],
                        'session_token': session_result['session_token'],
                        'security_enabled': True,
                        'permission_level': session_result['permission_level'],
                        'expires_at': session_result['expires_at']
                    }
                else:
                    return session_result
            
            else:
                # Create basic session
                import uuid
                session_id = str(uuid.uuid4())
                session_token = "basic_session_token"
                
                session = SecureDebugSession(
                    session_id=session_id,
                    session_token=session_token,
                    user_id=user_id,
                    security_integration=None
                )
                
                self.active_sessions[session_id] = session
                
                return {
                    'success': True,
                    'session_id': session_id,
                    'session_token': session_token,
                    'security_enabled': False,
                    'permission_level': 'full',
                    'expires_at': None
                }
                
        except Exception as e:
            self.logger.error(f"Failed to create debug session: {e}")
            return {
                'success': False,
                'error': str(e),
                'error_code': 'SESSION_CREATION_ERROR'
            }
    
    def get_session(self, session_id: str) -> Optional[SecureDebugSession]:
        """Get debug session by ID"""
        return self.active_sessions.get(session_id)
    
    async def execute_debug_code(
        self,
        session_id: str,
        code: str,
        language: str = 'python',
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute debug code in session"""
        session = self.get_session(session_id)
        if not session:
            return {
                'success': False,
                'error': 'Session not found',
                'error_code': 'SESSION_NOT_FOUND'
            }
        
        return await session.execute_code(code, language, context)
    
    async def profile_code(
        self,
        session_id: str,
        code: str,
        language: str = 'python'
    ) -> Dict[str, Any]:
        """Profile code execution"""
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        try:
            # Execute and profile
            with session.performance_profiler.profile("user_code"):
                result = await session.execute_code(code, language)
            
            # Get profiling results
            profile_stats = session.performance_profiler.get_profile_stats()
            performance_report = session.performance_profiler.get_performance_report()
            
            return {
                'execution_result': result,
                'profile_stats': profile_stats,
                'performance_report': performance_report
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def debug_exception(
        self,
        session_id: str,
        exception: Exception
    ) -> Dict[str, Any]:
        """Debug an exception"""
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        return session.analyze_exception(exception)
    
    def inspect_object(
        self,
        session_id: str,
        obj: Any,
        depth: int = 2
    ) -> Dict[str, Any]:
        """Inspect an object"""
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        return session.inspect_object(obj, depth)
    
    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get session status"""
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        status = {
            'session_id': session.session_id,
            'user_id': session.user_id,
            'is_secure': session.is_secure,
            'active': True
        }
        
        if session.is_secure and self.security_integration:
            try:
                security_status = await self.security_integration.get_security_status()
                status['security_status'] = security_status
            except Exception as e:
                status['security_error'] = str(e)
        
        return status
    
    async def terminate_session(
        self,
        session_id: str,
        reason: str = "User request"
    ) -> bool:
        """Terminate a debug session"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        try:
            success = await session.terminate(reason)
            
            # Remove from active sessions
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to terminate session {session_id}: {e}")
            return False
    
    async def get_security_status(self) -> Dict[str, Any]:
        """Get overall security status"""
        status = {
            'security_enabled': self.enable_security,
            'security_available': SECURITY_AVAILABLE,
            'active_sessions': len(self.active_sessions),
            'initialized': self._initialized
        }
        
        if self.enable_security and self.security_integration:
            try:
                security_status = await self.security_integration.get_security_status()
                status.update(security_status)
            except Exception as e:
                status['security_error'] = str(e)
        
        return status
    
    async def generate_compliance_report(
        self,
        report_type: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """Generate compliance report"""
        if not (self.enable_security and self.security_integration):
            return None
        
        try:
            return await self.security_integration.generate_compliance_report(
                report_type, start_date, end_date
            )
        except Exception as e:
            self.logger.error(f"Failed to generate compliance report: {e}")
            return None
    
    def list_active_sessions(self) -> List[Dict[str, Any]]:
        """List all active debug sessions"""
        sessions = []
        
        for session_id, session in self.active_sessions.items():
            sessions.append({
                'session_id': session_id,
                'user_id': session.user_id,
                'is_secure': session.is_secure,
                'active': True
            })
        
        return sessions
    
    async def shutdown(self):
        """Shutdown the secure debugger"""
        try:
            # Terminate all sessions
            session_ids = list(self.active_sessions.keys())
            for session_id in session_ids:
                await self.terminate_session(session_id, "System shutdown")
            
            # Shutdown security integration
            if self.security_integration:
                await self.security_integration.shutdown()
            
            self._initialized = False
            self.logger.info("Secure debugger shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")


# Global instance for backward compatibility
_secure_debugger: Optional[SecureEnterpriseDebugger] = None

async def get_secure_debugger(
    project_path: Path,
    security_level: Optional[DebugSecurityLevel] = None,
    enable_security: bool = True
) -> SecureEnterpriseDebugger:
    """Get or create global secure debugger instance"""
    global _secure_debugger
    
    if _secure_debugger is None:
        security_level = security_level or (DebugSecurityLevel.PRODUCTION if SECURITY_AVAILABLE else None)
        _secure_debugger = SecureEnterpriseDebugger(
            project_path=project_path,
            security_level=security_level,
            enable_security=enable_security
        )
        await _secure_debugger.initialize()
    
    return _secure_debugger

# Convenience functions for backward compatibility
async def create_debug_session(
    project_path: Path,
    user_id: str,
    role_id: str = "developer",
    **kwargs
) -> Dict[str, Any]:
    """Create a debug session"""
    debugger = await get_secure_debugger(project_path)
    return await debugger.create_debug_session(user_id, role_id, **kwargs)

async def execute_debug_code(
    project_path: Path,
    session_id: str,
    code: str,
    language: str = 'python',
    **kwargs
) -> Dict[str, Any]:
    """Execute debug code"""
    debugger = await get_secure_debugger(project_path)
    return await debugger.execute_debug_code(session_id, code, language, **kwargs)

# Export security availability flag
SECURE_DEBUG_AVAILABLE = SECURITY_AVAILABLE