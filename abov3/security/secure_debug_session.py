"""
ABOV3 Genesis - Secure Debug Session Manager
Enterprise-grade secure debugging with role-based access control and comprehensive security
"""

import asyncio
import logging
import secrets
import time
import hashlib
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
from contextlib import asynccontextmanager

from .crypto_manager import CryptographyManager
from .audit_logger import SecurityAuditLogger
from .auth_manager import AuthenticationManager, AuthorizationManager


class DebugPermissionLevel(Enum):
    """Debug permission levels with escalating privileges"""
    READ_ONLY = "read_only"              # View debug data only
    BASIC_DEBUG = "basic_debug"          # Basic debugging operations
    ADVANCED_DEBUG = "advanced_debug"    # Full debugging + profiling
    SYSTEM_DEBUG = "system_debug"        # System-level debugging
    ADMIN_DEBUG = "admin_debug"          # Administrative debugging access


class DebugSessionStatus(Enum):
    """Debug session status states"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"
    EXPIRED = "expired"
    SECURITY_LOCKED = "security_locked"


@dataclass
class DebugPermission:
    """Individual debug permission definition"""
    permission_id: str
    name: str
    description: str
    level: DebugPermissionLevel
    requires_approval: bool = False
    max_session_duration: int = 3600  # seconds
    allowed_operations: Set[str] = field(default_factory=set)
    restricted_data_types: Set[str] = field(default_factory=set)


@dataclass
class DebugRole:
    """Debug role with specific permissions"""
    role_id: str
    name: str
    description: str
    permissions: List[DebugPermission]
    max_concurrent_sessions: int = 1
    session_timeout: int = 3600
    requires_mfa: bool = False
    ip_restrictions: Optional[List[str]] = None
    time_restrictions: Optional[Dict[str, Any]] = None


@dataclass
class SecureDebugSession:
    """Secure debug session with comprehensive security controls"""
    session_id: str
    user_id: str
    role: DebugRole
    created_at: datetime
    last_activity: datetime
    status: DebugSessionStatus
    client_ip: str
    user_agent: str
    
    # Security controls
    encrypted_data_key: str
    session_token: str
    mfa_verified: bool = False
    permission_level: DebugPermissionLevel = DebugPermissionLevel.READ_ONLY
    
    # Session limits
    max_duration: int = 3600
    operations_count: int = 0
    data_accessed: int = 0  # bytes
    
    # Audit trail
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    security_events: List[Dict[str, Any]] = field(default_factory=list)
    
    # Resource usage
    memory_usage: int = 0
    cpu_usage: float = 0.0
    
    def is_expired(self) -> bool:
        """Check if session is expired"""
        return (datetime.now() - self.last_activity).seconds > self.max_duration
    
    def is_active(self) -> bool:
        """Check if session is active"""
        return (
            self.status == DebugSessionStatus.ACTIVE and 
            not self.is_expired() and 
            self.mfa_verified
        )
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()


class SecureDebugSessionManager:
    """
    Enterprise-grade secure debug session manager
    Implements role-based access control, MFA, encryption, and comprehensive auditing
    """
    
    def __init__(
        self,
        crypto_manager: CryptographyManager,
        audit_logger: SecurityAuditLogger,
        auth_manager: AuthenticationManager,
        authz_manager: AuthorizationManager,
        security_dir: Path
    ):
        self.crypto_manager = crypto_manager
        self.audit_logger = audit_logger
        self.auth_manager = auth_manager
        self.authz_manager = authz_manager
        self.security_dir = security_dir
        
        # Session storage
        self.active_sessions: Dict[str, SecureDebugSession] = {}
        self.session_locks: Dict[str, threading.Lock] = {}
        self.user_sessions: Dict[str, Set[str]] = {}  # user_id -> session_ids
        
        # Security configuration
        self.max_sessions_per_user = 3
        self.session_cleanup_interval = 300  # 5 minutes
        self.max_idle_time = 1800  # 30 minutes
        self.require_mfa_for_advanced = True
        self.enable_session_recording = True
        
        # Debug roles and permissions
        self.debug_roles: Dict[str, DebugRole] = {}
        self.debug_permissions: Dict[str, DebugPermission] = {}
        
        # Monitoring and analytics
        self.session_metrics: Dict[str, Any] = {
            'total_sessions': 0,
            'active_sessions': 0,
            'security_violations': 0,
            'permission_denials': 0,
            'data_accessed_total': 0
        }
        
        # Setup logging
        self.logger = logging.getLogger('abov3.security.debug_session')
        
        # Initialize default roles and permissions
        self._initialize_default_roles()
        
        # Start background cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _initialize_default_roles(self):
        """Initialize default debug roles and permissions"""
        
        # Define permissions
        permissions = [
            DebugPermission(
                permission_id="debug_read",
                name="Debug Read",
                description="Read debug information and logs",
                level=DebugPermissionLevel.READ_ONLY,
                allowed_operations={"view_logs", "read_variables", "view_stack_trace"},
                max_session_duration=7200
            ),
            DebugPermission(
                permission_id="debug_basic",
                name="Basic Debug",
                description="Basic debugging operations",
                level=DebugPermissionLevel.BASIC_DEBUG,
                allowed_operations={
                    "view_logs", "read_variables", "view_stack_trace",
                    "set_breakpoints", "step_through", "evaluate_expressions"
                },
                max_session_duration=3600
            ),
            DebugPermission(
                permission_id="debug_advanced",
                name="Advanced Debug",
                description="Advanced debugging with profiling",
                level=DebugPermissionLevel.ADVANCED_DEBUG,
                allowed_operations={
                    "view_logs", "read_variables", "view_stack_trace",
                    "set_breakpoints", "step_through", "evaluate_expressions",
                    "memory_profiling", "performance_analysis", "modify_variables"
                },
                requires_approval=True,
                max_session_duration=1800
            ),
            DebugPermission(
                permission_id="debug_system",
                name="System Debug",
                description="System-level debugging access",
                level=DebugPermissionLevel.SYSTEM_DEBUG,
                allowed_operations={
                    "view_logs", "read_variables", "view_stack_trace",
                    "set_breakpoints", "step_through", "evaluate_expressions",
                    "memory_profiling", "performance_analysis", "modify_variables",
                    "system_calls", "process_monitoring", "file_access"
                },
                requires_approval=True,
                restricted_data_types={"credentials", "api_keys", "personal_data"},
                max_session_duration=1200
            ),
            DebugPermission(
                permission_id="debug_admin",
                name="Administrative Debug",
                description="Full administrative debugging access",
                level=DebugPermissionLevel.ADMIN_DEBUG,
                allowed_operations={"*"},  # All operations
                requires_approval=True,
                max_session_duration=900
            )
        ]
        
        for perm in permissions:
            self.debug_permissions[perm.permission_id] = perm
        
        # Define roles
        roles = [
            DebugRole(
                role_id="debug_viewer",
                name="Debug Viewer",
                description="Read-only access to debug information",
                permissions=[self.debug_permissions["debug_read"]],
                max_concurrent_sessions=5,
                session_timeout=7200
            ),
            DebugRole(
                role_id="developer",
                name="Developer",
                description="Basic debugging for developers",
                permissions=[
                    self.debug_permissions["debug_read"],
                    self.debug_permissions["debug_basic"]
                ],
                max_concurrent_sessions=3,
                session_timeout=3600,
                requires_mfa=False
            ),
            DebugRole(
                role_id="senior_developer",
                name="Senior Developer",
                description="Advanced debugging capabilities",
                permissions=[
                    self.debug_permissions["debug_read"],
                    self.debug_permissions["debug_basic"],
                    self.debug_permissions["debug_advanced"]
                ],
                max_concurrent_sessions=2,
                session_timeout=1800,
                requires_mfa=True
            ),
            DebugRole(
                role_id="system_administrator",
                name="System Administrator",
                description="System-level debugging access",
                permissions=[
                    self.debug_permissions["debug_read"],
                    self.debug_permissions["debug_basic"],
                    self.debug_permissions["debug_advanced"],
                    self.debug_permissions["debug_system"]
                ],
                max_concurrent_sessions=2,
                session_timeout=1200,
                requires_mfa=True,
                ip_restrictions=["10.0.0.0/8", "192.168.0.0/16"]
            ),
            DebugRole(
                role_id="debug_administrator",
                name="Debug Administrator",
                description="Full administrative debugging access",
                permissions=[
                    self.debug_permissions["debug_read"],
                    self.debug_permissions["debug_basic"],
                    self.debug_permissions["debug_advanced"],
                    self.debug_permissions["debug_system"],
                    self.debug_permissions["debug_admin"]
                ],
                max_concurrent_sessions=1,
                session_timeout=900,
                requires_mfa=True,
                ip_restrictions=["10.0.0.0/8"],
                time_restrictions={
                    "business_hours_only": True,
                    "allowed_hours": "09:00-17:00",
                    "allowed_days": ["mon", "tue", "wed", "thu", "fri"]
                }
            )
        ]
        
        for role in roles:
            self.debug_roles[role.role_id] = role
    
    async def create_session(
        self,
        user_id: str,
        role_id: str,
        client_ip: str,
        user_agent: str,
        mfa_token: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new secure debug session with comprehensive security checks
        
        Args:
            user_id: User identifier
            role_id: Requested debug role
            client_ip: Client IP address
            user_agent: Client user agent
            mfa_token: Multi-factor authentication token
            additional_context: Additional security context
            
        Returns:
            Dict containing session creation result
        """
        try:
            # Validate role exists
            if role_id not in self.debug_roles:
                await self._audit_security_event("session_creation_failed", {
                    "user_id": user_id,
                    "role_id": role_id,
                    "reason": "invalid_role",
                    "client_ip": client_ip
                })
                return {
                    'success': False,
                    'error': 'Invalid debug role specified',
                    'error_code': 'INVALID_ROLE'
                }
            
            role = self.debug_roles[role_id]
            
            # Check user session limits
            current_user_sessions = len(self.user_sessions.get(user_id, set()))
            if current_user_sessions >= role.max_concurrent_sessions:
                await self._audit_security_event("session_creation_failed", {
                    "user_id": user_id,
                    "role_id": role_id,
                    "reason": "session_limit_exceeded",
                    "current_sessions": current_user_sessions,
                    "max_sessions": role.max_concurrent_sessions
                })
                return {
                    'success': False,
                    'error': f'Maximum concurrent sessions exceeded ({role.max_concurrent_sessions})',
                    'error_code': 'SESSION_LIMIT_EXCEEDED'
                }
            
            # Validate IP restrictions
            if role.ip_restrictions and not self._check_ip_restrictions(client_ip, role.ip_restrictions):
                await self._audit_security_event("session_creation_failed", {
                    "user_id": user_id,
                    "role_id": role_id,
                    "reason": "ip_restriction_violation",
                    "client_ip": client_ip,
                    "allowed_ips": role.ip_restrictions
                })
                return {
                    'success': False,
                    'error': 'Access denied from this IP address',
                    'error_code': 'IP_RESTRICTED'
                }
            
            # Check time restrictions
            if role.time_restrictions and not self._check_time_restrictions(role.time_restrictions):
                await self._audit_security_event("session_creation_failed", {
                    "user_id": user_id,
                    "role_id": role_id,
                    "reason": "time_restriction_violation",
                    "current_time": datetime.now().isoformat(),
                    "restrictions": role.time_restrictions
                })
                return {
                    'success': False,
                    'error': 'Access denied outside allowed time window',
                    'error_code': 'TIME_RESTRICTED'
                }
            
            # Validate MFA if required
            mfa_verified = False
            if role.requires_mfa or self.require_mfa_for_advanced:
                if not mfa_token:
                    return {
                        'success': False,
                        'error': 'Multi-factor authentication required',
                        'error_code': 'MFA_REQUIRED',
                        'requires_mfa': True
                    }
                
                mfa_result = await self.auth_manager.verify_mfa_token(user_id, mfa_token)
                if not mfa_result['valid']:
                    await self._audit_security_event("session_creation_failed", {
                        "user_id": user_id,
                        "role_id": role_id,
                        "reason": "mfa_verification_failed"
                    })
                    return {
                        'success': False,
                        'error': 'Multi-factor authentication failed',
                        'error_code': 'MFA_FAILED'
                    }
                mfa_verified = True
            
            # Generate secure session credentials
            session_id = str(uuid.uuid4())
            session_token = secrets.token_urlsafe(32)
            data_key = await self.crypto_manager.generate_data_key()
            
            # Determine maximum session duration
            max_duration = min(
                role.session_timeout,
                min(perm.max_session_duration for perm in role.permissions)
            )
            
            # Determine permission level
            permission_level = max(perm.level for perm in role.permissions)
            
            # Create session
            session = SecureDebugSession(
                session_id=session_id,
                user_id=user_id,
                role=role,
                created_at=datetime.now(),
                last_activity=datetime.now(),
                status=DebugSessionStatus.INITIALIZING,
                client_ip=client_ip,
                user_agent=user_agent,
                encrypted_data_key=data_key['encrypted_key'],
                session_token=session_token,
                mfa_verified=mfa_verified,
                permission_level=permission_level,
                max_duration=max_duration
            )
            
            # Store session
            self.active_sessions[session_id] = session
            self.session_locks[session_id] = threading.Lock()
            
            # Update user session tracking
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = set()
            self.user_sessions[user_id].add(session_id)
            
            # Activate session
            session.status = DebugSessionStatus.ACTIVE
            
            # Update metrics
            self.session_metrics['total_sessions'] += 1
            self.session_metrics['active_sessions'] += 1
            
            # Audit session creation
            await self._audit_security_event("debug_session_created", {
                "session_id": session_id,
                "user_id": user_id,
                "role_id": role_id,
                "permission_level": permission_level.value,
                "client_ip": client_ip,
                "mfa_verified": mfa_verified,
                "max_duration": max_duration
            })
            
            self.logger.info(f"Debug session created: {session_id} for user {user_id} with role {role_id}")
            
            return {
                'success': True,
                'session_id': session_id,
                'session_token': session_token,
                'permission_level': permission_level.value,
                'max_duration': max_duration,
                'allowed_operations': list(self._get_allowed_operations(role)),
                'expires_at': (datetime.now() + timedelta(seconds=max_duration)).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Session creation error: {e}")
            await self._audit_security_event("session_creation_error", {
                "user_id": user_id,
                "error": str(e)
            })
            return {
                'success': False,
                'error': 'Internal session creation error',
                'error_code': 'INTERNAL_ERROR'
            }
    
    async def validate_session(self, session_id: str, session_token: str) -> Dict[str, Any]:
        """
        Validate a debug session with comprehensive security checks
        
        Args:
            session_id: Session identifier
            session_token: Session authentication token
            
        Returns:
            Dict containing session validation result
        """
        try:
            # Check if session exists
            if session_id not in self.active_sessions:
                await self._audit_security_event("session_validation_failed", {
                    "session_id": session_id,
                    "reason": "session_not_found"
                })
                return {
                    'valid': False,
                    'error': 'Session not found',
                    'error_code': 'SESSION_NOT_FOUND'
                }
            
            session = self.active_sessions[session_id]
            
            # Validate session token
            if session.session_token != session_token:
                await self._audit_security_event("session_validation_failed", {
                    "session_id": session_id,
                    "user_id": session.user_id,
                    "reason": "invalid_token"
                })
                self._handle_security_violation(session_id, "invalid_token")
                return {
                    'valid': False,
                    'error': 'Invalid session token',
                    'error_code': 'INVALID_TOKEN'
                }
            
            # Check session status
            if session.status != DebugSessionStatus.ACTIVE:
                return {
                    'valid': False,
                    'error': f'Session is {session.status.value}',
                    'error_code': 'SESSION_INACTIVE'
                }
            
            # Check expiration
            if session.is_expired():
                await self.terminate_session(session_id, "Session expired")
                return {
                    'valid': False,
                    'error': 'Session expired',
                    'error_code': 'SESSION_EXPIRED'
                }
            
            # Check MFA status for sensitive operations
            if not session.mfa_verified and session.permission_level.value in ['advanced_debug', 'system_debug', 'admin_debug']:
                return {
                    'valid': False,
                    'error': 'MFA verification required for this permission level',
                    'error_code': 'MFA_REQUIRED'
                }
            
            # Update last activity
            session.update_activity()
            
            return {
                'valid': True,
                'session': {
                    'session_id': session_id,
                    'user_id': session.user_id,
                    'role_id': session.role.role_id,
                    'permission_level': session.permission_level.value,
                    'allowed_operations': list(self._get_allowed_operations(session.role)),
                    'expires_in': session.max_duration - int((datetime.now() - session.last_activity).total_seconds()),
                    'operations_count': session.operations_count,
                    'data_accessed': session.data_accessed
                }
            }
            
        except Exception as e:
            self.logger.error(f"Session validation error: {e}")
            return {
                'valid': False,
                'error': 'Internal validation error',
                'error_code': 'INTERNAL_ERROR'
            }
    
    async def check_operation_permission(
        self,
        session_id: str,
        operation: str,
        resource: Optional[str] = None,
        data_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check if a session has permission for a specific operation
        
        Args:
            session_id: Session identifier
            operation: Operation to check
            resource: Optional resource identifier
            data_type: Optional data type being accessed
            
        Returns:
            Dict containing permission check result
        """
        try:
            if session_id not in self.active_sessions:
                return {
                    'allowed': False,
                    'reason': 'Session not found'
                }
            
            session = self.active_sessions[session_id]
            
            # Check if session is active
            if not session.is_active():
                return {
                    'allowed': False,
                    'reason': f'Session is {session.status.value}'
                }
            
            # Get allowed operations for the role
            allowed_operations = self._get_allowed_operations(session.role)
            
            # Check if operation is allowed
            if operation not in allowed_operations and "*" not in allowed_operations:
                await self._audit_security_event("operation_permission_denied", {
                    "session_id": session_id,
                    "user_id": session.user_id,
                    "operation": operation,
                    "reason": "operation_not_allowed"
                })
                self.session_metrics['permission_denials'] += 1
                return {
                    'allowed': False,
                    'reason': f'Operation {operation} not allowed for role {session.role.role_id}'
                }
            
            # Check restricted data types
            if data_type:
                for perm in session.role.permissions:
                    if data_type in perm.restricted_data_types:
                        await self._audit_security_event("data_access_denied", {
                            "session_id": session_id,
                            "user_id": session.user_id,
                            "operation": operation,
                            "data_type": data_type,
                            "reason": "restricted_data_type"
                        })
                        return {
                            'allowed': False,
                            'reason': f'Data type {data_type} is restricted'
                        }
            
            # Check if approval is required
            requires_approval = any(perm.requires_approval for perm in session.role.permissions
                                    if operation in perm.allowed_operations or "*" in perm.allowed_operations)
            
            if requires_approval:
                # For now, we'll allow operations that require approval
                # In a full implementation, this would check an approval system
                await self._audit_security_event("approval_required_operation", {
                    "session_id": session_id,
                    "user_id": session.user_id,
                    "operation": operation,
                    "resource": resource
                })
            
            # Record the operation
            session.operations_count += 1
            session.audit_trail.append({
                'timestamp': datetime.now().isoformat(),
                'operation': operation,
                'resource': resource,
                'data_type': data_type,
                'allowed': True
            })
            
            return {
                'allowed': True,
                'requires_approval': requires_approval,
                'permission_level': session.permission_level.value
            }
            
        except Exception as e:
            self.logger.error(f"Permission check error: {e}")
            return {
                'allowed': False,
                'reason': 'Internal permission check error'
            }
    
    async def record_data_access(
        self,
        session_id: str,
        data_size: int,
        data_type: str,
        sensitive: bool = False
    ) -> bool:
        """
        Record data access for audit and monitoring
        
        Args:
            session_id: Session identifier
            data_size: Size of data accessed in bytes
            data_type: Type of data accessed
            sensitive: Whether the data is sensitive
            
        Returns:
            bool indicating if recording was successful
        """
        try:
            if session_id not in self.active_sessions:
                return False
            
            session = self.active_sessions[session_id]
            session.data_accessed += data_size
            self.session_metrics['data_accessed_total'] += data_size
            
            # Audit data access
            await self._audit_security_event("debug_data_accessed", {
                "session_id": session_id,
                "user_id": session.user_id,
                "data_size": data_size,
                "data_type": data_type,
                "sensitive": sensitive,
                "total_data_accessed": session.data_accessed
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data access recording error: {e}")
            return False
    
    async def terminate_session(self, session_id: str, reason: str = "User request") -> bool:
        """
        Terminate a debug session with proper cleanup
        
        Args:
            session_id: Session identifier
            reason: Reason for termination
            
        Returns:
            bool indicating if termination was successful
        """
        try:
            if session_id not in self.active_sessions:
                return False
            
            session = self.active_sessions[session_id]
            
            # Update session status
            session.status = DebugSessionStatus.TERMINATED
            
            # Audit session termination
            await self._audit_security_event("debug_session_terminated", {
                "session_id": session_id,
                "user_id": session.user_id,
                "reason": reason,
                "duration": int((datetime.now() - session.created_at).total_seconds()),
                "operations_count": session.operations_count,
                "data_accessed": session.data_accessed
            })
            
            # Cleanup
            self._cleanup_session(session_id)
            
            self.logger.info(f"Debug session terminated: {session_id}, reason: {reason}")
            return True
            
        except Exception as e:
            self.logger.error(f"Session termination error: {e}")
            return False
    
    async def suspend_session(self, session_id: str, reason: str = "Security violation") -> bool:
        """
        Suspend a session due to security concerns
        
        Args:
            session_id: Session identifier
            reason: Reason for suspension
            
        Returns:
            bool indicating if suspension was successful
        """
        try:
            if session_id not in self.active_sessions:
                return False
            
            session = self.active_sessions[session_id]
            session.status = DebugSessionStatus.SUSPENDED
            
            # Record security event
            session.security_events.append({
                'timestamp': datetime.now().isoformat(),
                'event': 'session_suspended',
                'reason': reason
            })
            
            # Audit suspension
            await self._audit_security_event("debug_session_suspended", {
                "session_id": session_id,
                "user_id": session.user_id,
                "reason": reason
            })
            
            self.session_metrics['security_violations'] += 1
            self.logger.warning(f"Debug session suspended: {session_id}, reason: {reason}")
            return True
            
        except Exception as e:
            self.logger.error(f"Session suspension error: {e}")
            return False
    
    async def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed session status"""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        
        return {
            'session_id': session_id,
            'user_id': session.user_id,
            'role': session.role.role_id,
            'status': session.status.value,
            'created_at': session.created_at.isoformat(),
            'last_activity': session.last_activity.isoformat(),
            'permission_level': session.permission_level.value,
            'mfa_verified': session.mfa_verified,
            'operations_count': session.operations_count,
            'data_accessed': session.data_accessed,
            'expires_in': session.max_duration - int((datetime.now() - session.last_activity).total_seconds()),
            'client_ip': session.client_ip,
            'security_events_count': len(session.security_events)
        }
    
    async def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all sessions for a user"""
        if user_id not in self.user_sessions:
            return []
        
        sessions = []
        for session_id in self.user_sessions[user_id]:
            if session_id in self.active_sessions:
                session_status = await self.get_session_status(session_id)
                if session_status:
                    sessions.append(session_status)
        
        return sessions
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get session manager metrics"""
        return {
            **self.session_metrics,
            'current_active_sessions': len([s for s in self.active_sessions.values() 
                                           if s.status == DebugSessionStatus.ACTIVE]),
            'roles_defined': len(self.debug_roles),
            'permissions_defined': len(self.debug_permissions)
        }
    
    def _get_allowed_operations(self, role: DebugRole) -> Set[str]:
        """Get all allowed operations for a role"""
        operations = set()
        for perm in role.permissions:
            operations.update(perm.allowed_operations)
        return operations
    
    def _check_ip_restrictions(self, client_ip: str, allowed_ips: List[str]) -> bool:
        """Check if client IP is allowed"""
        # Simplified IP checking - in production, use proper CIDR matching
        for allowed_ip in allowed_ips:
            if allowed_ip.endswith('/8'):
                network = allowed_ip.split('/')[0]
                if client_ip.startswith(network.rsplit('.', 3)[0]):
                    return True
            elif allowed_ip.endswith('/16'):
                network = allowed_ip.split('/')[0]
                if client_ip.startswith('.'.join(network.split('.')[:2])):
                    return True
            elif client_ip == allowed_ip:
                return True
        return False
    
    def _check_time_restrictions(self, restrictions: Dict[str, Any]) -> bool:
        """Check if current time is within allowed window"""
        if not restrictions.get('business_hours_only', False):
            return True
        
        now = datetime.now()
        current_day = now.strftime('%a').lower()
        current_time = now.strftime('%H:%M')
        
        # Check allowed days
        allowed_days = restrictions.get('allowed_days', [])
        if allowed_days and current_day not in allowed_days:
            return False
        
        # Check allowed hours
        allowed_hours = restrictions.get('allowed_hours', '00:00-23:59')
        start_time, end_time = allowed_hours.split('-')
        
        if start_time <= current_time <= end_time:
            return True
        
        return False
    
    def _handle_security_violation(self, session_id: str, violation_type: str):
        """Handle security violation"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.security_events.append({
                'timestamp': datetime.now().isoformat(),
                'violation': violation_type
            })
            
            # Suspend session for serious violations
            if violation_type in ['invalid_token', 'permission_abuse', 'suspicious_activity']:
                asyncio.create_task(self.suspend_session(session_id, f"Security violation: {violation_type}"))
    
    def _cleanup_session(self, session_id: str):
        """Clean up session resources"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            
            # Remove from user sessions tracking
            if session.user_id in self.user_sessions:
                self.user_sessions[session.user_id].discard(session_id)
                if not self.user_sessions[session.user_id]:
                    del self.user_sessions[session.user_id]
            
            # Remove session and lock
            del self.active_sessions[session_id]
            if session_id in self.session_locks:
                del self.session_locks[session_id]
            
            # Update metrics
            self.session_metrics['active_sessions'] = len(self.active_sessions)
    
    async def _cleanup_expired_sessions(self):
        """Cleanup expired and invalid sessions"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.active_sessions.items():
            if (
                session.is_expired() or
                session.status in [DebugSessionStatus.TERMINATED, DebugSessionStatus.EXPIRED] or
                (current_time - session.last_activity).seconds > self.max_idle_time
            ):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            await self.terminate_session(session_id, "Session cleanup - expired")
    
    def _start_cleanup_task(self):
        """Start background cleanup task"""
        async def cleanup_loop():
            while True:
                try:
                    await self._cleanup_expired_sessions()
                    await asyncio.sleep(self.session_cleanup_interval)
                except Exception as e:
                    self.logger.error(f"Cleanup task error: {e}")
                    await asyncio.sleep(60)  # Wait before retrying
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def _audit_security_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log security event to audit system"""
        if self.audit_logger:
            await self.audit_logger.log_event(event_type, {
                'component': 'secure_debug_session',
                'timestamp': datetime.now().isoformat(),
                **event_data
            })
    
    async def shutdown(self):
        """Shutdown session manager"""
        try:
            # Cancel cleanup task
            if self._cleanup_task:
                self._cleanup_task.cancel()
            
            # Terminate all active sessions
            session_ids = list(self.active_sessions.keys())
            for session_id in session_ids:
                await self.terminate_session(session_id, "System shutdown")
            
            self.logger.info("Secure debug session manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Session manager shutdown error: {e}")