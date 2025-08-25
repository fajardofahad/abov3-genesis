"""
ABOV3 Genesis - Authentication and Authorization Manager
Enterprise-grade authentication and authorization system with role-based access control
"""

import asyncio
import hashlib
import secrets
import json
import jwt
import bcrypt
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Union
from enum import Enum
from dataclasses import dataclass, asdict
import uuid
import time
from cryptography.fernet import Fernet


class UserRole(Enum):
    """User roles with hierarchical permissions"""
    GUEST = "guest"
    USER = "user"
    DEVELOPER = "developer"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class Permission(Enum):
    """System permissions"""
    # File operations
    READ_FILES = "read_files"
    WRITE_FILES = "write_files"
    DELETE_FILES = "delete_files"
    EXECUTE_FILES = "execute_files"
    
    # AI operations
    USE_AI = "use_ai"
    CONFIGURE_AI = "configure_ai"
    ACCESS_AI_LOGS = "access_ai_logs"
    
    # Project operations
    CREATE_PROJECT = "create_project"
    DELETE_PROJECT = "delete_project"
    CONFIGURE_PROJECT = "configure_project"
    
    # System operations
    VIEW_LOGS = "view_logs"
    CONFIGURE_SECURITY = "configure_security"
    MANAGE_USERS = "manage_users"
    SYSTEM_ADMIN = "system_admin"
    
    # Advanced operations
    BYPASS_SECURITY = "bypass_security"
    EMERGENCY_ACCESS = "emergency_access"


@dataclass
class User:
    """User account information"""
    user_id: str
    username: str
    email: str
    password_hash: str
    salt: str
    role: UserRole
    permissions: Set[Permission]
    created_at: datetime
    last_login: Optional[datetime] = None
    login_attempts: int = 0
    locked_until: Optional[datetime] = None
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    session_timeout: int = 3600  # seconds
    is_active: bool = True


@dataclass
class Session:
    """User session information"""
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True


@dataclass
class LoginAttempt:
    """Login attempt tracking"""
    username: str
    ip_address: str
    timestamp: datetime
    success: bool
    failure_reason: Optional[str] = None


class AuthenticationManager:
    """
    Comprehensive Authentication Manager
    Handles user authentication, password management, MFA, and session management
    """
    
    def __init__(self, security_dir: Path, crypto_manager=None, audit_logger=None,
                 max_attempts: int = 5, lockout_duration: int = 900, session_timeout: int = 3600):
        self.security_dir = security_dir
        self.crypto_manager = crypto_manager
        self.audit_logger = audit_logger
        self.max_attempts = max_attempts
        self.lockout_duration = lockout_duration
        self.session_timeout = session_timeout
        
        # Storage
        self.users_file = security_dir / 'users.json'
        self.sessions_file = security_dir / 'sessions.json'
        self.login_attempts_file = security_dir / 'login_attempts.json'
        
        # In-memory caches
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        self.login_attempts: List[LoginAttempt] = []
        
        # JWT configuration
        self.jwt_secret = self._generate_jwt_secret()
        self.jwt_algorithm = 'HS256'
        
        # Statistics
        self.auth_stats = {
            'total_login_attempts': 0,
            'successful_logins': 0,
            'failed_logins': 0,
            'active_sessions': 0,
            'locked_accounts': 0,
            'mfa_enabled_users': 0
        }
        
        # Initialize
        asyncio.create_task(self._initialize())
    
    async def _initialize(self):
        """Initialize authentication system"""
        # Load existing data
        await self._load_users()
        await self._load_sessions()
        await self._load_login_attempts()
        
        # Create default admin user if no users exist
        if not self.users:
            await self._create_default_admin()
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_expired_sessions())
    
    def _generate_jwt_secret(self) -> str:
        """Generate JWT secret key"""
        secret_file = self.security_dir / 'jwt_secret.key'
        
        if secret_file.exists():
            return secret_file.read_text()
        else:
            secret = secrets.token_urlsafe(64)
            secret_file.write_text(secret)
            secret_file.chmod(0o600)  # Readable only by owner
            return secret
    
    async def _create_default_admin(self):
        """Create default admin user"""
        default_password = secrets.token_urlsafe(16)
        
        admin_user = await self.create_user(
            username="admin",
            email="admin@abov3.local",
            password=default_password,
            role=UserRole.SUPER_ADMIN
        )
        
        if admin_user['success']:
            # Save default password to secure file
            creds_file = self.security_dir / 'default_admin_creds.txt'
            creds_file.write_text(f"Username: admin\nPassword: {default_password}\n")
            creds_file.chmod(0o600)
            
            if self.audit_logger:
                await self.audit_logger.log_event("default_admin_created", {
                    "username": "admin",
                    "credentials_file": str(creds_file)
                })
    
    async def create_user(self, username: str, email: str, password: str, 
                         role: UserRole, permissions: Optional[Set[Permission]] = None) -> Dict[str, Any]:
        """Create a new user account"""
        try:
            # Validate input
            if not username or len(username) < 3:
                return {'success': False, 'error': 'Username must be at least 3 characters'}
            
            if not email or '@' not in email:
                return {'success': False, 'error': 'Invalid email address'}
            
            if not password or len(password) < 8:
                return {'success': False, 'error': 'Password must be at least 8 characters'}
            
            # Check if user already exists
            if any(u.username.lower() == username.lower() for u in self.users.values()):
                return {'success': False, 'error': 'Username already exists'}
            
            if any(u.email.lower() == email.lower() for u in self.users.values()):
                return {'success': False, 'error': 'Email already exists'}
            
            # Generate secure password hash
            salt = bcrypt.gensalt()
            password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
            
            # Create user
            user_id = str(uuid.uuid4())
            user = User(
                user_id=user_id,
                username=username,
                email=email,
                password_hash=password_hash.decode('utf-8'),
                salt=salt.decode('utf-8'),
                role=role,
                permissions=permissions or self._get_default_permissions(role),
                created_at=datetime.now()
            )
            
            # Store user
            self.users[user_id] = user
            await self._save_users()
            
            # Update statistics
            if user.mfa_enabled:
                self.auth_stats['mfa_enabled_users'] += 1
            
            # Audit log
            if self.audit_logger:
                await self.audit_logger.log_event("user_created", {
                    "user_id": user_id,
                    "username": username,
                    "email": email,
                    "role": role.value
                })
            
            return {
                'success': True,
                'user_id': user_id,
                'message': 'User created successfully'
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Failed to create user: {str(e)}'}
    
    async def authenticate(self, username: str, password: str, ip_address: str = None,
                          user_agent: str = None, mfa_token: str = None) -> Dict[str, Any]:
        """Authenticate user with username/password and optional MFA"""
        self.auth_stats['total_login_attempts'] += 1
        
        try:
            # Find user
            user = None
            for u in self.users.values():
                if u.username.lower() == username.lower():
                    user = u
                    break
            
            # Record login attempt
            attempt = LoginAttempt(
                username=username,
                ip_address=ip_address or 'unknown',
                timestamp=datetime.now(),
                success=False
            )
            
            if not user:
                attempt.failure_reason = 'User not found'
                self.login_attempts.append(attempt)
                self.auth_stats['failed_logins'] += 1
                await self._save_login_attempts()
                return {'success': False, 'error': 'Invalid credentials'}
            
            # Check if account is locked
            if user.locked_until and datetime.now() < user.locked_until:
                attempt.failure_reason = 'Account locked'
                self.login_attempts.append(attempt)
                self.auth_stats['failed_logins'] += 1
                await self._save_login_attempts()
                return {
                    'success': False, 
                    'error': f'Account locked until {user.locked_until.isoformat()}'
                }
            
            # Check if account is active
            if not user.is_active:
                attempt.failure_reason = 'Account disabled'
                self.login_attempts.append(attempt)
                self.auth_stats['failed_logins'] += 1
                await self._save_login_attempts()
                return {'success': False, 'error': 'Account disabled'}
            
            # Verify password
            if not bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
                user.login_attempts += 1
                
                # Lock account if too many failed attempts
                if user.login_attempts >= self.max_attempts:
                    user.locked_until = datetime.now() + timedelta(seconds=self.lockout_duration)
                    self.auth_stats['locked_accounts'] += 1
                    attempt.failure_reason = 'Invalid password - account locked'
                else:
                    attempt.failure_reason = 'Invalid password'
                
                self.login_attempts.append(attempt)
                self.auth_stats['failed_logins'] += 1
                await self._save_users()
                await self._save_login_attempts()
                
                return {'success': False, 'error': 'Invalid credentials'}
            
            # MFA verification if enabled
            if user.mfa_enabled:
                if not mfa_token:
                    return {
                        'success': False,
                        'error': 'MFA token required',
                        'require_mfa': True
                    }
                
                if not self._verify_mfa_token(user.mfa_secret, mfa_token):
                    attempt.failure_reason = 'Invalid MFA token'
                    self.login_attempts.append(attempt)
                    self.auth_stats['failed_logins'] += 1
                    await self._save_login_attempts()
                    return {'success': False, 'error': 'Invalid MFA token'}
            
            # Successful authentication
            user.login_attempts = 0
            user.locked_until = None
            user.last_login = datetime.now()
            
            # Create session
            session_result = await self._create_session(user, ip_address, user_agent)
            
            attempt.success = True
            self.login_attempts.append(attempt)
            self.auth_stats['successful_logins'] += 1
            
            await self._save_users()
            await self._save_login_attempts()
            
            # Audit log
            if self.audit_logger:
                await self.audit_logger.log_event("user_authenticated", {
                    "user_id": user.user_id,
                    "username": user.username,
                    "ip_address": ip_address,
                    "mfa_used": user.mfa_enabled
                })
            
            return {
                'success': True,
                'user_id': user.user_id,
                'username': user.username,
                'role': user.role.value,
                'permissions': [p.value for p in user.permissions],
                'session_id': session_result['session_id'],
                'token': session_result['token'],
                'expires_at': session_result['expires_at']
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Authentication failed: {str(e)}'}
    
    async def _create_session(self, user: User, ip_address: str = None, 
                             user_agent: str = None) -> Dict[str, Any]:
        """Create new user session"""
        session_id = str(uuid.uuid4())
        now = datetime.now()
        expires_at = now + timedelta(seconds=user.session_timeout)
        
        session = Session(
            session_id=session_id,
            user_id=user.user_id,
            created_at=now,
            last_activity=now,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.sessions[session_id] = session
        self.auth_stats['active_sessions'] += 1
        await self._save_sessions()
        
        # Generate JWT token
        token_payload = {
            'session_id': session_id,
            'user_id': user.user_id,
            'username': user.username,
            'role': user.role.value,
            'permissions': [p.value for p in user.permissions],
            'iat': int(now.timestamp()),
            'exp': int(expires_at.timestamp())
        }
        
        token = jwt.encode(token_payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        
        return {
            'session_id': session_id,
            'token': token,
            'expires_at': expires_at.isoformat()
        }
    
    async def validate_token(self, token: str) -> Dict[str, Any]:
        """Validate JWT token and return user information"""
        try:
            # Decode token
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            session_id = payload.get('session_id')
            user_id = payload.get('user_id')
            
            # Check session exists and is valid
            session = self.sessions.get(session_id)
            if not session or not session.is_active:
                return {'valid': False, 'error': 'Invalid session'}
            
            # Check session expiration
            if datetime.now() > session.expires_at:
                session.is_active = False
                self.auth_stats['active_sessions'] -= 1
                await self._save_sessions()
                return {'valid': False, 'error': 'Session expired'}
            
            # Check user exists and is active
            user = self.users.get(user_id)
            if not user or not user.is_active:
                return {'valid': False, 'error': 'User not found or inactive'}
            
            # Update session activity
            session.last_activity = datetime.now()
            await self._save_sessions()
            
            return {
                'valid': True,
                'user_id': user.user_id,
                'username': user.username,
                'role': user.role.value,
                'permissions': [p.value for p in user.permissions],
                'session_id': session_id
            }
            
        except jwt.ExpiredSignatureError:
            return {'valid': False, 'error': 'Token expired'}
        except jwt.InvalidTokenError:
            return {'valid': False, 'error': 'Invalid token'}
        except Exception as e:
            return {'valid': False, 'error': f'Token validation failed: {str(e)}'}
    
    async def logout(self, session_id: str) -> Dict[str, Any]:
        """Logout user by invalidating session"""
        try:
            session = self.sessions.get(session_id)
            if not session:
                return {'success': False, 'error': 'Session not found'}
            
            session.is_active = False
            self.auth_stats['active_sessions'] -= 1
            await self._save_sessions()
            
            # Audit log
            if self.audit_logger:
                await self.audit_logger.log_event("user_logout", {
                    "session_id": session_id,
                    "user_id": session.user_id
                })
            
            return {'success': True, 'message': 'Logged out successfully'}
            
        except Exception as e:
            return {'success': False, 'error': f'Logout failed: {str(e)}'}
    
    async def change_password(self, user_id: str, current_password: str, 
                             new_password: str) -> Dict[str, Any]:
        """Change user password"""
        try:
            user = self.users.get(user_id)
            if not user:
                return {'success': False, 'error': 'User not found'}
            
            # Verify current password
            if not bcrypt.checkpw(current_password.encode('utf-8'), user.password_hash.encode('utf-8')):
                return {'success': False, 'error': 'Current password incorrect'}
            
            # Validate new password
            if len(new_password) < 8:
                return {'success': False, 'error': 'New password must be at least 8 characters'}
            
            # Generate new password hash
            salt = bcrypt.gensalt()
            password_hash = bcrypt.hashpw(new_password.encode('utf-8'), salt)
            
            user.password_hash = password_hash.decode('utf-8')
            user.salt = salt.decode('utf-8')
            
            await self._save_users()
            
            # Audit log
            if self.audit_logger:
                await self.audit_logger.log_event("password_changed", {
                    "user_id": user_id,
                    "username": user.username
                })
            
            return {'success': True, 'message': 'Password changed successfully'}
            
        except Exception as e:
            return {'success': False, 'error': f'Password change failed: {str(e)}'}
    
    async def enable_mfa(self, user_id: str) -> Dict[str, Any]:
        """Enable MFA for user"""
        try:
            user = self.users.get(user_id)
            if not user:
                return {'success': False, 'error': 'User not found'}
            
            # Generate MFA secret
            mfa_secret = secrets.token_urlsafe(32)
            user.mfa_secret = mfa_secret
            user.mfa_enabled = True
            
            self.auth_stats['mfa_enabled_users'] += 1
            await self._save_users()
            
            # Generate QR code URL for authenticator apps
            issuer = "ABOV3-Genesis"
            qr_url = f"otpauth://totp/{issuer}:{user.username}?secret={mfa_secret}&issuer={issuer}"
            
            # Audit log
            if self.audit_logger:
                await self.audit_logger.log_event("mfa_enabled", {
                    "user_id": user_id,
                    "username": user.username
                })
            
            return {
                'success': True,
                'mfa_secret': mfa_secret,
                'qr_url': qr_url,
                'message': 'MFA enabled successfully'
            }
            
        except Exception as e:
            return {'success': False, 'error': f'MFA enable failed: {str(e)}'}
    
    def _verify_mfa_token(self, secret: str, token: str) -> bool:
        """Verify TOTP MFA token"""
        try:
            import pyotp
            totp = pyotp.TOTP(secret)
            return totp.verify(token, valid_window=1)  # Allow 30-second window
        except:
            return False
    
    def _get_default_permissions(self, role: UserRole) -> Set[Permission]:
        """Get default permissions for role"""
        role_permissions = {
            UserRole.GUEST: {
                Permission.READ_FILES,
            },
            UserRole.USER: {
                Permission.READ_FILES,
                Permission.WRITE_FILES,
                Permission.USE_AI,
            },
            UserRole.DEVELOPER: {
                Permission.READ_FILES,
                Permission.WRITE_FILES,
                Permission.DELETE_FILES,
                Permission.USE_AI,
                Permission.CONFIGURE_AI,
                Permission.CREATE_PROJECT,
                Permission.CONFIGURE_PROJECT,
            },
            UserRole.ADMIN: {
                Permission.READ_FILES,
                Permission.WRITE_FILES,
                Permission.DELETE_FILES,
                Permission.EXECUTE_FILES,
                Permission.USE_AI,
                Permission.CONFIGURE_AI,
                Permission.ACCESS_AI_LOGS,
                Permission.CREATE_PROJECT,
                Permission.DELETE_PROJECT,
                Permission.CONFIGURE_PROJECT,
                Permission.VIEW_LOGS,
                Permission.CONFIGURE_SECURITY,
                Permission.MANAGE_USERS,
            },
            UserRole.SUPER_ADMIN: set(Permission)  # All permissions
        }
        
        return role_permissions.get(role, set())
    
    async def _load_users(self):
        """Load users from file"""
        try:
            if self.users_file.exists():
                with open(self.users_file, 'r') as f:
                    data = json.load(f)
                
                for user_data in data:
                    user = User(
                        user_id=user_data['user_id'],
                        username=user_data['username'],
                        email=user_data['email'],
                        password_hash=user_data['password_hash'],
                        salt=user_data['salt'],
                        role=UserRole(user_data['role']),
                        permissions={Permission(p) for p in user_data['permissions']},
                        created_at=datetime.fromisoformat(user_data['created_at']),
                        last_login=datetime.fromisoformat(user_data['last_login']) if user_data.get('last_login') else None,
                        login_attempts=user_data.get('login_attempts', 0),
                        locked_until=datetime.fromisoformat(user_data['locked_until']) if user_data.get('locked_until') else None,
                        mfa_enabled=user_data.get('mfa_enabled', False),
                        mfa_secret=user_data.get('mfa_secret'),
                        session_timeout=user_data.get('session_timeout', self.session_timeout),
                        is_active=user_data.get('is_active', True)
                    )
                    self.users[user.user_id] = user
        except Exception as e:
            print(f"Error loading users: {e}")
    
    async def _save_users(self):
        """Save users to file"""
        try:
            data = []
            for user in self.users.values():
                user_dict = {
                    'user_id': user.user_id,
                    'username': user.username,
                    'email': user.email,
                    'password_hash': user.password_hash,
                    'salt': user.salt,
                    'role': user.role.value,
                    'permissions': [p.value for p in user.permissions],
                    'created_at': user.created_at.isoformat(),
                    'last_login': user.last_login.isoformat() if user.last_login else None,
                    'login_attempts': user.login_attempts,
                    'locked_until': user.locked_until.isoformat() if user.locked_until else None,
                    'mfa_enabled': user.mfa_enabled,
                    'mfa_secret': user.mfa_secret,
                    'session_timeout': user.session_timeout,
                    'is_active': user.is_active
                }
                data.append(user_dict)
            
            with open(self.users_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving users: {e}")
    
    async def _load_sessions(self):
        """Load sessions from file"""
        try:
            if self.sessions_file.exists():
                with open(self.sessions_file, 'r') as f:
                    data = json.load(f)
                
                for session_data in data:
                    if session_data.get('is_active', True):
                        session = Session(
                            session_id=session_data['session_id'],
                            user_id=session_data['user_id'],
                            created_at=datetime.fromisoformat(session_data['created_at']),
                            last_activity=datetime.fromisoformat(session_data['last_activity']),
                            expires_at=datetime.fromisoformat(session_data['expires_at']),
                            ip_address=session_data.get('ip_address'),
                            user_agent=session_data.get('user_agent'),
                            is_active=session_data.get('is_active', True)
                        )
                        self.sessions[session.session_id] = session
                        self.auth_stats['active_sessions'] += 1
        except Exception as e:
            print(f"Error loading sessions: {e}")
    
    async def _save_sessions(self):
        """Save sessions to file"""
        try:
            data = []
            for session in self.sessions.values():
                session_dict = {
                    'session_id': session.session_id,
                    'user_id': session.user_id,
                    'created_at': session.created_at.isoformat(),
                    'last_activity': session.last_activity.isoformat(),
                    'expires_at': session.expires_at.isoformat(),
                    'ip_address': session.ip_address,
                    'user_agent': session.user_agent,
                    'is_active': session.is_active
                }
                data.append(session_dict)
            
            with open(self.sessions_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving sessions: {e}")
    
    async def _load_login_attempts(self):
        """Load login attempts from file"""
        try:
            if self.login_attempts_file.exists():
                with open(self.login_attempts_file, 'r') as f:
                    data = json.load(f)
                
                # Only keep recent attempts (last 30 days)
                cutoff = datetime.now() - timedelta(days=30)
                
                for attempt_data in data:
                    timestamp = datetime.fromisoformat(attempt_data['timestamp'])
                    if timestamp > cutoff:
                        attempt = LoginAttempt(
                            username=attempt_data['username'],
                            ip_address=attempt_data['ip_address'],
                            timestamp=timestamp,
                            success=attempt_data['success'],
                            failure_reason=attempt_data.get('failure_reason')
                        )
                        self.login_attempts.append(attempt)
        except Exception as e:
            print(f"Error loading login attempts: {e}")
    
    async def _save_login_attempts(self):
        """Save login attempts to file"""
        try:
            data = []
            for attempt in self.login_attempts[-1000:]:  # Keep last 1000 attempts
                attempt_dict = {
                    'username': attempt.username,
                    'ip_address': attempt.ip_address,
                    'timestamp': attempt.timestamp.isoformat(),
                    'success': attempt.success,
                    'failure_reason': attempt.failure_reason
                }
                data.append(attempt_dict)
            
            with open(self.login_attempts_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving login attempts: {e}")
    
    async def _cleanup_expired_sessions(self):
        """Cleanup expired sessions (background task)"""
        while True:
            try:
                now = datetime.now()
                expired_sessions = []
                
                for session_id, session in self.sessions.items():
                    if now > session.expires_at or not session.is_active:
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    del self.sessions[session_id]
                    self.auth_stats['active_sessions'] -= 1
                
                if expired_sessions:
                    await self._save_sessions()
                
                # Sleep for 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                print(f"Error in session cleanup: {e}")
                await asyncio.sleep(60)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get authentication statistics"""
        return self.auth_stats.copy()


class AuthorizationManager:
    """
    Role-Based Access Control Authorization Manager
    Handles permission checking and resource access control
    """
    
    def __init__(self, security_dir: Path, audit_logger=None):
        self.security_dir = security_dir
        self.audit_logger = audit_logger
        
        # Resource permissions
        self.resource_permissions_file = security_dir / 'resource_permissions.json'
        self.resource_permissions: Dict[str, Dict[str, Set[Permission]]] = {}
        
        # Authorization statistics
        self.authz_stats = {
            'total_permission_checks': 0,
            'granted_permissions': 0,
            'denied_permissions': 0,
            'by_permission_type': {}
        }
        
        # Load resource permissions
        asyncio.create_task(self._load_resource_permissions())
    
    async def check_permission(self, user_id: str, permission: Union[Permission, str], 
                              resource: str = None, context: Dict[str, Any] = None) -> bool:
        """Check if user has specific permission"""
        self.authz_stats['total_permission_checks'] += 1
        
        try:
            # Convert string to Permission enum if needed
            if isinstance(permission, str):
                try:
                    permission = Permission(permission)
                except ValueError:
                    self.authz_stats['denied_permissions'] += 1
                    return False
            
            # Get user from authentication manager (would need reference)
            # For now, assume we have user permissions available
            
            # Track statistics
            perm_type = permission.value
            self.authz_stats['by_permission_type'][perm_type] = \
                self.authz_stats['by_permission_type'].get(perm_type, 0) + 1
            
            # Check resource-specific permissions
            if resource and resource in self.resource_permissions:
                resource_perms = self.resource_permissions[resource].get(user_id, set())
                if permission in resource_perms:
                    self.authz_stats['granted_permissions'] += 1
                    return True
            
            # Would implement actual permission checking logic here
            # This is a simplified version
            
            self.authz_stats['denied_permissions'] += 1
            return False
            
        except Exception as e:
            self.authz_stats['denied_permissions'] += 1
            return False
    
    async def _load_resource_permissions(self):
        """Load resource permissions from file"""
        try:
            if self.resource_permissions_file.exists():
                with open(self.resource_permissions_file, 'r') as f:
                    data = json.load(f)
                
                for resource, user_perms in data.items():
                    self.resource_permissions[resource] = {}
                    for user_id, perms in user_perms.items():
                        self.resource_permissions[resource][user_id] = {Permission(p) for p in perms}
        except Exception as e:
            print(f"Error loading resource permissions: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get authorization statistics"""
        return self.authz_stats.copy()