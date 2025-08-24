"""
ABOV3 Genesis Permission Manager
Manages user permissions and consent for various operations
"""

import asyncio
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from enum import Enum

class PermissionType(Enum):
    FILE_CREATE = "file_create"
    FILE_MODIFY = "file_modify"
    FILE_DELETE = "file_delete"
    PACKAGE_INSTALL = "package_install"
    SHELL_COMMAND = "shell_command"
    NETWORK_ACCESS = "network_access"
    SYSTEM_CHANGE = "system_change"

class PermissionLevel(Enum):
    ALWAYS_ASK = "always_ask"
    ALLOW_ONCE = "allow_once"
    ALLOW_SESSION = "allow_session"
    ALWAYS_ALLOW = "always_allow"
    ALWAYS_DENY = "always_deny"

class PermissionManager:
    """
    Permission Manager for ABOV3 Genesis
    Handles user consent and permission validation
    """
    
    def __init__(self, project_path: Path):
        self.project_path = Path(project_path)
        self.abov3_dir = self.project_path / '.abov3'
        self.permissions_dir = self.abov3_dir / 'permissions'
        self.preferences_file = self.permissions_dir / 'preferences.yaml'
        
        # Ensure directories exist
        self.permissions_dir.mkdir(parents=True, exist_ok=True)
        
        # Permission preferences
        self.preferences = {}
        
        # Session permissions (temporary for current session)
        self.session_permissions = {}
        
        # Consent callback function
        self.consent_callback: Optional[Callable] = None
        
        # Load preferences
        self.load_preferences()
    
    def load_preferences(self):
        """Load permission preferences"""
        if self.preferences_file.exists():
            try:
                with open(self.preferences_file, 'r') as f:
                    self.preferences = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Error loading permission preferences: {e}")
                self.preferences = {}
        
        # Ensure default structure
        if not self.preferences:
            self.preferences = {
                'version': '1.0.0',
                'created': datetime.now().isoformat(),
                'permissions': {},
                'global_settings': {
                    'safe_mode': True,
                    'auto_confirm_safe_operations': False,
                    'prompt_timeout': 30
                }
            }
            self.save_preferences()
    
    def save_preferences(self):
        """Save permission preferences"""
        try:
            self.preferences['last_updated'] = datetime.now().isoformat()
            with open(self.preferences_file, 'w') as f:
                yaml.dump(self.preferences, f, default_flow_style=False)
        except Exception as e:
            print(f"Error saving permission preferences: {e}")
    
    def set_consent_callback(self, callback: Callable):
        """Set the callback function for requesting user consent"""
        self.consent_callback = callback
    
    async def request_permission(
        self,
        permission_type: PermissionType,
        description: str,
        details: Dict[str, Any] = None,
        danger_level: str = "medium"
    ) -> bool:
        """Request permission for an operation"""
        
        # Check existing preferences
        permission_key = self._get_permission_key(permission_type, details)
        
        # Check if we have a stored preference
        stored_level = self._get_stored_permission(permission_key)
        if stored_level:
            return self._evaluate_permission(stored_level)
        
        # Check session permissions
        if permission_key in self.session_permissions:
            return self.session_permissions[permission_key]
        
        # Safe mode checks
        if self.preferences.get('global_settings', {}).get('safe_mode', True):
            if self._is_dangerous_operation(permission_type, details, danger_level):
                return await self._request_user_consent(
                    permission_type, description, details, danger_level
                )
        
        # Auto-confirm safe operations if enabled
        if (self.preferences.get('global_settings', {}).get('auto_confirm_safe_operations', False) and
            danger_level == "low"):
            return True
        
        # Request user consent
        return await self._request_user_consent(
            permission_type, description, details, danger_level
        )
    
    def _get_permission_key(self, permission_type: PermissionType, details: Dict[str, Any] = None) -> str:
        """Generate a key for permission storage"""
        base_key = permission_type.value
        
        if details:
            # Add specific context to the key
            if permission_type == PermissionType.FILE_CREATE:
                if 'file_extension' in details:
                    base_key += f"_{details['file_extension']}"
            elif permission_type == PermissionType.PACKAGE_INSTALL:
                if 'package_manager' in details:
                    base_key += f"_{details['package_manager']}"
            elif permission_type == PermissionType.SHELL_COMMAND:
                if 'command' in details:
                    # Use first word of command for grouping
                    cmd_word = details['command'].split()[0] if details['command'] else 'unknown'
                    base_key += f"_{cmd_word}"
        
        return base_key
    
    def _get_stored_permission(self, permission_key: str) -> Optional[PermissionLevel]:
        """Get stored permission level for a key"""
        permissions = self.preferences.get('permissions', {})
        if permission_key in permissions:
            level_str = permissions[permission_key].get('level')
            if level_str:
                try:
                    return PermissionLevel(level_str)
                except ValueError:
                    pass
        return None
    
    def _evaluate_permission(self, level: PermissionLevel) -> bool:
        """Evaluate permission based on level"""
        if level == PermissionLevel.ALWAYS_ALLOW:
            return True
        elif level == PermissionLevel.ALWAYS_DENY:
            return False
        elif level == PermissionLevel.ALLOW_SESSION:
            # This should trigger a new consent request if not in session
            return False
        else:
            # ALWAYS_ASK or ALLOW_ONCE should trigger new consent
            return False
    
    def _is_dangerous_operation(
        self, 
        permission_type: PermissionType, 
        details: Dict[str, Any] = None,
        danger_level: str = "medium"
    ) -> bool:
        """Check if an operation is considered dangerous"""
        
        # High danger operations always require confirmation
        if danger_level == "high":
            return True
        
        # Specific dangerous patterns
        dangerous_patterns = {
            PermissionType.FILE_DELETE: True,
            PermissionType.SHELL_COMMAND: self._is_dangerous_command(details),
            PermissionType.SYSTEM_CHANGE: True,
            PermissionType.PACKAGE_INSTALL: self._is_dangerous_package(details),
            PermissionType.NETWORK_ACCESS: danger_level != "low"
        }
        
        return dangerous_patterns.get(permission_type, False)
    
    def _is_dangerous_command(self, details: Dict[str, Any] = None) -> bool:
        """Check if a shell command is dangerous"""
        if not details or 'command' not in details:
            return True
        
        command = details['command'].lower()
        dangerous_commands = [
            'rm ', 'del ', 'rmdir', 'rd ', 'format', 'fdisk',
            'sudo ', 'su ', 'chmod 777', 'chown root',
            'kill ', 'killall', 'pkill', 'shutdown', 'reboot',
            'dd ', 'mkfs', 'fsck'
        ]
        
        return any(dangerous_cmd in command for dangerous_cmd in dangerous_commands)
    
    def _is_dangerous_package(self, details: Dict[str, Any] = None) -> bool:
        """Check if a package installation is dangerous"""
        if not details:
            return False
        
        # System-level package managers are more dangerous
        dangerous_managers = ['apt', 'yum', 'dnf', 'pacman', 'brew']
        package_manager = details.get('package_manager', '').lower()
        
        return package_manager in dangerous_managers
    
    async def _request_user_consent(
        self,
        permission_type: PermissionType,
        description: str,
        details: Dict[str, Any] = None,
        danger_level: str = "medium"
    ) -> bool:
        """Request consent from user via callback"""
        if not self.consent_callback:
            # No callback available, use default safe behavior
            return danger_level == "low"
        
        try:
            # Prepare consent request
            consent_request = {
                'type': permission_type.value,
                'description': description,
                'details': details or {},
                'danger_level': danger_level,
                'timestamp': datetime.now().isoformat()
            }
            
            # Call consent callback
            result = await self.consent_callback(consent_request)
            
            # Handle result
            if isinstance(result, dict):
                granted = result.get('granted', False)
                remember_choice = result.get('remember', False)
                
                if remember_choice:
                    level = PermissionLevel.ALWAYS_ALLOW if granted else PermissionLevel.ALWAYS_DENY
                    self._store_permission(permission_type, details, level)
                else:
                    # Store for session only
                    permission_key = self._get_permission_key(permission_type, details)
                    self.session_permissions[permission_key] = granted
                
                return granted
            else:
                # Simple boolean result
                return bool(result)
                
        except Exception as e:
            print(f"Error in consent callback: {e}")
            return False
    
    def _store_permission(
        self,
        permission_type: PermissionType,
        details: Dict[str, Any] = None,
        level: PermissionLevel = PermissionLevel.ALWAYS_ASK
    ):
        """Store a permission preference"""
        permission_key = self._get_permission_key(permission_type, details)
        
        if 'permissions' not in self.preferences:
            self.preferences['permissions'] = {}
        
        self.preferences['permissions'][permission_key] = {
            'level': level.value,
            'type': permission_type.value,
            'created': datetime.now().isoformat(),
            'details': details or {}
        }
        
        self.save_preferences()
    
    def grant_permission(
        self,
        permission_type: PermissionType,
        level: PermissionLevel = PermissionLevel.ALLOW_SESSION,
        details: Dict[str, Any] = None
    ):
        """Manually grant a permission"""
        if level == PermissionLevel.ALLOW_SESSION:
            permission_key = self._get_permission_key(permission_type, details)
            self.session_permissions[permission_key] = True
        else:
            self._store_permission(permission_type, details, level)
    
    def revoke_permission(
        self,
        permission_type: PermissionType,
        details: Dict[str, Any] = None
    ):
        """Revoke a permission"""
        permission_key = self._get_permission_key(permission_type, details)
        
        # Remove from session permissions
        if permission_key in self.session_permissions:
            del self.session_permissions[permission_key]
        
        # Remove from stored permissions
        if 'permissions' in self.preferences and permission_key in self.preferences['permissions']:
            del self.preferences['permissions'][permission_key]
            self.save_preferences()
    
    def list_permissions(self) -> Dict[str, Any]:
        """List all permission preferences"""
        return {
            'stored_permissions': self.preferences.get('permissions', {}),
            'session_permissions': self.session_permissions,
            'global_settings': self.preferences.get('global_settings', {})
        }
    
    def clear_session_permissions(self):
        """Clear all session permissions"""
        self.session_permissions.clear()
    
    def clear_all_permissions(self):
        """Clear all stored permissions"""
        self.preferences['permissions'] = {}
        self.session_permissions.clear()
        self.save_preferences()
    
    def set_global_setting(self, key: str, value: Any):
        """Set a global permission setting"""
        if 'global_settings' not in self.preferences:
            self.preferences['global_settings'] = {}
        
        self.preferences['global_settings'][key] = value
        self.save_preferences()
    
    def get_global_setting(self, key: str, default: Any = None) -> Any:
        """Get a global permission setting"""
        return self.preferences.get('global_settings', {}).get(key, default)
    
    def export_permissions(self) -> Dict[str, Any]:
        """Export permission preferences"""
        return {
            'preferences': self.preferences,
            'session_permissions': self.session_permissions,
            'exported': datetime.now().isoformat()
        }
    
    def import_permissions(self, data: Dict[str, Any]) -> bool:
        """Import permission preferences"""
        try:
            if 'preferences' in data:
                self.preferences = data['preferences']
                self.save_preferences()
            
            if 'session_permissions' in data:
                self.session_permissions = data['session_permissions']
            
            return True
        except Exception as e:
            print(f"Error importing permissions: {e}")
            return False
    
    def get_permission_stats(self) -> Dict[str, Any]:
        """Get permission statistics"""
        stored_perms = self.preferences.get('permissions', {})
        
        # Count by type
        type_counts = {}
        level_counts = {}
        
        for perm_data in stored_perms.values():
            perm_type = perm_data.get('type', 'unknown')
            perm_level = perm_data.get('level', 'unknown')
            
            type_counts[perm_type] = type_counts.get(perm_type, 0) + 1
            level_counts[perm_level] = level_counts.get(perm_level, 0) + 1
        
        return {
            'total_stored': len(stored_perms),
            'session_permissions': len(self.session_permissions),
            'type_counts': type_counts,
            'level_counts': level_counts,
            'safe_mode': self.get_global_setting('safe_mode', True),
            'auto_confirm_safe': self.get_global_setting('auto_confirm_safe_operations', False)
        }
    
    async def check_file_operation_permission(
        self,
        operation: str,
        file_path: str,
        danger_level: str = "medium"
    ) -> bool:
        """Helper method for file operation permissions"""
        operation_map = {
            'create': PermissionType.FILE_CREATE,
            'modify': PermissionType.FILE_MODIFY,
            'delete': PermissionType.FILE_DELETE
        }
        
        perm_type = operation_map.get(operation)
        if not perm_type:
            return False
        
        # Determine file extension for context
        file_ext = Path(file_path).suffix
        
        details = {
            'file_path': file_path,
            'file_extension': file_ext
        }
        
        description = f"{operation.capitalize()} file: {file_path}"
        
        return await self.request_permission(perm_type, description, details, danger_level)
    
    async def check_shell_command_permission(
        self,
        command: str,
        danger_level: str = "medium"
    ) -> bool:
        """Helper method for shell command permissions"""
        details = {'command': command}
        description = f"Run shell command: {command}"
        
        return await self.request_permission(
            PermissionType.SHELL_COMMAND, description, details, danger_level
        )
    
    async def check_package_install_permission(
        self,
        package_name: str,
        package_manager: str = "pip",
        danger_level: str = "medium"
    ) -> bool:
        """Helper method for package installation permissions"""
        details = {
            'package_name': package_name,
            'package_manager': package_manager
        }
        description = f"Install package: {package_name} via {package_manager}"
        
        return await self.request_permission(
            PermissionType.PACKAGE_INSTALL, description, details, danger_level
        )