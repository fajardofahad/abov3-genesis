"""
ABOV3 Genesis - Development Environment Infrastructure
Auto-setup, dependency management, and environment configuration
"""

import asyncio
import os
import sys
import subprocess
import platform
import shutil
import json
import tempfile
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
import re
import hashlib
import time
from collections import defaultdict
import aiofiles
import yaml
import toml

logger = logging.getLogger(__name__)

class EnvironmentType(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

class PackageManager(Enum):
    """Package managers"""
    PIP = "pip"
    CONDA = "conda"
    NPM = "npm"
    YARN = "yarn"
    PNPM = "pnpm"
    POETRY = "poetry"
    PIPENV = "pipenv"

class PlatformType(Enum):
    """Platform types"""
    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"
    DOCKER = "docker"

@dataclass
class DependencySpec:
    """Dependency specification"""
    name: str
    version: Optional[str] = None
    extras: List[str] = field(default_factory=list)
    source: Optional[str] = None  # PyPI, conda-forge, etc.
    optional: bool = False
    dev_only: bool = False
    platform_specific: Optional[str] = None  # 'windows', 'linux', 'macos'
    python_version_constraint: Optional[str] = None
    environment_variables: Dict[str, str] = field(default_factory=dict)

@dataclass
class EnvironmentConfig:
    """Environment configuration"""
    name: str
    environment_type: EnvironmentType
    python_version: str = "3.9+"
    package_managers: List[PackageManager] = field(default_factory=lambda: [PackageManager.PIP])
    dependencies: List[DependencySpec] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    system_requirements: List[str] = field(default_factory=list)
    pre_install_commands: List[str] = field(default_factory=list)
    post_install_commands: List[str] = field(default_factory=list)
    health_checks: List[str] = field(default_factory=list)
    config_templates: Dict[str, str] = field(default_factory=dict)

class SystemDetector:
    """
    Detect system capabilities and requirements
    """

    @staticmethod
    def get_platform() -> PlatformType:
        """Detect platform type"""
        system = platform.system().lower()
        
        if system == "windows":
            return PlatformType.WINDOWS
        elif system == "darwin":
            return PlatformType.MACOS
        elif system == "linux":
            return PlatformType.LINUX
        else:
            # Default to Linux for unknown systems
            return PlatformType.LINUX

    @staticmethod
    def get_python_version() -> str:
        """Get current Python version"""
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get comprehensive system information"""
        return {
            'platform': SystemDetector.get_platform().value,
            'python_version': SystemDetector.get_python_version(),
            'python_executable': sys.executable,
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'node_info': platform.node(),
            'cpu_count': os.cpu_count(),
            'available_package_managers': SystemDetector.detect_package_managers()
        }

    @staticmethod
    def detect_package_managers() -> List[str]:
        """Detect available package managers"""
        managers = []
        
        # Python package managers
        if shutil.which('pip'):
            managers.append('pip')
        if shutil.which('conda'):
            managers.append('conda')
        if shutil.which('poetry'):
            managers.append('poetry')
        if shutil.which('pipenv'):
            managers.append('pipenv')
        
        # Node.js package managers
        if shutil.which('npm'):
            managers.append('npm')
        if shutil.which('yarn'):
            managers.append('yarn')
        if shutil.which('pnpm'):
            managers.append('pnpm')
        
        return managers

    @staticmethod
    def check_system_requirements(requirements: List[str]) -> Dict[str, bool]:
        """Check if system requirements are met"""
        results = {}
        
        for requirement in requirements:
            if requirement.startswith('command:'):
                # Check if command exists
                command = requirement[8:]  # Remove 'command:' prefix
                results[requirement] = shutil.which(command) is not None
            
            elif requirement.startswith('python_version:'):
                # Check Python version
                version_spec = requirement[15:]  # Remove 'python_version:' prefix
                results[requirement] = SystemDetector._check_python_version(version_spec)
            
            elif requirement.startswith('env_var:'):
                # Check environment variable
                var_name = requirement[8:]  # Remove 'env_var:' prefix
                results[requirement] = var_name in os.environ
            
            elif requirement.startswith('port:'):
                # Check if port is available
                port = int(requirement[5:])  # Remove 'port:' prefix
                results[requirement] = SystemDetector._check_port_available(port)
            
            else:
                # Assume it's a command check
                results[requirement] = shutil.which(requirement) is not None
        
        return results

    @staticmethod
    def _check_python_version(version_spec: str) -> bool:
        """Check if Python version meets specification"""
        import packaging.version
        import packaging.specifiers
        
        try:
            spec_set = packaging.specifiers.SpecifierSet(version_spec)
            current_version = packaging.version.Version(SystemDetector.get_python_version())
            return current_version in spec_set
        except Exception:
            return False

    @staticmethod
    def _check_port_available(port: int) -> bool:
        """Check if port is available"""
        import socket
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                return result != 0  # Port is available if connection fails
        except Exception:
            return False

class DependencyManager:
    """
    Advanced dependency management with conflict resolution
    """

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.platform = SystemDetector.get_platform()
        self.available_managers = SystemDetector.detect_package_managers()
        
        # Dependency resolution cache
        self._resolution_cache = {}
        self._lock_file_path = project_path / '.abov3' / 'dependencies.lock'
        
        # Installation history
        self._installation_history = []

    async def resolve_dependencies(
        self,
        dependencies: List[DependencySpec],
        environment_type: EnvironmentType = EnvironmentType.DEVELOPMENT
    ) -> Dict[str, Any]:
        """Resolve dependencies and check for conflicts"""
        
        # Filter dependencies by platform and environment
        filtered_deps = self._filter_dependencies(dependencies, environment_type)
        
        # Group by package manager
        manager_groups = self._group_by_manager(filtered_deps)
        
        # Resolve each group
        resolution_results = {}
        
        for manager, deps in manager_groups.items():
            try:
                result = await self._resolve_manager_dependencies(manager, deps)
                resolution_results[manager.value] = result
            except Exception as e:
                logger.error(f"Failed to resolve {manager.value} dependencies: {e}")
                resolution_results[manager.value] = {'error': str(e)}
        
        return {
            'platform': self.platform.value,
            'environment_type': environment_type.value,
            'resolution_results': resolution_results,
            'total_dependencies': len(filtered_deps),
            'conflicts': self._detect_conflicts(resolution_results)
        }

    def _filter_dependencies(
        self,
        dependencies: List[DependencySpec],
        environment_type: EnvironmentType
    ) -> List[DependencySpec]:
        """Filter dependencies by platform and environment"""
        filtered = []
        
        for dep in dependencies:
            # Check platform compatibility
            if dep.platform_specific and dep.platform_specific != self.platform.value:
                continue
            
            # Check environment type
            if dep.dev_only and environment_type == EnvironmentType.PRODUCTION:
                continue
            
            # Check Python version constraint
            if dep.python_version_constraint:
                if not SystemDetector._check_python_version(dep.python_version_constraint):
                    logger.warning(f"Skipping {dep.name}: Python version constraint not met")
                    continue
            
            filtered.append(dep)
        
        return filtered

    def _group_by_manager(self, dependencies: List[DependencySpec]) -> Dict[PackageManager, List[DependencySpec]]:
        """Group dependencies by package manager"""
        groups = defaultdict(list)
        
        for dep in dependencies:
            # Determine best package manager for this dependency
            manager = self._select_package_manager(dep)
            if manager:
                groups[manager].append(dep)
        
        return dict(groups)

    def _select_package_manager(self, dep: DependencySpec) -> Optional[PackageManager]:
        """Select best available package manager for dependency"""
        
        # Check if specific source is requested
        if dep.source:
            if 'conda' in dep.source and 'conda' in self.available_managers:
                return PackageManager.CONDA
            elif 'npm' in dep.source and 'npm' in self.available_managers:
                return PackageManager.NPM
        
        # Default to pip for Python packages
        if 'pip' in self.available_managers:
            return PackageManager.PIP
        
        # Fallback to conda
        if 'conda' in self.available_managers:
            return PackageManager.CONDA
        
        return None

    async def _resolve_manager_dependencies(
        self,
        manager: PackageManager,
        dependencies: List[DependencySpec]
    ) -> Dict[str, Any]:
        """Resolve dependencies for specific package manager"""
        
        if manager == PackageManager.PIP:
            return await self._resolve_pip_dependencies(dependencies)
        elif manager == PackageManager.CONDA:
            return await self._resolve_conda_dependencies(dependencies)
        elif manager == PackageManager.NPM:
            return await self._resolve_npm_dependencies(dependencies)
        else:
            return {'error': f'Unsupported package manager: {manager.value}'}

    async def _resolve_pip_dependencies(self, dependencies: List[DependencySpec]) -> Dict[str, Any]:
        """Resolve pip dependencies"""
        try:
            # Create requirements list
            requirements = []
            for dep in dependencies:
                req_str = dep.name
                if dep.version:
                    req_str += f"=={dep.version}"
                if dep.extras:
                    req_str += f"[{','.join(dep.extras)}]"
                requirements.append(req_str)
            
            # Use pip-tools or similar for resolution (simplified here)
            return {
                'manager': 'pip',
                'requirements': requirements,
                'resolved': True,
                'installation_command': f'pip install {" ".join(requirements)}'
            }
            
        except Exception as e:
            return {'error': str(e)}

    async def _resolve_conda_dependencies(self, dependencies: List[DependencySpec]) -> Dict[str, Any]:
        """Resolve conda dependencies"""
        try:
            requirements = []
            for dep in dependencies:
                req_str = dep.name
                if dep.version:
                    req_str += f"={dep.version}"
                requirements.append(req_str)
            
            return {
                'manager': 'conda',
                'requirements': requirements,
                'resolved': True,
                'installation_command': f'conda install {" ".join(requirements)}'
            }
            
        except Exception as e:
            return {'error': str(e)}

    async def _resolve_npm_dependencies(self, dependencies: List[DependencySpec]) -> Dict[str, Any]:
        """Resolve npm dependencies"""
        try:
            requirements = []
            for dep in dependencies:
                req_str = dep.name
                if dep.version:
                    req_str += f"@{dep.version}"
                requirements.append(req_str)
            
            return {
                'manager': 'npm',
                'requirements': requirements,
                'resolved': True,
                'installation_command': f'npm install {" ".join(requirements)}'
            }
            
        except Exception as e:
            return {'error': str(e)}

    def _detect_conflicts(self, resolution_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect dependency conflicts"""
        conflicts = []
        
        # Simple conflict detection (could be enhanced)
        all_packages = {}
        
        for manager, result in resolution_results.items():
            if 'requirements' in result:
                for req in result['requirements']:
                    package_name = req.split('==')[0].split('=')[0].split('@')[0]
                    
                    if package_name in all_packages:
                        conflicts.append({
                            'package': package_name,
                            'managers': [all_packages[package_name], manager],
                            'type': 'duplicate_package'
                        })
                    else:
                        all_packages[package_name] = manager
        
        return conflicts

    async def install_dependencies(
        self,
        resolution_result: Dict[str, Any],
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Install resolved dependencies"""
        
        installation_results = {}
        total_installed = 0
        total_failed = 0
        
        for manager, result in resolution_result['resolution_results'].items():
            if 'error' in result:
                installation_results[manager] = result
                continue
            
            try:
                install_result = await self._install_manager_dependencies(
                    PackageManager(manager),
                    result,
                    dry_run
                )
                
                installation_results[manager] = install_result
                
                if install_result.get('success'):
                    total_installed += install_result.get('installed_count', 0)
                else:
                    total_failed += 1
                    
            except Exception as e:
                logger.error(f"Failed to install {manager} dependencies: {e}")
                installation_results[manager] = {'error': str(e)}
                total_failed += 1
        
        return {
            'installation_results': installation_results,
            'total_installed': total_installed,
            'total_failed': total_failed,
            'dry_run': dry_run
        }

    async def _install_manager_dependencies(
        self,
        manager: PackageManager,
        resolution: Dict[str, Any],
        dry_run: bool
    ) -> Dict[str, Any]:
        """Install dependencies for specific manager"""
        
        command = resolution.get('installation_command')
        if not command:
            return {'error': 'No installation command available'}
        
        if dry_run:
            return {
                'success': True,
                'dry_run': True,
                'command': command,
                'installed_count': len(resolution.get('requirements', []))
            }
        
        try:
            # Execute installation command
            result = await self._execute_command(command)
            
            self._installation_history.append({
                'timestamp': time.time(),
                'manager': manager.value,
                'command': command,
                'success': result['success'],
                'output': result.get('output', ''),
                'error': result.get('error', '')
            })
            
            return {
                'success': result['success'],
                'command': command,
                'output': result.get('output', ''),
                'error': result.get('error', ''),
                'installed_count': len(resolution.get('requirements', []))
            }
            
        except Exception as e:
            return {'error': str(e)}

    async def _execute_command(self, command: str) -> Dict[str, Any]:
        """Execute shell command asynchronously"""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_path
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                'success': process.returncode == 0,
                'returncode': process.returncode,
                'output': stdout.decode('utf-8'),
                'error': stderr.decode('utf-8')
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    async def save_lock_file(self, resolution_result: Dict[str, Any]):
        """Save dependency lock file"""
        try:
            lock_data = {
                'generated_at': time.time(),
                'platform': resolution_result.get('platform'),
                'environment_type': resolution_result.get('environment_type'),
                'dependencies': resolution_result.get('resolution_results', {}),
                'system_info': SystemDetector.get_system_info()
            }
            
            # Ensure directory exists
            self._lock_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            async with aiofiles.open(self._lock_file_path, 'w') as f:
                await f.write(json.dumps(lock_data, indent=2))
                
            logger.info(f"Saved dependency lock file: {self._lock_file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save lock file: {e}")

    async def load_lock_file(self) -> Optional[Dict[str, Any]]:
        """Load dependency lock file"""
        try:
            if not self._lock_file_path.exists():
                return None
            
            async with aiofiles.open(self._lock_file_path, 'r') as f:
                content = await f.read()
                return json.loads(content)
                
        except Exception as e:
            logger.error(f"Failed to load lock file: {e}")
            return None

class EnvironmentSetup:
    """
    Complete environment setup and configuration
    """

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.dependency_manager = DependencyManager(project_path)
        
        # Setup configurations
        self._config_templates = {}
        self._setup_history = []

    async def setup_environment(
        self,
        config: EnvironmentConfig,
        force: bool = False
    ) -> Dict[str, Any]:
        """Setup complete development environment"""
        
        logger.info(f"Setting up {config.environment_type.value} environment: {config.name}")
        
        setup_results = {
            'environment_name': config.name,
            'environment_type': config.environment_type.value,
            'started_at': time.time(),
            'steps': []
        }
        
        try:
            # Step 1: System requirements check
            step_result = await self._check_system_requirements(config)
            setup_results['steps'].append(step_result)
            
            if not step_result['success'] and not force:
                setup_results['success'] = False
                setup_results['error'] = 'System requirements not met'
                return setup_results
            
            # Step 2: Pre-install commands
            if config.pre_install_commands:
                step_result = await self._execute_pre_install_commands(config)
                setup_results['steps'].append(step_result)
            
            # Step 3: Dependency resolution
            step_result = await self._resolve_and_install_dependencies(config)
            setup_results['steps'].append(step_result)
            
            # Step 4: Environment variables
            step_result = await self._setup_environment_variables(config)
            setup_results['steps'].append(step_result)
            
            # Step 5: Configuration files
            step_result = await self._create_config_files(config)
            setup_results['steps'].append(step_result)
            
            # Step 6: Post-install commands
            if config.post_install_commands:
                step_result = await self._execute_post_install_commands(config)
                setup_results['steps'].append(step_result)
            
            # Step 7: Health checks
            step_result = await self._run_health_checks(config)
            setup_results['steps'].append(step_result)
            
            # Summary
            setup_results['success'] = all(step.get('success', False) for step in setup_results['steps'])
            setup_results['completed_at'] = time.time()
            setup_results['duration'] = setup_results['completed_at'] - setup_results['started_at']
            
            # Save setup history
            self._setup_history.append(setup_results)
            
            await self._save_environment_info(config, setup_results)
            
            return setup_results
            
        except Exception as e:
            logger.error(f"Environment setup failed: {e}")
            setup_results['success'] = False
            setup_results['error'] = str(e)
            return setup_results

    async def _check_system_requirements(self, config: EnvironmentConfig) -> Dict[str, Any]:
        """Check system requirements"""
        logger.info("Checking system requirements...")
        
        # Check Python version
        python_check = SystemDetector._check_python_version(config.python_version)
        
        # Check system requirements
        system_checks = SystemDetector.check_system_requirements(config.system_requirements)
        
        # Check available package managers
        required_managers = [pm.value for pm in config.package_managers]
        available_managers = SystemDetector.detect_package_managers()
        
        manager_checks = {}
        for manager in required_managers:
            manager_checks[manager] = manager in available_managers
        
        all_passed = (
            python_check and
            all(system_checks.values()) and
            all(manager_checks.values())
        )
        
        return {
            'step': 'system_requirements',
            'success': all_passed,
            'python_version_check': python_check,
            'system_requirements': system_checks,
            'package_managers': manager_checks,
            'details': {
                'current_python': SystemDetector.get_python_version(),
                'required_python': config.python_version,
                'platform': SystemDetector.get_platform().value
            }
        }

    async def _execute_pre_install_commands(self, config: EnvironmentConfig) -> Dict[str, Any]:
        """Execute pre-install commands"""
        logger.info("Executing pre-install commands...")
        
        results = []
        
        for command in config.pre_install_commands:
            try:
                result = await self.dependency_manager._execute_command(command)
                results.append({
                    'command': command,
                    'success': result['success'],
                    'output': result.get('output', ''),
                    'error': result.get('error', '')
                })
            except Exception as e:
                results.append({
                    'command': command,
                    'success': False,
                    'error': str(e)
                })
        
        all_success = all(r['success'] for r in results)
        
        return {
            'step': 'pre_install_commands',
            'success': all_success,
            'commands': results,
            'total_commands': len(config.pre_install_commands)
        }

    async def _resolve_and_install_dependencies(self, config: EnvironmentConfig) -> Dict[str, Any]:
        """Resolve and install dependencies"""
        logger.info("Resolving and installing dependencies...")
        
        try:
            # Resolve dependencies
            resolution = await self.dependency_manager.resolve_dependencies(
                config.dependencies,
                config.environment_type
            )
            
            # Install dependencies
            installation = await self.dependency_manager.install_dependencies(resolution)
            
            # Save lock file
            await self.dependency_manager.save_lock_file(resolution)
            
            return {
                'step': 'dependencies',
                'success': installation['total_failed'] == 0,
                'resolution': resolution,
                'installation': installation,
                'total_dependencies': resolution['total_dependencies'],
                'conflicts': len(resolution['conflicts'])
            }
            
        except Exception as e:
            return {
                'step': 'dependencies',
                'success': False,
                'error': str(e)
            }

    async def _setup_environment_variables(self, config: EnvironmentConfig) -> Dict[str, Any]:
        """Setup environment variables"""
        logger.info("Setting up environment variables...")
        
        try:
            # Create .env file
            env_file_path = self.project_path / '.abov3' / '.env'
            env_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            env_content = []
            env_content.append(f"# ABOV3 Genesis Environment Variables")
            env_content.append(f"# Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}")
            env_content.append(f"# Environment: {config.environment_type.value}")
            env_content.append("")
            
            for key, value in config.environment_variables.items():
                env_content.append(f"{key}={value}")
                
                # Also set in current process for immediate use
                os.environ[key] = value
            
            # Add dependency-specific environment variables
            for dep in config.dependencies:
                for key, value in dep.environment_variables.items():
                    env_content.append(f"{key}={value}")
                    os.environ[key] = value
            
            async with aiofiles.open(env_file_path, 'w') as f:
                await f.write('\n'.join(env_content))
            
            return {
                'step': 'environment_variables',
                'success': True,
                'env_file': str(env_file_path),
                'variables_set': len(config.environment_variables),
                'total_variables': len(env_content) - 4  # Exclude header lines
            }
            
        except Exception as e:
            return {
                'step': 'environment_variables',
                'success': False,
                'error': str(e)
            }

    async def _create_config_files(self, config: EnvironmentConfig) -> Dict[str, Any]:
        """Create configuration files from templates"""
        logger.info("Creating configuration files...")
        
        try:
            created_files = []
            
            for file_path, template_content in config.config_templates.items():
                full_path = self.project_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Process template (simple variable substitution)
                processed_content = self._process_template(template_content, config)
                
                async with aiofiles.open(full_path, 'w') as f:
                    await f.write(processed_content)
                
                created_files.append(str(full_path))
            
            return {
                'step': 'config_files',
                'success': True,
                'created_files': created_files,
                'total_files': len(config.config_templates)
            }
            
        except Exception as e:
            return {
                'step': 'config_files',
                'success': False,
                'error': str(e)
            }

    def _process_template(self, template: str, config: EnvironmentConfig) -> str:
        """Process configuration template"""
        replacements = {
            '{{PROJECT_PATH}}': str(self.project_path),
            '{{ENVIRONMENT_TYPE}}': config.environment_type.value,
            '{{ENVIRONMENT_NAME}}': config.name,
            '{{PYTHON_VERSION}}': config.python_version,
            '{{PLATFORM}}': SystemDetector.get_platform().value
        }
        
        # Add environment variables to replacements
        for key, value in config.environment_variables.items():
            replacements[f'{{ENV.{key}}}'] = value
        
        # Simple template processing
        result = template
        for placeholder, value in replacements.items():
            result = result.replace(placeholder, value)
        
        return result

    async def _execute_post_install_commands(self, config: EnvironmentConfig) -> Dict[str, Any]:
        """Execute post-install commands"""
        logger.info("Executing post-install commands...")
        
        results = []
        
        for command in config.post_install_commands:
            try:
                result = await self.dependency_manager._execute_command(command)
                results.append({
                    'command': command,
                    'success': result['success'],
                    'output': result.get('output', ''),
                    'error': result.get('error', '')
                })
            except Exception as e:
                results.append({
                    'command': command,
                    'success': False,
                    'error': str(e)
                })
        
        all_success = all(r['success'] for r in results)
        
        return {
            'step': 'post_install_commands',
            'success': all_success,
            'commands': results,
            'total_commands': len(config.post_install_commands)
        }

    async def _run_health_checks(self, config: EnvironmentConfig) -> Dict[str, Any]:
        """Run environment health checks"""
        logger.info("Running health checks...")
        
        try:
            check_results = []
            
            for check_command in config.health_checks:
                try:
                    result = await self.dependency_manager._execute_command(check_command)
                    check_results.append({
                        'check': check_command,
                        'success': result['success'],
                        'output': result.get('output', ''),
                        'error': result.get('error', '')
                    })
                except Exception as e:
                    check_results.append({
                        'check': check_command,
                        'success': False,
                        'error': str(e)
                    })
            
            all_healthy = all(r['success'] for r in check_results)
            
            return {
                'step': 'health_checks',
                'success': all_healthy,
                'checks': check_results,
                'total_checks': len(config.health_checks),
                'health_status': 'healthy' if all_healthy else 'unhealthy'
            }
            
        except Exception as e:
            return {
                'step': 'health_checks',
                'success': False,
                'error': str(e)
            }

    async def _save_environment_info(self, config: EnvironmentConfig, setup_results: Dict[str, Any]):
        """Save environment information"""
        try:
            env_info = {
                'config': {
                    'name': config.name,
                    'environment_type': config.environment_type.value,
                    'python_version': config.python_version,
                    'package_managers': [pm.value for pm in config.package_managers]
                },
                'setup_results': setup_results,
                'system_info': SystemDetector.get_system_info(),
                'created_at': time.time()
            }
            
            info_path = self.project_path / '.abov3' / 'environment.json'
            info_path.parent.mkdir(parents=True, exist_ok=True)
            
            async with aiofiles.open(info_path, 'w') as f:
                await f.write(json.dumps(env_info, indent=2))
                
            logger.info(f"Saved environment info: {info_path}")
            
        except Exception as e:
            logger.error(f"Failed to save environment info: {e}")

class EnvironmentManager:
    """
    Main environment management coordinator
    """

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.setup = EnvironmentSetup(project_path)
        
        # Predefined environment configurations
        self._predefined_configs = self._create_predefined_configs()

    def _create_predefined_configs(self) -> Dict[str, EnvironmentConfig]:
        """Create predefined environment configurations"""
        
        # Base ABOV3 dependencies
        base_deps = [
            DependencySpec(name="ollama", version=">=0.2.0"),
            DependencySpec(name="prompt_toolkit", version=">=3.0.36"),
            DependencySpec(name="rich", version=">=13.5.0"),
            DependencySpec(name="pygments", version=">=2.15.0"),
            DependencySpec(name="click", version=">=8.1.0"),
            DependencySpec(name="pyyaml", version=">=6.0.0"),
            DependencySpec(name="jinja2", version=">=3.1.0"),
            DependencySpec(name="aiofiles", version=">=23.0.0"),
            DependencySpec(name="psutil", version=">=5.9.0"),
            DependencySpec(name="gitpython", version=">=3.1.0"),
            DependencySpec(name="aiohttp", version=">=3.8.0"),
            DependencySpec(name="packaging", version=">=21.0")
        ]
        
        # Development dependencies
        dev_deps = [
            DependencySpec(name="pytest", version=">=7.0.0", dev_only=True),
            DependencySpec(name="pytest-asyncio", version=">=0.21.0", dev_only=True),
            DependencySpec(name="black", version=">=23.0.0", dev_only=True),
            DependencySpec(name="flake8", version=">=6.0.0", dev_only=True),
            DependencySpec(name="mypy", version=">=1.0.0", dev_only=True),
            DependencySpec(name="coverage", version=">=7.0.0", dev_only=True)
        ]
        
        configs = {
            'development': EnvironmentConfig(
                name='ABOV3 Genesis Development',
                environment_type=EnvironmentType.DEVELOPMENT,
                python_version='>=3.9',
                dependencies=base_deps + dev_deps,
                environment_variables={
                    'ABOV3_ENV': 'development',
                    'ABOV3_DEBUG': 'true',
                    'ABOV3_LOG_LEVEL': 'DEBUG'
                },
                system_requirements=[
                    'command:git',
                    'python_version:>=3.9'
                ],
                health_checks=[
                    'python --version',
                    'pip --version'
                ],
                config_templates={
                    '.abov3/config.yaml': '''# ABOV3 Genesis Configuration
environment: {{ENVIRONMENT_TYPE}}
debug: true
log_level: DEBUG
project_path: {{PROJECT_PATH}}
''',
                    'pyproject.toml': '''[tool.black]
line-length = 88
target-version = ['py39']

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
'''
                }
            ),
            
            'production': EnvironmentConfig(
                name='ABOV3 Genesis Production',
                environment_type=EnvironmentType.PRODUCTION,
                python_version='>=3.9',
                dependencies=base_deps,
                environment_variables={
                    'ABOV3_ENV': 'production',
                    'ABOV3_DEBUG': 'false',
                    'ABOV3_LOG_LEVEL': 'INFO'
                },
                system_requirements=[
                    'command:git',
                    'python_version:>=3.9',
                    'port:11434'  # Ollama port
                ],
                health_checks=[
                    'python --version',
                    'pip --version',
                    'ollama list'
                ],
                config_templates={
                    '.abov3/config.yaml': '''# ABOV3 Genesis Production Configuration
environment: {{ENVIRONMENT_TYPE}}
debug: false
log_level: INFO
project_path: {{PROJECT_PATH}}
'''
                }
            )
        }
        
        return configs

    async def setup_development_environment(self, force: bool = False) -> Dict[str, Any]:
        """Setup development environment"""
        config = self._predefined_configs['development']
        return await self.setup.setup_environment(config, force)

    async def setup_production_environment(self, force: bool = False) -> Dict[str, Any]:
        """Setup production environment"""
        config = self._predefined_configs['production']
        return await self.setup.setup_environment(config, force)

    async def setup_custom_environment(
        self,
        config: EnvironmentConfig,
        force: bool = False
    ) -> Dict[str, Any]:
        """Setup custom environment"""
        return await self.setup.setup_environment(config, force)

    async def get_environment_status(self) -> Dict[str, Any]:
        """Get current environment status"""
        try:
            info_path = self.project_path / '.abov3' / 'environment.json'
            
            if not info_path.exists():
                return {
                    'status': 'not_setup',
                    'message': 'Environment has not been set up'
                }
            
            async with aiofiles.open(info_path, 'r') as f:
                content = await f.read()
                env_info = json.loads(content)
            
            # Check if environment is healthy
            system_info = SystemDetector.get_system_info()
            
            return {
                'status': 'setup',
                'environment_info': env_info,
                'current_system': system_info,
                'last_setup': env_info.get('created_at'),
                'environment_type': env_info.get('config', {}).get('environment_type'),
                'environment_name': env_info.get('config', {}).get('name')
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    async def validate_environment(self) -> Dict[str, Any]:
        """Validate current environment"""
        status = await self.get_environment_status()
        
        if status['status'] != 'setup':
            return {
                'valid': False,
                'issues': ['Environment not set up']
            }
        
        issues = []
        
        # Check Python version
        env_info = status['environment_info']
        required_python = env_info.get('config', {}).get('python_version', '>=3.9')
        
        if not SystemDetector._check_python_version(required_python):
            issues.append(f'Python version does not meet requirement: {required_python}')
        
        # Check dependencies (simplified)
        lock_file_path = self.project_path / '.abov3' / 'dependencies.lock'
        if not lock_file_path.exists():
            issues.append('Dependency lock file not found')
        
        # Check Ollama availability
        if not shutil.which('ollama'):
            issues.append('Ollama command not found')
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'environment_type': env_info.get('config', {}).get('environment_type'),
            'last_validation': time.time()
        }

# Context manager for environment setup
class environment_context:
    """Context manager for environment setup"""
    
    def __init__(self, project_path: Path, environment_type: str = 'development'):
        self.manager = EnvironmentManager(project_path)
        self.environment_type = environment_type

    async def __aenter__(self):
        # Setup environment if not already done
        status = await self.manager.get_environment_status()
        
        if status['status'] != 'setup':
            logger.info(f"Setting up {self.environment_type} environment...")
            
            if self.environment_type == 'development':
                result = await self.manager.setup_development_environment()
            elif self.environment_type == 'production':
                result = await self.manager.setup_production_environment()
            else:
                raise ValueError(f"Unknown environment type: {self.environment_type}")
            
            if not result.get('success'):
                raise RuntimeError(f"Environment setup failed: {result.get('error')}")
        
        # Validate environment
        validation = await self.manager.validate_environment()
        if not validation['valid']:
            logger.warning(f"Environment validation issues: {validation['issues']}")
        
        return self.manager

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup if needed
        pass