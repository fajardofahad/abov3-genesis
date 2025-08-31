"""
ABOV3 Genesis - Secure Sandbox Executor
Enterprise-grade sandboxed code execution environment for secure debugging
"""

import asyncio
import os
import sys
import subprocess
import tempfile
import shutil
import threading
import time
import signal
import resource
import logging
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import uuid
import psutil
import docker
from docker.models.containers import Container

from .audit_logger import SecurityAuditLogger
from .crypto_manager import CryptographyManager


class SandboxType(Enum):
    """Types of sandbox environments"""
    PROCESS = "process"          # Process-level isolation
    CONTAINER = "container"      # Docker container isolation
    VM = "virtual_machine"       # Virtual machine isolation
    CHROOT = "chroot"           # Chroot jail isolation


class ExecutionMode(Enum):
    """Code execution modes"""
    INTERACTIVE = "interactive"   # Interactive debugging session
    BATCH = "batch"              # Batch execution
    STEP_BY_STEP = "step_by_step" # Step-by-step debugging
    READ_ONLY = "read_only"      # Read-only analysis


class SecurityProfile(Enum):
    """Security profile levels"""
    MINIMAL = "minimal"       # Basic sandboxing
    STANDARD = "standard"     # Standard security controls
    HIGH = "high"            # High security controls
    MAXIMUM = "maximum"      # Maximum security isolation


@dataclass
class SandboxLimits:
    """Resource limits for sandbox execution"""
    max_cpu_percent: float = 50.0      # Maximum CPU usage percentage
    max_memory_mb: int = 512           # Maximum memory in MB
    max_disk_mb: int = 100             # Maximum disk usage in MB
    max_execution_time: int = 300      # Maximum execution time in seconds
    max_network_connections: int = 0    # Maximum network connections (0 = none)
    max_file_descriptors: int = 64     # Maximum file descriptors
    max_processes: int = 5             # Maximum child processes
    max_threads: int = 10              # Maximum threads
    allowed_syscalls: Optional[Set[str]] = None  # Allowed system calls
    blocked_syscalls: Optional[Set[str]] = None  # Blocked system calls


@dataclass
class SandboxResult:
    """Result of sandbox execution"""
    execution_id: str
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    memory_used: int
    cpu_usage: float
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    security_violations: List[str] = field(default_factory=list)
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    network_activity: List[Dict[str, Any]] = field(default_factory=list)
    blocked_operations: List[str] = field(default_factory=list)


class SecureSandboxExecutor:
    """
    Enterprise-grade secure sandbox executor for debug code execution
    Provides multiple isolation levels and comprehensive security controls
    """
    
    def __init__(
        self,
        audit_logger: SecurityAuditLogger,
        crypto_manager: Optional[CryptographyManager] = None,
        temp_dir: Optional[Path] = None
    ):
        self.audit_logger = audit_logger
        self.crypto_manager = crypto_manager
        self.temp_dir = temp_dir or Path(tempfile.gettempdir()) / "abov3_sandbox"
        
        # Create sandbox directory
        self.sandbox_dir = self.temp_dir / "sandboxes"
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        
        # Docker client for container sandboxing
        self.docker_client = None
        self._init_docker_client()
        
        # Active sandboxes tracking
        self.active_sandboxes: Dict[str, Dict[str, Any]] = {}
        self.sandbox_locks: Dict[str, threading.Lock] = {}
        
        # Security profiles
        self.security_profiles = self._init_security_profiles()
        
        # Monitoring and metrics
        self.execution_metrics = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'security_violations': 0,
            'resource_violations': 0,
            'blocked_operations': 0
        }
        
        # Setup logging
        self.logger = logging.getLogger('abov3.security.sandbox')
        
        # Background monitoring
        self._monitor_task = None
        self._start_monitoring()
    
    def _init_docker_client(self):
        """Initialize Docker client for container sandboxing"""
        try:
            self.docker_client = docker.from_env()
            # Test Docker connection
            self.docker_client.ping()
            self.logger.info("Docker client initialized successfully")
        except Exception as e:
            self.logger.warning(f"Docker not available: {e}")
            self.docker_client = None
    
    def _init_security_profiles(self) -> Dict[SecurityProfile, SandboxLimits]:
        """Initialize predefined security profiles"""
        return {
            SecurityProfile.MINIMAL: SandboxLimits(
                max_cpu_percent=25.0,
                max_memory_mb=128,
                max_disk_mb=50,
                max_execution_time=60,
                max_network_connections=0,
                max_processes=2,
                max_threads=5
            ),
            SecurityProfile.STANDARD: SandboxLimits(
                max_cpu_percent=50.0,
                max_memory_mb=256,
                max_disk_mb=100,
                max_execution_time=300,
                max_network_connections=0,
                max_processes=3,
                max_threads=8
            ),
            SecurityProfile.HIGH: SandboxLimits(
                max_cpu_percent=30.0,
                max_memory_mb=128,
                max_disk_mb=50,
                max_execution_time=120,
                max_network_connections=0,
                max_processes=1,
                max_threads=3,
                blocked_syscalls={'fork', 'exec', 'socket', 'bind', 'connect'}
            ),
            SecurityProfile.MAXIMUM: SandboxLimits(
                max_cpu_percent=20.0,
                max_memory_mb=64,
                max_disk_mb=25,
                max_execution_time=60,
                max_network_connections=0,
                max_processes=1,
                max_threads=1,
                blocked_syscalls={'fork', 'exec', 'socket', 'bind', 'connect', 'open', 'write'}
            )
        }
    
    async def execute_code(
        self,
        code: str,
        language: str,
        user_id: str,
        session_id: str,
        sandbox_type: SandboxType = SandboxType.PROCESS,
        security_profile: SecurityProfile = SecurityProfile.STANDARD,
        execution_mode: ExecutionMode = ExecutionMode.BATCH,
        custom_limits: Optional[SandboxLimits] = None,
        allowed_imports: Optional[Set[str]] = None,
        blocked_functions: Optional[Set[str]] = None
    ) -> SandboxResult:
        """
        Execute code in a secure sandbox environment
        
        Args:
            code: Code to execute
            language: Programming language
            user_id: User requesting execution
            session_id: Debug session ID
            sandbox_type: Type of sandbox to use
            security_profile: Security profile to apply
            execution_mode: Execution mode
            custom_limits: Custom resource limits
            allowed_imports: Allowed import modules
            blocked_functions: Blocked function names
            
        Returns:
            SandboxResult with execution results and security information
        """
        execution_id = str(uuid.uuid4())
        
        try:
            # Get security limits
            limits = custom_limits or self.security_profiles[security_profile]
            
            # Audit execution request
            await self.audit_logger.log_event(
                event_type="code_execution_requested",
                user_id=user_id,
                session_id=session_id,
                context={
                    'execution_id': execution_id,
                    'language': language,
                    'sandbox_type': sandbox_type.value,
                    'security_profile': security_profile.value,
                    'execution_mode': execution_mode.value,
                    'code_length': len(code)
                }
            )
            
            # Pre-execution security checks
            security_result = await self._pre_execution_security_check(
                code, language, allowed_imports, blocked_functions
            )
            
            if not security_result['safe']:
                result = SandboxResult(
                    execution_id=execution_id,
                    success=False,
                    exit_code=-1,
                    stdout="",
                    stderr="Security check failed",
                    execution_time=0,
                    memory_used=0,
                    cpu_usage=0,
                    errors=security_result['errors'],
                    security_violations=security_result['violations']
                )
                
                await self._audit_execution_result(user_id, session_id, result)
                self.execution_metrics['security_violations'] += 1
                return result
            
            # Execute based on sandbox type
            if sandbox_type == SandboxType.CONTAINER and self.docker_client:
                result = await self._execute_in_container(
                    execution_id, code, language, limits, user_id, session_id
                )
            elif sandbox_type == SandboxType.PROCESS:
                result = await self._execute_in_process(
                    execution_id, code, language, limits, user_id, session_id
                )
            elif sandbox_type == SandboxType.CHROOT:
                result = await self._execute_in_chroot(
                    execution_id, code, language, limits, user_id, session_id
                )
            else:
                raise ValueError(f"Unsupported sandbox type: {sandbox_type}")
            
            # Post-execution analysis
            result = await self._post_execution_analysis(result, limits)
            
            # Update metrics
            self._update_metrics(result)
            
            # Audit execution result
            await self._audit_execution_result(user_id, session_id, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Sandbox execution failed: {e}")
            
            result = SandboxResult(
                execution_id=execution_id,
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"Sandbox execution error: {str(e)}",
                execution_time=0,
                memory_used=0,
                cpu_usage=0,
                errors=[str(e)]
            )
            
            await self._audit_execution_result(user_id, session_id, result)
            self.execution_metrics['failed_executions'] += 1
            return result
    
    async def _pre_execution_security_check(
        self,
        code: str,
        language: str,
        allowed_imports: Optional[Set[str]],
        blocked_functions: Optional[Set[str]]
    ) -> Dict[str, Any]:
        """
        Perform pre-execution security analysis of code
        
        Args:
            code: Code to analyze
            language: Programming language
            allowed_imports: Allowed import modules
            blocked_functions: Blocked function names
            
        Returns:
            Dict with safety analysis results
        """
        result = {
            'safe': True,
            'violations': [],
            'errors': [],
            'warnings': []
        }
        
        try:
            if language.lower() == 'python':
                result = await self._check_python_security(code, allowed_imports, blocked_functions)
            elif language.lower() in ['javascript', 'js']:
                result = await self._check_javascript_security(code, blocked_functions)
            elif language.lower() in ['bash', 'sh']:
                result = await self._check_shell_security(code)
            else:
                result['warnings'].append(f"No specific security checks for language: {language}")
            
        except Exception as e:
            result['safe'] = False
            result['errors'].append(f"Security check error: {str(e)}")
        
        return result
    
    async def _check_python_security(
        self,
        code: str,
        allowed_imports: Optional[Set[str]],
        blocked_functions: Optional[Set[str]]
    ) -> Dict[str, Any]:
        """Check Python code for security issues"""
        import ast
        
        result = {
            'safe': True,
            'violations': [],
            'errors': [],
            'warnings': []
        }
        
        try:
            # Parse AST
            tree = ast.parse(code)
            
            # Dangerous patterns to check
            dangerous_patterns = {
                'exec': 'Dynamic code execution',
                'eval': 'Dynamic expression evaluation',
                '__import__': 'Dynamic imports',
                'compile': 'Code compilation',
                'globals': 'Global namespace access',
                'locals': 'Local namespace access',
                'vars': 'Variable namespace access',
                'dir': 'Object introspection',
                'getattr': 'Dynamic attribute access',
                'setattr': 'Dynamic attribute setting',
                'delattr': 'Dynamic attribute deletion',
                'hasattr': 'Attribute existence check'
            }
            
            # Dangerous modules
            dangerous_modules = {
                'os', 'sys', 'subprocess', 'socket', 'urllib', 'requests',
                'pickle', 'marshal', 'shelve', 'dbm', 'sqlite3',
                'threading', 'multiprocessing', 'concurrent',
                'ctypes', 'importlib', 'pkgutil', 'runpy'
            }
            
            class SecurityAnalyzer(ast.NodeVisitor):
                def __init__(self):
                    self.violations = []
                    self.imports = set()
                    self.function_calls = set()
                
                def visit_Import(self, node):
                    for alias in node.names:
                        module_name = alias.name.split('.')[0]
                        self.imports.add(module_name)
                        
                        if module_name in dangerous_modules:
                            if not allowed_imports or module_name not in allowed_imports:
                                self.violations.append(f"Dangerous module import: {module_name}")
                    
                    self.generic_visit(node)
                
                def visit_ImportFrom(self, node):
                    if node.module:
                        module_name = node.module.split('.')[0]
                        self.imports.add(module_name)
                        
                        if module_name in dangerous_modules:
                            if not allowed_imports or module_name not in allowed_imports:
                                self.violations.append(f"Dangerous module import: {module_name}")
                    
                    self.generic_visit(node)
                
                def visit_Call(self, node):
                    # Check function calls
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        self.function_calls.add(func_name)
                        
                        if func_name in dangerous_patterns:
                            if not blocked_functions or func_name in blocked_functions:
                                self.violations.append(f"Dangerous function call: {func_name} - {dangerous_patterns[func_name]}")
                    
                    elif isinstance(node.func, ast.Attribute):
                        # Handle method calls like os.system()
                        if isinstance(node.func.value, ast.Name):
                            module_name = node.func.value.id
                            method_name = node.func.attr
                            full_name = f"{module_name}.{method_name}"
                            
                            if module_name in dangerous_modules:
                                self.violations.append(f"Dangerous method call: {full_name}")
                    
                    self.generic_visit(node)
                
                def visit_Attribute(self, node):
                    # Check dangerous attribute access
                    if isinstance(node.value, ast.Name):
                        if node.value.id == '__builtins__':
                            self.violations.append("Access to __builtins__ detected")
                    
                    self.generic_visit(node)
                
                def visit_Subscript(self, node):
                    # Check dangerous subscript operations
                    if isinstance(node.value, ast.Name):
                        if node.value.id == '__builtins__':
                            self.violations.append("Subscript access to __builtins__ detected")
                    
                    self.generic_visit(node)
            
            analyzer = SecurityAnalyzer()
            analyzer.visit(tree)
            
            # Check results
            if analyzer.violations:
                result['safe'] = False
                result['violations'].extend(analyzer.violations)
            
            # Additional checks
            if 'open' in analyzer.function_calls and ('w' in code or 'a' in code):
                result['warnings'].append("File write operation detected")
            
            if any(keyword in code.lower() for keyword in ['password', 'secret', 'key', 'token']):
                result['warnings'].append("Potential sensitive data in code")
            
        except SyntaxError as e:
            result['errors'].append(f"Python syntax error: {str(e)}")
        except Exception as e:
            result['errors'].append(f"Security analysis error: {str(e)}")
        
        return result
    
    async def _check_javascript_security(
        self,
        code: str,
        blocked_functions: Optional[Set[str]]
    ) -> Dict[str, Any]:
        """Check JavaScript code for security issues"""
        result = {
            'safe': True,
            'violations': [],
            'errors': [],
            'warnings': []
        }
        
        # Dangerous JavaScript patterns
        dangerous_patterns = [
            'eval(',
            'Function(',
            'setTimeout(',
            'setInterval(',
            'document.write(',
            'innerHTML',
            'outerHTML',
            'document.domain',
            'location.href',
            'XMLHttpRequest',
            'fetch(',
            'require(',
            'import('
        ]
        
        code_lower = code.lower()
        
        for pattern in dangerous_patterns:
            if pattern.lower() in code_lower:
                result['violations'].append(f"Dangerous JavaScript pattern: {pattern}")
                result['safe'] = False
        
        # Check for blocked functions
        if blocked_functions:
            for func in blocked_functions:
                if func.lower() in code_lower:
                    result['violations'].append(f"Blocked function detected: {func}")
                    result['safe'] = False
        
        return result
    
    async def _check_shell_security(self, code: str) -> Dict[str, Any]:
        """Check shell script for security issues"""
        result = {
            'safe': False,  # Shell scripts are inherently dangerous
            'violations': [],
            'errors': [],
            'warnings': ['Shell script execution is inherently risky']
        }
        
        # Dangerous shell patterns
        dangerous_patterns = [
            'rm ', 'rmdir', 'del ', 'format',
            'wget', 'curl', 'nc ', 'netcat',
            'su ', 'sudo ', 'chmod', 'chown',
            '/etc/', '/proc/', '/sys/',
            'export', 'env ', 'PATH=',
            '>', '>>', 'tee ', 'dd ',
            'mount', 'umount', 'fdisk'
        ]
        
        code_lower = code.lower()
        
        for pattern in dangerous_patterns:
            if pattern in code_lower:
                result['violations'].append(f"Dangerous shell command: {pattern}")
        
        # Always block shell scripts in high security mode
        result['violations'].append("Shell script execution blocked for security")
        
        return result
    
    async def _execute_in_container(
        self,
        execution_id: str,
        code: str,
        language: str,
        limits: SandboxLimits,
        user_id: str,
        session_id: str
    ) -> SandboxResult:
        """Execute code in Docker container"""
        if not self.docker_client:
            raise RuntimeError("Docker not available for container execution")
        
        container = None
        start_time = time.time()
        
        try:
            # Create temporary directory for code
            exec_dir = self.sandbox_dir / execution_id
            exec_dir.mkdir(exist_ok=True)
            
            # Write code to file
            if language.lower() == 'python':
                code_file = exec_dir / "code.py"
                image = "python:3.9-alpine"
                command = ["python", "/workspace/code.py"]
            elif language.lower() in ['javascript', 'js']:
                code_file = exec_dir / "code.js"
                image = "node:16-alpine"
                command = ["node", "/workspace/code.js"]
            else:
                raise ValueError(f"Unsupported language for container execution: {language}")
            
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # Container configuration
            container_config = {
                'image': image,
                'command': command,
                'working_dir': '/workspace',
                'volumes': {str(exec_dir): {'bind': '/workspace', 'mode': 'ro'}},
                'mem_limit': f"{limits.max_memory_mb}m",
                'nano_cpus': int(limits.max_cpu_percent * 10000000),  # Convert to nano CPUs
                'network_disabled': limits.max_network_connections == 0,
                'remove': True,
                'stdout': True,
                'stderr': True,
                'user': 'nobody',
                'environment': {
                    'PYTHONDONTWRITEBYTECODE': '1',
                    'PYTHONUNBUFFERED': '1'
                }
            }
            
            # Security options
            security_opt = [
                'no-new-privileges:true',
                'seccomp=unconfined'  # In production, use a proper seccomp profile
            ]
            
            container_config['security_opt'] = security_opt
            container_config['cap_drop'] = ['ALL']
            container_config['read_only'] = True
            
            # Run container
            container = self.docker_client.containers.run(**container_config, detach=True)
            
            # Wait for completion with timeout
            try:
                exit_code = container.wait(timeout=limits.max_execution_time)['StatusCode']
            except Exception:
                # Timeout - kill container
                container.kill()
                exit_code = -1
            
            # Get output
            stdout = container.logs(stdout=True, stderr=False).decode('utf-8', errors='ignore')
            stderr = container.logs(stdout=False, stderr=True).decode('utf-8', errors='ignore')
            
            execution_time = time.time() - start_time
            
            # Get container stats (if available)
            memory_used = 0
            cpu_usage = 0.0
            
            try:
                stats = container.stats(stream=False)
                memory_stats = stats.get('memory_stats', {})
                memory_used = memory_stats.get('usage', 0) // (1024 * 1024)  # Convert to MB
                
                cpu_stats = stats.get('cpu_stats', {})
                precpu_stats = stats.get('precpu_stats', {})
                if cpu_stats and precpu_stats:
                    cpu_delta = cpu_stats.get('cpu_usage', {}).get('total_usage', 0) - \
                               precpu_stats.get('cpu_usage', {}).get('total_usage', 0)
                    system_delta = cpu_stats.get('system_cpu_usage', 0) - \
                                  precpu_stats.get('system_cpu_usage', 0)
                    if system_delta > 0:
                        cpu_usage = (cpu_delta / system_delta) * 100.0
            except:
                pass
            
            # Cleanup
            shutil.rmtree(exec_dir, ignore_errors=True)
            
            return SandboxResult(
                execution_id=execution_id,
                success=exit_code == 0,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                execution_time=execution_time,
                memory_used=memory_used,
                cpu_usage=cpu_usage
            )
            
        except Exception as e:
            if container:
                try:
                    container.kill()
                    container.remove()
                except:
                    pass
            
            raise e
    
    async def _execute_in_process(
        self,
        execution_id: str,
        code: str,
        language: str,
        limits: SandboxLimits,
        user_id: str,
        session_id: str
    ) -> SandboxResult:
        """Execute code in isolated process"""
        start_time = time.time()
        
        try:
            # Create temporary directory
            exec_dir = self.sandbox_dir / execution_id
            exec_dir.mkdir(exist_ok=True)
            
            # Write code to file
            if language.lower() == 'python':
                code_file = exec_dir / "code.py"
                cmd = [sys.executable, str(code_file)]
            elif language.lower() in ['javascript', 'js']:
                code_file = exec_dir / "code.js"
                cmd = ["node", str(code_file)]
            else:
                raise ValueError(f"Unsupported language for process execution: {language}")
            
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # Set resource limits
            def set_limits():
                # Memory limit
                if hasattr(resource, 'RLIMIT_AS'):
                    resource.setrlimit(resource.RLIMIT_AS, (
                        limits.max_memory_mb * 1024 * 1024,
                        limits.max_memory_mb * 1024 * 1024
                    ))
                
                # CPU time limit
                if hasattr(resource, 'RLIMIT_CPU'):
                    resource.setrlimit(resource.RLIMIT_CPU, (
                        limits.max_execution_time,
                        limits.max_execution_time
                    ))
                
                # File size limit
                if hasattr(resource, 'RLIMIT_FSIZE'):
                    resource.setrlimit(resource.RLIMIT_FSIZE, (
                        limits.max_disk_mb * 1024 * 1024,
                        limits.max_disk_mb * 1024 * 1024
                    ))
                
                # Process limit
                if hasattr(resource, 'RLIMIT_NPROC'):
                    resource.setrlimit(resource.RLIMIT_NPROC, (
                        limits.max_processes,
                        limits.max_processes
                    ))
            
            # Execute with timeout and resource limits
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=exec_dir,
                preexec_fn=set_limits if os.name != 'nt' else None,
                text=True,
                encoding='utf-8'
            )
            
            # Track process for monitoring
            self.active_sandboxes[execution_id] = {
                'process': process,
                'start_time': start_time,
                'limits': limits,
                'user_id': user_id,
                'session_id': session_id
            }
            
            try:
                stdout, stderr = process.communicate(timeout=limits.max_execution_time)
                exit_code = process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                exit_code = -1
                stderr += "\nExecution timed out"
            finally:
                if execution_id in self.active_sandboxes:
                    del self.active_sandboxes[execution_id]
            
            execution_time = time.time() - start_time
            
            # Get resource usage
            memory_used = 0
            cpu_usage = 0.0
            
            try:
                if process.pid:
                    proc_info = psutil.Process(process.pid)
                    memory_info = proc_info.memory_info()
                    memory_used = memory_info.rss // (1024 * 1024)  # MB
                    cpu_usage = proc_info.cpu_percent()
            except:
                pass
            
            # Cleanup
            shutil.rmtree(exec_dir, ignore_errors=True)
            
            return SandboxResult(
                execution_id=execution_id,
                success=exit_code == 0,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                execution_time=execution_time,
                memory_used=memory_used,
                cpu_usage=cpu_usage
            )
            
        except Exception as e:
            # Cleanup on error
            if execution_id in self.active_sandboxes:
                del self.active_sandboxes[execution_id]
            
            exec_dir = self.sandbox_dir / execution_id
            if exec_dir.exists():
                shutil.rmtree(exec_dir, ignore_errors=True)
            
            raise e
    
    async def _execute_in_chroot(
        self,
        execution_id: str,
        code: str,
        language: str,
        limits: SandboxLimits,
        user_id: str,
        session_id: str
    ) -> SandboxResult:
        """Execute code in chroot jail (Unix only)"""
        if os.name == 'nt':
            raise RuntimeError("Chroot sandboxing not available on Windows")
        
        # For now, fallback to process execution
        # Full chroot implementation would require root privileges
        return await self._execute_in_process(
            execution_id, code, language, limits, user_id, session_id
        )
    
    async def _post_execution_analysis(
        self,
        result: SandboxResult,
        limits: SandboxLimits
    ) -> SandboxResult:
        """Analyze execution result for security violations"""
        
        # Check resource violations
        if result.memory_used > limits.max_memory_mb:
            result.security_violations.append(f"Memory limit exceeded: {result.memory_used}MB > {limits.max_memory_mb}MB")
        
        if result.cpu_usage > limits.max_cpu_percent:
            result.security_violations.append(f"CPU limit exceeded: {result.cpu_usage}% > {limits.max_cpu_percent}%")
        
        if result.execution_time > limits.max_execution_time:
            result.security_violations.append(f"Time limit exceeded: {result.execution_time}s > {limits.max_execution_time}s")
        
        # Analyze output for suspicious content
        suspicious_patterns = [
            'error:', 'exception:', 'traceback',
            'permission denied', 'access denied',
            'connection refused', 'timeout',
            'segmentation fault', 'core dumped'
        ]
        
        output_text = (result.stdout + result.stderr).lower()
        for pattern in suspicious_patterns:
            if pattern in output_text:
                result.warnings.append(f"Suspicious output pattern: {pattern}")
        
        return result
    
    def _update_metrics(self, result: SandboxResult):
        """Update execution metrics"""
        self.execution_metrics['total_executions'] += 1
        
        if result.success:
            self.execution_metrics['successful_executions'] += 1
        else:
            self.execution_metrics['failed_executions'] += 1
        
        if result.security_violations:
            self.execution_metrics['security_violations'] += len(result.security_violations)
        
        if result.blocked_operations:
            self.execution_metrics['blocked_operations'] += len(result.blocked_operations)
    
    async def _audit_execution_result(
        self,
        user_id: str,
        session_id: str,
        result: SandboxResult
    ):
        """Audit the execution result"""
        await self.audit_logger.log_event(
            event_type="code_execution_completed",
            user_id=user_id,
            session_id=session_id,
            result="success" if result.success else "failure",
            context={
                'execution_id': result.execution_id,
                'exit_code': result.exit_code,
                'execution_time': result.execution_time,
                'memory_used': result.memory_used,
                'cpu_usage': result.cpu_usage,
                'security_violations': len(result.security_violations),
                'warnings': len(result.warnings),
                'errors': len(result.errors)
            }
        )
        
        # Log security violations separately
        if result.security_violations:
            for violation in result.security_violations:
                await self.audit_logger.log_security_event(
                    security_event_type="violation",
                    user_id=user_id,
                    session_id=session_id,
                    threat_level="medium",
                    additional_context={
                        'execution_id': result.execution_id,
                        'violation': violation
                    }
                )
    
    async def terminate_execution(self, execution_id: str, reason: str = "User request") -> bool:
        """Terminate a running execution"""
        if execution_id not in self.active_sandboxes:
            return False
        
        try:
            sandbox_info = self.active_sandboxes[execution_id]
            process = sandbox_info.get('process')
            
            if process and process.poll() is None:
                process.terminate()
                # Give it time to terminate gracefully
                time.sleep(1)
                if process.poll() is None:
                    process.kill()
            
            # Cleanup
            del self.active_sandboxes[execution_id]
            
            # Audit termination
            await self.audit_logger.log_event(
                event_type="code_execution_terminated",
                user_id=sandbox_info.get('user_id'),
                session_id=sandbox_info.get('session_id'),
                context={
                    'execution_id': execution_id,
                    'reason': reason
                }
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to terminate execution {execution_id}: {e}")
            return False
    
    def get_active_executions(self) -> List[Dict[str, Any]]:
        """Get information about active executions"""
        active = []
        current_time = time.time()
        
        for execution_id, info in self.active_sandboxes.items():
            active.append({
                'execution_id': execution_id,
                'user_id': info.get('user_id'),
                'session_id': info.get('session_id'),
                'start_time': info.get('start_time'),
                'elapsed_time': current_time - info.get('start_time', current_time),
                'limits': info.get('limits').__dict__ if info.get('limits') else {}
            })
        
        return active
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get sandbox execution metrics"""
        return {
            **self.execution_metrics,
            'active_executions': len(self.active_sandboxes),
            'docker_available': self.docker_client is not None
        }
    
    def _start_monitoring(self):
        """Start background monitoring of sandbox executions"""
        async def monitor_loop():
            while True:
                try:
                    await self._monitor_active_executions()
                    await asyncio.sleep(10)  # Check every 10 seconds
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
                    await asyncio.sleep(30)
        
        self._monitor_task = asyncio.create_task(monitor_loop())
    
    async def _monitor_active_executions(self):
        """Monitor active executions for violations"""
        current_time = time.time()
        terminated = []
        
        for execution_id, info in list(self.active_sandboxes.items()):
            try:
                # Check time limits
                elapsed_time = current_time - info.get('start_time', current_time)
                limits = info.get('limits')
                
                if limits and elapsed_time > limits.max_execution_time:
                    await self.terminate_execution(execution_id, "Time limit exceeded")
                    terminated.append(execution_id)
                    continue
                
                # Check if process is still running
                process = info.get('process')
                if process and process.poll() is not None:
                    # Process has finished
                    terminated.append(execution_id)
                    continue
                
                # Check resource usage if possible
                if process and hasattr(process, 'pid'):
                    try:
                        proc_info = psutil.Process(process.pid)
                        memory_mb = proc_info.memory_info().rss / (1024 * 1024)
                        
                        if limits and memory_mb > limits.max_memory_mb * 2:  # 2x limit as kill threshold
                            await self.terminate_execution(execution_id, "Excessive memory usage")
                            terminated.append(execution_id)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        # Process no longer exists
                        terminated.append(execution_id)
                        
            except Exception as e:
                self.logger.error(f"Error monitoring execution {execution_id}: {e}")
        
        # Cleanup terminated executions
        for execution_id in terminated:
            if execution_id in self.active_sandboxes:
                del self.active_sandboxes[execution_id]
    
    async def shutdown(self):
        """Shutdown sandbox executor"""
        try:
            # Cancel monitoring task
            if self._monitor_task:
                self._monitor_task.cancel()
            
            # Terminate all active executions
            for execution_id in list(self.active_sandboxes.keys()):
                await self.terminate_execution(execution_id, "System shutdown")
            
            # Cleanup temporary directories
            if self.sandbox_dir.exists():
                shutil.rmtree(self.sandbox_dir, ignore_errors=True)
            
            # Close Docker client
            if self.docker_client:
                self.docker_client.close()
            
            self.logger.info("Secure sandbox executor shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Sandbox executor shutdown error: {e}")