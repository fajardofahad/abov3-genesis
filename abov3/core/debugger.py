"""
ABOV3 Genesis - Advanced Debugging Toolkit
Comprehensive debugging, profiling, and diagnostics
"""

import sys
import os
import time
import traceback
import inspect
import logging
import cProfile
import pstats
import io
import json
import asyncio
from typing import Any, Dict, List, Optional, Callable, Union
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from functools import wraps
import threading
import psutil


class PerformanceProfiler:
    """Performance profiling and monitoring"""
    
    def __init__(self):
        self.profiles = {}
        self.metrics = {
            'function_calls': {},
            'execution_times': {},
            'memory_usage': {},
            'bottlenecks': []
        }
        self.profiler = cProfile.Profile()
    
    @contextmanager
    def profile(self, name: str):
        """Context manager for profiling code blocks"""
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        
        yield
        
        end_time = time.perf_counter()
        end_memory = self._get_memory_usage()
        
        duration = end_time - start_time
        memory_delta = end_memory - start_memory
        
        # Update metrics
        if name not in self.metrics['execution_times']:
            self.metrics['execution_times'][name] = []
        self.metrics['execution_times'][name].append(duration)
        
        if name not in self.metrics['memory_usage']:
            self.metrics['memory_usage'][name] = []
        self.metrics['memory_usage'][name].append(memory_delta)
        
        # Detect bottlenecks
        if duration > 1.0:  # More than 1 second
            self.metrics['bottlenecks'].append({
                'name': name,
                'duration': duration,
                'memory': memory_delta,
                'timestamp': datetime.now().isoformat()
            })
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile function execution"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"
            
            # Count function calls
            if func_name not in self.metrics['function_calls']:
                self.metrics['function_calls'][func_name] = 0
            self.metrics['function_calls'][func_name] += 1
            
            # Profile execution
            with self.profile(func_name):
                result = func(*args, **kwargs)
            
            return result
        
        return wrapper
    
    def profile_async_function(self, func: Callable) -> Callable:
        """Decorator to profile async function execution"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"
            
            # Count function calls
            if func_name not in self.metrics['function_calls']:
                self.metrics['function_calls'][func_name] = 0
            self.metrics['function_calls'][func_name] += 1
            
            # Profile execution
            start_time = time.perf_counter()
            start_memory = self._get_memory_usage()
            
            result = await func(*args, **kwargs)
            
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Update metrics
            if func_name not in self.metrics['execution_times']:
                self.metrics['execution_times'][func_name] = []
            self.metrics['execution_times'][func_name].append(duration)
            
            if func_name not in self.metrics['memory_usage']:
                self.metrics['memory_usage'][func_name] = []
            self.metrics['memory_usage'][func_name].append(memory_delta)
            
            return result
        
        return wrapper
    
    def start_profiling(self):
        """Start CPU profiling"""
        self.profiler.enable()
    
    def stop_profiling(self):
        """Stop CPU profiling"""
        self.profiler.disable()
    
    def get_profile_stats(self) -> str:
        """Get profiling statistics"""
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        return s.getvalue()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'summary': self._generate_summary(),
            'hotspots': self._identify_hotspots(),
            'bottlenecks': self.metrics['bottlenecks'],
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except:
            return 0
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate performance summary"""
        total_calls = sum(self.metrics['function_calls'].values())
        
        avg_execution_times = {}
        for func, times in self.metrics['execution_times'].items():
            if times:
                avg_execution_times[func] = sum(times) / len(times)
        
        return {
            'total_function_calls': total_calls,
            'unique_functions': len(self.metrics['function_calls']),
            'average_execution_times': avg_execution_times,
            'total_bottlenecks': len(self.metrics['bottlenecks'])
        }
    
    def _identify_hotspots(self) -> List[Dict[str, Any]]:
        """Identify performance hotspots"""
        hotspots = []
        
        # Functions with high call counts
        for func, count in self.metrics['function_calls'].items():
            if count > 100:
                hotspots.append({
                    'type': 'high_call_count',
                    'function': func,
                    'calls': count
                })
        
        # Functions with long execution times
        for func, times in self.metrics['execution_times'].items():
            if times:
                avg_time = sum(times) / len(times)
                if avg_time > 0.1:  # More than 100ms average
                    hotspots.append({
                        'type': 'slow_execution',
                        'function': func,
                        'average_time': avg_time
                    })
        
        # Functions with high memory usage
        for func, usages in self.metrics['memory_usage'].items():
            if usages:
                avg_usage = sum(usages) / len(usages)
                if avg_usage > 10:  # More than 10MB average
                    hotspots.append({
                        'type': 'high_memory',
                        'function': func,
                        'average_memory_mb': avg_usage
                    })
        
        return hotspots
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Check for bottlenecks
        if self.metrics['bottlenecks']:
            recommendations.append(f"Found {len(self.metrics['bottlenecks'])} performance bottlenecks - consider optimization")
        
        # Check for high-frequency calls
        high_freq_funcs = [f for f, c in self.metrics['function_calls'].items() if c > 1000]
        if high_freq_funcs:
            recommendations.append(f"Consider caching results for frequently called functions: {', '.join(high_freq_funcs[:3])}")
        
        # Check for memory issues
        high_mem_funcs = []
        for func, usages in self.metrics['memory_usage'].items():
            if usages and max(usages) > 100:  # More than 100MB
                high_mem_funcs.append(func)
        
        if high_mem_funcs:
            recommendations.append(f"High memory usage detected in: {', '.join(high_mem_funcs[:3])}")
        
        return recommendations


class CodeDebugger:
    """Advanced code debugging utilities"""
    
    def __init__(self):
        self.breakpoints = {}
        self.watch_variables = {}
        self.execution_trace = []
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup debug logger"""
        logger = logging.getLogger('ABOV3.Debugger')
        logger.setLevel(logging.DEBUG)
        
        # Console handler with color support
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        
        # Format with colors
        formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        return logger
    
    def trace_execution(self, func: Callable) -> Callable:
        """Decorator to trace function execution"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            
            # Log entry
            self.logger.debug(f"→ Entering {func_name}")
            self._log_arguments(func_name, args, kwargs)
            
            # Record in trace
            trace_entry = {
                'function': func_name,
                'args': str(args)[:100],
                'kwargs': str(kwargs)[:100],
                'timestamp': time.time(),
                'thread': threading.current_thread().name
            }
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Log exit
                self.logger.debug(f"← Exiting {func_name} with result: {str(result)[:100]}")
                trace_entry['result'] = str(result)[:100]
                trace_entry['success'] = True
                
                return result
                
            except Exception as e:
                # Log exception
                self.logger.error(f"✗ Exception in {func_name}: {str(e)}")
                trace_entry['error'] = str(e)
                trace_entry['success'] = False
                raise
            
            finally:
                self.execution_trace.append(trace_entry)
        
        return wrapper
    
    def trace_async_execution(self, func: Callable) -> Callable:
        """Decorator to trace async function execution"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            func_name = func.__name__
            
            # Log entry
            self.logger.debug(f"→ Entering async {func_name}")
            self._log_arguments(func_name, args, kwargs)
            
            # Record in trace
            trace_entry = {
                'function': func_name,
                'async': True,
                'args': str(args)[:100],
                'kwargs': str(kwargs)[:100],
                'timestamp': time.time()
            }
            
            try:
                # Execute function
                result = await func(*args, **kwargs)
                
                # Log exit
                self.logger.debug(f"← Exiting async {func_name} with result: {str(result)[:100]}")
                trace_entry['result'] = str(result)[:100]
                trace_entry['success'] = True
                
                return result
                
            except Exception as e:
                # Log exception
                self.logger.error(f"✗ Exception in async {func_name}: {str(e)}")
                trace_entry['error'] = str(e)
                trace_entry['success'] = False
                raise
            
            finally:
                self.execution_trace.append(trace_entry)
        
        return wrapper
    
    def _log_arguments(self, func_name: str, args: tuple, kwargs: dict):
        """Log function arguments"""
        if args:
            self.logger.debug(f"  Args: {str(args)[:200]}")
        if kwargs:
            self.logger.debug(f"  Kwargs: {str(kwargs)[:200]}")
    
    def set_breakpoint(self, file: str, line: int, condition: Optional[str] = None):
        """Set a conditional breakpoint"""
        key = f"{file}:{line}"
        self.breakpoints[key] = {
            'file': file,
            'line': line,
            'condition': condition,
            'hits': 0
        }
        self.logger.info(f"Breakpoint set at {key}")
    
    def watch_variable(self, var_name: str, callback: Optional[Callable] = None):
        """Watch a variable for changes"""
        self.watch_variables[var_name] = {
            'callback': callback,
            'values': [],
            'changes': 0
        }
        self.logger.info(f"Watching variable: {var_name}")
    
    def inspect_object(self, obj: Any, depth: int = 2) -> Dict[str, Any]:
        """Deep inspection of an object"""
        inspection = {
            'type': type(obj).__name__,
            'module': type(obj).__module__,
            'id': id(obj),
            'size': sys.getsizeof(obj),
            'attributes': {},
            'methods': [],
            'properties': {}
        }
        
        # Get attributes
        for attr_name in dir(obj):
            if not attr_name.startswith('_'):
                try:
                    attr_value = getattr(obj, attr_name)
                    if callable(attr_value):
                        inspection['methods'].append(attr_name)
                    else:
                        inspection['attributes'][attr_name] = {
                            'type': type(attr_value).__name__,
                            'value': str(attr_value)[:100]
                        }
                except:
                    pass
        
        # Get properties
        for name, prop in inspect.getmembers(type(obj), lambda x: isinstance(x, property)):
            try:
                value = getattr(obj, name)
                inspection['properties'][name] = {
                    'type': type(value).__name__,
                    'value': str(value)[:100]
                }
            except:
                inspection['properties'][name] = {'error': 'Could not access'}
        
        return inspection
    
    def analyze_stack_trace(self, exception: Exception) -> Dict[str, Any]:
        """Analyze exception stack trace"""
        analysis = {
            'exception_type': type(exception).__name__,
            'message': str(exception),
            'traceback': [],
            'local_variables': {},
            'suggestions': []
        }
        
        # Get traceback
        tb = traceback.extract_tb(exception.__traceback__)
        
        for frame in tb:
            analysis['traceback'].append({
                'file': frame.filename,
                'line': frame.lineno,
                'function': frame.name,
                'code': frame.line
            })
        
        # Try to get local variables from the last frame
        if exception.__traceback__:
            frame = exception.__traceback__.tb_frame
            while frame.f_back:
                frame = frame.f_back
            
            analysis['local_variables'] = {
                k: str(v)[:100] for k, v in frame.f_locals.items()
                if not k.startswith('__')
            }
        
        # Generate suggestions based on exception type
        analysis['suggestions'] = self._generate_debug_suggestions(exception)
        
        return analysis
    
    def _generate_debug_suggestions(self, exception: Exception) -> List[str]:
        """Generate debugging suggestions based on exception"""
        suggestions = []
        
        if isinstance(exception, AttributeError):
            suggestions.append("Check if the object has the attribute you're trying to access")
            suggestions.append("Verify object initialization")
        elif isinstance(exception, KeyError):
            suggestions.append("Check if the key exists in the dictionary")
            suggestions.append("Use .get() method with a default value")
        elif isinstance(exception, IndexError):
            suggestions.append("Check list/array bounds")
            suggestions.append("Verify list is not empty before accessing")
        elif isinstance(exception, TypeError):
            suggestions.append("Check argument types passed to function")
            suggestions.append("Verify function signature matches call")
        elif isinstance(exception, ValueError):
            suggestions.append("Validate input data format")
            suggestions.append("Check for type conversions")
        elif isinstance(exception, ImportError):
            suggestions.append("Install missing package with pip")
            suggestions.append("Check Python path and module location")
        elif isinstance(exception, FileNotFoundError):
            suggestions.append("Verify file path is correct")
            suggestions.append("Check file exists before opening")
        
        return suggestions
    
    def get_execution_report(self) -> Dict[str, Any]:
        """Get execution trace report"""
        return {
            'total_calls': len(self.execution_trace),
            'successful_calls': sum(1 for t in self.execution_trace if t.get('success')),
            'failed_calls': sum(1 for t in self.execution_trace if not t.get('success')),
            'recent_trace': self.execution_trace[-10:],
            'breakpoints': self.breakpoints,
            'watched_variables': self.watch_variables
        }


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


class SystemDebugger:
    """System-level debugging and diagnostics"""
    
    def __init__(self):
        self.system_info = self._collect_system_info()
        self.resource_monitor = ResourceMonitor()
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information"""
        info = {
            'platform': sys.platform,
            'python_version': sys.version,
            'python_executable': sys.executable,
            'working_directory': os.getcwd(),
            'environment_variables': dict(os.environ),
            'cpu_count': os.cpu_count(),
            'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown'
        }
        
        try:
            import psutil
            info['memory_total'] = psutil.virtual_memory().total / (1024**3)  # GB
            info['memory_available'] = psutil.virtual_memory().available / (1024**3)  # GB
            info['disk_usage'] = psutil.disk_usage('/').percent
            info['cpu_percent'] = psutil.cpu_percent(interval=1)
        except ImportError:
            pass
        
        return info
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check installed dependencies"""
        dependencies = {}
        
        required_packages = [
            'ollama', 'flask', 'fastapi', 'pyyaml', 'psutil',
            'numpy', 'pandas', 'requests', 'aiohttp'
        ]
        
        for package in required_packages:
            try:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                dependencies[package] = {
                    'installed': True,
                    'version': version
                }
            except ImportError:
                dependencies[package] = {
                    'installed': False,
                    'version': None
                }
        
        return dependencies
    
    def diagnose_issue(self, error_type: str) -> Dict[str, Any]:
        """Diagnose common issues"""
        diagnosis = {
            'issue': error_type,
            'checks': [],
            'recommendations': []
        }
        
        if error_type == 'model_not_found':
            # Check Ollama installation
            diagnosis['checks'].append({
                'name': 'Ollama installed',
                'result': os.path.exists('/usr/local/bin/ollama') or os.path.exists('C:\\Program Files\\Ollama\\ollama.exe')
            })
            
            diagnosis['recommendations'].extend([
                "Install Ollama from https://ollama.ai",
                "Pull required model: ollama pull llama3",
                "Check Ollama service is running: ollama serve"
            ])
        
        elif error_type == 'high_memory_usage':
            diagnosis['checks'].append({
                'name': 'Available memory',
                'result': f"{self.system_info.get('memory_available', 0):.2f} GB"
            })
            
            diagnosis['recommendations'].extend([
                "Close unnecessary applications",
                "Reduce batch size or model size",
                "Consider using streaming responses"
            ])
        
        elif error_type == 'slow_performance':
            diagnosis['checks'].append({
                'name': 'CPU usage',
                'result': f"{self.system_info.get('cpu_percent', 0)}%"
            })
            
            diagnosis['recommendations'].extend([
                "Profile code to identify bottlenecks",
                "Use caching for repeated operations",
                "Consider async operations for I/O tasks"
            ])
        
        return diagnosis


class ResourceMonitor:
    """Monitor system resources"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = []
    
    def start_monitoring(self, interval: float = 1.0):
        """Start resource monitoring"""
        self.monitoring = True
        
        def monitor():
            while self.monitoring:
                metrics = self._collect_metrics()
                self.metrics.append(metrics)
                
                # Keep only last 1000 measurements
                if len(self.metrics) > 1000:
                    self.metrics = self.metrics[-1000:]
                
                time.sleep(interval)
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current resource metrics"""
        metrics = {
            'timestamp': time.time(),
            'memory_percent': 0,
            'cpu_percent': 0,
            'disk_io': 0
        }
        
        try:
            import psutil
            metrics['memory_percent'] = psutil.virtual_memory().percent
            metrics['cpu_percent'] = psutil.cpu_percent(interval=0.1)
            
            disk_io = psutil.disk_io_counters()
            if disk_io:
                metrics['disk_io'] = disk_io.read_bytes + disk_io.write_bytes
        except:
            pass
        
        return metrics
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get resource usage summary"""
        if not self.metrics:
            return {}
        
        memory_values = [m['memory_percent'] for m in self.metrics]
        cpu_values = [m['cpu_percent'] for m in self.metrics]
        
        return {
            'memory': {
                'current': memory_values[-1] if memory_values else 0,
                'average': sum(memory_values) / len(memory_values) if memory_values else 0,
                'peak': max(memory_values) if memory_values else 0
            },
            'cpu': {
                'current': cpu_values[-1] if cpu_values else 0,
                'average': sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                'peak': max(cpu_values) if cpu_values else 0
            }
        }


# Global instances
_performance_profiler = None
_code_debugger = None
_system_debugger = None

def get_performance_profiler() -> PerformanceProfiler:
    """Get global performance profiler"""
    global _performance_profiler
    if _performance_profiler is None:
        _performance_profiler = PerformanceProfiler()
    return _performance_profiler

def get_code_debugger() -> CodeDebugger:
    """Get global code debugger"""
    global _code_debugger
    if _code_debugger is None:
        _code_debugger = CodeDebugger()
    return _code_debugger

def get_system_debugger() -> SystemDebugger:
    """Get global system debugger"""
    global _system_debugger
    if _system_debugger is None:
        _system_debugger = SystemDebugger()
    return _system_debugger