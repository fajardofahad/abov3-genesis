"""
ABOV3 Genesis - Resilience & Error Handling Infrastructure
Enterprise-grade error handling, circuit breakers, and recovery mechanisms
"""

import asyncio
import time
import logging
import traceback
import functools
import sys
from typing import Dict, Any, Optional, List, Callable, Union, Type
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import aiofiles
from collections import defaultdict, deque
import threading
import uuid
import signal

# Optional imports
try:
    import aiohttp
except ImportError:
    aiohttp = None

import os

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Failing, rejecting requests  
    HALF_OPEN = "half_open" # Testing if service recovered

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"           # Minor issues, retry automatically
    MEDIUM = "medium"     # Moderate issues, user notification
    HIGH = "high"         # Major issues, fallback required
    CRITICAL = "critical" # System threatening, immediate attention

class RecoveryStrategy(Enum):
    """Recovery strategies"""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    GRACEFUL_DEGRADE = "graceful_degrade"
    FAIL_FAST = "fail_fast"

@dataclass
class ErrorContext:
    """Error context information"""
    error_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    error_type: str = ""
    message: str = ""
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    component: str = ""
    operation: str = ""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    traceback: Optional[str] = None
    context_data: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    recovery_attempted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage"""
        return {
            'error_id': self.error_id,
            'timestamp': self.timestamp,
            'error_type': self.error_type,
            'message': self.message,
            'severity': self.severity.value,
            'component': self.component,
            'operation': self.operation,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'traceback': self.traceback,
            'context_data': self.context_data,
            'retry_count': self.retry_count,
            'recovery_attempted': self.recovery_attempted
        }

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5      # Failures before opening circuit
    recovery_timeout: float = 60.0  # Time before trying half-open
    success_threshold: int = 3       # Successes needed to close circuit
    timeout: float = 30.0           # Request timeout
    monitor_window: float = 300.0   # Monitoring window (5 minutes)

class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance
    Prevents cascading failures by stopping requests to failing services
    """

    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        
        # Failure tracking
        self._failures = deque(maxlen=100)  # Keep last 100 failures
        self._successes = deque(maxlen=100)
        self._last_failure_time = 0.0
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        
        # Statistics
        self._total_requests = 0
        self._total_failures = 0
        self._state_changes = []
        
        self._lock = threading.Lock()

    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self._lock:
            self._total_requests += 1
            
            # Check circuit state
            if self.state == CircuitState.OPEN:
                if time.time() - self._last_failure_time < self.config.recovery_timeout:
                    raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is OPEN")
                else:
                    # Try half-open
                    self.state = CircuitState.HALF_OPEN
                    self._state_changes.append((time.time(), CircuitState.HALF_OPEN))
            
            elif self.state == CircuitState.HALF_OPEN:
                if self._consecutive_successes >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self._consecutive_failures = 0
                    self._state_changes.append((time.time(), CircuitState.CLOSED))
        
        # Execute function
        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.config.timeout)
            else:
                result = func(*args, **kwargs)
            
            # Record success
            with self._lock:
                self._successes.append(time.time())
                self._consecutive_successes += 1
                self._consecutive_failures = 0
                
                # Clean old successes
                self._clean_old_records(self._successes)
            
            return result
            
        except Exception as e:
            # Record failure
            with self._lock:
                self._failures.append(time.time())
                self._total_failures += 1
                self._consecutive_failures += 1
                self._consecutive_successes = 0
                self._last_failure_time = time.time()
                
                # Clean old failures
                self._clean_old_records(self._failures)
                
                # Check if we should open the circuit
                recent_failures = len([f for f in self._failures if time.time() - f < self.config.monitor_window])
                
                if (self._consecutive_failures >= self.config.failure_threshold or
                    recent_failures >= self.config.failure_threshold):
                    
                    if self.state != CircuitState.OPEN:
                        self.state = CircuitState.OPEN
                        self._state_changes.append((time.time(), CircuitState.OPEN))
                        logger.warning(f"Circuit breaker '{self.name}' opened due to failures")
            
            raise e

    def _clean_old_records(self, records: deque):
        """Remove records older than monitoring window"""
        current_time = time.time()
        while records and current_time - records[0] > self.config.monitor_window:
            records.popleft()

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        with self._lock:
            current_time = time.time()
            recent_failures = len([f for f in self._failures if current_time - f < self.config.monitor_window])
            recent_successes = len([s for s in self._successes if current_time - s < self.config.monitor_window])
            
            return {
                'name': self.name,
                'state': self.state.value,
                'total_requests': self._total_requests,
                'total_failures': self._total_failures,
                'recent_failures': recent_failures,
                'recent_successes': recent_successes,
                'consecutive_failures': self._consecutive_failures,
                'consecutive_successes': self._consecutive_successes,
                'failure_rate': self._total_failures / max(1, self._total_requests),
                'recent_failure_rate': recent_failures / max(1, recent_failures + recent_successes),
                'last_failure_time': self._last_failure_time,
                'state_changes': self._state_changes[-10:]  # Last 10 state changes
            }

    def reset(self):
        """Reset circuit breaker to closed state"""
        with self._lock:
            self.state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._consecutive_successes = 0
            self._failures.clear()
            self._successes.clear()
            self._state_changes.append((time.time(), CircuitState.CLOSED))

class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass

class RetryPolicy:
    """
    Configurable retry policy with exponential backoff
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[List[Type[Exception]]] = None
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or [
            ConnectionError, TimeoutError, asyncio.TimeoutError,
            aiohttp.ClientError if aiohttp else Exception
        ]

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Check if exception should be retried"""
        if attempt >= self.max_attempts:
            return False
        
        return any(isinstance(exception, exc_type) for exc_type in self.retryable_exceptions)

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for next attempt"""
        delay = self.base_delay * (self.backoff_factor ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add random jitter to avoid thundering herd
            import random
            delay *= (0.5 + random.random() * 0.5)
        
        return delay

class ErrorRecoveryManager:
    """
    Centralized error handling and recovery management
    Coordinates circuit breakers, retries, and fallback mechanisms
    """

    def __init__(self, project_path: Optional[Path] = None):
        self.project_path = project_path
        
        # Circuit breakers by component
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Error history and analytics
        self._error_history: List[ErrorContext] = []
        self._error_patterns = defaultdict(int)
        
        # Recovery strategies by error type
        self._recovery_strategies: Dict[str, RecoveryStrategy] = {
            'ConnectionError': RecoveryStrategy.RETRY,
            'TimeoutError': RecoveryStrategy.RETRY,
            'HTTPError': RecoveryStrategy.RETRY,
            'OllamaConnectionError': RecoveryStrategy.CIRCUIT_BREAK,
            'AIModelNotAvailable': RecoveryStrategy.FALLBACK,
            'FileSystemError': RecoveryStrategy.GRACEFUL_DEGRADE,
            'MemoryError': RecoveryStrategy.GRACEFUL_DEGRADE,
            'CriticalSystemError': RecoveryStrategy.FAIL_FAST
        }
        
        # Fallback handlers
        self._fallback_handlers: Dict[str, Callable] = {}
        
        # Error persistence
        self._error_log_path = None
        if project_path:
            self._error_log_path = project_path / '.abov3' / 'logs' / 'errors.jsonl'
            self._error_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Background error analysis
        self._analysis_task = None
        self._start_error_analysis()

    def _start_error_analysis(self):
        """Start background error analysis"""
        if self._analysis_task is None:
            self._analysis_task = asyncio.create_task(self._analyze_errors_loop())

    async def _analyze_errors_loop(self):
        """Background error analysis and reporting"""
        while True:
            try:
                await asyncio.sleep(300)  # Analyze every 5 minutes
                await self._analyze_error_patterns()
                await self._check_system_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error analysis failed: {e}")

    async def _analyze_error_patterns(self):
        """Analyze error patterns and suggest optimizations"""
        if len(self._error_history) < 10:
            return
        
        # Recent errors (last hour)
        current_time = time.time()
        recent_errors = [
            err for err in self._error_history
            if current_time - err.timestamp < 3600
        ]
        
        if len(recent_errors) > 10:
            # High error rate detected
            error_types = defaultdict(int)
            for error in recent_errors:
                error_types[error.error_type] += 1
            
            # Log warning for high error rates
            most_common = max(error_types.items(), key=lambda x: x[1])
            if most_common[1] > 5:
                logger.warning(
                    f"High error rate detected: {most_common[0]} "
                    f"occurred {most_common[1]} times in the last hour"
                )

    async def _check_system_health(self):
        """Check overall system health"""
        unhealthy_circuits = []
        
        for name, circuit in self._circuit_breakers.items():
            stats = circuit.get_stats()
            if stats['state'] == 'open' or stats['recent_failure_rate'] > 0.5:
                unhealthy_circuits.append((name, stats))
        
        if unhealthy_circuits:
            logger.warning(f"Unhealthy circuits detected: {[name for name, _ in unhealthy_circuits]}")

    def get_circuit_breaker(self, component: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create circuit breaker for component"""
        if component not in self._circuit_breakers:
            config = config or CircuitBreakerConfig()
            self._circuit_breakers[component] = CircuitBreaker(component, config)
        
        return self._circuit_breakers[component]

    def register_fallback(self, error_type: str, handler: Callable):
        """Register fallback handler for error type"""
        self._fallback_handlers[error_type] = handler

    async def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any] = None,
        component: str = "unknown",
        operation: str = "unknown"
    ) -> Optional[Any]:
        """
        Central error handling with recovery attempts
        Returns recovery result or None if no recovery possible
        """
        error_context = ErrorContext(
            error_type=type(error).__name__,
            message=str(error),
            component=component,
            operation=operation,
            traceback=traceback.format_exc(),
            context_data=context or {}
        )
        
        # Determine severity
        error_context.severity = self._assess_severity(error, context)
        
        # Log error
        logger.error(f"Error in {component}.{operation}: {error}", extra=error_context.to_dict())
        
        # Record error
        self._error_history.append(error_context)
        await self._persist_error(error_context)
        
        # Attempt recovery based on strategy
        recovery_strategy = self._recovery_strategies.get(
            error_context.error_type,
            RecoveryStrategy.FAIL_FAST
        )
        
        recovery_result = None
        
        try:
            if recovery_strategy == RecoveryStrategy.RETRY:
                recovery_result = await self._attempt_retry(error_context)
            elif recovery_strategy == RecoveryStrategy.FALLBACK:
                recovery_result = await self._attempt_fallback(error_context)
            elif recovery_strategy == RecoveryStrategy.GRACEFUL_DEGRADE:
                recovery_result = await self._graceful_degrade(error_context)
            elif recovery_strategy == RecoveryStrategy.CIRCUIT_BREAK:
                # Circuit breaker will handle this automatically
                pass
            # FAIL_FAST: Let error propagate
            
            error_context.recovery_attempted = True
            
        except Exception as recovery_error:
            logger.error(f"Recovery attempt failed: {recovery_error}")
        
        return recovery_result

    def _assess_severity(self, error: Exception, context: Dict[str, Any] = None) -> ErrorSeverity:
        """Assess error severity"""
        error_type = type(error).__name__
        
        # Critical errors
        if error_type in ['MemoryError', 'SystemExit', 'KeyboardInterrupt']:
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if error_type in ['FileNotFoundError', 'PermissionError', 'ConnectionRefusedError']:
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if error_type in ['TimeoutError', 'HTTPError', 'ValidationError']:
            return ErrorSeverity.MEDIUM
        
        # Default to low for unknown errors
        return ErrorSeverity.LOW

    async def _attempt_retry(self, error_context: ErrorContext) -> Optional[Any]:
        """Attempt retry with exponential backoff"""
        # This would be implemented by the calling code using the retry decorator
        # Here we just log the retry attempt
        error_context.retry_count += 1
        logger.info(f"Retry attempt {error_context.retry_count} for error {error_context.error_id}")
        return None

    async def _attempt_fallback(self, error_context: ErrorContext) -> Optional[Any]:
        """Attempt fallback recovery"""
        fallback_handler = self._fallback_handlers.get(error_context.error_type)
        
        if fallback_handler:
            try:
                if asyncio.iscoroutinefunction(fallback_handler):
                    result = await fallback_handler(error_context)
                else:
                    result = fallback_handler(error_context)
                
                logger.info(f"Fallback successful for error {error_context.error_id}")
                return result
                
            except Exception as e:
                logger.error(f"Fallback failed for error {error_context.error_id}: {e}")
        
        return None

    async def _graceful_degrade(self, error_context: ErrorContext) -> Optional[Any]:
        """Gracefully degrade functionality"""
        # Return simplified/cached response
        logger.info(f"Gracefully degrading for error {error_context.error_id}")
        
        # Could return cached responses, simplified outputs, etc.
        return {
            'degraded': True,
            'message': 'Service temporarily degraded due to system issues',
            'error_id': error_context.error_id
        }

    async def _persist_error(self, error_context: ErrorContext):
        """Persist error to log file"""
        if not self._error_log_path:
            return
        
        try:
            error_line = json.dumps(error_context.to_dict()) + '\n'
            
            async with aiofiles.open(self._error_log_path, 'a', encoding='utf-8') as f:
                await f.write(error_line)
                
        except Exception as e:
            logger.error(f"Failed to persist error: {e}")

    async def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        current_time = time.time()
        
        # Time-based error counts
        last_hour = [e for e in self._error_history if current_time - e.timestamp < 3600]
        last_day = [e for e in self._error_history if current_time - e.timestamp < 86400]
        
        # Error type distribution
        error_types = defaultdict(int)
        severity_counts = defaultdict(int)
        component_errors = defaultdict(int)
        
        for error in self._error_history:
            error_types[error.error_type] += 1
            severity_counts[error.severity.value] += 1
            component_errors[error.component] += 1
        
        # Circuit breaker stats
        circuit_stats = {}
        for name, circuit in self._circuit_breakers.items():
            circuit_stats[name] = circuit.get_stats()
        
        return {
            'total_errors': len(self._error_history),
            'errors_last_hour': len(last_hour),
            'errors_last_day': len(last_day),
            'error_rate_per_hour': len(last_hour),
            'error_rate_per_day': len(last_day),
            'error_types': dict(error_types),
            'severity_distribution': dict(severity_counts),
            'component_errors': dict(component_errors),
            'circuit_breakers': circuit_stats,
            'recovery_rate': len([e for e in self._error_history if e.recovery_attempted]) / max(1, len(self._error_history))
        }

    async def cleanup(self):
        """Cleanup resources"""
        if self._analysis_task:
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass


# Decorators for error handling

def with_circuit_breaker(component: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator to add circuit breaker protection"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get error manager from context
            error_manager = getattr(asyncio.current_task(), '_error_manager', None)
            if error_manager:
                circuit = error_manager.get_circuit_breaker(component, config)
                return await circuit.call(func, *args, **kwargs)
            else:
                return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def with_retry(policy: Optional[RetryPolicy] = None):
    """Decorator to add retry logic"""
    if policy is None:
        policy = RetryPolicy()
    
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(policy.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if not policy.should_retry(e, attempt + 1):
                        break
                    
                    if attempt < policy.max_attempts - 1:
                        delay = policy.get_delay(attempt)
                        logger.info(f"Retrying {func.__name__} in {delay:.2f}s (attempt {attempt + 1})")
                        await asyncio.sleep(delay)
            
            # All retries exhausted
            if last_exception:
                raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(policy.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if not policy.should_retry(e, attempt + 1):
                        break
                    
                    if attempt < policy.max_attempts - 1:
                        delay = policy.get_delay(attempt)
                        logger.info(f"Retrying {func.__name__} in {delay:.2f}s (attempt {attempt + 1})")
                        time.sleep(delay)
            
            if last_exception:
                raise last_exception
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def with_error_handling(component: str = "unknown", operation: str = "unknown"):
    """Decorator to add centralized error handling"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Get error manager from context
                error_manager = getattr(asyncio.current_task(), '_error_manager', None)
                if error_manager:
                    recovery_result = await error_manager.handle_error(
                        e, 
                        context={'args': str(args)[:100], 'kwargs': str(kwargs)[:100]},
                        component=component,
                        operation=operation
                    )
                    if recovery_result is not None:
                        return recovery_result
                
                # Re-raise if no recovery
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {component}.{operation}: {e}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


class resilience_context:
    """Context manager for error handling and resilience"""
    
    def __init__(self, error_manager: ErrorRecoveryManager):
        self.error_manager = error_manager
        self.previous_error_manager = None

    async def __aenter__(self):
        # Inject error manager into current task
        current_task = asyncio.current_task()
        if current_task:
            self.previous_error_manager = getattr(current_task, '_error_manager', None)
            current_task._error_manager = self.error_manager
        return self.error_manager

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Handle any unhandled exceptions
        if exc_type and exc_val:
            await self.error_manager.handle_error(
                exc_val,
                context={'exc_type': exc_type.__name__},
                component='context_manager',
                operation='__aexit__'
            )
        
        # Restore previous error manager
        current_task = asyncio.current_task()
        if current_task:
            if self.previous_error_manager:
                current_task._error_manager = self.previous_error_manager
            else:
                if hasattr(current_task, '_error_manager'):
                    delattr(current_task, '_error_manager')


# Global error handler for unhandled exceptions
def setup_global_exception_handler(error_manager: ErrorRecoveryManager):
    """Setup global exception handlers"""
    
    def handle_exception(exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions"""
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
        
        # Attempt graceful shutdown
        try:
            asyncio.create_task(error_manager.handle_error(
                exc_value,
                component='global',
                operation='uncaught_exception'
            ))
        except:
            pass
    
    def handle_signal(signum, frame):
        """Handle system signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        
        # Trigger cleanup
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(error_manager.cleanup())
        except:
            pass
    
    # Install handlers
    sys.excepthook = handle_exception
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)