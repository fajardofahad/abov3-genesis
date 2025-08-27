"""
ABOV3 Genesis - NL2Code Error Handler
Comprehensive error handling and logging system for the NL2Code module
"""

import asyncio
import traceback
import logging
import json
from typing import Dict, List, Any, Optional, Type, Union, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from enum import Enum
import functools

# Configure module logger
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification"""
    VALIDATION = "validation"
    PARSING = "parsing"
    GENERATION = "generation"
    INTEGRATION = "integration"
    CONFIGURATION = "configuration"
    EXTERNAL_SERVICE = "external_service"
    RESOURCE = "resource"
    SECURITY = "security"
    PERFORMANCE = "performance"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for error handling"""
    module: str
    function: str
    user_input: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    memory_usage: Optional[int] = None
    stack_trace: Optional[str] = None
    correlation_id: Optional[str] = None


@dataclass
class NL2CodeError:
    """Structured error information"""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    details: str
    context: ErrorContext
    recoverable: bool
    recovery_suggestions: List[str]
    user_friendly_message: str
    technical_details: Dict[str, Any]


class NL2CodeException(Exception):
    """Base exception for NL2Code module"""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        recoverable: bool = True,
        recovery_suggestions: Optional[List[str]] = None,
        user_friendly_message: Optional[str] = None,
        technical_details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.severity = severity
        self.category = category
        self.recoverable = recoverable
        self.recovery_suggestions = recovery_suggestions or []
        self.user_friendly_message = user_friendly_message or message
        self.technical_details = technical_details or {}


class ValidationError(NL2CodeException):
    """Input validation errors"""
    
    def __init__(self, message: str, field: str = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.VALIDATION,
            **kwargs
        )
        self.field = field


class ParsingError(NL2CodeException):
    """Natural language parsing errors"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.PARSING,
            **kwargs
        )


class CodeGenerationError(NL2CodeException):
    """Code generation errors"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.GENERATION,
            **kwargs
        )


class IntegrationError(NL2CodeException):
    """Integration errors with external systems"""
    
    def __init__(self, message: str, service: str = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.INTEGRATION,
            **kwargs
        )
        self.service = service


class ConfigurationError(NL2CodeException):
    """Configuration and setup errors"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.CONFIGURATION,
            recoverable=False,
            **kwargs
        )


class ResourceError(NL2CodeException):
    """Resource-related errors (memory, disk, network)"""
    
    def __init__(self, message: str, resource_type: str = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.RESOURCE,
            **kwargs
        )
        self.resource_type = resource_type


class SecurityError(NL2CodeException):
    """Security-related errors"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.SECURITY,
            recoverable=False,
            **kwargs
        )


class PerformanceError(NL2CodeException):
    """Performance-related errors"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.PERFORMANCE,
            **kwargs
        )


class ErrorHandler:
    """
    Comprehensive error handling system for NL2Code module
    Provides logging, recovery, and user-friendly error reporting
    """
    
    def __init__(self, log_file_path: Optional[Path] = None):
        self.log_file_path = log_file_path or Path("nl2code_errors.log")
        self.error_history: List[NL2CodeError] = []
        self.recovery_strategies: Dict[ErrorCategory, Callable] = {}
        self.error_counters: Dict[ErrorCategory, int] = {}
        self._setup_logging()
        self._initialize_recovery_strategies()
    
    def _setup_logging(self):
        """Setup comprehensive logging configuration"""
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler for error logs
        file_handler = logging.FileHandler(self.log_file_path, encoding='utf-8')
        file_handler.setLevel(logging.ERROR)
        file_handler.setFormatter(formatter)
        
        # Console handler for all logs
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Configure logger
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        logger.info("Error handling system initialized")
    
    def _initialize_recovery_strategies(self):
        """Initialize recovery strategies for different error categories"""
        
        self.recovery_strategies = {
            ErrorCategory.VALIDATION: self._recover_validation_error,
            ErrorCategory.PARSING: self._recover_parsing_error,
            ErrorCategory.GENERATION: self._recover_generation_error,
            ErrorCategory.INTEGRATION: self._recover_integration_error,
            ErrorCategory.CONFIGURATION: self._recover_configuration_error,
            ErrorCategory.RESOURCE: self._recover_resource_error,
            ErrorCategory.EXTERNAL_SERVICE: self._recover_external_service_error
        }
    
    def handle_error(
        self,
        error: Union[Exception, NL2CodeException],
        context: Optional[ErrorContext] = None,
        attempt_recovery: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive error handling with logging and recovery
        """
        
        # Generate unique error ID
        error_id = f"NL2C-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{len(self.error_history):04d}"
        
        # Extract error information
        if isinstance(error, NL2CodeException):
            severity = error.severity
            category = error.category
            message = str(error)
            recoverable = error.recoverable
            recovery_suggestions = error.recovery_suggestions
            user_friendly_message = error.user_friendly_message
            technical_details = error.technical_details
        else:
            severity = ErrorSeverity.HIGH
            category = self._categorize_error(error)
            message = str(error)
            recoverable = True
            recovery_suggestions = []
            user_friendly_message = self._generate_user_friendly_message(error)
            technical_details = {"exception_type": type(error).__name__}
        
        # Create error record
        error_record = NL2CodeError(
            error_id=error_id,
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            message=message,
            details=traceback.format_exc() if not isinstance(error, NL2CodeException) else str(error),
            context=context or ErrorContext(module="unknown", function="unknown"),
            recoverable=recoverable,
            recovery_suggestions=recovery_suggestions,
            user_friendly_message=user_friendly_message,
            technical_details=technical_details
        )
        
        # Log the error
        self._log_error(error_record)
        
        # Store in history
        self.error_history.append(error_record)
        
        # Update counters
        self.error_counters[category] = self.error_counters.get(category, 0) + 1
        
        # Attempt recovery if requested and error is recoverable
        recovery_result = None
        if attempt_recovery and recoverable:
            recovery_result = self._attempt_recovery(error_record)
        
        return {
            "error_id": error_id,
            "severity": severity.value,
            "category": category.value,
            "user_message": user_friendly_message,
            "recoverable": recoverable,
            "recovery_suggestions": recovery_suggestions,
            "recovery_attempted": attempt_recovery and recoverable,
            "recovery_successful": recovery_result is not None and recovery_result.get('success', False),
            "recovery_details": recovery_result,
            "technical_details": technical_details
        }
    
    def _log_error(self, error_record: NL2CodeError):
        """Log error with appropriate level and detail"""
        
        log_message = f"[{error_record.error_id}] {error_record.message}"
        
        # Create structured log data
        log_data = {
            "error_id": error_record.error_id,
            "severity": error_record.severity.value,
            "category": error_record.category.value,
            "context": asdict(error_record.context),
            "recoverable": error_record.recoverable,
            "technical_details": error_record.technical_details
        }
        
        if error_record.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"{log_message} | Data: {json.dumps(log_data, indent=2)}")
        elif error_record.severity == ErrorSeverity.HIGH:
            logger.error(f"{log_message} | Data: {json.dumps(log_data, indent=2)}")
        elif error_record.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"{log_message} | Data: {json.dumps(log_data, indent=2)}")
        else:
            logger.info(f"{log_message} | Data: {json.dumps(log_data, indent=2)}")
        
        # Log detailed stack trace for high severity errors
        if error_record.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            if error_record.details:
                logger.error(f"Stack trace for {error_record.error_id}:\n{error_record.details}")
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize error based on exception type and message"""
        
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Categorization by exception type
        if error_type in ['ValueError', 'TypeError', 'AssertionError']:
            return ErrorCategory.VALIDATION
        elif error_type in ['SyntaxError', 'ParseError']:
            return ErrorCategory.PARSING
        elif error_type in ['FileNotFoundError', 'PermissionError', 'OSError']:
            return ErrorCategory.RESOURCE
        elif error_type in ['ConnectionError', 'TimeoutError', 'HTTPError']:
            return ErrorCategory.EXTERNAL_SERVICE
        elif error_type in ['MemoryError', 'ResourceExhaustedError']:
            return ErrorCategory.RESOURCE
        
        # Categorization by message content
        if any(keyword in error_message for keyword in ['config', 'setting', 'parameter']):
            return ErrorCategory.CONFIGURATION
        elif any(keyword in error_message for keyword in ['generate', 'create', 'build']):
            return ErrorCategory.GENERATION
        elif any(keyword in error_message for keyword in ['parse', 'analyze', 'understand']):
            return ErrorCategory.PARSING
        elif any(keyword in error_message for keyword in ['auth', 'permission', 'access']):
            return ErrorCategory.SECURITY
        elif any(keyword in error_message for keyword in ['slow', 'timeout', 'performance']):
            return ErrorCategory.PERFORMANCE
        elif any(keyword in error_message for keyword in ['api', 'service', 'connection']):
            return ErrorCategory.INTEGRATION
        
        return ErrorCategory.UNKNOWN
    
    def _generate_user_friendly_message(self, error: Exception) -> str:
        """Generate user-friendly error message"""
        
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Common user-friendly messages
        if 'file not found' in error_message:
            return "The required file could not be found. Please check the file path and try again."
        elif 'permission denied' in error_message:
            return "Permission denied. Please check your file permissions and try again."
        elif 'connection' in error_message:
            return "Connection error occurred. Please check your internet connection and try again."
        elif 'timeout' in error_message:
            return "The operation took too long to complete. Please try again or simplify your request."
        elif 'memory' in error_message:
            return "Not enough memory to complete the operation. Please try with a smaller request."
        elif 'invalid' in error_message or 'value error' in error_message:
            return "Invalid input provided. Please check your input and try again."
        elif 'syntax' in error_message:
            return "There was an error parsing your request. Please rephrase and try again."
        
        # Default message
        return "An unexpected error occurred. Please try again or contact support if the problem persists."
    
    def _attempt_recovery(self, error_record: NL2CodeError) -> Optional[Dict[str, Any]]:
        """Attempt to recover from error using appropriate strategy"""
        
        recovery_strategy = self.recovery_strategies.get(error_record.category)
        
        if not recovery_strategy:
            logger.warning(f"No recovery strategy for category {error_record.category.value}")
            return None
        
        try:
            logger.info(f"Attempting recovery for error {error_record.error_id}")
            recovery_result = recovery_strategy(error_record)
            
            if recovery_result and recovery_result.get('success', False):
                logger.info(f"Recovery successful for error {error_record.error_id}")
            else:
                logger.warning(f"Recovery failed for error {error_record.error_id}")
            
            return recovery_result
            
        except Exception as recovery_error:
            logger.error(f"Recovery attempt failed for {error_record.error_id}: {recovery_error}")
            return {"success": False, "error": str(recovery_error)}
    
    # Recovery strategy implementations
    def _recover_validation_error(self, error_record: NL2CodeError) -> Dict[str, Any]:
        """Recovery strategy for validation errors"""
        
        suggestions = [
            "Check input format and ensure all required fields are provided",
            "Verify data types match expected values",
            "Review input length and size constraints"
        ]
        
        return {
            "success": False,  # Validation errors typically need user intervention
            "strategy": "user_correction",
            "suggestions": suggestions,
            "auto_correctable": False
        }
    
    def _recover_parsing_error(self, error_record: NL2CodeError) -> Dict[str, Any]:
        """Recovery strategy for parsing errors"""
        
        suggestions = [
            "Try rephrasing your request with simpler language",
            "Break complex requests into smaller parts",
            "Use more specific technical terms",
            "Provide additional context about your requirements"
        ]
        
        return {
            "success": False,
            "strategy": "request_clarification",
            "suggestions": suggestions,
            "auto_correctable": False
        }
    
    def _recover_generation_error(self, error_record: NL2CodeError) -> Dict[str, Any]:
        """Recovery strategy for code generation errors"""
        
        # Try simpler generation approach
        suggestions = [
            "Reduce the complexity of the requested feature",
            "Generate components separately instead of all at once",
            "Use simpler technology stack",
            "Provide more specific requirements"
        ]
        
        return {
            "success": False,
            "strategy": "simplified_generation",
            "suggestions": suggestions,
            "auto_correctable": True,
            "fallback_approach": "incremental_generation"
        }
    
    def _recover_integration_error(self, error_record: NL2CodeError) -> Dict[str, Any]:
        """Recovery strategy for integration errors"""
        
        return {
            "success": False,
            "strategy": "retry_with_backoff",
            "suggestions": [
                "Check network connectivity",
                "Verify service credentials and configuration",
                "Try again in a few moments"
            ],
            "auto_correctable": True,
            "retry_count": 3,
            "backoff_seconds": [1, 5, 15]
        }
    
    def _recover_configuration_error(self, error_record: NL2CodeError) -> Dict[str, Any]:
        """Recovery strategy for configuration errors"""
        
        return {
            "success": False,
            "strategy": "configuration_validation",
            "suggestions": [
                "Check environment variables and configuration files",
                "Verify all required settings are provided",
                "Review configuration documentation"
            ],
            "auto_correctable": False
        }
    
    def _recover_resource_error(self, error_record: NL2CodeError) -> Dict[str, Any]:
        """Recovery strategy for resource errors"""
        
        return {
            "success": False,
            "strategy": "resource_optimization",
            "suggestions": [
                "Free up system memory by closing other applications",
                "Use smaller input data or reduce complexity",
                "Consider processing in smaller chunks"
            ],
            "auto_correctable": True,
            "optimization_strategies": ["chunking", "memory_cleanup", "simplified_processing"]
        }
    
    def _recover_external_service_error(self, error_record: NL2CodeError) -> Dict[str, Any]:
        """Recovery strategy for external service errors"""
        
        return {
            "success": False,
            "strategy": "service_fallback",
            "suggestions": [
                "Check service status and connectivity",
                "Verify API keys and authentication",
                "Try using alternative service or local processing"
            ],
            "auto_correctable": True,
            "fallback_options": ["local_processing", "alternative_service", "cached_results"]
        }
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and trends"""
        
        total_errors = len(self.error_history)
        if total_errors == 0:
            return {"total_errors": 0, "categories": {}, "severities": {}}
        
        # Count by category
        category_counts = {}
        severity_counts = {}
        recent_errors = []
        
        # Last 24 hours
        cutoff_time = datetime.now().timestamp() - (24 * 60 * 60)
        
        for error in self.error_history:
            # Category counts
            cat = error.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
            
            # Severity counts
            sev = error.severity.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            
            # Recent errors
            if error.timestamp.timestamp() > cutoff_time:
                recent_errors.append({
                    "error_id": error.error_id,
                    "timestamp": error.timestamp.isoformat(),
                    "category": error.category.value,
                    "severity": error.severity.value,
                    "message": error.user_friendly_message
                })
        
        return {
            "total_errors": total_errors,
            "categories": category_counts,
            "severities": severity_counts,
            "recent_errors_24h": len(recent_errors),
            "recent_errors": recent_errors[-10:],  # Last 10 recent errors
            "recovery_rate": len([e for e in self.error_history if e.recoverable]) / total_errors,
            "critical_errors": len([e for e in self.error_history if e.severity == ErrorSeverity.CRITICAL])
        }
    
    def export_error_report(self, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Export comprehensive error report"""
        
        output_path = output_path or Path(f"nl2code_error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        report_data = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_errors": len(self.error_history),
                "report_version": "1.0"
            },
            "statistics": self.get_error_statistics(),
            "detailed_errors": [
                {
                    "error_id": error.error_id,
                    "timestamp": error.timestamp.isoformat(),
                    "severity": error.severity.value,
                    "category": error.category.value,
                    "message": error.message,
                    "user_friendly_message": error.user_friendly_message,
                    "recoverable": error.recoverable,
                    "recovery_suggestions": error.recovery_suggestions,
                    "context": asdict(error.context),
                    "technical_details": error.technical_details
                }
                for error in self.error_history
            ]
        }
        
        # Write report to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Error report exported to {output_path}")
        
        return report_data
    
    def clear_error_history(self, older_than_days: int = 30):
        """Clear old errors from history"""
        
        cutoff_time = datetime.now().timestamp() - (older_than_days * 24 * 60 * 60)
        
        original_count = len(self.error_history)
        self.error_history = [
            error for error in self.error_history
            if error.timestamp.timestamp() > cutoff_time
        ]
        
        cleared_count = original_count - len(self.error_history)
        logger.info(f"Cleared {cleared_count} old errors from history")


# Global error handler instance
_global_error_handler = None


def get_error_handler() -> ErrorHandler:
    """Get global error handler instance"""
    global _global_error_handler
    
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    
    return _global_error_handler


def handle_errors(
    error_context: Optional[Dict[str, Any]] = None,
    attempt_recovery: bool = True,
    reraise: bool = False
):
    """
    Decorator for automatic error handling
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            context = ErrorContext(
                module=func.__module__,
                function=func.__name__,
                parameters={"args": str(args), "kwargs": str(kwargs)}
            )
            
            if error_context:
                for key, value in error_context.items():
                    setattr(context, key, value)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler = get_error_handler()
                result = error_handler.handle_error(e, context, attempt_recovery)
                
                if reraise:
                    raise
                
                return {"error": True, "error_details": result}
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            context = ErrorContext(
                module=func.__module__,
                function=func.__name__,
                parameters={"args": str(args), "kwargs": str(kwargs)}
            )
            
            if error_context:
                for key, value in error_context.items():
                    setattr(context, key, value)
            
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_handler = get_error_handler()
                result = error_handler.handle_error(e, context, attempt_recovery)
                
                if reraise:
                    raise
                
                return {"error": True, "error_details": result}
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def log_performance(func: Callable) -> Callable:
    """
    Decorator to log performance metrics
    """
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        import time
        import psutil
        import os
        
        start_time = time.time()
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = process.memory_info().rss
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            logger.info(
                f"Performance: {func.__name__} executed in {execution_time:.3f}s, "
                f"memory delta: {memory_delta / 1024 / 1024:.2f}MB"
            )
            
            return result
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            context = ErrorContext(
                module=func.__module__,
                function=func.__name__,
                execution_time=execution_time
            )
            
            error_handler = get_error_handler()
            error_handler.handle_error(e, context)
            raise
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        import time
        import psutil
        import os
        
        start_time = time.time()
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss
        
        try:
            result = await func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = process.memory_info().rss
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            logger.info(
                f"Performance: {func.__name__} executed in {execution_time:.3f}s, "
                f"memory delta: {memory_delta / 1024 / 1024:.2f}MB"
            )
            
            return result
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            context = ErrorContext(
                module=func.__module__,
                function=func.__name__,
                execution_time=execution_time
            )
            
            error_handler = get_error_handler()
            error_handler.handle_error(e, context)
            raise
    
    # Return appropriate wrapper based on function type
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


# Utility functions
def validate_input(
    value: Any,
    expected_type: Type,
    field_name: str,
    required: bool = True,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None
) -> Any:
    """Validate input with comprehensive error handling"""
    
    if value is None:
        if required:
            raise ValidationError(
                f"Field '{field_name}' is required but was not provided",
                field=field_name,
                recovery_suggestions=[
                    f"Provide a valid {expected_type.__name__} value for {field_name}",
                    "Check your input parameters and try again"
                ]
            )
        return None
    
    if not isinstance(value, expected_type):
        raise ValidationError(
            f"Field '{field_name}' must be of type {expected_type.__name__}, got {type(value).__name__}",
            field=field_name,
            recovery_suggestions=[
                f"Convert {field_name} to {expected_type.__name__}",
                "Check the data type of your input"
            ]
        )
    
    if isinstance(value, str) and min_length is not None and len(value) < min_length:
        raise ValidationError(
            f"Field '{field_name}' must be at least {min_length} characters long",
            field=field_name,
            recovery_suggestions=[
                f"Provide a longer value for {field_name}",
                f"Minimum length is {min_length} characters"
            ]
        )
    
    if isinstance(value, str) and max_length is not None and len(value) > max_length:
        raise ValidationError(
            f"Field '{field_name}' must be no more than {max_length} characters long",
            field=field_name,
            recovery_suggestions=[
                f"Provide a shorter value for {field_name}",
                f"Maximum length is {max_length} characters"
            ]
        )
    
    return value


def require_configuration(config_keys: List[str], config_dict: Dict[str, Any]):
    """Require specific configuration keys"""
    
    missing_keys = []
    for key in config_keys:
        if key not in config_dict or config_dict[key] is None:
            missing_keys.append(key)
    
    if missing_keys:
        raise ConfigurationError(
            f"Missing required configuration keys: {', '.join(missing_keys)}",
            recovery_suggestions=[
                "Check your configuration file or environment variables",
                "Ensure all required settings are provided",
                f"Required keys: {', '.join(config_keys)}"
            ]
        )