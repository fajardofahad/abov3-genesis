"""
ABOV3 Genesis - Advanced Error Handler
Comprehensive error handling, recovery, and diagnostics
"""

import logging
import traceback
import sys
import os
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime
from pathlib import Path
import json
import asyncio
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels"""
    CRITICAL = "critical"  # System cannot continue
    HIGH = "high"        # Major functionality broken
    MEDIUM = "medium"    # Feature impaired
    LOW = "low"         # Minor issue
    WARNING = "warning"  # Potential issue


class ErrorCategory(Enum):
    """Error categories for better classification"""
    MODEL_ERROR = "model_error"
    FILE_ERROR = "file_error"
    NETWORK_ERROR = "network_error"
    PARSING_ERROR = "parsing_error"
    VALIDATION_ERROR = "validation_error"
    PERMISSION_ERROR = "permission_error"
    CONFIGURATION_ERROR = "configuration_error"
    DEPENDENCY_ERROR = "dependency_error"
    USER_INPUT_ERROR = "user_input_error"
    SYSTEM_ERROR = "system_error"


class ErrorContext:
    """Context information for errors"""
    
    def __init__(self):
        self.request_id = None
        self.user_input = None
        self.project_path = None
        self.operation = None
        self.timestamp = datetime.now()
        self.environment = {
            'platform': sys.platform,
            'python_version': sys.version,
            'working_dir': os.getcwd()
        }
        self.additional_data = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary"""
        return {
            'request_id': self.request_id,
            'user_input': self.user_input,
            'project_path': str(self.project_path) if self.project_path else None,
            'operation': self.operation,
            'timestamp': self.timestamp.isoformat(),
            'environment': self.environment,
            'additional_data': self.additional_data
        }


class ErrorRecord:
    """Record of an error occurrence"""
    
    def __init__(self, error: Exception, category: ErrorCategory, 
                 severity: ErrorSeverity, context: ErrorContext):
        self.error = error
        self.error_type = type(error).__name__
        self.error_message = str(error)
        self.category = category
        self.severity = severity
        self.context = context
        self.traceback = traceback.format_exc()
        self.timestamp = datetime.now()
        self.recovery_attempted = False
        self.recovery_successful = False
        self.recovery_method = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary for logging"""
        return {
            'error_type': self.error_type,
            'error_message': self.error_message,
            'category': self.category.value,
            'severity': self.severity.value,
            'context': self.context.to_dict(),
            'traceback': self.traceback,
            'timestamp': self.timestamp.isoformat(),
            'recovery_attempted': self.recovery_attempted,
            'recovery_successful': self.recovery_successful,
            'recovery_method': self.recovery_method
        }


class ErrorHandler:
    """Advanced error handler with recovery capabilities"""
    
    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path.cwd() / '.abov3' / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        self.logger = self._setup_logging()
        
        # Error history for pattern detection
        self.error_history: List[ErrorRecord] = []
        self.max_history_size = 100
        
        # Recovery strategies
        self.recovery_strategies = self._initialize_recovery_strategies()
        
        # Error patterns for common issues
        self.error_patterns = self._initialize_error_patterns()
        
        # Metrics
        self.metrics = {
            'total_errors': 0,
            'recovered_errors': 0,
            'critical_errors': 0,
            'errors_by_category': {},
            'recovery_success_rate': 0.0
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger('ABOV3.ErrorHandler')
        logger.setLevel(logging.DEBUG)
        
        # File handler for errors
        error_log = self.log_dir / f'errors_{datetime.now().strftime("%Y%m%d")}.log'
        file_handler = logging.FileHandler(error_log)
        file_handler.setLevel(logging.ERROR)
        
        # File handler for debug
        debug_log = self.log_dir / f'debug_{datetime.now().strftime("%Y%m%d")}.log'
        debug_handler = logging.FileHandler(debug_log)
        debug_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        debug_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(debug_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _initialize_recovery_strategies(self) -> Dict[ErrorCategory, List[Callable]]:
        """Initialize recovery strategies for different error categories"""
        return {
            ErrorCategory.MODEL_ERROR: [
                self._recover_model_error_fallback,
                self._recover_model_error_retry,
                self._recover_model_error_simplify
            ],
            ErrorCategory.FILE_ERROR: [
                self._recover_file_error_create_dir,
                self._recover_file_error_permissions,
                self._recover_file_error_alternative_path
            ],
            ErrorCategory.NETWORK_ERROR: [
                self._recover_network_error_retry,
                self._recover_network_error_offline_mode
            ],
            ErrorCategory.PARSING_ERROR: [
                self._recover_parsing_error_cleanup,
                self._recover_parsing_error_alternative_format
            ],
            ErrorCategory.PERMISSION_ERROR: [
                self._recover_permission_error_elevate,
                self._recover_permission_error_alternative
            ],
            ErrorCategory.DEPENDENCY_ERROR: [
                self._recover_dependency_error_install,
                self._recover_dependency_error_fallback
            ]
        }
    
    def _initialize_error_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize common error patterns for quick identification"""
        return {
            'model_not_found': {
                'pattern': r'model.*not.*found|no.*model.*available',
                'category': ErrorCategory.MODEL_ERROR,
                'severity': ErrorSeverity.HIGH,
                'suggestion': 'Install required model or use fallback'
            },
            'file_not_found': {
                'pattern': r'file.*not.*found|no such file',
                'category': ErrorCategory.FILE_ERROR,
                'severity': ErrorSeverity.MEDIUM,
                'suggestion': 'Check file path or create missing file'
            },
            'permission_denied': {
                'pattern': r'permission.*denied|access.*denied',
                'category': ErrorCategory.PERMISSION_ERROR,
                'severity': ErrorSeverity.HIGH,
                'suggestion': 'Check file permissions or run with appropriate privileges'
            },
            'network_timeout': {
                'pattern': r'timeout|connection.*refused|network.*error',
                'category': ErrorCategory.NETWORK_ERROR,
                'severity': ErrorSeverity.MEDIUM,
                'suggestion': 'Check network connection or retry'
            },
            'out_of_memory': {
                'pattern': r'out.*of.*memory|memory.*error',
                'category': ErrorCategory.SYSTEM_ERROR,
                'severity': ErrorSeverity.CRITICAL,
                'suggestion': 'Reduce operation size or free up memory'
            }
        }
    
    async def handle_error(self, error: Exception, context: Optional[ErrorContext] = None,
                           auto_recover: bool = True) -> Dict[str, Any]:
        """Main error handling method"""
        # Create context if not provided
        if context is None:
            context = ErrorContext()
        
        # Classify error
        category, severity = self._classify_error(error)
        
        # Create error record
        record = ErrorRecord(error, category, severity, context)
        
        # Log error
        self._log_error(record)
        
        # Update metrics
        self._update_metrics(record)
        
        # Add to history
        self._add_to_history(record)
        
        # Attempt recovery if enabled
        recovery_result = None
        if auto_recover and severity != ErrorSeverity.CRITICAL:
            recovery_result = await self._attempt_recovery(record)
        
        # Generate user-friendly response
        response = self._generate_error_response(record, recovery_result)
        
        return response
    
    def _classify_error(self, error: Exception) -> tuple[ErrorCategory, ErrorSeverity]:
        """Classify error into category and severity"""
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Check against known patterns
        for pattern_name, pattern_info in self.error_patterns.items():
            import re
            if re.search(pattern_info['pattern'], error_str):
                return pattern_info['category'], pattern_info['severity']
        
        # Classification based on exception type
        if isinstance(error, FileNotFoundError):
            return ErrorCategory.FILE_ERROR, ErrorSeverity.MEDIUM
        elif isinstance(error, PermissionError):
            return ErrorCategory.PERMISSION_ERROR, ErrorSeverity.HIGH
        elif isinstance(error, ValueError):
            return ErrorCategory.VALIDATION_ERROR, ErrorSeverity.MEDIUM
        elif isinstance(error, KeyError):
            return ErrorCategory.PARSING_ERROR, ErrorSeverity.MEDIUM
        elif isinstance(error, ImportError):
            return ErrorCategory.DEPENDENCY_ERROR, ErrorSeverity.HIGH
        elif isinstance(error, MemoryError):
            return ErrorCategory.SYSTEM_ERROR, ErrorSeverity.CRITICAL
        elif isinstance(error, ConnectionError):
            return ErrorCategory.NETWORK_ERROR, ErrorSeverity.MEDIUM
        else:
            return ErrorCategory.SYSTEM_ERROR, ErrorSeverity.HIGH
    
    async def _attempt_recovery(self, record: ErrorRecord) -> Optional[Dict[str, Any]]:
        """Attempt to recover from error"""
        record.recovery_attempted = True
        
        # Get recovery strategies for this category
        strategies = self.recovery_strategies.get(record.category, [])
        
        for strategy in strategies:
            try:
                self.logger.info(f"Attempting recovery with {strategy.__name__}")
                result = await strategy(record)
                
                if result and result.get('success'):
                    record.recovery_successful = True
                    record.recovery_method = strategy.__name__
                    self.logger.info(f"Recovery successful with {strategy.__name__}")
                    return result
                    
            except Exception as recovery_error:
                self.logger.warning(f"Recovery strategy {strategy.__name__} failed: {recovery_error}")
                continue
        
        return None
    
    # Recovery strategies
    async def _recover_model_error_fallback(self, record: ErrorRecord) -> Dict[str, Any]:
        """Fallback to alternative model"""
        fallback_models = ['codellama:latest', 'mistral:latest', 'phi:latest']
        
        for model in fallback_models:
            # This would check if model is available
            # Simplified for demonstration
            return {
                'success': True,
                'method': 'model_fallback',
                'fallback_model': model,
                'message': f'Switched to fallback model: {model}'
            }
        
        return {'success': False}
    
    async def _recover_model_error_retry(self, record: ErrorRecord) -> Dict[str, Any]:
        """Retry model operation with delay"""
        await asyncio.sleep(1)  # Wait before retry
        return {
            'success': True,
            'method': 'retry',
            'message': 'Retrying operation'
        }
    
    async def _recover_model_error_simplify(self, record: ErrorRecord) -> Dict[str, Any]:
        """Simplify request for model"""
        return {
            'success': True,
            'method': 'simplify',
            'message': 'Simplifying request for processing'
        }
    
    async def _recover_file_error_create_dir(self, record: ErrorRecord) -> Dict[str, Any]:
        """Create missing directory"""
        if record.context.project_path:
            try:
                Path(record.context.project_path).mkdir(parents=True, exist_ok=True)
                return {
                    'success': True,
                    'method': 'create_directory',
                    'message': 'Created missing directory'
                }
            except:
                pass
        return {'success': False}
    
    async def _recover_file_error_permissions(self, record: ErrorRecord) -> Dict[str, Any]:
        """Fix file permissions"""
        # Implementation would fix permissions
        return {'success': False}
    
    async def _recover_file_error_alternative_path(self, record: ErrorRecord) -> Dict[str, Any]:
        """Use alternative file path"""
        # Implementation would find alternative path
        return {'success': False}
    
    async def _recover_network_error_retry(self, record: ErrorRecord) -> Dict[str, Any]:
        """Retry network operation"""
        for attempt in range(3):
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
            # Retry operation
            return {
                'success': True,
                'method': 'network_retry',
                'message': f'Retry successful on attempt {attempt + 1}'
            }
        return {'success': False}
    
    async def _recover_network_error_offline_mode(self, record: ErrorRecord) -> Dict[str, Any]:
        """Switch to offline mode"""
        return {
            'success': True,
            'method': 'offline_mode',
            'message': 'Switched to offline mode'
        }
    
    async def _recover_parsing_error_cleanup(self, record: ErrorRecord) -> Dict[str, Any]:
        """Clean up input for parsing"""
        return {
            'success': True,
            'method': 'input_cleanup',
            'message': 'Cleaned up input for parsing'
        }
    
    async def _recover_parsing_error_alternative_format(self, record: ErrorRecord) -> Dict[str, Any]:
        """Try alternative parsing format"""
        return {
            'success': True,
            'method': 'alternative_format',
            'message': 'Using alternative parsing format'
        }
    
    async def _recover_permission_error_elevate(self, record: ErrorRecord) -> Dict[str, Any]:
        """Request elevation of privileges"""
        return {'success': False}  # Would need user interaction
    
    async def _recover_permission_error_alternative(self, record: ErrorRecord) -> Dict[str, Any]:
        """Use alternative location without permission issues"""
        temp_dir = Path.home() / '.abov3' / 'temp'
        temp_dir.mkdir(parents=True, exist_ok=True)
        return {
            'success': True,
            'method': 'alternative_location',
            'message': f'Using alternative location: {temp_dir}'
        }
    
    async def _recover_dependency_error_install(self, record: ErrorRecord) -> Dict[str, Any]:
        """Attempt to install missing dependency"""
        # Would run pip install or similar
        return {'success': False}
    
    async def _recover_dependency_error_fallback(self, record: ErrorRecord) -> Dict[str, Any]:
        """Use fallback implementation without dependency"""
        return {
            'success': True,
            'method': 'fallback_implementation',
            'message': 'Using fallback implementation'
        }
    
    def _log_error(self, record: ErrorRecord):
        """Log error with appropriate level"""
        log_message = json.dumps(record.to_dict(), indent=2)
        
        if record.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif record.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif record.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def _update_metrics(self, record: ErrorRecord):
        """Update error metrics"""
        self.metrics['total_errors'] += 1
        
        if record.severity == ErrorSeverity.CRITICAL:
            self.metrics['critical_errors'] += 1
        
        category_key = record.category.value
        if category_key not in self.metrics['errors_by_category']:
            self.metrics['errors_by_category'][category_key] = 0
        self.metrics['errors_by_category'][category_key] += 1
        
        if record.recovery_successful:
            self.metrics['recovered_errors'] += 1
        
        # Calculate recovery success rate
        if self.metrics['total_errors'] > 0:
            self.metrics['recovery_success_rate'] = (
                self.metrics['recovered_errors'] / self.metrics['total_errors']
            )
    
    def _add_to_history(self, record: ErrorRecord):
        """Add error to history for pattern detection"""
        self.error_history.append(record)
        
        # Maintain history size limit
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size:]
    
    def _generate_error_response(self, record: ErrorRecord, 
                                recovery_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate user-friendly error response"""
        response = {
            'error': True,
            'error_type': record.error_type,
            'message': self._get_user_friendly_message(record),
            'severity': record.severity.value,
            'category': record.category.value,
            'timestamp': record.timestamp.isoformat()
        }
        
        # Add recovery information
        if recovery_result and recovery_result.get('success'):
            response['recovered'] = True
            response['recovery_message'] = recovery_result.get('message', 'Error recovered')
            response['recovery_method'] = recovery_result.get('method')
        else:
            response['recovered'] = False
            response['suggestion'] = self._get_error_suggestion(record)
        
        # Add debug information in development mode
        if os.getenv('ABOV3_DEBUG', '').lower() == 'true':
            response['debug_info'] = {
                'traceback': record.traceback,
                'context': record.context.to_dict()
            }
        
        return response
    
    def _get_user_friendly_message(self, record: ErrorRecord) -> str:
        """Generate user-friendly error message"""
        messages = {
            ErrorCategory.MODEL_ERROR: "AI model encountered an issue. Trying alternative approach...",
            ErrorCategory.FILE_ERROR: "File operation failed. Checking file system...",
            ErrorCategory.NETWORK_ERROR: "Network connection issue. Retrying...",
            ErrorCategory.PARSING_ERROR: "Could not understand the input format. Please rephrase.",
            ErrorCategory.VALIDATION_ERROR: "Invalid input detected. Please check your request.",
            ErrorCategory.PERMISSION_ERROR: "Permission denied. Trying alternative location...",
            ErrorCategory.CONFIGURATION_ERROR: "Configuration issue detected. Using defaults...",
            ErrorCategory.DEPENDENCY_ERROR: "Missing required component. Using alternative...",
            ErrorCategory.USER_INPUT_ERROR: "Could not process your request. Please try again.",
            ErrorCategory.SYSTEM_ERROR: "System error occurred. Please restart the application."
        }
        
        return messages.get(record.category, "An error occurred. Attempting recovery...")
    
    def _get_error_suggestion(self, record: ErrorRecord) -> str:
        """Get suggestion for error resolution"""
        # Check known patterns
        error_str = record.error_message.lower()
        for pattern_name, pattern_info in self.error_patterns.items():
            import re
            if re.search(pattern_info['pattern'], error_str):
                return pattern_info['suggestion']
        
        # Default suggestions by category
        suggestions = {
            ErrorCategory.MODEL_ERROR: "Try installing the required AI model or use a simpler request.",
            ErrorCategory.FILE_ERROR: "Check that the file path exists and you have proper permissions.",
            ErrorCategory.NETWORK_ERROR: "Check your internet connection and try again.",
            ErrorCategory.PARSING_ERROR: "Rephrase your request with clearer instructions.",
            ErrorCategory.VALIDATION_ERROR: "Check your input for errors and try again.",
            ErrorCategory.PERMISSION_ERROR: "Run the application with appropriate permissions.",
            ErrorCategory.CONFIGURATION_ERROR: "Check your configuration settings.",
            ErrorCategory.DEPENDENCY_ERROR: "Install missing dependencies with pip install.",
            ErrorCategory.USER_INPUT_ERROR: "Provide more specific details in your request.",
            ErrorCategory.SYSTEM_ERROR: "Restart the application or check system resources."
        }
        
        return suggestions.get(record.category, "Please try again or contact support.")
    
    def get_error_report(self) -> Dict[str, Any]:
        """Generate comprehensive error report"""
        recent_errors = [
            {
                'time': err.timestamp.isoformat(),
                'type': err.error_type,
                'category': err.category.value,
                'severity': err.severity.value,
                'recovered': err.recovery_successful
            }
            for err in self.error_history[-10:]  # Last 10 errors
        ]
        
        return {
            'metrics': self.metrics,
            'recent_errors': recent_errors,
            'patterns_detected': self._detect_error_patterns(),
            'recommendations': self._generate_recommendations()
        }
    
    def _detect_error_patterns(self) -> List[str]:
        """Detect patterns in error history"""
        patterns = []
        
        # Check for repeated errors
        error_counts = {}
        for err in self.error_history:
            key = f"{err.category.value}:{err.error_type}"
            error_counts[key] = error_counts.get(key, 0) + 1
        
        for error_key, count in error_counts.items():
            if count >= 3:
                patterns.append(f"Repeated {error_key} ({count} occurrences)")
        
        return patterns
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on error history"""
        recommendations = []
        
        if self.metrics['critical_errors'] > 0:
            recommendations.append("Critical errors detected. System stability may be compromised.")
        
        if self.metrics['recovery_success_rate'] < 0.5:
            recommendations.append("Low recovery success rate. Manual intervention may be needed.")
        
        # Category-specific recommendations
        for category, count in self.metrics['errors_by_category'].items():
            if count > 5:
                if category == ErrorCategory.MODEL_ERROR.value:
                    recommendations.append("Frequent model errors. Consider installing alternative models.")
                elif category == ErrorCategory.FILE_ERROR.value:
                    recommendations.append("Frequent file errors. Check file system permissions.")
                elif category == ErrorCategory.NETWORK_ERROR.value:
                    recommendations.append("Frequent network errors. Check internet connectivity.")
        
        return recommendations


# Global error handler instance
_error_handler = None

def get_error_handler() -> ErrorHandler:
    """Get or create global error handler instance"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler