"""
ABOV3 Genesis Core Package
Core AI processing, assistant functionality, and enhanced debugging capabilities
"""

# Original imports
from .assistant import Assistant
from .ollama_client import OllamaClient

# Import enhanced debugging components
from .enterprise_debugger import (
    EnterpriseDebugEngine,
    get_debug_engine,
    debug,
    analyze_error,
    ask_debug,
    profile
)

from .enhanced_ml_debugger import (
    EnhancedMLDebugger,
    get_enhanced_debugger,
    debug_with_ml,
    ask_debug_question,
    generate_tests
)

from .error_resolution_engine import (
    ErrorResolutionEngine,
    ErrorSeverity,
    ResolutionStatus,
    ErrorCategory,
    get_resolution_engine,
    track_error,
    resolve_error,
    auto_resolve,
    get_error_statistics
)

from .issue_tracking_system import (
    IssueTrackingSystem,
    Issue,
    IssueType,
    IssuePriority,
    IssueStatus,
    get_issue_tracker,
    create_issue,
    get_issue
)

from .enhanced_debug_integration import (
    EnhancedDebugIntegration,
    DebugMode,
    IntegrationEvent,
    get_debug_integration,
    handle_error,
    debug_query,
    get_debug_report
)

# Try to import secure debugger if available
try:
    from .secure_debugger import (
        SecureEnterpriseDebugger,
        get_secure_debugger,
        SECURE_DEBUG_AVAILABLE
    )
except ImportError:
    SECURE_DEBUG_AVAILABLE = False

__all__ = [
    # Original exports
    "Assistant",
    "OllamaClient",
    
    # Enterprise debugging
    'EnterpriseDebugEngine',
    'get_debug_engine',
    'debug',
    'analyze_error',
    'ask_debug',
    'profile',
    
    # ML-enhanced debugging
    'EnhancedMLDebugger',
    'get_enhanced_debugger',
    'debug_with_ml',
    'ask_debug_question',
    'generate_tests',
    
    # Error resolution
    'ErrorResolutionEngine',
    'ErrorSeverity',
    'ResolutionStatus',
    'ErrorCategory',
    'get_resolution_engine',
    'track_error',
    'resolve_error',
    'auto_resolve',
    'get_error_statistics',
    
    # Issue tracking
    'IssueTrackingSystem',
    'Issue',
    'IssueType',
    'IssuePriority',
    'IssueStatus',
    'get_issue_tracker',
    'create_issue',
    'get_issue',
    
    # Integrated debugging
    'EnhancedDebugIntegration',
    'DebugMode',
    'IntegrationEvent',
    'get_debug_integration',
    'handle_error',
    'debug_query',
    'get_debug_report',
    
    # Security flag
    'SECURE_DEBUG_AVAILABLE'
]

# Add secure debugger to exports if available
if SECURE_DEBUG_AVAILABLE:
    __all__.extend([
        'SecureEnterpriseDebugger',
        'get_secure_debugger'
    ])