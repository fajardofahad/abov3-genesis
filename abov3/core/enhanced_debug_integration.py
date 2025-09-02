"""
ABOV3 Genesis - Enhanced Debug Integration Module
Seamless integration of all debugging, error resolution, and issue tracking components
"""

import asyncio
import logging
import threading
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum, auto
import json
import traceback

# Import existing components
from .enterprise_debugger import (
    EnterpriseDebugEngine,
    IntelligentErrorAnalyzer,
    InteractiveDebugger,
    NaturalLanguageDebugger,
    get_debug_engine
)

from .enhanced_ml_debugger import (
    EnhancedMLDebugger,
    get_enhanced_debugger
)

# Try importing error resolution engine
try:
    from .error_resolution_engine import (
        ErrorResolutionEngine,
        ErrorSeverity,
        ResolutionStatus,
        ErrorCategory,
        get_resolution_engine
    )
    HAS_RESOLUTION_ENGINE = True
except ImportError:
    # Fallback definitions if module doesn't exist
    HAS_RESOLUTION_ENGINE = False
    from enum import Enum
    
    class ErrorSeverity(Enum):
        INFO = 1
        LOW = 2
        MEDIUM = 3
        HIGH = 4
        CRITICAL = 5
    
    class ResolutionStatus(Enum):
        PENDING = 1
        IN_PROGRESS = 2
        RESOLVED = 3
        FAILED = 4
    
    class ErrorCategory(Enum):
        SYNTAX = 1
        RUNTIME = 2
        LOGIC = 3
        CONFIGURATION = 4
        DEPENDENCY = 5
    
    def get_resolution_engine(project_path):
        return None

# Try importing issue tracking system
try:
    from .issue_tracking_system import (
        IssueTrackingSystem,
        Issue,
        IssueType,
        IssuePriority,
        IssueStatus,
        get_issue_tracker
    )
    HAS_ISSUE_TRACKER = True
except ImportError:
    # Fallback definitions if module doesn't exist
    HAS_ISSUE_TRACKER = False
    from enum import Enum
    from dataclasses import dataclass
    
    class IssueType(Enum):
        BUG = 1
        FEATURE = 2
        IMPROVEMENT = 3
        TASK = 4
    
    class IssuePriority(Enum):
        TRIVIAL = 1
        LOW = 2
        MEDIUM = 3
        HIGH = 4
        CRITICAL = 5
    
    class IssueStatus(Enum):
        OPEN = 1
        IN_PROGRESS = 2
        RESOLVED = 3
        CLOSED = 4
    
    @dataclass
    class Issue:
        issue_id: str
        title: str
        description: str
        issue_type: IssueType
        priority: IssuePriority
        status: IssueStatus
    
    def get_issue_tracker(project_path):
        return None

# Optional secure debugger
try:
    from .secure_debugger import (
        SecureEnterpriseDebugger,
        get_secure_debugger
    )
    HAS_SECURE_DEBUG = True
except ImportError:
    HAS_SECURE_DEBUG = False


class DebugMode(Enum):
    """Debug operation modes"""
    DEVELOPMENT = auto()
    TESTING = auto()
    STAGING = auto()
    PRODUCTION = auto()
    EMERGENCY = auto()


class IntegrationEvent(Enum):
    """Events for component integration"""
    ERROR_DETECTED = auto()
    ERROR_RESOLVED = auto()
    ISSUE_CREATED = auto()
    ISSUE_UPDATED = auto()
    PATTERN_LEARNED = auto()
    ALERT_TRIGGERED = auto()
    AUTO_FIX_APPLIED = auto()
    MANUAL_INTERVENTION = auto()


@dataclass
class DebugContext:
    """Unified debug context across all components"""
    session_id: str
    mode: DebugMode
    user_id: str
    project_path: Path
    timestamp: datetime
    active_errors: List[str] = field(default_factory=list)
    active_issues: List[str] = field(default_factory=list)
    ml_insights: Dict[str, Any] = field(default_factory=dict)
    security_context: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    automation_enabled: bool = True
    learning_enabled: bool = True


@dataclass
class ResolutionWorkflow:
    """Workflow for error resolution"""
    workflow_id: str
    name: str
    steps: List[Dict[str, Any]]
    triggers: List[IntegrationEvent]
    conditions: Dict[str, Any]
    actions: List[Callable]
    enabled: bool = True


class EnhancedDebugIntegration:
    """
    Master integration class for all debugging components
    Provides Claude-level intelligent debugging with seamless component orchestration
    """
    
    def __init__(self, project_path: Optional[Path] = None, mode: DebugMode = DebugMode.DEVELOPMENT):
        self.project_path = project_path or Path.cwd()
        self.mode = mode
        
        # Initialize all components
        self.debug_engine = get_debug_engine()
        self.ml_debugger = get_enhanced_debugger()
        
        # Initialize optional components
        self.resolution_engine = None
        if HAS_RESOLUTION_ENGINE:
            self.resolution_engine = get_resolution_engine(self.project_path)
        
        self.issue_tracker = None
        if HAS_ISSUE_TRACKER:
            self.issue_tracker = get_issue_tracker(self.project_path)
        
        # Optional secure debugger
        self.secure_debugger = None
        if HAS_SECURE_DEBUG:
            asyncio.create_task(self._init_secure_debugger())
        
        # Integration state
        self.active_contexts: Dict[str, DebugContext] = {}
        self.workflows: Dict[str, ResolutionWorkflow] = {}
        self.event_handlers: Dict[IntegrationEvent, List[Callable]] = {
            event: [] for event in IntegrationEvent
        }
        
        # Configuration
        self.config = self._load_configuration()
        
        # Monitoring
        self.monitoring_enabled = False
        self.monitor_thread = None
        self._stop_monitoring = threading.Event()
        
        # Metrics
        self.metrics = {
            'total_errors_handled': 0,
            'auto_resolutions': 0,
            'issues_created': 0,
            'ml_predictions_made': 0,
            'average_resolution_time': 0.0,
            'success_rate': 0.0
        }
        
        # Setup
        self._setup_logging()
        self._setup_default_workflows()
        self._register_event_handlers()
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        self.logger = logging.getLogger('abov3.debug_integration')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load integration configuration"""
        config = {
            'auto_resolution_enabled': True,
            'ml_analysis_enabled': True,
            'issue_creation_threshold': ErrorSeverity.HIGH,
            'pattern_learning_enabled': True,
            'security_checks_enabled': self.mode == DebugMode.PRODUCTION,
            'performance_profiling_enabled': True,
            'natural_language_enabled': True,
            'max_auto_fix_attempts': 3,
            'alert_on_critical': True,
            'batch_processing_size': 10
        }
        
        # Adjust based on mode
        if self.mode == DebugMode.PRODUCTION:
            config['auto_resolution_enabled'] = False  # Manual approval required
            config['security_checks_enabled'] = True
        elif self.mode == DebugMode.EMERGENCY:
            config['auto_resolution_enabled'] = True
            config['max_auto_fix_attempts'] = 5
            config['alert_on_critical'] = True
        
        return config
    
    async def _init_secure_debugger(self):
        """Initialize secure debugger asynchronously"""
        if HAS_SECURE_DEBUG:
            try:
                self.secure_debugger = await get_secure_debugger(
                    self.project_path,
                    enable_security=self.config['security_checks_enabled']
                )
                self.logger.info("Secure debugger initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize secure debugger: {e}")
    
    def _setup_default_workflows(self):
        """Setup default resolution workflows"""
        # Auto-resolution workflow
        auto_resolution = ResolutionWorkflow(
            workflow_id='auto_resolution',
            name='Automatic Error Resolution',
            steps=[
                {'action': 'analyze_error', 'component': 'ml_debugger'},
                {'action': 'find_pattern', 'component': 'resolution_engine'},
                {'action': 'generate_fix', 'component': 'resolution_engine'},
                {'action': 'validate_fix', 'component': 'debug_engine'},
                {'action': 'apply_fix', 'component': 'resolution_engine'}
            ],
            triggers=[IntegrationEvent.ERROR_DETECTED],
            conditions={'severity': ['LOW', 'MEDIUM']},
            actions=[self._execute_auto_resolution],
            enabled=self.config['auto_resolution_enabled']
        )
        self.workflows['auto_resolution'] = auto_resolution
        
        # Issue creation workflow
        issue_creation = ResolutionWorkflow(
            workflow_id='issue_creation',
            name='Automatic Issue Creation',
            steps=[
                {'action': 'assess_severity', 'component': 'resolution_engine'},
                {'action': 'check_duplicates', 'component': 'issue_tracker'},
                {'action': 'create_issue', 'component': 'issue_tracker'},
                {'action': 'link_error', 'component': 'issue_tracker'}
            ],
            triggers=[IntegrationEvent.ERROR_DETECTED],
            conditions={'severity': ['HIGH', 'CRITICAL']},
            actions=[self._execute_issue_creation],
            enabled=True
        )
        self.workflows['issue_creation'] = issue_creation
        
        # Learning workflow
        learning_workflow = ResolutionWorkflow(
            workflow_id='pattern_learning',
            name='Pattern Learning Workflow',
            steps=[
                {'action': 'extract_pattern', 'component': 'resolution_engine'},
                {'action': 'update_ml_model', 'component': 'ml_debugger'},
                {'action': 'store_pattern', 'component': 'resolution_engine'}
            ],
            triggers=[IntegrationEvent.ERROR_RESOLVED],
            conditions={},
            actions=[self._execute_learning],
            enabled=self.config['pattern_learning_enabled']
        )
        self.workflows['pattern_learning'] = learning_workflow
    
    def _register_event_handlers(self):
        """Register event handlers for component integration"""
        # Error detection handlers
        self.register_event_handler(
            IntegrationEvent.ERROR_DETECTED,
            self._handle_error_detection
        )
        
        # Resolution handlers
        self.register_event_handler(
            IntegrationEvent.ERROR_RESOLVED,
            self._handle_error_resolution
        )
        
        # Issue handlers
        self.register_event_handler(
            IntegrationEvent.ISSUE_CREATED,
            self._handle_issue_creation
        )
        
        # Alert handlers
        self.register_event_handler(
            IntegrationEvent.ALERT_TRIGGERED,
            self._handle_alert
        )
    
    def create_debug_context(self, user_id: str = 'system') -> DebugContext:
        """Create a new debug context"""
        import uuid
        session_id = str(uuid.uuid4())
        
        context = DebugContext(
            session_id=session_id,
            mode=self.mode,
            user_id=user_id,
            project_path=self.project_path,
            timestamp=datetime.now(),
            automation_enabled=self.config['auto_resolution_enabled'],
            learning_enabled=self.config['pattern_learning_enabled']
        )
        
        self.active_contexts[session_id] = context
        
        # Create ML debug session
        if self.ml_debugger:
            ml_session_id = self.ml_debugger.create_debug_session(
                code='',  # Will be updated with actual code
                session_name=f"Integration_{session_id}"
            )
            context.ml_insights['ml_session_id'] = ml_session_id
        
        # Create debug engine session
        engine_session_id = self.debug_engine.create_debug_session(
            name=f"Integration_{session_id}"
        )
        context.ml_insights['engine_session_id'] = engine_session_id
        
        self.logger.info(f"Created debug context: {session_id}")
        
        return context
    
    def handle_error_sync(self, error: Exception, context_id: Optional[str] = None, 
                         **kwargs) -> Dict[str, Any]:
        """
        Main entry point for error handling
        Orchestrates all components for comprehensive error resolution
        """
        # Get or create context
        if context_id and context_id in self.active_contexts:
            context = self.active_contexts[context_id]
        else:
            context = self.create_debug_context()
            context_id = context.session_id
        
        result = {
            'context_id': context_id,
            'error_id': None,
            'ml_analysis': {},
            'resolution_result': None,
            'issue_created': None,
            'recommendations': [],
            'success': False
        }
        
        try:
            # Track error in resolution engine
            error_id = self.resolution_engine.track_error(error, **kwargs)
            result['error_id'] = error_id
            context.active_errors.append(error_id)
            
            # ML analysis
            if self.config['ml_analysis_enabled']:
                ml_analysis = self.ml_debugger.analyze_error_with_ml(
                    error,
                    code_context=kwargs.get('code_context', ''),
                    **kwargs
                )
                result['ml_analysis'] = ml_analysis
                context.ml_insights['last_analysis'] = ml_analysis
            
            # Basic analysis
            basic_analysis = self.debug_engine.analyze_exception(error, **kwargs)
            
            # Trigger error detected event
            self._trigger_event(IntegrationEvent.ERROR_DETECTED, {
                'error': error,
                'error_id': error_id,
                'context': context,
                'analysis': {**basic_analysis, **result['ml_analysis']}
            })
            
            # Assess severity
            severity = self._assess_integrated_severity(error, basic_analysis, result['ml_analysis'])
            
            # Auto-resolution attempt
            if self.config['auto_resolution_enabled'] and severity.value <= ErrorSeverity.MEDIUM.value:
                resolution_result = self._attempt_auto_resolution(
                    error_id, error, context
                )
                result['resolution_result'] = resolution_result
                
                if resolution_result and resolution_result.success:
                    result['success'] = True
                    self._trigger_event(IntegrationEvent.ERROR_RESOLVED, {
                        'error_id': error_id,
                        'resolution': resolution_result,
                        'context': context
                    })
            
            # Create issue if needed
            if severity.value >= self.config['issue_creation_threshold'].value:
                issue = self._create_issue_from_error(
                    error, error_id, severity, context
                )
                if issue:
                    result['issue_created'] = issue.issue_id
                    context.active_issues.append(issue.issue_id)
            
            # Generate recommendations
            result['recommendations'] = self._generate_recommendations(
                error, basic_analysis, result['ml_analysis'], context
            )
            
            # Update metrics
            self.metrics['total_errors_handled'] += 1
            if result['success']:
                self.metrics['auto_resolutions'] += 1
            
            # Calculate success rate
            if self.metrics['total_errors_handled'] > 0:
                self.metrics['success_rate'] = (
                    self.metrics['auto_resolutions'] / 
                    self.metrics['total_errors_handled']
                )
            
        except Exception as e:
            self.logger.error(f"Error handling failed: {e}")
            result['error'] = str(e)
        
        return result
    
    def _assess_integrated_severity(self, error: Exception, 
                                   basic_analysis: Dict[str, Any],
                                   ml_analysis: Dict[str, Any]) -> ErrorSeverity:
        """Assess severity using all available information"""
        # Get basic severity
        basic_severity = basic_analysis.get('severity', 3)
        
        # Get ML confidence
        ml_confidence = ml_analysis.get('confidence', 0.5)
        
        # Adjust based on ML insights
        if ml_confidence > 0.8:
            # High confidence in ML analysis
            ml_severity = ml_analysis.get('basic_analysis', {}).get('severity', basic_severity)
            # Weight ML analysis higher
            final_severity = int((basic_severity * 0.3 + ml_severity * 0.7))
        else:
            final_severity = basic_severity
        
        # Map to enum
        severity_map = {
            1: ErrorSeverity.LOW,
            2: ErrorSeverity.LOW,
            3: ErrorSeverity.MEDIUM,
            4: ErrorSeverity.HIGH,
            5: ErrorSeverity.CRITICAL
        }
        
        return severity_map.get(final_severity, ErrorSeverity.MEDIUM)
    
    def _attempt_auto_resolution(self, error_id: str, error: Exception, 
                                context: DebugContext) -> Optional[Any]:
        """Attempt automatic error resolution"""
        try:
            # Get fix suggestions from ML
            ml_fixes = []
            if context.ml_insights.get('last_analysis'):
                ml_fixes = context.ml_insights['last_analysis'].get('fix_suggestions', [])
            
            # Attempt resolution
            resolution_result = self.resolution_engine.resolve_error(error_id)
            
            # If basic resolution fails, try ML suggestions
            if not resolution_result.success and ml_fixes:
                for fix in ml_fixes[:self.config['max_auto_fix_attempts']]:
                    if fix.get('confidence', 0) > 0.7:
                        # Apply ML fix (simplified - would need actual implementation)
                        self.logger.info(f"Attempting ML fix: {fix.get('explanation')}")
                        # In production, this would apply the actual fix
                        break
            
            return resolution_result
            
        except Exception as e:
            self.logger.error(f"Auto-resolution failed: {e}")
            return None
    
    def _create_issue_from_error(self, error: Exception, error_id: str,
                                severity: ErrorSeverity, 
                                context: DebugContext) -> Optional[Issue]:
        """Create issue from error"""
        try:
            # Map severity to priority
            priority_map = {
                ErrorSeverity.CRITICAL: IssuePriority.CRITICAL,
                ErrorSeverity.HIGH: IssuePriority.HIGH,
                ErrorSeverity.MEDIUM: IssuePriority.MEDIUM,
                ErrorSeverity.LOW: IssuePriority.LOW,
                ErrorSeverity.INFO: IssuePriority.TRIVIAL
            }
            
            priority = priority_map.get(severity, IssuePriority.MEDIUM)
            
            # Create issue
            issue = self.issue_tracker.create_issue(
                title=f"{type(error).__name__}: {str(error)[:100]}",
                description=self._format_error_description(error, error_id, context),
                issue_type=IssueType.BUG,
                priority=priority,
                created_by=context.user_id,
                error_ids=[error_id],
                components=self._identify_components(error),
                labels=self._generate_labels(error, severity)
            )
            
            self._trigger_event(IntegrationEvent.ISSUE_CREATED, {
                'issue': issue,
                'error_id': error_id,
                'context': context
            })
            
            self.metrics['issues_created'] += 1
            
            return issue
            
        except Exception as e:
            self.logger.error(f"Issue creation failed: {e}")
            return None
    
    def _format_error_description(self, error: Exception, error_id: str,
                                 context: DebugContext) -> str:
        """Format error description for issue"""
        description = f"""
## Error Details
- **Error ID**: {error_id}
- **Type**: {type(error).__name__}
- **Message**: {str(error)}
- **Session**: {context.session_id}
- **Timestamp**: {datetime.now().isoformat()}

## Stack Trace
```
{traceback.format_exc()}
```

## Context
- **Mode**: {context.mode.name}
- **User**: {context.user_id}
- **Project**: {context.project_path}

## ML Analysis
{json.dumps(context.ml_insights.get('last_analysis', {}), indent=2)[:1000]}

## Active Errors
{', '.join(context.active_errors)}

## Automation Status
- **Auto-resolution**: {'Enabled' if context.automation_enabled else 'Disabled'}
- **Learning**: {'Enabled' if context.learning_enabled else 'Disabled'}
"""
        return description
    
    def _identify_components(self, error: Exception) -> List[str]:
        """Identify affected components from error"""
        components = []
        
        error_str = str(error).lower()
        type_name = type(error).__name__.lower()
        
        # Component mapping
        component_map = {
            'import': ['dependencies', 'modules'],
            'attribute': ['api', 'interface'],
            'key': ['data', 'configuration'],
            'index': ['data', 'arrays'],
            'type': ['validation', 'types'],
            'file': ['io', 'filesystem'],
            'connection': ['network', 'database'],
            'memory': ['performance', 'resources'],
            'permission': ['security', 'access']
        }
        
        for keyword, comps in component_map.items():
            if keyword in error_str or keyword in type_name:
                components.extend(comps)
        
        return list(set(components))
    
    def _generate_labels(self, error: Exception, severity: ErrorSeverity) -> List[str]:
        """Generate labels for issue"""
        labels = [
            f"error:{type(error).__name__}",
            f"severity:{severity.name.lower()}",
            f"auto-generated"
        ]
        
        if self.mode == DebugMode.PRODUCTION:
            labels.append("production")
        elif self.mode == DebugMode.EMERGENCY:
            labels.append("emergency")
        
        return labels
    
    def _generate_recommendations(self, error: Exception, 
                                 basic_analysis: Dict[str, Any],
                                 ml_analysis: Dict[str, Any],
                                 context: DebugContext) -> List[str]:
        """Generate comprehensive recommendations"""
        recommendations = []
        
        # From basic analysis
        if 'solutions' in basic_analysis:
            recommendations.extend(basic_analysis['solutions'][:3])
        
        # From ML analysis
        if 'fix_suggestions' in ml_analysis:
            for fix in ml_analysis['fix_suggestions'][:2]:
                if fix.get('explanation'):
                    recommendations.append(fix['explanation'])
        
        # From prevention suggestions
        if 'prevention_suggestions' in basic_analysis:
            recommendations.extend(basic_analysis['prevention_suggestions'][:2])
        
        # Context-specific recommendations
        if context.mode == DebugMode.PRODUCTION:
            recommendations.append("Consider implementing additional monitoring")
            recommendations.append("Review deployment rollback procedures")
        
        return recommendations
    
    def _execute_auto_resolution(self, event_data: Dict[str, Any]):
        """Execute auto-resolution workflow"""
        error = event_data.get('error')
        error_id = event_data.get('error_id')
        context = event_data.get('context')
        
        if not all([error, error_id, context]):
            return
        
        # Execute workflow steps
        self.logger.info(f"Executing auto-resolution for error {error_id}")
        
        # This would execute the actual workflow steps
        # For now, it's handled in _attempt_auto_resolution
    
    def _execute_issue_creation(self, event_data: Dict[str, Any]):
        """Execute issue creation workflow"""
        # Already handled in _create_issue_from_error
        pass
    
    def _execute_learning(self, event_data: Dict[str, Any]):
        """Execute learning workflow"""
        error_id = event_data.get('error_id')
        resolution = event_data.get('resolution')
        
        if not all([error_id, resolution]):
            return
        
        self.logger.info(f"Learning from resolution of error {error_id}")
        
        # Update ML models with resolution outcome
        if self.ml_debugger and hasattr(self.ml_debugger, 'learning_system'):
            # This would update the learning system
            pass
    
    def _handle_error_detection(self, event_data: Dict[str, Any]):
        """Handle error detection event"""
        self.logger.info(f"Error detected: {event_data.get('error_id')}")
        
        # Check if alert needed
        analysis = event_data.get('analysis', {})
        if analysis.get('severity', 0) >= 4 and self.config['alert_on_critical']:
            self._trigger_event(IntegrationEvent.ALERT_TRIGGERED, event_data)
    
    def _handle_error_resolution(self, event_data: Dict[str, Any]):
        """Handle error resolution event"""
        self.logger.info(f"Error resolved: {event_data.get('error_id')}")
        
        # Update related issue if exists
        context = event_data.get('context')
        if context and context.active_issues:
            for issue_id in context.active_issues:
                self.issue_tracker.add_comment(
                    issue_id,
                    author='system',
                    content=f"Associated error {event_data.get('error_id')} has been resolved.",
                    is_internal=False
                )
    
    def _handle_issue_creation(self, event_data: Dict[str, Any]):
        """Handle issue creation event"""
        issue = event_data.get('issue')
        if issue:
            self.logger.info(f"Issue created: {issue.issue_id}")
    
    def _handle_alert(self, event_data: Dict[str, Any]):
        """Handle alert event"""
        self.logger.warning(f"ALERT: Critical error detected - {event_data.get('error_id')}")
        
        # In production, this would send actual alerts (email, Slack, etc.)
    
    def register_event_handler(self, event: IntegrationEvent, handler: Callable):
        """Register event handler"""
        self.event_handlers[event].append(handler)
    
    def register_context(self, context: DebugContext):
        """Register a debug context"""
        self.active_contexts[context.session_id] = context
        self.logger.info(f"Registered debug context: {context.session_id}")
    
    def enable_automatic_resolution(self):
        """Enable automatic error resolution"""
        self.config['auto_resolution_enabled'] = True
        if 'auto_resolution' in self.workflows:
            self.workflows['auto_resolution'].enabled = True
        self.logger.info("Automatic error resolution enabled")
    
    def start_monitoring(self):
        """Start system monitoring"""
        if not self.monitoring_enabled:
            self.monitoring_enabled = True
            self._stop_monitoring.clear()
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.info("System monitoring started")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        import time
        while not self._stop_monitoring.is_set():
            try:
                # Monitor system health
                if hasattr(self, 'debug_engine'):
                    health = self.debug_engine.get_system_health()
                    if health.get('memory_usage', 0) > 90:
                        self.logger.warning("High memory usage detected")
                    if health.get('cpu_usage', 0) > 90:
                        self.logger.warning("High CPU usage detected")
                
                # Check for hanging operations
                for context_id, context in self.active_contexts.items():
                    if context.active_errors:
                        age = (datetime.now() - context.timestamp).total_seconds()
                        if age > 300:  # 5 minutes
                            self.logger.warning(f"Long-running context detected: {context_id}")
                
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(30)
    
    async def track_operation(self, operation_name: str, details: Dict[str, Any]):
        """Track an operation for debugging"""
        try:
            # Log operation
            self.logger.debug(f"Operation: {operation_name} - {details}")
            
            # Track in debug engine if available
            if hasattr(self, 'debug_engine'):
                self.debug_engine.track_operation(operation_name, details)
            
            # Update metrics
            if 'operations' not in self.metrics:
                self.metrics['operations'] = {}
            if operation_name not in self.metrics['operations']:
                self.metrics['operations'][operation_name] = 0
            self.metrics['operations'][operation_name] += 1
            
        except Exception as e:
            self.logger.error(f"Failed to track operation: {e}")
    
    @contextmanager
    def monitored_operation(self, operation_name: str):
        """Context manager for monitoring operations"""
        start_time = datetime.now()
        operation_id = f"{operation_name}_{start_time.timestamp()}"
        
        try:
            # Start monitoring
            self.logger.debug(f"Starting operation: {operation_name}")
            if hasattr(self, 'debug_engine'):
                self.debug_engine.start_operation_monitoring(operation_id)
            
            yield operation_id
            
            # Success
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.debug(f"Operation {operation_name} completed in {duration:.2f}s")
            
        except Exception as e:
            # Failure
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Operation {operation_name} failed after {duration:.2f}s: {e}")
            raise
        
        finally:
            # Stop monitoring
            if hasattr(self, 'debug_engine'):
                self.debug_engine.stop_operation_monitoring(operation_id)
    
    async def analyze_response(self, response: Any):
        """Analyze a response for potential issues"""
        try:
            if not response:
                self.logger.warning("Empty response detected")
                return
            
            # Check response size
            response_str = str(response)
            if len(response_str) > 1000000:  # 1MB
                self.logger.warning(f"Large response detected: {len(response_str)} bytes")
            
            # Check for common error patterns
            error_patterns = [
                'error', 'exception', 'failed', 'unable to', 'could not',
                'traceback', 'stack trace', 'null pointer', 'undefined'
            ]
            
            response_lower = response_str.lower()
            for pattern in error_patterns:
                if pattern in response_lower:
                    self.logger.debug(f"Potential issue pattern found: {pattern}")
            
            # ML analysis if available
            if self.ml_debugger and self.config['ml_analysis_enabled']:
                ml_analysis = self.ml_debugger.analyze_text(response_str[:5000])
                if ml_analysis.get('issues_detected'):
                    self.logger.warning(f"ML detected issues in response: {ml_analysis['issues']}")
            
        except Exception as e:
            self.logger.error(f"Failed to analyze response: {e}")
    
    async def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Async wrapper for error handling"""
        # Convert to sync call for compatibility  
        return self.handle_error_sync(error, context_id=None)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of the debug session"""
        summary = {
            'errors_handled': self.metrics.get('total_errors_handled', 0),
            'auto_resolutions': self.metrics.get('auto_resolutions', 0),
            'issues_created': self.metrics.get('issues_created', 0),
            'patterns_learned': len(self.resolution_engine.error_patterns) if hasattr(self, 'resolution_engine') else 0,
            'active_contexts': len(self.active_contexts),
            'success_rate': self.metrics.get('success_rate', 0.0),
            'operations': self.metrics.get('operations', {})
        }
        
        # Add ML insights
        if self.ml_debugger:
            ml_stats = self.ml_debugger.get_statistics()
            summary['ml_predictions'] = ml_stats.get('total_predictions', 0)
            summary['ml_accuracy'] = ml_stats.get('accuracy', 0.0)
        
        # Add issue tracker stats
        if self.issue_tracker:
            issue_stats = self.issue_tracker.get_statistics()
            summary['total_issues'] = issue_stats.get('total_issues', 0)
            summary['open_issues'] = issue_stats.get('open_issues', 0)
        
        return summary
    
    async def shutdown(self):
        """Shutdown the debug integration"""
        self.logger.info("Shutting down debug integration")
        
        # Stop monitoring
        if self.monitoring_enabled:
            self.monitoring_enabled = False
            self._stop_monitoring.set()
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)
        
        # Close all active contexts
        for context_id in list(self.active_contexts.keys()):
            self.close_context(context_id)
        
        # Shutdown components
        if hasattr(self, 'ml_debugger') and hasattr(self.ml_debugger, 'cleanup'):
            self.ml_debugger.cleanup()
        
        if hasattr(self, 'debug_engine') and hasattr(self.debug_engine, 'cleanup'):
            self.debug_engine.cleanup()
        
        if hasattr(self, 'resolution_engine') and hasattr(self.resolution_engine, 'shutdown'):
            await self.resolution_engine.shutdown()
        
        if hasattr(self, 'issue_tracker') and hasattr(self.issue_tracker, 'close'):
            self.issue_tracker.close()
        
        self.logger.info("Debug integration shutdown complete")
    
    def close_context(self, context_id: str):
        """Close a debug context"""
        if context_id in self.active_contexts:
            context = self.active_contexts[context_id]
            
            # Close ML session
            if context.ml_insights.get('ml_session_id'):
                self.ml_debugger.close_session(context.ml_insights['ml_session_id'])
            
            # Close engine session  
            if context.ml_insights.get('engine_session_id'):
                self.debug_engine.close_session(context.ml_insights['engine_session_id'])
            
            del self.active_contexts[context_id]
            self.logger.info(f"Closed debug context: {context_id}")
    
    def _trigger_event(self, event: IntegrationEvent, data: Dict[str, Any]):
        """Trigger an integration event"""
        for handler in self.event_handlers.get(event, []):
            try:
                handler(data)
            except Exception as e:
                self.logger.error(f"Event handler error for {event}: {e}")
    
    def _trigger_event(self, event: IntegrationEvent, data: Dict[str, Any]):
        """Trigger event and execute handlers"""
        for handler in self.event_handlers[event]:
            try:
                handler(data)
            except Exception as e:
                self.logger.error(f"Event handler failed: {e}")
        
        # Execute workflows triggered by this event
        for workflow in self.workflows.values():
            if event in workflow.triggers and workflow.enabled:
                if self._check_workflow_conditions(workflow, data):
                    for action in workflow.actions:
                        try:
                            action(data)
                        except Exception as e:
                            self.logger.error(f"Workflow action failed: {e}")
    
    def _check_workflow_conditions(self, workflow: ResolutionWorkflow, 
                                  data: Dict[str, Any]) -> bool:
        """Check if workflow conditions are met"""
        # Simple condition checking - would be more sophisticated in production
        if 'severity' in workflow.conditions:
            analysis = data.get('analysis', {})
            severity = analysis.get('severity', 0)
            severity_name = ErrorSeverity(severity).name if severity else 'LOW'
            return severity_name in workflow.conditions['severity']
        
        return True
    
    def natural_language_query(self, query: str, context_id: Optional[str] = None) -> str:
        """Process natural language debug query"""
        context = None
        if context_id and context_id in self.active_contexts:
            context = self.active_contexts[context_id]
        
        # Try ML debugger first
        if self.ml_debugger:
            result = self.ml_debugger.ask_natural_language(
                query,
                code_context=context.ml_insights.get('code_context') if context else None
            )
            if result and 'response' in result:
                return result['response']
        
        # Fallback to basic NL debugger
        return self.debug_engine.query(query)
    
    def get_integrated_statistics(self) -> Dict[str, Any]:
        """Get statistics from all components"""
        stats = {
            'integration_metrics': self.metrics,
            'active_contexts': len(self.active_contexts),
            'workflows': {
                wf_id: {
                    'name': wf.name,
                    'enabled': wf.enabled,
                    'triggers': [t.name for t in wf.triggers]
                }
                for wf_id, wf in self.workflows.items()
            }
        }
        
        # Add component statistics
        stats['debug_engine'] = self.debug_engine.get_debug_report()
        stats['resolution_engine'] = self.resolution_engine.get_statistics()
        stats['issue_tracker'] = self.issue_tracker.get_statistics()
        
        if self.ml_debugger:
            stats['ml_debugger'] = self.ml_debugger.get_learning_insights()
        
        return stats
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive debug report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'mode': self.mode.name,
            'statistics': self.get_integrated_statistics(),
            'recent_errors': self.resolution_engine.get_error_history(limit=20),
            'open_issues': self.issue_tracker.search_issues(status=IssueStatus.OPEN),
            'recommendations': self._generate_system_recommendations()
        }
        
        return report
    
    def _generate_system_recommendations(self) -> List[str]:
        """Generate system-level recommendations"""
        recommendations = []
        
        # Based on metrics
        if self.metrics['success_rate'] < 0.5:
            recommendations.append("Low auto-resolution success rate - consider updating patterns")
        
        if self.metrics['total_errors_handled'] > 100:
            recommendations.append("High error volume - review common error patterns")
        
        # Based on mode
        if self.mode == DebugMode.PRODUCTION:
            recommendations.append("Production mode - ensure monitoring and alerting are active")
        
        return recommendations


# Global instance
_debug_integration = None

def get_debug_integration(project_path: Optional[Path] = None, 
                         mode: DebugMode = DebugMode.DEVELOPMENT) -> EnhancedDebugIntegration:
    """Get global debug integration instance"""
    global _debug_integration
    if _debug_integration is None:
        _debug_integration = EnhancedDebugIntegration(project_path, mode)
    return _debug_integration


# Convenience functions
def handle_error(error: Exception, **kwargs) -> Dict[str, Any]:
    """Handle error with full integration"""
    integration = get_debug_integration()
    return integration.handle_error_sync(error, **kwargs)


def debug_query(query: str) -> str:
    """Process natural language debug query"""
    integration = get_debug_integration()
    return integration.natural_language_query(query)


def get_debug_report() -> Dict[str, Any]:
    """Get comprehensive debug report"""
    integration = get_debug_integration()
    return integration.generate_comprehensive_report()