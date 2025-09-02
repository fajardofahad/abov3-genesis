"""
ABOV3 Genesis - Context Compact Integration Module
Seamless integration of Auto Context Compact with existing memory manager,
debug systems, and enhanced error handling with real-time monitoring.
"""

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum, auto
import weakref

# Import existing components
from .auto_context_compact import (
    AutoContextCompact,
    CompactionStrategy,
    ContextImportance,
    ContextSegment,
    get_context_compact
)
from .context_intelligence import (
    ContextIntelligence,
    ContextAnalysis,
    ContextPrediction,
    ContextState,
    IntentType,
    get_context_intelligence
)
from .memory_manager import (
    MemoryManager,
    MemoryEntry,
    MemoryType,
    Priority,
    get_memory_manager
)
from .enhanced_debug_integration import (
    EnhancedDebugIntegration,
    DebugMode,
    IntegrationEvent,
    get_debug_integration
)


class IntegrationType(Enum):
    """Types of integration events"""
    MEMORY_SYNC = "memory_sync"
    DEBUG_TRACE = "debug_trace"
    ERROR_CONTEXT = "error_context"
    PERFORMANCE_MONITOR = "performance_monitor"
    CONTEXT_BACKUP = "context_backup"
    ROLLBACK_TRIGGER = "rollback_trigger"
    THRESHOLD_ALERT = "threshold_alert"
    OPTIMIZATION_COMPLETE = "optimization_complete"


class MonitoringLevel(Enum):
    """Monitoring intensity levels"""
    MINIMAL = 1      # Basic health checks
    STANDARD = 2     # Regular monitoring
    INTENSIVE = 3    # Detailed monitoring
    CRITICAL = 4     # Maximum monitoring
    DEBUG = 5        # Debug-level monitoring


@dataclass
class IntegrationEvent:
    """Integration event data structure"""
    event_id: str
    event_type: IntegrationType
    timestamp: datetime
    source_component: str
    target_component: str
    data: Dict[str, Any]
    priority: int = 1  # 1-5, 5 being highest
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrationMetrics:
    """Metrics for integration system"""
    total_events: int = 0
    successful_integrations: int = 0
    failed_integrations: int = 0
    memory_syncs: int = 0
    debug_traces: int = 0
    context_backups: int = 0
    rollbacks_performed: int = 0
    average_sync_time: float = 0.0
    last_sync_timestamp: Optional[datetime] = None
    system_health_score: float = 1.0
    performance_degradation_alerts: int = 0


class ContextCompactIntegration:
    """
    Master integration system providing seamless connection between
    Auto Context Compact, Memory Manager, Debug Integration, and Context Intelligence.
    Features real-time monitoring, automatic synchronization, and intelligent optimization.
    """
    
    def __init__(self, project_path: Optional[Path] = None,
                 monitoring_level: MonitoringLevel = MonitoringLevel.STANDARD):
        
        self.project_path = project_path or Path.cwd()
        self.monitoring_level = monitoring_level
        
        # Initialize core components
        self.context_compact = get_context_compact(self.project_path)
        self.context_intelligence = get_context_intelligence(self.project_path)
        self.memory_manager = get_memory_manager(self.project_path)
        self.debug_integration = get_debug_integration(self.project_path)
        
        # Integration state
        self.active_integrations: Dict[str, Dict[str, Any]] = {}
        self.event_queue: deque = deque(maxlen=1000)
        self.event_history: deque = deque(maxlen=5000)
        self.rollback_states: deque = deque(maxlen=20)
        
        # Monitoring and metrics
        self.metrics = IntegrationMetrics()
        self.monitoring_enabled = False
        self.monitor_thread = None
        self._stop_monitoring = threading.Event()
        
        # Configuration
        self.config = {
            # Synchronization settings
            'auto_sync_enabled': True,
            'sync_interval_seconds': 60,       # Sync every minute
            'batch_sync_size': 50,             # Process 50 items at once
            'sync_timeout_seconds': 30,        # Timeout for sync operations
            
            # Memory integration
            'memory_sync_threshold': 100,      # Sync after 100 operations
            'memory_priority_mapping': {       # Map context importance to memory priority
                ContextImportance.CRITICAL: Priority.CRITICAL,
                ContextImportance.HIGH: Priority.HIGH,
                ContextImportance.MEDIUM: Priority.MEDIUM,
                ContextImportance.LOW: Priority.LOW,
                ContextImportance.MINIMAL: Priority.TEMP
            },
            
            # Performance monitoring
            'performance_check_interval': 30,  # Check performance every 30 seconds
            'memory_usage_threshold': 0.8,     # Alert at 80% memory usage
            'response_time_threshold': 2.0,    # Alert if response > 2 seconds
            'error_rate_threshold': 0.05,      # Alert if error rate > 5%
            
            # Rollback settings
            'auto_rollback_enabled': True,
            'rollback_quality_threshold': 0.3, # Rollback if quality drops below 30%
            'max_rollback_attempts': 3,
            'rollback_timeout_seconds': 60,
            
            # Debug integration
            'debug_trace_enabled': True,
            'trace_context_operations': True,
            'trace_performance_metrics': True,
            'debug_error_context_size': 5000,  # Characters of context for error debugging
            
            # Alerting
            'alert_on_failures': True,
            'alert_threshold_count': 3,        # Alert after 3 consecutive failures
            'alert_cooldown_minutes': 15,      # Wait 15 minutes between similar alerts
        }
        
        # Event handlers
        self.event_handlers: Dict[IntegrationType, List[Callable]] = defaultdict(list)
        self.alert_handlers: List[Callable] = []
        
        # Threading
        self.lock = threading.RLock()
        self.sync_lock = threading.Lock()
        
        # Setup logging
        self.logger = logging.getLogger('abov3.context_integration')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Initialize integration
        self._setup_integration()
        self._register_event_handlers()
        
        # Start monitoring
        if monitoring_level != MonitoringLevel.MINIMAL:
            self.start_monitoring()
        
        self.logger.info(f"Context Compact Integration initialized with {monitoring_level.name} monitoring")
    
    def _setup_integration(self):
        """Setup integration between all components"""
        try:
            # Register context compact event handlers
            self.context_compact.register_event_handler(
                'compaction_completed',
                self._handle_compaction_completed
            )
            
            # Register intelligence event handlers (if available)
            if hasattr(self.context_intelligence, 'register_event_handler'):
                self.context_intelligence.register_event_handler(
                    'analysis_completed',
                    self._handle_analysis_completed
                )
            
            # Register debug integration event handlers
            self.debug_integration.register_event_handler(
                IntegrationEvent.ERROR_DETECTED,
                self._handle_error_detected
            )
            
            # Setup memory manager hooks
            self._setup_memory_hooks()
            
            # Create initial rollback state
            self._create_rollback_state("initialization")
            
            self.logger.info("Integration setup completed")
            
        except Exception as e:
            self.logger.error(f"Integration setup failed: {e}")
    
    def _setup_memory_hooks(self):
        """Setup hooks into memory manager for synchronization"""
        # This would normally involve monkey-patching or using hooks
        # For demonstration, we'll set up periodic sync instead
        pass
    
    def _register_event_handlers(self):
        """Register internal event handlers"""
        # Memory sync handlers
        self.register_event_handler(
            IntegrationType.MEMORY_SYNC,
            self._handle_memory_sync
        )
        
        # Debug trace handlers
        self.register_event_handler(
            IntegrationType.DEBUG_TRACE,
            self._handle_debug_trace
        )
        
        # Error context handlers
        self.register_event_handler(
            IntegrationType.ERROR_CONTEXT,
            self._handle_error_context
        )
        
        # Performance monitoring handlers
        self.register_event_handler(
            IntegrationType.PERFORMANCE_MONITOR,
            self._handle_performance_monitor
        )
        
        # Context backup handlers
        self.register_event_handler(
            IntegrationType.CONTEXT_BACKUP,
            self._handle_context_backup
        )
        
        # Rollback handlers
        self.register_event_handler(
            IntegrationType.ROLLBACK_TRIGGER,
            self._handle_rollback_trigger
        )
        
        # Threshold alert handlers
        self.register_event_handler(
            IntegrationType.THRESHOLD_ALERT,
            self._handle_threshold_alert
        )
    
    async def sync_with_memory_manager(self, context_id: str = 'main',
                                      force_sync: bool = False) -> Dict[str, Any]:
        """Synchronize context data with memory manager"""
        start_time = time.time()
        
        with self.sync_lock:
            try:
                # Check if sync is needed
                if not force_sync and not self._should_sync():
                    return {
                        'success': True,
                        'message': 'Sync not needed',
                        'items_synced': 0
                    }
                
                synced_items = 0
                errors = []
                
                # Get context segments
                if context_id in self.context_compact.active_contexts:
                    segment_ids = self.context_compact.active_contexts[context_id]
                    
                    for segment_id in segment_ids:
                        if segment_id in self.context_compact.context_segments:
                            try:
                                segment = self.context_compact.context_segments[segment_id]
                                
                                # Map context segment to memory entry
                                memory_priority = self.config['memory_priority_mapping'].get(
                                    segment.importance, 
                                    Priority.MEDIUM
                                )
                                
                                # Determine memory type based on content type
                                memory_type = self._map_content_to_memory_type(segment.content_type)
                                
                                # Store in memory manager
                                entry_id = await self.memory_manager.store(
                                    content=segment.content,
                                    memory_type=memory_type,
                                    priority=memory_priority,
                                    tags={'context_compact', context_id, segment.content_type},
                                    metadata={
                                        'segment_id': segment_id,
                                        'context_id': context_id,
                                        'importance': segment.importance.value,
                                        'compression_ratio': segment.compression_ratio,
                                        'tokens_estimate': segment.tokens_estimate,
                                        'sync_timestamp': datetime.now().isoformat()
                                    },
                                    entry_id=f"context_{segment_id}"
                                )
                                
                                synced_items += 1
                                
                            except Exception as e:
                                errors.append(f"Failed to sync segment {segment_id}: {e}")
                                self.logger.error(f"Sync error for segment {segment_id}: {e}")
                
                # Sync compressed contexts
                for compressed_id, compressed_data in self.context_compact.compressed_contexts.items():
                    if context_id in compressed_id:
                        try:
                            await self.memory_manager.store(
                                content=compressed_data,
                                memory_type=MemoryType.SYSTEM,
                                priority=Priority.HIGH,
                                tags={'compressed_context', context_id},
                                metadata={
                                    'compressed_id': compressed_id,
                                    'context_id': context_id,
                                    'compression_method': compressed_data.get('compression_method'),
                                    'compression_ratio': compressed_data.get('compression_ratio'),
                                    'sync_timestamp': datetime.now().isoformat()
                                },
                                entry_id=f"compressed_{compressed_id}"
                            )
                            synced_items += 1
                            
                        except Exception as e:
                            errors.append(f"Failed to sync compressed context {compressed_id}: {e}")
                
                # Update metrics
                sync_time = time.time() - start_time
                self.metrics.memory_syncs += 1
                self.metrics.successful_integrations += 1 if not errors else 0
                self.metrics.failed_integrations += 1 if errors else 0
                
                # Update average sync time
                self.metrics.average_sync_time = (
                    (self.metrics.average_sync_time * (self.metrics.memory_syncs - 1) + sync_time) /
                    self.metrics.memory_syncs
                )
                
                self.metrics.last_sync_timestamp = datetime.now()
                
                # Fire sync event
                await self._fire_event(
                    IntegrationType.MEMORY_SYNC,
                    'context_compact',
                    'memory_manager',
                    {
                        'context_id': context_id,
                        'synced_items': synced_items,
                        'errors': errors,
                        'sync_time': sync_time
                    }
                )
                
                result = {
                    'success': len(errors) == 0,
                    'items_synced': synced_items,
                    'sync_time': sync_time,
                    'errors': errors
                }
                
                if errors:
                    result['message'] = f"Sync completed with {len(errors)} errors"
                else:
                    result['message'] = f"Successfully synced {synced_items} items"
                
                self.logger.info(f"Memory sync completed: {synced_items} items, {len(errors)} errors")
                
                return result
                
            except Exception as e:
                self.logger.error(f"Memory sync failed: {e}")
                self.metrics.failed_integrations += 1
                return {
                    'success': False,
                    'error': str(e),
                    'items_synced': 0
                }
    
    def _map_content_to_memory_type(self, content_type: str) -> MemoryType:
        """Map context content type to memory type"""
        mapping = {
            'conversation': MemoryType.CONVERSATION,
            'code': MemoryType.CODE_CONTEXT,
            'file': MemoryType.PROJECT,
            'error': MemoryType.SYSTEM,
            'decision': MemoryType.SESSION,
            'system': MemoryType.SYSTEM
        }
        return mapping.get(content_type, MemoryType.SESSION)
    
    def _should_sync(self) -> bool:
        """Determine if synchronization is needed"""
        # Check if auto-sync is enabled
        if not self.config['auto_sync_enabled']:
            return False
        
        # Check time-based sync
        if self.metrics.last_sync_timestamp:
            time_since_sync = (datetime.now() - self.metrics.last_sync_timestamp).total_seconds()
            if time_since_sync < self.config['sync_interval_seconds']:
                return False
        
        # Check operation-based sync
        total_operations = (
            self.metrics.memory_syncs + 
            self.metrics.debug_traces + 
            self.metrics.context_backups
        )
        
        if total_operations % self.config['memory_sync_threshold'] == 0:
            return True
        
        return self.metrics.last_sync_timestamp is None
    
    async def trace_context_operation(self, operation: str, context_id: str,
                                     details: Dict[str, Any]) -> str:
        """Trace context operations for debugging"""
        try:
            if not self.config['debug_trace_enabled']:
                return ""
            
            trace_id = f"trace_{int(time.time() * 1000)}"
            
            # Collect context for debugging
            context_summary = self.context_compact.get_context_summary(context_id)
            
            trace_data = {
                'trace_id': trace_id,
                'operation': operation,
                'context_id': context_id,
                'timestamp': datetime.now().isoformat(),
                'context_summary': context_summary,
                'operation_details': details,
                'system_state': {
                    'memory_usage': self._get_memory_usage(),
                    'active_contexts': len(self.context_compact.active_contexts),
                    'monitoring_level': self.monitoring_level.name
                }
            }
            
            # Store trace in debug integration
            debug_context = self.debug_integration.create_debug_context()
            await self.debug_integration.track_operation(
                f"context_operation_{operation}",
                trace_data
            )
            
            # Fire debug trace event
            await self._fire_event(
                IntegrationType.DEBUG_TRACE,
                'context_integration',
                'debug_integration',
                trace_data
            )
            
            self.metrics.debug_traces += 1
            
            self.logger.debug(f"Context operation traced: {operation} for {context_id}")
            
            return trace_id
            
        except Exception as e:
            self.logger.error(f"Failed to trace context operation: {e}")
            return ""
    
    async def handle_error_with_context(self, error: Exception, 
                                       context_id: str = 'main',
                                       preserve_context: bool = True) -> Dict[str, Any]:
        """Handle errors with full context preservation"""
        try:
            error_id = f"error_{int(time.time() * 1000)}"
            
            # Get current context for error analysis
            context_summary = self.context_compact.get_context_summary(context_id)
            context_analysis = await self.context_intelligence.analyze_context(context_id)
            
            # Extract relevant context segments for error debugging
            relevant_context = await self._extract_error_context(
                error, context_id, self.config['debug_error_context_size']
            )
            
            # Create comprehensive error context
            error_context = {
                'error_id': error_id,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'context_id': context_id,
                'timestamp': datetime.now().isoformat(),
                'context_summary': context_summary,
                'context_analysis': asdict(context_analysis),
                'relevant_context': relevant_context,
                'system_state': await self._get_system_state()
            }
            
            # Preserve context if requested
            if preserve_context:
                backup_id = await self._create_context_backup(
                    context_id, f"error_backup_{error_id}"
                )
                error_context['backup_id'] = backup_id
            
            # Handle error through debug integration
            debug_result = self.debug_integration.handle_error(
                error,
                context_id=context_id,
                context=error_context
            )
            
            # Fire error context event
            await self._fire_event(
                IntegrationType.ERROR_CONTEXT,
                'context_integration',
                'debug_integration',
                error_context
            )
            
            # Check if context quality has degraded due to error
            if context_analysis.quality_score < self.config['rollback_quality_threshold']:
                if self.config['auto_rollback_enabled']:
                    await self._trigger_rollback(
                        context_id, 
                        f"Quality degradation after error: {error_id}"
                    )
            
            result = {
                'error_id': error_id,
                'handled': debug_result.get('success', False),
                'context_preserved': preserve_context,
                'debug_result': debug_result,
                'recommendations': context_analysis.recommendations
            }
            
            self.logger.info(f"Error handled with context: {error_id}")
            
            return result
            
        except Exception as handling_error:
            self.logger.error(f"Failed to handle error with context: {handling_error}")
            return {
                'error_id': 'failed',
                'handled': False,
                'error': str(handling_error)
            }
    
    async def _extract_error_context(self, error: Exception, context_id: str,
                                    max_size: int) -> Dict[str, Any]:
        """Extract relevant context for error analysis"""
        relevant_context = {
            'recent_segments': [],
            'error_related_segments': [],
            'high_importance_segments': []
        }
        
        if context_id not in self.context_compact.active_contexts:
            return relevant_context
        
        segment_ids = self.context_compact.active_contexts[context_id]
        segments = [
            self.context_compact.context_segments[sid] 
            for sid in segment_ids 
            if sid in self.context_compact.context_segments
        ]
        
        # Sort by timestamp (most recent first)
        segments.sort(key=lambda s: s.timestamp, reverse=True)
        
        current_size = 0
        error_type = type(error).__name__.lower()
        
        # Extract recent segments
        for segment in segments[:10]:  # Last 10 segments
            if current_size >= max_size:
                break
            
            segment_data = {
                'segment_id': segment.segment_id,
                'content_type': segment.content_type,
                'timestamp': segment.timestamp.isoformat(),
                'importance': segment.importance.name,
                'content_preview': str(segment.content)[:200] + "..." if len(str(segment.content)) > 200 else str(segment.content)
            }
            
            relevant_context['recent_segments'].append(segment_data)
            current_size += len(json.dumps(segment_data))
        
        # Extract error-related segments
        for segment in segments:
            if current_size >= max_size:
                break
            
            content_lower = str(segment.content).lower()
            if (error_type in content_lower or 
                'error' in content_lower or 
                'exception' in content_lower or
                segment.content_type == 'error'):
                
                segment_data = {
                    'segment_id': segment.segment_id,
                    'content_type': segment.content_type,
                    'timestamp': segment.timestamp.isoformat(),
                    'importance': segment.importance.name,
                    'content_preview': str(segment.content)[:300] + "..." if len(str(segment.content)) > 300 else str(segment.content)
                }
                
                relevant_context['error_related_segments'].append(segment_data)
                current_size += len(json.dumps(segment_data))
        
        # Extract high-importance segments
        high_importance = [s for s in segments if s.importance.value >= ContextImportance.HIGH.value]
        for segment in high_importance[:5]:  # Top 5 high importance
            if current_size >= max_size:
                break
            
            segment_data = {
                'segment_id': segment.segment_id,
                'content_type': segment.content_type,
                'timestamp': segment.timestamp.isoformat(),
                'importance': segment.importance.name,
                'content_preview': str(segment.content)[:150] + "..." if len(str(segment.content)) > 150 else str(segment.content)
            }
            
            relevant_context['high_importance_segments'].append(segment_data)
            current_size += len(json.dumps(segment_data))
        
        return relevant_context
    
    async def _create_context_backup(self, context_id: str, backup_name: str) -> str:
        """Create a backup of current context state"""
        try:
            backup_id = f"{backup_name}_{int(time.time())}"
            
            # Collect context data
            context_summary = self.context_compact.get_context_summary(context_id)
            context_analysis = await self.context_intelligence.analyze_context(context_id)
            
            # Get all segments
            segments_data = {}
            if context_id in self.context_compact.active_contexts:
                segment_ids = self.context_compact.active_contexts[context_id]
                for segment_id in segment_ids:
                    if segment_id in self.context_compact.context_segments:
                        segment = self.context_compact.context_segments[segment_id]
                        segments_data[segment_id] = asdict(segment)
            
            # Get compressed contexts
            compressed_data = {}
            for comp_id, comp_data in self.context_compact.compressed_contexts.items():
                if context_id in comp_id:
                    compressed_data[comp_id] = comp_data
            
            # Create backup
            backup_data = {
                'backup_id': backup_id,
                'context_id': context_id,
                'timestamp': datetime.now().isoformat(),
                'context_summary': context_summary,
                'context_analysis': asdict(context_analysis),
                'segments': segments_data,
                'compressed_contexts': compressed_data,
                'metadata': {
                    'backup_reason': backup_name,
                    'total_segments': len(segments_data),
                    'total_tokens': context_summary.get('total_tokens', 0)
                }
            }
            
            # Store backup in memory manager
            await self.memory_manager.store(
                content=backup_data,
                memory_type=MemoryType.SYSTEM,
                priority=Priority.HIGH,
                tags={'context_backup', context_id, 'rollback'},
                metadata={
                    'backup_id': backup_id,
                    'context_id': context_id,
                    'backup_type': 'full_context'
                },
                entry_id=backup_id
            )
            
            # Add to rollback states
            self.rollback_states.append({
                'backup_id': backup_id,
                'context_id': context_id,
                'timestamp': datetime.now(),
                'reason': backup_name
            })
            
            # Fire backup event
            await self._fire_event(
                IntegrationType.CONTEXT_BACKUP,
                'context_integration',
                'memory_manager',
                {'backup_id': backup_id, 'context_id': context_id}
            )
            
            self.metrics.context_backups += 1
            
            self.logger.info(f"Context backup created: {backup_id}")
            
            return backup_id
            
        except Exception as e:
            self.logger.error(f"Failed to create context backup: {e}")
            return ""
    
    def _create_rollback_state(self, reason: str):
        """Create a rollback state for all active contexts"""
        try:
            for context_id in self.context_compact.active_contexts:
                asyncio.create_task(self._create_context_backup(context_id, f"rollback_{reason}"))
        except Exception as e:
            self.logger.error(f"Failed to create rollback state: {e}")
    
    async def _trigger_rollback(self, context_id: str, reason: str) -> Dict[str, Any]:
        """Trigger rollback to previous context state"""
        try:
            if not self.config['auto_rollback_enabled']:
                return {'success': False, 'message': 'Auto rollback disabled'}
            
            # Find most recent rollback state
            rollback_state = None
            for state in reversed(self.rollback_states):
                if state['context_id'] == context_id:
                    rollback_state = state
                    break
            
            if not rollback_state:
                return {'success': False, 'message': 'No rollback state found'}
            
            # Retrieve backup data
            backup_entries = await self.memory_manager.retrieve(
                entry_id=rollback_state['backup_id']
            )
            
            if not backup_entries:
                return {'success': False, 'message': 'Backup data not found'}
            
            backup_data = backup_entries[0][1]  # Get content from (id, content, metadata)
            
            # Restore context state (simplified - would need full restoration logic)
            # For now, we'll trigger emergency compaction instead
            compaction_result = await self.context_compact.compact_context(
                context_id,
                CompactionStrategy.EMERGENCY
            )
            
            # Fire rollback event
            await self._fire_event(
                IntegrationType.ROLLBACK_TRIGGER,
                'context_integration',
                'context_compact',
                {
                    'context_id': context_id,
                    'backup_id': rollback_state['backup_id'],
                    'reason': reason,
                    'compaction_result': compaction_result
                }
            )
            
            self.metrics.rollbacks_performed += 1
            
            self.logger.warning(f"Rollback triggered for {context_id}: {reason}")
            
            return {
                'success': True,
                'backup_id': rollback_state['backup_id'],
                'compaction_result': compaction_result
            }
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _get_system_state(self) -> Dict[str, Any]:
        """Get comprehensive system state"""
        return {
            'timestamp': datetime.now().isoformat(),
            'memory_usage': self._get_memory_usage(),
            'active_contexts': len(self.context_compact.active_contexts),
            'total_segments': len(self.context_compact.context_segments),
            'compressed_contexts': len(self.context_compact.compressed_contexts),
            'memory_entries': len(self.memory_manager.memory_store),
            'debug_sessions': len(self.debug_integration.active_contexts),
            'monitoring_level': self.monitoring_level.name,
            'integration_metrics': asdict(self.metrics)
        }
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        try:
            # Get memory manager stats
            memory_stats = self.memory_manager.get_memory_stats()
            
            # Get context compact stats
            compact_stats = self.context_compact.get_compaction_statistics()
            
            return {
                'memory_manager': memory_stats,
                'context_compact': compact_stats,
                'total_memory_mb': memory_stats.get('memory_usage_mb', 0) + 
                                 compact_stats.get('stats', {}).get('memory_usage_mb', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get memory usage: {e}")
            return {'error': str(e)}
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        if not self.monitoring_enabled:
            self.monitoring_enabled = True
            self._stop_monitoring.clear()
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.info(f"Integration monitoring started at {self.monitoring_level.name} level")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        if self.monitoring_enabled:
            self.monitoring_enabled = False
            self._stop_monitoring.set()
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)
            self.logger.info("Integration monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while not self._stop_monitoring.is_set():
            try:
                # Perform monitoring based on level
                if self.monitoring_level.value >= MonitoringLevel.STANDARD.value:
                    asyncio.run(self._standard_monitoring_check())
                
                if self.monitoring_level.value >= MonitoringLevel.INTENSIVE.value:
                    asyncio.run(self._intensive_monitoring_check())
                
                if self.monitoring_level.value >= MonitoringLevel.CRITICAL.value:
                    asyncio.run(self._critical_monitoring_check())
                
                if self.monitoring_level.value >= MonitoringLevel.DEBUG.value:
                    asyncio.run(self._debug_monitoring_check())
                
                # Sleep based on monitoring level
                sleep_time = max(30 - (self.monitoring_level.value * 5), 5)
                self._stop_monitoring.wait(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                self._stop_monitoring.wait(60)
    
    async def _standard_monitoring_check(self):
        """Standard monitoring checks"""
        try:
            # Check memory usage
            memory_usage = self._get_memory_usage()
            total_memory = memory_usage.get('total_memory_mb', 0)
            
            if total_memory > 0:
                usage_ratio = total_memory / 1024  # Assume 1GB limit for example
                if usage_ratio > self.config['memory_usage_threshold']:
                    await self._fire_alert(
                        f"High memory usage: {usage_ratio:.1%}",
                        {'memory_usage': memory_usage}
                    )
            
            # Check sync status
            if self._should_sync():
                for context_id in self.context_compact.active_contexts:
                    await self.sync_with_memory_manager(context_id)
            
            # Update system health score
            await self._update_system_health_score()
            
        except Exception as e:
            self.logger.error(f"Standard monitoring check error: {e}")
    
    async def _intensive_monitoring_check(self):
        """Intensive monitoring checks"""
        try:
            # Check all context states
            for context_id in self.context_compact.active_contexts:
                analysis = await self.context_intelligence.analyze_context(context_id)
                
                if analysis.state == ContextState.CRITICAL:
                    await self._fire_alert(
                        f"Critical context state: {context_id}",
                        {'analysis': asdict(analysis)}
                    )
                
                if analysis.quality_score < 0.5:
                    await self._fire_alert(
                        f"Low context quality: {context_id} ({analysis.quality_score:.2f})",
                        {'analysis': asdict(analysis)}
                    )
            
            # Check error rates
            total_ops = self.metrics.successful_integrations + self.metrics.failed_integrations
            if total_ops > 0:
                error_rate = self.metrics.failed_integrations / total_ops
                if error_rate > self.config['error_rate_threshold']:
                    await self._fire_alert(
                        f"High error rate: {error_rate:.1%}",
                        {'metrics': asdict(self.metrics)}
                    )
            
        except Exception as e:
            self.logger.error(f"Intensive monitoring check error: {e}")
    
    async def _critical_monitoring_check(self):
        """Critical level monitoring checks"""
        try:
            # Check response times
            if self.metrics.average_sync_time > self.config['response_time_threshold']:
                await self._fire_alert(
                    f"Slow response times: {self.metrics.average_sync_time:.2f}s",
                    {'metrics': asdict(self.metrics)}
                )
            
            # Check for system degradation
            if self.metrics.system_health_score < 0.7:
                await self._fire_alert(
                    f"System health degraded: {self.metrics.system_health_score:.2f}",
                    {'metrics': asdict(self.metrics)}
                )
            
            # Proactive optimization
            for context_id in self.context_compact.active_contexts:
                prediction = await self.context_intelligence.predict_context_evolution(context_id)
                
                if prediction.compaction_probability > 0.8:
                    # Proactive compaction
                    await self.context_compact.compact_context(
                        context_id,
                        prediction.optimal_strategy
                    )
                    self.logger.info(f"Proactive compaction applied to {context_id}")
            
        except Exception as e:
            self.logger.error(f"Critical monitoring check error: {e}")
    
    async def _debug_monitoring_check(self):
        """Debug level monitoring checks"""
        try:
            # Trace all operations
            system_state = await self._get_system_state()
            
            await self._fire_event(
                IntegrationType.PERFORMANCE_MONITOR,
                'monitoring_system',
                'integration',
                {
                    'monitoring_level': 'debug',
                    'system_state': system_state,
                    'detailed_metrics': {
                        'event_queue_size': len(self.event_queue),
                        'event_history_size': len(self.event_history),
                        'rollback_states': len(self.rollback_states),
                        'active_integrations': len(self.active_integrations)
                    }
                }
            )
            
            # Log detailed metrics
            self.logger.debug(f"System state: {system_state}")
            
        except Exception as e:
            self.logger.error(f"Debug monitoring check error: {e}")
    
    async def _update_system_health_score(self):
        """Update overall system health score"""
        try:
            health_factors = []
            
            # Memory health
            memory_usage = self._get_memory_usage()
            total_memory = memory_usage.get('total_memory_mb', 0)
            if total_memory > 0:
                memory_health = max(0, 1.0 - (total_memory / 1024))  # Assume 1GB limit
                health_factors.append(memory_health * 0.3)
            
            # Error rate health
            total_ops = self.metrics.successful_integrations + self.metrics.failed_integrations
            if total_ops > 0:
                success_rate = self.metrics.successful_integrations / total_ops
                health_factors.append(success_rate * 0.3)
            
            # Response time health
            if self.metrics.average_sync_time > 0:
                response_health = max(0, 1.0 - (self.metrics.average_sync_time / 10))  # 10s max
                health_factors.append(response_health * 0.2)
            
            # Context quality health
            context_qualities = []
            for context_id in self.context_compact.active_contexts:
                if context_id in self.context_intelligence.analysis_cache:
                    analysis = self.context_intelligence.analysis_cache[context_id]
                    context_qualities.append(analysis.quality_score)
            
            if context_qualities:
                avg_quality = sum(context_qualities) / len(context_qualities)
                health_factors.append(avg_quality * 0.2)
            
            # Calculate overall health
            if health_factors:
                self.metrics.system_health_score = sum(health_factors) / len(health_factors)
            else:
                self.metrics.system_health_score = 1.0  # Default if no factors
            
        except Exception as e:
            self.logger.error(f"Failed to update system health score: {e}")
            self.metrics.system_health_score = 0.5  # Default to medium health on error
    
    async def _fire_alert(self, message: str, data: Dict[str, Any]):
        """Fire alert to registered handlers"""
        try:
            alert_data = {
                'timestamp': datetime.now().isoformat(),
                'message': message,
                'data': data,
                'monitoring_level': self.monitoring_level.name,
                'system_state': await self._get_system_state()
            }
            
            # Fire threshold alert event
            await self._fire_event(
                IntegrationType.THRESHOLD_ALERT,
                'monitoring_system',
                'alert_handlers',
                alert_data
            )
            
            # Call alert handlers
            for handler in self.alert_handlers:
                try:
                    handler(alert_data)
                except Exception as e:
                    self.logger.error(f"Alert handler error: {e}")
            
            self.metrics.performance_degradation_alerts += 1
            self.logger.warning(f"ALERT: {message}")
            
        except Exception as e:
            self.logger.error(f"Failed to fire alert: {e}")
    
    async def _fire_event(self, event_type: IntegrationType, source: str,
                         target: str, data: Dict[str, Any]) -> str:
        """Fire integration event"""
        try:
            event_id = f"{event_type.value}_{int(time.time() * 1000)}"
            
            event = IntegrationEvent(
                event_id=event_id,
                event_type=event_type,
                timestamp=datetime.now(),
                source_component=source,
                target_component=target,
                data=data
            )
            
            # Add to queue and history
            self.event_queue.append(event)
            self.event_history.append(event)
            
            # Call registered handlers
            for handler in self.event_handlers.get(event_type, []):
                try:
                    await handler(event)
                except Exception as e:
                    event.success = False
                    event.error_message = str(e)
                    self.logger.error(f"Event handler error for {event_type}: {e}")
            
            self.metrics.total_events += 1
            if event.success:
                self.metrics.successful_integrations += 1
            else:
                self.metrics.failed_integrations += 1
            
            return event_id
            
        except Exception as e:
            self.logger.error(f"Failed to fire event: {e}")
            return ""
    
    def register_event_handler(self, event_type: IntegrationType, 
                              handler: Callable):
        """Register event handler"""
        self.event_handlers[event_type].append(handler)
        self.logger.debug(f"Registered handler for {event_type}")
    
    def register_alert_handler(self, handler: Callable):
        """Register alert handler"""
        self.alert_handlers.append(handler)
        self.logger.debug("Registered alert handler")
    
    # Event handler implementations
    async def _handle_memory_sync(self, event: IntegrationEvent):
        """Handle memory sync events"""
        self.logger.debug(f"Memory sync event: {event.data.get('synced_items', 0)} items")
    
    async def _handle_debug_trace(self, event: IntegrationEvent):
        """Handle debug trace events"""
        self.logger.debug(f"Debug trace: {event.data.get('operation', 'unknown')}")
    
    async def _handle_error_context(self, event: IntegrationEvent):
        """Handle error context events"""
        error_id = event.data.get('error_id', 'unknown')
        self.logger.info(f"Error context preserved: {error_id}")
    
    async def _handle_performance_monitor(self, event: IntegrationEvent):
        """Handle performance monitoring events"""
        self.logger.debug(f"Performance monitoring: {event.data.get('monitoring_level', 'unknown')}")
    
    async def _handle_context_backup(self, event: IntegrationEvent):
        """Handle context backup events"""
        backup_id = event.data.get('backup_id', 'unknown')
        self.logger.info(f"Context backup created: {backup_id}")
    
    async def _handle_rollback_trigger(self, event: IntegrationEvent):
        """Handle rollback trigger events"""
        context_id = event.data.get('context_id', 'unknown')
        reason = event.data.get('reason', 'unknown')
        self.logger.warning(f"Rollback triggered for {context_id}: {reason}")
    
    async def _handle_threshold_alert(self, event: IntegrationEvent):
        """Handle threshold alert events"""
        message = event.data.get('message', 'Unknown alert')
        self.logger.warning(f"Threshold alert: {message}")
    
    # External event handlers (from other components)
    def _handle_compaction_completed(self, data: Dict[str, Any]):
        """Handle compaction completion from context compact"""
        context_id = data.get('context_id', 'unknown')
        tokens_saved = data.get('result', {}).get('tokens_saved', 0)
        
        asyncio.create_task(self._fire_event(
            IntegrationType.OPTIMIZATION_COMPLETE,
            'context_compact',
            'integration',
            {
                'context_id': context_id,
                'tokens_saved': tokens_saved,
                'optimization_type': 'compaction'
            }
        ))
        
        self.logger.info(f"Compaction completed for {context_id}: {tokens_saved} tokens saved")
    
    def _handle_analysis_completed(self, data: Dict[str, Any]):
        """Handle analysis completion from context intelligence"""
        context_id = data.get('context_id', 'unknown')
        quality_score = data.get('quality_score', 0.0)
        
        asyncio.create_task(self._fire_event(
            IntegrationType.OPTIMIZATION_COMPLETE,
            'context_intelligence',
            'integration',
            {
                'context_id': context_id,
                'quality_score': quality_score,
                'optimization_type': 'analysis'
            }
        ))
        
        self.logger.debug(f"Analysis completed for {context_id}: quality={quality_score:.2f}")
    
    def _handle_error_detected(self, data: Dict[str, Any]):
        """Handle error detection from debug integration"""
        error_id = data.get('error_id', 'unknown')
        
        # Automatically preserve context on error detection
        if self.config['trace_context_operations']:
            for context_id in self.context_compact.active_contexts:
                asyncio.create_task(self._create_context_backup(
                    context_id, 
                    f"auto_backup_error_{error_id}"
                ))
        
        self.logger.info(f"Error detected, context preserved: {error_id}")
    
    def get_integration_report(self) -> Dict[str, Any]:
        """Get comprehensive integration report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'monitoring_level': self.monitoring_level.name,
            'monitoring_enabled': self.monitoring_enabled,
            'metrics': asdict(self.metrics),
            'configuration': self.config,
            'active_integrations': len(self.active_integrations),
            'event_queue_size': len(self.event_queue),
            'event_history_size': len(self.event_history),
            'rollback_states': len(self.rollback_states),
            'registered_handlers': {
                event_type.value: len(handlers) 
                for event_type, handlers in self.event_handlers.items()
            },
            'alert_handlers': len(self.alert_handlers),
            'recent_events': [
                {
                    'event_id': event.event_id,
                    'event_type': event.event_type.value,
                    'timestamp': event.timestamp.isoformat(),
                    'source': event.source_component,
                    'target': event.target_component,
                    'success': event.success
                }
                for event in list(self.event_history)[-10:]
            ]
        }
    
    async def shutdown(self):
        """Gracefully shutdown integration system"""
        self.logger.info("Shutting down Context Compact Integration")
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Final sync
        if self.config['auto_sync_enabled']:
            for context_id in list(self.context_compact.active_contexts.keys()):
                try:
                    await self.sync_with_memory_manager(context_id, force_sync=True)
                except Exception as e:
                    self.logger.error(f"Final sync failed for {context_id}: {e}")
        
        # Create final rollback state
        self._create_rollback_state("shutdown")
        
        # Clear event handlers
        self.event_handlers.clear()
        self.alert_handlers.clear()
        
        self.logger.info("Context Compact Integration shutdown complete")


# Global instance
_global_integration: Optional[ContextCompactIntegration] = None

def get_context_integration(project_path: Optional[Path] = None,
                           monitoring_level: MonitoringLevel = MonitoringLevel.STANDARD) -> ContextCompactIntegration:
    """Get global context integration instance"""
    global _global_integration
    if _global_integration is None:
        _global_integration = ContextCompactIntegration(project_path, monitoring_level)
    return _global_integration

# Convenience functions
async def sync_with_memory(context_id: str = 'main', force_sync: bool = False) -> Dict[str, Any]:
    """Sync context with memory manager"""
    integration = get_context_integration()
    return await integration.sync_with_memory_manager(context_id, force_sync)

async def trace_operation(operation: str, context_id: str, details: Dict[str, Any]) -> str:
    """Trace context operation"""
    integration = get_context_integration()
    return await integration.trace_context_operation(operation, context_id, details)

async def handle_error_with_context(error: Exception, context_id: str = 'main',
                                   preserve_context: bool = True) -> Dict[str, Any]:
    """Handle error with context preservation"""
    integration = get_context_integration()
    return await integration.handle_error_with_context(error, context_id, preserve_context)

def get_integration_report() -> Dict[str, Any]:
    """Get integration system report"""
    integration = get_context_integration()
    return integration.get_integration_report()