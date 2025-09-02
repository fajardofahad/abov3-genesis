"""
ABOV3 Genesis - Claude Coder Integration Module
Seamlessly integrates keyboard controls, memory management, and feedback loops
"""

import asyncio
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from datetime import datetime
from contextlib import asynccontextmanager

# Import our three core modules
from .memory_manager import MemoryManager, MemoryType, Priority, get_memory_manager
from .feedback_loop import FeedbackLoop, ExecutionResult, get_feedback_loop
from ..ui.keyboard_handler import KeyboardHandler, KeyEvent, get_keyboard_handler

@dataclass
class ClaudeIntegrationConfig:
    """Configuration for Claude-level integration"""
    # Keyboard settings
    enable_keyboard_controls: bool = True
    interrupt_timeout: float = 0.5
    
    # Memory settings
    max_memory_mb: int = 500
    context_window_size: int = 128000
    auto_compress_threshold: float = 0.8
    
    # Feedback settings
    max_execution_time: float = 30.0
    max_iterations: int = 10
    success_threshold: float = 0.8
    
    # Integration settings
    auto_save_interval: float = 30.0
    background_processing: bool = True
    debug_mode: bool = False

class ClaudeIntegration:
    """
    Master integration class that provides Claude Coder-level functionality
    by seamlessly coordinating keyboard controls, memory management, and feedback loops.
    """
    
    def __init__(self, project_path: Path, config: ClaudeIntegrationConfig = None):
        self.project_path = project_path
        self.config = config or ClaudeIntegrationConfig()
        
        # Core modules
        self.keyboard_handler: Optional[KeyboardHandler] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.feedback_loop: Optional[FeedbackLoop] = None
        
        # Integration state
        self.initialized = False
        self.active_operation: Optional[str] = None
        self.operation_start_time: float = 0.0
        self.todo_visible: bool = True
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.running = False
        
        # Callbacks and hooks
        self.on_interrupt: Optional[Callable] = None
        self.on_todo_toggle: Optional[Callable] = None
        self.on_feedback_received: Optional[Callable] = None
        self.on_memory_event: Optional[Callable] = None
        
        # Performance metrics
        self.metrics = {
            'operations_completed': 0,
            'interrupts_handled': 0,
            'memory_operations': 0,
            'feedback_cycles': 0,
            'session_start': datetime.now(),
            'response_times': []
        }
        
        # Integration lock for thread safety
        self.lock = asyncio.Lock()
    
    async def initialize(self, loop: asyncio.AbstractEventLoop = None) -> bool:
        """Initialize all systems with full integration"""
        if self.initialized:
            return True
        
        try:
            print("🎯 Initializing Claude-level integration...")
            
            # Initialize memory manager first
            print("🧠 Starting advanced memory system...")
            self.memory_manager = get_memory_manager(
                self.project_path, 
                self.config.max_memory_mb
            )
            
            # Store initialization in memory
            await self.memory_manager.store(
                {
                    'event': 'system_initialization',
                    'timestamp': datetime.now().isoformat(),
                    'config': {
                        'max_memory_mb': self.config.max_memory_mb,
                        'context_window_size': self.config.context_window_size,
                        'keyboard_enabled': self.config.enable_keyboard_controls,
                        'feedback_enabled': True
                    }
                },
                MemoryType.SYSTEM,
                Priority.HIGH,
                tags={'initialization', 'system'}
            )
            
            # Initialize feedback loop with memory integration
            print("🔄 Starting write-run-debug feedback system...")
            self.feedback_loop = get_feedback_loop(self.project_path, self.memory_manager)
            
            # Setup feedback callbacks
            self.feedback_loop.on_execution_complete = self._on_execution_complete
            self.feedback_loop.on_feedback_analysis = self._on_feedback_analysis
            
            # Initialize keyboard controls if enabled
            if self.config.enable_keyboard_controls:
                print("⌨️  Activating keyboard controls...")
                self.keyboard_handler = get_keyboard_handler()
                
                # Setup keyboard callbacks
                self.keyboard_handler.set_todo_toggle_callback(self._on_todo_toggle)
                self.keyboard_handler.add_interrupt_callback(self._on_keyboard_interrupt)
                
                # Initialize keyboard system
                await self.keyboard_handler.initialize(loop)
            
            # Start background integration tasks
            if self.config.background_processing:
                await self._start_background_tasks()
            
            self.running = True
            self.initialized = True
            
            print("✅ Claude-level integration active!")
            print("   ESC     → Emergency interrupt")
            print("   Ctrl+T  → Toggle todo list") 
            print("   Memory  → Intelligent context management")
            print("   Feedback→ Automatic write-run-debug cycles")
            
            return True
            
        except Exception as e:
            print(f"❌ Integration initialization failed: {e}")
            await self._cleanup_partial_init()
            return False
    
    async def _start_background_tasks(self):
        """Start background integration tasks"""
        tasks = [
            self._auto_save_task(),
            self._memory_optimization_task(),
            self._performance_monitoring_task(),
            self._integration_health_check()
        ]
        
        for coro in tasks:
            task = asyncio.create_task(coro)
            self.background_tasks.append(task)
    
    async def _auto_save_task(self):
        """Background task for auto-saving memory and session state"""
        while self.running:
            try:
                await asyncio.sleep(self.config.auto_save_interval)
                
                if self.memory_manager:
                    # Store session checkpoint
                    await self.memory_manager.store(
                        {
                            'event': 'auto_save_checkpoint',
                            'timestamp': datetime.now().isoformat(),
                            'active_operation': self.active_operation,
                            'metrics': self.get_performance_metrics()
                        },
                        MemoryType.SESSION,
                        Priority.MEDIUM,
                        tags={'auto_save', 'checkpoint'}
                    )
                
            except Exception as e:
                if self.config.debug_mode:
                    print(f"Auto-save error: {e}")
                await asyncio.sleep(5)  # Retry delay
    
    async def _memory_optimization_task(self):
        """Background task for memory optimization"""
        while self.running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                if self.memory_manager:
                    stats = self.memory_manager.get_memory_stats()
                    
                    # Compress context if approaching limit
                    if stats['memory_usage_percent'] > self.config.auto_compress_threshold * 100:
                        compression_ratio = await self.memory_manager.compress_context_window()
                        
                        if compression_ratio > 0:
                            print(f"🗜️  Memory compressed: {compression_ratio:.1%} reduction")
                    
                    # Clean old entries periodically
                    if stats['total_entries'] > 1000:
                        cleaned = await self.memory_manager.cleanup_old_entries(7)
                        if cleaned > 0:
                            print(f"🧹 Cleaned {cleaned} old memory entries")
                
            except Exception as e:
                if self.config.debug_mode:
                    print(f"Memory optimization error: {e}")
                await asyncio.sleep(30)  # Retry delay
    
    async def _performance_monitoring_task(self):
        """Background task for performance monitoring"""
        while self.running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Update performance metrics
                self.metrics['uptime_seconds'] = (datetime.now() - self.metrics['session_start']).total_seconds()
                
                # Log performance data to memory
                if self.memory_manager:
                    perf_data = self.get_performance_metrics()
                    await self.memory_manager.store(
                        perf_data,
                        MemoryType.SYSTEM,
                        Priority.LOW,
                        tags={'performance', 'monitoring'}
                    )
                
            except Exception:
                await asyncio.sleep(30)
    
    async def _integration_health_check(self):
        """Background task for system health monitoring"""
        while self.running:
            try:
                await asyncio.sleep(120)  # Check every 2 minutes
                
                health_status = {
                    'timestamp': datetime.now().isoformat(),
                    'keyboard_active': self.keyboard_handler is not None and self.keyboard_handler.running,
                    'memory_usage': self.memory_manager.get_memory_stats() if self.memory_manager else {},
                    'feedback_metrics': self.feedback_loop.get_cycle_metrics() if self.feedback_loop else {},
                    'background_tasks_count': len([t for t in self.background_tasks if not t.done()])
                }
                
                # Store health check
                if self.memory_manager:
                    await self.memory_manager.store(
                        health_status,
                        MemoryType.SYSTEM,
                        Priority.MEDIUM,
                        tags={'health_check', 'system_status'}
                    )
                
            except Exception:
                await asyncio.sleep(60)
    
    @asynccontextmanager
    async def operation_context(self, operation_name: str, auto_feedback: bool = True):
        """Context manager for Claude-level operation tracking"""
        async with self.lock:
            # Start operation
            self.active_operation = operation_name
            self.operation_start_time = time.time()
            
            if self.keyboard_handler:
                self.keyboard_handler.set_current_operation(operation_name)
            
            # Store operation start in memory
            if self.memory_manager:
                await self.memory_manager.store(
                    {
                        'event': 'operation_start',
                        'operation': operation_name,
                        'timestamp': datetime.now().isoformat()
                    },
                    MemoryType.SESSION,
                    Priority.MEDIUM,
                    tags={'operation', 'start'}
                )
        
        try:
            yield self
        finally:
            async with self.lock:
                # End operation
                duration = time.time() - self.operation_start_time
                self.metrics['operations_completed'] += 1
                self.metrics['response_times'].append(duration)
                
                # Keep only recent response times
                if len(self.metrics['response_times']) > 100:
                    self.metrics['response_times'] = self.metrics['response_times'][-50:]
                
                # Store operation completion
                if self.memory_manager:
                    await self.memory_manager.store(
                        {
                            'event': 'operation_complete',
                            'operation': operation_name,
                            'duration': duration,
                            'timestamp': datetime.now().isoformat()
                        },
                        MemoryType.SESSION,
                        Priority.MEDIUM,
                        tags={'operation', 'complete'}
                    )
                
                # Clear operation state
                self.active_operation = None
                self.operation_start_time = 0.0
                
                if self.keyboard_handler:
                    self.keyboard_handler.clear_current_operation()
    
    async def execute_with_feedback(
        self, 
        file_path: Path, 
        auto_iterate: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Execute code with full feedback loop integration"""
        if not self.feedback_loop:
            raise RuntimeError("Feedback loop not initialized")
        
        async with self.operation_context(f"Executing {file_path.name}"):
            # Check for interrupt before starting
            if self.keyboard_handler and self.keyboard_handler.is_interrupt_requested():
                return {
                    'success': False,
                    'error': 'Operation interrupted by user',
                    'interrupted': True
                }
            
            # Store execution intent in memory
            if self.memory_manager:
                await self.memory_manager.store(
                    {
                        'event': 'code_execution_start',
                        'file_path': str(file_path),
                        'auto_iterate': auto_iterate,
                        'timestamp': datetime.now().isoformat()
                    },
                    MemoryType.CODE_CONTEXT,
                    Priority.HIGH,
                    tags={'execution', 'feedback'}
                )
            
            # Setup interrupt handling
            if self.keyboard_handler:
                self.feedback_loop.add_interrupt_callback(
                    lambda: self.keyboard_handler.is_interrupt_requested()
                )
            
            # Execute with feedback
            result = await self.feedback_loop.execute_feedback_cycle(
                file_path,
                auto_fix=auto_iterate,
                progress_callback=progress_callback
            )
            
            # Store execution results in memory
            if self.memory_manager:
                await self.memory_manager.store(
                    {
                        'event': 'code_execution_complete',
                        'file_path': str(file_path),
                        'result': result,
                        'timestamp': datetime.now().isoformat()
                    },
                    MemoryType.CODE_CONTEXT,
                    Priority.HIGH,
                    tags={'execution', 'results'}
                )
            
            self.metrics['feedback_cycles'] += 1
            return result
    
    async def store_conversation_context(
        self, 
        user_input: str, 
        assistant_response: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """Store conversation with intelligent context management"""
        if not self.memory_manager:
            return
        
        # Enhanced context with system state
        enhanced_context = context or {}
        enhanced_context.update({
            'active_operation': self.active_operation,
            'todo_visible': self.todo_visible,
            'keyboard_active': self.keyboard_handler is not None and self.keyboard_handler.running,
            'system_metrics': self.get_performance_metrics()
        })
        
        await self.memory_manager.store_conversation_turn(
            user_input,
            assistant_response,
            enhanced_context
        )
        
        self.metrics['memory_operations'] += 1
    
    async def retrieve_relevant_context(
        self, 
        query: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant context using intelligent search"""
        if not self.memory_manager:
            return []
        
        # Search across different memory types
        results = []
        
        # Search conversation history
        conversations = await self.memory_manager.retrieve(
            content_query=query,
            memory_type=MemoryType.CONVERSATION,
            limit=limit // 2
        )
        
        for entry_id, content, metadata in conversations:
            results.append({
                'type': 'conversation',
                'content': content,
                'metadata': metadata,
                'relevance': 'high'
            })
        
        # Search code context
        code_context = await self.memory_manager.retrieve(
            content_query=query,
            memory_type=MemoryType.CODE_CONTEXT,
            limit=limit // 2
        )
        
        for entry_id, content, metadata in code_context:
            results.append({
                'type': 'code_context',
                'content': content,
                'metadata': metadata,
                'relevance': 'high'
            })
        
        return results[:limit]
    
    async def _on_execution_complete(self, metrics):
        """Handle feedback execution completion"""
        if self.on_feedback_received:
            await self._safe_callback(self.on_feedback_received, 'execution_complete', metrics)
        
        # Store in memory
        if self.memory_manager:
            await self.memory_manager.store(
                {
                    'event': 'execution_feedback',
                    'metrics': {
                        'duration': metrics.duration_seconds,
                        'exit_code': metrics.exit_code,
                        'result': metrics.result.value,
                        'stdout_length': len(metrics.stdout),
                        'stderr_length': len(metrics.stderr)
                    },
                    'timestamp': datetime.now().isoformat()
                },
                MemoryType.SYSTEM,
                Priority.MEDIUM,
                tags={'feedback', 'execution'}
            )
    
    async def _on_feedback_analysis(self, analyses):
        """Handle feedback analysis results"""
        if self.on_feedback_received:
            await self._safe_callback(self.on_feedback_received, 'analysis', analyses)
        
        # Store analysis in memory
        if self.memory_manager:
            analysis_data = []
            for analysis in analyses:
                analysis_data.append({
                    'feedback_type': analysis.feedback_type.value,
                    'severity': analysis.severity,
                    'message': analysis.message,
                    'confidence': analysis.confidence,
                    'suggestions': analysis.suggestions[:3]  # Limit for storage
                })
            
            await self.memory_manager.store(
                {
                    'event': 'feedback_analysis',
                    'analyses': analysis_data,
                    'timestamp': datetime.now().isoformat()
                },
                MemoryType.SYSTEM,
                Priority.HIGH,
                tags={'feedback', 'analysis'}
            )
    
    async def _on_keyboard_interrupt(self):
        """Handle keyboard interrupt events"""
        self.metrics['interrupts_handled'] += 1
        
        if self.on_interrupt:
            await self._safe_callback(self.on_interrupt)
        
        # Interrupt feedback loop if active
        if self.feedback_loop:
            self.feedback_loop.interrupt_cycle()
        
        # Store interrupt in memory
        if self.memory_manager:
            await self.memory_manager.store(
                {
                    'event': 'keyboard_interrupt',
                    'active_operation': self.active_operation,
                    'timestamp': datetime.now().isoformat()
                },
                MemoryType.SYSTEM,
                Priority.HIGH,
                tags={'interrupt', 'keyboard'}
            )
    
    async def _on_todo_toggle(self, visible: bool):
        """Handle todo visibility toggle"""
        self.todo_visible = visible
        
        if self.on_todo_toggle:
            await self._safe_callback(self.on_todo_toggle, visible)
        
        # Store state change in memory
        if self.memory_manager:
            await self.memory_manager.store(
                {
                    'event': 'todo_visibility_change',
                    'visible': visible,
                    'timestamp': datetime.now().isoformat()
                },
                MemoryType.SESSION,
                Priority.LOW,
                tags={'ui', 'todo'}
            )
    
    async def _safe_callback(self, callback: Callable, *args, **kwargs):
        """Safely execute callback with error handling"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args, **kwargs)
            else:
                callback(*args, **kwargs)
        except Exception as e:
            if self.config.debug_mode:
                print(f"Callback error: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        metrics = {
            'operations_completed': self.metrics['operations_completed'],
            'interrupts_handled': self.metrics['interrupts_handled'],
            'memory_operations': self.metrics['memory_operations'],
            'feedback_cycles': self.metrics['feedback_cycles'],
            'uptime_seconds': (datetime.now() - self.metrics['session_start']).total_seconds(),
            'average_response_time': sum(self.metrics['response_times']) / max(len(self.metrics['response_times']), 1),
            'active_operation': self.active_operation,
            'todo_visible': self.todo_visible
        }
        
        # Add subsystem metrics
        if self.keyboard_handler:
            metrics['keyboard'] = self.keyboard_handler.get_performance_metrics()
        
        if self.memory_manager:
            metrics['memory'] = self.memory_manager.get_memory_stats()
        
        if self.feedback_loop:
            metrics['feedback'] = self.feedback_loop.get_cycle_metrics()
        
        return metrics
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'initialized': self.initialized,
            'running': self.running,
            'active_operation': self.active_operation,
            'modules': {
                'keyboard': self.keyboard_handler is not None,
                'memory': self.memory_manager is not None,
                'feedback': self.feedback_loop is not None
            },
            'background_tasks': len([t for t in self.background_tasks if not t.done()]),
            'performance': self.get_performance_metrics()
        }
    
    async def _cleanup_partial_init(self):
        """Clean up partially initialized systems"""
        if self.keyboard_handler:
            self.keyboard_handler.shutdown()
        
        if self.memory_manager:
            await self.memory_manager.shutdown()
        
        # Cancel any started background tasks
        for task in self.background_tasks:
            task.cancel()
    
    async def shutdown(self):
        """Gracefully shutdown all integrated systems"""
        if not self.initialized:
            return
        
        print("🔄 Shutting down Claude-level integration...")
        
        self.running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Shutdown subsystems
        if self.keyboard_handler:
            self.keyboard_handler.shutdown()
        
        if self.memory_manager:
            await self.memory_manager.shutdown()
        
        # Final metrics
        final_metrics = self.get_performance_metrics()
        print(f"📊 Session completed:")
        print(f"   Operations: {final_metrics['operations_completed']}")
        print(f"   Memory ops: {final_metrics['memory_operations']}")
        print(f"   Feedback cycles: {final_metrics['feedback_cycles']}")
        print(f"   Uptime: {final_metrics['uptime_seconds']:.1f}s")
        
        self.initialized = False
        print("✅ Claude-level integration shutdown complete")

# Global integration instance
_global_claude_integration: Optional[ClaudeIntegration] = None

def get_claude_integration(
    project_path: Path = None, 
    config: ClaudeIntegrationConfig = None
) -> ClaudeIntegration:
    """Get the global Claude integration instance"""
    global _global_claude_integration
    if _global_claude_integration is None:
        if project_path is None:
            project_path = Path.cwd()
        _global_claude_integration = ClaudeIntegration(project_path, config)
    return _global_claude_integration

async def initialize_claude_integration(
    project_path: Path, 
    config: ClaudeIntegrationConfig = None
) -> ClaudeIntegration:
    """Initialize the global Claude integration system"""
    global _global_claude_integration
    _global_claude_integration = ClaudeIntegration(project_path, config)
    
    if await _global_claude_integration.initialize():
        return _global_claude_integration
    else:
        _global_claude_integration = None
        raise RuntimeError("Failed to initialize Claude integration")