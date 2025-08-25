"""
ABOV3 Genesis - Monitoring & Observability Infrastructure
Enterprise-grade monitoring, logging, and metrics collection
"""

import asyncio
import logging
import time
import json
import threading
import multiprocessing
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import aiofiles
import psutil
import traceback
from collections import defaultdict, deque
import hashlib
import gzip
import os
import sys
import platform
from datetime import datetime, timezone
import queue
import pickle

# Configure structured logging
class LogLevel(Enum):
    """Log levels"""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

class MetricType(Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    SET = "set"

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: float
    level: LogLevel
    logger_name: str
    message: str
    module: str
    function: str
    line_number: int
    thread_id: int
    process_id: int
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    component: str = "unknown"
    operation: str = "unknown"
    tags: Dict[str, str] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[Dict[str, str]] = None
    duration_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        entry_dict = {
            'timestamp': self.timestamp,
            'iso_timestamp': datetime.fromtimestamp(self.timestamp, timezone.utc).isoformat(),
            'level': self.level.name,
            'logger': self.logger_name,
            'message': self.message,
            'module': self.module,
            'function': self.function,
            'line': self.line_number,
            'thread_id': self.thread_id,
            'process_id': self.process_id,
            'component': self.component,
            'operation': self.operation
        }
        
        # Add optional fields
        if self.correlation_id:
            entry_dict['correlation_id'] = self.correlation_id
        if self.user_id:
            entry_dict['user_id'] = self.user_id
        if self.session_id:
            entry_dict['session_id'] = self.session_id
        if self.tags:
            entry_dict['tags'] = self.tags
        if self.context:
            entry_dict['context'] = self.context
        if self.exception:
            entry_dict['exception'] = self.exception
        if self.duration_ms is not None:
            entry_dict['duration_ms'] = self.duration_ms
        
        return entry_dict

@dataclass
class Metric:
    """Metric data point"""
    name: str
    metric_type: MetricType
    value: Union[int, float]
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'type': self.metric_type.value,
            'value': self.value,
            'timestamp': self.timestamp,
            'iso_timestamp': datetime.fromtimestamp(self.timestamp, timezone.utc).isoformat(),
            'tags': self.tags,
            'unit': self.unit
        }

@dataclass
class Alert:
    """Alert definition"""
    alert_id: str
    name: str
    condition: str
    severity: AlertSeverity
    threshold: Union[int, float]
    comparison: str  # '>', '<', '>=', '<=', '==', '!='
    metric_name: str
    window_minutes: int = 5
    enabled: bool = True
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_triggered: Optional[float] = None
    trigger_count: int = 0

class StructuredLogger:
    """
    High-performance structured logger with async I/O
    """

    def __init__(
        self,
        name: str,
        log_file: Optional[Path] = None,
        level: LogLevel = LogLevel.INFO,
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        max_files: int = 10,
        compress_old: bool = True,
        async_writing: bool = True
    ):
        self.name = name
        self.log_file = log_file
        self.level = level
        self.max_file_size = max_file_size
        self.max_files = max_files
        self.compress_old = compress_old
        self.async_writing = async_writing
        
        # Async log writing
        self._log_queue = asyncio.Queue(maxsize=10000) if async_writing else None
        self._writer_task = None
        self._current_file_size = 0
        self._lock = asyncio.Lock() if async_writing else threading.Lock()
        
        # Context tracking
        self._context_stack = []
        
        # Initialize file
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            if log_file.exists():
                self._current_file_size = log_file.stat().st_size
        
        # Start async writer
        if async_writing and log_file:
            self._start_async_writer()

    def _start_async_writer(self):
        """Start async log writer"""
        if not self._writer_task or self._writer_task.done():
            self._writer_task = asyncio.create_task(self._write_logs_loop())

    async def _write_logs_loop(self):
        """Async log writing loop"""
        while True:
            try:
                # Get log entry from queue
                log_entry = await self._log_queue.get()
                
                if log_entry is None:  # Shutdown signal
                    break
                
                # Write to file
                await self._write_log_entry(log_entry)
                
                # Mark task as done
                self._log_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Log writing error: {e}", file=sys.stderr)

    async def _write_log_entry(self, log_entry: LogEntry):
        """Write log entry to file"""
        if not self.log_file:
            return
        
        try:
            # Check for log rotation
            if self._current_file_size > self.max_file_size:
                await self._rotate_logs()
            
            # Serialize log entry
            log_line = json.dumps(log_entry.to_dict()) + '\n'
            
            # Write to file
            async with aiofiles.open(self.log_file, 'a', encoding='utf-8') as f:
                await f.write(log_line)
            
            # Update file size
            self._current_file_size += len(log_line.encode('utf-8'))
            
        except Exception as e:
            print(f"Failed to write log entry: {e}", file=sys.stderr)

    async def _rotate_logs(self):
        """Rotate log files"""
        try:
            # Move current log files
            for i in range(self.max_files - 1, 0, -1):
                old_file = self.log_file.with_suffix(f'.{i}{self.log_file.suffix}')
                new_file = self.log_file.with_suffix(f'.{i + 1}{self.log_file.suffix}')
                
                if old_file.exists():
                    if i + 1 >= self.max_files:
                        # Delete oldest file
                        old_file.unlink()
                    else:
                        old_file.rename(new_file)
            
            # Move current log to .1
            if self.log_file.exists():
                rotated_file = self.log_file.with_suffix(f'.1{self.log_file.suffix}')
                self.log_file.rename(rotated_file)
                
                # Compress if enabled
                if self.compress_old:
                    await self._compress_log_file(rotated_file)
            
            # Reset file size counter
            self._current_file_size = 0
            
        except Exception as e:
            print(f"Log rotation failed: {e}", file=sys.stderr)

    async def _compress_log_file(self, file_path: Path):
        """Compress log file"""
        try:
            compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
            
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            
            # Remove original file
            file_path.unlink()
            
        except Exception as e:
            print(f"Log compression failed: {e}", file=sys.stderr)

    def _create_log_entry(
        self,
        level: LogLevel,
        message: str,
        **kwargs
    ) -> LogEntry:
        """Create structured log entry"""
        
        # Get caller information
        frame = sys._getframe(3)  # Skip internal function calls
        
        # Get current context
        context = {}
        tags = {}
        
        if self._context_stack:
            for ctx in self._context_stack:
                context.update(ctx.get('context', {}))
                tags.update(ctx.get('tags', {}))
        
        # Merge with provided context
        context.update(kwargs.get('context', {}))
        tags.update(kwargs.get('tags', {}))
        
        # Create log entry
        log_entry = LogEntry(
            timestamp=time.time(),
            level=level,
            logger_name=self.name,
            message=message,
            module=frame.f_globals.get('__name__', 'unknown'),
            function=frame.f_code.co_name,
            line_number=frame.f_lineno,
            thread_id=threading.get_ident(),
            process_id=os.getpid(),
            correlation_id=kwargs.get('correlation_id'),
            user_id=kwargs.get('user_id'),
            session_id=kwargs.get('session_id'),
            component=kwargs.get('component', 'unknown'),
            operation=kwargs.get('operation', 'unknown'),
            tags=tags,
            context=context,
            duration_ms=kwargs.get('duration_ms')
        )
        
        # Add exception information if available
        if kwargs.get('exc_info') and sys.exc_info()[0]:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            log_entry.exception = {
                'type': exc_type.__name__,
                'message': str(exc_value),
                'traceback': traceback.format_exception(exc_type, exc_value, exc_traceback)
            }
        
        return log_entry

    async def log(self, level: LogLevel, message: str, **kwargs):
        """Log message at specified level"""
        if level.value < self.level.value:
            return
        
        log_entry = self._create_log_entry(level, message, **kwargs)
        
        # Print to console for debugging
        if level.value >= LogLevel.WARNING.value:
            print(f"[{level.name}] {message}", file=sys.stderr)
        
        if self.async_writing and self._log_queue:
            try:
                await self._log_queue.put(log_entry)
            except asyncio.QueueFull:
                print(f"Log queue full, dropping message: {message}", file=sys.stderr)
        elif self.log_file:
            await self._write_log_entry(log_entry)

    async def trace(self, message: str, **kwargs):
        """Log trace message"""
        await self.log(LogLevel.TRACE, message, **kwargs)

    async def debug(self, message: str, **kwargs):
        """Log debug message"""
        await self.log(LogLevel.DEBUG, message, **kwargs)

    async def info(self, message: str, **kwargs):
        """Log info message"""
        await self.log(LogLevel.INFO, message, **kwargs)

    async def warning(self, message: str, **kwargs):
        """Log warning message"""
        await self.log(LogLevel.WARNING, message, **kwargs)

    async def error(self, message: str, **kwargs):
        """Log error message"""
        await self.log(LogLevel.ERROR, message, **kwargs)

    async def critical(self, message: str, **kwargs):
        """Log critical message"""
        await self.log(LogLevel.CRITICAL, message, **kwargs)

    def push_context(self, **context):
        """Push logging context"""
        self._context_stack.append({
            'context': context.get('context', {}),
            'tags': context.get('tags', {})
        })

    def pop_context(self):
        """Pop logging context"""
        if self._context_stack:
            self._context_stack.pop()

    async def close(self):
        """Close logger and cleanup"""
        if self._writer_task:
            # Signal shutdown
            await self._log_queue.put(None)
            
            # Wait for queue to be processed
            await self._log_queue.join()
            
            # Cancel writer task
            self._writer_task.cancel()
            try:
                await self._writer_task
            except asyncio.CancelledError:
                pass

class MetricsCollector:
    """
    High-performance metrics collection and aggregation
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        flush_interval: float = 10.0,
        max_metrics_in_memory: int = 100000
    ):
        self.storage_path = storage_path
        self.flush_interval = flush_interval
        self.max_metrics_in_memory = max_metrics_in_memory
        
        # Metrics storage
        self._metrics: List[Metric] = []
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._timers: Dict[str, List[float]] = defaultdict(list)
        self._sets: Dict[str, set] = defaultdict(set)
        
        # Background processing
        self._flush_task = None
        self._lock = asyncio.Lock()
        
        # Statistics
        self._stats = {
            'metrics_collected': 0,
            'metrics_flushed': 0,
            'last_flush': 0
        }
        
        # Start background flushing
        self._start_flushing()

    def _start_flushing(self):
        """Start background metrics flushing"""
        if not self._flush_task or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._flush_loop())

    async def _flush_loop(self):
        """Background metrics flushing loop"""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Metrics flush error: {e}", file=sys.stderr)

    async def record_metric(
        self,
        name: str,
        value: Union[int, float],
        metric_type: MetricType,
        tags: Dict[str, str] = None,
        unit: str = None
    ):
        """Record a metric"""
        async with self._lock:
            metric = Metric(
                name=name,
                metric_type=metric_type,
                value=value,
                timestamp=time.time(),
                tags=tags or {},
                unit=unit
            )
            
            # Update internal aggregations
            if metric_type == MetricType.COUNTER:
                self._counters[name] += value
            elif metric_type == MetricType.GAUGE:
                self._gauges[name] = value
            elif metric_type == MetricType.HISTOGRAM:
                self._histograms[name].append(value)
            elif metric_type == MetricType.TIMER:
                self._timers[name].append(value)
            elif metric_type == MetricType.SET:
                self._sets[name].add(value)
            
            # Store raw metric
            self._metrics.append(metric)
            self._stats['metrics_collected'] += 1
            
            # Check if we need to flush
            if len(self._metrics) >= self.max_metrics_in_memory:
                await self._flush_metrics()

    async def increment(
        self,
        name: str,
        value: Union[int, float] = 1,
        tags: Dict[str, str] = None
    ):
        """Increment a counter"""
        await self.record_metric(name, value, MetricType.COUNTER, tags)

    async def gauge(
        self,
        name: str,
        value: Union[int, float],
        tags: Dict[str, str] = None,
        unit: str = None
    ):
        """Record a gauge value"""
        await self.record_metric(name, value, MetricType.GAUGE, tags, unit)

    async def histogram(
        self,
        name: str,
        value: Union[int, float],
        tags: Dict[str, str] = None,
        unit: str = None
    ):
        """Record a histogram value"""
        await self.record_metric(name, value, MetricType.HISTOGRAM, tags, unit)

    async def timer(
        self,
        name: str,
        value: Union[int, float],
        tags: Dict[str, str] = None
    ):
        """Record a timer value (in milliseconds)"""
        await self.record_metric(name, value, MetricType.TIMER, tags, 'ms')

    async def unique(
        self,
        name: str,
        value: Union[str, int],
        tags: Dict[str, str] = None
    ):
        """Record a unique value (set)"""
        await self.record_metric(name, value, MetricType.SET, tags)

    async def _flush_metrics(self):
        """Flush metrics to storage"""
        if not self.storage_path or not self._metrics:
            return
        
        try:
            # Create metrics data
            flush_data = {
                'timestamp': time.time(),
                'metrics': [m.to_dict() for m in self._metrics],
                'aggregations': {
                    'counters': dict(self._counters),
                    'gauges': dict(self._gauges),
                    'histograms': {k: {
                        'count': len(v),
                        'min': min(v) if v else 0,
                        'max': max(v) if v else 0,
                        'avg': sum(v) / len(v) if v else 0,
                        'sum': sum(v)
                    } for k, v in self._histograms.items()},
                    'timers': {k: {
                        'count': len(v),
                        'min': min(v) if v else 0,
                        'max': max(v) if v else 0,
                        'avg': sum(v) / len(v) if v else 0,
                        'p50': sorted(v)[len(v)//2] if v else 0,
                        'p95': sorted(v)[int(len(v)*0.95)] if v else 0,
                        'p99': sorted(v)[int(len(v)*0.99)] if v else 0
                    } for k, v in self._timers.items()},
                    'sets': {k: len(v) for k, v in self._sets.items()}
                }
            }
            
            # Write to file
            timestamp = int(time.time())
            metrics_file = self.storage_path / f'metrics_{timestamp}.json'
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            
            async with aiofiles.open(metrics_file, 'w') as f:
                await f.write(json.dumps(flush_data, indent=2))
            
            # Update stats
            self._stats['metrics_flushed'] += len(self._metrics)
            self._stats['last_flush'] = time.time()
            
            # Clear metrics
            self._metrics.clear()
            
            # Reset aggregations for counters only (keep gauges, reset histograms)
            self._histograms.clear()
            self._timers.clear()
            self._sets.clear()
            
        except Exception as e:
            print(f"Failed to flush metrics: {e}", file=sys.stderr)

    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metric aggregations"""
        async with self._lock:
            return {
                'counters': dict(self._counters),
                'gauges': dict(self._gauges),
                'histograms': {k: len(v) for k, v in self._histograms.items()},
                'timers': {k: len(v) for k, v in self._timers.items()},
                'sets': {k: len(v) for k, v in self._sets.items()},
                'stats': self._stats.copy()
            }

    async def close(self):
        """Close metrics collector"""
        if self._flush_task:
            # Final flush
            await self._flush_metrics()
            
            # Cancel flush task
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

class SystemMonitor:
    """
    System resource monitoring
    """

    def __init__(
        self,
        metrics_collector: MetricsCollector,
        monitor_interval: float = 5.0
    ):
        self.metrics_collector = metrics_collector
        self.monitor_interval = monitor_interval
        
        # Monitoring task
        self._monitor_task = None
        
        # Previous values for calculating deltas
        self._prev_cpu_times = None
        self._prev_network_io = None
        self._prev_disk_io = None
        
        # Start monitoring
        self._start_monitoring()

    def _start_monitoring(self):
        """Start system monitoring"""
        if not self._monitor_task or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def _monitor_loop(self):
        """System monitoring loop"""
        while True:
            try:
                await asyncio.sleep(self.monitor_interval)
                await self._collect_system_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"System monitoring error: {e}", file=sys.stderr)

    async def _collect_system_metrics(self):
        """Collect system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            await self.metrics_collector.gauge('system.cpu.usage_percent', cpu_percent)
            
            cpu_count = psutil.cpu_count()
            await self.metrics_collector.gauge('system.cpu.count', cpu_count)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            await self.metrics_collector.gauge('system.memory.total', memory.total)
            await self.metrics_collector.gauge('system.memory.used', memory.used)
            await self.metrics_collector.gauge('system.memory.available', memory.available)
            await self.metrics_collector.gauge('system.memory.usage_percent', memory.percent)
            
            # Disk metrics
            disk_usage = psutil.disk_usage('/')
            await self.metrics_collector.gauge('system.disk.total', disk_usage.total)
            await self.metrics_collector.gauge('system.disk.used', disk_usage.used)
            await self.metrics_collector.gauge('system.disk.free', disk_usage.free)
            await self.metrics_collector.gauge('system.disk.usage_percent', disk_usage.used / disk_usage.total * 100)
            
            # Network I/O
            network_io = psutil.net_io_counters()
            if network_io:
                await self.metrics_collector.gauge('system.network.bytes_sent', network_io.bytes_sent)
                await self.metrics_collector.gauge('system.network.bytes_recv', network_io.bytes_recv)
                await self.metrics_collector.gauge('system.network.packets_sent', network_io.packets_sent)
                await self.metrics_collector.gauge('system.network.packets_recv', network_io.packets_recv)
                
                # Calculate rates if we have previous values
                if self._prev_network_io:
                    time_delta = self.monitor_interval
                    bytes_sent_rate = (network_io.bytes_sent - self._prev_network_io.bytes_sent) / time_delta
                    bytes_recv_rate = (network_io.bytes_recv - self._prev_network_io.bytes_recv) / time_delta
                    
                    await self.metrics_collector.gauge('system.network.bytes_sent_per_sec', bytes_sent_rate)
                    await self.metrics_collector.gauge('system.network.bytes_recv_per_sec', bytes_recv_rate)
                
                self._prev_network_io = network_io
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                await self.metrics_collector.gauge('system.disk.read_bytes', disk_io.read_bytes)
                await self.metrics_collector.gauge('system.disk.write_bytes', disk_io.write_bytes)
                await self.metrics_collector.gauge('system.disk.read_count', disk_io.read_count)
                await self.metrics_collector.gauge('system.disk.write_count', disk_io.write_count)
                
                # Calculate rates
                if self._prev_disk_io:
                    time_delta = self.monitor_interval
                    read_bytes_rate = (disk_io.read_bytes - self._prev_disk_io.read_bytes) / time_delta
                    write_bytes_rate = (disk_io.write_bytes - self._prev_disk_io.write_bytes) / time_delta
                    
                    await self.metrics_collector.gauge('system.disk.read_bytes_per_sec', read_bytes_rate)
                    await self.metrics_collector.gauge('system.disk.write_bytes_per_sec', write_bytes_rate)
                
                self._prev_disk_io = disk_io
            
            # Process metrics
            current_process = psutil.Process()
            
            # Process CPU and memory
            await self.metrics_collector.gauge('process.cpu.usage_percent', current_process.cpu_percent())
            
            memory_info = current_process.memory_info()
            await self.metrics_collector.gauge('process.memory.rss', memory_info.rss)
            await self.metrics_collector.gauge('process.memory.vms', memory_info.vms)
            await self.metrics_collector.gauge('process.memory.percent', current_process.memory_percent())
            
            # Process file descriptors (Unix only)
            if hasattr(current_process, 'num_fds'):
                try:
                    await self.metrics_collector.gauge('process.file_descriptors', current_process.num_fds())
                except:
                    pass
            
            # Thread count
            await self.metrics_collector.gauge('process.threads', current_process.num_threads())
            
        except Exception as e:
            print(f"Failed to collect system metrics: {e}", file=sys.stderr)

    async def stop_monitoring(self):
        """Stop system monitoring"""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

class AlertManager:
    """
    Alert management and notification system
    """

    def __init__(
        self,
        metrics_collector: MetricsCollector,
        logger: StructuredLogger,
        check_interval: float = 60.0
    ):
        self.metrics_collector = metrics_collector
        self.logger = logger
        self.check_interval = check_interval
        
        # Alert definitions
        self._alerts: Dict[str, Alert] = {}
        
        # Alert history
        self._alert_history = deque(maxlen=1000)
        
        # Alert checking task
        self._check_task = None
        
        # Notification handlers
        self._notification_handlers: List[Callable] = []
        
        # Start alert checking
        self._start_alert_checking()

    def _start_alert_checking(self):
        """Start alert checking"""
        if not self._check_task or self._check_task.done():
            self._check_task = asyncio.create_task(self._check_alerts_loop())

    async def _check_alerts_loop(self):
        """Alert checking loop"""
        while True:
            try:
                await asyncio.sleep(self.check_interval)
                await self._check_all_alerts()
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self.logger.error(f"Alert checking error: {e}", exc_info=True)

    def add_alert(self, alert: Alert):
        """Add an alert definition"""
        self._alerts[alert.alert_id] = alert

    def remove_alert(self, alert_id: str):
        """Remove an alert definition"""
        self._alerts.pop(alert_id, None)

    def add_notification_handler(self, handler: Callable):
        """Add notification handler"""
        self._notification_handlers.append(handler)

    async def _check_all_alerts(self):
        """Check all alerts"""
        current_metrics = await self.metrics_collector.get_current_metrics()
        
        for alert in self._alerts.values():
            if not alert.enabled:
                continue
            
            try:
                await self._check_alert(alert, current_metrics)
            except Exception as e:
                await self.logger.error(f"Error checking alert {alert.alert_id}: {e}")

    async def _check_alert(self, alert: Alert, metrics: Dict[str, Any]):
        """Check individual alert"""
        # Get metric value
        metric_value = self._get_metric_value(alert.metric_name, metrics)
        
        if metric_value is None:
            return
        
        # Check condition
        triggered = self._evaluate_condition(
            metric_value,
            alert.comparison,
            alert.threshold
        )
        
        if triggered:
            # Update alert
            alert.last_triggered = time.time()
            alert.trigger_count += 1
            
            # Create alert event
            alert_event = {
                'alert_id': alert.alert_id,
                'alert_name': alert.name,
                'severity': alert.severity.value,
                'metric_name': alert.metric_name,
                'metric_value': metric_value,
                'threshold': alert.threshold,
                'comparison': alert.comparison,
                'condition': alert.condition,
                'triggered_at': time.time(),
                'trigger_count': alert.trigger_count,
                'description': alert.description,
                'tags': alert.tags
            }
            
            # Add to history
            self._alert_history.append(alert_event)
            
            # Log alert
            await self.logger.warning(
                f"Alert triggered: {alert.name}",
                component="alert_manager",
                operation="alert_check",
                context=alert_event
            )
            
            # Send notifications
            await self._send_notifications(alert_event)

    def _get_metric_value(self, metric_name: str, metrics: Dict[str, Any]) -> Optional[float]:
        """Get metric value from metrics data"""
        # Check counters
        if metric_name in metrics.get('counters', {}):
            return metrics['counters'][metric_name]
        
        # Check gauges
        if metric_name in metrics.get('gauges', {}):
            return metrics['gauges'][metric_name]
        
        # Check histograms (use count)
        if metric_name in metrics.get('histograms', {}):
            return metrics['histograms'][metric_name]
        
        # Check timers (use count)
        if metric_name in metrics.get('timers', {}):
            return metrics['timers'][metric_name]
        
        # Check sets (use cardinality)
        if metric_name in metrics.get('sets', {}):
            return metrics['sets'][metric_name]
        
        return None

    def _evaluate_condition(
        self,
        value: float,
        comparison: str,
        threshold: float
    ) -> bool:
        """Evaluate alert condition"""
        if comparison == '>':
            return value > threshold
        elif comparison == '<':
            return value < threshold
        elif comparison == '>=':
            return value >= threshold
        elif comparison == '<=':
            return value <= threshold
        elif comparison == '==':
            return value == threshold
        elif comparison == '!=':
            return value != threshold
        else:
            return False

    async def _send_notifications(self, alert_event: Dict[str, Any]):
        """Send alert notifications"""
        for handler in self._notification_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert_event)
                else:
                    handler(alert_event)
            except Exception as e:
                await self.logger.error(f"Notification handler error: {e}")

    async def get_alert_status(self) -> Dict[str, Any]:
        """Get alert system status"""
        return {
            'total_alerts': len(self._alerts),
            'enabled_alerts': len([a for a in self._alerts.values() if a.enabled]),
            'recent_triggers': list(self._alert_history)[-10:],  # Last 10 triggers
            'alert_definitions': {aid: {
                'name': a.name,
                'severity': a.severity.value,
                'enabled': a.enabled,
                'trigger_count': a.trigger_count,
                'last_triggered': a.last_triggered
            } for aid, a in self._alerts.items()}
        }

    async def stop_alert_checking(self):
        """Stop alert checking"""
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass

class ObservabilityManager:
    """
    Main observability coordinator
    Integrates logging, metrics, and alerting
    """

    def __init__(self, project_path: Path):
        self.project_path = project_path
        
        # Initialize components
        logs_path = project_path / '.abov3' / 'logs'
        metrics_path = project_path / '.abov3' / 'metrics'
        
        # Create directories
        logs_path.mkdir(parents=True, exist_ok=True)
        metrics_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = StructuredLogger(
            name="abov3.genesis",
            log_file=logs_path / "application.log",
            level=LogLevel.INFO
        )
        
        # Initialize metrics collector
        self.metrics_collector = MetricsCollector(storage_path=metrics_path)
        
        # Initialize system monitor
        self.system_monitor = SystemMonitor(self.metrics_collector)
        
        # Initialize alert manager
        self.alert_manager = AlertManager(self.metrics_collector, self.logger)
        
        # Setup default alerts
        self._setup_default_alerts()
        
        # Application metrics
        self._app_start_time = time.time()

    def _setup_default_alerts(self):
        """Setup default system alerts"""
        default_alerts = [
            Alert(
                alert_id="high_cpu_usage",
                name="High CPU Usage",
                condition="system.cpu.usage_percent > 80",
                severity=AlertSeverity.HIGH,
                threshold=80.0,
                comparison=">",
                metric_name="system.cpu.usage_percent",
                description="System CPU usage is above 80%"
            ),
            Alert(
                alert_id="high_memory_usage",
                name="High Memory Usage", 
                condition="system.memory.usage_percent > 90",
                severity=AlertSeverity.CRITICAL,
                threshold=90.0,
                comparison=">",
                metric_name="system.memory.usage_percent",
                description="System memory usage is above 90%"
            ),
            Alert(
                alert_id="disk_space_low",
                name="Low Disk Space",
                condition="system.disk.usage_percent > 85",
                severity=AlertSeverity.HIGH,
                threshold=85.0,
                comparison=">",
                metric_name="system.disk.usage_percent",
                description="Disk space usage is above 85%"
            )
        ]
        
        for alert in default_alerts:
            self.alert_manager.add_alert(alert)

    async def record_application_event(
        self,
        event_name: str,
        duration_ms: Optional[float] = None,
        **context
    ):
        """Record application event"""
        # Log the event
        await self.logger.info(
            f"Application event: {event_name}",
            component="application",
            operation=event_name,
            duration_ms=duration_ms,
            context=context
        )
        
        # Record metrics
        await self.metrics_collector.increment(f"app.events.{event_name}")
        
        if duration_ms is not None:
            await self.metrics_collector.timer(f"app.duration.{event_name}", duration_ms)

    async def record_user_action(
        self,
        action: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **context
    ):
        """Record user action"""
        await self.logger.info(
            f"User action: {action}",
            component="user_interface",
            operation=action,
            user_id=user_id,
            session_id=session_id,
            context=context
        )
        
        await self.metrics_collector.increment("app.user_actions", tags={'action': action})
        
        if user_id:
            await self.metrics_collector.unique("app.active_users", user_id)

    async def record_ai_request(
        self,
        model_name: str,
        request_type: str,
        duration_ms: float,
        success: bool,
        tokens_used: int = 0,
        **context
    ):
        """Record AI model request"""
        await self.logger.info(
            f"AI request: {request_type} with {model_name}",
            component="ai_integration",
            operation=request_type,
            duration_ms=duration_ms,
            context={
                'model_name': model_name,
                'success': success,
                'tokens_used': tokens_used,
                **context
            }
        )
        
        # Metrics
        await self.metrics_collector.increment("app.ai_requests", tags={
            'model': model_name,
            'type': request_type,
            'status': 'success' if success else 'error'
        })
        
        await self.metrics_collector.timer("app.ai_request_duration", duration_ms, tags={
            'model': model_name,
            'type': request_type
        })
        
        if tokens_used > 0:
            await self.metrics_collector.gauge("app.ai_tokens_used", tokens_used, tags={
                'model': model_name
            })

    async def get_observability_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive observability dashboard"""
        # Get current metrics
        metrics = await self.metrics_collector.get_current_metrics()
        
        # Get alert status
        alert_status = await self.alert_manager.get_alert_status()
        
        # Application uptime
        uptime = time.time() - self._app_start_time
        
        # System info
        system_info = {
            'platform': platform.system(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'uptime_seconds': uptime
        }
        
        return {
            'system_info': system_info,
            'metrics': metrics,
            'alerts': alert_status,
            'uptime_seconds': uptime,
            'logs_location': str(self.project_path / '.abov3' / 'logs'),
            'metrics_location': str(self.project_path / '.abov3' / 'metrics')
        }

    async def cleanup(self):
        """Cleanup all observability components"""
        await self.logger.close()
        await self.metrics_collector.close()
        await self.system_monitor.stop_monitoring()
        await self.alert_manager.stop_alert_checking()

# Context managers
class log_context:
    """Context manager for structured logging"""
    
    def __init__(self, logger: StructuredLogger, **context):
        self.logger = logger
        self.context = context

    def __enter__(self):
        self.logger.push_context(**self.context)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.pop_context()

class timer_context:
    """Context manager for timing operations"""
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        metric_name: str,
        tags: Dict[str, str] = None
    ):
        self.metrics_collector = metrics_collector
        self.metric_name = metric_name
        self.tags = tags or {}
        self.start_time = None

    async def __aenter__(self):
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (time.time() - self.start_time) * 1000  # Convert to ms
            await self.metrics_collector.timer(self.metric_name, duration, self.tags)