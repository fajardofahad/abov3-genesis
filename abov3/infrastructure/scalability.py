"""
ABOV3 Genesis - Scalability Infrastructure
Enterprise-grade scalability, load balancing, and resource management
"""

import asyncio
import time
import threading
import multiprocessing
from typing import Dict, Any, Optional, List, Callable, Union, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import hashlib
from pathlib import Path
import aiofiles
from collections import defaultdict, deque
import psutil
import weakref
import heapq
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import uuid
import pickle

logger = logging.getLogger(__name__)

T = TypeVar('T')

class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESOURCE_BASED = "resource_based"
    CONSISTENT_HASH = "consistent_hash"

class ResourceType(Enum):
    """Resource types for management"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    GPU = "gpu"
    AI_MODEL = "ai_model"

class ScalingDirection(Enum):
    """Scaling directions"""
    UP = "up"      # Scale up resources
    DOWN = "down"  # Scale down resources
    OUT = "out"    # Scale out instances
    IN = "in"      # Scale in instances

@dataclass
class WorkerNode:
    """Represents a worker node in the distributed system"""
    node_id: str
    host: str
    port: int
    weight: float = 1.0
    current_connections: int = 0
    max_connections: int = 100
    avg_response_time: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    last_heartbeat: float = field(default_factory=time.time)
    is_healthy: bool = True
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def utilization(self) -> float:
        """Get overall node utilization"""
        connection_util = self.current_connections / max(1, self.max_connections)
        resource_util = (self.cpu_usage + self.memory_usage) / 200.0
        return (connection_util + resource_util) / 2.0

    @property
    def is_available(self) -> bool:
        """Check if node is available for new requests"""
        return (self.is_healthy and 
                self.current_connections < self.max_connections and
                time.time() - self.last_heartbeat < 30)

@dataclass
class ResourceQuota:
    """Resource quota and limits"""
    cpu_cores: Optional[float] = None
    memory_mb: Optional[int] = None
    disk_mb: Optional[int] = None
    network_mbps: Optional[float] = None
    gpu_memory_mb: Optional[int] = None
    concurrent_requests: Optional[int] = None
    requests_per_minute: Optional[int] = None

@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration"""
    min_instances: int = 1
    max_instances: int = 10
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    target_response_time_ms: float = 1000.0
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 30.0
    cooldown_period: float = 300.0  # 5 minutes
    scale_up_step: int = 1
    scale_down_step: int = 1

class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4

@dataclass
class Task:
    """Distributed task representation"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    function: Callable = None
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    timeout: Optional[float] = None
    retries: int = 0
    max_retries: int = 3
    resource_requirements: Optional[ResourceQuota] = None
    affinity: Optional[List[str]] = None  # Node capabilities required
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        """For priority queue ordering"""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.created_at < other.created_at

class LoadBalancer:
    """
    Intelligent load balancer with multiple strategies
    Routes requests to optimal worker nodes
    """

    def __init__(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_CONNECTIONS,
        health_check_interval: float = 30.0
    ):
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        
        # Worker nodes
        self._nodes: Dict[str, WorkerNode] = {}
        self._node_order: List[str] = []  # For round robin
        self._current_index = 0
        
        # Consistent hashing ring (for consistent hash strategy)
        self._hash_ring: List[tuple] = []  # (hash_value, node_id)
        
        # Statistics
        self._request_count = 0
        self._total_response_time = 0.0
        self._node_selection_history = deque(maxlen=1000)
        
        # Health checking
        self._health_check_task = None
        self._lock = asyncio.Lock()
        
        # Start health checking
        self._start_health_checking()

    def _start_health_checking(self):
        """Start background health checking"""
        if self._health_check_task is None:
            self._health_check_task = asyncio.create_task(self._health_check_loop())

    async def _health_check_loop(self):
        """Background health checking loop"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._check_node_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def _check_node_health(self):
        """Check health of all nodes"""
        current_time = time.time()
        unhealthy_nodes = []
        
        async with self._lock:
            for node_id, node in self._nodes.items():
                # Mark as unhealthy if no heartbeat for 30 seconds
                if current_time - node.last_heartbeat > 30:
                    if node.is_healthy:
                        node.is_healthy = False
                        unhealthy_nodes.append(node_id)
                        logger.warning(f"Node {node_id} marked as unhealthy")
                
                # Auto-recover if heartbeat resumes
                elif not node.is_healthy and current_time - node.last_heartbeat < 10:
                    node.is_healthy = True
                    logger.info(f"Node {node_id} recovered")
        
        # Update hash ring if needed
        if unhealthy_nodes and self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
            await self._rebuild_hash_ring()

    async def add_node(self, node: WorkerNode):
        """Add a worker node"""
        async with self._lock:
            self._nodes[node.node_id] = node
            self._node_order.append(node.node_id)
            
            # Update hash ring for consistent hashing
            if self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
                await self._rebuild_hash_ring()
            
            logger.info(f"Added node {node.node_id} ({node.host}:{node.port})")

    async def remove_node(self, node_id: str):
        """Remove a worker node"""
        async with self._lock:
            if node_id in self._nodes:
                del self._nodes[node_id]
                if node_id in self._node_order:
                    self._node_order.remove(node_id)
                
                # Update hash ring
                if self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
                    await self._rebuild_hash_ring()
                
                logger.info(f"Removed node {node_id}")

    async def _rebuild_hash_ring(self):
        """Rebuild consistent hash ring"""
        self._hash_ring.clear()
        
        # Add multiple points for each node (virtual nodes)
        for node_id, node in self._nodes.items():
            if node.is_available:
                # Create virtual nodes based on weight
                virtual_nodes = max(1, int(node.weight * 100))
                for i in range(virtual_nodes):
                    hash_key = f"{node_id}:{i}"
                    hash_value = int(hashlib.md5(hash_key.encode()).hexdigest(), 16)
                    self._hash_ring.append((hash_value, node_id))
        
        # Sort by hash value
        self._hash_ring.sort()

    async def select_node(self, task: Optional[Task] = None, routing_key: str = None) -> Optional[WorkerNode]:
        """Select optimal node based on strategy"""
        async with self._lock:
            available_nodes = [
                node for node in self._nodes.values()
                if node.is_available
            ]
            
            if not available_nodes:
                return None
            
            # Filter by affinity if specified
            if task and task.affinity:
                available_nodes = [
                    node for node in available_nodes
                    if any(capability in node.capabilities for capability in task.affinity)
                ]
                
                if not available_nodes:
                    return None
            
            selected_node = None
            
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                selected_node = await self._select_round_robin(available_nodes)
            
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                selected_node = min(available_nodes, key=lambda n: n.current_connections)
            
            elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
                selected_node = min(available_nodes, key=lambda n: n.avg_response_time)
            
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                selected_node = await self._select_weighted_round_robin(available_nodes)
            
            elif self.strategy == LoadBalancingStrategy.RESOURCE_BASED:
                selected_node = min(available_nodes, key=lambda n: n.utilization)
            
            elif self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
                selected_node = await self._select_consistent_hash(available_nodes, routing_key or str(task.task_id if task else random.random()))
            
            if selected_node:
                selected_node.current_connections += 1
                self._node_selection_history.append((time.time(), selected_node.node_id))
            
            return selected_node

    async def _select_round_robin(self, available_nodes: List[WorkerNode]) -> WorkerNode:
        """Round robin selection"""
        if not self._node_order:
            return available_nodes[0]
        
        # Find next available node in order
        start_index = self._current_index
        for _ in range(len(self._node_order)):
            node_id = self._node_order[self._current_index]
            self._current_index = (self._current_index + 1) % len(self._node_order)
            
            if node_id in self._nodes and self._nodes[node_id] in available_nodes:
                return self._nodes[node_id]
        
        # Fallback to first available
        return available_nodes[0]

    async def _select_weighted_round_robin(self, available_nodes: List[WorkerNode]) -> WorkerNode:
        """Weighted round robin selection"""
        # Create weighted list
        weighted_nodes = []
        for node in available_nodes:
            weight = max(1, int(node.weight * 10))
            weighted_nodes.extend([node] * weight)
        
        if weighted_nodes:
            index = self._current_index % len(weighted_nodes)
            self._current_index += 1
            return weighted_nodes[index]
        
        return available_nodes[0]

    async def _select_consistent_hash(self, available_nodes: List[WorkerNode], routing_key: str) -> WorkerNode:
        """Consistent hash selection"""
        if not self._hash_ring:
            return available_nodes[0]
        
        # Hash the routing key
        key_hash = int(hashlib.md5(routing_key.encode()).hexdigest(), 16)
        
        # Find the first node with hash >= key_hash
        for hash_value, node_id in self._hash_ring:
            if hash_value >= key_hash and node_id in self._nodes:
                node = self._nodes[node_id]
                if node in available_nodes:
                    return node
        
        # Wrap around to first node
        if self._hash_ring:
            _, node_id = self._hash_ring[0]
            if node_id in self._nodes:
                node = self._nodes[node_id]
                if node in available_nodes:
                    return node
        
        return available_nodes[0]

    async def release_node(self, node: WorkerNode, response_time: float = None):
        """Release node after request completion"""
        async with self._lock:
            if node.current_connections > 0:
                node.current_connections -= 1
            
            # Update response time moving average
            if response_time is not None:
                if node.avg_response_time == 0:
                    node.avg_response_time = response_time
                else:
                    # Exponential moving average
                    alpha = 0.1
                    node.avg_response_time = alpha * response_time + (1 - alpha) * node.avg_response_time

    async def update_node_stats(self, node_id: str, cpu_usage: float, memory_usage: float):
        """Update node resource statistics"""
        async with self._lock:
            if node_id in self._nodes:
                node = self._nodes[node_id]
                node.cpu_usage = cpu_usage
                node.memory_usage = memory_usage
                node.last_heartbeat = time.time()

    async def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        async with self._lock:
            node_stats = {}
            total_connections = 0
            healthy_nodes = 0
            
            for node_id, node in self._nodes.items():
                node_stats[node_id] = {
                    'is_healthy': node.is_healthy,
                    'is_available': node.is_available,
                    'current_connections': node.current_connections,
                    'utilization': node.utilization,
                    'avg_response_time': node.avg_response_time,
                    'cpu_usage': node.cpu_usage,
                    'memory_usage': node.memory_usage
                }
                
                total_connections += node.current_connections
                if node.is_healthy:
                    healthy_nodes += 1
            
            return {
                'strategy': self.strategy.value,
                'total_nodes': len(self._nodes),
                'healthy_nodes': healthy_nodes,
                'total_connections': total_connections,
                'request_count': self._request_count,
                'avg_response_time': self._total_response_time / max(1, self._request_count),
                'node_stats': node_stats
            }

class ResourceManager:
    """
    Advanced resource management and allocation
    Handles CPU, memory, disk, and custom resource quotas
    """

    def __init__(self, project_path: Optional[Path] = None):
        self.project_path = project_path
        
        # Resource tracking
        self._resource_usage: Dict[str, Dict[ResourceType, float]] = defaultdict(lambda: defaultdict(float))
        self._resource_limits: Dict[str, ResourceQuota] = {}
        self._resource_reservations: Dict[str, Dict[ResourceType, float]] = defaultdict(lambda: defaultdict(float))
        
        # System resources
        self._system_resources = {
            ResourceType.CPU: psutil.cpu_count(),
            ResourceType.MEMORY: psutil.virtual_memory().total // (1024 * 1024),  # MB
            ResourceType.DISK_IO: 1000,  # Arbitrary units
            ResourceType.NETWORK_IO: 1000  # Arbitrary units
        }
        
        # Resource monitoring
        self._monitor_task = None
        self._resource_history = deque(maxlen=1000)
        self._lock = asyncio.Lock()
        
        # Start monitoring
        self._start_resource_monitoring()

    def _start_resource_monitoring(self):
        """Start background resource monitoring"""
        if self._monitor_task is None:
            self._monitor_task = asyncio.create_task(self._monitor_resources_loop())

    async def _monitor_resources_loop(self):
        """Background resource monitoring"""
        while True:
            try:
                await asyncio.sleep(10)  # Monitor every 10 seconds
                await self._update_system_resources()
                await self._check_resource_violations()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")

    async def _update_system_resources(self):
        """Update system resource measurements"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_usage = (cpu_percent / 100.0) * self._system_resources[ResourceType.CPU]
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = (memory.used // (1024 * 1024))  # MB
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_usage = (disk_io.read_bytes + disk_io.write_bytes) if disk_io else 0
            
            # Network I/O
            net_io = psutil.net_io_counters()
            network_usage = (net_io.bytes_sent + net_io.bytes_recv) if net_io else 0
            
            # Record snapshot
            snapshot = {
                'timestamp': time.time(),
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'disk_io': disk_usage,
                'network_io': network_usage
            }
            
            self._resource_history.append(snapshot)
            
        except Exception as e:
            logger.error(f"Failed to update system resources: {e}")

    async def _check_resource_violations(self):
        """Check for resource quota violations"""
        violations = []
        
        async with self._lock:
            for component_id, limits in self._resource_limits.items():
                usage = self._resource_usage[component_id]
                
                # Check each resource type
                if limits.cpu_cores and usage[ResourceType.CPU] > limits.cpu_cores:
                    violations.append(f"{component_id}: CPU usage {usage[ResourceType.CPU]:.1f} > limit {limits.cpu_cores}")
                
                if limits.memory_mb and usage[ResourceType.MEMORY] > limits.memory_mb:
                    violations.append(f"{component_id}: Memory usage {usage[ResourceType.MEMORY]:.1f}MB > limit {limits.memory_mb}MB")
        
        if violations:
            logger.warning(f"Resource violations detected: {violations}")

    async def allocate_resources(self, component_id: str, requirements: ResourceQuota) -> bool:
        """Allocate resources for a component"""
        async with self._lock:
            # Check if resources are available
            if not await self._can_allocate(requirements):
                return False
            
            # Reserve resources
            reservations = self._resource_reservations[component_id]
            if requirements.cpu_cores:
                reservations[ResourceType.CPU] += requirements.cpu_cores
            if requirements.memory_mb:
                reservations[ResourceType.MEMORY] += requirements.memory_mb
            
            # Set limits
            self._resource_limits[component_id] = requirements
            
            logger.info(f"Allocated resources for {component_id}: {requirements}")
            return True

    async def _can_allocate(self, requirements: ResourceQuota) -> bool:
        """Check if resources can be allocated"""
        # Calculate total reserved resources
        total_reserved = defaultdict(float)
        for reservations in self._resource_reservations.values():
            for resource_type, amount in reservations.items():
                total_reserved[resource_type] += amount
        
        # Check availability
        if requirements.cpu_cores:
            available_cpu = self._system_resources[ResourceType.CPU] - total_reserved[ResourceType.CPU]
            if requirements.cpu_cores > available_cpu:
                return False
        
        if requirements.memory_mb:
            available_memory = self._system_resources[ResourceType.MEMORY] - total_reserved[ResourceType.MEMORY]
            if requirements.memory_mb > available_memory:
                return False
        
        return True

    async def update_usage(self, component_id: str, resource_type: ResourceType, usage: float):
        """Update resource usage for a component"""
        async with self._lock:
            self._resource_usage[component_id][resource_type] = usage

    async def release_resources(self, component_id: str):
        """Release all resources for a component"""
        async with self._lock:
            if component_id in self._resource_reservations:
                del self._resource_reservations[component_id]
            if component_id in self._resource_limits:
                del self._resource_limits[component_id]
            if component_id in self._resource_usage:
                del self._resource_usage[component_id]
            
            logger.info(f"Released resources for {component_id}")

    async def get_resource_report(self) -> Dict[str, Any]:
        """Get comprehensive resource usage report"""
        async with self._lock:
            # Calculate total usage and reservations
            total_usage = defaultdict(float)
            total_reserved = defaultdict(float)
            
            for usage in self._resource_usage.values():
                for resource_type, amount in usage.items():
                    total_usage[resource_type] += amount
            
            for reservations in self._resource_reservations.values():
                for resource_type, amount in reservations.items():
                    total_reserved[resource_type] += amount
            
            # System utilization
            system_util = {}
            for resource_type, total in self._system_resources.items():
                used = total_usage.get(resource_type, 0)
                system_util[resource_type.value] = {
                    'total': total,
                    'used': used,
                    'reserved': total_reserved.get(resource_type, 0),
                    'utilization_percent': (used / max(1, total)) * 100
                }
            
            # Component details
            component_details = {}
            for component_id in set(self._resource_usage.keys()) | set(self._resource_limits.keys()):
                usage = self._resource_usage[component_id]
                limits = self._resource_limits.get(component_id)
                
                component_details[component_id] = {
                    'usage': {rt.value: amount for rt, amount in usage.items()},
                    'limits': limits.__dict__ if limits else None,
                    'violations': []
                }
                
                # Check for violations
                if limits:
                    if limits.cpu_cores and usage[ResourceType.CPU] > limits.cpu_cores:
                        component_details[component_id]['violations'].append('cpu_exceeded')
                    if limits.memory_mb and usage[ResourceType.MEMORY] > limits.memory_mb:
                        component_details[component_id]['violations'].append('memory_exceeded')
            
            return {
                'system_resources': system_util,
                'components': component_details,
                'total_components': len(component_details),
                'violated_components': len([c for c in component_details.values() if c['violations']])
            }

class AutoScaler:
    """
    Automatic scaling based on resource usage and performance metrics
    Handles both vertical (scale up/down) and horizontal (scale out/in) scaling
    """

    def __init__(
        self,
        resource_manager: ResourceManager,
        load_balancer: LoadBalancer,
        scaling_policy: ScalingPolicy
    ):
        self.resource_manager = resource_manager
        self.load_balancer = load_balancer
        self.policy = scaling_policy
        
        # Scaling state
        self.current_instances = 1
        self.last_scale_time = 0.0
        self.scaling_history = deque(maxlen=100)
        
        # Monitoring
        self._monitor_task = None
        self._metrics_window = deque(maxlen=60)  # 10 minutes of data (10s intervals)
        self._lock = asyncio.Lock()
        
        # Start monitoring
        self._start_scaling_monitoring()

    def _start_scaling_monitoring(self):
        """Start scaling monitoring"""
        if self._monitor_task is None:
            self._monitor_task = asyncio.create_task(self._scaling_monitor_loop())

    async def _scaling_monitor_loop(self):
        """Background scaling monitoring and decision making"""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                await self._collect_metrics()
                await self._evaluate_scaling_decision()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-scaling error: {e}")

    async def _collect_metrics(self):
        """Collect current metrics"""
        # Get resource report
        resource_report = await self.resource_manager.get_resource_report()
        
        # Get load balancer stats
        lb_stats = await self.load_balancer.get_stats()
        
        # Create metrics snapshot
        metrics = {
            'timestamp': time.time(),
            'cpu_utilization': resource_report['system_resources']['cpu']['utilization_percent'],
            'memory_utilization': resource_report['system_resources']['memory']['utilization_percent'],
            'avg_response_time': lb_stats['avg_response_time'],
            'active_connections': lb_stats['total_connections'],
            'healthy_nodes': lb_stats['healthy_nodes']
        }
        
        self._metrics_window.append(metrics)

    async def _evaluate_scaling_decision(self):
        """Evaluate if scaling is needed"""
        if len(self._metrics_window) < 5:  # Need at least 5 data points
            return
        
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scale_time < self.policy.cooldown_period:
            return
        
        # Calculate average metrics over last 5 minutes
        recent_metrics = [m for m in self._metrics_window if current_time - m['timestamp'] < 300]
        
        if not recent_metrics:
            return
        
        metrics_count = max(1, len(recent_metrics))
        avg_cpu = sum(m['cpu_utilization'] for m in recent_metrics) / metrics_count
        avg_memory = sum(m['memory_utilization'] for m in recent_metrics) / metrics_count
        avg_response_time = sum(m['avg_response_time'] for m in recent_metrics) / metrics_count
        
        # Scale up conditions
        should_scale_up = (
            avg_cpu > self.policy.scale_up_threshold or
            avg_memory > self.policy.scale_up_threshold or
            avg_response_time > self.policy.target_response_time_ms
        )
        
        # Scale down conditions
        should_scale_down = (
            avg_cpu < self.policy.scale_down_threshold and
            avg_memory < self.policy.scale_down_threshold and
            avg_response_time < self.policy.target_response_time_ms * 0.5
        )
        
        # Execute scaling decision
        if should_scale_up and self.current_instances < self.policy.max_instances:
            await self._scale_out()
        elif should_scale_down and self.current_instances > self.policy.min_instances:
            await self._scale_in()

    async def _scale_out(self):
        """Scale out (add instances)"""
        new_instances = min(
            self.current_instances + self.policy.scale_up_step,
            self.policy.max_instances
        )
        
        logger.info(f"Scaling out from {self.current_instances} to {new_instances} instances")
        
        # This would typically create new worker instances
        # For now, we just update the count and log
        self.current_instances = new_instances
        self.last_scale_time = time.time()
        
        # Record scaling event
        self.scaling_history.append({
            'timestamp': time.time(),
            'direction': ScalingDirection.OUT,
            'from_instances': self.current_instances - self.policy.scale_up_step,
            'to_instances': self.current_instances,
            'trigger': 'high_resource_usage'
        })

    async def _scale_in(self):
        """Scale in (remove instances)"""
        new_instances = max(
            self.current_instances - self.policy.scale_down_step,
            self.policy.min_instances
        )
        
        logger.info(f"Scaling in from {self.current_instances} to {new_instances} instances")
        
        # This would typically terminate worker instances gracefully
        self.current_instances = new_instances
        self.last_scale_time = time.time()
        
        # Record scaling event
        self.scaling_history.append({
            'timestamp': time.time(),
            'direction': ScalingDirection.IN,
            'from_instances': self.current_instances + self.policy.scale_down_step,
            'to_instances': self.current_instances,
            'trigger': 'low_resource_usage'
        })

    async def get_scaling_report(self) -> Dict[str, Any]:
        """Get auto-scaling report"""
        if not self._metrics_window:
            return {'error': 'No metrics available'}
        
        latest_metrics = self._metrics_window[-1]
        
        return {
            'current_instances': self.current_instances,
            'min_instances': self.policy.min_instances,
            'max_instances': self.policy.max_instances,
            'last_scale_time': self.last_scale_time,
            'scaling_history': list(self.scaling_history)[-10:],  # Last 10 events
            'current_metrics': latest_metrics,
            'scaling_policy': {
                'target_cpu_utilization': self.policy.target_cpu_utilization,
                'target_memory_utilization': self.policy.target_memory_utilization,
                'target_response_time_ms': self.policy.target_response_time_ms,
                'scale_up_threshold': self.policy.scale_up_threshold,
                'scale_down_threshold': self.policy.scale_down_threshold,
                'cooldown_period': self.policy.cooldown_period
            }
        }

class DistributedTaskQueue:
    """
    Distributed task queue with priority handling and load balancing
    """

    def __init__(self, load_balancer: LoadBalancer, resource_manager: ResourceManager):
        self.load_balancer = load_balancer
        self.resource_manager = resource_manager
        
        # Task queues by priority
        self._task_queues: Dict[TaskPriority, List[Task]] = {
            priority: [] for priority in TaskPriority
        }
        
        # Task tracking
        self._active_tasks: Dict[str, Task] = {}
        self._completed_tasks = deque(maxlen=1000)
        self._failed_tasks = deque(maxlen=1000)
        
        # Processing
        self._processor_task = None
        self._lock = asyncio.Lock()
        
        # Start task processor
        self._start_task_processor()

    def _start_task_processor(self):
        """Start background task processor"""
        if self._processor_task is None:
            self._processor_task = asyncio.create_task(self._process_tasks_loop())

    async def _process_tasks_loop(self):
        """Background task processing loop"""
        while True:
            try:
                await asyncio.sleep(0.1)  # Process tasks quickly
                await self._process_next_task()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Task processing error: {e}")

    async def _process_next_task(self):
        """Process the next highest priority task"""
        task = None
        
        async with self._lock:
            # Find highest priority task
            for priority in TaskPriority:
                if self._task_queues[priority]:
                    task = heapq.heappop(self._task_queues[priority])
                    break
        
        if not task:
            return
        
        # Select node for execution
        node = await self.load_balancer.select_node(task)
        
        if not node:
            # No available nodes, requeue task
            async with self._lock:
                heapq.heappush(self._task_queues[task.priority], task)
            return
        
        # Execute task
        await self._execute_task(task, node)

    async def _execute_task(self, task: Task, node: WorkerNode):
        """Execute task on selected node"""
        start_time = time.time()
        
        try:
            # Mark as active
            async with self._lock:
                self._active_tasks[task.task_id] = task
            
            # Check timeout
            timeout = task.timeout or 300.0  # 5 minute default timeout
            
            # Execute function (simulate execution for now)
            if asyncio.iscoroutinefunction(task.function):
                result = await asyncio.wait_for(
                    task.function(*task.args, **task.kwargs),
                    timeout=timeout
                )
            else:
                result = task.function(*task.args, **task.kwargs)
            
            # Mark as completed
            execution_time = time.time() - start_time
            
            async with self._lock:
                if task.task_id in self._active_tasks:
                    del self._active_tasks[task.task_id]
                
                self._completed_tasks.append({
                    'task_id': task.task_id,
                    'execution_time': execution_time,
                    'node_id': node.node_id,
                    'completed_at': time.time(),
                    'result': str(result)[:100]  # Truncated result
                })
            
            # Update node stats
            await self.load_balancer.release_node(node, execution_time)
            
            logger.debug(f"Task {task.task_id} completed in {execution_time:.2f}s on node {node.node_id}")
            
        except Exception as e:
            # Handle failure
            execution_time = time.time() - start_time
            
            async with self._lock:
                if task.task_id in self._active_tasks:
                    del self._active_tasks[task.task_id]
                
                self._failed_tasks.append({
                    'task_id': task.task_id,
                    'error': str(e),
                    'execution_time': execution_time,
                    'node_id': node.node_id,
                    'failed_at': time.time(),
                    'retries': task.retries
                })
            
            # Retry if possible
            if task.retries < task.max_retries:
                task.retries += 1
                logger.info(f"Retrying task {task.task_id} (attempt {task.retries + 1})")
                
                # Requeue with delay
                await asyncio.sleep(2 ** task.retries)  # Exponential backoff
                await self.submit_task(task)
            else:
                logger.error(f"Task {task.task_id} failed permanently after {task.retries} retries: {e}")
            
            # Release node
            await self.load_balancer.release_node(node, execution_time)

    async def submit_task(self, task: Task) -> str:
        """Submit task for execution"""
        async with self._lock:
            heapq.heappush(self._task_queues[task.priority], task)
        
        logger.debug(f"Submitted task {task.task_id} with priority {task.priority.name}")
        return task.task_id

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        async with self._lock:
            # Check active tasks
            if task_id in self._active_tasks:
                task = self._active_tasks[task_id]
                return {
                    'task_id': task_id,
                    'status': 'running',
                    'priority': task.priority.name,
                    'started_at': time.time() - (time.time() - task.created_at),
                    'retries': task.retries
                }
            
            # Check completed tasks
            for completed in self._completed_tasks:
                if completed['task_id'] == task_id:
                    return {
                        'task_id': task_id,
                        'status': 'completed',
                        'completed_at': completed['completed_at'],
                        'execution_time': completed['execution_time'],
                        'node_id': completed['node_id']
                    }
            
            # Check failed tasks
            for failed in self._failed_tasks:
                if failed['task_id'] == task_id:
                    return {
                        'task_id': task_id,
                        'status': 'failed',
                        'failed_at': failed['failed_at'],
                        'error': failed['error'],
                        'retries': failed['retries']
                    }
        
        return None

    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        async with self._lock:
            queue_lengths = {
                priority.name: len(tasks) 
                for priority, tasks in self._task_queues.items()
            }
            
            return {
                'queue_lengths': queue_lengths,
                'total_queued': sum(queue_lengths.values()),
                'active_tasks': len(self._active_tasks),
                'completed_tasks': len(self._completed_tasks),
                'failed_tasks': len(self._failed_tasks),
                'success_rate': len(self._completed_tasks) / max(1, len(self._completed_tasks) + len(self._failed_tasks))
            }

class ScalabilityManager:
    """
    Main scalability coordinator
    Integrates load balancing, resource management, and auto-scaling
    """

    def __init__(self, project_path: Optional[Path] = None):
        self.project_path = project_path
        
        # Initialize components
        self.resource_manager = ResourceManager(project_path)
        self.load_balancer = LoadBalancer()
        
        # Auto-scaling configuration
        scaling_policy = ScalingPolicy(
            min_instances=1,
            max_instances=10,
            target_cpu_utilization=70.0,
            scale_up_threshold=80.0,
            scale_down_threshold=30.0
        )
        
        self.auto_scaler = AutoScaler(
            self.resource_manager,
            self.load_balancer,
            scaling_policy
        )
        
        # Task queue
        self.task_queue = DistributedTaskQueue(
            self.load_balancer,
            self.resource_manager
        )
        
        # Performance tracking
        self._performance_metrics = deque(maxlen=1000)
        self._start_time = time.time()

    async def add_worker_node(
        self,
        host: str,
        port: int,
        weight: float = 1.0,
        capabilities: List[str] = None
    ) -> str:
        """Add a new worker node"""
        node_id = f"{host}:{port}"
        
        node = WorkerNode(
            node_id=node_id,
            host=host,
            port=port,
            weight=weight,
            capabilities=capabilities or []
        )
        
        await self.load_balancer.add_node(node)
        return node_id

    async def submit_task(
        self,
        function: Callable,
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[float] = None,
        resource_requirements: Optional[ResourceQuota] = None,
        **kwargs
    ) -> str:
        """Submit a task for distributed execution"""
        task = Task(
            function=function,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout,
            resource_requirements=resource_requirements
        )
        
        return await self.task_queue.submit_task(task)

    async def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive scalability report"""
        # Collect reports from all components
        resource_report = await self.resource_manager.get_resource_report()
        load_balancer_stats = await self.load_balancer.get_stats()
        scaling_report = await self.auto_scaler.get_scaling_report()
        queue_stats = await self.task_queue.get_queue_stats()
        
        # Calculate uptime
        uptime = time.time() - self._start_time
        
        return {
            'uptime_seconds': uptime,
            'resource_management': resource_report,
            'load_balancing': load_balancer_stats,
            'auto_scaling': scaling_report,
            'task_queue': queue_stats,
            'recommendations': self._generate_scalability_recommendations(
                resource_report, load_balancer_stats, scaling_report
            )
        }

    def _generate_scalability_recommendations(
        self,
        resource_report: Dict[str, Any],
        lb_stats: Dict[str, Any],
        scaling_report: Dict[str, Any]
    ) -> List[str]:
        """Generate scalability recommendations"""
        recommendations = []
        
        # Resource recommendations
        cpu_util = resource_report['system_resources']['cpu']['utilization_percent']
        memory_util = resource_report['system_resources']['memory']['utilization_percent']
        
        if cpu_util > 80:
            recommendations.append("High CPU utilization. Consider adding more worker nodes or upgrading CPU.")
        
        if memory_util > 85:
            recommendations.append("High memory utilization. Consider increasing memory or optimizing memory usage.")
        
        # Load balancing recommendations
        if lb_stats['healthy_nodes'] < 2:
            recommendations.append("Consider adding redundant worker nodes for high availability.")
        
        if lb_stats['avg_response_time'] > 2000:  # 2 seconds
            recommendations.append("High response times detected. Consider optimizing code or scaling out.")
        
        # Scaling recommendations
        if scaling_report.get('current_instances', 0) == scaling_report.get('max_instances', 1):
            recommendations.append("At maximum instance limit. Consider increasing max_instances or optimizing resource usage.")
        
        return recommendations

    async def cleanup(self):
        """Cleanup all components"""
        if self.resource_manager._monitor_task:
            self.resource_manager._monitor_task.cancel()
        if self.load_balancer._health_check_task:
            self.load_balancer._health_check_task.cancel()
        if self.auto_scaler._monitor_task:
            self.auto_scaler._monitor_task.cancel()
        if self.task_queue._processor_task:
            self.task_queue._processor_task.cancel()