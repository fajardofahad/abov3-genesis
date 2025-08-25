"""
ABOV3 Genesis - Performance Optimization Infrastructure
Enterprise-grade performance optimizations for production deployment
"""

import asyncio
import time
import threading
import weakref
from typing import Dict, Any, Optional, List, Callable, TypeVar, Generic
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
import logging
import hashlib
import pickle
import json
from pathlib import Path
import aiohttp
import aiofiles
from collections import defaultdict, OrderedDict
import psutil
import gc

logger = logging.getLogger(__name__)

T = TypeVar('T')

class CacheStrategy(Enum):
    """Cache eviction strategies"""
    LRU = "lru"          # Least Recently Used
    LFU = "lfu"          # Least Frequently Used  
    TTL = "ttl"          # Time To Live
    FIFO = "fifo"        # First In First Out

class PerformanceLevel(Enum):
    """Performance optimization levels"""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    ENTERPRISE = "enterprise"

@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with metadata"""
    value: T
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[float] = None
    size: int = 0

    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    def touch(self):
        """Update access statistics"""
        self.last_accessed = time.time()
        self.access_count += 1

class CacheManager:
    """
    High-performance caching system with multiple eviction strategies
    Optimized for AI model responses and project data
    """

    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: int = 512,
        strategy: CacheStrategy = CacheStrategy.LRU,
        default_ttl: Optional[float] = None,
        persistence_path: Optional[Path] = None
    ):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.strategy = strategy
        self.default_ttl = default_ttl
        self.persistence_path = persistence_path
        
        self._cache: Dict[str, CacheEntry] = OrderedDict()
        self._memory_usage = 0
        self._lock = asyncio.Lock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
        
        # Background cleanup task
        self._cleanup_task = None
        self._cleanup_interval = 60  # seconds
        
        # Initialize background cleanup
        asyncio.create_task(self._start_cleanup_task())

    async def _start_cleanup_task(self):
        """Start background cleanup task"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_expired()
                await self._check_memory_usage()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        async with self._lock:
            self._stats['total_requests'] += 1
            
            if key not in self._cache:
                self._stats['misses'] += 1
                return None
                
            entry = self._cache[key]
            
            # Check if expired
            if entry.is_expired():
                del self._cache[key]
                self._memory_usage -= entry.size
                self._stats['misses'] += 1
                return None
            
            # Update access stats
            entry.touch()
            self._stats['hits'] += 1
            
            # Move to end for LRU
            if self.strategy == CacheStrategy.LRU:
                self._cache.move_to_end(key)
            
            return entry.value

    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache"""
        async with self._lock:
            # Calculate size
            try:
                size = len(pickle.dumps(value))
            except:
                size = len(str(value).encode('utf-8'))
            
            # Check if we need to evict
            while (len(self._cache) >= self.max_size or 
                   self._memory_usage + size > self.max_memory_bytes):
                if not await self._evict_one():
                    # Can't evict more, reject this entry
                    return False
            
            # Create cache entry
            entry = CacheEntry(
                value=value,
                ttl=ttl or self.default_ttl,
                size=size
            )
            
            # Remove existing entry if present
            if key in self._cache:
                self._memory_usage -= self._cache[key].size
            
            # Add new entry
            self._cache[key] = entry
            self._memory_usage += size
            
            return True

    async def _evict_one(self) -> bool:
        """Evict one entry based on strategy"""
        if not self._cache:
            return False
            
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used (first in OrderedDict)
            key = next(iter(self._cache))
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            key = min(self._cache.keys(), key=lambda k: self._cache[k].access_count)
        elif self.strategy == CacheStrategy.FIFO:
            # Remove first in (first in OrderedDict)
            key = next(iter(self._cache))
        else:  # TTL or default
            # Remove oldest entry
            key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
        
        # Remove entry
        entry = self._cache.pop(key)
        self._memory_usage -= entry.size
        self._stats['evictions'] += 1
        
        return True

    async def _cleanup_expired(self):
        """Remove expired entries"""
        expired_keys = []
        
        for key, entry in self._cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            entry = self._cache.pop(key)
            self._memory_usage -= entry.size

    async def _check_memory_usage(self):
        """Check and optimize memory usage"""
        if self._memory_usage > self.max_memory_bytes * 0.8:
            # Evict 10% of entries when approaching limit
            evict_count = max(1, len(self._cache) // 10)
            for _ in range(evict_count):
                if not await self._evict_one():
                    break

    async def clear(self):
        """Clear all cache entries"""
        async with self._lock:
            self._cache.clear()
            self._memory_usage = 0

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        async with self._lock:
            hit_rate = 0
            if self._stats['total_requests'] > 0:
                hit_rate = self._stats['hits'] / self._stats['total_requests']
            
            return {
                'size': len(self._cache),
                'memory_usage_mb': self._memory_usage / (1024 * 1024),
                'hit_rate': hit_rate,
                'stats': self._stats.copy(),
                'strategy': self.strategy.value
            }

    async def save_to_disk(self):
        """Save cache to disk for persistence"""
        if not self.persistence_path:
            return
            
        try:
            cache_data = {
                'entries': {},
                'stats': self._stats,
                'timestamp': time.time()
            }
            
            # Only save non-expired entries
            for key, entry in self._cache.items():
                if not entry.is_expired():
                    cache_data['entries'][key] = {
                        'value': entry.value,
                        'created_at': entry.created_at,
                        'ttl': entry.ttl
                    }
            
            async with aiofiles.open(self.persistence_path, 'wb') as f:
                await f.write(pickle.dumps(cache_data))
                
        except Exception as e:
            logger.error(f"Failed to save cache to disk: {e}")

    async def load_from_disk(self):
        """Load cache from disk"""
        if not self.persistence_path or not self.persistence_path.exists():
            return
            
        try:
            async with aiofiles.open(self.persistence_path, 'rb') as f:
                data = pickle.loads(await f.read())
            
            # Restore entries that haven't expired
            current_time = time.time()
            for key, entry_data in data.get('entries', {}).items():
                created_at = entry_data.get('created_at', current_time)
                ttl = entry_data.get('ttl')
                
                # Check if still valid
                if ttl is None or current_time - created_at < ttl:
                    await self.set(key, entry_data['value'], ttl)
                    
        except Exception as e:
            logger.error(f"Failed to load cache from disk: {e}")


class ConnectionPool:
    """
    High-performance connection pool for HTTP requests and database connections
    Optimized for Ollama API calls and external services
    """

    def __init__(
        self,
        max_connections: int = 100,
        max_connections_per_host: int = 30,
        keepalive_timeout: int = 30,
        connect_timeout: int = 10,
        read_timeout: int = 30,
        enable_compression: bool = True
    ):
        self.max_connections = max_connections
        self.max_connections_per_host = max_connections_per_host
        self.keepalive_timeout = keepalive_timeout
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        
        # Connection pool configuration
        connector = aiohttp.TCPConnector(
            limit=max_connections,
            limit_per_host=max_connections_per_host,
            keepalive_timeout=keepalive_timeout,
            enable_cleanup_closed=True,
            use_dns_cache=True,
            ttl_dns_cache=300,  # 5 minutes DNS cache
            family=0  # Allow both IPv4 and IPv6
        )
        
        # Timeout configuration
        timeout = aiohttp.ClientTimeout(
            total=None,  # No total timeout
            connect=connect_timeout,
            sock_read=read_timeout
        )
        
        # HTTP session with optimizations
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'ABOV3-Genesis/1.0.0',
                'Connection': 'keep-alive'
            }
        )
        
        self._connection_stats = {
            'total_requests': 0,
            'active_connections': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0
        }
        
        self._response_times = []
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> aiohttp.ClientResponse:
        """Make HTTP request with connection pooling"""
        start_time = time.time()
        
        try:
            async with self._lock:
                self._connection_stats['total_requests'] += 1
                self._connection_stats['active_connections'] += 1
            
            # Make request using the pooled session
            async with self._session.request(method, url, **kwargs) as response:
                # Record response time
                response_time = time.time() - start_time
                self._response_times.append(response_time)
                
                # Keep only last 1000 response times for average calculation
                if len(self._response_times) > 1000:
                    self._response_times = self._response_times[-1000:]
                
                # Update statistics
                async with self._lock:
                    self._connection_stats['active_connections'] -= 1
                    self._connection_stats['avg_response_time'] = sum(self._response_times) / len(self._response_times)
                
                return response
                
        except Exception as e:
            async with self._lock:
                self._connection_stats['failed_requests'] += 1
                self._connection_stats['active_connections'] -= 1
            raise e

    async def get(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """HTTP GET request"""
        return await self.request('GET', url, **kwargs)

    async def post(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """HTTP POST request"""
        return await self.request('POST', url, **kwargs)

    async def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        async with self._lock:
            connector_info = self._session.connector._conns_per_host if hasattr(self._session.connector, '_conns_per_host') else {}
            
            return {
                'total_requests': self._connection_stats['total_requests'],
                'active_connections': self._connection_stats['active_connections'],
                'failed_requests': self._connection_stats['failed_requests'],
                'avg_response_time': self._connection_stats['avg_response_time'],
                'success_rate': 1.0 - (self._connection_stats['failed_requests'] / max(1, self._connection_stats['total_requests'])),
                'pool_size': len(connector_info),
                'max_connections': self.max_connections,
                'max_connections_per_host': self.max_connections_per_host
            }

    async def close(self):
        """Close connection pool"""
        if self._session and not self._session.closed:
            await self._session.close()


class PerformanceOptimizer:
    """
    Main performance optimization controller
    Coordinates caching, connection pooling, and resource management
    """

    def __init__(
        self,
        performance_level: PerformanceLevel = PerformanceLevel.PRODUCTION,
        project_path: Optional[Path] = None
    ):
        self.performance_level = performance_level
        self.project_path = project_path
        
        # Initialize components based on performance level
        self._init_for_performance_level()
        
        # Performance monitoring
        self._performance_metrics = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'disk_io': 0.0,
            'network_io': 0.0,
            'active_tasks': 0
        }
        
        # Background monitoring task
        self._monitor_task = None
        
        # Thread and process pools for CPU-intensive tasks
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
        self._process_pool = ProcessPoolExecutor(max_workers=2)
        
        # Start monitoring
        asyncio.create_task(self._start_monitoring())

    def _init_for_performance_level(self):
        """Initialize components based on performance level"""
        cache_configs = {
            PerformanceLevel.DEVELOPMENT: {
                'max_size': 100,
                'max_memory_mb': 128,
                'strategy': CacheStrategy.LRU,
                'default_ttl': 300  # 5 minutes
            },
            PerformanceLevel.PRODUCTION: {
                'max_size': 1000,
                'max_memory_mb': 512,
                'strategy': CacheStrategy.LRU,
                'default_ttl': 1800  # 30 minutes
            },
            PerformanceLevel.ENTERPRISE: {
                'max_size': 10000,
                'max_memory_mb': 2048,
                'strategy': CacheStrategy.LRU,
                'default_ttl': 3600  # 1 hour
            }
        }
        
        pool_configs = {
            PerformanceLevel.DEVELOPMENT: {
                'max_connections': 20,
                'max_connections_per_host': 10
            },
            PerformanceLevel.PRODUCTION: {
                'max_connections': 100,
                'max_connections_per_host': 30
            },
            PerformanceLevel.ENTERPRISE: {
                'max_connections': 500,
                'max_connections_per_host': 100
            }
        }
        
        # Initialize cache
        cache_config = cache_configs[self.performance_level]
        persistence_path = None
        if self.project_path:
            persistence_path = self.project_path / '.abov3' / 'cache' / 'performance.cache'
            persistence_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.cache = CacheManager(
            persistence_path=persistence_path,
            **cache_config
        )
        
        # Initialize connection pool
        pool_config = pool_configs[self.performance_level]
        self.connection_pool = ConnectionPool(**pool_config)

    async def _start_monitoring(self):
        """Start performance monitoring"""
        if self._monitor_task is None:
            self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def _monitor_loop(self):
        """Background performance monitoring"""
        while True:
            try:
                await asyncio.sleep(10)  # Monitor every 10 seconds
                await self._update_metrics()
                await self._optimize_if_needed()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")

    async def _update_metrics(self):
        """Update performance metrics"""
        try:
            # System metrics
            self._performance_metrics['cpu_usage'] = psutil.cpu_percent(interval=1)
            self._performance_metrics['memory_usage'] = psutil.virtual_memory().percent
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                self._performance_metrics['disk_io'] = disk_io.read_bytes + disk_io.write_bytes
            
            # Network I/O
            net_io = psutil.net_io_counters()
            if net_io:
                self._performance_metrics['network_io'] = net_io.bytes_sent + net_io.bytes_recv
            
            # Active tasks count
            current_task = asyncio.current_task()
            all_tasks = asyncio.all_tasks(asyncio.get_event_loop())
            self._performance_metrics['active_tasks'] = len([t for t in all_tasks if not t.done()])
            
        except Exception as e:
            logger.error(f"Metrics update error: {e}")

    async def _optimize_if_needed(self):
        """Perform optimization if needed"""
        # Memory optimization
        if self._performance_metrics['memory_usage'] > 80:
            await self._optimize_memory()
        
        # CPU optimization
        if self._performance_metrics['cpu_usage'] > 90:
            await self._optimize_cpu()

    async def _optimize_memory(self):
        """Optimize memory usage"""
        try:
            # Trigger garbage collection
            gc.collect()
            
            # Clear cache if memory is very high
            if self._performance_metrics['memory_usage'] > 90:
                cache_stats = await self.cache.get_stats()
                if cache_stats['memory_usage_mb'] > 100:
                    # Clear 50% of cache entries
                    cache_size = cache_stats['size']
                    for _ in range(cache_size // 2):
                        if not await self.cache._evict_one():
                            break
                    
                    logger.info(f"Emergency cache cleanup: freed memory due to high usage")
                    
        except Exception as e:
            logger.error(f"Memory optimization error: {e}")

    async def _optimize_cpu(self):
        """Optimize CPU usage"""
        try:
            # Reduce thread pool size temporarily
            current_workers = self._thread_pool._max_workers
            if current_workers > 2:
                # Create new pool with fewer workers
                old_pool = self._thread_pool
                self._thread_pool = ThreadPoolExecutor(max_workers=max(2, current_workers // 2))
                
                # Schedule old pool shutdown
                def shutdown_old_pool():
                    old_pool.shutdown(wait=False)
                
                asyncio.get_event_loop().call_later(60, shutdown_old_pool)
                
                logger.info(f"Reduced thread pool size due to high CPU usage: {current_workers} -> {self._thread_pool._max_workers}")
                
        except Exception as e:
            logger.error(f"CPU optimization error: {e}")

    async def cache_response(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None
    ) -> bool:
        """Cache AI response or computation result"""
        return await self.cache.set(key, value, ttl)

    async def get_cached_response(self, key: str) -> Optional[Any]:
        """Get cached AI response or computation result"""
        return await self.cache.get(key)

    def create_cache_key(self, *args, **kwargs) -> str:
        """Create cache key from arguments"""
        # Create deterministic hash from arguments
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else {}
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]

    async def execute_in_thread(self, func: Callable, *args, **kwargs):
        """Execute CPU-intensive function in thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._thread_pool, func, *args, **kwargs)

    async def execute_in_process(self, func: Callable, *args, **kwargs):
        """Execute CPU-intensive function in process pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._process_pool, func, *args, **kwargs)

    async def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        cache_stats = await self.cache.get_stats()
        pool_stats = await self.connection_pool.get_stats()
        
        return {
            'performance_level': self.performance_level.value,
            'system_metrics': self._performance_metrics.copy(),
            'cache': cache_stats,
            'connection_pool': pool_stats,
            'thread_pool': {
                'max_workers': self._thread_pool._max_workers,
                'active_threads': self._thread_pool._threads
            },
            'recommendations': self._generate_recommendations()
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        # Memory recommendations
        if self._performance_metrics['memory_usage'] > 75:
            recommendations.append("High memory usage detected. Consider increasing system RAM or reducing cache size.")
        
        # CPU recommendations
        if self._performance_metrics['cpu_usage'] > 80:
            recommendations.append("High CPU usage detected. Consider scaling to multiple instances or upgrading hardware.")
        
        # Cache recommendations
        if hasattr(self, '_last_cache_stats'):
            hit_rate = self._last_cache_stats.get('hit_rate', 0)
            if hit_rate < 0.5:
                recommendations.append("Low cache hit rate. Consider adjusting cache strategy or increasing cache size.")
        
        return recommendations

    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Cancel monitoring task
            if self._monitor_task:
                self._monitor_task.cancel()
                try:
                    await self._monitor_task
                except asyncio.CancelledError:
                    pass
            
            # Save cache to disk
            await self.cache.save_to_disk()
            
            # Close connection pool
            await self.connection_pool.close()
            
            # Shutdown thread pools
            self._thread_pool.shutdown(wait=False)
            self._process_pool.shutdown(wait=False)
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# Decorator for caching function results
def cache_result(ttl: Optional[float] = None, key_prefix: str = ""):
    """Decorator to cache function results"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            # This will be injected by the performance optimizer
            optimizer = getattr(asyncio.current_task(), '_performance_optimizer', None)
            
            if optimizer:
                cache_key = key_prefix + optimizer.create_cache_key(func.__name__, *args, **kwargs)
                
                # Try to get from cache
                cached_result = await optimizer.get_cached_response(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = await func(*args, **kwargs)
                await optimizer.cache_response(cache_key, result, ttl)
                return result
            else:
                # No optimizer available, execute directly
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            # For synchronous functions, can't easily cache without async context
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Context manager for performance optimization
class performance_context:
    """Context manager to inject performance optimization"""
    
    def __init__(self, optimizer: PerformanceOptimizer):
        self.optimizer = optimizer
        self.previous_optimizer = None

    async def __aenter__(self):
        # Inject optimizer into current task
        current_task = asyncio.current_task()
        if current_task:
            self.previous_optimizer = getattr(current_task, '_performance_optimizer', None)
            current_task._performance_optimizer = self.optimizer
        return self.optimizer

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Restore previous optimizer
        current_task = asyncio.current_task()
        if current_task:
            if self.previous_optimizer:
                current_task._performance_optimizer = self.previous_optimizer
            else:
                delattr(current_task, '_performance_optimizer')