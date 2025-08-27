"""
ABOV3 Genesis - Performance Optimizer
Optimizes context-aware comprehension for large codebases (1M+ lines)
"""

import asyncio
import logging
import time
import psutil
import gc
from typing import Dict, List, Any, Optional, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count
import threading
import queue
import weakref
from collections import deque, OrderedDict
import sys

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""
    memory_usage_mb: float
    cpu_usage_percent: float
    processing_time: float
    files_processed: int
    cache_hit_rate: float
    throughput_files_per_second: float
    peak_memory_mb: float
    gc_collections: int

class MemoryManager:
    """Manages memory usage for large codebase processing"""
    
    def __init__(self, max_memory_mb: int = 4096):
        self.max_memory_mb = max_memory_mb
        self.current_memory_mb = 0
        self.cache_registry = weakref.WeakSet()
        self._lock = threading.Lock()
    
    def check_memory_usage(self) -> float:
        """Check current memory usage"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        self.current_memory_mb = memory_mb
        return memory_mb
    
    def should_clear_cache(self) -> bool:
        """Check if cache should be cleared to free memory"""
        return self.check_memory_usage() > self.max_memory_mb * 0.8
    
    def force_garbage_collection(self):
        """Force garbage collection"""
        gc.collect()
        
    def clear_caches(self):
        """Clear all registered caches"""
        with self._lock:
            for cache_obj in self.cache_registry:
                if hasattr(cache_obj, 'clear'):
                    cache_obj.clear()
            self.force_garbage_collection()

class BatchProcessor:
    """Processes files in optimized batches"""
    
    def __init__(self, batch_size: int = 100, max_workers: int = None):
        self.batch_size = batch_size
        self.max_workers = max_workers or min(32, cpu_count() * 2)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=min(cpu_count(), 8))
        
    async def process_files_batch(
        self,
        files: List[Path],
        processor_func: Callable,
        use_multiprocessing: bool = False
    ) -> List[Any]:
        """Process files in optimized batches"""
        results = []
        
        # Split files into batches
        batches = [files[i:i + self.batch_size] for i in range(0, len(files), self.batch_size)]
        
        executor = self.process_pool if use_multiprocessing else self.thread_pool
        
        for batch in batches:
            # Process batch
            loop = asyncio.get_event_loop()
            batch_tasks = [
                loop.run_in_executor(executor, processor_func, file_path)
                for file_path in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Filter out exceptions and collect results
            for result in batch_results:
                if not isinstance(result, Exception):
                    results.append(result)
            
            # Memory management between batches
            if len(results) % (self.batch_size * 5) == 0:  # Every 5 batches
                gc.collect()
        
        return results
    
    def __del__(self):
        """Clean up executors"""
        try:
            self.thread_pool.shutdown(wait=False)
            self.process_pool.shutdown(wait=False)
        except:
            pass

class HierarchicalIndexer:
    """Hierarchical indexing for large monorepos"""
    
    def __init__(self):
        self.directory_summaries = {}
        self.file_priorities = {}
        self.hot_files = set()  # Frequently accessed files
        
    async def build_hierarchy(self, root_path: Path) -> Dict[str, Any]:
        """Build hierarchical view of codebase"""
        hierarchy = {}
        
        # Walk directory tree and create hierarchy
        for root, dirs, files in root_path.rglob('*'):
            # Skip common ignore directories early
            dirs[:] = [d for d in dirs if not self._should_ignore_directory(d)]
            
            relative_root = root.relative_to(root_path)
            current_level = hierarchy
            
            # Build nested structure
            for part in relative_root.parts:
                if part not in current_level:
                    current_level[part] = {'dirs': {}, 'files': [], 'summary': {}}
                current_level = current_level[part]['dirs']
            
            # Add files to current level
            code_files = [f for f in files if self._is_code_file(Path(f))]
            if relative_root.parts:
                # Navigate to the correct level for adding files
                target_level = hierarchy
                for part in relative_root.parts[:-1]:
                    target_level = target_level[part]['dirs']
                target_level[relative_root.parts[-1]]['files'] = code_files
            else:
                hierarchy['files'] = code_files
        
        # Calculate summaries
        await self._calculate_directory_summaries(hierarchy, root_path)
        
        return hierarchy
    
    def _should_ignore_directory(self, dirname: str) -> bool:
        """Check if directory should be ignored"""
        ignore_dirs = {
            '.git', '__pycache__', 'node_modules', '.venv', 'venv',
            'build', 'dist', '.next', '.nuxt', 'target', '.idea', '.vscode'
        }
        return dirname in ignore_dirs or dirname.startswith('.')
    
    def _is_code_file(self, file_path: Path) -> bool:
        """Check if file is a code file"""
        code_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h',
            '.cs', '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.scala'
        }
        return file_path.suffix.lower() in code_extensions
    
    async def _calculate_directory_summaries(self, hierarchy: Dict, root_path: Path):
        """Calculate summaries for each directory"""
        def calculate_summary(node, path_parts):
            file_count = len(node.get('files', []))
            total_dirs = len(node.get('dirs', {}))
            
            # Recursively calculate for subdirectories
            total_files = file_count
            total_subdirs = total_dirs
            
            for subdir_name, subdir_node in node.get('dirs', {}).items():
                sub_summary = calculate_summary(subdir_node, path_parts + [subdir_name])
                total_files += sub_summary['total_files']
                total_subdirs += sub_summary['total_dirs']
            
            summary = {
                'direct_files': file_count,
                'total_files': total_files,
                'direct_dirs': total_dirs,
                'total_dirs': total_subdirs,
                'depth': len(path_parts),
                'priority': self._calculate_directory_priority(file_count, total_files)
            }
            
            node['summary'] = summary
            return summary
        
        calculate_summary(hierarchy, [])
    
    def _calculate_directory_priority(self, direct_files: int, total_files: int) -> float:
        """Calculate priority score for a directory"""
        # Higher priority for directories with more files
        file_score = min(1.0, total_files / 100)
        
        # Higher priority for directories with balanced distribution
        balance_score = 0.5
        if total_files > 0:
            balance_score = min(1.0, direct_files / total_files * 2)
        
        return (file_score + balance_score) / 2
    
    def get_priority_directories(self, hierarchy: Dict, top_k: int = 20) -> List[Tuple[str, float]]:
        """Get top priority directories for processing"""
        priorities = []
        
        def collect_priorities(node, path):
            if 'summary' in node:
                priority = node['summary']['priority']
                priorities.append((path, priority))
            
            for subdir_name, subdir_node in node.get('dirs', {}).items():
                subpath = f"{path}/{subdir_name}" if path else subdir_name
                collect_priorities(subdir_node, subpath)
        
        collect_priorities(hierarchy, "")
        
        # Sort by priority and return top k
        priorities.sort(key=lambda x: x[1], reverse=True)
        return priorities[:top_k]

class StreamingProcessor:
    """Processes files in streaming fashion for memory efficiency"""
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.processing_queue = queue.Queue(maxsize=buffer_size)
        self.result_queue = queue.Queue(maxsize=buffer_size)
        self.active_processors = 0
        
    async def stream_process_files(
        self,
        files: List[Path],
        processor_func: Callable,
        result_callback: Callable[[Any], None]
    ):
        """Stream process files with bounded memory usage"""
        
        # Start background processor threads
        num_processors = min(4, cpu_count())
        processor_threads = []
        
        for _ in range(num_processors):
            thread = threading.Thread(
                target=self._processor_worker,
                args=(processor_func,),
                daemon=True
            )
            thread.start()
            processor_threads.append(thread)
        
        # Start result handler thread
        result_thread = threading.Thread(
            target=self._result_worker,
            args=(result_callback,),
            daemon=True
        )
        result_thread.start()
        
        # Queue files for processing
        for file_path in files:
            self.processing_queue.put(file_path)
        
        # Signal end of input
        for _ in range(num_processors):
            self.processing_queue.put(None)
        
        # Wait for processing to complete
        for thread in processor_threads:
            thread.join()
        
        # Signal end of results
        self.result_queue.put(None)
        result_thread.join()
    
    def _processor_worker(self, processor_func: Callable):
        """Worker thread for processing files"""
        while True:
            file_path = self.processing_queue.get()
            if file_path is None:
                break
            
            try:
                result = processor_func(file_path)
                self.result_queue.put(('success', result))
            except Exception as e:
                self.result_queue.put(('error', str(e)))
            finally:
                self.processing_queue.task_done()
    
    def _result_worker(self, result_callback: Callable):
        """Worker thread for handling results"""
        while True:
            result = self.result_queue.get()
            if result is None:
                break
            
            status, data = result
            if status == 'success':
                result_callback(data)
            # Ignore errors for now, could add error handling
            
            self.result_queue.task_done()

class CacheManager:
    """Advanced caching with size limits and eviction policies"""
    
    def __init__(self, max_size: int = 10000, max_memory_mb: int = 1024):
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.cache = OrderedDict()
        self.access_counts = defaultdict(int)
        self.cache_sizes = {}
        self.total_memory_mb = 0
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with LRU update"""
        with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.access_counts[key] += 1
                return value
            return None
    
    def put(self, key: str, value: Any, size_mb: float = 0.1):
        """Put item in cache with eviction if needed"""
        with self._lock:
            # Remove existing if present
            if key in self.cache:
                self.total_memory_mb -= self.cache_sizes.get(key, 0)
                del self.cache[key]
            
            # Evict if necessary
            while (len(self.cache) >= self.max_size or 
                   self.total_memory_mb + size_mb > self.max_memory_mb):
                if not self.cache:
                    break
                self._evict_least_valuable()
            
            # Add new item
            self.cache[key] = value
            self.cache_sizes[key] = size_mb
            self.total_memory_mb += size_mb
    
    def _evict_least_valuable(self):
        """Evict least valuable item (LRU + access count hybrid)"""
        if not self.cache:
            return
        
        # Find least valuable item
        min_score = float('inf')
        min_key = None
        
        for i, key in enumerate(self.cache.keys()):
            # Score based on recency (lower index = older) and access count
            recency_score = i / len(self.cache)  # 0 to 1
            access_score = min(1.0, self.access_counts[key] / 10)  # 0 to 1
            combined_score = recency_score + access_score
            
            if combined_score < min_score:
                min_score = combined_score
                min_key = key
        
        # Remove least valuable
        if min_key:
            self.total_memory_mb -= self.cache_sizes.get(min_key, 0)
            del self.cache[min_key]
            del self.cache_sizes[min_key]
            del self.access_counts[min_key]
    
    def clear(self):
        """Clear all cache"""
        with self._lock:
            self.cache.clear()
            self.cache_sizes.clear()
            self.access_counts.clear()
            self.total_memory_mb = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'memory_mb': self.total_memory_mb,
                'max_memory_mb': self.max_memory_mb,
                'utilization': len(self.cache) / self.max_size,
                'memory_utilization': self.total_memory_mb / self.max_memory_mb
            }

class PerformanceOptimizer:
    """
    Main performance optimizer for large codebase processing
    Coordinates all performance optimization strategies
    """
    
    def __init__(self, max_memory_mb: int = 4096, batch_size: int = 100):
        self.max_memory_mb = max_memory_mb
        self.batch_size = batch_size
        
        # Initialize components
        self.memory_manager = MemoryManager(max_memory_mb)
        self.batch_processor = BatchProcessor(batch_size)
        self.hierarchical_indexer = HierarchicalIndexer()
        self.streaming_processor = StreamingProcessor()
        self.cache_manager = CacheManager(max_memory_mb=max_memory_mb // 4)
        
        # Performance tracking
        self.metrics = PerformanceMetrics(
            memory_usage_mb=0,
            cpu_usage_percent=0,
            processing_time=0,
            files_processed=0,
            cache_hit_rate=0,
            throughput_files_per_second=0,
            peak_memory_mb=0,
            gc_collections=0
        )
        
        self.start_time = time.time()
        
        logger.info(f"PerformanceOptimizer initialized with {max_memory_mb}MB memory limit")
    
    async def optimize_file_processing(
        self,
        files: List[Path],
        processor_func: Callable,
        use_hierarchy: bool = True,
        use_streaming: bool = False
    ) -> List[Any]:
        """Optimize file processing for large numbers of files"""
        start_time = time.time()
        
        logger.info(f"Starting optimized processing of {len(files)} files")
        
        # Build hierarchy if requested
        if use_hierarchy and files:
            hierarchy = await self.hierarchical_indexer.build_hierarchy(files[0].parent)
            priority_dirs = self.hierarchical_indexer.get_priority_directories(hierarchy)
            
            # Reorder files based on directory priorities
            files = self._reorder_files_by_priority(files, priority_dirs)
        
        results = []
        
        if use_streaming:
            # Use streaming processing for very large file sets
            result_collector = []
            
            def collect_result(result):
                result_collector.append(result)
                
                # Periodic memory check
                if len(result_collector) % 1000 == 0:
                    if self.memory_manager.should_clear_cache():
                        self.cache_manager.clear()
                        self.memory_manager.force_garbage_collection()
            
            await self.streaming_processor.stream_process_files(
                files, processor_func, collect_result
            )
            results = result_collector
        else:
            # Use batch processing
            results = await self.batch_processor.process_files_batch(
                files, processor_func, use_multiprocessing=len(files) > 1000
            )
        
        # Update metrics
        processing_time = time.time() - start_time
        self.metrics.processing_time = processing_time
        self.metrics.files_processed = len(results)
        self.metrics.throughput_files_per_second = len(results) / max(processing_time, 0.001)
        self.metrics.memory_usage_mb = self.memory_manager.check_memory_usage()
        self.metrics.peak_memory_mb = max(self.metrics.peak_memory_mb, self.metrics.memory_usage_mb)
        
        logger.info(f"Processed {len(results)} files in {processing_time:.2f}s "
                   f"({self.metrics.throughput_files_per_second:.1f} files/sec)")
        
        return results
    
    def _reorder_files_by_priority(
        self, 
        files: List[Path], 
        priority_dirs: List[Tuple[str, float]]
    ) -> List[Path]:
        """Reorder files based on directory priorities"""
        # Create priority mapping
        dir_priorities = {path: priority for path, priority in priority_dirs}
        
        def get_file_priority(file_path: Path) -> float:
            # Find the most specific directory priority
            path_str = str(file_path.parent)
            best_priority = 0.0
            
            for dir_path, priority in dir_priorities.items():
                if path_str.startswith(dir_path):
                    best_priority = max(best_priority, priority)
            
            return best_priority
        
        # Sort files by priority (descending)
        return sorted(files, key=get_file_priority, reverse=True)
    
    async def optimize_memory_usage(self, components: List[Any]):
        """Optimize memory usage across components"""
        current_memory = self.memory_manager.check_memory_usage()
        
        if current_memory > self.max_memory_mb * 0.7:  # 70% threshold
            logger.info(f"Memory usage high ({current_memory:.1f}MB), optimizing...")
            
            # Clear caches
            self.cache_manager.clear()
            
            # Clear component caches if they have clear methods
            for component in components:
                if hasattr(component, 'clear_cache'):
                    await component.clear_cache()
            
            # Force garbage collection
            self.memory_manager.force_garbage_collection()
            
            new_memory = self.memory_manager.check_memory_usage()
            logger.info(f"Memory optimization complete: {current_memory:.1f}MB -> {new_memory:.1f}MB")
    
    def optimize_for_large_monorepo(self, root_path: Path, target_files: Optional[List[Path]] = None) -> Dict[str, Any]:
        """Optimize processing strategy for large monorepos"""
        optimization_strategy = {
            'use_hierarchy': True,
            'use_streaming': False,
            'batch_size': self.batch_size,
            'max_concurrent_files': 50,
            'enable_aggressive_caching': True,
            'priority_processing': True
        }
        
        # Estimate repository size
        if target_files:
            file_count = len(target_files)
        else:
            # Quick estimation
            file_count = sum(1 for _ in root_path.rglob('*.py'))  # Just Python files for estimation
        
        # Adjust strategy based on size
        if file_count > 100000:  # Very large (100k+ files)
            optimization_strategy.update({
                'use_streaming': True,
                'batch_size': 50,  # Smaller batches
                'max_concurrent_files': 20,
                'enable_file_filtering': True
            })
            logger.info(f"Very large monorepo detected ({file_count} files), using streaming strategy")
        
        elif file_count > 10000:  # Large (10k+ files)
            optimization_strategy.update({
                'batch_size': 100,
                'max_concurrent_files': 30,
                'enable_progressive_indexing': True
            })
            logger.info(f"Large monorepo detected ({file_count} files), using batch strategy")
        
        else:  # Normal size
            optimization_strategy.update({
                'use_hierarchy': False,
                'batch_size': 200,
                'max_concurrent_files': 100
            })
            logger.info(f"Normal size repository ({file_count} files), using standard strategy")
        
        return optimization_strategy
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        current_memory = self.memory_manager.check_memory_usage()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        uptime = time.time() - self.start_time
        
        cache_stats = self.cache_manager.get_stats()
        
        return {
            'system_metrics': {
                'current_memory_mb': current_memory,
                'peak_memory_mb': self.metrics.peak_memory_mb,
                'memory_limit_mb': self.max_memory_mb,
                'memory_utilization': current_memory / self.max_memory_mb,
                'cpu_usage_percent': cpu_percent,
                'uptime_seconds': uptime
            },
            'processing_metrics': {
                'files_processed': self.metrics.files_processed,
                'processing_time': self.metrics.processing_time,
                'throughput_files_per_second': self.metrics.throughput_files_per_second,
                'batch_size': self.batch_size,
                'max_workers': self.batch_processor.max_workers
            },
            'cache_metrics': cache_stats,
            'optimization_recommendations': self._generate_optimization_recommendations(
                current_memory, cpu_percent, cache_stats
            )
        }
    
    def _generate_optimization_recommendations(
        self, 
        current_memory: float, 
        cpu_percent: float, 
        cache_stats: Dict[str, Any]
    ) -> List[str]:
        """Generate optimization recommendations based on current metrics"""
        recommendations = []
        
        # Memory recommendations
        if current_memory / self.max_memory_mb > 0.8:
            recommendations.append("Consider increasing memory limit or reducing batch size")
        
        if cache_stats['memory_utilization'] > 0.9:
            recommendations.append("Cache memory usage is high, consider clearing less important caches")
        
        # CPU recommendations
        if cpu_percent < 50:
            recommendations.append("CPU usage is low, consider increasing batch size or worker count")
        elif cpu_percent > 90:
            recommendations.append("CPU usage is high, consider reducing worker count")
        
        # Throughput recommendations
        if self.metrics.throughput_files_per_second < 10:
            recommendations.append("Low throughput detected, consider optimizing file processing pipeline")
        
        # Cache recommendations
        if cache_stats['utilization'] < 0.3:
            recommendations.append("Cache utilization is low, consider adjusting cache size")
        
        return recommendations
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            self.cache_manager.clear()
            self.memory_manager.force_garbage_collection()
            
            # Cleanup batch processor
            del self.batch_processor
            
            logger.info("Performance optimizer cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            asyncio.create_task(self.cleanup())
        except:
            pass