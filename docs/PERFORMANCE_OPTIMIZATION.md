# ABOV3 Genesis Performance Optimization Guide

## Overview

This comprehensive guide covers performance optimization strategies for ABOV3 Genesis, helping you achieve optimal response times, resource utilization, and scalability. Learn how to tune your system for maximum efficiency across different hardware configurations and use cases.

## Table of Contents

1. [Performance Fundamentals](#performance-fundamentals)
2. [System Requirements Optimization](#system-requirements-optimization)
3. [Ollama Performance Tuning](#ollama-performance-tuning)
4. [ABOV3 Application Optimization](#abov3-application-optimization)
5. [Hardware Optimization](#hardware-optimization)
6. [Network and I/O Optimization](#network-and-io-optimization)
7. [Memory Management](#memory-management)
8. [Caching Strategies](#caching-strategies)
9. [Monitoring and Profiling](#monitoring-and-profiling)
10. [Scaling Strategies](#scaling-strategies)
11. [Platform-Specific Optimizations](#platform-specific-optimizations)
12. [Troubleshooting Performance Issues](#troubleshooting-performance-issues)

## Performance Fundamentals

### Understanding ABOV3 Performance Metrics

#### Key Performance Indicators

**Response Time Metrics:**
- **Cold Start**: Time to load model and generate first response (target: <10s)
- **Warm Response**: Time for subsequent responses (target: <3s)  
- **Code Generation**: Time to generate complete code files (target: <30s)
- **Project Analysis**: Time to analyze project context (target: <5s)

**Resource Utilization:**
- **CPU Usage**: Should stay below 80% average
- **Memory Usage**: Model-dependent, typically 4-16GB
- **GPU Utilization**: 70-90% when using GPU acceleration
- **Disk I/O**: Minimal during steady-state operation

**Throughput Metrics:**
- **Requests per Minute**: Number of user requests processed
- **Concurrent Users**: Number of simultaneous users supported
- **Model Throughput**: Tokens generated per second

#### Performance Baselines

| System Configuration | Cold Start | Warm Response | Code Generation |
|---------------------|------------|---------------|-----------------|
| **Basic** (8GB RAM, CPU only) | 8-15s | 3-8s | 30-60s |
| **Recommended** (16GB RAM, GPU) | 3-8s | 1-3s | 10-30s |
| **High-End** (32GB RAM, High-end GPU) | 2-5s | 0.5-2s | 5-15s |

### Performance Architecture

```
User Request
     │
     ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ABOV3 Core    │────│  Ollama Client  │────│   AI Models     │
│   - Caching     │    │  - Connection   │    │   - Model Load  │
│   - Context     │    │  - Request      │    │   - Inference   │
│   - Routing     │    │  - Response     │    │   - Generation  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
     │                           │                           │
     ▼                           ▼                           ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   File System   │    │    Network      │    │  GPU/CPU/Memory │
│   - I/O Caching │    │   - Latency     │    │   - Utilization │
│   - SSD/NVMe    │    │   - Bandwidth   │    │   - Scheduling  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## System Requirements Optimization

### Hardware Sizing Guide

#### CPU Requirements

**Minimum Configuration:**
```yaml
CPU: 4 cores, 2.5GHz
Use Case: Single user, simple tasks
Performance: Basic functionality with slower responses

Optimization:
- Use smaller models (gemma:2b, phi3)
- Enable CPU performance mode
- Limit background processes
```

**Recommended Configuration:**
```yaml  
CPU: 6-8 cores, 3.0GHz+
Use Case: Regular development work
Performance: Good response times for most tasks

Optimization:
- Use mid-size models (llama3, codellama)
- Enable parallel processing
- Configure CPU affinity for Ollama
```

**High-Performance Configuration:**
```yaml
CPU: 12+ cores, 3.5GHz+
Use Case: Heavy development, multiple users
Performance: Fast responses, concurrent operations

Optimization:  
- Use any models including large ones
- Enable NUMA optimization
- Configure advanced CPU scheduling
```

#### Memory Requirements

**Memory Sizing by Model:**

```yaml
Small Models (2B parameters):
  RAM Required: 4-6GB
  Recommended: 8GB total system RAM
  Models: gemma:2b, phi3
  
Medium Models (7-8B parameters):
  RAM Required: 6-10GB
  Recommended: 16GB total system RAM  
  Models: llama3, codellama, mistral
  
Large Models (13B+ parameters):
  RAM Required: 12-20GB
  Recommended: 32GB total system RAM
  Models: deepseek-coder, yi-coder
```

**Memory Optimization:**
```bash
# Linux: Configure swap for larger models
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Configure swappiness for better performance
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf

# Enable memory compression (if available)
echo 'vm.compressed_memory=1' | sudo tee -a /etc/sysctl.conf
```

#### Storage Requirements

**SSD vs HDD Performance:**

```yaml
NVMe SSD:
  Model Loading: 2-5 seconds
  File Operations: <100ms
  Best For: Primary storage, model storage

SATA SSD:
  Model Loading: 5-10 seconds  
  File Operations: <200ms
  Best For: Budget option, good performance

Traditional HDD:
  Model Loading: 15-30 seconds
  File Operations: 1-5 seconds
  Best For: Not recommended for models
```

**Storage Optimization:**
```bash
# Move models to fastest drive
export OLLAMA_MODELS_PATH="/path/to/nvme/models"

# Enable SSD optimizations (Linux)
echo noop | sudo tee /sys/block/nvme0n1/queue/scheduler
echo 1 | sudo tee /sys/block/nvme0n1/queue/rotational

# Disable unnecessary file indexing
sudo systemctl disable updatedb.timer
```

## Ollama Performance Tuning

### Model Selection Optimization

#### Performance vs Quality Trade-offs

```python
# Performance-oriented model selection
performance_models = {
    "quick_tasks": "gemma:2b",        # Fastest, good quality
    "code_generation": "codellama",   # Balanced speed/quality  
    "general_coding": "llama3:q4_k",  # Quantized for speed
    "complex_analysis": "deepseek-coder:q8"  # High quality, slower
}

# Configure automatic model switching
def select_optimal_model(task_type: str, performance_priority: bool) -> str:
    if performance_priority:
        return performance_models.get(task_type, "gemma:2b")
    else:
        return quality_models.get(task_type, "llama3")
```

#### Model Quantization Strategies

```bash
# Download quantized versions for better performance
ollama pull llama3:q4_k_m    # 4-bit quantization, good balance
ollama pull llama3:q2_k      # 2-bit quantization, fastest
ollama pull llama3:q8_0      # 8-bit quantization, best quality

# Performance comparison
# q2_k:   75% size reduction, 3x faster, some quality loss
# q4_k_m: 50% size reduction, 2x faster, minimal quality loss  
# q8_0:   25% size reduction, 1.5x faster, negligible quality loss
```

### Ollama Server Configuration

#### Environment Variables

```bash
# Performance-oriented Ollama configuration
export OLLAMA_MAX_MEMORY=8192        # Limit memory usage (MB)
export OLLAMA_CONTEXT_SIZE=4096      # Context window size
export OLLAMA_BATCH_SIZE=512         # Batch processing size
export OLLAMA_THREADS=6              # Number of CPU threads
export OLLAMA_MMAP=true              # Enable memory mapping
export OLLAMA_PREDICT=-1             # No prediction limit
export OLLAMA_ROPE_FREQ_BASE=10000   # RoPE frequency base
export OLLAMA_ROPE_FREQ_SCALE=1.0    # RoPE frequency scaling

# GPU-specific settings (if available)
export CUDA_VISIBLE_DEVICES=0        # Use specific GPU
export OLLAMA_GPU_LAYERS=35          # Number of layers on GPU
export OLLAMA_GPU_MEMORY=4096        # GPU memory limit (MB)

# Network optimization  
export OLLAMA_HOST=0.0.0.0:11434     # Listen on all interfaces
export OLLAMA_TIMEOUT=300            # Request timeout (seconds)
export OLLAMA_KEEP_ALIVE=24h         # Keep models loaded
```

#### Advanced Configuration File

Create `~/.ollama/config.yaml`:

```yaml
# Ollama performance configuration
server:
  host: "0.0.0.0"
  port: 11434
  timeout: 300
  keep_alive: "24h"
  
memory:
  max_memory: 8192      # MB
  mmap: true
  swap_size: 4096       # MB
  
processing:
  context_size: 4096
  batch_size: 512
  threads: 6
  predict: -1
  
gpu:
  enabled: true
  device: 0
  memory: 4096          # MB
  layers: 35
  
caching:
  enabled: true
  max_entries: 1000
  ttl: 3600            # seconds
  
logging:
  level: "info"
  file: "ollama.log"
```

### Model Pre-loading and Warming

#### Pre-load Strategy

```bash
#!/bin/bash
# preload_models.sh - Pre-load models for better performance

echo "Pre-loading models for optimal performance..."

# Primary models for different tasks
ollama pull gemma:2b          # Quick tasks
ollama pull llama3:q4_k       # General coding  
ollama pull codellama         # Code generation
ollama pull qwen2-coder       # Debugging

# Warm up models (load into memory)
echo "Warming up models..."
echo "test" | ollama run gemma:2b --keep-alive=24h > /dev/null &
echo "test" | ollama run llama3:q4_k --keep-alive=24h > /dev/null &  
echo "test" | ollama run codellama --keep-alive=24h > /dev/null &
echo "test" | ollama run qwen2-coder --keep-alive=24h > /dev/null &

wait
echo "Models pre-loaded and warmed up!"
```

#### Automatic Model Management

```python
# Python script for intelligent model management
import asyncio
import time
from typing import Dict, List
import psutil

class ModelManager:
    def __init__(self):
        self.loaded_models: Dict[str, float] = {}  # model -> last_used
        self.memory_threshold = 0.8  # 80% memory usage
        self.model_priority = {
            "gemma:2b": 1,      # Highest priority (fastest)
            "llama3:q4_k": 2,   # Medium priority
            "codellama": 3,     # Lower priority
            "deepseek-coder": 4 # Lowest priority (largest)
        }
    
    async def optimize_loaded_models(self):
        """Optimize which models are kept in memory"""
        memory_usage = psutil.virtual_memory().percent / 100
        
        if memory_usage > self.memory_threshold:
            # Unload least recently used, lowest priority models
            await self.unload_lru_models()
    
    async def preload_for_task(self, task_type: str):
        """Pre-load optimal model for specific task"""
        model = self.select_model_for_task(task_type)
        if model not in self.loaded_models:
            await self.load_model(model)
    
    def select_model_for_task(self, task_type: str) -> str:
        """Select optimal model based on task type and current load"""
        current_load = psutil.cpu_percent()
        memory_available = psutil.virtual_memory().available / (1024**3)  # GB
        
        if current_load > 80 or memory_available < 4:
            return "gemma:2b"  # Use fastest model under load
        elif task_type in ["debug", "analysis"]:
            return "qwen2-coder"
        elif task_type in ["code", "generation"]:
            return "codellama" 
        else:
            return "llama3:q4_k"
```

## ABOV3 Application Optimization

### Context Management Optimization

#### Efficient Context Handling

```python
# Optimized context management
class OptimizedContextManager:
    def __init__(self, max_context_size: int = 4096):
        self.max_context_size = max_context_size
        self.context_cache = {}
        self.compression_ratio = 0.7  # Target compression
    
    def optimize_context(self, context: str) -> str:
        """Optimize context for better performance"""
        # 1. Remove redundant information
        context = self.remove_redundancy(context)
        
        # 2. Compress long contexts
        if len(context) > self.max_context_size:
            context = self.compress_context(context)
        
        # 3. Cache frequently used contexts
        context_hash = hash(context)
        if context_hash in self.context_cache:
            return self.context_cache[context_hash]
        
        self.context_cache[context_hash] = context
        return context
    
    def compress_context(self, context: str) -> str:
        """Intelligently compress context while preserving important information"""
        # Extract key information
        important_parts = self.extract_important_parts(context)
        
        # Summarize less important sections
        summarized_parts = self.summarize_sections(context)
        
        # Combine for optimal context
        return self.combine_context_parts(important_parts, summarized_parts)
```

#### Smart Caching System

```python
# Advanced caching for ABOV3
import hashlib
import pickle
from typing import Any, Optional
import redis  # Optional for distributed caching

class SmartCache:
    def __init__(self, redis_url: Optional[str] = None):
        self.local_cache = {}
        self.max_local_size = 1000
        self.redis_client = redis.from_url(redis_url) if redis_url else None
    
    def get_cache_key(self, prompt: str, model: str, context: str) -> str:
        """Generate cache key from request parameters"""
        content = f"{prompt}|{model}|{context}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cached response"""
        # Try local cache first (fastest)
        if key in self.local_cache:
            return self.local_cache[key]['data']
        
        # Try Redis cache (if available)
        if self.redis_client:
            cached = self.redis_client.get(key)
            if cached:
                return pickle.loads(cached)
        
        return None
    
    async def set(self, key: str, data: Any, ttl: int = 3600):
        """Cache response with TTL"""
        # Store in local cache
        if len(self.local_cache) >= self.max_local_size:
            # Remove oldest entry
            oldest_key = min(self.local_cache.keys(), 
                           key=lambda k: self.local_cache[k]['timestamp'])
            del self.local_cache[oldest_key]
        
        self.local_cache[key] = {
            'data': data,
            'timestamp': time.time()
        }
        
        # Store in Redis (if available)
        if self.redis_client:
            self.redis_client.setex(key, ttl, pickle.dumps(data))
```

### Async Processing Optimization

#### Concurrent Request Handling

```python
# Optimized async processing for ABOV3
import asyncio
from asyncio import Queue, Semaphore
from typing import List, Callable, Any
import time

class OptimizedProcessor:
    def __init__(self, max_concurrent: int = 3):
        self.max_concurrent = max_concurrent
        self.semaphore = Semaphore(max_concurrent)
        self.request_queue = Queue()
        self.processing_times = []
        
    async def process_request(self, request_func: Callable, *args, **kwargs) -> Any:
        """Process request with concurrency control and monitoring"""
        async with self.semaphore:
            start_time = time.time()
            try:
                result = await request_func(*args, **kwargs)
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                return result
            except Exception as e:
                # Log error and potentially retry
                await self.handle_processing_error(e, request_func, *args, **kwargs)
                raise
    
    async def batch_process(self, requests: List[tuple]) -> List[Any]:
        """Process multiple requests concurrently"""
        tasks = []
        for request_func, args, kwargs in requests:
            task = asyncio.create_task(
                self.process_request(request_func, *args, **kwargs)
            )
            tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_performance_stats(self) -> dict:
        """Get processing performance statistics"""
        if not self.processing_times:
            return {"avg_time": 0, "min_time": 0, "max_time": 0}
            
        return {
            "avg_time": sum(self.processing_times) / len(self.processing_times),
            "min_time": min(self.processing_times),
            "max_time": max(self.processing_times),
            "request_count": len(self.processing_times)
        }
```

## Hardware Optimization

### GPU Acceleration Setup

#### NVIDIA GPU Configuration

```bash
# Install NVIDIA drivers and CUDA toolkit
# Ubuntu/Debian:
sudo apt update
sudo apt install nvidia-driver-470 nvidia-cuda-toolkit

# Verify installation
nvidia-smi
nvcc --version

# Configure Ollama for GPU
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_GPU_LAYERS=35
export OLLAMA_GPU_MEMORY=4096

# Test GPU acceleration
ollama run llama3 "test gpu performance" --verbose
```

#### Multi-GPU Setup

```bash
# Configure multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OLLAMA_MULTI_GPU=true
export OLLAMA_GPU_SPLIT="3,2,2,1"  # Memory split across GPUs

# Load balance across GPUs
export OLLAMA_GPU_SCHEDULER="round_robin"
```

#### Apple Silicon Optimization

```bash
# macOS Metal acceleration (automatic with Ollama)
# Verify Metal usage
system_profiler SPDisplaysDataType | grep "Metal"

# Monitor GPU usage  
sudo powermetrics -n 1 -i 1000 --samplers gpu_power

# Optimize for Apple Silicon
export OLLAMA_METAL=1
export OLLAMA_METAL_EMBED_LIBRARY=1
```

### CPU Optimization

#### CPU Affinity and Scheduling

```bash
# Linux CPU optimization
# Set CPU affinity for Ollama process
ollama serve &
OLLAMA_PID=$!
taskset -cp 0-7 $OLLAMA_PID  # Use first 8 cores

# Set CPU scheduling priority
sudo renice -10 $OLLAMA_PID  # Higher priority

# Configure CPU governor for performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable CPU throttling
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

# Configure NUMA (if applicable)
numactl --cpubind=0 --membind=0 ollama serve
```

#### Windows CPU Optimization

```powershell
# Set high performance power plan
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

# Set process priority
$process = Get-Process -Name "ollama"
$process.PriorityClass = "High"

# Disable CPU parking
powercfg /setacvalueindex scheme_current sub_processor CPMINCORES 100
powercfg /setactive scheme_current
```

### Memory Optimization

#### Memory Allocation Strategies

```python
# Python memory optimization for ABOV3
import gc
import psutil
from typing import Optional

class MemoryManager:
    def __init__(self, max_memory_percent: float = 80.0):
        self.max_memory_percent = max_memory_percent
        self.memory_warning_threshold = 75.0
        
    def check_memory_usage(self) -> dict:
        """Check current memory usage"""
        memory = psutil.virtual_memory()
        return {
            "total": memory.total / (1024**3),  # GB
            "available": memory.available / (1024**3),  # GB
            "percent_used": memory.percent,
            "warning": memory.percent > self.memory_warning_threshold
        }
    
    def optimize_memory(self):
        """Optimize memory usage"""
        # Force garbage collection
        gc.collect()
        
        # Clear Python caches
        import sys
        sys.intern.__clear__()
        
        # Clear module caches (if safe)
        import importlib
        importlib.invalidate_caches()
    
    def should_use_smaller_model(self) -> bool:
        """Determine if should switch to smaller model based on memory"""
        memory_info = self.check_memory_usage()
        return memory_info["percent_used"] > self.max_memory_percent
```

#### Swap Configuration

```bash
# Linux swap optimization
# Create optimized swap file
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Optimize swap settings
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
echo 'vm.vfs_cache_pressure=50' | sudo tee -a /etc/sysctl.conf
echo 'vm.dirty_background_ratio=5' | sudo tee -a /etc/sysctl.conf
echo 'vm.dirty_ratio=10' | sudo tee -a /etc/sysctl.conf

# Apply settings
sudo sysctl -p
```

## Network and I/O Optimization

### Network Configuration

#### TCP Optimization for API Calls

```bash
# Linux network optimization
# Increase TCP buffer sizes
echo 'net.core.rmem_max=134217728' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max=134217728' | sudo tee -a /etc/sysctl.conf
echo 'net.ipv4.tcp_rmem=4096 65536 134217728' | sudo tee -a /etc/sysctl.conf
echo 'net.ipv4.tcp_wmem=4096 65536 134217728' | sudo tee -a /etc/sysctl.conf

# Enable TCP window scaling
echo 'net.ipv4.tcp_window_scaling=1' | sudo tee -a /etc/sysctl.conf

# Reduce TCP timeout
echo 'net.ipv4.tcp_keepalive_time=120' | sudo tee -a /etc/sysctl.conf

# Apply settings
sudo sysctl -p
```

#### Connection Pooling

```python
# Optimized HTTP client for Ollama API
import aiohttp
import asyncio
from typing import Optional

class OptimizedOllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.connector_limits = aiohttp.TCPConnector(
            limit=100,              # Total connection limit
            limit_per_host=20,      # Per-host connection limit
            ttl_dns_cache=300,      # DNS cache TTL
            use_dns_cache=True,     # Enable DNS caching
            keepalive_timeout=30,   # Keep connections alive
            enable_cleanup_closed=True
        )
    
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=300, connect=10)
        self.session = aiohttp.ClientSession(
            connector=self.connector_limits,
            timeout=timeout,
            headers={"Connection": "keep-alive"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def generate(self, model: str, prompt: str, **kwargs) -> dict:
        """Generate response with optimized connection handling"""
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            **kwargs
        }
        
        async with self.session.post(f"{self.base_url}/api/generate", 
                                   json=data) as response:
            return await response.json()
```

### File I/O Optimization

#### Async File Operations

```python
# Optimized file I/O for ABOV3
import aiofiles
import aiofiles.os
from pathlib import Path
import asyncio
from typing import List, Union

class OptimizedFileManager:
    def __init__(self, max_concurrent_files: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent_files)
    
    async def read_file_async(self, file_path: Union[str, Path]) -> str:
        """Read file asynchronously with concurrency control"""
        async with self.semaphore:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                return await f.read()
    
    async def write_file_async(self, file_path: Union[str, Path], 
                             content: str) -> None:
        """Write file asynchronously with concurrency control"""
        async with self.semaphore:
            # Ensure directory exists
            await aiofiles.os.makedirs(Path(file_path).parent, exist_ok=True)
            
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(content)
    
    async def batch_read_files(self, file_paths: List[Union[str, Path]]) -> List[str]:
        """Read multiple files concurrently"""
        tasks = [self.read_file_async(path) for path in file_paths]
        return await asyncio.gather(*tasks)
    
    async def batch_write_files(self, file_data: List[tuple]) -> None:
        """Write multiple files concurrently"""
        tasks = [self.write_file_async(path, content) for path, content in file_data]
        await asyncio.gather(*tasks)
```

#### SSD Optimization

```bash
# SSD optimization for model storage
# Enable TRIM support
sudo systemctl enable fstrim.timer
sudo systemctl start fstrim.timer

# Set optimal I/O scheduler for SSD
echo noop | sudo tee /sys/block/nvme0n1/queue/scheduler

# Disable access time updates (reduces writes)
# Add 'noatime' to fstab
sudo sed -i 's/defaults/defaults,noatime/g' /etc/fstab

# Set optimal read-ahead values
echo 256 | sudo tee /sys/block/nvme0n1/queue/read_ahead_kb
```

## Memory Management

### Garbage Collection Optimization

```python
# Optimized garbage collection for ABOV3
import gc
import weakref
from typing import Dict, Any, Optional

class MemoryOptimizer:
    def __init__(self):
        self.weak_cache: Dict[str, weakref.ref] = {}
        self.gc_threshold = (700, 10, 10)  # Adjusted thresholds
        
    def configure_gc(self):
        """Configure garbage collection for better performance"""
        # Set custom thresholds
        gc.set_threshold(*self.gc_threshold)
        
        # Enable debug flags in development
        if __debug__:
            gc.set_debug(gc.DEBUG_STATS)
    
    def periodic_cleanup(self):
        """Perform periodic memory cleanup"""
        # Force garbage collection
        collected = gc.collect()
        
        # Clear weak references to deleted objects
        dead_refs = []
        for key, ref in self.weak_cache.items():
            if ref() is None:
                dead_refs.append(key)
        
        for key in dead_refs:
            del self.weak_cache[key]
        
        return {
            "objects_collected": collected,
            "weak_refs_cleaned": len(dead_refs),
            "current_objects": len(gc.get_objects())
        }
    
    def get_memory_stats(self) -> dict:
        """Get detailed memory statistics"""
        import psutil
        import sys
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss": memory_info.rss / (1024**2),  # MB
            "vms": memory_info.vms / (1024**2),  # MB
            "python_objects": len(gc.get_objects()),
            "gc_counts": gc.get_count(),
            "sys_getsizeof": sys.getsizeof(gc.get_objects())
        }
```

### Memory Pool Management

```python
# Custom memory pool for frequent allocations
import mmap
from typing import List, Optional
import struct

class MemoryPool:
    def __init__(self, block_size: int = 4096, pool_size: int = 1024):
        self.block_size = block_size
        self.pool_size = pool_size
        self.total_size = block_size * pool_size
        
        # Create memory mapped pool
        self.pool = mmap.mmap(-1, self.total_size)
        self.free_blocks: List[int] = list(range(pool_size))
        self.allocated_blocks: set = set()
    
    def allocate_block(self) -> Optional[int]:
        """Allocate a memory block"""
        if not self.free_blocks:
            return None
        
        block_id = self.free_blocks.pop()
        self.allocated_blocks.add(block_id)
        return block_id
    
    def free_block(self, block_id: int):
        """Free a memory block"""
        if block_id in self.allocated_blocks:
            self.allocated_blocks.remove(block_id)
            self.free_blocks.append(block_id)
            
            # Clear the block
            offset = block_id * self.block_size
            self.pool[offset:offset + self.block_size] = b'\x00' * self.block_size
    
    def write_block(self, block_id: int, data: bytes) -> bool:
        """Write data to a memory block"""
        if block_id not in self.allocated_blocks or len(data) > self.block_size:
            return False
        
        offset = block_id * self.block_size
        self.pool[offset:offset + len(data)] = data
        return True
    
    def read_block(self, block_id: int, size: int = None) -> bytes:
        """Read data from a memory block"""
        if block_id not in self.allocated_blocks:
            return b''
        
        offset = block_id * self.block_size
        if size is None:
            size = self.block_size
        
        return self.pool[offset:offset + size]
    
    def get_stats(self) -> dict:
        """Get memory pool statistics"""
        return {
            "total_blocks": self.pool_size,
            "free_blocks": len(self.free_blocks),
            "allocated_blocks": len(self.allocated_blocks),
            "memory_usage": len(self.allocated_blocks) * self.block_size / (1024**2)  # MB
        }
    
    def __del__(self):
        if hasattr(self, 'pool'):
            self.pool.close()
```

## Caching Strategies

### Multi-Level Caching System

```python
# Advanced multi-level caching for ABOV3
import hashlib
import json
import sqlite3
import pickle
from typing import Any, Optional, Union
from datetime import datetime, timedelta
import asyncio

class MultiLevelCache:
    def __init__(self, db_path: str = "abov3_cache.db"):
        self.memory_cache = {}  # L1 Cache
        self.disk_cache_path = db_path  # L2 Cache
        self.max_memory_items = 500
        self.default_ttl = 3600  # 1 hour
        
        self._init_disk_cache()
    
    def _init_disk_cache(self):
        """Initialize SQLite disk cache"""
        self.conn = sqlite3.connect(self.disk_cache_path, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value BLOB,
                created_at DATETIME,
                ttl INTEGER,
                access_count INTEGER DEFAULT 0,
                last_accessed DATETIME
            )
        """)
        self.conn.commit()
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        content = json.dumps([args, sorted(kwargs.items())], sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache (L1 -> L2)"""
        # Try L1 cache first
        if key in self.memory_cache:
            item = self.memory_cache[key]
            if not self._is_expired(item):
                item['access_count'] += 1
                item['last_accessed'] = datetime.now()
                return item['value']
            else:
                del self.memory_cache[key]
        
        # Try L2 cache
        cursor = self.conn.execute(
            "SELECT value, created_at, ttl FROM cache WHERE key = ?", (key,)
        )
        row = cursor.fetchone()
        
        if row:
            value, created_at, ttl = row
            created_dt = datetime.fromisoformat(created_at)
            
            if datetime.now() - created_dt < timedelta(seconds=ttl):
                # Move to L1 cache for faster access
                unpickled_value = pickle.loads(value)
                await self._set_memory_cache(key, unpickled_value, ttl)
                
                # Update access statistics
                self.conn.execute(
                    "UPDATE cache SET access_count = access_count + 1, last_accessed = ? WHERE key = ?",
                    (datetime.now().isoformat(), key)
                )
                self.conn.commit()
                
                return unpickled_value
            else:
                # Remove expired item
                self.conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                self.conn.commit()
        
        return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> None:
        """Set item in cache (L1 and L2)"""
        ttl = ttl or self.default_ttl
        
        # Set in L1 cache
        await self._set_memory_cache(key, value, ttl)
        
        # Set in L2 cache
        pickled_value = pickle.dumps(value)
        self.conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, created_at, ttl) VALUES (?, ?, ?, ?)",
            (key, pickled_value, datetime.now().isoformat(), ttl)
        )
        self.conn.commit()
    
    async def _set_memory_cache(self, key: str, value: Any, ttl: int):
        """Set item in memory cache with eviction"""
        if len(self.memory_cache) >= self.max_memory_items:
            # Evict least recently accessed item
            lru_key = min(self.memory_cache.keys(),
                         key=lambda k: self.memory_cache[k]['last_accessed'])
            del self.memory_cache[lru_key]
        
        self.memory_cache[key] = {
            'value': value,
            'created_at': datetime.now(),
            'ttl': ttl,
            'access_count': 0,
            'last_accessed': datetime.now()
        }
    
    def _is_expired(self, item: dict) -> bool:
        """Check if cache item is expired"""
        age = datetime.now() - item['created_at']
        return age.total_seconds() > item['ttl']
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        cursor = self.conn.execute("SELECT COUNT(*), AVG(access_count) FROM cache")
        disk_count, avg_access = cursor.fetchone()
        
        return {
            "l1_cache_size": len(self.memory_cache),
            "l2_cache_size": disk_count or 0,
            "avg_access_count": avg_access or 0,
            "hit_ratio": self._calculate_hit_ratio()
        }
    
    def cleanup_expired(self):
        """Remove expired items from disk cache"""
        cutoff = datetime.now() - timedelta(hours=24)  # Remove items older than 24h
        self.conn.execute(
            "DELETE FROM cache WHERE datetime(created_at) < ?",
            (cutoff.isoformat(),)
        )
        self.conn.commit()
    
    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()
```

### Smart Cache Invalidation

```python
# Intelligent cache invalidation system
import fnmatch
from typing import Set, List, Pattern
import re

class SmartCacheInvalidation:
    def __init__(self, cache: MultiLevelCache):
        self.cache = cache
        self.invalidation_patterns: List[Pattern] = []
        self.dependency_graph: dict = {}  # key -> set of dependent keys
    
    def add_dependency(self, key: str, depends_on: List[str]):
        """Add cache dependency relationship"""
        for dep in depends_on:
            if dep not in self.dependency_graph:
                self.dependency_graph[dep] = set()
            self.dependency_graph[dep].add(key)
    
    def invalidate_key(self, key: str):
        """Invalidate a specific key and its dependents"""
        # Remove from cache
        self.cache.delete(key)
        
        # Invalidate dependent keys
        if key in self.dependency_graph:
            for dependent_key in self.dependency_graph[key]:
                self.invalidate_key(dependent_key)
    
    def invalidate_pattern(self, pattern: str):
        """Invalidate keys matching a pattern"""
        # Get all cache keys
        all_keys = self.cache.get_all_keys()
        
        # Find matching keys
        matching_keys = fnmatch.filter(all_keys, pattern)
        
        # Invalidate each matching key
        for key in matching_keys:
            self.invalidate_key(key)
    
    def register_pattern(self, pattern: str):
        """Register a pattern for automatic invalidation"""
        self.invalidation_patterns.append(re.compile(pattern))
    
    def auto_invalidate(self, event_type: str, context: dict):
        """Automatically invalidate based on events"""
        if event_type == "file_modified":
            file_path = context.get("file_path", "")
            # Invalidate caches related to this file
            self.invalidate_pattern(f"*{file_path}*")
        
        elif event_type == "model_changed":
            model_name = context.get("model", "")
            # Invalidate all caches for this model
            self.invalidate_pattern(f"*model:{model_name}*")
        
        elif event_type == "project_modified":
            project_path = context.get("project_path", "")
            # Invalidate all project-related caches
            self.invalidate_pattern(f"*project:{project_path}*")
```

## Monitoring and Profiling

### Performance Monitoring System

```python
# Comprehensive performance monitoring for ABOV3
import time
import psutil
import asyncio
from typing import Dict, List, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
import statistics

@dataclass
class PerformanceMetrics:
    operation: str
    start_time: float
    end_time: float
    duration: float
    memory_before: float
    memory_after: float
    cpu_percent: float
    success: bool
    error: str = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class PerformanceMonitor:
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.active_operations: Dict[str, float] = {}
        self.alert_thresholds = {
            "response_time": 10.0,  # seconds
            "memory_usage": 85.0,   # percentage
            "cpu_usage": 90.0       # percentage
        }
        self.alerts: List[dict] = []
    
    def start_operation(self, operation_id: str, operation_name: str):
        """Start monitoring an operation"""
        self.active_operations[operation_id] = {
            "name": operation_name,
            "start_time": time.time(),
            "memory_before": psutil.virtual_memory().percent,
            "cpu_before": psutil.cpu_percent()
        }
    
    def end_operation(self, operation_id: str, success: bool = True, 
                     error: str = None, metadata: Dict[str, Any] = None):
        """End monitoring an operation"""
        if operation_id not in self.active_operations:
            return
        
        op_data = self.active_operations.pop(operation_id)
        end_time = time.time()
        
        metrics = PerformanceMetrics(
            operation=op_data["name"],
            start_time=op_data["start_time"],
            end_time=end_time,
            duration=end_time - op_data["start_time"],
            memory_before=op_data["memory_before"],
            memory_after=psutil.virtual_memory().percent,
            cpu_percent=psutil.cpu_percent(),
            success=success,
            error=error,
            metadata=metadata or {}
        )
        
        self.metrics.append(metrics)
        self._check_alerts(metrics)
    
    def _check_alerts(self, metrics: PerformanceMetrics):
        """Check for performance alerts"""
        alerts = []
        
        if metrics.duration > self.alert_thresholds["response_time"]:
            alerts.append({
                "type": "slow_response",
                "message": f"Operation '{metrics.operation}' took {metrics.duration:.2f}s",
                "timestamp": datetime.now(),
                "severity": "warning"
            })
        
        if metrics.memory_after > self.alert_thresholds["memory_usage"]:
            alerts.append({
                "type": "high_memory",
                "message": f"Memory usage reached {metrics.memory_after:.1f}%",
                "timestamp": datetime.now(),
                "severity": "warning"
            })
        
        if metrics.cpu_percent > self.alert_thresholds["cpu_usage"]:
            alerts.append({
                "type": "high_cpu",
                "message": f"CPU usage reached {metrics.cpu_percent:.1f}%",
                "timestamp": datetime.now(),
                "severity": "warning"
            })
        
        self.alerts.extend(alerts)
    
    def get_performance_summary(self, operation_name: str = None) -> dict:
        """Get performance summary for operations"""
        if operation_name:
            relevant_metrics = [m for m in self.metrics if m.operation == operation_name]
        else:
            relevant_metrics = self.metrics
        
        if not relevant_metrics:
            return {}
        
        durations = [m.duration for m in relevant_metrics]
        success_rate = sum(1 for m in relevant_metrics if m.success) / len(relevant_metrics)
        
        return {
            "operation": operation_name or "all",
            "total_calls": len(relevant_metrics),
            "success_rate": success_rate * 100,
            "avg_duration": statistics.mean(durations),
            "median_duration": statistics.median(durations),
            "p95_duration": statistics.quantiles(durations, n=20)[18] if len(durations) > 20 else max(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "recent_alerts": len([a for a in self.alerts if (datetime.now() - a["timestamp"]).seconds < 300])
        }
    
    def export_metrics(self, format: str = "json", file_path: str = None) -> str:
        """Export metrics to file or string"""
        data = {
            "export_time": datetime.now().isoformat(),
            "metrics": [
                {
                    "operation": m.operation,
                    "duration": m.duration,
                    "memory_usage": m.memory_after - m.memory_before,
                    "cpu_percent": m.cpu_percent,
                    "success": m.success,
                    "timestamp": datetime.fromtimestamp(m.start_time).isoformat(),
                    "error": m.error,
                    "metadata": m.metadata
                }
                for m in self.metrics
            ],
            "alerts": self.alerts,
            "summary": self.get_performance_summary()
        }
        
        if format == "json":
            import json
            output = json.dumps(data, indent=2, default=str)
        else:
            output = str(data)
        
        if file_path:
            with open(file_path, 'w') as f:
                f.write(output)
        
        return output

# Usage decorator for easy monitoring
def monitor_performance(operation_name: str, monitor: PerformanceMonitor):
    def decorator(func: Callable):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                operation_id = f"{operation_name}_{time.time()}"
                monitor.start_operation(operation_id, operation_name)
                
                try:
                    result = await func(*args, **kwargs)
                    monitor.end_operation(operation_id, success=True)
                    return result
                except Exception as e:
                    monitor.end_operation(operation_id, success=False, error=str(e))
                    raise
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                operation_id = f"{operation_name}_{time.time()}"
                monitor.start_operation(operation_id, operation_name)
                
                try:
                    result = func(*args, **kwargs)
                    monitor.end_operation(operation_id, success=True)
                    return result
                except Exception as e:
                    monitor.end_operation(operation_id, success=False, error=str(e))
                    raise
            return sync_wrapper
    return decorator
```

### Real-time Performance Dashboard

```python
# Real-time performance dashboard
import threading
import time
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.columns import Columns
import psutil

class PerformanceDashboard:
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.console = Console()
        self.update_interval = 1.0  # seconds
        self.running = False
    
    def create_dashboard(self) -> Panel:
        """Create the dashboard display"""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        system_table = Table(title="System Resources")
        system_table.add_column("Metric", style="cyan")
        system_table.add_column("Value", style="green")
        system_table.add_column("Status", style="yellow")
        
        # CPU status
        cpu_status = "🟢 Good" if cpu_percent < 70 else "🟡 High" if cpu_percent < 90 else "🔴 Critical"
        system_table.add_row("CPU Usage", f"{cpu_percent:.1f}%", cpu_status)
        
        # Memory status
        mem_status = "🟢 Good" if memory.percent < 70 else "🟡 High" if memory.percent < 90 else "🔴 Critical"
        system_table.add_row("Memory Usage", f"{memory.percent:.1f}%", mem_status)
        
        # Disk status
        disk_status = "🟢 Good" if disk.percent < 70 else "🟡 High" if disk.percent < 90 else "🔴 Critical"
        system_table.add_row("Disk Usage", f"{disk.percent:.1f}%", disk_status)
        
        # Performance metrics
        perf_summary = self.monitor.get_performance_summary()
        perf_table = Table(title="ABOV3 Performance")
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="green")
        
        if perf_summary:
            perf_table.add_row("Total Requests", str(perf_summary.get("total_calls", 0)))
            perf_table.add_row("Success Rate", f"{perf_summary.get('success_rate', 0):.1f}%")
            perf_table.add_row("Avg Response Time", f"{perf_summary.get('avg_duration', 0):.2f}s")
            perf_table.add_row("P95 Response Time", f"{perf_summary.get('p95_duration', 0):.2f}s")
        else:
            perf_table.add_row("Status", "No data yet")
        
        # Recent alerts
        recent_alerts = [a for a in self.monitor.alerts if (time.time() - a["timestamp"].timestamp()) < 300]
        alert_table = Table(title="Recent Alerts (5 min)")
        alert_table.add_column("Time", style="cyan")
        alert_table.add_column("Type", style="yellow")
        alert_table.add_column("Message", style="red")
        
        for alert in recent_alerts[-5:]:  # Show last 5 alerts
            alert_table.add_row(
                alert["timestamp"].strftime("%H:%M:%S"),
                alert["type"],
                alert["message"]
            )
        
        if not recent_alerts:
            alert_table.add_row("", "", "No recent alerts")
        
        # Combine tables
        dashboard = Columns([
            Panel(system_table, title="System", border_style="blue"),
            Panel(perf_table, title="Performance", border_style="green"),
            Panel(alert_table, title="Alerts", border_style="red")
        ])
        
        return Panel(dashboard, title="ABOV3 Genesis Performance Dashboard", border_style="bright_blue")
    
    def start_dashboard(self):
        """Start the real-time dashboard"""
        self.running = True
        
        with Live(self.create_dashboard(), refresh_per_second=1) as live:
            while self.running:
                live.update(self.create_dashboard())
                time.sleep(self.update_interval)
    
    def stop_dashboard(self):
        """Stop the dashboard"""
        self.running = False

# Usage example
if __name__ == "__main__":
    monitor = PerformanceMonitor()
    dashboard = PerformanceDashboard(monitor)
    
    # Start dashboard in separate thread
    dashboard_thread = threading.Thread(target=dashboard.start_dashboard, daemon=True)
    dashboard_thread.start()
    
    # Your ABOV3 application continues running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        dashboard.stop_dashboard()
```

## Scaling Strategies

### Horizontal Scaling

#### Load Balancer Configuration

```nginx
# nginx.conf for ABOV3 Genesis load balancing
upstream abov3_backend {
    # Weighted round-robin
    server abov3-1:8080 weight=3;
    server abov3-2:8080 weight=2;
    server abov3-3:8080 weight=1;
    
    # Health checks
    keepalive 32;
    keepalive_requests 100;
    keepalive_timeout 60s;
}

server {
    listen 80;
    server_name abov3.yourdomain.com;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
    
    location / {
        proxy_pass http://abov3_backend;
        
        # Connection settings
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
        
        # Buffering
        proxy_buffering on;
        proxy_buffer_size 128k;
        proxy_buffers 4 256k;
        proxy_busy_buffers_size 256k;
    }
    
    # Health check endpoint
    location /health {
        proxy_pass http://abov3_backend/health;
        proxy_method GET;
    }
    
    # Static file serving
    location /static/ {
        root /var/www/abov3;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

#### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: abov3-genesis
  namespace: abov3
spec:
  replicas: 3
  selector:
    matchLabels:
      app: abov3-genesis
  template:
    metadata:
      labels:
        app: abov3-genesis
    spec:
      containers:
      - name: abov3-genesis
        image: abov3/genesis:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            cpu: 500m
            memory: 2Gi
          limits:
            cpu: 2000m
            memory: 8Gi
        env:
        - name: OLLAMA_HOST
          value: "ollama-service:11434"
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: model-cache
          mountPath: /app/.ollama
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: abov3-service
  namespace: abov3
spec:
  selector:
    app: abov3-genesis
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: abov3-hpa
  namespace: abov3
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: abov3-genesis
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

### Vertical Scaling

#### Dynamic Resource Adjustment

```python
# Dynamic resource scaling based on load
import psutil
import asyncio
import subprocess
from typing import Dict, Tuple

class VerticalScaler:
    def __init__(self):
        self.current_limits = {
            "cpu_percent": 70,
            "memory_percent": 80,
            "ollama_memory": 8192  # MB
        }
        self.scaling_history = []
        
    async def monitor_and_scale(self):
        """Continuously monitor and scale resources"""
        while True:
            current_usage = self.get_current_usage()
            scaling_decision = self.make_scaling_decision(current_usage)
            
            if scaling_decision:
                await self.apply_scaling(scaling_decision)
            
            await asyncio.sleep(30)  # Check every 30 seconds
    
    def get_current_usage(self) -> Dict[str, float]:
        """Get current resource usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Get Ollama process usage
        ollama_usage = self.get_ollama_usage()
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "ollama_cpu": ollama_usage["cpu"],
            "ollama_memory": ollama_usage["memory"]
        }
    
    def get_ollama_usage(self) -> Dict[str, float]:
        """Get Ollama-specific resource usage"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
                if proc.info['name'] and 'ollama' in proc.info['name'].lower():
                    return {
                        "cpu": proc.info['cpu_percent'],
                        "memory": proc.info['memory_info'].rss / (1024**2)  # MB
                    }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        
        return {"cpu": 0, "memory": 0}
    
    def make_scaling_decision(self, usage: Dict[str, float]) -> Dict[str, any]:
        """Decide if scaling is needed"""
        decisions = {}
        
        # CPU scaling
        if usage["cpu_percent"] > 85:
            decisions["action"] = "scale_up"
            decisions["reason"] = "High CPU usage"
        elif usage["cpu_percent"] < 30 and len(self.scaling_history) > 0:
            decisions["action"] = "scale_down"
            decisions["reason"] = "Low CPU usage"
        
        # Memory scaling for Ollama
        if usage["ollama_memory"] > self.current_limits["ollama_memory"] * 0.9:
            decisions["ollama_memory"] = min(16384, self.current_limits["ollama_memory"] * 1.5)
            decisions["reason"] = "High Ollama memory usage"
        
        return decisions if decisions else None
    
    async def apply_scaling(self, decision: Dict[str, any]):
        """Apply scaling decisions"""
        if "ollama_memory" in decision:
            await self.scale_ollama_memory(decision["ollama_memory"])
        
        if decision.get("action") == "scale_up":
            await self.scale_up_resources()
        elif decision.get("action") == "scale_down":
            await self.scale_down_resources()
        
        self.scaling_history.append({
            "timestamp": asyncio.get_event_loop().time(),
            "decision": decision,
            "usage_before": self.get_current_usage()
        })
    
    async def scale_ollama_memory(self, new_limit: int):
        """Scale Ollama memory limit"""
        try:
            # Update environment variable
            import os
            os.environ["OLLAMA_MAX_MEMORY"] = str(new_limit)
            
            # Restart Ollama service (implementation depends on your setup)
            await self.restart_ollama_service()
            
            self.current_limits["ollama_memory"] = new_limit
            print(f"Scaled Ollama memory to {new_limit}MB")
            
        except Exception as e:
            print(f"Failed to scale Ollama memory: {e}")
    
    async def restart_ollama_service(self):
        """Restart Ollama service (platform-specific)"""
        try:
            # Linux systemd
            await asyncio.create_subprocess_exec(
                "sudo", "systemctl", "restart", "ollama"
            )
        except Exception as e:
            print(f"Failed to restart Ollama: {e}")
```

## Platform-Specific Optimizations

### Windows Optimizations

```powershell
# Windows performance optimization script
# Run as Administrator

# Set high performance power plan
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

# Disable Windows Defender real-time protection for ABOV3 directories
Add-MpPreference -ExclusionPath "$env:USERPROFILE\.abov3"
Add-MpPreference -ExclusionPath "$env:USERPROFILE\projects"
Add-MpPreference -ExclusionProcess "ollama.exe"
Add-MpPreference -ExclusionProcess "python.exe"

# Optimize virtual memory
$TotalRAM = (Get-CimInstance Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum).Sum / 1GB
$PageFileSize = [math]::Round($TotalRAM * 1.5) * 1024  # 1.5x RAM in MB
$pagefile = Get-WmiObject -Class Win32_ComputerSystem -EnableAllPrivileges
$pagefile.AutomaticManagedPagefile = $false
$pagefile.Put()

# Set specific page file size
$pagefileset = Get-WmiObject -Class Win32_PageFileSetting
$pagefileset.InitialSize = $PageFileSize
$pagefileset.MaximumSize = $PageFileSize
$pagefileset.Put()

# Optimize system responsiveness
Set-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Multimedia\SystemProfile" -Name "SystemResponsiveness" -Value 0

# Set process scheduling to optimize for programs
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\PriorityControl" -Name "Win32PrioritySeparation" -Value 38

# Disable unnecessary services
$ServicesToStop = @(
    "WSearch",      # Windows Search
    "SysMain",      # Superfetch
    "Themes",       # Windows Themes
    "TabletInputService"  # Tablet Input Service
)

foreach ($Service in $ServicesToStop) {
    Set-Service -Name $Service -StartupType Disabled -ErrorAction SilentlyContinue
    Stop-Service -Name $Service -Force -ErrorAction SilentlyContinue
}

Write-Host "Windows optimization complete. Restart required."
```

### Linux Optimizations

```bash
#!/bin/bash
# Linux performance optimization script

echo "Optimizing Linux for ABOV3 Genesis..."

# CPU optimizations
echo "Configuring CPU performance..."

# Set CPU governor to performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable CPU power saving
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || true

# Set CPU affinity for better performance
echo "Configuring CPU affinity..."
cat > /tmp/cpu_affinity.conf << 'EOF'
# ABOV3 Genesis CPU affinity
kernel.sched_autogroup_enabled = 0
kernel.sched_child_runs_first = 1
kernel.sched_latency_ns = 1000000
kernel.sched_min_granularity_ns = 100000
kernel.sched_wakeup_granularity_ns = 50000
EOF

sudo cp /tmp/cpu_affinity.conf /etc/sysctl.d/99-abov3-cpu.conf

# Memory optimizations
echo "Configuring memory settings..."
cat > /tmp/memory_opt.conf << 'EOF'
# Memory optimization for ABOV3
vm.swappiness = 10
vm.vfs_cache_pressure = 50
vm.dirty_background_ratio = 5
vm.dirty_ratio = 10
vm.dirty_writeback_centisecs = 100
vm.dirty_expire_centisecs = 300
vm.page-cluster = 3
vm.overcommit_memory = 1
vm.overcommit_ratio = 80
EOF

sudo cp /tmp/memory_opt.conf /etc/sysctl.d/99-abov3-memory.conf

# Network optimizations
echo "Configuring network settings..."
cat > /tmp/network_opt.conf << 'EOF'
# Network optimization for ABOV3
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 65536 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.ipv4.tcp_window_scaling = 1
net.ipv4.tcp_timestamps = 1
net.ipv4.tcp_sack = 1
net.ipv4.tcp_congestion_control = bbr
net.core.default_qdisc = fq
net.ipv4.tcp_keepalive_time = 120
net.ipv4.tcp_keepalive_intvl = 30
net.ipv4.tcp_keepalive_probes = 3
EOF

sudo cp /tmp/network_opt.conf /etc/sysctl.d/99-abov3-network.conf

# I/O optimizations
echo "Configuring I/O settings..."

# Set I/O scheduler for SSDs
for disk in /sys/block/sd*; do
    if [ -f "$disk/queue/rotational" ] && [ "$(cat $disk/queue/rotational)" = "0" ]; then
        echo noop | sudo tee $disk/queue/scheduler
        echo "Set noop scheduler for SSD: $(basename $disk)"
    fi
done

for disk in /sys/block/nvme*; do
    if [ -d "$disk" ]; then
        echo none | sudo tee $disk/queue/scheduler 2>/dev/null || true
        echo "Set none scheduler for NVMe: $(basename $disk)"
    fi
done

# Optimize read-ahead
echo "Setting optimal read-ahead values..."
for disk in /sys/block/sd* /sys/block/nvme*; do
    if [ -d "$disk" ]; then
        echo 256 | sudo tee $disk/queue/read_ahead_kb 2>/dev/null || true
    fi
done

# File system optimizations
echo "Configuring file system..."
cat > /tmp/fs_opt.conf << 'EOF'
# File system optimization for ABOV3
fs.file-max = 2097152
fs.nr_open = 1048576
fs.inotify.max_user_watches = 524288
fs.inotify.max_user_instances = 512
EOF

sudo cp /tmp/fs_opt.conf /etc/sysctl.d/99-abov3-fs.conf

# Apply all sysctl changes
sudo sysctl -p /etc/sysctl.d/99-abov3-*.conf

# Ulimit optimizations
echo "Configuring process limits..."
cat > /tmp/limits.conf << 'EOF'
# ABOV3 Genesis limits
* soft nofile 65536
* hard nofile 65536
* soft nproc 32768
* hard nproc 32768
* soft memlock unlimited
* hard memlock unlimited
EOF

sudo cp /tmp/limits.conf /etc/security/limits.d/99-abov3.conf

# Create systemd service for Ollama with optimizations
echo "Creating optimized Ollama service..."
sudo tee /etc/systemd/system/ollama-optimized.service > /dev/null << 'EOF'
[Unit]
Description=Ollama Service (Optimized for ABOV3)
After=network.target

[Service]
Type=simple
User=ollama
Group=ollama
ExecStart=/usr/local/bin/ollama serve
Restart=always
RestartSec=3
Environment=OLLAMA_MAX_MEMORY=8192
Environment=OLLAMA_THREADS=8
Environment=OLLAMA_MMAP=true
Environment=OLLAMA_KEEP_ALIVE=24h
LimitNOFILE=65536
LimitNPROC=32768
LimitMEMLOCK=infinity
Nice=-10
IOSchedulingClass=1
IOSchedulingPriority=4
CPUSchedulingPolicy=1
CPUSchedulingPriority=50

[Install]
WantedBy=multi-user.target
EOF

# Create ollama user if it doesn't exist
sudo useradd -r -s /bin/false -d /usr/share/ollama ollama 2>/dev/null || true

# Enable and start the optimized service
sudo systemctl daemon-reload
sudo systemctl enable ollama-optimized.service
sudo systemctl stop ollama.service 2>/dev/null || true  # Stop default service
sudo systemctl disable ollama.service 2>/dev/null || true
sudo systemctl start ollama-optimized.service

# Install additional performance tools
echo "Installing performance monitoring tools..."
sudo apt update
sudo apt install -y htop iotop nethogs sysstat nvtop 2>/dev/null || true

echo "Linux optimization complete!"
echo "Reboot recommended for all changes to take effect."
echo ""
echo "Performance monitoring commands:"
echo "  htop          - CPU and memory usage"
echo "  iotop         - I/O usage"
echo "  nethogs       - Network usage by process"
echo "  iostat 1      - I/O statistics"
echo "  nvidia-smi    - GPU usage (if NVIDIA GPU present)"
echo "  nvtop         - GPU monitoring"
```

### macOS Optimizations

```bash
#!/bin/bash
# macOS performance optimization script

echo "Optimizing macOS for ABOV3 Genesis..."

# Disable unnecessary visual effects
echo "Disabling visual effects..."
defaults write com.apple.dock minimize-to-application -bool true
defaults write com.apple.dock expose-animation-duration -float 0.1
defaults write com.apple.dock autohide-delay -float 0
defaults write com.apple.dock autohide-time-modifier -float 0.5
defaults write NSGlobalDomain NSWindowResizeTime -float 0.001

# Optimize energy settings
echo "Configuring energy settings..."
sudo pmset -a standby 0
sudo pmset -a hibernatemode 0
sudo pmset -a autopoweroff 0
sudo pmset -a powernap 0
sudo pmset -a proximitywake 0
sudo pmset -a tcpkeepalive 0

# Set high performance mode
sudo pmset -a lowpowermode 0

# Increase file limits
echo "Configuring file limits..."
echo 'limit maxfiles 65536 65536' | sudo tee -a /etc/launchd.conf
echo 'limit maxproc 2048 2048' | sudo tee -a /etc/launchd.conf

# Create launch daemon for file limits
sudo tee /Library/LaunchDaemons/limit.maxfiles.plist > /dev/null << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>limit.maxfiles</string>
    <key>ProgramArguments</key>
    <array>
      <string>launchctl</string>
      <string>limit</string>
      <string>maxfiles</string>
      <string>65536</string>
      <string>65536</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>ServiceIPC</key>
    <false/>
  </dict>
</plist>
EOF

sudo launchctl load -w /Library/LaunchDaemons/limit.maxfiles.plist

# Optimize Ollama for Apple Silicon
echo "Configuring Ollama for Apple Silicon..."
if [[ $(uname -m) == "arm64" ]]; then
    export OLLAMA_METAL=1
    export OLLAMA_METAL_EMBED_LIBRARY=1
    echo "export OLLAMA_METAL=1" >> ~/.zshrc
    echo "export OLLAMA_METAL_EMBED_LIBRARY=1" >> ~/.zshrc
fi

# Create optimized Ollama launch daemon
sudo tee /Library/LaunchDaemons/com.ollama.optimized.plist > /dev/null << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.ollama.optimized</string>
    <key>ProgramArguments</key>
    <array>
        <string>/opt/homebrew/bin/ollama</string>
        <string>serve</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>OLLAMA_MAX_MEMORY</key>
        <string>8192</string>
        <key>OLLAMA_METAL</key>
        <string>1</string>
        <key>OLLAMA_KEEP_ALIVE</key>
        <string>24h</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/var/log/ollama.log</string>
    <key>StandardErrorPath</key>
    <string>/var/log/ollama.error.log</string>
    <key>Nice</key>
    <integer>-10</integer>
</dict>
</plist>
EOF

# Load the optimized service
sudo launchctl load -w /Library/LaunchDaemons/com.ollama.optimized.plist

# Disable spotlight indexing for project directories (optional)
echo "Configuring Spotlight..."
sudo mdutil -i off /Users/$(whoami)/projects 2>/dev/null || true
sudo mdutil -i off /Users/$(whoami)/.abov3 2>/dev/null || true

# Restart dock to apply changes
killall Dock

echo "macOS optimization complete!"
echo "Some changes require a restart to take effect."
echo ""
echo "Performance monitoring commands:"
echo "  Activity Monitor - GUI system monitor"
echo "  sudo powermetrics -n 1 -i 1000 --samplers cpu_power,gpu_power - Power usage"
echo "  top -o cpu - Process CPU usage"
echo "  vm_stat 1 - Virtual memory statistics"
```

## Troubleshooting Performance Issues

### Common Performance Problems and Solutions

#### Problem 1: Slow Cold Start Times

**Symptoms:**
- First response takes 10+ seconds
- Models loading slowly
- High disk I/O during startup

**Solutions:**

```bash
# 1. Pre-load models
ollama pull gemma:2b
ollama run gemma:2b "warmup" --keep-alive=24h &

# 2. Move models to faster storage
export OLLAMA_MODELS_PATH="/path/to/nvme/drive"

# 3. Use smaller models for initial responses
# In ABOV3: /model switch gemma:2b

# 4. Enable model caching
export OLLAMA_MMAP=true
export OLLAMA_CACHE_MODELS=true
```

#### Problem 2: High Memory Usage

**Symptoms:**
- System running out of RAM
- Ollama crashes with OOM errors
- Slow system performance

**Solutions:**

```bash
# 1. Limit Ollama memory usage
export OLLAMA_MAX_MEMORY=4096  # 4GB limit

# 2. Use quantized models
ollama pull llama3:q4_k_m  # 4-bit quantization

# 3. Configure swap space
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 4. Monitor and cleanup
python3 -c "
import gc
import psutil
print('Memory usage:', psutil.virtual_memory().percent)
gc.collect()
print('After cleanup:', psutil.virtual_memory().percent)
"
```

#### Problem 3: Slow Response Times

**Symptoms:**
- Responses taking 30+ seconds
- High CPU usage
- System becomes unresponsive

**Solutions:**

```bash
# 1. Use faster models
ollama pull gemma:2b  # Fastest model

# 2. Reduce context size
export OLLAMA_CONTEXT_SIZE=2048

# 3. Optimize CPU usage
taskset -c 0-3 ollama serve  # Limit to first 4 cores
nice -n -10 ollama serve     # Higher priority

# 4. Enable GPU acceleration (if available)
export CUDA_VISIBLE_DEVICES=0
nvidia-smi  # Verify GPU usage
```

#### Problem 4: Network Latency Issues

**Symptoms:**
- Intermittent connection failures
- Timeout errors
- Inconsistent response times

**Solutions:**

```bash
# 1. Increase timeouts
export OLLAMA_TIMEOUT=300  # 5 minutes

# 2. Test local connection
curl -X GET http://localhost:11434/api/tags

# 3. Configure connection pooling
# In ABOV3 configuration:
export ABOV3_MAX_CONNECTIONS=10
export ABOV3_KEEPALIVE=true

# 4. Monitor network usage
nethogs  # Linux
netstat -i  # All platforms
```

### Performance Debugging Tools

#### System Performance Profiler

```python
# comprehensive_profiler.py
import time
import psutil
import threading
import json
from typing import Dict, List, Callable
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class ProfileSnapshot:
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_bytes_sent: int
    network_bytes_recv: int
    ollama_cpu: float
    ollama_memory_mb: float
    active_connections: int
    
class SystemProfiler:
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.snapshots: List[ProfileSnapshot] = []
        self.running = False
        self.thread = None
        self.previous_disk_io = None
        self.previous_network_io = None
    
    def start_profiling(self):
        """Start continuous profiling"""
        self.running = True
        self.thread = threading.Thread(target=self._profiling_loop, daemon=True)
        self.thread.start()
        print(f"Started profiling with {self.interval}s interval")
    
    def stop_profiling(self):
        """Stop profiling and return results"""
        self.running = False
        if self.thread:
            self.thread.join()
        print(f"Stopped profiling. Collected {len(self.snapshots)} snapshots")
        return self.snapshots
    
    def _profiling_loop(self):
        """Main profiling loop"""
        while self.running:
            try:
                snapshot = self._take_snapshot()
                self.snapshots.append(snapshot)
                time.sleep(self.interval)
            except Exception as e:
                print(f"Profiling error: {e}")
                time.sleep(self.interval)
    
    def _take_snapshot(self) -> ProfileSnapshot:
        """Take a single performance snapshot"""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_read_mb = 0
        disk_write_mb = 0
        
        if self.previous_disk_io:
            disk_read_mb = (disk_io.read_bytes - self.previous_disk_io.read_bytes) / (1024**2)
            disk_write_mb = (disk_io.write_bytes - self.previous_disk_io.write_bytes) / (1024**2)
        
        self.previous_disk_io = disk_io
        
        # Network I/O
        network_io = psutil.net_io_counters()
        
        # Ollama process stats
        ollama_stats = self._get_ollama_stats()
        
        # Active connections
        connections = len(psutil.net_connections(kind='inet'))
        
        return ProfileSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available_gb=memory.available / (1024**3),
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            network_bytes_sent=network_io.bytes_sent,
            network_bytes_recv=network_io.bytes_recv,
            ollama_cpu=ollama_stats["cpu"],
            ollama_memory_mb=ollama_stats["memory"],
            active_connections=connections
        )
    
    def _get_ollama_stats(self) -> Dict[str, float]:
        """Get Ollama process statistics"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
                if proc.info['name'] and 'ollama' in proc.info['name'].lower():
                    return {
                        "cpu": proc.info['cpu_percent'],
                        "memory": proc.info['memory_info'].rss / (1024**2)
                    }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        
        return {"cpu": 0, "memory": 0}
    
    def analyze_performance(self) -> Dict[str, any]:
        """Analyze collected performance data"""
        if not self.snapshots:
            return {}
        
        # Calculate statistics
        cpu_values = [s.cpu_percent for s in self.snapshots]
        memory_values = [s.memory_percent for s in self.snapshots]
        ollama_cpu_values = [s.ollama_cpu for s in self.snapshots if s.ollama_cpu > 0]
        
        analysis = {
            "duration_seconds": self.snapshots[-1].timestamp - self.snapshots[0].timestamp,
            "total_snapshots": len(self.snapshots),
            "cpu_stats": {
                "avg": sum(cpu_values) / len(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values),
                "spikes": len([c for c in cpu_values if c > 90])
            },
            "memory_stats": {
                "avg": sum(memory_values) / len(memory_values),
                "max": max(memory_values),
                "min": min(memory_values),
                "critical_usage": len([m for m in memory_values if m > 90])
            },
            "ollama_stats": {
                "avg_cpu": sum(ollama_cpu_values) / len(ollama_cpu_values) if ollama_cpu_values else 0,
                "max_cpu": max(ollama_cpu_values) if ollama_cpu_values else 0,
                "avg_memory": sum(s.ollama_memory_mb for s in self.snapshots) / len(self.snapshots)
            },
            "performance_issues": []
        }
        
        # Identify performance issues
        if analysis["cpu_stats"]["avg"] > 80:
            analysis["performance_issues"].append("High average CPU usage")
        
        if analysis["memory_stats"]["avg"] > 80:
            analysis["performance_issues"].append("High average memory usage")
        
        if analysis["cpu_stats"]["spikes"] > len(self.snapshots) * 0.1:
            analysis["performance_issues"].append("Frequent CPU spikes")
        
        return analysis
    
    def export_data(self, filename: str = "performance_profile.json"):
        """Export profiling data to JSON file"""
        data = {
            "metadata": {
                "start_time": datetime.fromtimestamp(self.snapshots[0].timestamp).isoformat(),
                "end_time": datetime.fromtimestamp(self.snapshots[-1].timestamp).isoformat(),
                "interval": self.interval,
                "total_snapshots": len(self.snapshots)
            },
            "snapshots": [asdict(snapshot) for snapshot in self.snapshots],
            "analysis": self.analyze_performance()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Exported performance data to {filename}")

# Usage example
if __name__ == "__main__":
    profiler = SystemProfiler(interval=0.5)  # 500ms intervals
    
    try:
        profiler.start_profiling()
        
        # Let it run for 60 seconds
        time.sleep(60)
        
        profiler.stop_profiling()
        analysis = profiler.analyze_performance()
        
        print("\nPerformance Analysis:")
        print(f"Duration: {analysis['duration_seconds']:.1f} seconds")
        print(f"Average CPU: {analysis['cpu_stats']['avg']:.1f}%")
        print(f"Average Memory: {analysis['memory_stats']['avg']:.1f}%")
        print(f"Ollama Average CPU: {analysis['ollama_stats']['avg_cpu']:.1f}%")
        
        if analysis['performance_issues']:
            print("\nPerformance Issues Detected:")
            for issue in analysis['performance_issues']:
                print(f"  - {issue}")
        
        profiler.export_data()
        
    except KeyboardInterrupt:
        profiler.stop_profiling()
        print("\nProfiling interrupted by user")
```

This comprehensive performance optimization guide provides you with everything needed to achieve optimal performance with ABOV3 Genesis. Remember that optimization is an iterative process - start with the most impactful changes for your specific use case and continue refining based on your performance monitoring results.

The key to successful optimization is continuous monitoring, measurement, and adjustment based on your specific hardware, workload, and performance requirements.

---

**Happy optimizing!** For additional help, check our [Troubleshooting Guide](TROUBLESHOOTING.md) or join our community discussions.