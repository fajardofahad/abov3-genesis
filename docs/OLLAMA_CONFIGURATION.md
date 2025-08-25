# ABOV3 Genesis - Ollama Configuration Guide

## Overview

This comprehensive guide covers everything you need to know about configuring and optimizing Ollama for use with ABOV3 Genesis. Proper configuration ensures optimal performance, model selection, and seamless integration with your AI coding assistant.

## Table of Contents

1. [Understanding Ollama](#understanding-ollama)
2. [Model Selection Guide](#model-selection-guide)
3. [Performance Configuration](#performance-configuration)
4. [Advanced Model Management](#advanced-model-management)
5. [Optimization Techniques](#optimization-techniques)
6. [Multi-Model Strategies](#multi-model-strategies)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

## Understanding Ollama

### What is Ollama?

Ollama is a lightweight, extensible framework for running large language models locally. It provides:

- **Local Execution**: Run AI models entirely on your machine
- **Model Management**: Easy installation and switching between models
- **API Access**: RESTful API for programmatic access
- **GPU Acceleration**: Support for NVIDIA and Apple Silicon GPUs
- **Memory Optimization**: Efficient memory usage and model loading

### Ollama Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ABOV3 Genesis │────│   Ollama API    │────│    AI Models    │
│                 │    │  (Port 11434)   │    │  (llama3, etc.) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   System RAM    │
                       │   GPU Memory    │
                       │   Disk Storage  │
                       └─────────────────┘
```

## Model Selection Guide

### Recommended Models for Different Tasks

#### 🚀 **General Purpose (Recommended)**

**llama3 (4.7GB)**
```bash
ollama pull llama3
```
- **Best for**: General coding, explanations, documentation
- **RAM Required**: 8GB
- **Performance**: Excellent balance of speed and quality
- **Use Cases**: Web development, scripting, debugging

**llama3.1 (4.7GB)**
```bash
ollama pull llama3.1
```
- **Best for**: Latest improvements, better reasoning
- **RAM Required**: 8GB
- **Performance**: Enhanced version of llama3
- **Use Cases**: Complex problem solving, architecture design

#### 💻 **Code Specialists**

**codellama (3.8GB)**
```bash
ollama pull codellama
```
- **Best for**: Code generation, completion
- **RAM Required**: 8GB
- **Specialization**: Programming tasks, syntax
- **Languages**: Python, JavaScript, Java, C++, etc.

**deepseek-coder (6.2GB)**
```bash
ollama pull deepseek-coder
```
- **Best for**: Complex applications, enterprise code
- **RAM Required**: 12GB
- **Performance**: Superior code quality
- **Use Cases**: Full-stack applications, system design

**qwen2-coder (4.2GB)**
```bash
ollama pull qwen2-coder
```
- **Best for**: Code debugging, analysis
- **RAM Required**: 8GB
- **Specialization**: Bug detection, optimization
- **Languages**: Multi-language support

#### ⚡ **Performance Optimized**

**gemma:2b (1.7GB)**
```bash
ollama pull gemma:2b
```
- **Best for**: Quick tasks, testing, low-resource systems
- **RAM Required**: 4GB
- **Speed**: Very fast responses
- **Use Cases**: Simple scripts, quick questions

**phi3 (2.3GB)**
```bash
ollama pull phi3
```
- **Best for**: Efficient reasoning, mobile deployment
- **RAM Required**: 6GB
- **Quality**: High quality despite small size
- **Use Cases**: Resource-constrained environments

#### 🌍 **Specialized Models**

**mistral (4.1GB)**
```bash
ollama pull mistral
```
- **Best for**: Multilingual projects, European languages
- **RAM Required**: 8GB
- **Languages**: English, French, German, Spanish, Italian
- **Use Cases**: International projects

**yi-coder (4.8GB)**
```bash
ollama pull yi-coder
```
- **Best for**: Chinese/English bilingual coding
- **RAM Required**: 10GB
- **Languages**: Mandarin, English
- **Use Cases**: Chinese market development

### Model Comparison Matrix

| Model | Size | RAM | Speed | Code Quality | Use Case |
|-------|------|-----|--------|--------------|----------|
| **gemma:2b** | 1.7GB | 4GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Quick tasks |
| **phi3** | 2.3GB | 6GB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Efficient coding |
| **codellama** | 3.8GB | 8GB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Code generation |
| **llama3** | 4.7GB | 8GB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | General purpose |
| **mistral** | 4.1GB | 8GB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Multilingual |
| **deepseek-coder** | 6.2GB | 12GB | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Complex apps |

## Performance Configuration

### System Requirements by Model

#### For 2B-4B Models (gemma:2b, phi3)
```yaml
Minimum Requirements:
  RAM: 4GB
  CPU: 2 cores
  Storage: 5GB free
  
Recommended:
  RAM: 8GB
  CPU: 4 cores
  SSD: 20GB free
```

#### For 7B-8B Models (llama3, codellama)
```yaml
Minimum Requirements:
  RAM: 8GB
  CPU: 4 cores  
  Storage: 10GB free
  
Recommended:
  RAM: 16GB
  CPU: 6+ cores
  GPU: 4GB VRAM
  SSD: 50GB free
```

#### For 13B+ Models (deepseek-coder)
```yaml
Minimum Requirements:
  RAM: 12GB
  CPU: 6 cores
  Storage: 15GB free
  
Recommended:
  RAM: 24GB
  CPU: 8+ cores
  GPU: 8GB+ VRAM
  SSD: 100GB free
```

### GPU Configuration

#### NVIDIA GPU Setup

```bash
# Check GPU compatibility
nvidia-smi

# Install CUDA toolkit (if not present)
# Windows: Download from nvidia.com
# Linux: 
sudo apt install nvidia-cuda-toolkit

# Configure Ollama for GPU usage
export CUDA_VISIBLE_DEVICES=0
ollama serve
```

#### Apple Silicon (M1/M2/M3)

```bash
# Ollama automatically uses Metal acceleration
# No additional configuration needed
ollama serve

# Verify Metal usage
ollama run llama3 "test" --verbose
```

#### AMD GPU (Linux only)

```bash
# Install ROCm
sudo apt install rocm-dev

# Set environment variables
export HIP_VISIBLE_DEVICES=0
export ROCM_PATH=/opt/rocm

ollama serve
```

### Memory Optimization

#### Ollama Memory Settings

```bash
# Set maximum memory usage (in MB)
export OLLAMA_MAX_MEMORY=8192

# Set context window size
export OLLAMA_CONTEXT_SIZE=4096

# Enable memory mapping
export OLLAMA_MMAP=true

# Configure batch size
export OLLAMA_BATCH_SIZE=512
```

#### System Memory Management

**Windows:**
```powershell
# Increase virtual memory
# Control Panel → System → Advanced → Performance Settings → Advanced → Virtual Memory
# Set to 1.5x your RAM size
```

**Linux:**
```bash
# Create swap file if needed
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Add to /etc/fstab for persistence
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

**macOS:**
```bash
# macOS handles virtual memory automatically
# Monitor with Activity Monitor
```

## Advanced Model Management

### Custom Model Creation

#### Creating Specialized Models

```bash
# Create a Modelfile
cat > Modelfile << EOF
FROM llama3

# Custom parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "```"

# Specialized system prompt
SYSTEM """You are a Python specialist AI assistant. You excel at:
- Writing clean, Pythonic code
- Following PEP 8 standards
- Creating comprehensive docstrings
- Implementing proper error handling
- Optimizing performance

Always provide complete, runnable code with explanations."""
EOF

# Create the custom model
ollama create python-specialist -f ./Modelfile
```

#### Model Variants

**Frontend Specialist:**
```dockerfile
FROM codellama
PARAMETER temperature 0.6
SYSTEM """You are a frontend development specialist focused on:
- React, Vue, Angular frameworks
- Modern CSS (Tailwind, CSS-in-JS)
- JavaScript ES6+ features
- Responsive design
- Performance optimization
- Accessibility best practices"""
```

**Backend Specialist:**
```dockerfile
FROM deepseek-coder
PARAMETER temperature 0.8
SYSTEM """You are a backend development expert specializing in:
- RESTful API design
- Database optimization
- Authentication & authorization
- Microservices architecture
- Cloud deployment
- Security best practices"""
```

### Model Quantization

#### Understanding Quantization

Different quantization levels offer trade-offs between quality and resource usage:

```bash
# Q2_K - Smallest, fastest, lower quality
ollama pull llama3:q2_k

# Q4_K_M - Good balance (recommended)
ollama pull llama3:q4_k_m

# Q8_0 - High quality, larger size
ollama pull llama3:q8_0

# No quantization (FP16) - Best quality, largest
ollama pull llama3:latest
```

#### Quantization Comparison

| Quantization | Size Reduction | Quality Loss | Speed Gain |
|--------------|----------------|--------------|------------|
| **Q2_K** | 75% | High | Very High |
| **Q4_K_M** | 50% | Low | High |
| **Q5_K_M** | 40% | Very Low | Medium |
| **Q8_0** | 25% | Minimal | Low |
| **FP16** | 0% | None | Baseline |

### Multi-Model Orchestration

#### Automatic Model Selection

Configure ABOV3 Genesis to automatically select models:

```yaml
# ~/.abov3/config.yaml
model_selection:
  strategy: "auto"
  
  task_models:
    code_generation: "deepseek-coder"
    debugging: "qwen2-coder" 
    documentation: "llama3"
    quick_tasks: "gemma:2b"
    frontend: "codellama"
    backend: "deepseek-coder"
  
  fallback_model: "llama3"
  
  performance_thresholds:
    response_time: 30  # seconds
    memory_usage: 80   # percentage
```

#### Model Ensemble

Use multiple models for complex tasks:

```python
# Example: Code review with multiple perspectives
models = [
    "deepseek-coder",  # Code quality
    "qwen2-coder",     # Bug detection  
    "llama3"           # Documentation
]

# Each model provides different insights
for model in models:
    result = await ollama_client.generate(model, prompt)
    combined_analysis.append(result)
```

## Optimization Techniques

### Performance Tuning

#### CPU Optimization

```bash
# Set CPU affinity (Linux)
taskset -c 0-3 ollama serve

# Adjust process priority
nice -n -10 ollama serve

# Configure CPU governor (Linux)
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

#### Memory Optimization

```bash
# Pre-load models to avoid cold starts
ollama pull llama3
ollama run llama3 "preload" --keep-alive=24h

# Configure memory limits
export OLLAMA_MAX_MEMORY=8192
export OLLAMA_SWAP_SIZE=4096
```

#### Disk I/O Optimization

**SSD Optimization:**
```bash
# Enable TRIM (Linux)
sudo systemctl enable fstrim.timer

# Adjust SSD scheduler
echo noop | sudo tee /sys/block/nvme0n1/queue/scheduler
```

**Model Storage Location:**
```bash
# Move models to fastest drive
export OLLAMA_MODELS_PATH="/path/to/fast/ssd"

# Create symlink if needed
ln -s /fast/ssd/models ~/.ollama/models
```

### Network Optimization

#### Local Network Configuration

```bash
# Increase API timeout
export OLLAMA_REQUEST_TIMEOUT=300

# Configure connection pooling
export OLLAMA_MAX_CONNECTIONS=10

# Enable keep-alive
export OLLAMA_KEEP_ALIVE=true
```

#### Remote Ollama Setup

```bash
# Configure remote Ollama server
export OLLAMA_HOST=192.168.1.100:11434

# Load balancing with multiple servers
export OLLAMA_HOSTS="server1:11434,server2:11434,server3:11434"
```

### Caching Strategies

#### Response Caching

```yaml
# ABOV3 caching configuration
caching:
  enabled: true
  ttl: 3600  # 1 hour
  max_size: 1000  # number of responses
  
  cache_policies:
    code_generation: 7200    # 2 hours
    documentation: 3600      # 1 hour  
    debugging: 1800          # 30 minutes
```

#### Context Caching

```bash
# Enable context caching in Ollama
export OLLAMA_CACHE_CONTEXT=true
export OLLAMA_CONTEXT_CACHE_SIZE=1000
```

## Multi-Model Strategies

### Task-Specific Model Routing

#### Configuration Example

```yaml
# models.yaml
routing_rules:
  - pattern: "create.*app|build.*application"
    model: "deepseek-coder"
    reason: "Complex application generation"
    
  - pattern: "fix.*bug|debug.*error"  
    model: "qwen2-coder"
    reason: "Debugging specialist"
    
  - pattern: "write.*test|create.*test"
    model: "codellama"
    reason: "Test generation"
    
  - pattern: "explain|document|comment"
    model: "llama3"
    reason: "Documentation and explanation"
    
  - pattern: "quick.*script|simple.*function"
    model: "gemma:2b"
    reason: "Fast, simple tasks"
```

### Load Balancing

#### Round-Robin Model Selection

```python
# Implement load balancing
class ModelLoadBalancer:
    def __init__(self):
        self.models = ["llama3", "codellama", "deepseek-coder"]
        self.current = 0
        
    def get_next_model(self):
        model = self.models[self.current]
        self.current = (self.current + 1) % len(self.models)
        return model
```

#### Weighted Model Selection

```python
# Weight models by performance
model_weights = {
    "gemma:2b": 0.4,      # Fast for quick tasks
    "llama3": 0.3,        # General purpose
    "codellama": 0.2,     # Code focused
    "deepseek-coder": 0.1 # Complex tasks only
}
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: Model Loading Fails

**Symptoms:**
- "Model not found" error
- Slow loading times
- Out of memory errors

**Solutions:**
```bash
# Check available models
ollama list

# Verify model integrity
ollama show llama3

# Re-download corrupted models
ollama rm llama3
ollama pull llama3

# Check system resources
free -h     # Linux/macOS
wmic OS get TotalVisibleMemorySize,FreePhysicalMemory  # Windows
```

#### Issue: Slow Response Times

**Symptoms:**
- Responses taking > 30 seconds
- High CPU/memory usage
- System freezing

**Solutions:**
```bash
# Switch to smaller model
ollama pull gemma:2b

# Reduce context size
export OLLAMA_CONTEXT_SIZE=2048

# Enable GPU acceleration
export CUDA_VISIBLE_DEVICES=0

# Check system load
htop    # Linux/macOS  
taskmgr # Windows
```

#### Issue: API Connection Problems

**Symptoms:**
- "Connection refused" errors
- Timeout errors
- Inconsistent responses

**Solutions:**
```bash
# Check Ollama service status
ps aux | grep ollama          # Linux/macOS
Get-Process -Name "ollama"    # Windows

# Restart Ollama service
killall ollama && ollama serve  # Linux/macOS
taskkill /F /IM ollama.exe && ollama serve  # Windows

# Verify port availability
netstat -an | grep 11434
```

### Diagnostic Commands

#### Health Check Script

```bash
#!/bin/bash
# ollama_health_check.sh

echo "=== Ollama Health Check ==="

# Check service status
echo "1. Service Status:"
if pgrep -x "ollama" > /dev/null; then
    echo "   ✓ Ollama service is running"
else
    echo "   ✗ Ollama service is not running"
fi

# Check API availability
echo "2. API Availability:"
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "   ✓ API is responding"
else
    echo "   ✗ API is not responding"
fi

# Check models
echo "3. Available Models:"
ollama list | tail -n +2 | while read model size date; do
    echo "   • $model ($size)"
done

# Check system resources
echo "4. System Resources:"
echo "   Memory: $(free -h | grep '^Mem:' | awk '{print $3 "/" $2}')"
echo "   Disk: $(df -h ~/.ollama 2>/dev/null | tail -1 | awk '{print $3 "/" $2 " (" $5 " used)"}')"

# Test model response
echo "5. Model Test:"
if echo "Hello" | ollama run llama3 > /dev/null 2>&1; then
    echo "   ✓ Model responds correctly"
else
    echo "   ✗ Model response failed"
fi

echo "=== Health Check Complete ==="
```

#### Performance Monitoring

```python
# performance_monitor.py
import time
import psutil
import requests

class OllamaMonitor:
    def __init__(self):
        self.base_url = "http://localhost:11434"
        
    def monitor_response_time(self, model, prompt):
        start_time = time.time()
        
        response = requests.post(f"{self.base_url}/api/generate", json={
            "model": model,
            "prompt": prompt,
            "stream": False
        })
        
        end_time = time.time()
        response_time = end_time - start_time
        
        return {
            "response_time": response_time,
            "status_code": response.status_code,
            "memory_usage": psutil.virtual_memory().percent,
            "cpu_usage": psutil.cpu_percent()
        }

# Usage
monitor = OllamaMonitor()
stats = monitor.monitor_response_time("llama3", "Hello world")
print(f"Response time: {stats['response_time']:.2f}s")
```

## Best Practices

### Model Selection Guidelines

1. **Start with llama3** for general development
2. **Use gemma:2b** for quick tasks and testing
3. **Switch to deepseek-coder** for complex applications
4. **Use qwen2-coder** specifically for debugging
5. **Consider codellama** for pure code generation

### Performance Optimization

1. **Keep models loaded** with `--keep-alive` flag
2. **Use appropriate quantization** levels
3. **Monitor system resources** regularly
4. **Implement caching** for repeated requests
5. **Optimize context size** for your use case

### Resource Management

1. **Monitor memory usage** and set limits
2. **Use SSD storage** for model storage
3. **Configure appropriate swap** space
4. **Clean up unused models** periodically
5. **Implement graceful degradation**

### Security Considerations

1. **Restrict network access** to Ollama API
2. **Use authentication** for remote access
3. **Monitor API usage** and set rate limits
4. **Keep models updated** for security patches
5. **Implement input validation**

### Maintenance Tasks

#### Weekly Maintenance

```bash
# Update models
ollama pull llama3
ollama pull codellama

# Clean up old conversations
rm -rf ~/.ollama/conversations/*

# Check disk usage
du -sh ~/.ollama/
```

#### Monthly Maintenance

```bash
# Full health check
./ollama_health_check.sh

# Performance benchmark
python performance_benchmark.py

# Update Ollama
# Download latest version from ollama.ai
```

## Integration with ABOV3 Genesis

### Automatic Configuration

ABOV3 Genesis automatically detects and configures Ollama models:

```python
# ABOV3 automatically:
# 1. Detects available models
# 2. Selects optimal model for task
# 3. Manages context and memory
# 4. Handles errors and fallbacks
# 5. Optimizes performance
```

### Manual Override

You can manually configure model selection in ABOV3:

```bash
# In ABOV3 Genesis
/model switch deepseek-coder  # Switch to specific model
/model list                   # Show available models
/model help                   # Show model commands
```

This comprehensive configuration guide ensures you get the best performance from Ollama with ABOV3 Genesis. Remember that optimal configuration depends on your specific hardware, use cases, and performance requirements.

---

**Need Help?** Check our [Troubleshooting Guide](TROUBLESHOOTING.md) or join our community for support!