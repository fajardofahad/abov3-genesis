# ABOV3 Genesis Troubleshooting Guide

## Overview

This comprehensive troubleshooting guide helps you diagnose and resolve common issues with ABOV3 Genesis. Issues are organized by category with step-by-step solutions, diagnostic commands, and prevention strategies.

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Installation Issues](#installation-issues)
3. [Ollama Connection Problems](#ollama-connection-problems)
4. [Model-Related Issues](#model-related-issues)
5. [Performance Problems](#performance-problems)
6. [Project and Session Issues](#project-and-session-issues)
7. [Code Generation Problems](#code-generation-problems)
8. [System Resource Issues](#system-resource-issues)
9. [Error Messages Guide](#error-messages-guide)
10. [Advanced Troubleshooting](#advanced-troubleshooting)

## Quick Diagnostics

### Health Check Commands

Run these commands to quickly assess system health:

```bash
# 1. Check ABOV3 Genesis installation
abov3 --version

# 2. Check Ollama status
ollama --version
curl -s http://localhost:11434/api/tags

# 3. Check available models
ollama list

# 4. Test basic functionality
echo "Hello" | ollama run llama3

# 5. Check system resources
# Linux/macOS:
free -h && df -h
# Windows:
systeminfo | findstr "Available Physical Memory"
```

### System Information Collection

When reporting issues, collect this information:

```bash
# Create diagnostic report
cat > diagnostic_report.txt << EOF
=== ABOV3 Genesis Diagnostic Report ===
Date: $(date)

System Information:
- OS: $(uname -a 2>/dev/null || systeminfo | head -5)
- Python: $(python3 --version)
- ABOV3: $(abov3 --version)
- Ollama: $(ollama --version)

Available Models:
$(ollama list)

System Resources:
$(free -h 2>/dev/null || systeminfo | grep Memory)

Recent Errors:
$(tail -50 ~/.abov3/logs/error.log 2>/dev/null || echo "No error log found")
EOF

cat diagnostic_report.txt
```

## Installation Issues

### Issue: Python Not Found

**Symptoms:**
```
'python3' is not recognized as an internal or external command
python: command not found
```

**Solutions:**

**Windows:**
```powershell
# Install Python from python.org or use winget
winget install Python.Python.3.11

# Add to PATH manually
$env:PATH += ";$env:LOCALAPPDATA\Programs\Python\Python311;$env:LOCALAPPDATA\Programs\Python\Python311\Scripts"

# Restart terminal and verify
python3 --version
```

**macOS:**
```bash
# Install via Homebrew
brew install python@3.11

# Or download from python.org
# Add to shell profile
echo 'export PATH="/usr/local/opt/python@3.11/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install python3 python3-pip -y

# RHEL/CentOS
sudo yum install python3 python3-pip -y

# Arch Linux
sudo pacman -S python python-pip
```

### Issue: Pip Installation Fails

**Symptoms:**
```
pip: command not found
Permission denied when installing packages
```

**Solutions:**

```bash
# Install pip if missing
python3 -m ensurepip --upgrade

# Use virtual environment (recommended)
python3 -m venv abov3-env
source abov3-env/bin/activate  # Linux/macOS
abov3-env\Scripts\activate     # Windows

# Install ABOV3 Genesis
pip install abov3-genesis

# Or use user installation
pip install --user abov3-genesis
```

### Issue: Module Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'abov3'
ImportError: attempted relative import with no known parent package
```

**Solutions:**

```bash
# Check installation
pip list | grep abov3

# Reinstall ABOV3 Genesis
pip uninstall abov3-genesis
pip install abov3-genesis

# Check Python path
python3 -c "import sys; print('\n'.join(sys.path))"

# Install in development mode if using source
cd abov3-genesis
pip install -e .
```

## Ollama Connection Problems

### Issue: Ollama Not Running

**Symptoms:**
```
Connection refused on localhost:11434
Failed to connect to Ollama server
```

**Solutions:**

```bash
# Check if Ollama is running
# Linux/macOS:
ps aux | grep ollama
pgrep ollama

# Windows:
tasklist | findstr ollama
Get-Process -Name "ollama" -ErrorAction SilentlyContinue

# Start Ollama if not running
ollama serve

# Or start as background service
# Linux (systemd):
sudo systemctl start ollama
sudo systemctl enable ollama

# macOS (launchd):
brew services start ollama

# Windows (as service):
# Install Ollama from ollama.ai (includes service)
```

### Issue: Ollama API Not Responding

**Symptoms:**
```
Timeout waiting for response
API endpoint not available
```

**Solutions:**

```bash
# Test API directly
curl -X GET http://localhost:11434/api/tags

# Check port availability
# Linux/macOS:
netstat -tlnp | grep 11434
lsof -i :11434

# Windows:
netstat -an | findstr :11434
netsh int ipv4 show tcpconnections

# Check firewall settings
# Linux (ufw):
sudo ufw allow 11434/tcp

# Windows:
# Go to Windows Firewall → Allow app → Add port 11434

# macOS:
# System Preferences → Security & Privacy → Firewall → Options
```

### Issue: Ollama Wrong Version

**Symptoms:**
```
Incompatible Ollama version
API version mismatch
```

**Solutions:**

```bash
# Check current version
ollama --version

# Update Ollama
# Download latest from https://ollama.ai

# Linux/macOS update script:
curl -fsSL https://ollama.ai/install.sh | sh

# Verify update
ollama --version
```

## Model-Related Issues

### Issue: Model Not Found

**Symptoms:**
```
Model 'llama3' not found
No such model available
```

**Solutions:**

```bash
# List available models locally
ollama list

# Pull missing model
ollama pull llama3

# Check model availability online
curl -s https://ollama.ai/api/models | jq '.models[].name'

# Pull specific version
ollama pull llama3:latest
ollama pull llama3:8b
```

### Issue: Model Download Fails

**Symptoms:**
```
Failed to download model
Network timeout during download
Incomplete model download
```

**Solutions:**

```bash
# Check internet connection
ping ollama.ai

# Check disk space
df -h  # Linux/macOS
dir C:\ # Windows

# Resume interrupted download
ollama pull llama3 --resume

# Use alternative download location
export OLLAMA_MODELS_PATH="/path/to/models"
ollama pull llama3

# Manual download (if needed)
# Download from alternative source and import
ollama import llama3 /path/to/model/file
```

### Issue: Model Loading Slow/Fails

**Symptoms:**
```
Model takes too long to load
Out of memory when loading model
Model crashes during initialization
```

**Solutions:**

```bash
# Check available memory
free -h  # Linux/macOS
wmic OS get TotalVisibleMemorySize,FreePhysicalMemory  # Windows

# Use smaller model
ollama pull gemma:2b
ollama pull phi3

# Adjust memory settings
export OLLAMA_MAX_MEMORY=4096  # MB
ollama serve

# Enable memory mapping
export OLLAMA_MMAP=true
ollama serve

# Use quantized models
ollama pull llama3:q4_k_m  # Smaller, faster
```

## Performance Problems

### Issue: Slow Response Times

**Symptoms:**
- Responses taking > 30 seconds
- High CPU usage during requests
- System becomes unresponsive

**Solutions:**

```bash
# 1. Switch to faster model
ollama pull gemma:2b
# In ABOV3: /model switch gemma:2b

# 2. Reduce context size
export OLLAMA_CONTEXT_SIZE=2048
export OLLAMA_BATCH_SIZE=256

# 3. Enable GPU acceleration (if available)
# Check GPU availability:
nvidia-smi  # NVIDIA
system_profiler SPDisplaysDataType | grep Chipset  # macOS

# Configure GPU usage:
export CUDA_VISIBLE_DEVICES=0  # NVIDIA
# Apple Silicon automatically uses Metal

# 4. Optimize system performance
# Linux: Set CPU governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# macOS: Reset SMC
sudo pmset -a standby 0
sudo pmset -a hibernatemode 0

# Windows: Set high performance mode
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
```

### Issue: High Memory Usage

**Symptoms:**
- System runs out of memory
- Ollama crashes with OOM error
- Other applications become slow

**Solutions:**

```bash
# 1. Monitor memory usage
htop  # Linux/macOS
taskmgr  # Windows

# 2. Set memory limits
export OLLAMA_MAX_MEMORY=6144  # 6GB limit
export OLLAMA_SWAP_SIZE=2048   # 2GB swap

# 3. Use memory-efficient models
ollama pull gemma:2b     # 1.7GB
ollama pull phi3         # 2.3GB
ollama pull llama3:q4_k  # Quantized version

# 4. Clear model cache periodically
ollama stop
rm -rf ~/.ollama/cache/*  # Linux/macOS
rmdir /s %USERPROFILE%\.ollama\cache  # Windows
ollama serve

# 5. Configure swap (Linux)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## Project and Session Issues

### Issue: Project Not Loading

**Symptoms:**
```
Failed to initialize project
Project directory not found
Invalid project structure
```

**Solutions:**

```bash
# 1. Check project path
ls -la /path/to/project
ls -la /path/to/project/.abov3

# 2. Verify permissions
# Linux/macOS:
ls -la ~/.abov3
chmod -R 755 ~/.abov3
chown -R $USER:$USER ~/.abov3

# Windows:
icacls %USERPROFILE%\.abov3 /grant %USERNAME%:F /T

# 3. Recreate project structure
mkdir -p /path/to/project/.abov3/{agents,sessions,history,genesis_flow,tasks,permissions,dependencies}

# 4. Check project configuration
cat /path/to/project/.abov3/project.yaml
```

### Issue: Session Recovery Failed

**Symptoms:**
```
Could not restore previous session
Session data corrupted
Missing conversation history
```

**Solutions:**

```bash
# 1. Check session files
ls -la ~/.abov3/sessions/
ls -la /project/.abov3/sessions/

# 2. Clear corrupted sessions
rm -rf ~/.abov3/sessions/current.session
rm -rf /project/.abov3/sessions/corrupted.session

# 3. Backup and reset session data
cp -r ~/.abov3/sessions ~/.abov3/sessions.backup
rm -rf ~/.abov3/sessions/*

# 4. Start fresh session
abov3 --new

# 5. Verify session manager
python3 -c "
from abov3.session.manager import SessionManager
from pathlib import Path
sm = SessionManager(Path('.'))
print('Session manager initialized successfully')
"
```

### Issue: Agent Switch Failed

**Symptoms:**
```
Agent not found
Failed to switch to agent
Invalid agent configuration
```

**Solutions:**

```bash
# 1. List available agents
# In ABOV3: /agents list

# 2. Check agent files
ls -la ~/.abov3/agents/
cat ~/.abov3/agents/current.yaml

# 3. Recreate default agents
# In ABOV3: This will recreate Genesis agents
/agents list  # Triggers creation if missing

# 4. Create custom agent
# In ABOV3:
/agents create my-agent llama3 "Description" "System prompt"

# 5. Reset agent configuration
rm -rf ~/.abov3/agents/*
# Restart ABOV3 to recreate defaults
```

## Code Generation Problems

### Issue: Poor Code Quality

**Symptoms:**
- Generated code has syntax errors
- Code doesn't follow best practices
- Missing imports or dependencies

**Solutions:**

```bash
# 1. Switch to code-specialized model
# In ABOV3: /model switch deepseek-coder

# 2. Provide more specific prompts
# Instead of: "create an app"
# Use: "create a React todo app with TypeScript, Redux state management, and responsive design"

# 3. Request code review
# "Review this code and fix any issues: [paste code]"

# 4. Ask for best practices
# "Rewrite this code following Python best practices and PEP 8"

# 5. Enable project intelligence
# Ensure project analysis runs for context
```

### Issue: Incomplete Code Generation

**Symptoms:**
- Code stops mid-generation
- Missing function implementations
- Truncated responses

**Solutions:**

```bash
# 1. Increase context size
export OLLAMA_CONTEXT_SIZE=8192  # Increase context window

# 2. Request continuation
# "Continue the previous code where it left off"
# "Complete the implementation of the function above"

# 3. Break down requests
# Instead of: "create entire application"
# Use: "create the user authentication module"

# 4. Use streaming responses
# Check if streaming is enabled in settings

# 5. Increase timeout
export OLLAMA_REQUEST_TIMEOUT=300  # 5 minutes
```

## System Resource Issues

### Issue: Disk Space Full

**Symptoms:**
```
No space left on device
Disk full error
Cannot write files
```

**Solutions:**

```bash
# 1. Check disk usage
df -h  # Linux/macOS
dir C:\ # Windows

# 2. Clean Ollama models
ollama list
ollama rm unused-model-name

# 3. Clear ABOV3 cache
rm -rf ~/.abov3/cache/*
rm -rf ~/.abov3/logs/*.log.old

# 4. Clean system temp files
# Linux/macOS:
sudo rm -rf /tmp/ollama*
sudo rm -rf /var/tmp/abov3*

# Windows:
del /Q /S %TEMP%\ollama*
del /Q /S %TEMP%\abov3*

# 5. Move models to different drive
export OLLAMA_MODELS_PATH="/path/to/larger/drive"
ollama list  # Models will be moved automatically
```

### Issue: CPU Overheating

**Symptoms:**
- System becomes very hot
- CPU throttling occurs
- Performance degrades over time

**Solutions:**

```bash
# 1. Monitor temperatures
# Linux:
sensors
cat /proc/cpuinfo | grep MHz

# macOS:
sudo powermetrics -n 1 -i 1000 | grep -i temp

# 2. Limit CPU usage
# Linux:
cpulimit -l 50 -p $(pgrep ollama)  # Limit to 50% CPU

# 3. Use smaller models
ollama pull gemma:2b  # Less CPU intensive

# 4. Adjust process priority
# Linux/macOS:
nice -n 10 ollama serve  # Lower priority

# Windows:
# Task Manager → Process → Set Priority → Below Normal

# 5. Improve cooling
# Clean computer fans and vents
# Ensure good airflow
# Consider laptop cooling pad
```

## Error Messages Guide

### Common Error Messages and Solutions

#### `ConnectionError: Failed to connect to Ollama`

**Cause:** Ollama service not running or not accessible

**Solution:**
```bash
# Check if Ollama is running
ps aux | grep ollama  # Linux/macOS
tasklist | findstr ollama  # Windows

# Start Ollama
ollama serve

# Check port binding
netstat -tlnp | grep 11434
```

#### `ModelNotFoundError: Model 'llama3' not found`

**Cause:** Requested model not installed

**Solution:**
```bash
# Pull the model
ollama pull llama3

# Verify installation
ollama list

# Check available models online
curl -s https://ollama.ai/api/models
```

#### `OutOfMemoryError: Cannot allocate memory`

**Cause:** Insufficient RAM for model

**Solution:**
```bash
# Check available memory
free -h

# Use smaller model
ollama pull gemma:2b

# Add swap space (Linux)
sudo fallocate -l 4G /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### `TimeoutError: Request timed out`

**Cause:** Model taking too long to respond

**Solution:**
```bash
# Increase timeout
export OLLAMA_REQUEST_TIMEOUT=300

# Use faster model
ollama pull gemma:2b

# Check system load
htop  # Linux/macOS
taskmgr  # Windows
```

#### `PermissionError: Access denied`

**Cause:** Insufficient file permissions

**Solution:**
```bash
# Fix permissions (Linux/macOS)
chmod -R 755 ~/.abov3
chown -R $USER:$USER ~/.abov3

# Windows: Run as administrator
# Right-click → "Run as administrator"
```

#### `ValidationError: Invalid configuration`

**Cause:** Corrupted configuration files

**Solution:**
```bash
# Backup and reset configuration
cp ~/.abov3/config.yaml ~/.abov3/config.yaml.backup
rm ~/.abov3/config.yaml

# Restart ABOV3 to recreate defaults
abov3
```

## Advanced Troubleshooting

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
# Enable debug mode
export ABOV3_DEBUG=true
export OLLAMA_DEBUG=true

# Run with verbose output
abov3 --verbose

# Check debug logs
tail -f ~/.abov3/logs/debug.log
```

### Network Troubleshooting

```bash
# Test network connectivity
ping ollama.ai
curl -I https://ollama.ai

# Check DNS resolution
nslookup ollama.ai
dig ollama.ai

# Test local API
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3","prompt":"test","stream":false}'
```

### Process Monitoring

```bash
# Monitor system processes
top -p $(pgrep ollama)  # Linux
htop -p $(pgrep ollama)  # Linux/macOS

# Windows Task Manager
tasklist /FI "IMAGENAME eq ollama.exe"

# Monitor file descriptors (Linux/macOS)
lsof -p $(pgrep ollama)

# Check memory maps
cat /proc/$(pgrep ollama)/smaps  # Linux
```

### Log Analysis

```bash
# ABOV3 Genesis logs
tail -f ~/.abov3/logs/error.log
tail -f ~/.abov3/logs/debug.log

# Ollama logs (Linux systemd)
journalctl -f -u ollama

# System logs
# Linux: /var/log/syslog
# macOS: /var/log/system.log
# Windows: Event Viewer
```

### Performance Profiling

```python
# Create performance test script
import time
import psutil
import requests
import json

def profile_ollama_request(model, prompt):
    start_time = time.time()
    start_memory = psutil.virtual_memory().used
    
    response = requests.post("http://localhost:11434/api/generate", 
                           json={"model": model, "prompt": prompt, "stream": False})
    
    end_time = time.time()
    end_memory = psutil.virtual_memory().used
    
    return {
        "response_time": end_time - start_time,
        "memory_delta": end_memory - start_memory,
        "status_code": response.status_code,
        "cpu_percent": psutil.cpu_percent()
    }

# Test different models
models = ["gemma:2b", "llama3", "deepseek-coder"]
prompt = "Write a simple Python function"

for model in models:
    try:
        stats = profile_ollama_request(model, prompt)
        print(f"{model}: {stats}")
    except Exception as e:
        print(f"{model}: Error - {e}")
```

### Recovery Procedures

#### Complete Reset

If all else fails, perform a complete reset:

```bash
# 1. Backup important data
mkdir ~/abov3-backup
cp -r ~/.abov3 ~/abov3-backup/
cp -r ~/projects ~/abov3-backup/

# 2. Stop all processes
pkill ollama  # Linux/macOS
taskkill /F /IM ollama.exe  # Windows

# 3. Remove all ABOV3 data
rm -rf ~/.abov3

# 4. Reinstall ABOV3 Genesis
pip uninstall abov3-genesis
pip install abov3-genesis

# 5. Reinstall Ollama
# Download fresh installer from ollama.ai

# 6. Pull fresh models
ollama pull llama3
ollama pull codellama

# 7. Test basic functionality
abov3 --version
echo "test" | ollama run llama3
```

### Getting Help

When reporting issues, include:

1. **System Information**
   ```bash
   # Run and include output
   uname -a  # Linux/macOS
   systeminfo  # Windows
   python3 --version
   abov3 --version
   ollama --version
   ```

2. **Error Logs**
   ```bash
   # Include recent logs
   tail -50 ~/.abov3/logs/error.log
   tail -50 ~/.abov3/logs/debug.log
   ```

3. **Configuration Files**
   ```bash
   # Sanitize and include
   cat ~/.abov3/config.yaml
   cat ~/.abov3/agents/current.yaml
   ```

4. **Steps to Reproduce**
   - Exact commands that cause the issue
   - Expected vs. actual behavior
   - Frequency of the issue

5. **Environment Details**
   - Hardware specifications
   - Network configuration
   - Other running software

### Support Channels

- **GitHub Issues**: https://github.com/fajardofahad/abov3-genesis/issues
- **Documentation**: Check `/docs` folder
- **Community**: Discord/Reddit (links in README)
- **Email**: support@abov3genesis.com

---

**Remember:** Most issues can be resolved by ensuring Ollama is running, models are properly installed, and sufficient system resources are available. When in doubt, try the complete reset procedure above.