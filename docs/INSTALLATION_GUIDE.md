# ABOV3 Genesis Installation Guide

## Overview

This guide provides comprehensive installation instructions for ABOV3 Genesis on different platforms. ABOV3 Genesis is designed to work locally with Ollama AI models, providing complete privacy and control over your AI coding assistant.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Installation](#quick-installation)
3. [Platform-Specific Installation](#platform-specific-installation)
4. [Ollama Setup](#ollama-setup)
5. [Verification and Testing](#verification-and-testing)
6. [Post-Installation Configuration](#post-installation-configuration)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Installation](#advanced-installation)

## System Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| **Operating System** | Windows 10+, macOS 10.15+, Ubuntu 18.04+ |
| **Python** | Python 3.8 or higher |
| **RAM** | 8GB (16GB recommended for large models) |
| **Storage** | 10GB free space (20GB+ for multiple models) |
| **Internet** | Required for initial setup and model downloads |

### Recommended Requirements

| Component | Recommendation |
|-----------|----------------|
| **RAM** | 16GB or higher |
| **GPU** | NVIDIA GPU with 4GB+ VRAM (optional but recommended) |
| **CPU** | 4+ cores, 2.5GHz+ |
| **Storage** | SSD with 50GB+ free space |

### GPU Support

ABOV3 Genesis supports GPU acceleration through Ollama:
- **NVIDIA GPUs**: CUDA support (GTX 1060+, RTX series recommended)
- **Apple Silicon**: Native Metal acceleration (M1, M2, M3)
- **AMD GPUs**: Limited support through ROCm (Linux only)

## Quick Installation

For users who want to get started quickly:

### 1. Install Ollama

Visit [https://ollama.ai](https://ollama.ai) and download the installer for your platform.

### 2. Pull an AI Model

```bash
ollama pull llama3
```

### 3. Install ABOV3 Genesis

```bash
pip install abov3-genesis
```

### 4. Start ABOV3

```bash
abov3
```

That's it! You should see the ABOV3 Genesis interface.

## Platform-Specific Installation

### Windows Installation

#### Option 1: Using Installer (Recommended)

1. **Download ABOV3 Genesis Installer**
   ```
   Download from: https://github.com/fajardofahad/abov3-genesis/releases
   File: abov3-genesis-windows-installer.exe
   ```

2. **Run Installer**
   - Right-click installer → "Run as administrator"
   - Follow installation wizard
   - Choose installation directory (default: `C:\Program Files\ABOV3 Genesis`)

3. **Install Ollama**
   - Download from: https://ollama.ai/download/windows
   - Run `OllamaSetup.exe`
   - Follow installation prompts

#### Option 2: Manual Installation

1. **Install Python 3.8+**
   ```powershell
   # Download from python.org or use winget
   winget install Python.Python.3.11
   ```

2. **Install Git (Optional but recommended)**
   ```powershell
   winget install Git.Git
   ```

3. **Install ABOV3 Genesis**
   ```powershell
   # Open PowerShell as Administrator
   pip install abov3-genesis
   
   # Or install from source
   git clone https://github.com/fajardofahad/abov3-genesis.git
   cd abov3-genesis
   pip install -e .
   ```

4. **Install Ollama**
   ```powershell
   # Download and install from https://ollama.ai
   # Or use chocolatey
   choco install ollama
   ```

5. **Create Desktop Shortcut**
   ```batch
   # Create abov3.bat in user directory
   @echo off
   cd /d "%USERPROFILE%\Documents"
   abov3
   pause
   ```

#### Windows-Specific Configuration

1. **Add to PATH**
   ```powershell
   # Add Python Scripts to PATH if not already added
   $env:PATH += ";$env:LOCALAPPDATA\Programs\Python\Python311\Scripts"
   ```

2. **Configure Windows Defender**
   ```powershell
   # Add exclusion for ABOV3 directory
   Add-MpPreference -ExclusionPath "C:\Users\%USERNAME%\.abov3"
   Add-MpPreference -ExclusionPath "C:\Users\%USERNAME%\projects"
   ```

### macOS Installation

#### Option 1: Using Homebrew (Recommended)

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.8+
brew install python@3.11

# Install Ollama
brew install ollama

# Install ABOV3 Genesis
pip3 install abov3-genesis
```

#### Option 2: Manual Installation

1. **Install Python 3.8+**
   ```bash
   # Download from python.org or use pyenv
   pyenv install 3.11.0
   pyenv global 3.11.0
   ```

2. **Install Ollama**
   ```bash
   # Download from https://ollama.ai/download/mac
   # Or install manually
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

3. **Install ABOV3 Genesis**
   ```bash
   pip3 install abov3-genesis
   
   # Or from source
   git clone https://github.com/fajardofahad/abov3-genesis.git
   cd abov3-genesis
   pip3 install -e .
   ```

#### macOS-Specific Configuration

1. **Configure Terminal**
   ```bash
   # Add to ~/.zshrc or ~/.bash_profile
   echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc
   echo 'alias abov3-dev="cd ~/projects && abov3"' >> ~/.zshrc
   ```

2. **Grant Permissions**
   ```bash
   # Allow Ollama to run
   xattr -rd com.apple.quarantine /usr/local/bin/ollama
   ```

### Linux Installation

#### Ubuntu/Debian

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.8+
sudo apt install python3 python3-pip python3-venv -y

# Install Git
sudo apt install git -y

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Install ABOV3 Genesis
pip3 install abov3-genesis

# Or from source
git clone https://github.com/fajardofahad/abov3-genesis.git
cd abov3-genesis
pip3 install -e .
```

#### RHEL/CentOS/Fedora

```bash
# RHEL/CentOS
sudo yum update -y
sudo yum install python3 python3-pip git -y

# Fedora
sudo dnf update -y
sudo dnf install python3 python3-pip git -y

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Install ABOV3 Genesis
pip3 install abov3-genesis
```

#### Arch Linux

```bash
# Update system
sudo pacman -Syu

# Install dependencies
sudo pacman -S python python-pip git

# Install Ollama (AUR)
yay -S ollama

# Install ABOV3 Genesis
pip install abov3-genesis
```

#### Linux-Specific Configuration

1. **Configure Systemd Service (Optional)**
   ```bash
   # Create service file
   sudo tee /etc/systemd/system/ollama.service > /dev/null <<EOF
   [Unit]
   Description=Ollama Service
   After=network.target
   
   [Service]
   Type=simple
   User=ollama
   Group=ollama
   ExecStart=/usr/local/bin/ollama serve
   Restart=always
   RestartSec=3
   
   [Install]
   WantedBy=multi-user.target
   EOF
   
   # Enable and start service
   sudo systemctl enable ollama
   sudo systemctl start ollama
   ```

2. **Configure Firewall**
   ```bash
   # Ubuntu/Debian (UFW)
   sudo ufw allow 11434/tcp
   
   # RHEL/CentOS (Firewalld)
   sudo firewall-cmd --permanent --add-port=11434/tcp
   sudo firewall-cmd --reload
   ```

## Ollama Setup

### Installing Ollama Models

After installing Ollama, you need to download AI models:

#### Recommended Models

```bash
# General purpose coding (recommended for beginners)
ollama pull llama3

# Specialized code generation
ollama pull codellama

# Advanced coding model
ollama pull deepseek-coder

# Fast and efficient
ollama pull gemma

# Multilingual support
ollama pull mistral

# Code debugging specialist
ollama pull qwen2-coder
```

#### Model Size Guide

| Model | Size | RAM Required | Use Case |
|-------|------|--------------|----------|
| **gemma:2b** | 1.7GB | 4GB | Quick tasks, testing |
| **llama3** | 4.7GB | 8GB | General coding (recommended) |
| **codellama** | 3.8GB | 8GB | Code generation |
| **deepseek-coder** | 6.2GB | 12GB | Complex applications |
| **mistral** | 4.1GB | 8GB | Multilingual projects |

#### Custom Model Configuration

```bash
# Create custom model with specific parameters
ollama create mymodel -f ./Modelfile

# Example Modelfile
cat > Modelfile << EOF
FROM llama3
PARAMETER temperature 0.7
PARAMETER top_p 0.9
SYSTEM You are a specialized coding assistant focused on Python development.
EOF

ollama create python-specialist -f ./Modelfile
```

### Starting Ollama Service

#### Windows
```powershell
# Ollama runs automatically as a service after installation
# To check status:
Get-Service -Name "Ollama"

# To start manually:
ollama serve
```

#### macOS
```bash
# Start Ollama (runs in background)
ollama serve

# Or use launchd for automatic startup
brew services start ollama
```

#### Linux
```bash
# Start Ollama service
sudo systemctl start ollama

# Enable automatic startup
sudo systemctl enable ollama

# Check status
sudo systemctl status ollama
```

## Verification and Testing

### 1. Verify Installation

```bash
# Check ABOV3 Genesis installation
abov3 --version

# Check Ollama installation
ollama --version

# Check Python version
python3 --version
```

### 2. Test Ollama Connection

```bash
# List installed models
ollama list

# Test model response
ollama run llama3 "Hello, can you help me code?"
```

### 3. Test ABOV3 Genesis

```bash
# Start ABOV3 Genesis
abov3

# In ABOV3, try these commands:
# /model list       - Check available models
# /help            - Show available commands
# create a simple Python function to add two numbers
```

### 4. Run System Tests

```bash
# Navigate to ABOV3 installation directory
cd path/to/abov3-genesis

# Run comprehensive tests
python run_tests.py

# Run specific component tests
python -m pytest tests/ -v
```

## Post-Installation Configuration

### 1. Configure Default Settings

Create global configuration file:

```yaml
# ~/.abov3/config.yaml
default_agent: genesis-architect
auto_save_interval: 30
max_recent_projects: 10
theme: genesis
genz_messages: true
ai_model: llama3:latest

preferences:
  show_welcome: true
  auto_detect_language: true
  enable_gpu: true
  
models:
  coding: codellama
  general: llama3
  debugging: qwen2-coder
```

### 2. Setup Project Directories

```bash
# Create projects directory
mkdir -p ~/projects/abov3

# Set permissions (Linux/macOS)
chmod 755 ~/projects/abov3
```

### 3. Configure Environment Variables

#### Windows (PowerShell)
```powershell
# Add to PowerShell profile
$profile_path = $PROFILE.AllUsersAllHosts
Add-Content -Path $profile_path -Value '$env:ABOV3_HOME = "$env:USERPROFILE\.abov3"'
Add-Content -Path $profile_path -Value '$env:ABOV3_PROJECTS = "$env:USERPROFILE\projects"'
```

#### macOS/Linux (Bash/Zsh)
```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'export ABOV3_HOME="$HOME/.abov3"' >> ~/.bashrc
echo 'export ABOV3_PROJECTS="$HOME/projects"' >> ~/.bashrc
echo 'export OLLAMA_HOST="127.0.0.1:11434"' >> ~/.bashrc
```

### 4. Install Optional Dependencies

```bash
# Enhanced features
pip install abov3-genesis[full]

# Development tools
pip install abov3-genesis[dev]

# GPU acceleration (NVIDIA)
pip install abov3-genesis[cuda]
```

## Troubleshooting

### Common Installation Issues

#### Issue: Python Not Found
```bash
# Solution: Install Python 3.8+
# Windows: Download from python.org
# macOS: brew install python@3.11  
# Linux: apt install python3
```

#### Issue: Ollama Connection Failed
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if not running
ollama serve

# Check firewall settings
# Windows: Allow port 11434 in Windows Firewall
# Linux: sudo ufw allow 11434
```

#### Issue: Model Download Failed
```bash
# Check internet connection
ping ollama.ai

# Try downloading specific model
ollama pull llama3 --verbose

# Check disk space
df -h  # Linux/macOS
dir C:\  # Windows
```

#### Issue: Permission Denied
```bash
# Linux/macOS: Fix permissions
sudo chown -R $USER:$USER ~/.abov3
chmod -R 755 ~/.abov3

# Windows: Run as Administrator
# Right-click Command Prompt → "Run as administrator"
```

#### Issue: Module Import Error
```python
# Solution: Install in virtual environment
python -m venv abov3-env
source abov3-env/bin/activate  # Linux/macOS
abov3-env\Scripts\activate     # Windows
pip install abov3-genesis
```

### Performance Optimization

#### GPU Configuration
```bash
# NVIDIA GPU setup
nvidia-smi  # Check GPU status

# Install CUDA toolkit
# Windows/Linux: Download from nvidia.com
# Configure Ollama to use GPU
export OLLAMA_GPU=true
```

#### Memory Optimization
```bash
# Adjust Ollama memory usage
export OLLAMA_RUNNER_ARGS="--memory 8192"

# Configure model context size
export OLLAMA_CONTEXT_SIZE=4096
```

## Advanced Installation

### Docker Installation

#### Using Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_GPU=true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  abov3-genesis:
    build: .
    ports:
      - "8080:8080"
    depends_on:
      - ollama
    environment:
      - OLLAMA_HOST=ollama:11434
    volumes:
      - ./projects:/app/projects

volumes:
  ollama_data:
```

#### Running with Docker

```bash
# Build and start services
docker-compose up -d

# Pull models
docker-compose exec ollama ollama pull llama3

# Access ABOV3 Genesis
docker-compose exec abov3-genesis abov3
```

### Kubernetes Installation

```yaml
# k8s/ollama-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ollama
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ollama
  template:
    metadata:
      labels:
        app: ollama
    spec:
      containers:
      - name: ollama
        image: ollama/ollama:latest
        ports:
        - containerPort: 11434
        resources:
          requests:
            memory: "8Gi"
          limits:
            memory: "16Gi"
            nvidia.com/gpu: 1
---
apiVersion: v1
kind: Service
metadata:
  name: ollama-service
spec:
  selector:
    app: ollama
  ports:
  - port: 11434
    targetPort: 11434
  type: ClusterIP
```

### Enterprise Installation

#### Multi-User Setup

```bash
# Install with enterprise features
pip install abov3-genesis[enterprise]

# Configure multi-user database
export ABOV3_DB_URL="postgresql://user:pass@localhost/abov3"

# Setup authentication
export ABOV3_AUTH_PROVIDER="ldap"
export ABOV3_LDAP_SERVER="ldap.company.com"
```

#### Load Balancer Configuration

```nginx
# nginx.conf
upstream abov3_backend {
    server abov3-1:8080;
    server abov3-2:8080;
    server abov3-3:8080;
}

server {
    listen 80;
    server_name abov3.company.com;
    
    location / {
        proxy_pass http://abov3_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Next Steps

After successful installation:

1. **Read the [User Guide](USER_GUIDE.md)** to learn how to use ABOV3 Genesis
2. **Check out [Examples](examples/)** to see what you can build
3. **Configure [Ollama Models](OLLAMA_CONFIGURATION.md)** for optimal performance
4. **Join the community** for support and updates

## Support

If you encounter issues during installation:

1. **Check the logs**: `~/.abov3/logs/installation.log`
2. **Search existing issues**: [GitHub Issues](https://github.com/fajardofahad/abov3-genesis/issues)
3. **Create a bug report** with system information and error logs
4. **Join our Discord** for community support

---

**Congratulations!** You now have ABOV3 Genesis installed and ready to transform your ideas into reality! 🚀