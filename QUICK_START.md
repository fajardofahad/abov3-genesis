# 🚀 ABOV3 Genesis - Quick Start Guide

## Running Without Installation

Follow these steps to run ABOV3 Genesis directly from source code:

### 1. Prerequisites

#### Install Ollama (Required)
```bash
# Download from https://ollama.ai/
# Or use package managers:

# Windows (Chocolatey)
choco install ollama

# macOS (Homebrew)
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh
```

#### Pull an AI Model
```bash
# Pull Llama 3 (recommended)
ollama pull llama3

# Or try other models:
ollama pull codellama    # For coding
ollama pull gemma:7b     # Smaller, faster
ollama pull mistral      # Good balance
```

#### Start Ollama Server
```bash
ollama serve
```

### 2. Install Python Dependencies

```bash
# Navigate to the project directory
cd C:\Users\fajar\Documents\ABOV3\abov3-Genesis\abov3-genesis-v1.0.0

# Install minimal dependencies
pip install -r requirements-dev.txt

# Or install manually if needed:
pip install ollama prompt_toolkit rich click pyyaml
```

### 3. Test Your Setup

```bash
# Run the setup test
python test_setup.py
```

You should see output like:
```
🚀 ABOV3 Genesis - Setup Test

🧪 Testing ABOV3 Genesis imports...
  ✅ GenZ Status messages
  ✅ Genesis Engine
  ✅ Project Registry
  ✅ Ollama Client
  ✅ Agent Manager

🎭 Testing GenZ status messages...
  💭 Thinking: 🧠 Big brain time fr fr...
  🏗️  Building: 🏗️ From idea to reality, watch this...
  ✨ Success: ✨ From idea to reality - absolutely slayed! 💅

🤖 Testing Ollama connection...
  ✅ Ollama is running with 3 models
    - llama3:latest
    - codellama:latest
    - gemma:7b

📊 Test Results: 3/3 tests passed
🎉 All tests passed! ABOV3 Genesis is ready to run.
```

### 4. Run ABOV3 Genesis

Choose your preferred method:

#### Option A: Python Script (Recommended)
```bash
python run_abov3.py
```

#### Option B: Windows Batch File
```bash
run_abov3.bat
```

#### Option C: Direct Python Module
```bash
python -m abov3.main
```

### 5. First Time Setup

When you run ABOV3 Genesis for the first time:

1. **You'll see the Genesis banner**:
```
╔══════════════════════════════════════════════════════╗
║              ABOV3 Genesis v1.0.0                    ║
║         From Idea to Built Reality                   ║
╚══════════════════════════════════════════════════════╝
```

2. **Choose "Create new project"** (option 1)

3. **Enter your idea**:
   - Example: "I want to build a simple calculator app"
   - Or: "Create a todo list web application"
   - Or: "Build a file organizer CLI tool"

4. **Give your project a name** or press Enter for auto-generated

5. **Choose project location** or press Enter for default

6. **Watch the magic happen!** 🎉

### 6. Example Session

```bash
$ python run_abov3.py

Choose your genesis path:
🌟 Start Fresh:
  1. 💡 Create new project (Start your genesis)

Choose your destiny > 1

💡 Genesis: Create New Project
First, tell me your idea:
> I want to build a task management web app

Project name [task-management-web]: my-tasks
Project location [C:\Users\fajar\projects\my-tasks]: 

🏗️ From idea to reality, watch this...
✅ Genesis initiated for 'my-tasks'
💡 Idea captured: I want to build a task management web app

✓ ABOV3 Genesis [my-tasks/genesis-architect/💡]> build my idea

🌟 Starting Genesis Workflow
📐 Entering Design Phase...

[Genesis Architect creates system architecture...]
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. "Module not found" errors
```bash
# Make sure you're in the right directory
cd C:\Users\fajar\Documents\ABOV3\abov3-Genesis\abov3-genesis-v1.0.0

# Install dependencies
pip install -r requirements-dev.txt
```

#### 2. "Ollama not available"
```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# Start Ollama if not running
ollama serve

# Pull a model if none available
ollama pull llama3
```

#### 3. Unicode/encoding errors
```bash
# Try running with explicit UTF-8 encoding
set PYTHONIOENCODING=utf-8
python run_abov3.py
```

#### 4. Permission errors on Windows
```bash
# Run as administrator or check your Python installation
# Make sure Python is in your PATH
python --version
```

### Still Having Issues?

1. **Run the test script**: `python test_setup.py`
2. **Check the error messages** for specific guidance
3. **Make sure all prerequisites are installed**
4. **Try installing in a virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # macOS/Linux
   pip install -r requirements-dev.txt
   python run_abov3.py
   ```

## 🎯 Next Steps

Once ABOV3 Genesis is running:

1. **Try different ideas** - Web apps, CLI tools, APIs, games
2. **Explore different agents** - Use `/agents list` to see specialists
3. **Check out commands** - Type `/help` for all available commands
4. **Build something amazing** - Transform your ideas into reality!

---

**✨ From Idea to Built Reality - Let's cook something amazing! 🔥**