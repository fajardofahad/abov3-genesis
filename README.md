# ABOV3 Genesis v1.0.0
## From Idea to Built Reality

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ollama](https://img.shields.io/badge/Powered_by-Ollama-green.svg)](https://ollama.ai/)

**ABOV3 Genesis** is a revolutionary AI-powered coding assistant that transforms your ideas into fully built, working applications. No cap, we're about to cook! 🔥

## ✨ What Makes ABOV3 Genesis Special?

- **💡 From Idea to Reality**: Complete workflow transformation through 5 phases
- **🤖 Local AI Power**: Uses Ollama models for private, offline AI assistance  
- **📁 Project-Centric**: Each project gets its own isolated workspace and history
- **🎯 Genesis Agents**: Specialized AI agents for every phase of development
- **💅 GenZ Vibes**: Fun, engaging status messages that keep you motivated
- **🚀 Enterprise-Ready**: Built for serious development with professional features

## 🌟 The Genesis Workflow

Transform any idea through our proven 5-phase workflow:

```
💡 Idea Phase    → Capture and analyze your concept
📐 Design Phase  → Create system architecture  
🔨 Build Phase   → Generate production code
🧪 Test Phase    → Ensure quality and reliability
🚀 Deploy Phase  → Launch to the world
```

## 🚀 Quick Start

### Prerequisites

1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai/)
2. **Pull a model**: `ollama pull llama3` (or any preferred model)
3. **Python 3.8+**: Make sure Python is installed

### Installation

```bash
pip install abov3-genesis
```

### Your First Genesis

```bash
# Start ABOV3 Genesis
abov3

# Choose "Create new project" and enter your idea
# Example: "I want to build a task management app"

# Watch the magic happen as Genesis transforms your idea into reality!
```

## 🎯 Core Features

### 🏗️ Genesis Engine
The heart of ABOV3 - transforms abstract ideas into concrete implementations through intelligent workflow management.

### 🤖 Specialized Agents
- **Genesis Architect**: Designs system architecture from ideas
- **Genesis Builder**: Writes production-ready code
- **Genesis Designer**: Creates beautiful user interfaces
- **Genesis Optimizer**: Perfects and optimizes code
- **Genesis Deployer**: Handles deployment and production

### 📁 Project Management
- **Isolated Workspaces**: Each project has its own `.abov3` directory
- **Session Recovery**: Pick up exactly where you left off
- **Project Registry**: Easy switching between projects
- **History Tracking**: Complete conversation and progress history

### 💅 GenZ Status Messages
Stay motivated with our iconic status messages:
- "🧠 Big brain time fr fr..."
- "🔥 Your idea said 'make me iconic'..."
- "✨ From idea to reality - absolutely slayed! 💅"

## 📖 Usage Examples

### Creating a Web App
```bash
abov3

# Enter idea: "I want to build a recipe sharing platform"
# Genesis will:
# 1. Design the architecture (database, API, frontend)
# 2. Generate complete code (React frontend, FastAPI backend)
# 3. Create comprehensive tests
# 4. Set up deployment configuration
```

### Building an API
```bash
abov3

# Enter idea: "I need a REST API for managing inventory"
# Genesis will:
# 1. Design the data models and endpoints
# 2. Implement the API with proper validation
# 3. Add authentication and security
# 4. Generate API documentation
```

### CLI Tool Development
```bash
abov3

# Enter idea: "I want to create a CLI tool for file organization"
# Genesis will:
# 1. Design the command structure
# 2. Implement the CLI with click/argparse
# 3. Add comprehensive help and documentation
# 4. Create installation and packaging scripts
```

## 🔧 Commands

### Basic Commands
```bash
abov3                    # Start Genesis (interactive project selection)
abov3 /path/to/project   # Open specific project
abov3 --new             # Create new project immediately
abov3 --version         # Show version
```

### In-Genesis Commands
```bash
# Genesis Workflow
build my idea           # Start the Genesis transformation
continue genesis        # Move to next phase
genesis status         # Show current progress

# Project Management
/project               # Show current project info
/project switch        # Switch to different project
/project list          # Show all projects

# Agent Management  
/agents               # Show current agent
/agents list          # List all available agents
/agents switch <name> # Switch to specific agent

# Other
/help                 # Show all commands
/vibe                 # Get motivated!
/clear                # Clear screen
/exit                 # Save and exit
```

## 🏗️ Project Structure

When you create a Genesis project, here's what gets set up:

```
your-project/
├── .abov3/                    # Genesis workspace
│   ├── genesis.yaml           # Genesis metadata
│   ├── project.yaml           # Project configuration
│   ├── agents/                # Custom agents
│   ├── sessions/              # Session data
│   ├── history/               # Conversation history
│   ├── genesis_flow/          # Workflow tracking
│   ├── tasks/                 # Task management
│   ├── permissions/           # Permission preferences
│   └── dependencies/          # Dependency management
├── src/                       # Your application code
├── tests/                     # Generated tests
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
```

## 🎨 Customization

### Custom Agents
Create specialized agents for your specific needs:

```bash
/agents create my-specialist llama3:latest "Description" "System prompt"
```

### Genesis Templates
Start with pre-built templates:
- Web App Genesis (React + FastAPI)
- API Genesis (FastAPI + SQLAlchemy) 
- CLI Genesis (Click + Rich)
- Data Pipeline Genesis (Pandas + Airflow)
- ML Model Genesis (PyTorch + MLflow)

### Permission Management
Control what Genesis can do:
```bash
# Safe mode (asks permission for everything)
# Auto-confirm safe operations
# Remember choices for similar operations
```

## 🔌 Integrations

### Supported AI Models
- **llama3**: General purpose development
- **codellama**: Code generation specialist  
- **deepseek-coder**: Advanced coding model
- **gemma**: Fast and efficient
- **mistral**: Multilingual development

### Package Managers
- pip (Python)
- npm/yarn (JavaScript)
- cargo (Rust)
- composer (PHP)

### Version Control
- Git integration
- Automatic .gitignore creation
- Commit message generation

## 🛠️ Configuration

### Global Settings
Located at `~/.abov3/config.yaml`:

```yaml
default_agent: genesis-architect
auto_save_interval: 30
max_recent_projects: 10
theme: genesis
genz_messages: true
preferences:
  show_welcome: true
  auto_detect_language: true
```

### Project Settings
Located at `project/.abov3/project.yaml`:

```yaml
name: my-project
version: 0.1.0
genesis: true
settings:
  auto_save: true
  auto_format: true
  auto_test: false
```

## 🚨 Troubleshooting

### Ollama Not Found
```bash
# Install Ollama from https://ollama.ai/
# Pull a model:
ollama pull llama3

# Start Ollama service:
ollama serve
```

### Permission Issues
```bash
# Reset all permissions:
# In Genesis: /permissions reset

# Check current permissions:
# In Genesis: /permissions list
```

### Session Recovery Failed
```bash
# Clear corrupted session:
rm -rf .abov3/sessions/current.session

# Restart Genesis
abov3
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Setup
```bash
git clone https://github.com/fajardofahad/abov3-genesis.git
cd abov3-genesis
pip install -e .
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ollama Team** - For providing the local AI infrastructure
- **Rich** - For beautiful terminal UI components
- **Click** - For elegant command-line interfaces
- **The Python Community** - For the amazing ecosystem

## 📚 Documentation

### Quick Navigation
- **[📖 Documentation Hub](docs/README.md)** - Complete documentation index
- **[🚀 User Guide](docs/USER_GUIDE.md)** - Learn to use ABOV3 like Claude
- **[💾 Installation Guide](docs/INSTALLATION_GUIDE.md)** - Platform-specific setup
- **[🛠️ API Documentation](docs/API_DOCUMENTATION.md)** - Developer reference

### Essential Guides
- **[🎯 Examples & Tutorials](docs/EXAMPLES_AND_TUTORIALS.md)** - Practical examples for all use cases
- **[⚙️ Ollama Configuration](docs/OLLAMA_CONFIGURATION.md)** - AI model setup and optimization  
- **[⚡ Performance Optimization](docs/PERFORMANCE_OPTIMIZATION.md)** - Advanced tuning guide
- **[🔧 Troubleshooting](docs/TROUBLESHOOTING.md)** - Solutions for common issues
- **[🤝 Contributing](docs/CONTRIBUTING.md)** - How to contribute to the project

## 🔗 Links

- **[📚 Complete Documentation](docs/README.md)** - Start here for all documentation
- **GitHub**: https://github.com/fajardofahad/abov3-genesis
- **Issues**: https://github.com/fajardofahad/abov3-genesis/issues
- **Discussions**: https://github.com/fajardofahad/abov3-genesis/discussions
- **Discord**: [Community Coming Soon]

---

**✨ Transform your ideas into reality with ABOV3 Genesis - From Idea to Built Reality! ✨**

*Built with 💜 by developers, for developers*