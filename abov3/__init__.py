"""
ABOV3 Genesis - From Idea to Built Reality
Advanced AI Coding Assistant Package

Transform your ideas into working applications with AI-powered development.
"""

__version__ = "1.0.0"
__tagline__ = "From Idea to Built Reality"
__author__ = "ABOV3 Team"
__email__ = "team@abov3.dev"

# Core imports for easy access
from .genesis.engine import GenesisEngine
from .core.assistant import Assistant
from .project.manager import ProjectManager

__all__ = [
    "GenesisEngine",
    "Assistant", 
    "ProjectManager",
    "__version__",
    "__tagline__"
]