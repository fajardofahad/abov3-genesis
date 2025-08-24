"""
ABOV3 Genesis Core Package
Core AI processing and assistant functionality
"""

from .assistant import Assistant
from .ollama_client import OllamaClient

__all__ = ["Assistant", "OllamaClient"]