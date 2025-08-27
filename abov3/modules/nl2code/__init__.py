"""
ABOV3 Genesis - Natural Language to Code Module
Module 1: Natural Language to Code Generation System
"""

from .core.processor import NaturalLanguageProcessor
from .planning.engine import PlanningEngine
from .generation.engine import CodeGenerationEngine
from .testing.generator import TestGenerator

__version__ = "1.0.0"
__all__ = [
    'NaturalLanguageProcessor',
    'PlanningEngine', 
    'CodeGenerationEngine',
    'TestGenerator'
]