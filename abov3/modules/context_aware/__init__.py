"""
ABOV3 Genesis - Context-Aware Comprehension Module
Module 2: Intelligent codebase understanding and analysis system
"""

from .core.comprehension_engine import ComprehensionEngine
from .indexing.code_indexer import CodeIndexer
from .knowledge_graph.graph_builder import KnowledgeGraphBuilder
from .semantic_search.search_engine import SemanticSearchEngine
from .analysis.code_analyzer import CodeAnalyzer
from .refactoring.suggestion_engine import RefactoringSuggestionEngine

__version__ = "1.0.0"
__all__ = [
    'ComprehensionEngine',
    'CodeIndexer',
    'KnowledgeGraphBuilder', 
    'SemanticSearchEngine',
    'CodeAnalyzer',
    'RefactoringSuggestionEngine'
]