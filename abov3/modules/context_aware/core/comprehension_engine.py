"""
ABOV3 Genesis - Context-Aware Comprehension Engine
Core orchestration engine for intelligent codebase understanding
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import json

from ...core.context_manager import SmartContextManager
from ...core.ollama_client import OllamaClient
from ..indexing.code_indexer import CodeIndexer
from ..knowledge_graph.graph_builder import KnowledgeGraphBuilder
from ..semantic_search.search_engine import SemanticSearchEngine
from ..analysis.code_analyzer import CodeAnalyzer
from ..refactoring.suggestion_engine import RefactoringSuggestionEngine

logger = logging.getLogger(__name__)

class ComprehensionMode(Enum):
    """Different comprehension modes for various use cases"""
    QUICK_SCAN = "quick_scan"          # Fast overview for small files
    DEEP_ANALYSIS = "deep_analysis"    # Comprehensive analysis
    SEMANTIC_SEARCH = "semantic_search" # Find similar code patterns
    QA_MODE = "qa_mode"               # Answer questions about code
    REFACTOR_MODE = "refactor_mode"   # Suggest refactoring improvements
    MONOREPO_MODE = "monorepo_mode"   # Handle large monorepos efficiently

@dataclass
class ComprehensionRequest:
    """Request for code comprehension"""
    query: str
    mode: ComprehensionMode = ComprehensionMode.DEEP_ANALYSIS
    target_paths: List[str] = field(default_factory=list)
    include_tests: bool = True
    include_docs: bool = True
    max_files: int = 1000
    max_lines_per_file: int = 5000
    context_depth: int = 3  # How deep to traverse dependencies
    use_cache: bool = True
    priority_languages: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "node_modules", ".git", "__pycache__", ".venv", "venv", 
        "build", "dist", ".next", ".nuxt", "target"
    ])

@dataclass 
class ComprehensionResult:
    """Result of code comprehension"""
    query: str
    mode: ComprehensionMode
    answer: str
    confidence_score: float
    source_files: List[str]
    related_concepts: List[str]
    suggestions: List[str]
    execution_time: float
    tokens_used: int
    cache_hit: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

class ComprehensionEngine:
    """
    Main engine for context-aware code comprehension
    Orchestrates indexing, analysis, and intelligent understanding
    """
    
    def __init__(
        self, 
        workspace_path: str,
        ollama_client: OllamaClient,
        model_name: str = "deepseek-coder",
        enable_caching: bool = True,
        max_cache_size: int = 1000
    ):
        self.workspace_path = Path(workspace_path)
        self.ollama_client = ollama_client
        self.model_name = model_name
        self.enable_caching = enable_caching
        
        # Core components
        self.context_manager = SmartContextManager(max_context_tokens=16384, model_name=model_name)
        self.code_indexer = CodeIndexer(workspace_path, enable_caching)
        self.knowledge_graph = KnowledgeGraphBuilder()
        self.semantic_search = SemanticSearchEngine()
        self.code_analyzer = CodeAnalyzer()
        self.refactoring_engine = RefactoringSuggestionEngine()
        
        # Performance tracking
        self.query_cache: Dict[str, ComprehensionResult] = {}
        self.performance_stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "avg_response_time": 0.0,
            "total_files_indexed": 0,
            "last_index_update": None
        }
        
        # Configuration
        self.max_cache_size = max_cache_size
        self.index_update_interval = 300  # 5 minutes
        self._last_index_update = 0
        
        logger.info(f"ComprehensionEngine initialized for workspace: {workspace_path}")
    
    async def initialize(self):
        """Initialize the comprehension engine"""
        logger.info("Initializing ComprehensionEngine...")
        
        # Initialize components
        await self.code_indexer.initialize()
        await self.semantic_search.initialize()
        await self.knowledge_graph.initialize()
        
        # Build initial index
        await self.update_index()
        
        logger.info("ComprehensionEngine initialization complete")
    
    async def comprehend(
        self, 
        request: Union[str, ComprehensionRequest]
    ) -> ComprehensionResult:
        """
        Main comprehension method - understands code and answers queries
        """
        start_time = time.time()
        
        # Convert string query to request object
        if isinstance(request, str):
            request = ComprehensionRequest(query=request)
        
        # Check cache first
        if self.enable_caching and request.use_cache:
            cache_key = self._create_cache_key(request)
            if cache_key in self.query_cache:
                result = self.query_cache[cache_key]
                result.cache_hit = True
                self.performance_stats["cache_hits"] += 1
                return result
        
        # Update index if needed
        if time.time() - self._last_index_update > self.index_update_interval:
            await self.update_index()
        
        # Route to appropriate comprehension mode
        if request.mode == ComprehensionMode.QUICK_SCAN:
            result = await self._quick_scan_comprehension(request)
        elif request.mode == ComprehensionMode.DEEP_ANALYSIS:
            result = await self._deep_analysis_comprehension(request)
        elif request.mode == ComprehensionMode.SEMANTIC_SEARCH:
            result = await self._semantic_search_comprehension(request)
        elif request.mode == ComprehensionMode.QA_MODE:
            result = await self._qa_mode_comprehension(request)
        elif request.mode == ComprehensionMode.REFACTOR_MODE:
            result = await self._refactor_mode_comprehension(request)
        elif request.mode == ComprehensionMode.MONOREPO_MODE:
            result = await self._monorepo_mode_comprehension(request)
        else:
            result = await self._deep_analysis_comprehension(request)
        
        # Add execution metadata
        result.execution_time = time.time() - start_time
        result.cache_hit = False
        
        # Cache the result
        if self.enable_caching:
            cache_key = self._create_cache_key(request)
            self.query_cache[cache_key] = result
            
            # Maintain cache size
            if len(self.query_cache) > self.max_cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.query_cache))
                del self.query_cache[oldest_key]
        
        # Update performance stats
        self.performance_stats["total_queries"] += 1
        self._update_performance_stats(result.execution_time)
        
        logger.info(f"Comprehension completed in {result.execution_time:.2f}s")
        return result
    
    async def _quick_scan_comprehension(self, request: ComprehensionRequest) -> ComprehensionResult:
        """Quick scan mode for fast overview"""
        logger.debug("Performing quick scan comprehension")
        
        # Get file list based on request
        relevant_files = await self.code_indexer.find_relevant_files(
            request.query, 
            max_files=min(50, request.max_files)  # Limit for quick scan
        )
        
        # Build lightweight context
        context_parts = []
        source_files = []
        
        for file_info in relevant_files[:20]:  # Top 20 files
            file_path = file_info['path']
            if self._should_include_file(file_path, request):
                # Get file summary instead of full content
                summary = await self.code_indexer.get_file_summary(file_path)
                if summary:
                    context_parts.append(f"File: {file_path}\nSummary: {summary}\n")
                    source_files.append(str(file_path))
        
        # Build context for AI model
        context = "\n".join(context_parts)
        self.context_manager.add_project_info(context)
        
        # Generate response
        prompt = self._build_prompt(request.query, "quick_scan")
        full_context = self.context_manager.build_optimized_context(
            task_type="explanation",
            query=request.query,
            target_tokens=4096
        )
        
        response = await self._generate_ai_response(prompt, full_context)
        
        return ComprehensionResult(
            query=request.query,
            mode=request.mode,
            answer=response,
            confidence_score=0.8,  # Quick scan has lower confidence
            source_files=source_files,
            related_concepts=[],
            suggestions=[],
            execution_time=0.0,
            tokens_used=len(full_context.split())
        )
    
    async def _deep_analysis_comprehension(self, request: ComprehensionRequest) -> ComprehensionResult:
        """Deep analysis mode for comprehensive understanding"""
        logger.debug("Performing deep analysis comprehension")
        
        # Get comprehensive file analysis
        relevant_files = await self.code_indexer.find_relevant_files(
            request.query,
            max_files=request.max_files
        )
        
        # Analyze code structure and relationships
        analysis_results = []
        source_files = []
        
        for file_info in relevant_files:
            file_path = file_info['path']
            if self._should_include_file(file_path, request):
                # Deep analysis of each relevant file
                analysis = await self.code_analyzer.analyze_file(
                    file_path,
                    include_ast=True,
                    include_metrics=True,
                    include_dependencies=True
                )
                if analysis:
                    analysis_results.append(analysis)
                    source_files.append(str(file_path))
        
        # Build knowledge graph for the relevant code
        subgraph = await self.knowledge_graph.build_subgraph(
            source_files,
            depth=request.context_depth
        )
        
        # Extract key concepts and relationships
        concepts = self.knowledge_graph.extract_concepts(subgraph)
        relationships = self.knowledge_graph.extract_relationships(subgraph)
        
        # Build comprehensive context
        context_parts = []
        
        # Add architectural overview
        if len(analysis_results) > 5:
            arch_overview = self._generate_architectural_overview(analysis_results)
            context_parts.append(f"Architectural Overview:\n{arch_overview}\n")
        
        # Add key file analyses
        for analysis in analysis_results[:10]:  # Top 10 analyses
            file_context = self._format_file_analysis(analysis)
            context_parts.append(file_context)
        
        # Add concept relationships
        if concepts:
            concepts_text = f"Key Concepts: {', '.join(concepts[:20])}"
            context_parts.append(concepts_text)
        
        # Build AI context
        context = "\n\n".join(context_parts)
        self.context_manager.add_project_info(context)
        
        # Generate comprehensive response
        prompt = self._build_prompt(request.query, "deep_analysis")
        full_context = self.context_manager.build_optimized_context(
            task_type="architecture",
            query=request.query,
            target_tokens=12288
        )
        
        response = await self._generate_ai_response(prompt, full_context)
        
        # Generate suggestions
        suggestions = await self._generate_suggestions(analysis_results, request.query)
        
        return ComprehensionResult(
            query=request.query,
            mode=request.mode,
            answer=response,
            confidence_score=0.95,  # High confidence for deep analysis
            source_files=source_files,
            related_concepts=concepts[:10],
            suggestions=suggestions,
            execution_time=0.0,
            tokens_used=len(full_context.split()),
            metadata={
                "files_analyzed": len(analysis_results),
                "concepts_found": len(concepts),
                "relationships_found": len(relationships)
            }
        )
    
    async def _semantic_search_comprehension(self, request: ComprehensionRequest) -> ComprehensionResult:
        """Semantic search mode for finding similar patterns"""
        logger.debug("Performing semantic search comprehension")
        
        # Perform semantic search
        search_results = await self.semantic_search.search_similar_code(
            request.query,
            top_k=50,
            include_context=True
        )
        
        # Group results by similarity type
        grouped_results = self.semantic_search.group_by_similarity(search_results)
        
        # Build context from search results
        context_parts = []
        source_files = []
        
        for group_name, results in grouped_results.items():
            if len(context_parts) > 20:  # Limit context size
                break
                
            context_parts.append(f"## {group_name}")
            
            for result in results[:5]:  # Top 5 per group
                file_path = result['file_path']
                code_snippet = result['code_snippet']
                similarity_score = result['similarity_score']
                
                context_parts.append(
                    f"File: {file_path} (Similarity: {similarity_score:.2f})\n"
                    f"```{result.get('language', 'text')}\n{code_snippet}\n```\n"
                )
                
                if file_path not in source_files:
                    source_files.append(file_path)
        
        # Build AI context
        context = "\n\n".join(context_parts)
        self.context_manager.add_project_info(context)
        
        prompt = self._build_prompt(request.query, "semantic_search")
        full_context = self.context_manager.build_optimized_context(
            task_type="code_generation",
            query=request.query,
            target_tokens=10240
        )
        
        response = await self._generate_ai_response(prompt, full_context)
        
        return ComprehensionResult(
            query=request.query,
            mode=request.mode,
            answer=response,
            confidence_score=0.85,
            source_files=source_files,
            related_concepts=[],
            suggestions=[],
            execution_time=0.0,
            tokens_used=len(full_context.split()),
            metadata={
                "search_results": len(search_results),
                "similarity_groups": len(grouped_results)
            }
        )
    
    async def _qa_mode_comprehension(self, request: ComprehensionRequest) -> ComprehensionResult:
        """Q&A mode for answering specific questions about code"""
        logger.debug("Performing Q&A mode comprehension")
        
        # Extract intent and entities from the query
        intent = self._extract_intent(request.query)
        entities = self._extract_entities(request.query)
        
        # Find relevant code based on intent and entities
        if intent == "find_function":
            relevant_files = await self.code_indexer.find_functions(entities)
        elif intent == "find_class":
            relevant_files = await self.code_indexer.find_classes(entities)
        elif intent == "explain_error":
            relevant_files = await self.code_indexer.find_error_related_code(entities)
        elif intent == "find_usage":
            relevant_files = await self.code_indexer.find_usage_examples(entities)
        else:
            relevant_files = await self.code_indexer.find_relevant_files(request.query, max_files=30)
        
        # Build focused context for Q&A
        context_parts = []
        source_files = []
        
        for file_info in relevant_files:
            if len(context_parts) > 15:  # Keep context focused
                break
                
            file_path = file_info['path']
            if self._should_include_file(file_path, request):
                # Get relevant code snippets
                snippets = await self.code_indexer.get_code_snippets(
                    file_path, 
                    entities, 
                    context_lines=5
                )
                
                if snippets:
                    for snippet in snippets:
                        context_parts.append(
                            f"File: {file_path}\n"
                            f"```{snippet.get('language', 'text')}\n{snippet['code']}\n```\n"
                        )
                    source_files.append(str(file_path))
        
        # Add to context manager
        context = "\n\n".join(context_parts)
        self.context_manager.add_project_info(context)
        
        # Generate focused Q&A response
        prompt = self._build_prompt(request.query, "qa", intent, entities)
        full_context = self.context_manager.build_optimized_context(
            task_type="explanation",
            query=request.query,
            target_tokens=8192
        )
        
        response = await self._generate_ai_response(prompt, full_context)
        
        return ComprehensionResult(
            query=request.query,
            mode=request.mode,
            answer=response,
            confidence_score=0.9,
            source_files=source_files,
            related_concepts=entities,
            suggestions=[],
            execution_time=0.0,
            tokens_used=len(full_context.split()),
            metadata={
                "intent": intent,
                "entities": entities
            }
        )
    
    async def _refactor_mode_comprehension(self, request: ComprehensionRequest) -> ComprehensionResult:
        """Refactoring mode for suggesting improvements"""
        logger.debug("Performing refactoring mode comprehension")
        
        # Get files for refactoring analysis
        if request.target_paths:
            target_files = [Path(p) for p in request.target_paths]
        else:
            file_results = await self.code_indexer.find_relevant_files(
                request.query, 
                max_files=20
            )
            target_files = [Path(f['path']) for f in file_results]
        
        # Analyze each file for refactoring opportunities
        refactoring_suggestions = []
        source_files = []
        
        for file_path in target_files:
            if self._should_include_file(file_path, request):
                suggestions = await self.refactoring_engine.analyze_file(file_path)
                if suggestions:
                    refactoring_suggestions.extend(suggestions)
                    source_files.append(str(file_path))
        
        # Prioritize suggestions
        prioritized_suggestions = self.refactoring_engine.prioritize_suggestions(
            refactoring_suggestions
        )
        
        # Build context with refactoring analysis
        context_parts = []
        
        # Add file analyses
        for file_path in source_files[:10]:
            analysis = await self.code_analyzer.analyze_file(
                file_path, 
                include_metrics=True,
                include_issues=True
            )
            if analysis:
                context_parts.append(self._format_file_analysis(analysis))
        
        # Add refactoring suggestions
        suggestions_text = self._format_refactoring_suggestions(prioritized_suggestions)
        context_parts.append(f"Refactoring Suggestions:\n{suggestions_text}")
        
        # Build AI context
        context = "\n\n".join(context_parts)
        self.context_manager.add_project_info(context)
        
        prompt = self._build_prompt(request.query, "refactoring")
        full_context = self.context_manager.build_optimized_context(
            task_type="debugging",
            query=request.query,
            target_tokens=10240
        )
        
        response = await self._generate_ai_response(prompt, full_context)
        
        # Format suggestions for output
        suggestion_texts = [s.description for s in prioritized_suggestions[:10]]
        
        return ComprehensionResult(
            query=request.query,
            mode=request.mode,
            answer=response,
            confidence_score=0.88,
            source_files=source_files,
            related_concepts=[],
            suggestions=suggestion_texts,
            execution_time=0.0,
            tokens_used=len(full_context.split()),
            metadata={
                "total_suggestions": len(refactoring_suggestions),
                "high_priority_suggestions": len([s for s in prioritized_suggestions if s.priority == "high"])
            }
        )
    
    async def _monorepo_mode_comprehension(self, request: ComprehensionRequest) -> ComprehensionResult:
        """Monorepo mode for handling large repositories efficiently"""
        logger.debug("Performing monorepo mode comprehension")
        
        # Use hierarchical analysis for large codebases
        hierarchy = await self.code_indexer.build_project_hierarchy()
        
        # Identify relevant modules/packages first
        relevant_modules = await self._identify_relevant_modules(
            request.query, 
            hierarchy
        )
        
        # Focus analysis on relevant modules
        context_parts = []
        source_files = []
        module_summaries = []
        
        for module_path, relevance_score in relevant_modules[:20]:  # Top 20 modules
            # Get module summary
            module_summary = await self.code_indexer.get_module_summary(module_path)
            if module_summary:
                module_summaries.append({
                    'path': module_path,
                    'summary': module_summary,
                    'relevance': relevance_score
                })
                
                # Get key files from this module
                key_files = await self.code_indexer.get_key_files_in_module(
                    module_path, 
                    max_files=5
                )
                
                for file_info in key_files:
                    file_path = file_info['path']
                    if len(source_files) < 50:  # Limit total files
                        source_files.append(str(file_path))
        
        # Build high-level architectural overview
        arch_overview = self._build_monorepo_overview(module_summaries, hierarchy)
        context_parts.append(f"Monorepo Architecture:\n{arch_overview}")
        
        # Add module summaries
        for module_info in module_summaries[:15]:
            context_parts.append(
                f"Module: {module_info['path']}\n"
                f"Relevance: {module_info['relevance']:.2f}\n"
                f"Summary: {module_info['summary']}\n"
            )
        
        # Build AI context
        context = "\n\n".join(context_parts)
        self.context_manager.add_project_info(context)
        
        prompt = self._build_prompt(request.query, "monorepo")
        full_context = self.context_manager.build_optimized_context(
            task_type="architecture",
            query=request.query,
            target_tokens=14336
        )
        
        response = await self._generate_ai_response(prompt, full_context)
        
        return ComprehensionResult(
            query=request.query,
            mode=request.mode,
            answer=response,
            confidence_score=0.85,  # Slightly lower due to scale
            source_files=source_files,
            related_concepts=[m['path'] for m in module_summaries[:10]],
            suggestions=[],
            execution_time=0.0,
            tokens_used=len(full_context.split()),
            metadata={
                "modules_analyzed": len(relevant_modules),
                "total_hierarchy_depth": len(hierarchy),
                "files_in_scope": len(source_files)
            }
        )
    
    # Utility methods
    
    def _should_include_file(self, file_path: Path, request: ComprehensionRequest) -> bool:
        """Check if file should be included in analysis"""
        file_path_str = str(file_path)
        
        # Check exclude patterns
        for pattern in request.exclude_patterns:
            if pattern in file_path_str:
                return False
        
        # Check file size
        try:
            if file_path.stat().st_size > request.max_lines_per_file * 100:  # Rough estimate
                return False
        except:
            pass
        
        # Check if it's a test file and tests are excluded
        if not request.include_tests and ('test_' in file_path.name or '/test/' in file_path_str):
            return False
        
        # Check if it's a doc file and docs are excluded
        if not request.include_docs and file_path.suffix in ['.md', '.rst', '.txt']:
            return False
        
        return True
    
    def _build_prompt(self, query: str, mode: str, intent: str = None, entities: List[str] = None) -> str:
        """Build AI prompt based on comprehension mode"""
        base_prompt = f"Query: {query}\n\n"
        
        if mode == "quick_scan":
            base_prompt += (
                "Please provide a quick overview based on the file summaries provided. "
                "Focus on high-level architecture and key components."
            )
        elif mode == "deep_analysis":
            base_prompt += (
                "Please provide a comprehensive analysis of the codebase. "
                "Include architectural insights, key patterns, and relationships between components."
            )
        elif mode == "semantic_search":
            base_prompt += (
                "Based on the similar code patterns found, please explain the commonalities "
                "and provide insights about the code structure and best practices used."
            )
        elif mode == "qa":
            if intent and entities:
                base_prompt += (
                    f"Intent: {intent}\n"
                    f"Key entities: {', '.join(entities)}\n\n"
                    "Please provide a focused answer to the specific question based on the code provided."
                )
            else:
                base_prompt += "Please answer the question based on the code analysis provided."
        elif mode == "refactoring":
            base_prompt += (
                "Based on the code analysis and refactoring suggestions provided, "
                "please explain the recommended improvements and their benefits."
            )
        elif mode == "monorepo":
            base_prompt += (
                "Based on the monorepo structure analysis, please provide insights about "
                "the overall architecture, module relationships, and answer the query in context."
            )
        
        return base_prompt
    
    async def _generate_ai_response(self, prompt: str, context: str) -> str:
        """Generate AI response using Ollama"""
        try:
            full_prompt = f"{context}\n\n{prompt}"
            
            response = await self.ollama_client.generate(
                model=self.model_name,
                prompt=full_prompt,
                options={
                    "temperature": 0.1,  # Low temperature for factual responses
                    "top_p": 0.9,
                    "top_k": 40,
                    "repeat_penalty": 1.1
                }
            )
            
            if response and 'response' in response:
                return response['response']
            else:
                return "Unable to generate response. Please check the model and try again."
                
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return f"Error generating response: {str(e)}"
    
    async def update_index(self, force: bool = False):
        """Update the code index"""
        if not force and time.time() - self._last_index_update < self.index_update_interval:
            return
        
        logger.info("Updating code index...")
        await self.code_indexer.update_index()
        await self.semantic_search.update_embeddings()
        
        self._last_index_update = time.time()
        self.performance_stats["last_index_update"] = self._last_index_update
        self.performance_stats["total_files_indexed"] = await self.code_indexer.get_indexed_file_count()
        
        logger.info("Code index updated successfully")
    
    def _create_cache_key(self, request: ComprehensionRequest) -> str:
        """Create cache key for request"""
        key_parts = [
            request.query[:100],  # Truncate query
            request.mode.value,
            str(sorted(request.target_paths)),
            str(request.include_tests),
            str(request.include_docs),
            str(request.max_files),
            str(request.context_depth)
        ]
        return hash("|".join(key_parts))
    
    def _update_performance_stats(self, execution_time: float):
        """Update performance statistics"""
        total_queries = self.performance_stats["total_queries"]
        current_avg = self.performance_stats["avg_response_time"]
        
        # Calculate new average
        new_avg = (current_avg * (total_queries - 1) + execution_time) / total_queries
        self.performance_stats["avg_response_time"] = new_avg
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return {
            **self.performance_stats,
            "cache_hit_rate": self.performance_stats["cache_hits"] / max(1, self.performance_stats["total_queries"]),
            "cache_size": len(self.query_cache)
        }
    
    async def clear_cache(self):
        """Clear all caches"""
        self.query_cache.clear()
        await self.code_indexer.clear_cache()
        await self.semantic_search.clear_cache()
        logger.info("All caches cleared")
    
    # Placeholder methods for complex operations (to be implemented)
    
    def _generate_architectural_overview(self, analysis_results: List[Dict]) -> str:
        """Generate architectural overview from analysis results"""
        # This would analyze patterns across files to provide architectural insights
        return "Architectural analysis placeholder - to be implemented"
    
    def _format_file_analysis(self, analysis: Dict) -> str:
        """Format file analysis for context"""
        # Format analysis results in a context-friendly way
        return f"File analysis for {analysis.get('file_path', 'unknown')}: {analysis.get('summary', 'No summary')}"
    
    async def _generate_suggestions(self, analysis_results: List[Dict], query: str) -> List[str]:
        """Generate suggestions based on analysis"""
        # Generate contextual suggestions
        return ["Suggestion 1: Consider adding documentation", "Suggestion 2: Review error handling"]
    
    def _extract_intent(self, query: str) -> str:
        """Extract intent from query"""
        # Simple intent extraction - could be enhanced with NLP
        query_lower = query.lower()
        if "find function" in query_lower or "where is function" in query_lower:
            return "find_function"
        elif "find class" in query_lower or "where is class" in query_lower:
            return "find_class"
        elif "error" in query_lower or "exception" in query_lower or "bug" in query_lower:
            return "explain_error"
        elif "usage" in query_lower or "example" in query_lower or "how to use" in query_lower:
            return "find_usage"
        else:
            return "general_query"
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract entities (function names, class names, etc.) from query"""
        # Simple entity extraction - could be enhanced with NLP
        words = query.split()
        entities = []
        
        # Look for camelCase or snake_case patterns
        for word in words:
            if '_' in word or (any(c.isupper() for c in word) and any(c.islower() for c in word)):
                entities.append(word)
        
        return entities
    
    def _format_refactoring_suggestions(self, suggestions: List) -> str:
        """Format refactoring suggestions for context"""
        formatted = []
        for i, suggestion in enumerate(suggestions[:10], 1):
            formatted.append(f"{i}. {suggestion.description} (Priority: {suggestion.priority})")
        return "\n".join(formatted)
    
    async def _identify_relevant_modules(self, query: str, hierarchy: Dict) -> List[Tuple[str, float]]:
        """Identify relevant modules in monorepo"""
        # This would use semantic analysis to identify relevant modules
        # Placeholder implementation
        modules = []
        for module_path in hierarchy.keys():
            relevance_score = 0.5  # Placeholder scoring
            modules.append((module_path, relevance_score))
        
        return sorted(modules, key=lambda x: x[1], reverse=True)
    
    def _build_monorepo_overview(self, module_summaries: List[Dict], hierarchy: Dict) -> str:
        """Build overview of monorepo architecture"""
        overview_parts = []
        overview_parts.append(f"Total modules analyzed: {len(module_summaries)}")
        overview_parts.append(f"Hierarchy depth: {len(hierarchy)}")
        
        # Add top modules
        for module_info in module_summaries[:5]:
            overview_parts.append(f"- {module_info['path']}: {module_info['summary'][:100]}...")
        
        return "\n".join(overview_parts)