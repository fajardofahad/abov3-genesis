"""
ABOV3 Genesis - Semantic Search Engine
Vector-based semantic search for code similarity and pattern matching
"""

import asyncio
import logging
import json
import time
import pickle
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import hashlib
from concurrent.futures import ThreadPoolExecutor
import re

# For embeddings - using a lightweight approach that doesn't require external dependencies
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

@dataclass
class CodeSnippet:
    """Represents a code snippet for semantic search"""
    id: str
    content: str
    file_path: str
    language: str
    line_start: int
    line_end: int
    snippet_type: str  # function, class, method, etc.
    context: str = ""  # Surrounding context
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    keywords: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.id:
            # Generate unique ID
            id_string = f"{self.file_path}:{self.line_start}:{self.line_end}:{hash(self.content)}"
            self.id = hashlib.md5(id_string.encode()).hexdigest()[:12]

@dataclass
class SearchResult:
    """Search result with similarity score"""
    snippet: CodeSnippet
    similarity_score: float
    match_type: str  # exact, semantic, structural, etc.
    matched_keywords: List[str] = field(default_factory=list)
    explanation: str = ""

class SemanticSearchEngine:
    """
    High-performance semantic search engine for code
    Uses TF-IDF and word embeddings for similarity matching
    """
    
    def __init__(self, enable_caching: bool = True, embedding_dim: int = 300):
        self.enable_caching = enable_caching
        self.embedding_dim = embedding_dim
        
        # Storage
        self.snippets: Dict[str, CodeSnippet] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.snippet_ids: List[str] = []
        
        # TF-IDF components
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words='english',
            token_pattern=r'\b\w+\b',  # Include all word characters
            max_df=0.8,
            min_df=2
        )
        
        self.tfidf_matrix: Optional[np.ndarray] = None
        self.svd_transformer: Optional[TruncatedSVD] = None
        
        # Clustering for grouping similar code
        self.clusterer: Optional[KMeans] = None
        self.cluster_labels: Optional[np.ndarray] = None
        
        # Caches
        self.search_cache: Dict[str, List[SearchResult]] = {}
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
        
        # Performance tracking
        self.search_stats = {
            'total_searches': 0,
            'cache_hits': 0,
            'avg_search_time': 0.0,
            'total_snippets': 0,
            'last_index_time': 0.0
        }
        
        # Code pattern definitions
        self.code_patterns = self._initialize_code_patterns()
        
        logger.info("SemanticSearchEngine initialized")
    
    async def initialize(self):
        """Initialize the semantic search engine"""
        logger.info("Initializing SemanticSearchEngine...")
        # Any additional initialization
        logger.info("SemanticSearchEngine initialized")
    
    def _initialize_code_patterns(self) -> Dict[str, List[str]]:
        """Initialize common code patterns for better matching"""
        return {
            'api_endpoints': [
                r'@app\.route', r'@router\.(get|post|put|delete)', 
                r'app\.(get|post|put|delete)', r'router\.(get|post|put|delete)'
            ],
            'database_queries': [
                r'SELECT\s+.*\s+FROM', r'INSERT\s+INTO', r'UPDATE\s+.*\s+SET',
                r'DELETE\s+FROM', r'\.query\(', r'\.execute\('
            ],
            'async_patterns': [
                r'async\s+def', r'await\s+', r'asyncio\.', r'Task\[', r'Coroutine\['
            ],
            'error_handling': [
                r'try:', r'except\s+\w+:', r'raise\s+\w+', r'catch\s*\(',
                r'throw\s+new', r'\.catch\('
            ],
            'class_definitions': [
                r'class\s+\w+', r'def\s+__init__', r'@property', r'@staticmethod',
                r'@classmethod'
            ],
            'function_definitions': [
                r'def\s+\w+', r'function\s+\w+', r'=>\s*{', r'const\s+\w+\s*='
            ],
            'testing_patterns': [
                r'def\s+test_', r'@pytest\.', r'assert\s+', r'expect\(',
                r'it\(.*,.*=>', r'describe\('
            ],
            'import_patterns': [
                r'import\s+\w+', r'from\s+\w+\s+import', r'require\(',
                r'#include\s*<', r'using\s+namespace'
            ]
        }
    
    async def index_code_snippets(self, code_indexer) -> None:
        """Index code snippets from code indexer for semantic search"""
        start_time = time.time()
        logger.info("Indexing code snippets for semantic search...")
        
        # Clear existing data
        self.snippets.clear()
        self.snippet_ids.clear()
        
        # Extract snippets from all indexed files
        snippet_texts = []
        
        for file_path, file_info in code_indexer.file_cache.items():
            await self._extract_file_snippets(file_path, file_info, code_indexer)
        
        # Build text corpus for TF-IDF
        for snippet in self.snippets.values():
            snippet_texts.append(self._preprocess_code_text(snippet.content))
            self.snippet_ids.append(snippet.id)
        
        if snippet_texts:
            # Build TF-IDF matrix
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(snippet_texts)
            
            # Apply dimensionality reduction
            self.svd_transformer = TruncatedSVD(
                n_components=min(self.embedding_dim, self.tfidf_matrix.shape[1]),
                random_state=42
            )
            self.embeddings = self.svd_transformer.fit_transform(self.tfidf_matrix)
            
            # Perform clustering
            await self._cluster_snippets()
        
        self.search_stats['total_snippets'] = len(self.snippets)
        self.search_stats['last_index_time'] = time.time() - start_time
        
        logger.info(f"Indexed {len(self.snippets)} code snippets in {self.search_stats['last_index_time']:.2f}s")
    
    async def _extract_file_snippets(self, file_path: str, file_info, code_indexer):
        """Extract searchable snippets from a file"""
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            lines = content.splitlines()
            
            # Extract function snippets
            if file_path in code_indexer.function_cache:
                for func_info in code_indexer.function_cache[file_path]:
                    snippet_content = self._extract_snippet_content(
                        lines, func_info.line_start - 1, func_info.line_end - 1
                    )
                    
                    if snippet_content:
                        snippet = CodeSnippet(
                            id="",
                            content=snippet_content,
                            file_path=file_path,
                            language=file_info.language,
                            line_start=func_info.line_start,
                            line_end=func_info.line_end,
                            snippet_type="function",
                            context=self._extract_context(lines, func_info.line_start - 1, func_info.line_end - 1),
                            metadata={
                                'name': func_info.name,
                                'parameters': func_info.parameters,
                                'complexity': func_info.complexity,
                                'is_async': func_info.is_async,
                                'is_method': func_info.is_method,
                                'docstring': func_info.docstring
                            },
                            keywords=self._extract_keywords(snippet_content, file_info.language)
                        )
                        self.snippets[snippet.id] = snippet
            
            # Extract class snippets
            if file_path in code_indexer.class_cache:
                for class_info in code_indexer.class_cache[file_path]:
                    snippet_content = self._extract_snippet_content(
                        lines, class_info.line_start - 1, class_info.line_end - 1
                    )
                    
                    if snippet_content:
                        snippet = CodeSnippet(
                            id="",
                            content=snippet_content,
                            file_path=file_path,
                            language=file_info.language,
                            line_start=class_info.line_start,
                            line_end=class_info.line_end,
                            snippet_type="class",
                            context=self._extract_context(lines, class_info.line_start - 1, class_info.line_end - 1),
                            metadata={
                                'name': class_info.name,
                                'methods': class_info.methods,
                                'base_classes': class_info.base_classes,
                                'docstring': class_info.docstring,
                                'is_abstract': class_info.is_abstract
                            },
                            keywords=self._extract_keywords(snippet_content, file_info.language)
                        )
                        self.snippets[snippet.id] = snippet
            
            # Extract other interesting patterns (imports, API endpoints, etc.)
            await self._extract_pattern_snippets(lines, file_path, file_info)
            
        except Exception as e:
            logger.error(f"Error extracting snippets from {file_path}: {e}")
    
    def _extract_snippet_content(self, lines: List[str], start: int, end: int, max_lines: int = 100) -> str:
        """Extract snippet content with line limit"""
        start = max(0, start)
        end = min(len(lines), end)
        
        # Limit snippet size
        if end - start > max_lines:
            end = start + max_lines
        
        return '\n'.join(lines[start:end])
    
    def _extract_context(self, lines: List[str], start: int, end: int, context_lines: int = 5) -> str:
        """Extract surrounding context for a snippet"""
        context_start = max(0, start - context_lines)
        context_end = min(len(lines), end + context_lines)
        
        context_lines_list = []
        
        # Add context before
        if context_start < start:
            context_lines_list.extend(lines[context_start:start])
            context_lines_list.append("--- SNIPPET START ---")
        
        # Add context after
        if end < context_end:
            context_lines_list.append("--- SNIPPET END ---")
            context_lines_list.extend(lines[end:context_end])
        
        return '\n'.join(context_lines_list)
    
    def _extract_keywords(self, content: str, language: str) -> List[str]:
        """Extract keywords and important terms from code"""
        keywords = []
        
        # Language-specific keyword extraction
        if language == 'python':
            # Python keywords and patterns
            python_keywords = re.findall(r'\b(def|class|import|from|async|await|try|except|with|yield)\b', content)
            keywords.extend(python_keywords)
            
            # Function/class names
            func_names = re.findall(r'def\s+(\w+)', content)
            class_names = re.findall(r'class\s+(\w+)', content)
            keywords.extend(func_names + class_names)
            
        elif language in ['javascript', 'typescript']:
            # JavaScript/TypeScript keywords
            js_keywords = re.findall(r'\b(function|class|const|let|var|async|await|import|export)\b', content)
            keywords.extend(js_keywords)
            
            # Function names
            func_names = re.findall(r'function\s+(\w+)', content)
            arrow_funcs = re.findall(r'(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>', content)
            keywords.extend(func_names + arrow_funcs)
        
        # Common patterns across languages
        # API patterns
        api_patterns = re.findall(r'(GET|POST|PUT|DELETE|PATCH)', content, re.IGNORECASE)
        keywords.extend(api_patterns)
        
        # Database patterns
        db_patterns = re.findall(r'\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP)\b', content, re.IGNORECASE)
        keywords.extend(db_patterns)
        
        # Framework patterns
        framework_patterns = re.findall(r'\b(express|flask|django|react|vue|angular)\b', content, re.IGNORECASE)
        keywords.extend(framework_patterns)
        
        # Remove duplicates and return
        return list(set(keywords))
    
    async def _extract_pattern_snippets(self, lines: List[str], file_path: str, file_info):
        """Extract snippets based on interesting code patterns"""
        content = '\n'.join(lines)
        
        for pattern_name, pattern_regexes in self.code_patterns.items():
            for pattern_regex in pattern_regexes:
                matches = list(re.finditer(pattern_regex, content, re.IGNORECASE | re.MULTILINE))
                
                for match in matches:
                    # Find line numbers
                    start_pos = match.start()
                    line_start = content[:start_pos].count('\n')
                    
                    # Extract surrounding lines
                    context_start = max(0, line_start - 5)
                    context_end = min(len(lines), line_start + 10)
                    
                    snippet_content = '\n'.join(lines[context_start:context_end])
                    
                    if len(snippet_content.strip()) > 20:  # Minimum content length
                        snippet = CodeSnippet(
                            id="",
                            content=snippet_content,
                            file_path=file_path,
                            language=file_info.language,
                            line_start=context_start + 1,
                            line_end=context_end,
                            snippet_type=pattern_name,
                            metadata={
                                'pattern': pattern_regex,
                                'match_text': match.group(0)
                            },
                            keywords=self._extract_keywords(snippet_content, file_info.language)
                        )
                        self.snippets[snippet.id] = snippet
    
    def _preprocess_code_text(self, code: str) -> str:
        """Preprocess code text for better TF-IDF representation"""
        # Remove comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)  # Python comments
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)  # C-style comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)  # Multi-line comments
        
        # Extract identifiers and keywords
        identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)
        
        # Split camelCase and snake_case
        processed_tokens = []
        for identifier in identifiers:
            # Split camelCase
            camel_split = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', identifier)
            camel_split = re.sub('([a-z0-9])([A-Z])', r'\1 \2', camel_split)
            
            # Split snake_case
            snake_split = camel_split.replace('_', ' ')
            
            processed_tokens.append(snake_split.lower())
        
        # Add string literals (for API endpoints, error messages, etc.)
        strings = re.findall(r'["\']([^"\']{3,})["\']', code)
        processed_tokens.extend([s.lower() for s in strings if len(s) > 3])
        
        return ' '.join(processed_tokens)
    
    async def _cluster_snippets(self, n_clusters: int = None):
        """Cluster code snippets for better organization"""
        if self.embeddings is None or len(self.embeddings) < 5:
            return
        
        if n_clusters is None:
            # Determine optimal number of clusters
            n_clusters = min(20, max(5, len(self.embeddings) // 10))
        
        try:
            self.clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            self.cluster_labels = self.clusterer.fit_predict(self.embeddings)
            
            logger.debug(f"Clustered {len(self.snippets)} snippets into {n_clusters} clusters")
            
        except Exception as e:
            logger.error(f"Error clustering snippets: {e}")
            self.cluster_labels = None
    
    async def search_similar_code(
        self, 
        query: str, 
        top_k: int = 20, 
        include_context: bool = False,
        filter_language: str = None,
        filter_type: str = None
    ) -> List[SearchResult]:
        """Search for similar code snippets"""
        start_time = time.time()
        
        # Check cache
        cache_key = self._create_search_cache_key(query, top_k, filter_language, filter_type)
        if self.enable_caching and cache_key in self.search_cache:
            self.search_stats['cache_hits'] += 1
            return self.search_cache[cache_key]
        
        if not self.snippets or self.embeddings is None:
            return []
        
        # Preprocess query
        processed_query = self._preprocess_code_text(query)
        
        # Get query embedding
        query_tfidf = self.tfidf_vectorizer.transform([processed_query])
        query_embedding = self.svd_transformer.transform(query_tfidf)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get more for filtering
        
        results = []
        for idx in top_indices:
            if idx < len(self.snippet_ids):
                snippet_id = self.snippet_ids[idx]
                snippet = self.snippets[snippet_id]
                similarity_score = similarities[idx]
                
                # Apply filters
                if filter_language and snippet.language != filter_language:
                    continue
                if filter_type and snippet.snippet_type != filter_type:
                    continue
                
                # Skip very low similarity scores
                if similarity_score < 0.1:
                    continue
                
                # Determine match type
                match_type = self._determine_match_type(query, snippet, similarity_score)
                
                # Find matched keywords
                matched_keywords = self._find_matched_keywords(query, snippet)
                
                # Generate explanation
                explanation = self._generate_match_explanation(query, snippet, match_type, matched_keywords)
                
                result = SearchResult(
                    snippet=snippet,
                    similarity_score=similarity_score,
                    match_type=match_type,
                    matched_keywords=matched_keywords,
                    explanation=explanation
                )
                
                results.append(result)
                
                if len(results) >= top_k:
                    break
        
        # Sort by similarity score
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Cache results
        if self.enable_caching:
            self.search_cache[cache_key] = results
            if len(self.search_cache) > 1000:  # Limit cache size
                oldest_key = next(iter(self.search_cache))
                del self.search_cache[oldest_key]
        
        # Update stats
        self.search_stats['total_searches'] += 1
        search_time = time.time() - start_time
        self._update_search_stats(search_time)
        
        logger.debug(f"Search completed in {search_time:.3f}s, found {len(results)} results")
        return results
    
    def _determine_match_type(self, query: str, snippet: CodeSnippet, similarity_score: float) -> str:
        """Determine the type of match"""
        query_lower = query.lower()
        content_lower = snippet.content.lower()
        
        # Exact keyword match
        if any(word in content_lower for word in query_lower.split() if len(word) > 2):
            if similarity_score > 0.8:
                return "exact_high"
            else:
                return "exact_medium"
        
        # Structural similarity
        if similarity_score > 0.7:
            return "structural"
        
        # Semantic similarity
        if similarity_score > 0.5:
            return "semantic"
        
        # Pattern similarity
        if similarity_score > 0.3:
            return "pattern"
        
        return "weak"
    
    def _find_matched_keywords(self, query: str, snippet: CodeSnippet) -> List[str]:
        """Find keywords that matched between query and snippet"""
        query_words = set(query.lower().split())
        snippet_keywords = set(kw.lower() for kw in snippet.keywords)
        content_words = set(snippet.content.lower().split())
        
        # Direct keyword matches
        direct_matches = list(query_words & snippet_keywords)
        
        # Content word matches
        content_matches = list(query_words & content_words)
        
        # Combine and deduplicate
        all_matches = list(set(direct_matches + content_matches))
        
        # Filter out very short words
        meaningful_matches = [word for word in all_matches if len(word) > 2]
        
        return meaningful_matches[:10]  # Limit to top 10
    
    def _generate_match_explanation(
        self, 
        query: str, 
        snippet: CodeSnippet, 
        match_type: str, 
        matched_keywords: List[str]
    ) -> str:
        """Generate explanation for why this snippet matched"""
        explanations = []
        
        if match_type.startswith("exact"):
            explanations.append(f"Contains exact keywords: {', '.join(matched_keywords[:3])}")
        
        if snippet.snippet_type in ['function', 'method']:
            name = snippet.metadata.get('name', 'unknown')
            explanations.append(f"Function/method: {name}")
        elif snippet.snippet_type == 'class':
            name = snippet.metadata.get('name', 'unknown')
            explanations.append(f"Class: {name}")
        
        if snippet.language:
            explanations.append(f"Language: {snippet.language}")
        
        if match_type == "structural":
            explanations.append("Similar code structure")
        elif match_type == "semantic":
            explanations.append("Similar functionality")
        elif match_type == "pattern":
            explanations.append("Similar code patterns")
        
        return " | ".join(explanations) if explanations else "Similar code found"
    
    def group_by_similarity(self, search_results: List[SearchResult], similarity_threshold: float = 0.8) -> Dict[str, List[SearchResult]]:
        """Group search results by similarity"""
        groups = {}
        
        if self.cluster_labels is None:
            # Simple grouping by snippet type
            type_groups = defaultdict(list)
            for result in search_results:
                type_groups[result.snippet.snippet_type].append(result)
            
            return dict(type_groups)
        
        # Group by clusters
        snippet_id_to_cluster = {}
        for i, cluster_label in enumerate(self.cluster_labels):
            if i < len(self.snippet_ids):
                snippet_id_to_cluster[self.snippet_ids[i]] = cluster_label
        
        cluster_groups = defaultdict(list)
        for result in search_results:
            cluster_id = snippet_id_to_cluster.get(result.snippet.id, -1)
            cluster_groups[f"Cluster_{cluster_id}"].append(result)
        
        return dict(cluster_groups)
    
    async def find_code_patterns(self, pattern_name: str, language: str = None) -> List[SearchResult]:
        """Find code snippets matching specific patterns"""
        results = []
        
        for snippet in self.snippets.values():
            if language and snippet.language != language:
                continue
            
            if snippet.snippet_type == pattern_name:
                result = SearchResult(
                    snippet=snippet,
                    similarity_score=1.0,
                    match_type="pattern_exact",
                    explanation=f"Exact pattern match: {pattern_name}"
                )
                results.append(result)
        
        return results
    
    async def find_similar_functions(self, function_name: str, top_k: int = 10) -> List[SearchResult]:
        """Find functions similar to a given function name"""
        # Simple implementation - could be enhanced with more sophisticated matching
        query = f"function {function_name}"
        results = await self.search_similar_code(
            query, 
            top_k=top_k, 
            filter_type="function"
        )
        return results
    
    async def get_code_suggestions(self, context: str, intent: str = "implementation") -> List[SearchResult]:
        """Get code suggestions based on context and intent"""
        # Enhance query based on intent
        if intent == "implementation":
            query = f"implement {context}"
        elif intent == "testing":
            query = f"test {context}"
        elif intent == "error_handling":
            query = f"error handling {context}"
        else:
            query = context
        
        results = await self.search_similar_code(query, top_k=15)
        
        # Filter and rank based on intent
        filtered_results = []
        for result in results:
            if intent == "testing" and "test" in result.snippet.snippet_type:
                result.similarity_score *= 1.5  # Boost test results for testing intent
            elif intent == "error_handling" and any(
                keyword in result.snippet.content.lower() 
                for keyword in ['try', 'catch', 'except', 'error', 'exception']
            ):
                result.similarity_score *= 1.3
            
            filtered_results.append(result)
        
        # Re-sort by adjusted scores
        filtered_results.sort(key=lambda x: x.similarity_score, reverse=True)
        return filtered_results
    
    async def update_embeddings(self):
        """Update embeddings (called when index is updated)"""
        if self.snippets:
            snippet_texts = []
            self.snippet_ids.clear()
            
            for snippet in self.snippets.values():
                snippet_texts.append(self._preprocess_code_text(snippet.content))
                self.snippet_ids.append(snippet.id)
            
            if snippet_texts:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(snippet_texts)
                self.embeddings = self.svd_transformer.fit_transform(self.tfidf_matrix)
                await self._cluster_snippets()
    
    def _create_search_cache_key(self, query: str, top_k: int, filter_language: str, filter_type: str) -> str:
        """Create cache key for search query"""
        key_parts = [
            query[:100],  # Truncate query
            str(top_k),
            filter_language or "any",
            filter_type or "any"
        ]
        return hashlib.md5("|".join(key_parts).encode()).hexdigest()
    
    def _update_search_stats(self, search_time: float):
        """Update search performance statistics"""
        total_searches = self.search_stats['total_searches']
        current_avg = self.search_stats['avg_search_time']
        
        # Calculate new average
        new_avg = (current_avg * (total_searches - 1) + search_time) / total_searches
        self.search_stats['avg_search_time'] = new_avg
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search performance statistics"""
        cache_hit_rate = self.search_stats['cache_hits'] / max(1, self.search_stats['total_searches'])
        
        return {
            'total_snippets': self.search_stats['total_snippets'],
            'total_searches': self.search_stats['total_searches'],
            'cache_hits': self.search_stats['cache_hits'],
            'cache_hit_rate': cache_hit_rate,
            'avg_search_time': self.search_stats['avg_search_time'],
            'last_index_time': self.search_stats['last_index_time'],
            'embedding_dimensions': self.embedding_dim,
            'clusters': len(set(self.cluster_labels)) if self.cluster_labels is not None else 0
        }
    
    async def clear_cache(self):
        """Clear all caches"""
        self.search_cache.clear()
        self.similarity_cache.clear()
        logger.info("Semantic search caches cleared")
    
    async def save_index(self, file_path: str):
        """Save the search index to file"""
        try:
            index_data = {
                'snippets': {
                    sid: {
                        'content': s.content,
                        'file_path': s.file_path,
                        'language': s.language,
                        'line_start': s.line_start,
                        'line_end': s.line_end,
                        'snippet_type': s.snippet_type,
                        'context': s.context,
                        'metadata': s.metadata,
                        'keywords': s.keywords
                    } for sid, s in self.snippets.items()
                },
                'snippet_ids': self.snippet_ids,
                'search_stats': self.search_stats
            }
            
            # Save embeddings separately (binary format)
            embeddings_file = file_path.replace('.json', '_embeddings.npy')
            if self.embeddings is not None:
                np.save(embeddings_file, self.embeddings)
            
            # Save main index
            with open(file_path, 'w') as f:
                json.dump(index_data, f, indent=2)
            
            # Save TF-IDF vectorizer
            vectorizer_file = file_path.replace('.json', '_vectorizer.pkl')
            with open(vectorizer_file, 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
            
            # Save SVD transformer
            if self.svd_transformer:
                svd_file = file_path.replace('.json', '_svd.pkl')
                with open(svd_file, 'wb') as f:
                    pickle.dump(self.svd_transformer, f)
            
            logger.info(f"Search index saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving search index: {e}")
    
    async def load_index(self, file_path: str):
        """Load the search index from file"""
        try:
            # Load main index
            with open(file_path, 'r') as f:
                index_data = json.load(f)
            
            # Restore snippets
            self.snippets.clear()
            for sid, snippet_data in index_data.get('snippets', {}).items():
                snippet = CodeSnippet(
                    id=sid,
                    content=snippet_data['content'],
                    file_path=snippet_data['file_path'],
                    language=snippet_data['language'],
                    line_start=snippet_data['line_start'],
                    line_end=snippet_data['line_end'],
                    snippet_type=snippet_data['snippet_type'],
                    context=snippet_data.get('context', ''),
                    metadata=snippet_data.get('metadata', {}),
                    keywords=snippet_data.get('keywords', [])
                )
                self.snippets[sid] = snippet
            
            self.snippet_ids = index_data.get('snippet_ids', [])
            self.search_stats.update(index_data.get('search_stats', {}))
            
            # Load embeddings
            embeddings_file = file_path.replace('.json', '_embeddings.npy')
            if Path(embeddings_file).exists():
                self.embeddings = np.load(embeddings_file)
            
            # Load TF-IDF vectorizer
            vectorizer_file = file_path.replace('.json', '_vectorizer.pkl')
            if Path(vectorizer_file).exists():
                with open(vectorizer_file, 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
            
            # Load SVD transformer
            svd_file = file_path.replace('.json', '_svd.pkl')
            if Path(svd_file).exists():
                with open(svd_file, 'rb') as f:
                    self.svd_transformer = pickle.load(f)
            
            # Rebuild clusters
            if self.embeddings is not None:
                await self._cluster_snippets()
            
            logger.info(f"Search index loaded from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading search index: {e}")
    
    def export_search_results(self, results: List[SearchResult], format: str = 'json') -> str:
        """Export search results in specified format"""
        if format == 'json':
            data = []
            for result in results:
                data.append({
                    'snippet_id': result.snippet.id,
                    'file_path': result.snippet.file_path,
                    'language': result.snippet.language,
                    'snippet_type': result.snippet.snippet_type,
                    'similarity_score': result.similarity_score,
                    'match_type': result.match_type,
                    'matched_keywords': result.matched_keywords,
                    'explanation': result.explanation,
                    'content_preview': result.snippet.content[:200] + '...' if len(result.snippet.content) > 200 else result.snippet.content
                })
            return json.dumps(data, indent=2)
        
        elif format == 'markdown':
            md_lines = ["# Search Results\n"]
            for i, result in enumerate(results, 1):
                md_lines.extend([
                    f"## Result {i} (Score: {result.similarity_score:.3f})",
                    f"**File:** {result.snippet.file_path}",
                    f"**Language:** {result.snippet.language}",
                    f"**Type:** {result.snippet.snippet_type}",
                    f"**Explanation:** {result.explanation}",
                    f"**Keywords:** {', '.join(result.matched_keywords)}",
                    f"```{result.snippet.language}",
                    result.snippet.content[:500] + ('...' if len(result.snippet.content) > 500 else ''),
                    "```",
                    ""
                ])
            return '\n'.join(md_lines)
        
        return "Unsupported format"