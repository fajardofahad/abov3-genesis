# ABOV3 Genesis - Context-Aware Comprehension Module

## Overview

The Context-Aware Comprehension Module (Module 2) provides intelligent codebase understanding and analysis capabilities for ABOV3 Genesis. It can efficiently index, analyze, and reason over entire repositories including large monorepos (up to 1M+ lines of code) to answer questions, provide refactoring suggestions, and enable intelligent code navigation.

## Key Features

### 🧠 **Intelligent Code Comprehension**
- **Multi-mode Analysis**: Quick scan, deep analysis, semantic search, Q&A, refactoring analysis, and monorepo handling
- **Context-Aware Responses**: Understands code relationships, patterns, and architectural decisions
- **Cross-language Support**: Python, JavaScript, TypeScript, Java, C++, Go, Rust, and more

### 🗂️ **Advanced Code Indexing**
- **AST-based Parsing**: Deep structural analysis using Abstract Syntax Trees
- **Incremental Updates**: Efficient re-indexing of only changed files
- **Multi-language Support**: Language-specific parsing for accurate analysis
- **Persistent Storage**: SQLite-based indexing with fast retrieval

### 🕸️ **Knowledge Graph Construction**
- **Code Relationships**: Maps dependencies, inheritance, function calls, and imports
- **Entity Extraction**: Identifies classes, functions, modules, and their relationships
- **Graph Analytics**: Finds patterns, communities, and architectural insights
- **Subgraph Analysis**: Focus analysis on specific code regions

### 🔍 **Semantic Search Engine**
- **Vector Embeddings**: TF-IDF and SVD-based code similarity matching
- **Pattern Recognition**: Identifies common coding patterns and anti-patterns
- **Similarity Clustering**: Groups related code for better organization
- **Multi-modal Search**: Text, structure, and semantic similarity

### 🔧 **Intelligent Refactoring Suggestions**
- **Code Smell Detection**: Identifies maintainability issues and technical debt
- **Refactoring Recommendations**: Suggests specific improvements with rationale
- **Priority Ranking**: Orders suggestions by impact and effort required
- **Multiple Refactoring Types**: Extract method, split class, simplify conditionals, and more

### ⚡ **Performance Optimization**
- **Large Codebase Support**: Handles repositories with 1M+ lines efficiently
- **Memory Management**: Intelligent caching and memory usage optimization
- **Parallel Processing**: Multi-threaded and batch processing for speed
- **Streaming Analysis**: Memory-efficient processing for very large projects

## Architecture

```
context_aware/
├── core/                    # Core comprehension engine
│   └── comprehension_engine.py
├── indexing/               # Code indexing and AST parsing
│   └── code_indexer.py
├── knowledge_graph/        # Code relationship mapping
│   └── graph_builder.py
├── semantic_search/        # Vector-based code search
│   └── search_engine.py
├── analysis/              # Code analysis and metrics
│   └── code_analyzer.py
├── refactoring/           # Refactoring suggestions
│   └── suggestion_engine.py
├── utils/                 # Performance optimization
│   └── performance_optimizer.py
└── demo.py               # Interactive demonstration
```

## Quick Start

### Basic Usage

```python
import asyncio
from abov3.modules.context_aware import ComprehensionEngine, ComprehensionRequest, ComprehensionMode
from abov3.core.ollama_client import OllamaClient

async def analyze_codebase():
    # Initialize Ollama client
    ollama_client = OllamaClient()
    await ollama_client.connect()
    
    # Create comprehension engine
    engine = ComprehensionEngine(
        workspace_path="/path/to/your/project",
        ollama_client=ollama_client,
        model_name="deepseek-coder"
    )
    
    await engine.initialize()
    
    # Ask questions about your code
    request = ComprehensionRequest(
        query="What are the main components of this codebase?",
        mode=ComprehensionMode.DEEP_ANALYSIS
    )
    
    result = await engine.comprehend(request)
    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence_score:.2f}")
    print(f"Files analyzed: {len(result.source_files)}")

# Run the example
asyncio.run(analyze_codebase())
```

### Interactive Demo

Run the interactive demo to explore capabilities:

```bash
cd abov3/modules/context_aware
python demo.py /path/to/your/project
```

Or run from Python:

```python
from abov3.modules.context_aware.demo import run_interactive_session
import asyncio

asyncio.run(run_interactive_session("/path/to/your/project"))
```

## Comprehension Modes

### 1. **Quick Scan** (`QUICK_SCAN`)
Fast overview of codebase structure and key components.

```python
request = ComprehensionRequest(
    query="Give me a quick overview of this project",
    mode=ComprehensionMode.QUICK_SCAN,
    max_files=20
)
```

### 2. **Deep Analysis** (`DEEP_ANALYSIS`)
Comprehensive analysis with architectural insights and relationships.

```python
request = ComprehensionRequest(
    query="Analyze the architecture and design patterns",
    mode=ComprehensionMode.DEEP_ANALYSIS,
    context_depth=3
)
```

### 3. **Semantic Search** (`SEMANTIC_SEARCH`)
Find similar code patterns and implementations.

```python
request = ComprehensionRequest(
    query="Find error handling patterns",
    mode=ComprehensionMode.SEMANTIC_SEARCH,
    max_files=50
)
```

### 4. **Q&A Mode** (`QA_MODE`)
Answer specific questions about the codebase.

```python
request = ComprehensionRequest(
    query="How does the authentication system work?",
    mode=ComprehensionMode.QA_MODE
)
```

### 5. **Refactoring Analysis** (`REFACTOR_MODE`)
Identify improvement opportunities and suggest refactorings.

```python
request = ComprehensionRequest(
    query="What can be improved in this codebase?",
    mode=ComprehensionMode.REFACTOR_MODE,
    target_paths=["src/main.py", "src/utils.py"]
)
```

### 6. **Monorepo Mode** (`MONOREPO_MODE`)
Handle large monorepos with hierarchical analysis.

```python
request = ComprehensionRequest(
    query="Analyze this monorepo structure",
    mode=ComprehensionMode.MONOREPO_MODE,
    max_files=1000
)
```

## Advanced Features

### Code Indexing

```python
from abov3.modules.context_aware import CodeIndexer

indexer = CodeIndexer("/path/to/project")
await indexer.initialize()
await indexer.update_index()

# Find relevant files
results = await indexer.find_relevant_files("authentication", max_files=10)

# Search for specific functions
functions = await indexer.find_functions(["login", "authenticate"])

# Get file summary
summary = await indexer.get_file_summary("src/auth.py")
```

### Knowledge Graph Analysis

```python
from abov3.modules.context_aware import KnowledgeGraphBuilder

graph_builder = KnowledgeGraphBuilder()
await graph_builder.initialize()

# Build graph from code index
graph = await graph_builder.build_graph_from_index(indexer)

# Extract key concepts and relationships
concepts = graph_builder.extract_concepts(graph)
relationships = graph_builder.extract_relationships(graph)

# Find paths between entities
paths = await graph_builder.find_shortest_paths(entity1_id, entity2_id)
```

### Semantic Search

```python
from abov3.modules.context_aware import SemanticSearchEngine

search_engine = SemanticSearchEngine()
await search_engine.initialize()
await search_engine.index_code_snippets(indexer)

# Search for similar code
results = await search_engine.search_similar_code(
    "database connection handling",
    top_k=10,
    filter_language="python"
)

# Group results by similarity
grouped = search_engine.group_by_similarity(results)
```

### Code Analysis

```python
from abov3.modules.context_aware import CodeAnalyzer

analyzer = CodeAnalyzer()

# Analyze a single file
result = await analyzer.analyze_file(
    "src/complex_module.py",
    include_ast=True,
    include_metrics=True,
    include_issues=True
)

print(f"Complexity: {result.metrics.cyclomatic_complexity}")
print(f"Issues found: {len(result.issues)}")
print(f"Maintainability: {result.metrics.maintainability_index:.1f}")
```

### Refactoring Suggestions

```python
from abov3.modules.context_aware import RefactoringSuggestionEngine

refactoring_engine = RefactoringSuggestionEngine()

# Analyze file for refactoring opportunities
suggestions = await refactoring_engine.analyze_file("src/legacy_code.py")

# Prioritize suggestions
prioritized = refactoring_engine.prioritize_suggestions(suggestions)

# Filter by criteria
high_priority = refactoring_engine.filter_suggestions(
    suggestions,
    min_priority=Priority.HIGH,
    types=[RefactoringType.EXTRACT_METHOD, RefactoringType.SPLIT_LARGE_CLASS]
)
```

## Performance Optimization

For large codebases (1M+ lines), the module includes advanced performance optimizations:

### Memory Management
```python
from abov3.modules.context_aware.utils import PerformanceOptimizer

optimizer = PerformanceOptimizer(max_memory_mb=4096)

# Optimize file processing
results = await optimizer.optimize_file_processing(
    files=large_file_list,
    processor_func=analysis_function,
    use_hierarchy=True,
    use_streaming=True
)
```

### Batch Processing
```python
# Process files in optimized batches
batch_processor = BatchProcessor(batch_size=100, max_workers=8)
results = await batch_processor.process_files_batch(
    files, processor_function, use_multiprocessing=True
)
```

### Hierarchical Indexing
```python
# Build hierarchy for large monorepos
hierarchical_indexer = HierarchicalIndexer()
hierarchy = await hierarchical_indexer.build_hierarchy(repo_root)
priority_dirs = hierarchical_indexer.get_priority_directories(hierarchy)
```

## Configuration Options

### ComprehensionRequest Parameters

- **query**: The question or analysis request
- **mode**: Analysis mode (see modes above)
- **target_paths**: Specific files/directories to analyze
- **include_tests**: Whether to include test files (default: True)
- **include_docs**: Whether to include documentation files (default: True)
- **max_files**: Maximum number of files to analyze (default: 1000)
- **max_lines_per_file**: Maximum lines per file (default: 5000)
- **context_depth**: How deep to traverse dependencies (default: 3)
- **use_cache**: Whether to use caching (default: True)
- **priority_languages**: Languages to prioritize in analysis
- **exclude_patterns**: File patterns to exclude

### Performance Tuning

```python
# For large repositories
engine = ComprehensionEngine(
    workspace_path=repo_path,
    ollama_client=client,
    model_name="deepseek-coder",
    enable_caching=True,
    max_cache_size=2000  # Increase cache size
)

# Optimize for memory usage
request = ComprehensionRequest(
    query="Analyze main components",
    mode=ComprehensionMode.MONOREPO_MODE,
    max_files=500,  # Limit files for memory efficiency
    context_depth=2  # Reduce depth for faster processing
)
```

## Integration with ABOV3 Genesis

The Context-Aware Comprehension module integrates seamlessly with other ABOV3 components:

### With Natural Language to Code (Module 1)
```python
# Use comprehension results to inform code generation
comprehension_result = await engine.comprehend(analysis_request)
context_info = comprehension_result.answer

generation_request = CodeGenerationRequest(
    description="Create a new authentication handler",
    context=context_info,  # Use comprehension insights
    style_guide=project_style
)
```

### With Ollama Integration
```python
# The module uses existing Ollama client and models
ollama_client = OllamaClient()
await ollama_client.connect()

# Automatically selects best available model
engine = ComprehensionEngine(
    workspace_path=workspace,
    ollama_client=ollama_client,
    model_name="deepseek-coder"  # Or any available coding model
)
```

## Testing

Run the comprehensive test suite:

```bash
# Run all context-aware tests
pytest tests/test_context_aware.py -v

# Run specific test categories
pytest tests/test_context_aware.py::TestCodeIndexer -v
pytest tests/test_context_aware.py::TestKnowledgeGraphBuilder -v
pytest tests/test_context_aware.py::TestSemanticSearchEngine -v
```

## Performance Benchmarks

On a typical development machine with a 100k+ line codebase:

- **Initial Indexing**: ~2-5 minutes
- **Incremental Updates**: ~10-30 seconds  
- **Quick Scan Query**: ~1-3 seconds
- **Deep Analysis Query**: ~5-15 seconds
- **Semantic Search**: ~2-8 seconds
- **Memory Usage**: ~500MB-2GB (depending on codebase size)

For 1M+ line monorepos:
- **Initial Indexing**: ~15-30 minutes (with optimizations)
- **Query Response**: ~10-60 seconds
- **Memory Usage**: ~2-8GB (with streaming optimizations)

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   ```python
   # Enable streaming for very large codebases
   optimizer = PerformanceOptimizer(max_memory_mb=2048)
   # Use hierarchical processing
   request.mode = ComprehensionMode.MONOREPO_MODE
   ```

2. **Slow Indexing**
   ```python
   # Exclude unnecessary directories
   request.exclude_patterns = [
       "node_modules", ".git", "__pycache__", 
       "build", "dist", ".venv", "venv"
   ]
   ```

3. **Model Not Available**
   ```python
   # Check available models
   models = await ollama_client.list_models()
   print("Available models:", [m['name'] for m in models])
   ```

4. **Large File Issues**
   ```python
   # Limit file size and line count
   request.max_lines_per_file = 2000
   request.max_files = 200
   ```

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug mode in components
indexer = CodeIndexer(workspace, enable_caching=True)
# Check logs for detailed processing information
```

## Contributing

To extend the Context-Aware Comprehension module:

1. **Adding New Languages**: Extend `CodeIndexer.LANGUAGE_EXTENSIONS` and add language-specific parsing
2. **New Analysis Types**: Add new `ComprehensionMode` values and corresponding analysis logic
3. **Custom Refactoring Rules**: Extend `RefactoringSuggestionEngine` with new patterns and rules
4. **Performance Optimizations**: Contribute to `PerformanceOptimizer` for better large-scale handling

## API Reference

For detailed API documentation, see the docstrings in each module file. Key classes:

- `ComprehensionEngine`: Main orchestration engine
- `CodeIndexer`: File indexing and AST parsing
- `KnowledgeGraphBuilder`: Code relationship mapping
- `SemanticSearchEngine`: Vector-based similarity search
- `CodeAnalyzer`: Code quality and metrics analysis
- `RefactoringSuggestionEngine`: Refactoring recommendations
- `PerformanceOptimizer`: Large-scale optimization utilities

## License

Part of ABOV3 Genesis - AI-Powered Development Platform
Copyright (c) 2024 ABOV3 Team