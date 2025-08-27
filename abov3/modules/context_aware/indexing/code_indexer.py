"""
ABOV3 Genesis - Code Indexer
High-performance code indexing with AST parsing for large codebases
"""

import ast
import asyncio
import logging
import json
import hashlib
import sqlite3
import time
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import aiosqlite
from collections import defaultdict, Counter
import re
import mimetypes

logger = logging.getLogger(__name__)

@dataclass
class FileInfo:
    """Information about a code file"""
    path: str
    language: str
    size: int
    lines: int
    hash: str
    last_modified: float
    functions: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    complexity_score: float = 0.0
    maintainability_index: float = 0.0
    summary: str = ""
    ast_data: Optional[Dict] = None

@dataclass
class FunctionInfo:
    """Information about a function"""
    name: str
    file_path: str
    line_start: int
    line_end: int
    parameters: List[str]
    return_type: Optional[str]
    docstring: Optional[str]
    complexity: int
    calls: List[str] = field(default_factory=list)
    is_async: bool = False
    is_method: bool = False
    class_name: Optional[str] = None

@dataclass
class ClassInfo:
    """Information about a class"""
    name: str
    file_path: str
    line_start: int
    line_end: int
    methods: List[str]
    properties: List[str]
    base_classes: List[str]
    docstring: Optional[str]
    is_abstract: bool = False

class CodeIndexer:
    """
    High-performance code indexer with AST parsing capabilities
    Efficiently indexes large codebases (1M+ lines) with incremental updates
    """
    
    # Supported file extensions and their languages
    LANGUAGE_EXTENSIONS = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'jsx',
        '.tsx': 'tsx',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.hpp': 'cpp',
        '.cs': 'csharp',
        '.go': 'go',
        '.rs': 'rust',
        '.php': 'php',
        '.rb': 'ruby',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.r': 'r',
        '.sql': 'sql',
        '.sh': 'bash',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.json': 'json',
        '.xml': 'xml',
        '.html': 'html',
        '.css': 'css',
        '.md': 'markdown',
        '.rst': 'rst'
    }
    
    # Binary and ignore patterns
    IGNORE_PATTERNS = [
        r'\.git/',
        r'__pycache__/',
        r'node_modules/',
        r'\.venv/',
        r'venv/',
        r'build/',
        r'dist/',
        r'\.next/',
        r'\.nuxt/',
        r'target/',
        r'\.idea/',
        r'\.vscode/',
        r'\.DS_Store',
        r'\.pyc$',
        r'\.pyo$',
        r'\.pyd$',
        r'\.so$',
        r'\.dylib$',
        r'\.dll$',
        r'\.exe$',
        r'\.bin$',
        r'\.img$',
        r'\.iso$',
        r'\.jpg$',
        r'\.jpeg$',
        r'\.png$',
        r'\.gif$',
        r'\.svg$',
        r'\.ico$',
        r'\.pdf$',
        r'\.zip$',
        r'\.tar$',
        r'\.gz$',
        r'\.rar$',
        r'\.7z$'
    ]
    
    def __init__(self, workspace_path: str, enable_caching: bool = True, max_workers: int = None):
        self.workspace_path = Path(workspace_path)
        self.enable_caching = enable_caching
        self.max_workers = max_workers or min(32, (asyncio.get_event_loop()._default_executor._max_workers or 4))
        
        # Database for persistent indexing
        self.db_path = self.workspace_path / '.abov3' / 'code_index.db'
        self.db_path.parent.mkdir(exist_ok=True)
        
        # In-memory caches
        self.file_cache: Dict[str, FileInfo] = {}
        self.function_cache: Dict[str, List[FunctionInfo]] = {}
        self.class_cache: Dict[str, List[ClassInfo]] = {}
        
        # Performance tracking
        self.indexing_stats = {
            'total_files': 0,
            'indexed_files': 0,
            'skipped_files': 0,
            'total_lines': 0,
            'indexing_time': 0.0,
            'last_index_time': 0.0
        }
        
        # Compiled regex patterns for performance
        self.ignore_patterns = [re.compile(pattern) for pattern in self.IGNORE_PATTERNS]
        
        logger.info(f"CodeIndexer initialized for workspace: {workspace_path}")
    
    async def initialize(self):
        """Initialize the code indexer"""
        logger.info("Initializing CodeIndexer...")
        
        # Create database tables
        await self._create_database_schema()
        
        # Load existing index from database
        await self._load_index_from_db()
        
        logger.info(f"CodeIndexer initialized with {len(self.file_cache)} cached files")
    
    async def _create_database_schema(self):
        """Create database schema for persistent storage"""
        async with aiosqlite.connect(str(self.db_path)) as db:
            # Files table
            await db.execute('''
                CREATE TABLE IF NOT EXISTS files (
                    path TEXT PRIMARY KEY,
                    language TEXT,
                    size INTEGER,
                    lines INTEGER,
                    hash TEXT,
                    last_modified REAL,
                    functions TEXT,
                    classes TEXT,
                    imports TEXT,
                    exports TEXT,
                    dependencies TEXT,
                    complexity_score REAL,
                    maintainability_index REAL,
                    summary TEXT,
                    ast_data TEXT,
                    indexed_at REAL
                )
            ''')
            
            # Functions table
            await db.execute('''
                CREATE TABLE IF NOT EXISTS functions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    file_path TEXT,
                    line_start INTEGER,
                    line_end INTEGER,
                    parameters TEXT,
                    return_type TEXT,
                    docstring TEXT,
                    complexity INTEGER,
                    calls TEXT,
                    is_async BOOLEAN,
                    is_method BOOLEAN,
                    class_name TEXT,
                    FOREIGN KEY (file_path) REFERENCES files (path)
                )
            ''')
            
            # Classes table
            await db.execute('''
                CREATE TABLE IF NOT EXISTS classes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    file_path TEXT,
                    line_start INTEGER,
                    line_end INTEGER,
                    methods TEXT,
                    properties TEXT,
                    base_classes TEXT,
                    docstring TEXT,
                    is_abstract BOOLEAN,
                    FOREIGN KEY (file_path) REFERENCES files (path)
                )
            ''')
            
            # Create indexes for performance
            await db.execute('CREATE INDEX IF NOT EXISTS idx_files_language ON files(language)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_files_hash ON files(hash)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_functions_name ON functions(name)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_functions_file ON functions(file_path)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_classes_name ON classes(name)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_classes_file ON classes(file_path)')
            
            await db.commit()
    
    async def _load_index_from_db(self):
        """Load existing index from database"""
        async with aiosqlite.connect(str(self.db_path)) as db:
            # Load files
            async with db.execute('SELECT * FROM files') as cursor:
                async for row in cursor:
                    file_info = FileInfo(
                        path=row[0],
                        language=row[1],
                        size=row[2],
                        lines=row[3],
                        hash=row[4],
                        last_modified=row[5],
                        functions=json.loads(row[6]) if row[6] else [],
                        classes=json.loads(row[7]) if row[7] else [],
                        imports=json.loads(row[8]) if row[8] else [],
                        exports=json.loads(row[9]) if row[9] else [],
                        dependencies=json.loads(row[10]) if row[10] else [],
                        complexity_score=row[11] or 0.0,
                        maintainability_index=row[12] or 0.0,
                        summary=row[13] or "",
                        ast_data=json.loads(row[14]) if row[14] else None
                    )
                    self.file_cache[row[0]] = file_info
            
            # Load functions
            async with db.execute('SELECT * FROM functions') as cursor:
                async for row in cursor:
                    func_info = FunctionInfo(
                        name=row[1],
                        file_path=row[2],
                        line_start=row[3],
                        line_end=row[4],
                        parameters=json.loads(row[5]) if row[5] else [],
                        return_type=row[6],
                        docstring=row[7],
                        complexity=row[8] or 0,
                        calls=json.loads(row[9]) if row[9] else [],
                        is_async=bool(row[10]),
                        is_method=bool(row[11]),
                        class_name=row[12]
                    )
                    
                    if row[2] not in self.function_cache:
                        self.function_cache[row[2]] = []
                    self.function_cache[row[2]].append(func_info)
            
            # Load classes
            async with db.execute('SELECT * FROM classes') as cursor:
                async for row in cursor:
                    class_info = ClassInfo(
                        name=row[1],
                        file_path=row[2],
                        line_start=row[3],
                        line_end=row[4],
                        methods=json.loads(row[5]) if row[5] else [],
                        properties=json.loads(row[6]) if row[6] else [],
                        base_classes=json.loads(row[7]) if row[7] else [],
                        docstring=row[8],
                        is_abstract=bool(row[9])
                    )
                    
                    if row[2] not in self.class_cache:
                        self.class_cache[row[2]] = []
                    self.class_cache[row[2]].append(class_info)
    
    async def update_index(self, force_full_reindex: bool = False):
        """Update the code index incrementally or fully"""
        start_time = time.time()
        logger.info(f"Starting {'full' if force_full_reindex else 'incremental'} index update...")
        
        # Get all code files
        code_files = await self._discover_code_files()
        
        # Filter files that need indexing
        files_to_index = []
        
        if force_full_reindex:
            files_to_index = code_files
        else:
            for file_path in code_files:
                if await self._needs_reindexing(file_path):
                    files_to_index.append(file_path)
        
        logger.info(f"Found {len(files_to_index)} files to index out of {len(code_files)} total files")
        
        # Index files in parallel batches
        batch_size = self.max_workers * 2
        indexed_count = 0
        
        for i in range(0, len(files_to_index), batch_size):
            batch = files_to_index[i:i + batch_size]
            
            # Process batch in parallel
            tasks = [self._index_file(file_path) for file_path in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successful indexing
            for result in results:
                if not isinstance(result, Exception):
                    indexed_count += 1
            
            logger.debug(f"Processed batch {i//batch_size + 1}/{(len(files_to_index)-1)//batch_size + 1}")
        
        # Update statistics
        self.indexing_stats.update({
            'total_files': len(code_files),
            'indexed_files': indexed_count,
            'skipped_files': len(code_files) - indexed_count,
            'indexing_time': time.time() - start_time,
            'last_index_time': time.time()
        })
        
        logger.info(f"Index update completed in {self.indexing_stats['indexing_time']:.2f}s")
        logger.info(f"Indexed: {indexed_count}, Skipped: {self.indexing_stats['skipped_files']}")
    
    async def _discover_code_files(self) -> List[Path]:
        """Discover all code files in workspace"""
        code_files = []
        
        # Use os.walk for better performance on large directories
        def walk_directory():
            for root, dirs, files in self.workspace_path.rglob('*'):
                # Filter out ignored directories
                dirs[:] = [d for d in dirs if not any(pattern.search(str(Path(root) / d)) for pattern in self.ignore_patterns)]
                
                for file in files:
                    file_path = Path(root) / file
                    
                    # Check if file should be indexed
                    if self._should_index_file(file_path):
                        code_files.append(file_path)
        
        # Run discovery in thread pool for better performance
        with ThreadPoolExecutor(max_workers=4) as executor:
            await asyncio.get_event_loop().run_in_executor(executor, walk_directory)
        
        return code_files
    
    def _should_index_file(self, file_path: Path) -> bool:
        """Check if file should be indexed"""
        # Check ignore patterns
        file_path_str = str(file_path)
        if any(pattern.search(file_path_str) for pattern in self.ignore_patterns):
            return False
        
        # Check file extension
        if file_path.suffix.lower() not in self.LANGUAGE_EXTENSIONS:
            return False
        
        # Check file size (skip very large files)
        try:
            if file_path.stat().st_size > 10 * 1024 * 1024:  # 10MB limit
                return False
        except OSError:
            return False
        
        return True
    
    async def _needs_reindexing(self, file_path: Path) -> bool:
        """Check if file needs reindexing"""
        file_path_str = str(file_path)
        
        # Check if file exists in cache
        if file_path_str not in self.file_cache:
            return True
        
        try:
            # Check if file was modified
            current_mtime = file_path.stat().st_mtime
            cached_mtime = self.file_cache[file_path_str].last_modified
            
            if current_mtime > cached_mtime:
                return True
        except OSError:
            # File might have been deleted
            return False
        
        return False
    
    async def _index_file(self, file_path: Path) -> Optional[FileInfo]:
        """Index a single file"""
        try:
            file_path_str = str(file_path)
            
            # Get file stats
            stat = file_path.stat()
            file_size = stat.st_size
            last_modified = stat.st_mtime
            
            # Read file content
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Try with different encoding
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
            
            # Calculate file hash
            file_hash = hashlib.md5(content.encode('utf-8', errors='ignore')).hexdigest()
            
            # Determine language
            language = self.LANGUAGE_EXTENSIONS.get(file_path.suffix.lower(), 'text')
            
            # Count lines
            lines = len(content.splitlines())
            
            # Parse file based on language
            if language == 'python':
                ast_data, functions, classes = await self._parse_python_file(content, file_path_str)
            elif language in ['javascript', 'typescript', 'jsx', 'tsx']:
                ast_data, functions, classes = await self._parse_javascript_file(content, file_path_str, language)
            else:
                # Basic parsing for other languages
                ast_data, functions, classes = await self._parse_generic_file(content, file_path_str, language)
            
            # Calculate metrics
            complexity_score = self._calculate_complexity_score(content, functions)
            maintainability_index = self._calculate_maintainability_index(content, complexity_score)
            
            # Extract imports and dependencies
            imports = self._extract_imports(content, language)
            dependencies = self._extract_dependencies(content, language)
            exports = self._extract_exports(content, language)
            
            # Generate summary
            summary = self._generate_file_summary(content, functions, classes, language)
            
            # Create file info
            file_info = FileInfo(
                path=file_path_str,
                language=language,
                size=file_size,
                lines=lines,
                hash=file_hash,
                last_modified=last_modified,
                functions=[f.name for f in functions],
                classes=[c.name for c in classes],
                imports=imports,
                exports=exports,
                dependencies=dependencies,
                complexity_score=complexity_score,
                maintainability_index=maintainability_index,
                summary=summary,
                ast_data=ast_data
            )
            
            # Update caches
            self.file_cache[file_path_str] = file_info
            if functions:
                self.function_cache[file_path_str] = functions
            if classes:
                self.class_cache[file_path_str] = classes
            
            # Save to database
            await self._save_file_to_db(file_info, functions, classes)
            
            self.indexing_stats['total_lines'] += lines
            
            logger.debug(f"Indexed file: {file_path_str} ({lines} lines, {len(functions)} functions, {len(classes)} classes)")
            return file_info
            
        except Exception as e:
            logger.error(f"Error indexing file {file_path}: {e}")
            return None
    
    async def _parse_python_file(self, content: str, file_path: str) -> Tuple[Optional[Dict], List[FunctionInfo], List[ClassInfo]]:
        """Parse Python file using AST"""
        try:
            tree = ast.parse(content)
            
            functions = []
            classes = []
            ast_data = {
                'imports': [],
                'global_variables': [],
                'decorators': []
            }
            
            class PythonVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.current_class = None
                
                def visit_FunctionDef(self, node):
                    # Extract function information
                    func_info = FunctionInfo(
                        name=node.name,
                        file_path=file_path,
                        line_start=node.lineno,
                        line_end=getattr(node, 'end_lineno', node.lineno),
                        parameters=[arg.arg for arg in node.args.args],
                        return_type=None,  # Would need type hints parsing
                        docstring=ast.get_docstring(node),
                        complexity=self._calculate_cyclomatic_complexity(node),
                        calls=self._extract_function_calls(node),
                        is_async=isinstance(node, ast.AsyncFunctionDef),
                        is_method=self.current_class is not None,
                        class_name=self.current_class
                    )
                    functions.append(func_info)
                    self.generic_visit(node)
                
                def visit_AsyncFunctionDef(self, node):
                    self.visit_FunctionDef(node)
                
                def visit_ClassDef(self, node):
                    # Extract class information
                    methods = []
                    properties = []
                    
                    old_class = self.current_class
                    self.current_class = node.name
                    
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            methods.append(item.name)
                        elif isinstance(item, ast.Assign):
                            for target in item.targets:
                                if isinstance(target, ast.Name):
                                    properties.append(target.id)
                    
                    class_info = ClassInfo(
                        name=node.name,
                        file_path=file_path,
                        line_start=node.lineno,
                        line_end=getattr(node, 'end_lineno', node.lineno),
                        methods=methods,
                        properties=properties,
                        base_classes=[base.id if isinstance(base, ast.Name) else str(base) for base in node.bases],
                        docstring=ast.get_docstring(node),
                        is_abstract=any(isinstance(dec, ast.Name) and dec.id == 'abstractmethod' 
                                       for dec in getattr(node, 'decorator_list', []))
                    )
                    classes.append(class_info)
                    
                    self.generic_visit(node)
                    self.current_class = old_class
                
                def visit_Import(self, node):
                    for alias in node.names:
                        ast_data['imports'].append(alias.name)
                
                def visit_ImportFrom(self, node):
                    module = node.module or ''
                    for alias in node.names:
                        ast_data['imports'].append(f"{module}.{alias.name}")
            
            visitor = PythonVisitor()
            visitor.visit(tree)
            
            return ast_data, functions, classes
            
        except SyntaxError as e:
            logger.debug(f"Syntax error parsing Python file {file_path}: {e}")
            return None, [], []
        except Exception as e:
            logger.error(f"Error parsing Python file {file_path}: {e}")
            return None, [], []
    
    async def _parse_javascript_file(self, content: str, file_path: str, language: str) -> Tuple[Optional[Dict], List[FunctionInfo], List[ClassInfo]]:
        """Parse JavaScript/TypeScript file (basic parsing)"""
        # This is a simplified parser - for production, you'd want to use a proper JS/TS parser
        functions = []
        classes = []
        ast_data = {'imports': [], 'exports': []}
        
        lines = content.splitlines()
        
        # Simple regex-based parsing
        function_pattern = re.compile(r'^\s*(async\s+)?function\s+(\w+)\s*\(([^)]*)\)')
        arrow_function_pattern = re.compile(r'^\s*(const|let|var)\s+(\w+)\s*=\s*(async\s+)?\([^)]*\)\s*=>')
        class_pattern = re.compile(r'^\s*class\s+(\w+)')
        method_pattern = re.compile(r'^\s*(\w+)\s*\([^)]*\)\s*\{')
        
        current_class = None
        
        for i, line in enumerate(lines):
            # Function definitions
            func_match = function_pattern.match(line)
            if func_match:
                is_async = bool(func_match.group(1))
                func_name = func_match.group(2)
                params = [p.strip() for p in func_match.group(3).split(',') if p.strip()]
                
                functions.append(FunctionInfo(
                    name=func_name,
                    file_path=file_path,
                    line_start=i + 1,
                    line_end=i + 1,  # Would need bracket matching for accurate end
                    parameters=params,
                    return_type=None,
                    docstring=None,
                    complexity=1,  # Simplified
                    is_async=is_async,
                    is_method=current_class is not None,
                    class_name=current_class
                ))
            
            # Arrow functions
            arrow_match = arrow_function_pattern.match(line)
            if arrow_match:
                func_name = arrow_match.group(2)
                is_async = bool(arrow_match.group(3))
                
                functions.append(FunctionInfo(
                    name=func_name,
                    file_path=file_path,
                    line_start=i + 1,
                    line_end=i + 1,
                    parameters=[],  # Would need better parsing
                    return_type=None,
                    docstring=None,
                    complexity=1,
                    is_async=is_async,
                    is_method=False
                ))
            
            # Class definitions
            class_match = class_pattern.match(line)
            if class_match:
                class_name = class_match.group(1)
                current_class = class_name
                
                classes.append(ClassInfo(
                    name=class_name,
                    file_path=file_path,
                    line_start=i + 1,
                    line_end=i + 1,
                    methods=[],
                    properties=[],
                    base_classes=[],
                    docstring=None
                ))
        
        return ast_data, functions, classes
    
    async def _parse_generic_file(self, content: str, file_path: str, language: str) -> Tuple[Optional[Dict], List[FunctionInfo], List[ClassInfo]]:
        """Basic parsing for other file types"""
        # Very basic parsing - extract function-like patterns
        functions = []
        classes = []
        ast_data = {'content_type': 'generic'}
        
        # Language-specific patterns
        if language in ['java', 'csharp', 'cpp', 'c']:
            # Look for function/method patterns
            func_pattern = re.compile(r'\b(public|private|protected|static).*?\b(\w+)\s*\([^)]*\)')
            matches = func_pattern.finditer(content)
            
            for match in matches:
                func_name = match.group(2)
                if func_name not in ['if', 'while', 'for', 'switch']:  # Filter out control structures
                    functions.append(FunctionInfo(
                        name=func_name,
                        file_path=file_path,
                        line_start=content[:match.start()].count('\n') + 1,
                        line_end=content[:match.end()].count('\n') + 1,
                        parameters=[],
                        return_type=None,
                        docstring=None,
                        complexity=1
                    ))
        
        return ast_data, functions, classes
    
    def _calculate_cyclomatic_complexity(self, node) -> int:
        """Calculate cyclomatic complexity of AST node"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.With):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _extract_function_calls(self, node) -> List[str]:
        """Extract function calls from AST node"""
        calls = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(child.func.attr)
        
        return calls
    
    def _calculate_complexity_score(self, content: str, functions: List[FunctionInfo]) -> float:
        """Calculate overall complexity score for file"""
        if not functions:
            return 1.0
        
        total_complexity = sum(f.complexity for f in functions)
        avg_complexity = total_complexity / len(functions)
        
        # Normalize to 0-10 scale
        return min(10.0, avg_complexity * 2.0)
    
    def _calculate_maintainability_index(self, content: str, complexity_score: float) -> float:
        """Calculate maintainability index"""
        lines = len(content.splitlines())
        volume = len(content) * 0.1  # Simplified volume calculation
        
        # Simplified maintainability index
        mi = 171 - (5.2 * complexity_score) - (0.23 * volume) - (16.2 * lines / 1000)
        return max(0.0, min(100.0, mi))
    
    def _extract_imports(self, content: str, language: str) -> List[str]:
        """Extract imports/includes from file"""
        imports = []
        
        if language == 'python':
            import_pattern = re.compile(r'^\s*(import|from)\s+([\w\.]+)', re.MULTILINE)
            matches = import_pattern.finditer(content)
            imports = [match.group(2) for match in matches]
        
        elif language in ['javascript', 'typescript', 'jsx', 'tsx']:
            import_pattern = re.compile(r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]', re.MULTILINE)
            matches = import_pattern.finditer(content)
            imports = [match.group(1) for match in matches]
        
        elif language in ['java', 'csharp']:
            import_pattern = re.compile(r'import\s+([\w\.]+);', re.MULTILINE)
            matches = import_pattern.finditer(content)
            imports = [match.group(1) for match in matches]
        
        elif language in ['cpp', 'c']:
            include_pattern = re.compile(r'#include\s*[<"](.*?)[>"]', re.MULTILINE)
            matches = include_pattern.finditer(content)
            imports = [match.group(1) for match in matches]
        
        return imports
    
    def _extract_dependencies(self, content: str, language: str) -> List[str]:
        """Extract dependencies from file content"""
        # This is a simplified implementation
        dependencies = []
        
        if language == 'python':
            # Look for common library imports
            common_libs = ['numpy', 'pandas', 'requests', 'flask', 'django', 'fastapi', 'sqlalchemy']
            for lib in common_libs:
                if lib in content:
                    dependencies.append(lib)
        
        elif language in ['javascript', 'typescript']:
            # Look for require statements and common libraries
            require_pattern = re.compile(r'require\([\'"]([^\'"]+)[\'"]\)')
            matches = require_pattern.finditer(content)
            dependencies.extend(match.group(1) for match in matches)
        
        return list(set(dependencies))  # Remove duplicates
    
    def _extract_exports(self, content: str, language: str) -> List[str]:
        """Extract exports from file"""
        exports = []
        
        if language in ['javascript', 'typescript', 'jsx', 'tsx']:
            export_pattern = re.compile(r'export\s+(default\s+)?(\w+)', re.MULTILINE)
            matches = export_pattern.finditer(content)
            exports = [match.group(2) for match in matches if match.group(2) not in ['default']]
        
        return exports
    
    def _generate_file_summary(self, content: str, functions: List[FunctionInfo], classes: List[ClassInfo], language: str) -> str:
        """Generate a summary of the file"""
        summary_parts = []
        
        # File type and size
        lines = len(content.splitlines())
        summary_parts.append(f"{language.capitalize()} file with {lines} lines")
        
        # Functions and classes
        if classes:
            summary_parts.append(f"{len(classes)} classes: {', '.join(c.name for c in classes[:3])}")
            if len(classes) > 3:
                summary_parts[-1] += f" and {len(classes) - 3} more"
        
        if functions:
            func_count = len(functions)
            method_count = len([f for f in functions if f.is_method])
            free_func_count = func_count - method_count
            
            if free_func_count > 0:
                summary_parts.append(f"{free_func_count} functions")
            if method_count > 0:
                summary_parts.append(f"{method_count} methods")
        
        # Purpose detection (very basic)
        content_lower = content.lower()
        if 'test' in content_lower and ('assert' in content_lower or 'expect' in content_lower):
            summary_parts.append("Contains tests")
        elif 'main' in content_lower and language == 'python':
            summary_parts.append("Entry point script")
        elif 'class' in content and 'def __init__' in content:
            summary_parts.append("Defines classes with constructors")
        
        return "; ".join(summary_parts) if summary_parts else "Code file"
    
    async def _save_file_to_db(self, file_info: FileInfo, functions: List[FunctionInfo], classes: List[ClassInfo]):
        """Save file information to database"""
        async with aiosqlite.connect(str(self.db_path)) as db:
            # Save file info
            await db.execute('''
                INSERT OR REPLACE INTO files 
                (path, language, size, lines, hash, last_modified, functions, classes, 
                 imports, exports, dependencies, complexity_score, maintainability_index, 
                 summary, ast_data, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                file_info.path, file_info.language, file_info.size, file_info.lines,
                file_info.hash, file_info.last_modified, json.dumps(file_info.functions),
                json.dumps(file_info.classes), json.dumps(file_info.imports),
                json.dumps(file_info.exports), json.dumps(file_info.dependencies),
                file_info.complexity_score, file_info.maintainability_index,
                file_info.summary, json.dumps(file_info.ast_data), time.time()
            ))
            
            # Delete old functions and classes for this file
            await db.execute('DELETE FROM functions WHERE file_path = ?', (file_info.path,))
            await db.execute('DELETE FROM classes WHERE file_path = ?', (file_info.path,))
            
            # Save functions
            for func in functions:
                await db.execute('''
                    INSERT INTO functions 
                    (name, file_path, line_start, line_end, parameters, return_type, 
                     docstring, complexity, calls, is_async, is_method, class_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    func.name, func.file_path, func.line_start, func.line_end,
                    json.dumps(func.parameters), func.return_type, func.docstring,
                    func.complexity, json.dumps(func.calls), func.is_async,
                    func.is_method, func.class_name
                ))
            
            # Save classes
            for cls in classes:
                await db.execute('''
                    INSERT INTO classes 
                    (name, file_path, line_start, line_end, methods, properties, 
                     base_classes, docstring, is_abstract)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    cls.name, cls.file_path, cls.line_start, cls.line_end,
                    json.dumps(cls.methods), json.dumps(cls.properties),
                    json.dumps(cls.base_classes), cls.docstring, cls.is_abstract
                ))
            
            await db.commit()
    
    # Query methods
    
    async def find_relevant_files(self, query: str, max_files: int = 100) -> List[Dict[str, Any]]:
        """Find files relevant to the query"""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        results = []
        
        for file_path, file_info in self.file_cache.items():
            relevance_score = 0.0
            
            # Check file path
            if any(word in file_path.lower() for word in query_words):
                relevance_score += 2.0
            
            # Check file summary
            if file_info.summary and any(word in file_info.summary.lower() for word in query_words):
                relevance_score += 3.0
            
            # Check functions and classes
            all_symbols = file_info.functions + file_info.classes
            if any(any(word in symbol.lower() for word in query_words) for symbol in all_symbols):
                relevance_score += 4.0
            
            # Check imports and dependencies
            all_deps = file_info.imports + file_info.dependencies
            if any(any(word in dep.lower() for word in query_words) for dep in all_deps):
                relevance_score += 1.0
            
            if relevance_score > 0:
                results.append({
                    'path': file_path,
                    'relevance_score': relevance_score,
                    'file_info': file_info
                })
        
        # Sort by relevance and limit results
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:max_files]
    
    async def find_functions(self, function_names: List[str]) -> List[Dict[str, Any]]:
        """Find functions by name"""
        results = []
        
        for file_path, functions in self.function_cache.items():
            for func in functions:
                if any(name.lower() in func.name.lower() for name in function_names):
                    results.append({
                        'path': file_path,
                        'function': func,
                        'relevance_score': 5.0
                    })
        
        return results
    
    async def find_classes(self, class_names: List[str]) -> List[Dict[str, Any]]:
        """Find classes by name"""
        results = []
        
        for file_path, classes in self.class_cache.items():
            for cls in classes:
                if any(name.lower() in cls.name.lower() for name in class_names):
                    results.append({
                        'path': file_path,
                        'class': cls,
                        'relevance_score': 5.0
                    })
        
        return results
    
    async def get_file_summary(self, file_path: str) -> Optional[str]:
        """Get summary of a specific file"""
        file_info = self.file_cache.get(file_path)
        return file_info.summary if file_info else None
    
    async def get_indexed_file_count(self) -> int:
        """Get number of indexed files"""
        return len(self.file_cache)
    
    async def clear_cache(self):
        """Clear all caches"""
        self.file_cache.clear()
        self.function_cache.clear()
        self.class_cache.clear()
        
        # Optionally clear database
        async with aiosqlite.connect(str(self.db_path)) as db:
            await db.execute('DELETE FROM files')
            await db.execute('DELETE FROM functions')
            await db.execute('DELETE FROM classes')
            await db.commit()
        
        logger.info("Code index cache cleared")
    
    # Additional methods for comprehensive functionality
    
    async def find_error_related_code(self, error_terms: List[str]) -> List[Dict[str, Any]]:
        """Find code related to error handling"""
        results = []
        error_keywords = ['try', 'catch', 'except', 'error', 'exception', 'throw', 'raise']
        
        for file_path, file_info in self.file_cache.items():
            score = 0.0
            
            # Check if file contains error handling
            if any(keyword in file_info.summary.lower() for keyword in error_keywords):
                score += 2.0
            
            # Check function names for error handling
            for func_name in file_info.functions:
                if any(term.lower() in func_name.lower() for term in error_terms):
                    score += 3.0
                if any(keyword in func_name.lower() for keyword in error_keywords):
                    score += 1.0
            
            if score > 0:
                results.append({
                    'path': file_path,
                    'relevance_score': score,
                    'file_info': file_info
                })
        
        return sorted(results, key=lambda x: x['relevance_score'], reverse=True)
    
    async def find_usage_examples(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Find usage examples of symbols"""
        results = []
        
        for file_path, file_info in self.file_cache.items():
            score = 0.0
            
            # Check if symbols are used in the file
            for symbol in symbols:
                if symbol in file_info.functions:
                    score += 3.0
                if symbol in file_info.classes:
                    score += 3.0
                if any(symbol in dep for dep in file_info.dependencies):
                    score += 2.0
                if any(symbol in imp for imp in file_info.imports):
                    score += 2.0
            
            if score > 0:
                results.append({
                    'path': file_path,
                    'relevance_score': score,
                    'file_info': file_info
                })
        
        return sorted(results, key=lambda x: x['relevance_score'], reverse=True)
    
    async def get_code_snippets(self, file_path: str, symbols: List[str], context_lines: int = 3) -> List[Dict[str, Any]]:
        """Get code snippets containing specific symbols"""
        snippets = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines):
                for symbol in symbols:
                    if symbol in line:
                        start = max(0, i - context_lines)
                        end = min(len(lines), i + context_lines + 1)
                        
                        snippet_lines = lines[start:end]
                        snippet = ''.join(snippet_lines)
                        
                        language = self.LANGUAGE_EXTENSIONS.get(Path(file_path).suffix.lower(), 'text')
                        
                        snippets.append({
                            'symbol': symbol,
                            'line_number': i + 1,
                            'code': snippet,
                            'language': language,
                            'context_start': start + 1,
                            'context_end': end
                        })
        
        except Exception as e:
            logger.error(f"Error reading code snippets from {file_path}: {e}")
        
        return snippets
    
    async def build_project_hierarchy(self) -> Dict[str, Any]:
        """Build hierarchical view of project structure"""
        hierarchy = {}
        
        for file_path in self.file_cache.keys():
            path_parts = Path(file_path).parts
            current_level = hierarchy
            
            for part in path_parts[:-1]:  # Exclude filename
                if part not in current_level:
                    current_level[part] = {}
                current_level = current_level[part]
            
            # Add file info
            filename = path_parts[-1]
            current_level[filename] = self.file_cache[file_path]
        
        return hierarchy
    
    async def get_module_summary(self, module_path: str) -> Optional[str]:
        """Get summary of a module/directory"""
        module_files = [f for f in self.file_cache.keys() if f.startswith(module_path)]
        
        if not module_files:
            return None
        
        # Aggregate statistics
        total_lines = sum(self.file_cache[f].lines for f in module_files)
        total_functions = sum(len(self.file_cache[f].functions) for f in module_files)
        total_classes = sum(len(self.file_cache[f].classes) for f in module_files)
        languages = list(set(self.file_cache[f].language for f in module_files))
        
        summary = f"Module with {len(module_files)} files, {total_lines} total lines"
        
        if total_classes > 0:
            summary += f", {total_classes} classes"
        if total_functions > 0:
            summary += f", {total_functions} functions"
        
        if languages:
            summary += f". Languages: {', '.join(languages)}"
        
        return summary
    
    async def get_key_files_in_module(self, module_path: str, max_files: int = 10) -> List[Dict[str, Any]]:
        """Get key files in a module based on various metrics"""
        module_files = [(f, info) for f, info in self.file_cache.items() if f.startswith(module_path)]
        
        # Score files based on importance
        scored_files = []
        for file_path, file_info in module_files:
            score = 0.0
            
            # Size factor (larger files might be more important)
            score += min(5.0, file_info.lines / 100)
            
            # Complexity factor
            score += file_info.complexity_score * 0.5
            
            # Symbol count (functions + classes)
            symbol_count = len(file_info.functions) + len(file_info.classes)
            score += min(3.0, symbol_count * 0.5)
            
            # Special files
            filename = Path(file_path).name.lower()
            if filename in ['__init__.py', 'main.py', 'index.js', 'app.py']:
                score += 3.0
            elif filename.startswith('test_'):
                score += 1.0  # Tests are somewhat important
            
            scored_files.append({
                'path': file_path,
                'score': score,
                'file_info': file_info
            })
        
        # Sort by score and return top files
        scored_files.sort(key=lambda x: x['score'], reverse=True)
        return scored_files[:max_files]