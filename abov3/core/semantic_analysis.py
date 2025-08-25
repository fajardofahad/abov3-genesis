"""
ABOV3 Genesis - Semantic Code Analysis and Understanding
Advanced semantic analysis system for deep code understanding and intelligent suggestions
"""

import asyncio
import ast
import json
import re
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import logging
from enum import Enum
from collections import defaultdict, deque
import hashlib
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)

class SemanticType(Enum):
    """Types of semantic entities"""
    FUNCTION = "function"
    CLASS = "class"
    VARIABLE = "variable"
    MODULE = "module"
    CONSTANT = "constant"
    INTERFACE = "interface"
    ENUM = "enum"
    DECORATOR = "decorator"
    ANNOTATION = "annotation"
    IMPORT = "import"
    NAMESPACE = "namespace"

class RelationshipType(Enum):
    """Types of relationships between code entities"""
    CALLS = "calls"
    INHERITS = "inherits"
    IMPORTS = "imports"
    USES = "uses"
    DEFINES = "defines"
    OVERRIDES = "overrides"
    IMPLEMENTS = "implements"
    DEPENDS_ON = "depends_on"
    CONTAINS = "contains"
    REFERENCES = "references"
    THROWS = "throws"
    RETURNS = "returns"

class CodePattern(Enum):
    """Common code patterns"""
    SINGLETON = "singleton"
    FACTORY = "factory"
    OBSERVER = "observer"
    STRATEGY = "strategy"
    DECORATOR = "decorator"
    ADAPTER = "adapter"
    FACADE = "facade"
    TEMPLATE_METHOD = "template_method"
    COMMAND = "command"
    BUILDER = "builder"
    PROXY = "proxy"
    CHAIN_OF_RESPONSIBILITY = "chain_of_responsibility"

@dataclass
class SemanticEntity:
    """Represents a semantic entity in code"""
    name: str
    entity_type: SemanticType
    file_path: Optional[str] = None
    line_number: int = 0
    column: int = 0
    scope: str = "global"
    docstring: Optional[str] = None
    signature: Optional[str] = None
    return_type: Optional[str] = None
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    annotations: Dict[str, str] = field(default_factory=dict)
    decorators: List[str] = field(default_factory=list)
    complexity_score: float = 0.0
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SemanticRelationship:
    """Represents a relationship between semantic entities"""
    source: str
    target: str
    relationship_type: RelationshipType
    strength: float = 1.0
    context: Dict[str, Any] = field(default_factory=dict)
    file_path: Optional[str] = None
    line_number: int = 0

@dataclass
class CodeSemanticsAnalysis:
    """Complete semantic analysis of code"""
    entities: Dict[str, SemanticEntity] = field(default_factory=dict)
    relationships: List[SemanticRelationship] = field(default_factory=list)
    patterns: List[Tuple[CodePattern, float, Dict[str, Any]]] = field(default_factory=list)
    complexity_metrics: Dict[str, float] = field(default_factory=dict)
    quality_assessment: Dict[str, Any] = field(default_factory=dict)
    architectural_insights: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_graph: Optional[nx.DiGraph] = None

class SemanticCodeAnalyzer:
    """Advanced semantic code analysis system"""
    
    def __init__(self):
        # Language-specific analyzers
        self.language_analyzers = {
            'python': PythonSemanticAnalyzer(),
            'javascript': JavaScriptSemanticAnalyzer(),
            'typescript': TypeScriptSemanticAnalyzer(),
            'java': JavaSemanticAnalyzer(),
            'go': GoSemanticAnalyzer(),
            'rust': RustSemanticAnalyzer()
        }
        
        # Pattern recognizers
        self.pattern_recognizer = DesignPatternRecognizer()
        self.architecture_analyzer = ArchitectureAnalyzer()
        
        # Quality assessors
        self.quality_assessor = CodeQualityAssessor()
        self.complexity_analyzer = ComplexityAnalyzer()
        
        # Knowledge graph
        self.knowledge_graph_builder = KnowledgeGraphBuilder()
        
        # Semantic similarity
        self.similarity_analyzer = SemanticSimilarityAnalyzer()
        
        # Cache for performance
        self.analysis_cache = {}
        self.cache_ttl = 1800  # 30 minutes
    
    async def analyze_code_semantics(
        self,
        code_content: str,
        file_path: Optional[str] = None,
        language: str = "python",
        include_patterns: bool = True,
        include_architecture: bool = True
    ) -> CodeSemanticsAnalysis:
        """Comprehensive semantic analysis of code"""
        
        start_time = time.time()
        
        # Create cache key
        cache_key = self._create_cache_key(code_content, language, include_patterns, include_architecture)
        if cache_key in self.analysis_cache:
            cache_entry = self.analysis_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                logger.debug("Returning cached semantic analysis")
                return cache_entry['analysis']
        
        # Get language-specific analyzer
        analyzer = self.language_analyzers.get(language)
        if not analyzer:
            logger.warning(f"No semantic analyzer available for language: {language}")
            return CodeSemanticsAnalysis()
        
        # Parse code and extract entities
        entities = await analyzer.extract_entities(code_content, file_path)
        
        # Analyze relationships
        relationships = await analyzer.analyze_relationships(entities, code_content)
        
        # Build knowledge graph
        knowledge_graph = await self.knowledge_graph_builder.build_graph(entities, relationships)
        
        # Analyze complexity
        complexity_metrics = await self.complexity_analyzer.analyze_complexity(entities, relationships, code_content)
        
        # Assess quality
        quality_assessment = await self.quality_assessor.assess_quality(entities, relationships, code_content, language)
        
        # Recognize patterns
        patterns = []
        if include_patterns:
            patterns = await self.pattern_recognizer.recognize_patterns(entities, relationships, code_content)
        
        # Architectural analysis
        architectural_insights = {}
        if include_architecture:
            architectural_insights = await self.architecture_analyzer.analyze_architecture(entities, relationships, knowledge_graph)
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(
            entities, relationships, patterns, complexity_metrics, quality_assessment
        )
        
        # Create analysis result
        analysis = CodeSemanticsAnalysis(
            entities=entities,
            relationships=relationships,
            patterns=patterns,
            complexity_metrics=complexity_metrics,
            quality_assessment=quality_assessment,
            architectural_insights=architectural_insights,
            recommendations=recommendations,
            knowledge_graph=knowledge_graph
        )
        
        # Cache result
        self.analysis_cache[cache_key] = {
            'analysis': analysis,
            'timestamp': time.time()
        }
        
        logger.debug(f"Semantic analysis completed in {time.time() - start_time:.3f}s")
        return analysis
    
    async def compare_code_similarity(
        self,
        code1: str,
        code2: str,
        language: str = "python"
    ) -> Dict[str, Any]:
        """Compare semantic similarity between two code snippets"""
        
        # Analyze both code snippets
        analysis1 = await self.analyze_code_semantics(code1, language=language, include_patterns=False, include_architecture=False)
        analysis2 = await self.analyze_code_semantics(code2, language=language, include_patterns=False, include_architecture=False)
        
        # Calculate similarities
        similarity_result = await self.similarity_analyzer.calculate_similarity(analysis1, analysis2)
        
        return similarity_result
    
    async def find_code_clones(
        self,
        code_files: List[Tuple[str, str]],  # [(file_path, content), ...]
        language: str = "python",
        similarity_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Find code clones across multiple files"""
        
        clones = []
        analyses = {}
        
        # Analyze all files
        for file_path, content in code_files:
            analyses[file_path] = await self.analyze_code_semantics(
                content, file_path, language, include_patterns=False, include_architecture=False
            )
        
        # Compare all pairs
        file_paths = list(analyses.keys())
        for i in range(len(file_paths)):
            for j in range(i + 1, len(file_paths)):
                file1, file2 = file_paths[i], file_paths[j]
                similarity = await self.similarity_analyzer.calculate_similarity(analyses[file1], analyses[file2])
                
                if similarity['overall_similarity'] >= similarity_threshold:
                    clones.append({
                        'file1': file1,
                        'file2': file2,
                        'similarity': similarity,
                        'clone_type': self._classify_clone_type(similarity)
                    })
        
        return clones
    
    def _create_cache_key(self, code_content: str, language: str, include_patterns: bool, include_architecture: bool) -> str:
        """Create cache key for analysis"""
        key_components = [
            hashlib.md5(code_content.encode()).hexdigest()[:16],
            language,
            str(include_patterns),
            str(include_architecture)
        ]
        return '|'.join(key_components)
    
    def _classify_clone_type(self, similarity: Dict[str, Any]) -> str:
        """Classify type of code clone"""
        overall = similarity['overall_similarity']
        structural = similarity.get('structural_similarity', 0)
        semantic = similarity.get('semantic_similarity', 0)
        
        if overall > 0.95 and structural > 0.9:
            return "Type 1 - Exact Clone"
        elif overall > 0.85 and structural > 0.8:
            return "Type 2 - Renamed Clone"
        elif semantic > 0.8:
            return "Type 3 - Near Miss Clone"
        else:
            return "Type 4 - Semantic Clone"
    
    async def _generate_recommendations(
        self,
        entities: Dict[str, SemanticEntity],
        relationships: List[SemanticRelationship],
        patterns: List[Tuple[CodePattern, float, Dict[str, Any]]],
        complexity_metrics: Dict[str, float],
        quality_assessment: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on semantic analysis"""
        
        recommendations = []
        
        # Complexity-based recommendations
        if complexity_metrics.get('cyclomatic_complexity', 0) > 10:
            recommendations.append({
                'type': 'complexity',
                'severity': 'high',
                'title': 'High Cyclomatic Complexity',
                'description': 'Consider breaking down complex functions into smaller, more manageable pieces',
                'affected_entities': [name for name, entity in entities.items() if entity.complexity_score > 10]
            })
        
        # Quality-based recommendations
        if quality_assessment.get('maintainability_index', 100) < 60:
            recommendations.append({
                'type': 'maintainability',
                'severity': 'medium',
                'title': 'Low Maintainability',
                'description': 'Consider improving code structure, reducing complexity, and adding documentation',
                'suggestions': ['Add comprehensive docstrings', 'Reduce function complexity', 'Improve variable naming']
            })
        
        # Pattern-based recommendations
        singleton_patterns = [p for p in patterns if p[0] == CodePattern.SINGLETON and p[1] > 0.7]
        if len(singleton_patterns) > 1:
            recommendations.append({
                'type': 'pattern',
                'severity': 'low',
                'title': 'Multiple Singleton Patterns',
                'description': 'Multiple singleton patterns detected. Consider if all are necessary.',
                'pattern_locations': [p[2] for p in singleton_patterns]
            })
        
        # Relationship-based recommendations
        circular_deps = self._find_circular_dependencies(relationships)
        if circular_deps:
            recommendations.append({
                'type': 'architecture',
                'severity': 'high',
                'title': 'Circular Dependencies',
                'description': 'Circular dependencies detected between modules/classes',
                'circular_chains': circular_deps
            })
        
        # Entity-specific recommendations
        long_functions = [name for name, entity in entities.items() 
                         if entity.entity_type == SemanticType.FUNCTION and 
                         entity.metadata.get('line_count', 0) > 50]
        if long_functions:
            recommendations.append({
                'type': 'structure',
                'severity': 'medium',
                'title': 'Long Functions',
                'description': 'Functions with many lines should be broken down for better readability',
                'affected_functions': long_functions
            })
        
        return recommendations
    
    def _find_circular_dependencies(self, relationships: List[SemanticRelationship]) -> List[List[str]]:
        """Find circular dependencies in relationships"""
        
        # Build dependency graph
        graph = nx.DiGraph()
        for rel in relationships:
            if rel.relationship_type in [RelationshipType.DEPENDS_ON, RelationshipType.IMPORTS, RelationshipType.USES]:
                graph.add_edge(rel.source, rel.target)
        
        # Find cycles
        try:
            cycles = list(nx.simple_cycles(graph))
            return [cycle for cycle in cycles if len(cycle) > 1]
        except:
            return []

class PythonSemanticAnalyzer:
    """Python-specific semantic analyzer"""
    
    async def extract_entities(self, code_content: str, file_path: Optional[str] = None) -> Dict[str, SemanticEntity]:
        """Extract semantic entities from Python code"""
        
        entities = {}
        
        try:
            tree = ast.parse(code_content)
            
            # Walk AST and extract entities
            for node in ast.walk(tree):
                entity = None
                
                if isinstance(node, ast.FunctionDef):
                    entity = await self._create_function_entity(node, file_path)
                elif isinstance(node, ast.ClassDef):
                    entity = await self._create_class_entity(node, file_path)
                elif isinstance(node, ast.Assign):
                    entity = await self._create_variable_entity(node, file_path)
                elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    entity = await self._create_import_entity(node, file_path)
                
                if entity:
                    entities[entity.name] = entity
        
        except SyntaxError as e:
            logger.warning(f"Syntax error in Python code: {e}")
        
        return entities
    
    async def _create_function_entity(self, node: ast.FunctionDef, file_path: Optional[str]) -> SemanticEntity:
        """Create semantic entity for function"""
        
        # Extract parameters
        parameters = []
        for arg in node.args.args:
            param = {
                'name': arg.arg,
                'type': self._get_annotation_string(arg.annotation) if arg.annotation else None,
                'default': None
            }
            parameters.append(param)
        
        # Extract decorators
        decorators = [self._get_decorator_name(dec) for dec in node.decorator_list]
        
        # Get docstring
        docstring = ast.get_docstring(node)
        
        # Calculate complexity
        complexity = self._calculate_function_complexity(node)
        
        entity = SemanticEntity(
            name=node.name,
            entity_type=SemanticType.FUNCTION,
            file_path=file_path,
            line_number=node.lineno,
            column=node.col_offset,
            docstring=docstring,
            parameters=parameters,
            decorators=decorators,
            complexity_score=complexity,
            return_type=self._get_annotation_string(node.returns) if node.returns else None,
            metadata={
                'line_count': self._count_lines(node),
                'is_async': isinstance(node, ast.AsyncFunctionDef),
                'has_docstring': docstring is not None
            }
        )
        
        return entity
    
    async def _create_class_entity(self, node: ast.ClassDef, file_path: Optional[str]) -> SemanticEntity:
        """Create semantic entity for class"""
        
        # Extract base classes
        bases = [self._get_base_name(base) for base in node.bases]
        
        # Extract decorators
        decorators = [self._get_decorator_name(dec) for dec in node.decorator_list]
        
        # Get docstring
        docstring = ast.get_docstring(node)
        
        # Count methods
        methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
        
        entity = SemanticEntity(
            name=node.name,
            entity_type=SemanticType.CLASS,
            file_path=file_path,
            line_number=node.lineno,
            column=node.col_offset,
            docstring=docstring,
            decorators=decorators,
            metadata={
                'base_classes': bases,
                'method_count': len(methods),
                'line_count': self._count_lines(node),
                'has_docstring': docstring is not None
            }
        )
        
        return entity
    
    async def _create_variable_entity(self, node: ast.Assign, file_path: Optional[str]) -> SemanticEntity:
        """Create semantic entity for variable"""
        
        # Handle simple assignments only
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0]
            
            entity = SemanticEntity(
                name=target.id,
                entity_type=SemanticType.VARIABLE,
                file_path=file_path,
                line_number=node.lineno,
                column=node.col_offset,
                metadata={
                    'value_type': self._infer_value_type(node.value),
                    'is_constant': target.id.isupper()
                }
            )
            
            return entity
        
        return None
    
    async def _create_import_entity(self, node: Union[ast.Import, ast.ImportFrom], file_path: Optional[str]) -> SemanticEntity:
        """Create semantic entity for import"""
        
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        else:
            names = [f"{node.module}.{alias.name}" for alias in node.names]
        
        entity = SemanticEntity(
            name=names[0] if names else "unknown_import",
            entity_type=SemanticType.IMPORT,
            file_path=file_path,
            line_number=node.lineno,
            column=node.col_offset,
            metadata={
                'imported_names': names,
                'module': getattr(node, 'module', None)
            }
        )
        
        return entity
    
    async def analyze_relationships(
        self,
        entities: Dict[str, SemanticEntity],
        code_content: str
    ) -> List[SemanticRelationship]:
        """Analyze relationships between entities"""
        
        relationships = []
        
        try:
            tree = ast.parse(code_content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # Function calls
                    func_name = self._get_call_name(node.func)
                    if func_name and func_name in entities:
                        # Find caller context
                        caller = self._find_enclosing_function_or_class(node, tree)
                        if caller and caller in entities:
                            relationships.append(SemanticRelationship(
                                source=caller,
                                target=func_name,
                                relationship_type=RelationshipType.CALLS,
                                line_number=node.lineno
                            ))
                
                elif isinstance(node, ast.ClassDef):
                    # Inheritance relationships
                    for base in node.bases:
                        base_name = self._get_base_name(base)
                        if base_name and base_name in entities:
                            relationships.append(SemanticRelationship(
                                source=node.name,
                                target=base_name,
                                relationship_type=RelationshipType.INHERITS,
                                line_number=node.lineno
                            ))
        
        except SyntaxError:
            pass
        
        return relationships
    
    def _calculate_function_complexity(self, node: ast.FunctionDef) -> float:
        """Calculate cyclomatic complexity of function"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _count_lines(self, node: ast.AST) -> int:
        """Count lines in AST node"""
        if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
            return node.end_lineno - node.lineno + 1
        return 1
    
    def _get_annotation_string(self, annotation) -> str:
        """Get string representation of type annotation"""
        if annotation is None:
            return None
        
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Constant):
            return str(annotation.value)
        else:
            return ast.unparse(annotation) if hasattr(ast, 'unparse') else "unknown"
    
    def _get_decorator_name(self, decorator) -> str:
        """Get decorator name"""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return f"{decorator.value.id}.{decorator.attr}" if isinstance(decorator.value, ast.Name) else decorator.attr
        else:
            return "unknown_decorator"
    
    def _get_base_name(self, base) -> str:
        """Get base class name"""
        if isinstance(base, ast.Name):
            return base.id
        elif isinstance(base, ast.Attribute):
            return f"{base.value.id}.{base.attr}" if isinstance(base.value, ast.Name) else base.attr
        else:
            return "unknown_base"
    
    def _infer_value_type(self, value_node) -> str:
        """Infer type of value"""
        if isinstance(value_node, ast.Constant):
            return type(value_node.value).__name__
        elif isinstance(value_node, ast.List):
            return "list"
        elif isinstance(value_node, ast.Dict):
            return "dict"
        elif isinstance(value_node, ast.Call):
            func_name = self._get_call_name(value_node.func)
            return f"call_to_{func_name}" if func_name else "function_call"
        else:
            return "unknown"
    
    def _get_call_name(self, func_node) -> Optional[str]:
        """Get function call name"""
        if isinstance(func_node, ast.Name):
            return func_node.id
        elif isinstance(func_node, ast.Attribute):
            return func_node.attr
        else:
            return None
    
    def _find_enclosing_function_or_class(self, node: ast.AST, tree: ast.AST) -> Optional[str]:
        """Find enclosing function or class name"""
        # This is a simplified implementation
        # In practice, you'd need to track the AST traversal path
        return None

# Simplified implementations for other languages
class JavaScriptSemanticAnalyzer:
    """JavaScript-specific semantic analyzer"""
    
    async def extract_entities(self, code_content: str, file_path: Optional[str] = None) -> Dict[str, SemanticEntity]:
        # Simplified JavaScript analysis using regex patterns
        entities = {}
        
        # Find functions
        func_pattern = r'function\s+(\w+)\s*\([^)]*\)\s*{'
        for match in re.finditer(func_pattern, code_content):
            line_num = code_content[:match.start()].count('\n') + 1
            entities[match.group(1)] = SemanticEntity(
                name=match.group(1),
                entity_type=SemanticType.FUNCTION,
                file_path=file_path,
                line_number=line_num
            )
        
        # Find classes
        class_pattern = r'class\s+(\w+)(?:\s+extends\s+(\w+))?\s*{'
        for match in re.finditer(class_pattern, code_content):
            line_num = code_content[:match.start()].count('\n') + 1
            entities[match.group(1)] = SemanticEntity(
                name=match.group(1),
                entity_type=SemanticType.CLASS,
                file_path=file_path,
                line_number=line_num
            )
        
        return entities
    
    async def analyze_relationships(
        self,
        entities: Dict[str, SemanticEntity],
        code_content: str
    ) -> List[SemanticRelationship]:
        return []  # Simplified

class TypeScriptSemanticAnalyzer(JavaScriptSemanticAnalyzer):
    """TypeScript-specific semantic analyzer"""
    pass

class JavaSemanticAnalyzer:
    """Java-specific semantic analyzer"""
    
    async def extract_entities(self, code_content: str, file_path: Optional[str] = None) -> Dict[str, SemanticEntity]:
        entities = {}
        
        # Find classes
        class_pattern = r'(?:public\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([^{]+))?\s*{'
        for match in re.finditer(class_pattern, code_content):
            line_num = code_content[:match.start()].count('\n') + 1
            entities[match.group(1)] = SemanticEntity(
                name=match.group(1),
                entity_type=SemanticType.CLASS,
                file_path=file_path,
                line_number=line_num
            )
        
        # Find methods
        method_pattern = r'(?:public|private|protected)?\s*(?:static\s+)?(?:\w+\s+)+(\w+)\s*\([^)]*\)\s*{'
        for match in re.finditer(method_pattern, code_content):
            line_num = code_content[:match.start()].count('\n') + 1
            entities[match.group(1)] = SemanticEntity(
                name=match.group(1),
                entity_type=SemanticType.FUNCTION,
                file_path=file_path,
                line_number=line_num
            )
        
        return entities
    
    async def analyze_relationships(
        self,
        entities: Dict[str, SemanticEntity],
        code_content: str
    ) -> List[SemanticRelationship]:
        return []

class GoSemanticAnalyzer:
    """Go-specific semantic analyzer"""
    
    async def extract_entities(self, code_content: str, file_path: Optional[str] = None) -> Dict[str, SemanticEntity]:
        entities = {}
        
        # Find functions
        func_pattern = r'func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\([^)]*\)(?:\s*[^{]*)?{'
        for match in re.finditer(func_pattern, code_content):
            line_num = code_content[:match.start()].count('\n') + 1
            entities[match.group(1)] = SemanticEntity(
                name=match.group(1),
                entity_type=SemanticType.FUNCTION,
                file_path=file_path,
                line_number=line_num
            )
        
        # Find structs
        struct_pattern = r'type\s+(\w+)\s+struct\s*{'
        for match in re.finditer(struct_pattern, code_content):
            line_num = code_content[:match.start()].count('\n') + 1
            entities[match.group(1)] = SemanticEntity(
                name=match.group(1),
                entity_type=SemanticType.CLASS,  # Using CLASS for struct
                file_path=file_path,
                line_number=line_num
            )
        
        return entities
    
    async def analyze_relationships(
        self,
        entities: Dict[str, SemanticEntity],
        code_content: str
    ) -> List[SemanticRelationship]:
        return []

class RustSemanticAnalyzer:
    """Rust-specific semantic analyzer"""
    
    async def extract_entities(self, code_content: str, file_path: Optional[str] = None) -> Dict[str, SemanticEntity]:
        entities = {}
        
        # Find functions
        func_pattern = r'(?:pub\s+)?fn\s+(\w+)\s*\([^)]*\)(?:\s*->\s*[^{]+)?\s*{'
        for match in re.finditer(func_pattern, code_content):
            line_num = code_content[:match.start()].count('\n') + 1
            entities[match.group(1)] = SemanticEntity(
                name=match.group(1),
                entity_type=SemanticType.FUNCTION,
                file_path=file_path,
                line_number=line_num
            )
        
        # Find structs
        struct_pattern = r'(?:pub\s+)?struct\s+(\w+)(?:\s*<[^>]*>)?\s*{'
        for match in re.finditer(struct_pattern, code_content):
            line_num = code_content[:match.start()].count('\n') + 1
            entities[match.group(1)] = SemanticEntity(
                name=match.group(1),
                entity_type=SemanticType.CLASS,
                file_path=file_path,
                line_number=line_num
            )
        
        return entities
    
    async def analyze_relationships(
        self,
        entities: Dict[str, SemanticEntity],
        code_content: str
    ) -> List[SemanticRelationship]:
        return []

class DesignPatternRecognizer:
    """Recognizes design patterns in code"""
    
    async def recognize_patterns(
        self,
        entities: Dict[str, SemanticEntity],
        relationships: List[SemanticRelationship],
        code_content: str
    ) -> List[Tuple[CodePattern, float, Dict[str, Any]]]:
        """Recognize design patterns"""
        
        patterns = []
        
        # Singleton pattern detection
        singleton_confidence = await self._detect_singleton(entities, code_content)
        if singleton_confidence > 0.5:
            patterns.append((CodePattern.SINGLETON, singleton_confidence, {'classes': self._find_singleton_classes(entities, code_content)}))
        
        # Factory pattern detection
        factory_confidence = await self._detect_factory(entities, relationships)
        if factory_confidence > 0.5:
            patterns.append((CodePattern.FACTORY, factory_confidence, {}))
        
        # Observer pattern detection
        observer_confidence = await self._detect_observer(entities, relationships)
        if observer_confidence > 0.5:
            patterns.append((CodePattern.OBSERVER, observer_confidence, {}))
        
        return patterns
    
    async def _detect_singleton(self, entities: Dict[str, SemanticEntity], code_content: str) -> float:
        """Detect singleton pattern"""
        confidence = 0.0
        
        for entity in entities.values():
            if entity.entity_type == SemanticType.CLASS:
                # Check for singleton indicators
                if 'instance' in code_content.lower() and 'private' in code_content.lower():
                    confidence += 0.3
                if '__new__' in code_content or 'getInstance' in code_content:
                    confidence += 0.5
        
        return min(1.0, confidence)
    
    async def _detect_factory(self, entities: Dict[str, SemanticEntity], relationships: List[SemanticRelationship]) -> float:
        """Detect factory pattern"""
        confidence = 0.0
        
        # Look for factory method names
        factory_names = ['create', 'make', 'build', 'factory', 'getInstance']
        
        for entity in entities.values():
            if entity.entity_type == SemanticType.FUNCTION:
                if any(name in entity.name.lower() for name in factory_names):
                    confidence += 0.3
        
        return min(1.0, confidence)
    
    async def _detect_observer(self, entities: Dict[str, SemanticEntity], relationships: List[SemanticRelationship]) -> float:
        """Detect observer pattern"""
        confidence = 0.0
        
        # Look for observer-related methods
        observer_methods = ['notify', 'update', 'subscribe', 'unsubscribe', 'addObserver']
        
        for entity in entities.values():
            if entity.entity_type == SemanticType.FUNCTION:
                if any(method in entity.name for method in observer_methods):
                    confidence += 0.4
        
        return min(1.0, confidence)
    
    def _find_singleton_classes(self, entities: Dict[str, SemanticEntity], code_content: str) -> List[str]:
        """Find classes that might be singletons"""
        singleton_classes = []
        
        for name, entity in entities.items():
            if entity.entity_type == SemanticType.CLASS:
                if 'instance' in code_content.lower() and name.lower() in code_content.lower():
                    singleton_classes.append(name)
        
        return singleton_classes

class ArchitectureAnalyzer:
    """Analyzes architectural patterns and insights"""
    
    async def analyze_architecture(
        self,
        entities: Dict[str, SemanticEntity],
        relationships: List[SemanticRelationship],
        knowledge_graph: nx.DiGraph
    ) -> Dict[str, Any]:
        """Analyze architectural patterns"""
        
        insights = {
            'modularity': await self._analyze_modularity(entities, relationships),
            'coupling': await self._analyze_coupling(relationships, knowledge_graph),
            'cohesion': await self._analyze_cohesion(entities, relationships),
            'layering': await self._detect_layered_architecture(entities, relationships),
            'patterns': await self._detect_architectural_patterns(entities, relationships)
        }
        
        return insights
    
    async def _analyze_modularity(self, entities: Dict[str, SemanticEntity], relationships: List[SemanticRelationship]) -> Dict[str, Any]:
        """Analyze code modularity"""
        
        # Simple modularity analysis
        total_entities = len(entities)
        classes = len([e for e in entities.values() if e.entity_type == SemanticType.CLASS])
        functions = len([e for e in entities.values() if e.entity_type == SemanticType.FUNCTION])
        
        return {
            'total_entities': total_entities,
            'class_count': classes,
            'function_count': functions,
            'class_to_function_ratio': classes / max(1, functions),
            'modularity_score': min(1.0, classes / max(1, total_entities))
        }
    
    async def _analyze_coupling(self, relationships: List[SemanticRelationship], knowledge_graph: nx.DiGraph) -> Dict[str, Any]:
        """Analyze coupling between components"""
        
        if not knowledge_graph:
            return {'coupling_score': 0.5}
        
        # Calculate coupling metrics
        total_edges = knowledge_graph.number_of_edges()
        total_nodes = knowledge_graph.number_of_nodes()
        
        coupling_score = total_edges / max(1, total_nodes * (total_nodes - 1) / 2)
        
        return {
            'coupling_score': coupling_score,
            'total_relationships': total_edges,
            'entities': total_nodes
        }
    
    async def _analyze_cohesion(self, entities: Dict[str, SemanticEntity], relationships: List[SemanticRelationship]) -> Dict[str, Any]:
        """Analyze cohesion within components"""
        
        # Simplified cohesion analysis
        class_cohesion = {}
        
        for entity in entities.values():
            if entity.entity_type == SemanticType.CLASS:
                # Calculate cohesion based on method relationships
                related_methods = [r for r in relationships 
                                 if r.source.startswith(entity.name) or r.target.startswith(entity.name)]
                class_cohesion[entity.name] = len(related_methods)
        
        avg_cohesion = sum(class_cohesion.values()) / max(1, len(class_cohesion))
        
        return {
            'average_class_cohesion': avg_cohesion,
            'class_cohesion': class_cohesion
        }
    
    async def _detect_layered_architecture(self, entities: Dict[str, SemanticEntity], relationships: List[SemanticRelationship]) -> Dict[str, Any]:
        """Detect layered architecture patterns"""
        
        # Simple layer detection based on naming conventions
        layers = {
            'presentation': [],
            'business': [],
            'data': [],
            'utility': []
        }
        
        for name, entity in entities.items():
            name_lower = name.lower()
            if any(term in name_lower for term in ['view', 'controller', 'ui', 'frontend']):
                layers['presentation'].append(name)
            elif any(term in name_lower for term in ['service', 'manager', 'business', 'logic']):
                layers['business'].append(name)
            elif any(term in name_lower for term in ['repository', 'dao', 'database', 'model']):
                layers['data'].append(name)
            elif any(term in name_lower for term in ['util', 'helper', 'common']):
                layers['utility'].append(name)
        
        return {
            'detected_layers': {k: v for k, v in layers.items() if v},
            'is_layered': len([v for v in layers.values() if v]) >= 2
        }
    
    async def _detect_architectural_patterns(self, entities: Dict[str, SemanticEntity], relationships: List[SemanticRelationship]) -> List[str]:
        """Detect architectural patterns"""
        
        patterns = []
        
        # MVC pattern detection
        has_model = any('model' in entity.name.lower() for entity in entities.values())
        has_view = any('view' in entity.name.lower() for entity in entities.values())
        has_controller = any('controller' in entity.name.lower() for entity in entities.values())
        
        if has_model and has_view and has_controller:
            patterns.append('MVC')
        
        # Repository pattern detection
        if any('repository' in entity.name.lower() for entity in entities.values()):
            patterns.append('Repository')
        
        # Service pattern detection
        if any('service' in entity.name.lower() for entity in entities.values()):
            patterns.append('Service Layer')
        
        return patterns

class CodeQualityAssessor:
    """Assesses code quality based on semantic analysis"""
    
    async def assess_quality(
        self,
        entities: Dict[str, SemanticEntity],
        relationships: List[SemanticRelationship],
        code_content: str,
        language: str
    ) -> Dict[str, Any]:
        """Assess overall code quality"""
        
        quality_metrics = {
            'maintainability_index': await self._calculate_maintainability_index(entities, code_content),
            'documentation_coverage': await self._calculate_documentation_coverage(entities),
            'naming_consistency': await self._assess_naming_consistency(entities),
            'error_handling': await self._assess_error_handling(code_content, language),
            'test_coverage_indicators': await self._assess_test_coverage_indicators(entities, code_content),
            'security_indicators': await self._assess_security_indicators(code_content, language)
        }
        
        # Overall quality score
        scores = [v for v in quality_metrics.values() if isinstance(v, (int, float))]
        quality_metrics['overall_quality'] = sum(scores) / len(scores) if scores else 0.5
        
        return quality_metrics
    
    async def _calculate_maintainability_index(self, entities: Dict[str, SemanticEntity], code_content: str) -> float:
        """Calculate maintainability index"""
        
        # Simplified maintainability calculation
        total_complexity = sum(entity.complexity_score for entity in entities.values())
        total_entities = len(entities)
        avg_complexity = total_complexity / max(1, total_entities)
        
        # Lines of code
        loc = len(code_content.split('\n'))
        
        # Simple maintainability formula
        maintainability = max(0, 100 - (avg_complexity * 10) - (loc / 100))
        
        return min(100, max(0, maintainability))
    
    async def _calculate_documentation_coverage(self, entities: Dict[str, SemanticEntity]) -> float:
        """Calculate documentation coverage"""
        
        documented = sum(1 for entity in entities.values() 
                        if entity.docstring and entity.entity_type in [SemanticType.FUNCTION, SemanticType.CLASS])
        documentable = len([entity for entity in entities.values() 
                          if entity.entity_type in [SemanticType.FUNCTION, SemanticType.CLASS]])
        
        return documented / max(1, documentable)
    
    async def _assess_naming_consistency(self, entities: Dict[str, SemanticEntity]) -> float:
        """Assess naming consistency"""
        
        # Simple naming assessment
        consistent_names = 0
        total_names = 0
        
        for entity in entities.values():
            total_names += 1
            
            # Check naming conventions
            if entity.entity_type == SemanticType.CLASS:
                if entity.name[0].isupper():  # PascalCase for classes
                    consistent_names += 1
            elif entity.entity_type == SemanticType.FUNCTION:
                if entity.name.islower() or '_' in entity.name:  # snake_case or camelCase
                    consistent_names += 1
        
        return consistent_names / max(1, total_names)
    
    async def _assess_error_handling(self, code_content: str, language: str) -> float:
        """Assess error handling practices"""
        
        error_handling_score = 0.0
        
        if language == 'python':
            if 'try:' in code_content and 'except' in code_content:
                error_handling_score += 0.5
            if 'logging' in code_content or 'logger' in code_content:
                error_handling_score += 0.3
            if 'raise' in code_content:
                error_handling_score += 0.2
        elif language == 'javascript':
            if 'try {' in code_content and 'catch' in code_content:
                error_handling_score += 0.5
            if 'console.error' in code_content:
                error_handling_score += 0.3
            if 'throw' in code_content:
                error_handling_score += 0.2
        
        return min(1.0, error_handling_score)
    
    async def _assess_test_coverage_indicators(self, entities: Dict[str, SemanticEntity], code_content: str) -> float:
        """Assess indicators of test coverage"""
        
        test_indicators = 0.0
        
        # Look for test-related entities
        test_entities = [e for e in entities.values() if 'test' in e.name.lower()]
        if test_entities:
            test_indicators += 0.4
        
        # Look for test-related imports or frameworks
        test_frameworks = ['unittest', 'pytest', 'jest', 'mocha', 'junit', 'testng']
        if any(framework in code_content.lower() for framework in test_frameworks):
            test_indicators += 0.3
        
        # Look for assertion statements
        if any(assertion in code_content.lower() for assertion in ['assert', 'expect', 'should']):
            test_indicators += 0.3
        
        return min(1.0, test_indicators)
    
    async def _assess_security_indicators(self, code_content: str, language: str) -> float:
        """Assess security indicators"""
        
        security_score = 1.0  # Start with perfect score, deduct for issues
        
        # Common security issues
        security_issues = [
            'password', 'secret', 'api_key', 'token',  # Hardcoded credentials
            'eval(', 'exec(',  # Code injection
            'md5(', 'sha1(',  # Weak hashing
            'http://'  # Insecure protocols
        ]
        
        for issue in security_issues:
            if issue in code_content.lower():
                security_score -= 0.2
        
        return max(0.0, security_score)

class ComplexityAnalyzer:
    """Analyzes code complexity metrics"""
    
    async def analyze_complexity(
        self,
        entities: Dict[str, SemanticEntity],
        relationships: List[SemanticRelationship],
        code_content: str
    ) -> Dict[str, float]:
        """Analyze various complexity metrics"""
        
        metrics = {
            'cyclomatic_complexity': await self._calculate_cyclomatic_complexity(entities),
            'cognitive_complexity': await self._calculate_cognitive_complexity(code_content),
            'structural_complexity': await self._calculate_structural_complexity(relationships),
            'size_complexity': await self._calculate_size_complexity(entities, code_content)
        }
        
        return metrics
    
    async def _calculate_cyclomatic_complexity(self, entities: Dict[str, SemanticEntity]) -> float:
        """Calculate average cyclomatic complexity"""
        complexities = [entity.complexity_score for entity in entities.values() if entity.complexity_score > 0]
        return sum(complexities) / max(1, len(complexities))
    
    async def _calculate_cognitive_complexity(self, code_content: str) -> float:
        """Calculate cognitive complexity (simplified)"""
        # This is a simplified version
        nesting_indicators = code_content.count('if ') + code_content.count('for ') + code_content.count('while ')
        return min(20, nesting_indicators)  # Cap at 20
    
    async def _calculate_structural_complexity(self, relationships: List[SemanticRelationship]) -> float:
        """Calculate structural complexity based on relationships"""
        return len(relationships) / 10.0  # Normalized to ~1.0
    
    async def _calculate_size_complexity(self, entities: Dict[str, SemanticEntity], code_content: str) -> float:
        """Calculate size-based complexity"""
        loc = len(code_content.split('\n'))
        entity_count = len(entities)
        
        return (loc / 100) + (entity_count / 10)  # Normalized complexity

class KnowledgeGraphBuilder:
    """Builds knowledge graph from semantic entities and relationships"""
    
    async def build_graph(
        self,
        entities: Dict[str, SemanticEntity],
        relationships: List[SemanticRelationship]
    ) -> nx.DiGraph:
        """Build knowledge graph"""
        
        graph = nx.DiGraph()
        
        # Add nodes (entities)
        for name, entity in entities.items():
            graph.add_node(name, 
                          type=entity.entity_type.value,
                          complexity=entity.complexity_score,
                          line_number=entity.line_number)
        
        # Add edges (relationships)
        for rel in relationships:
            if rel.source in graph and rel.target in graph:
                graph.add_edge(rel.source, rel.target,
                              relationship=rel.relationship_type.value,
                              strength=rel.strength)
        
        return graph

class SemanticSimilarityAnalyzer:
    """Analyzes semantic similarity between code structures"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
    
    async def calculate_similarity(
        self,
        analysis1: CodeSemanticsAnalysis,
        analysis2: CodeSemanticsAnalysis
    ) -> Dict[str, Any]:
        """Calculate similarity between two code analyses"""
        
        similarity_result = {
            'overall_similarity': 0.0,
            'structural_similarity': 0.0,
            'semantic_similarity': 0.0,
            'entity_similarity': 0.0,
            'relationship_similarity': 0.0
        }
        
        # Entity similarity
        entity_sim = await self._calculate_entity_similarity(analysis1.entities, analysis2.entities)
        similarity_result['entity_similarity'] = entity_sim
        
        # Relationship similarity
        rel_sim = await self._calculate_relationship_similarity(analysis1.relationships, analysis2.relationships)
        similarity_result['relationship_similarity'] = rel_sim
        
        # Overall similarity
        similarity_result['overall_similarity'] = (entity_sim + rel_sim) / 2
        similarity_result['structural_similarity'] = similarity_result['overall_similarity']
        similarity_result['semantic_similarity'] = similarity_result['overall_similarity']
        
        return similarity_result
    
    async def _calculate_entity_similarity(
        self,
        entities1: Dict[str, SemanticEntity],
        entities2: Dict[str, SemanticEntity]
    ) -> float:
        """Calculate entity similarity"""
        
        names1 = set(entities1.keys())
        names2 = set(entities2.keys())
        
        if not names1 and not names2:
            return 1.0
        
        intersection = len(names1.intersection(names2))
        union = len(names1.union(names2))
        
        return intersection / max(1, union)
    
    async def _calculate_relationship_similarity(
        self,
        relationships1: List[SemanticRelationship],
        relationships2: List[SemanticRelationship]
    ) -> float:
        """Calculate relationship similarity"""
        
        rel_set1 = set((r.source, r.target, r.relationship_type.value) for r in relationships1)
        rel_set2 = set((r.source, r.target, r.relationship_type.value) for r in relationships2)
        
        if not rel_set1 and not rel_set2:
            return 1.0
        
        intersection = len(rel_set1.intersection(rel_set2))
        union = len(rel_set1.union(rel_set2))
        
        return intersection / max(1, union)

# Factory function
def create_semantic_analyzer() -> SemanticCodeAnalyzer:
    """Create and configure semantic code analyzer"""
    return SemanticCodeAnalyzer()