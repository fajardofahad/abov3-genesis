"""
ABOV3 Genesis - Knowledge Graph Builder
Constructs and manages knowledge graphs of code relationships
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, deque
import networkx as nx
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)

class RelationshipType(Enum):
    """Types of relationships between code entities"""
    IMPORTS = "imports"
    CALLS = "calls"
    INHERITS = "inherits"
    CONTAINS = "contains"
    USES = "uses"
    DEFINES = "defines"
    DEPENDS_ON = "depends_on"
    IMPLEMENTS = "implements"
    OVERRIDES = "overrides"
    REFERENCES = "references"
    SIMILAR_TO = "similar_to"
    MODULE_OF = "module_of"
    PACKAGE_OF = "package_of"

class EntityType(Enum):
    """Types of code entities"""
    FILE = "file"
    MODULE = "module"
    PACKAGE = "package"
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    INTERFACE = "interface"
    VARIABLE = "variable"
    CONSTANT = "constant"
    PROPERTY = "property"
    IMPORT = "import"
    NAMESPACE = "namespace"

@dataclass
class CodeEntity:
    """Represents a code entity in the knowledge graph"""
    id: str
    name: str
    entity_type: EntityType
    file_path: str
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    language: str = "unknown"
    signature: Optional[str] = None
    docstring: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            # Generate unique ID from entity properties
            id_string = f"{self.entity_type.value}:{self.file_path}:{self.name}:{self.line_start}"
            self.id = hashlib.md5(id_string.encode()).hexdigest()[:12]

@dataclass
class CodeRelationship:
    """Represents a relationship between code entities"""
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    strength: float = 1.0  # Relationship strength (0.0 to 1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def id(self) -> str:
        return f"{self.source_id}-{self.relationship_type.value}-{self.target_id}"

@dataclass
class GraphMetrics:
    """Metrics about the knowledge graph"""
    total_entities: int
    total_relationships: int
    entity_type_distribution: Dict[str, int]
    relationship_type_distribution: Dict[str, int]
    average_degree: float
    clustering_coefficient: float
    connected_components: int
    diameter: Optional[int]
    density: float

class KnowledgeGraphBuilder:
    """
    Builds and manages knowledge graphs of code relationships
    Uses NetworkX for efficient graph operations
    """
    
    def __init__(self, enable_caching: bool = True):
        self.enable_caching = enable_caching
        
        # Main graph
        self.graph = nx.MultiDiGraph()  # Directed multigraph to handle multiple relationship types
        
        # Entity and relationship storage
        self.entities: Dict[str, CodeEntity] = {}
        self.relationships: Dict[str, CodeRelationship] = {}
        
        # Caches for performance
        self.subgraph_cache: Dict[str, nx.Graph] = {}
        self.path_cache: Dict[Tuple[str, str], List[List[str]]] = {}
        self.community_cache: Optional[List[Set[str]]] = None
        
        # Performance tracking
        self.build_stats = {
            'entities_processed': 0,
            'relationships_processed': 0,
            'last_build_time': 0.0,
            'graph_build_time': 0.0
        }
        
        logger.info("KnowledgeGraphBuilder initialized")
    
    async def initialize(self):
        """Initialize the knowledge graph builder"""
        logger.info("Initializing KnowledgeGraphBuilder...")
        # Any initialization logic here
        logger.info("KnowledgeGraphBuilder initialized")
    
    async def build_graph_from_index(self, code_indexer) -> nx.Graph:
        """Build knowledge graph from code index"""
        start_time = time.time()
        logger.info("Building knowledge graph from code index...")
        
        # Clear existing graph
        self.graph.clear()
        self.entities.clear()
        self.relationships.clear()
        self._clear_caches()
        
        # Process all files in the index
        for file_path, file_info in code_indexer.file_cache.items():
            await self._process_file(file_info, code_indexer)
        
        # Add inter-file relationships
        await self._build_inter_file_relationships(code_indexer)
        
        # Calculate and cache graph metrics
        await self._calculate_graph_metrics()
        
        self.build_stats['graph_build_time'] = time.time() - start_time
        logger.info(f"Knowledge graph built in {self.build_stats['graph_build_time']:.2f}s")
        logger.info(f"Graph contains {len(self.entities)} entities and {len(self.relationships)} relationships")
        
        return self.graph
    
    async def _process_file(self, file_info, code_indexer):
        """Process a single file and add its entities/relationships to the graph"""
        try:
            # Create file entity
            file_entity = CodeEntity(
                id="",  # Will be auto-generated
                name=Path(file_info.path).name,
                entity_type=EntityType.FILE,
                file_path=file_info.path,
                language=file_info.language,
                metadata={
                    'size': file_info.size,
                    'lines': file_info.lines,
                    'complexity_score': file_info.complexity_score,
                    'maintainability_index': file_info.maintainability_index,
                    'summary': file_info.summary
                }
            )
            self._add_entity(file_entity)
            
            # Create module/package entities
            path_parts = Path(file_info.path).parts
            if len(path_parts) > 1:
                # Create package entities for directory structure
                current_path = ""
                parent_id = None
                
                for i, part in enumerate(path_parts[:-1]):  # Exclude filename
                    current_path = str(Path(current_path) / part) if current_path else part
                    
                    entity_type = EntityType.PACKAGE if i == 0 else EntityType.MODULE
                    package_entity = CodeEntity(
                        id="",
                        name=part,
                        entity_type=entity_type,
                        file_path=current_path,
                        language=file_info.language,
                        metadata={'is_directory': True}
                    )
                    
                    package_id = self._add_entity(package_entity)
                    
                    # Add containment relationship
                    if parent_id:
                        self._add_relationship(parent_id, package_id, RelationshipType.CONTAINS)
                    
                    parent_id = package_id
                
                # File is contained in its parent directory
                if parent_id:
                    self._add_relationship(parent_id, file_entity.id, RelationshipType.CONTAINS)
            
            # Process functions
            if file_info.path in code_indexer.function_cache:
                for func_info in code_indexer.function_cache[file_info.path]:
                    func_entity = CodeEntity(
                        id="",
                        name=func_info.name,
                        entity_type=EntityType.METHOD if func_info.is_method else EntityType.FUNCTION,
                        file_path=file_info.path,
                        line_start=func_info.line_start,
                        line_end=func_info.line_end,
                        language=file_info.language,
                        signature=f"{func_info.name}({', '.join(func_info.parameters)})",
                        docstring=func_info.docstring,
                        metadata={
                            'parameters': func_info.parameters,
                            'return_type': func_info.return_type,
                            'complexity': func_info.complexity,
                            'is_async': func_info.is_async,
                            'calls': func_info.calls
                        }
                    )
                    
                    func_id = self._add_entity(func_entity)
                    
                    # Function is defined in file
                    self._add_relationship(file_entity.id, func_id, RelationshipType.DEFINES)
                    
                    # Function calls relationships
                    for called_func in func_info.calls:
                        # Find the called function entity (simplified - would need better resolution)
                        called_id = self._find_entity_by_name(called_func, EntityType.FUNCTION)
                        if called_id:
                            self._add_relationship(func_id, called_id, RelationshipType.CALLS)
                    
                    # Method belongs to class
                    if func_info.is_method and func_info.class_name:
                        class_id = self._find_entity_by_name(func_info.class_name, EntityType.CLASS)
                        if class_id:
                            self._add_relationship(class_id, func_id, RelationshipType.CONTAINS)
            
            # Process classes
            if file_info.path in code_indexer.class_cache:
                for class_info in code_indexer.class_cache[file_info.path]:
                    class_entity = CodeEntity(
                        id="",
                        name=class_info.name,
                        entity_type=EntityType.CLASS,
                        file_path=file_info.path,
                        line_start=class_info.line_start,
                        line_end=class_info.line_end,
                        language=file_info.language,
                        docstring=class_info.docstring,
                        metadata={
                            'methods': class_info.methods,
                            'properties': class_info.properties,
                            'base_classes': class_info.base_classes,
                            'is_abstract': class_info.is_abstract
                        }
                    )
                    
                    class_id = self._add_entity(class_entity)
                    
                    # Class is defined in file
                    self._add_relationship(file_entity.id, class_id, RelationshipType.DEFINES)
                    
                    # Inheritance relationships
                    for base_class in class_info.base_classes:
                        base_id = self._find_entity_by_name(base_class, EntityType.CLASS)
                        if base_id:
                            self._add_relationship(class_id, base_id, RelationshipType.INHERITS)
            
            # Process imports
            for import_name in file_info.imports:
                import_entity = CodeEntity(
                    id="",
                    name=import_name,
                    entity_type=EntityType.IMPORT,
                    file_path=file_info.path,
                    language=file_info.language,
                    metadata={'import_type': 'module'}
                )
                
                import_id = self._add_entity(import_entity)
                self._add_relationship(file_entity.id, import_id, RelationshipType.IMPORTS)
            
            self.build_stats['entities_processed'] += 1
            
        except Exception as e:
            logger.error(f"Error processing file {file_info.path}: {e}")
    
    async def _build_inter_file_relationships(self, code_indexer):
        """Build relationships between entities across files"""
        logger.debug("Building inter-file relationships...")
        
        # Build import-to-definition relationships
        for entity_id, entity in self.entities.items():
            if entity.entity_type == EntityType.IMPORT:
                # Try to find the actual module/package being imported
                target_entity = self._find_module_entity(entity.name)
                if target_entity:
                    self._add_relationship(entity_id, target_entity, RelationshipType.REFERENCES)
        
        # Build call relationships across files
        await self._build_cross_file_calls(code_indexer)
        
        # Build similarity relationships
        await self._build_similarity_relationships()
    
    async def _build_cross_file_calls(self, code_indexer):
        """Build function call relationships across files"""
        # Create a name-to-entity mapping for functions
        function_map = defaultdict(list)
        
        for entity_id, entity in self.entities.items():
            if entity.entity_type in [EntityType.FUNCTION, EntityType.METHOD]:
                function_map[entity.name].append(entity_id)
        
        # Connect function calls
        for entity_id, entity in self.entities.items():
            if entity.entity_type in [EntityType.FUNCTION, EntityType.METHOD]:
                calls = entity.metadata.get('calls', [])
                for called_name in calls:
                    # Find potential targets
                    targets = function_map.get(called_name, [])
                    for target_id in targets:
                        if target_id != entity_id:  # Don't connect to self
                            # Calculate relationship strength based on context
                            strength = self._calculate_call_strength(entity, self.entities[target_id])
                            self._add_relationship(
                                entity_id, 
                                target_id, 
                                RelationshipType.CALLS,
                                strength
                            )
    
    async def _build_similarity_relationships(self):
        """Build similarity relationships between entities"""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated similarity measures
        
        entities_by_type = defaultdict(list)
        for entity_id, entity in self.entities.items():
            entities_by_type[entity.entity_type].append(entity_id)
        
        # Compare entities of the same type
        for entity_type, entity_ids in entities_by_type.items():
            if entity_type in [EntityType.FUNCTION, EntityType.CLASS]:
                for i, entity1_id in enumerate(entity_ids):
                    for entity2_id in entity_ids[i+1:]:
                        similarity = self._calculate_entity_similarity(
                            self.entities[entity1_id],
                            self.entities[entity2_id]
                        )
                        
                        if similarity > 0.7:  # High similarity threshold
                            self._add_relationship(
                                entity1_id, 
                                entity2_id, 
                                RelationshipType.SIMILAR_TO, 
                                similarity
                            )
    
    def _add_entity(self, entity: CodeEntity) -> str:
        """Add entity to graph and return its ID"""
        if not entity.id:
            # Generate ID
            entity.__post_init__()
        
        # Avoid duplicates
        if entity.id in self.entities:
            return entity.id
        
        self.entities[entity.id] = entity
        self.graph.add_node(
            entity.id,
            name=entity.name,
            type=entity.entity_type.value,
            file_path=entity.file_path,
            metadata=entity.metadata
        )
        
        return entity.id
    
    def _add_relationship(
        self, 
        source_id: str, 
        target_id: str, 
        relationship_type: RelationshipType, 
        strength: float = 1.0,
        metadata: Dict[str, Any] = None
    ):
        """Add relationship to graph"""
        relationship = CodeRelationship(
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            strength=strength,
            metadata=metadata or {}
        )
        
        # Avoid duplicate relationships
        if relationship.id in self.relationships:
            # Update strength if higher
            existing = self.relationships[relationship.id]
            if strength > existing.strength:
                existing.strength = strength
            return
        
        self.relationships[relationship.id] = relationship
        self.graph.add_edge(
            source_id,
            target_id,
            type=relationship_type.value,
            strength=strength,
            metadata=metadata or {}
        )
        
        self.build_stats['relationships_processed'] += 1
    
    def _find_entity_by_name(self, name: str, entity_type: EntityType) -> Optional[str]:
        """Find entity ID by name and type"""
        for entity_id, entity in self.entities.items():
            if entity.name == name and entity.entity_type == entity_type:
                return entity_id
        return None
    
    def _find_module_entity(self, module_name: str) -> Optional[str]:
        """Find module entity by import name"""
        # Simple heuristic - could be improved
        for entity_id, entity in self.entities.items():
            if entity.entity_type in [EntityType.MODULE, EntityType.PACKAGE, EntityType.FILE]:
                if module_name in entity.file_path or module_name == entity.name:
                    return entity_id
        return None
    
    def _calculate_call_strength(self, caller: CodeEntity, callee: CodeEntity) -> float:
        """Calculate strength of call relationship"""
        strength = 1.0
        
        # Same file = stronger relationship
        if caller.file_path == callee.file_path:
            strength *= 1.5
        
        # Same language = stronger relationship
        if caller.language == callee.language:
            strength *= 1.2
        
        return min(1.0, strength)
    
    def _calculate_entity_similarity(self, entity1: CodeEntity, entity2: CodeEntity) -> float:
        """Calculate similarity between two entities"""
        if entity1.entity_type != entity2.entity_type:
            return 0.0
        
        similarity = 0.0
        
        # Name similarity (simple)
        if entity1.name == entity2.name:
            similarity += 0.5
        elif entity1.name.lower() in entity2.name.lower() or entity2.name.lower() in entity1.name.lower():
            similarity += 0.3
        
        # Signature similarity (for functions)
        if entity1.signature and entity2.signature:
            if entity1.signature == entity2.signature:
                similarity += 0.3
            elif len(set(entity1.signature.split()) & set(entity2.signature.split())) > 0:
                similarity += 0.1
        
        # Same file = more similar
        if entity1.file_path == entity2.file_path:
            similarity += 0.2
        
        return min(1.0, similarity)
    
    async def build_subgraph(self, entity_ids: List[str], depth: int = 2) -> nx.Graph:
        """Build subgraph around specific entities"""
        cache_key = f"{hash(tuple(sorted(entity_ids)))}_{depth}"
        
        if self.enable_caching and cache_key in self.subgraph_cache:
            return self.subgraph_cache[cache_key]
        
        # Collect nodes within specified depth
        subgraph_nodes = set(entity_ids)
        current_level = set(entity_ids)
        
        for _ in range(depth):
            next_level = set()
            for node in current_level:
                if node in self.graph:
                    # Add neighbors
                    neighbors = set(self.graph.predecessors(node)) | set(self.graph.successors(node))
                    next_level.update(neighbors)
            
            subgraph_nodes.update(next_level)
            current_level = next_level
        
        # Create subgraph
        subgraph = self.graph.subgraph(subgraph_nodes).copy()
        
        if self.enable_caching:
            self.subgraph_cache[cache_key] = subgraph
        
        return subgraph
    
    def extract_concepts(self, graph: nx.Graph = None) -> List[str]:
        """Extract key concepts from the graph"""
        if graph is None:
            graph = self.graph
        
        concepts = []
        
        # Get high-degree nodes (important entities)
        degree_dict = dict(graph.degree())
        sorted_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)
        
        for node_id, degree in sorted_nodes[:20]:  # Top 20 by degree
            if node_id in self.entities:
                entity = self.entities[node_id]
                concepts.append(f"{entity.name} ({entity.entity_type.value})")
        
        return concepts
    
    def extract_relationships(self, graph: nx.Graph = None) -> List[str]:
        """Extract key relationships from the graph"""
        if graph is None:
            graph = self.graph
        
        relationships = []
        
        # Count relationship types
        relationship_counts = defaultdict(int)
        for edge in graph.edges(data=True):
            rel_type = edge[2].get('type', 'unknown')
            relationship_counts[rel_type] += 1
        
        for rel_type, count in sorted(relationship_counts.items(), key=lambda x: x[1], reverse=True):
            relationships.append(f"{rel_type}: {count} relationships")
        
        return relationships
    
    async def find_shortest_paths(self, source_id: str, target_id: str, max_paths: int = 5) -> List[List[str]]:
        """Find shortest paths between two entities"""
        cache_key = (source_id, target_id)
        
        if self.enable_caching and cache_key in self.path_cache:
            return self.path_cache[cache_key][:max_paths]
        
        try:
            # Find all simple paths (limited to avoid exponential explosion)
            paths = list(nx.all_simple_paths(
                self.graph, 
                source_id, 
                target_id, 
                cutoff=5  # Maximum path length
            ))
            
            # Sort by path length
            paths.sort(key=len)
            paths = paths[:max_paths]
            
            if self.enable_caching:
                self.path_cache[cache_key] = paths
            
            return paths
            
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    async def find_communities(self) -> List[Set[str]]:
        """Find communities/clusters in the graph"""
        if self.community_cache is not None:
            return self.community_cache
        
        try:
            # Convert to undirected for community detection
            undirected = self.graph.to_undirected()
            
            # Use a simple community detection algorithm
            # In practice, you might want to use more sophisticated algorithms
            communities = []
            
            # Simple approach: connected components
            for component in nx.connected_components(undirected):
                if len(component) > 1:  # Ignore singletons
                    communities.append(component)
            
            self.community_cache = communities
            return communities
            
        except Exception as e:
            logger.error(f"Error finding communities: {e}")
            return []
    
    async def _calculate_graph_metrics(self):
        """Calculate and cache graph metrics"""
        try:
            # Basic metrics
            num_nodes = self.graph.number_of_nodes()
            num_edges = self.graph.number_of_edges()
            
            # Entity type distribution
            entity_types = defaultdict(int)
            for entity in self.entities.values():
                entity_types[entity.entity_type.value] += 1
            
            # Relationship type distribution
            rel_types = defaultdict(int)
            for rel in self.relationships.values():
                rel_types[rel.relationship_type.value] += 1
            
            # Network metrics
            if num_nodes > 0:
                avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0
                density = nx.density(self.graph)
                
                # Connected components
                undirected = self.graph.to_undirected()
                connected_components = nx.number_connected_components(undirected)
                
                # Clustering coefficient (for undirected version)
                try:
                    clustering = nx.average_clustering(undirected)
                except:
                    clustering = 0.0
                
                # Diameter (for largest component)
                diameter = None
                if connected_components > 0:
                    largest_cc = max(nx.connected_components(undirected), key=len)
                    if len(largest_cc) > 1:
                        try:
                            subgraph = undirected.subgraph(largest_cc)
                            diameter = nx.diameter(subgraph)
                        except:
                            diameter = None
            else:
                avg_degree = 0
                density = 0
                connected_components = 0
                clustering = 0
                diameter = None
            
            self.graph_metrics = GraphMetrics(
                total_entities=num_nodes,
                total_relationships=num_edges,
                entity_type_distribution=dict(entity_types),
                relationship_type_distribution=dict(rel_types),
                average_degree=avg_degree,
                clustering_coefficient=clustering,
                connected_components=connected_components,
                diameter=diameter,
                density=density
            )
            
        except Exception as e:
            logger.error(f"Error calculating graph metrics: {e}")
    
    def get_entity_neighbors(self, entity_id: str, relationship_types: List[RelationshipType] = None) -> List[str]:
        """Get neighbors of an entity, optionally filtered by relationship type"""
        if entity_id not in self.graph:
            return []
        
        neighbors = []
        
        # Outgoing edges
        for target in self.graph.successors(entity_id):
            edge_data = self.graph.get_edge_data(entity_id, target)
            if edge_data:
                for edge in edge_data.values():
                    if not relationship_types or edge.get('type') in [rt.value for rt in relationship_types]:
                        neighbors.append(target)
                        break
        
        # Incoming edges
        for source in self.graph.predecessors(entity_id):
            edge_data = self.graph.get_edge_data(source, entity_id)
            if edge_data:
                for edge in edge_data.values():
                    if not relationship_types or edge.get('type') in [rt.value for rt in relationship_types]:
                        neighbors.append(source)
                        break
        
        return list(set(neighbors))  # Remove duplicates
    
    def get_entity_info(self, entity_id: str) -> Optional[CodeEntity]:
        """Get detailed information about an entity"""
        return self.entities.get(entity_id)
    
    def find_entities_by_type(self, entity_type: EntityType, file_path: str = None) -> List[str]:
        """Find entities by type, optionally filtered by file"""
        results = []
        
        for entity_id, entity in self.entities.items():
            if entity.entity_type == entity_type:
                if file_path is None or entity.file_path == file_path:
                    results.append(entity_id)
        
        return results
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        if hasattr(self, 'graph_metrics'):
            return {
                'entities': self.graph_metrics.total_entities,
                'relationships': self.graph_metrics.total_relationships,
                'entity_types': self.graph_metrics.entity_type_distribution,
                'relationship_types': self.graph_metrics.relationship_type_distribution,
                'avg_degree': self.graph_metrics.average_degree,
                'clustering': self.graph_metrics.clustering_coefficient,
                'connected_components': self.graph_metrics.connected_components,
                'diameter': self.graph_metrics.diameter,
                'density': self.graph_metrics.density,
                'build_stats': self.build_stats
            }
        else:
            return {
                'entities': len(self.entities),
                'relationships': len(self.relationships),
                'build_stats': self.build_stats
            }
    
    def _clear_caches(self):
        """Clear all caches"""
        self.subgraph_cache.clear()
        self.path_cache.clear()
        self.community_cache = None
    
    async def export_graph(self, file_path: str, format: str = 'gexf'):
        """Export graph to file"""
        try:
            if format == 'gexf':
                nx.write_gexf(self.graph, file_path)
            elif format == 'graphml':
                nx.write_graphml(self.graph, file_path)
            elif format == 'json':
                data = {
                    'entities': {eid: {
                        'name': e.name,
                        'type': e.entity_type.value,
                        'file_path': e.file_path,
                        'metadata': e.metadata
                    } for eid, e in self.entities.items()},
                    'relationships': {rid: {
                        'source': r.source_id,
                        'target': r.target_id,
                        'type': r.relationship_type.value,
                        'strength': r.strength,
                        'metadata': r.metadata
                    } for rid, r in self.relationships.items()}
                }
                
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
            
            logger.info(f"Graph exported to {file_path} in {format} format")
            
        except Exception as e:
            logger.error(f"Error exporting graph: {e}")
    
    async def import_graph(self, file_path: str, format: str = 'json'):
        """Import graph from file"""
        try:
            if format == 'json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Import entities
                for eid, entity_data in data.get('entities', {}).items():
                    entity = CodeEntity(
                        id=eid,
                        name=entity_data['name'],
                        entity_type=EntityType(entity_data['type']),
                        file_path=entity_data['file_path'],
                        metadata=entity_data.get('metadata', {})
                    )
                    self.entities[eid] = entity
                    self.graph.add_node(eid, **entity_data)
                
                # Import relationships
                for rid, rel_data in data.get('relationships', {}).items():
                    relationship = CodeRelationship(
                        source_id=rel_data['source'],
                        target_id=rel_data['target'],
                        relationship_type=RelationshipType(rel_data['type']),
                        strength=rel_data.get('strength', 1.0),
                        metadata=rel_data.get('metadata', {})
                    )
                    self.relationships[rid] = relationship
                    self.graph.add_edge(
                        rel_data['source'],
                        rel_data['target'],
                        type=rel_data['type'],
                        strength=rel_data.get('strength', 1.0),
                        metadata=rel_data.get('metadata', {})
                    )
            
            logger.info(f"Graph imported from {file_path}")
            
        except Exception as e:
            logger.error(f"Error importing graph: {e}")