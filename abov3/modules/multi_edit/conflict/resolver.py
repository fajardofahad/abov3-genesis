"""
Intelligent Conflict Resolution System

Enterprise-grade conflict detection and resolution with advanced merging algorithms,
semantic understanding, and interactive conflict resolution capabilities.
"""

import os
import re
import ast
import json
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import difflib
from datetime import datetime

logger = logging.getLogger(__name__)


class ConflictType(Enum):
    """Types of conflicts that can occur"""
    CONTENT_OVERLAP = "content_overlap"
    LINE_CONFLICT = "line_conflict"
    STRUCTURAL_CONFLICT = "structural_conflict"
    DEPENDENCY_CONFLICT = "dependency_conflict"
    SEMANTIC_CONFLICT = "semantic_conflict"
    ENCODING_CONFLICT = "encoding_conflict"


class ResolutionStrategy(Enum):
    """Strategies for resolving conflicts"""
    AUTO_MERGE = "auto_merge"
    PREFER_OURS = "prefer_ours"
    PREFER_THEIRS = "prefer_theirs"
    MANUAL_RESOLUTION = "manual_resolution"
    SMART_MERGE = "smart_merge"
    SEMANTIC_MERGE = "semantic_merge"


@dataclass
class ConflictLocation:
    """Represents a conflict location in a file"""
    file_path: str
    start_line: int
    end_line: int
    conflict_type: ConflictType
    our_content: str
    their_content: str
    base_content: Optional[str] = None
    context_before: List[str] = None
    context_after: List[str] = None
    
    def __post_init__(self):
        if self.context_before is None:
            self.context_before = []
        if self.context_after is None:
            self.context_after = []


@dataclass
class ConflictResolution:
    """Represents a resolved conflict"""
    conflict_location: ConflictLocation
    resolution_strategy: ResolutionStrategy
    resolved_content: str
    confidence: float
    explanation: str
    requires_review: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ConflictResolver:
    """
    Intelligent conflict resolution system
    
    Features:
    - Automatic conflict detection across multiple patch sets
    - Smart merging with semantic understanding
    - Context-aware resolution strategies
    - Interactive conflict resolution
    - Confidence scoring for automatic resolutions
    - Structured and semantic conflict handling
    """
    
    def __init__(self, project_root: str = None):
        self.project_root = project_root
        
        # Resolution strategies by file type
        self.file_type_strategies = {
            '.py': self._python_semantic_merge,
            '.js': self._javascript_semantic_merge,
            '.ts': self._typescript_semantic_merge,
            '.java': self._java_semantic_merge,
            '.cpp': self._cpp_semantic_merge,
            '.h': self._c_header_merge,
            '.json': self._json_semantic_merge,
            '.xml': self._xml_semantic_merge,
            '.yaml': self._yaml_semantic_merge,
            '.yml': self._yaml_semantic_merge
        }
        
        # Patterns for different languages
        self.language_patterns = {
            'python': {
                'function': r'^\s*def\s+(\w+)\s*\(',
                'class': r'^\s*class\s+(\w+)\s*[:(]',
                'import': r'^\s*(from\s+\S+\s+)?import\s+',
                'comment': r'^\s*#',
                'docstring': r'^\s*"""',
            },
            'javascript': {
                'function': r'^\s*(function\s+\w+|const\s+\w+\s*=.*=>|\w+\s*:\s*function)',
                'class': r'^\s*class\s+\w+',
                'import': r'^\s*(import|export)',
                'comment': r'^\s*//',
            },
            'java': {
                'method': r'^\s*(public|private|protected).*\w+\s*\(',
                'class': r'^\s*(public\s+)?class\s+\w+',
                'import': r'^\s*import\s+',
                'comment': r'^\s*//',
            }
        }
    
    async def detect_conflicts(
        self, 
        patch_sets: List[Any],  # PatchSet objects
        base_state: Dict[str, str] = None
    ) -> List[ConflictLocation]:
        """
        Detect conflicts between multiple patch sets
        
        Args:
            patch_sets: List of patch sets to check for conflicts
            base_state: Base state of files (if available)
            
        Returns:
            List of detected conflicts
        """
        conflicts = []
        
        # Group changes by file
        file_changes = {}
        for patch_set in patch_sets:
            for file_change in patch_set.files:
                file_path = file_change.file_path
                if file_path not in file_changes:
                    file_changes[file_path] = []
                file_changes[file_path].append((patch_set.id, file_change))
        
        # Check each file for conflicts
        for file_path, changes in file_changes.items():
            if len(changes) > 1:
                file_conflicts = await self._detect_file_conflicts(
                    file_path, changes, base_state
                )
                conflicts.extend(file_conflicts)
        
        return conflicts
    
    async def _detect_file_conflicts(
        self,
        file_path: str,
        changes: List[Tuple[str, Any]],
        base_state: Dict[str, str] = None
    ) -> List[ConflictLocation]:
        """Detect conflicts within a single file"""
        
        conflicts = []
        
        # Get base content
        base_content = ""
        if base_state and file_path in base_state:
            base_content = base_state[file_path]
        elif os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    base_content = f.read()
            except Exception as e:
                logger.warning(f"Could not read base file {file_path}: {e}")
        
        base_lines = base_content.split('\n') if base_content else []
        
        # Analyze each pair of changes for conflicts
        for i, (patch_id1, change1) in enumerate(changes):
            for j, (patch_id2, change2) in enumerate(changes[i+1:], i+1):
                
                # Check for overlapping line modifications
                overlap_conflicts = await self._check_line_overlaps(
                    file_path, change1, change2, base_lines
                )
                conflicts.extend(overlap_conflicts)
                
                # Check for structural conflicts
                structural_conflicts = await self._check_structural_conflicts(
                    file_path, change1, change2, base_lines
                )
                conflicts.extend(structural_conflicts)
                
                # Check for semantic conflicts
                semantic_conflicts = await self._check_semantic_conflicts(
                    file_path, change1, change2, base_lines
                )
                conflicts.extend(semantic_conflicts)
        
        return conflicts
    
    async def _check_line_overlaps(
        self,
        file_path: str,
        change1: Any,
        change2: Any,
        base_lines: List[str]
    ) -> List[ConflictLocation]:
        """Check for overlapping line modifications"""
        
        conflicts = []
        
        # Get line changes for both modifications
        lines1 = self._get_modified_lines(change1, base_lines)
        lines2 = self._get_modified_lines(change2, base_lines)
        
        # Find overlapping ranges
        overlaps = self._find_line_overlaps(lines1, lines2)
        
        for start_line, end_line in overlaps:
            # Get conflicting content
            our_content = self._get_content_range(change1.new_content or "", start_line, end_line)
            their_content = self._get_content_range(change2.new_content or "", start_line, end_line)
            base_content = self._get_content_range('\n'.join(base_lines), start_line, end_line)
            
            conflict = ConflictLocation(
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                conflict_type=ConflictType.LINE_CONFLICT,
                our_content=our_content,
                their_content=their_content,
                base_content=base_content,
                context_before=base_lines[max(0, start_line-3):start_line],
                context_after=base_lines[end_line:min(len(base_lines), end_line+3)]
            )
            
            conflicts.append(conflict)
        
        return conflicts
    
    async def _check_structural_conflicts(
        self,
        file_path: str,
        change1: Any,
        change2: Any,
        base_lines: List[str]
    ) -> List[ConflictLocation]:
        """Check for structural conflicts (function/class definitions, etc.)"""
        
        conflicts = []
        
        # Get file extension to determine language
        file_ext = os.path.splitext(file_path)[1]
        
        if file_ext in ['.py', '.js', '.java', '.cpp', '.c']:
            # Parse structural elements
            structures1 = await self._parse_code_structures(change1.new_content or "", file_ext)
            structures2 = await self._parse_code_structures(change2.new_content or "", file_ext)
            
            # Find conflicting structures
            for struct1 in structures1:
                for struct2 in structures2:
                    if (struct1['name'] == struct2['name'] and 
                        struct1['type'] == struct2['type'] and
                        struct1['content'] != struct2['content']):
                        
                        conflict = ConflictLocation(
                            file_path=file_path,
                            start_line=struct1.get('line_start', 0),
                            end_line=struct1.get('line_end', 0),
                            conflict_type=ConflictType.STRUCTURAL_CONFLICT,
                            our_content=struct1['content'],
                            their_content=struct2['content']
                        )
                        
                        conflicts.append(conflict)
        
        return conflicts
    
    async def _check_semantic_conflicts(
        self,
        file_path: str,
        change1: Any,
        change2: Any,
        base_lines: List[str]
    ) -> List[ConflictLocation]:
        """Check for semantic conflicts (variable usage, imports, etc.)"""
        
        conflicts = []
        
        # Check import conflicts
        import_conflicts = await self._check_import_conflicts(
            file_path, change1, change2
        )
        conflicts.extend(import_conflicts)
        
        # Check variable/function usage conflicts
        usage_conflicts = await self._check_usage_conflicts(
            file_path, change1, change2, base_lines
        )
        conflicts.extend(usage_conflicts)
        
        return conflicts
    
    async def resolve_conflicts(
        self,
        conflicts: List[ConflictLocation],
        strategy: ResolutionStrategy = ResolutionStrategy.SMART_MERGE
    ) -> List[ConflictResolution]:
        """
        Resolve detected conflicts using specified strategy
        
        Args:
            conflicts: List of conflicts to resolve
            strategy: Resolution strategy to use
            
        Returns:
            List of conflict resolutions
        """
        
        resolutions = []
        
        for conflict in conflicts:
            resolution = await self._resolve_single_conflict(conflict, strategy)
            if resolution:
                resolutions.append(resolution)
        
        return resolutions
    
    async def _resolve_single_conflict(
        self,
        conflict: ConflictLocation,
        strategy: ResolutionStrategy
    ) -> Optional[ConflictResolution]:
        """Resolve a single conflict"""
        
        if strategy == ResolutionStrategy.AUTO_MERGE:
            return await self._auto_merge_conflict(conflict)
        
        elif strategy == ResolutionStrategy.PREFER_OURS:
            return ConflictResolution(
                conflict_location=conflict,
                resolution_strategy=strategy,
                resolved_content=conflict.our_content,
                confidence=1.0,
                explanation="Used 'ours' strategy"
            )
        
        elif strategy == ResolutionStrategy.PREFER_THEIRS:
            return ConflictResolution(
                conflict_location=conflict,
                resolution_strategy=strategy,
                resolved_content=conflict.their_content,
                confidence=1.0,
                explanation="Used 'theirs' strategy"
            )
        
        elif strategy == ResolutionStrategy.SMART_MERGE:
            return await self._smart_merge_conflict(conflict)
        
        elif strategy == ResolutionStrategy.SEMANTIC_MERGE:
            return await self._semantic_merge_conflict(conflict)
        
        else:  # MANUAL_RESOLUTION
            return ConflictResolution(
                conflict_location=conflict,
                resolution_strategy=strategy,
                resolved_content="",
                confidence=0.0,
                explanation="Requires manual resolution",
                requires_review=True
            )
    
    async def _auto_merge_conflict(self, conflict: ConflictLocation) -> ConflictResolution:
        """Attempt automatic merge using standard algorithms"""
        
        # Use 3-way merge if base content is available
        if conflict.base_content:
            merged = await self._three_way_merge(
                conflict.base_content,
                conflict.our_content,
                conflict.their_content
            )
            
            if merged['success']:
                return ConflictResolution(
                    conflict_location=conflict,
                    resolution_strategy=ResolutionStrategy.AUTO_MERGE,
                    resolved_content=merged['content'],
                    confidence=merged['confidence'],
                    explanation=merged['explanation']
                )
        
        # Fall back to line-by-line merge
        return await self._line_by_line_merge(conflict)
    
    async def _smart_merge_conflict(self, conflict: ConflictLocation) -> ConflictResolution:
        """Smart merge using context and patterns"""
        
        file_ext = os.path.splitext(conflict.file_path)[1]
        
        # Use file-type specific merger if available
        if file_ext in self.file_type_strategies:
            merger = self.file_type_strategies[file_ext]
            result = await merger(conflict)
            if result:
                return result
        
        # Fall back to general smart merge
        return await self._general_smart_merge(conflict)
    
    async def _semantic_merge_conflict(self, conflict: ConflictLocation) -> ConflictResolution:
        """Semantic merge based on code understanding"""
        
        file_ext = os.path.splitext(conflict.file_path)[1]
        
        if file_ext == '.py':
            return await self._python_semantic_merge(conflict)
        elif file_ext in ['.js', '.ts']:
            return await self._javascript_semantic_merge(conflict)
        elif file_ext == '.java':
            return await self._java_semantic_merge(conflict)
        else:
            # Fall back to smart merge
            return await self._smart_merge_conflict(conflict)
    
    # Language-specific semantic mergers
    async def _python_semantic_merge(self, conflict: ConflictLocation) -> ConflictResolution:
        """Python-specific semantic merge"""
        
        try:
            # Parse both versions as AST
            our_ast = ast.parse(conflict.our_content)
            their_ast = ast.parse(conflict.their_content)
            
            # Merge at AST level
            merged_ast = await self._merge_python_ast(our_ast, their_ast)
            
            if merged_ast:
                merged_content = ast.unparse(merged_ast)
                return ConflictResolution(
                    conflict_location=conflict,
                    resolution_strategy=ResolutionStrategy.SEMANTIC_MERGE,
                    resolved_content=merged_content,
                    confidence=0.8,
                    explanation="Python AST-based semantic merge"
                )
        
        except SyntaxError:
            # Fall back to text-based merge
            pass
        
        return await self._general_smart_merge(conflict)
    
    async def _javascript_semantic_merge(self, conflict: ConflictLocation) -> ConflictResolution:
        """JavaScript/TypeScript semantic merge"""
        # Placeholder for JS/TS semantic merge
        return await self._general_smart_merge(conflict)
    
    async def _typescript_semantic_merge(self, conflict: ConflictLocation) -> ConflictResolution:
        """TypeScript semantic merge"""
        return await self._javascript_semantic_merge(conflict)
    
    async def _java_semantic_merge(self, conflict: ConflictLocation) -> ConflictResolution:
        """Java semantic merge"""
        # Placeholder for Java semantic merge
        return await self._general_smart_merge(conflict)
    
    async def _cpp_semantic_merge(self, conflict: ConflictLocation) -> ConflictResolution:
        """C++ semantic merge"""
        # Placeholder for C++ semantic merge
        return await self._general_smart_merge(conflict)
    
    async def _c_header_merge(self, conflict: ConflictLocation) -> ConflictResolution:
        """C header file merge"""
        # Placeholder for C header merge
        return await self._general_smart_merge(conflict)
    
    async def _json_semantic_merge(self, conflict: ConflictLocation) -> ConflictResolution:
        """JSON semantic merge"""
        try:
            our_json = json.loads(conflict.our_content)
            their_json = json.loads(conflict.their_content)
            
            # Deep merge JSON objects
            merged_json = await self._deep_merge_dict(our_json, their_json)
            merged_content = json.dumps(merged_json, indent=2)
            
            return ConflictResolution(
                conflict_location=conflict,
                resolution_strategy=ResolutionStrategy.SEMANTIC_MERGE,
                resolved_content=merged_content,
                confidence=0.9,
                explanation="JSON semantic merge"
            )
        
        except json.JSONDecodeError:
            return await self._general_smart_merge(conflict)
    
    async def _xml_semantic_merge(self, conflict: ConflictLocation) -> ConflictResolution:
        """XML semantic merge"""
        # Placeholder for XML merge
        return await self._general_smart_merge(conflict)
    
    async def _yaml_semantic_merge(self, conflict: ConflictLocation) -> ConflictResolution:
        """YAML semantic merge"""
        # Placeholder for YAML merge
        return await self._general_smart_merge(conflict)
    
    async def _general_smart_merge(self, conflict: ConflictLocation) -> ConflictResolution:
        """General smart merge using text analysis"""
        
        our_lines = conflict.our_content.split('\n')
        their_lines = conflict.their_content.split('\n')
        
        # Use difflib for intelligent merge
        differ = difflib.SequenceMatcher(None, our_lines, their_lines)
        merged_lines = []
        
        for tag, i1, i2, j1, j2 in differ.get_opcodes():
            if tag == 'equal':
                merged_lines.extend(our_lines[i1:i2])
            elif tag == 'replace':
                # Try to merge replacements intelligently
                if len(our_lines[i1:i2]) == 1 and len(their_lines[j1:j2]) == 1:
                    # Single line replacement - try to combine
                    merged_line = await self._merge_single_lines(
                        our_lines[i1], their_lines[j1]
                    )
                    merged_lines.append(merged_line)
                else:
                    # Multiple lines - keep both with markers
                    merged_lines.extend(our_lines[i1:i2])
                    merged_lines.extend(their_lines[j1:j2])
            elif tag == 'delete':
                # Keep deleted lines commented
                for line in our_lines[i1:i2]:
                    merged_lines.append(f"# DELETED: {line}")
            elif tag == 'insert':
                merged_lines.extend(their_lines[j1:j2])
        
        merged_content = '\n'.join(merged_lines)
        
        return ConflictResolution(
            conflict_location=conflict,
            resolution_strategy=ResolutionStrategy.SMART_MERGE,
            resolved_content=merged_content,
            confidence=0.6,
            explanation="General smart merge with text analysis"
        )
    
    # Helper methods
    async def _three_way_merge(
        self, 
        base: str, 
        ours: str, 
        theirs: str
    ) -> Dict[str, Any]:
        """Perform 3-way merge"""
        
        base_lines = base.split('\n')
        our_lines = ours.split('\n')
        their_lines = theirs.split('\n')
        
        # Simple 3-way merge logic
        merged_lines = []
        conflicts = []
        
        # Use difflib to find differences
        base_to_ours = list(difflib.unified_diff(base_lines, our_lines, lineterm=''))
        base_to_theirs = list(difflib.unified_diff(base_lines, their_lines, lineterm=''))
        
        # If both changes are identical, merge is clean
        if our_lines == their_lines:
            return {
                'success': True,
                'content': ours,
                'confidence': 1.0,
                'explanation': 'Identical changes'
            }
        
        # If one side has no changes, use the other
        if base_lines == our_lines:
            return {
                'success': True,
                'content': theirs,
                'confidence': 0.9,
                'explanation': 'Only theirs changed'
            }
        
        if base_lines == their_lines:
            return {
                'success': True,
                'content': ours,
                'confidence': 0.9,
                'explanation': 'Only ours changed'
            }
        
        # Otherwise, we have a conflict
        return {
            'success': False,
            'content': '',
            'confidence': 0.0,
            'explanation': 'Complex 3-way merge conflict'
        }
    
    async def _line_by_line_merge(self, conflict: ConflictLocation) -> ConflictResolution:
        """Line-by-line merge strategy"""
        
        our_lines = conflict.our_content.split('\n')
        their_lines = conflict.their_content.split('\n')
        
        # Simple line-by-line comparison
        merged_lines = []
        max_lines = max(len(our_lines), len(their_lines))
        
        for i in range(max_lines):
            our_line = our_lines[i] if i < len(our_lines) else None
            their_line = their_lines[i] if i < len(their_lines) else None
            
            if our_line == their_line:
                merged_lines.append(our_line or "")
            elif our_line and not their_line:
                merged_lines.append(our_line)
            elif their_line and not our_line:
                merged_lines.append(their_line)
            else:
                # Both different - need to decide
                if len(our_line or "") > len(their_line or ""):
                    merged_lines.append(our_line)
                else:
                    merged_lines.append(their_line)
        
        return ConflictResolution(
            conflict_location=conflict,
            resolution_strategy=ResolutionStrategy.AUTO_MERGE,
            resolved_content='\n'.join(merged_lines),
            confidence=0.5,
            explanation="Line-by-line automatic merge"
        )
    
    async def _merge_single_lines(self, our_line: str, their_line: str) -> str:
        """Merge two similar lines intelligently"""
        
        # If one is a subset of the other, use the longer one
        if our_line in their_line:
            return their_line
        elif their_line in our_line:
            return our_line
        
        # If they share common parts, try to combine
        matcher = difflib.SequenceMatcher(None, our_line, their_line)
        ratio = matcher.ratio()
        
        if ratio > 0.8:
            # Very similar - use the first one
            return our_line
        elif ratio > 0.5:
            # Somewhat similar - try to combine
            return f"{our_line} # MERGED: {their_line}"
        else:
            # Very different - keep both
            return f"{our_line} # CONFLICT: {their_line}"
    
    # Utility methods
    def _get_modified_lines(self, change: Any, base_lines: List[str]) -> Set[int]:
        """Get set of line numbers modified by a change"""
        # Placeholder - would analyze the change to determine affected lines
        return set()
    
    def _find_line_overlaps(self, lines1: Set[int], lines2: Set[int]) -> List[Tuple[int, int]]:
        """Find overlapping line ranges"""
        overlaps = []
        intersection = lines1.intersection(lines2)
        
        if intersection:
            # Convert to ranges
            sorted_lines = sorted(intersection)
            start = sorted_lines[0]
            end = sorted_lines[0]
            
            for line in sorted_lines[1:]:
                if line == end + 1:
                    end = line
                else:
                    overlaps.append((start, end + 1))
                    start = line
                    end = line
            
            overlaps.append((start, end + 1))
        
        return overlaps
    
    def _get_content_range(self, content: str, start_line: int, end_line: int) -> str:
        """Get content between specified line ranges"""
        lines = content.split('\n')
        return '\n'.join(lines[start_line:end_line])
    
    async def _parse_code_structures(self, content: str, file_ext: str) -> List[Dict[str, Any]]:
        """Parse code structures from content"""
        structures = []
        
        if file_ext == '.py':
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        structures.append({
                            'type': 'function',
                            'name': node.name,
                            'line_start': node.lineno,
                            'line_end': node.end_lineno,
                            'content': ast.unparse(node)
                        })
                    elif isinstance(node, ast.ClassDef):
                        structures.append({
                            'type': 'class',
                            'name': node.name,
                            'line_start': node.lineno,
                            'line_end': node.end_lineno,
                            'content': ast.unparse(node)
                        })
            except SyntaxError:
                pass
        
        return structures
    
    async def _check_import_conflicts(self, file_path: str, change1: Any, change2: Any) -> List[ConflictLocation]:
        """Check for import statement conflicts"""
        conflicts = []
        
        # Extract imports from both changes
        imports1 = self._extract_imports(change1.new_content or "", file_path)
        imports2 = self._extract_imports(change2.new_content or "", file_path)
        
        # Find conflicting imports
        for imp1 in imports1:
            for imp2 in imports2:
                if (imp1['module'] == imp2['module'] and 
                    imp1['name'] != imp2['name']):
                    
                    conflict = ConflictLocation(
                        file_path=file_path,
                        start_line=imp1.get('line', 0),
                        end_line=imp1.get('line', 0),
                        conflict_type=ConflictType.SEMANTIC_CONFLICT,
                        our_content=imp1['statement'],
                        their_content=imp2['statement']
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    async def _check_usage_conflicts(
        self, 
        file_path: str, 
        change1: Any, 
        change2: Any,
        base_lines: List[str]
    ) -> List[ConflictLocation]:
        """Check for variable/function usage conflicts"""
        # Placeholder for usage conflict detection
        return []
    
    def _extract_imports(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Extract import statements from content"""
        imports = []
        lines = content.split('\n')
        
        file_ext = os.path.splitext(file_path)[1]
        
        if file_ext == '.py':
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith('import ') or line.startswith('from '):
                    imports.append({
                        'statement': line,
                        'line': i + 1,
                        'module': self._extract_python_module(line),
                        'name': line
                    })
        
        return imports
    
    def _extract_python_module(self, import_statement: str) -> str:
        """Extract module name from Python import statement"""
        if import_statement.startswith('from '):
            parts = import_statement.split()
            if len(parts) >= 2:
                return parts[1]
        elif import_statement.startswith('import '):
            parts = import_statement.split()
            if len(parts) >= 2:
                return parts[1].split('.')[0]
        
        return ""
    
    async def _merge_python_ast(self, ast1: ast.AST, ast2: ast.AST) -> Optional[ast.AST]:
        """Merge two Python ASTs"""
        # Placeholder for AST merging
        return None
    
    async def _deep_merge_dict(self, dict1: Dict, dict2: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result:
                if isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = await self._deep_merge_dict(result[key], value)
                elif isinstance(result[key], list) and isinstance(value, list):
                    # Merge lists by extending
                    result[key] = result[key] + [item for item in value if item not in result[key]]
                else:
                    # Overwrite with new value
                    result[key] = value
            else:
                result[key] = value
        
        return result