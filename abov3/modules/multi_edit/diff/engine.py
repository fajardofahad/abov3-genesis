"""
Advanced Diff Engine

Enterprise-grade diff generation with context-aware analysis, intelligent line matching,
and comprehensive change visualization for multi-file patch sets.
"""

import re
import difflib
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Types of changes in a diff"""
    ADDED = "added"
    REMOVED = "removed" 
    MODIFIED = "modified"
    MOVED = "moved"
    UNCHANGED = "unchanged"


@dataclass
class DiffLine:
    """Represents a single line in a diff"""
    line_number_old: Optional[int]
    line_number_new: Optional[int]
    content: str
    change_type: ChangeType
    context_before: List[str] = None
    context_after: List[str] = None
    semantic_context: str = None  # Function, class, etc.
    similarity_score: float = 0.0  # For moved/modified lines
    
    def __post_init__(self):
        if self.context_before is None:
            self.context_before = []
        if self.context_after is None:
            self.context_after = []


@dataclass
class FileDiff:
    """Represents diff for a single file"""
    file_path: str
    old_file_path: Optional[str] = None  # For renames
    change_type: str = "modify"  # create, modify, delete, rename
    lines: List[DiffLine] = None
    hunks: List['DiffHunk'] = None
    statistics: Dict[str, int] = None
    encoding: str = "utf-8"
    
    def __post_init__(self):
        if self.lines is None:
            self.lines = []
        if self.hunks is None:
            self.hunks = []
        if self.statistics is None:
            self.statistics = {
                'lines_added': 0,
                'lines_removed': 0,
                'lines_modified': 0,
                'lines_unchanged': 0
            }


@dataclass 
class DiffHunk:
    """Represents a continuous block of changes"""
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: List[DiffLine]
    context: str = ""  # Semantic context (function name, etc.)


class DiffEngine:
    """
    Advanced diff engine with enterprise features
    
    Features:
    - Context-aware line analysis
    - Intelligent similarity detection
    - Semantic understanding of code structure
    - Side-by-side and unified diff formats
    - Statistical analysis of changes
    - Smart line matching algorithms
    """
    
    def __init__(self, context_lines: int = 3):
        self.context_lines = context_lines
        self.semantic_patterns = {
            'function': [
                r'^\s*def\s+(\w+)',      # Python
                r'^\s*function\s+(\w+)', # JavaScript
                r'^\s*(\w+)\s*\([^)]*\)\s*\{',  # C/C++/Java
                r'^\s*public\s+\w+\s+(\w+)\s*\(',  # Java methods
            ],
            'class': [
                r'^\s*class\s+(\w+)',    # Python/C++
                r'^\s*public\s+class\s+(\w+)',  # Java
            ],
            'interface': [
                r'^\s*interface\s+(\w+)', # TypeScript/Java
            ],
            'struct': [
                r'^\s*struct\s+(\w+)',   # C/C++
            ]
        }
    
    async def generate_file_diff(
        self, 
        old_content: str, 
        new_content: str, 
        file_path: str,
        old_file_path: str = None
    ) -> FileDiff:
        """
        Generate comprehensive diff for a single file
        
        Args:
            old_content: Original file content
            new_content: New file content
            file_path: Path to the file
            old_file_path: Original path (for renames)
            
        Returns:
            FileDiff object with detailed analysis
        """
        
        # Determine change type
        change_type = self._determine_change_type(old_content, new_content, old_file_path)
        
        # Create FileDiff object
        file_diff = FileDiff(
            file_path=file_path,
            old_file_path=old_file_path,
            change_type=change_type,
            encoding='utf-8'
        )
        
        if change_type == 'create':
            file_diff.lines = await self._generate_create_diff(new_content)
        elif change_type == 'delete':
            file_diff.lines = await self._generate_delete_diff(old_content)
        else:
            file_diff.lines = await self._generate_modify_diff(old_content, new_content)
        
        # Generate hunks from lines
        file_diff.hunks = await self._generate_hunks(file_diff.lines)
        
        # Calculate statistics
        file_diff.statistics = self._calculate_statistics(file_diff.lines)
        
        # Add semantic context
        await self._add_semantic_context(file_diff, old_content, new_content)
        
        return file_diff
    
    def _determine_change_type(self, old_content: str, new_content: str, old_file_path: str) -> str:
        """Determine the type of change"""
        if old_content is None or old_content == '':
            return 'create'
        elif new_content is None or new_content == '':
            return 'delete'
        elif old_file_path:
            return 'rename'
        else:
            return 'modify'
    
    async def _generate_create_diff(self, content: str) -> List[DiffLine]:
        """Generate diff lines for file creation"""
        lines = content.split('\n') if content else ['']
        diff_lines = []
        
        for i, line in enumerate(lines):
            diff_lines.append(DiffLine(
                line_number_old=None,
                line_number_new=i + 1,
                content=line,
                change_type=ChangeType.ADDED
            ))
        
        return diff_lines
    
    async def _generate_delete_diff(self, content: str) -> List[DiffLine]:
        """Generate diff lines for file deletion"""
        lines = content.split('\n') if content else ['']
        diff_lines = []
        
        for i, line in enumerate(lines):
            diff_lines.append(DiffLine(
                line_number_old=i + 1,
                line_number_new=None,
                content=line,
                change_type=ChangeType.REMOVED
            ))
        
        return diff_lines
    
    async def _generate_modify_diff(self, old_content: str, new_content: str) -> List[DiffLine]:
        """Generate diff lines for file modification"""
        old_lines = old_content.split('\n') if old_content else ['']
        new_lines = new_content.split('\n') if new_content else ['']
        
        # Use difflib for initial diff
        differ = difflib.SequenceMatcher(None, old_lines, new_lines)
        diff_lines = []
        
        old_line_num = 1
        new_line_num = 1
        
        for tag, i1, i2, j1, j2 in differ.get_opcodes():
            if tag == 'equal':
                # Unchanged lines
                for k in range(i1, i2):
                    diff_lines.append(DiffLine(
                        line_number_old=old_line_num,
                        line_number_new=new_line_num,
                        content=old_lines[k],
                        change_type=ChangeType.UNCHANGED
                    ))
                    old_line_num += 1
                    new_line_num += 1
            
            elif tag == 'delete':
                # Removed lines
                for k in range(i1, i2):
                    diff_lines.append(DiffLine(
                        line_number_old=old_line_num,
                        line_number_new=None,
                        content=old_lines[k],
                        change_type=ChangeType.REMOVED
                    ))
                    old_line_num += 1
            
            elif tag == 'insert':
                # Added lines
                for k in range(j1, j2):
                    diff_lines.append(DiffLine(
                        line_number_old=None,
                        line_number_new=new_line_num,
                        content=new_lines[k],
                        change_type=ChangeType.ADDED
                    ))
                    new_line_num += 1
            
            elif tag == 'replace':
                # Modified lines - try to match similar lines
                old_section = old_lines[i1:i2]
                new_section = new_lines[j1:j2]
                
                matched_pairs = await self._find_similar_lines(old_section, new_section)
                
                # Process matched pairs
                for old_idx, new_idx, similarity in matched_pairs:
                    if similarity > 0.5:  # Consider it a modification
                        diff_lines.append(DiffLine(
                            line_number_old=old_line_num + old_idx,
                            line_number_new=new_line_num + new_idx,
                            content=new_section[new_idx],
                            change_type=ChangeType.MODIFIED,
                            similarity_score=similarity
                        ))
                    else:
                        # Too different, treat as separate add/remove
                        diff_lines.append(DiffLine(
                            line_number_old=old_line_num + old_idx,
                            line_number_new=None,
                            content=old_section[old_idx],
                            change_type=ChangeType.REMOVED
                        ))
                        diff_lines.append(DiffLine(
                            line_number_old=None,
                            line_number_new=new_line_num + new_idx,
                            content=new_section[new_idx],
                            change_type=ChangeType.ADDED
                        ))
                
                # Handle unmatched lines
                matched_old = {pair[0] for pair in matched_pairs}
                matched_new = {pair[1] for pair in matched_pairs}
                
                for i, line in enumerate(old_section):
                    if i not in matched_old:
                        diff_lines.append(DiffLine(
                            line_number_old=old_line_num + i,
                            line_number_new=None,
                            content=line,
                            change_type=ChangeType.REMOVED
                        ))
                
                for i, line in enumerate(new_section):
                    if i not in matched_new:
                        diff_lines.append(DiffLine(
                            line_number_old=None,
                            line_number_new=new_line_num + i,
                            content=line,
                            change_type=ChangeType.ADDED
                        ))
                
                old_line_num += len(old_section)
                new_line_num += len(new_section)
        
        return diff_lines
    
    async def _find_similar_lines(self, old_lines: List[str], new_lines: List[str]) -> List[Tuple[int, int, float]]:
        """Find pairs of similar lines between old and new sections"""
        pairs = []
        
        for i, old_line in enumerate(old_lines):
            best_match = None
            best_similarity = 0.0
            
            for j, new_line in enumerate(new_lines):
                similarity = self._calculate_line_similarity(old_line, new_line)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = j
            
            if best_match is not None and best_similarity > 0.3:
                pairs.append((i, best_match, best_similarity))
        
        # Remove duplicate matches (keep highest similarity)
        unique_pairs = {}
        for old_idx, new_idx, similarity in pairs:
            key = (old_idx, new_idx)
            if key not in unique_pairs or similarity > unique_pairs[key][2]:
                unique_pairs[key] = (old_idx, new_idx, similarity)
        
        return list(unique_pairs.values())
    
    def _calculate_line_similarity(self, line1: str, line2: str) -> float:
        """Calculate similarity between two lines"""
        if line1 == line2:
            return 1.0
        
        if not line1.strip() and not line2.strip():
            return 1.0
        
        # Use sequence matcher for similarity
        matcher = difflib.SequenceMatcher(None, line1, line2)
        return matcher.ratio()
    
    async def _generate_hunks(self, diff_lines: List[DiffLine]) -> List[DiffHunk]:
        """Generate hunks from diff lines"""
        hunks = []
        current_hunk_lines = []
        hunk_start_old = None
        hunk_start_new = None
        
        i = 0
        while i < len(diff_lines):
            line = diff_lines[i]
            
            # Start new hunk if we hit a change
            if line.change_type != ChangeType.UNCHANGED:
                if not current_hunk_lines:
                    # Add context before
                    context_start = max(0, i - self.context_lines)
                    for j in range(context_start, i):
                        context_line = diff_lines[j]
                        if context_line.change_type == ChangeType.UNCHANGED:
                            current_hunk_lines.append(context_line)
                    
                    hunk_start_old = diff_lines[context_start].line_number_old
                    hunk_start_new = diff_lines[context_start].line_number_new
                
                current_hunk_lines.append(line)
            
            else:  # Unchanged line
                if current_hunk_lines:
                    # Add to current hunk as context
                    current_hunk_lines.append(line)
                    
                    # Check if we should end the hunk
                    unchanged_count = 0
                    j = i
                    while (j < len(diff_lines) and 
                           j < i + self.context_lines * 2 and
                           diff_lines[j].change_type == ChangeType.UNCHANGED):
                        unchanged_count += 1
                        j += 1
                    
                    # End hunk if we have enough unchanged context
                    if unchanged_count >= self.context_lines * 2 or i == len(diff_lines) - 1:
                        # Add context after
                        context_end = min(len(diff_lines), i + self.context_lines)
                        for k in range(i + 1, context_end):
                            current_hunk_lines.append(diff_lines[k])
                        
                        # Create hunk
                        hunk = await self._create_hunk(current_hunk_lines, hunk_start_old, hunk_start_new)
                        hunks.append(hunk)
                        
                        # Reset for next hunk
                        current_hunk_lines = []
                        hunk_start_old = None
                        hunk_start_new = None
                        
                        # Skip context lines we already processed
                        i = context_end - 1
            
            i += 1
        
        # Handle final hunk if exists
        if current_hunk_lines:
            hunk = await self._create_hunk(current_hunk_lines, hunk_start_old, hunk_start_new)
            hunks.append(hunk)
        
        return hunks
    
    async def _create_hunk(self, lines: List[DiffLine], start_old: int, start_new: int) -> DiffHunk:
        """Create a hunk from lines"""
        if not lines:
            return DiffHunk(0, 0, 0, 0, [])
        
        old_count = sum(1 for line in lines if line.line_number_old is not None)
        new_count = sum(1 for line in lines if line.line_number_new is not None)
        
        # Find semantic context
        context = await self._find_hunk_context(lines)
        
        return DiffHunk(
            old_start=start_old or 1,
            old_count=old_count,
            new_start=start_new or 1,
            new_count=new_count,
            lines=lines,
            context=context
        )
    
    async def _find_hunk_context(self, lines: List[DiffLine]) -> str:
        """Find semantic context for a hunk (function name, class, etc.)"""
        contexts = []
        
        for line in lines:
            if line.change_type == ChangeType.UNCHANGED:
                content = line.content.strip()
                
                # Check for semantic patterns
                for context_type, patterns in self.semantic_patterns.items():
                    for pattern in patterns:
                        match = re.match(pattern, content)
                        if match:
                            name = match.group(1) if match.groups() else context_type
                            contexts.append(f"{context_type} {name}")
                            break
        
        return contexts[0] if contexts else ""
    
    def _calculate_statistics(self, diff_lines: List[DiffLine]) -> Dict[str, int]:
        """Calculate statistics from diff lines"""
        stats = {
            'lines_added': 0,
            'lines_removed': 0,
            'lines_modified': 0,
            'lines_unchanged': 0
        }
        
        for line in diff_lines:
            if line.change_type == ChangeType.ADDED:
                stats['lines_added'] += 1
            elif line.change_type == ChangeType.REMOVED:
                stats['lines_removed'] += 1
            elif line.change_type == ChangeType.MODIFIED:
                stats['lines_modified'] += 1
            elif line.change_type == ChangeType.UNCHANGED:
                stats['lines_unchanged'] += 1
        
        return stats
    
    async def _add_semantic_context(self, file_diff: FileDiff, old_content: str, new_content: str):
        """Add semantic context to diff lines"""
        old_lines = old_content.split('\n') if old_content else []
        new_lines = new_content.split('\n') if new_content else []
        
        # Find function/class boundaries in old content
        old_contexts = await self._find_semantic_boundaries(old_lines)
        new_contexts = await self._find_semantic_boundaries(new_lines)
        
        # Apply context to diff lines
        for line in file_diff.lines:
            if line.line_number_old:
                context = self._find_context_for_line(line.line_number_old - 1, old_contexts)
                if context:
                    line.semantic_context = context
            elif line.line_number_new:
                context = self._find_context_for_line(line.line_number_new - 1, new_contexts)
                if context:
                    line.semantic_context = context
    
    async def _find_semantic_boundaries(self, lines: List[str]) -> Dict[int, str]:
        """Find semantic boundaries (functions, classes) in code"""
        boundaries = {}
        current_context = None
        indent_stack = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            
            # Get indentation level
            indent = len(line) - len(line.lstrip())
            
            # Check for semantic patterns
            for context_type, patterns in self.semantic_patterns.items():
                for pattern in patterns:
                    match = re.match(pattern, stripped)
                    if match:
                        name = match.group(1) if match.groups() else context_type
                        context_name = f"{context_type} {name}"
                        
                        # Update indent stack
                        while indent_stack and indent_stack[-1][1] >= indent:
                            indent_stack.pop()
                        
                        indent_stack.append((context_name, indent))
                        current_context = " -> ".join(ctx[0] for ctx in indent_stack)
                        break
            
            if current_context:
                boundaries[i] = current_context
        
        return boundaries
    
    def _find_context_for_line(self, line_number: int, boundaries: Dict[int, str]) -> Optional[str]:
        """Find the semantic context for a specific line"""
        context = None
        for boundary_line, boundary_context in boundaries.items():
            if boundary_line <= line_number:
                context = boundary_context
            else:
                break
        return context
    
    async def format_unified_diff(self, file_diff: FileDiff) -> str:
        """Format diff as unified diff text"""
        lines = []
        
        # Header
        old_path = file_diff.old_file_path or file_diff.file_path
        lines.append(f"--- {old_path}")
        lines.append(f"+++ {file_diff.file_path}")
        
        # Hunks
        for hunk in file_diff.hunks:
            lines.append(f"@@ -{hunk.old_start},{hunk.old_count} +{hunk.new_start},{hunk.new_count} @@ {hunk.context}")
            
            for line in hunk.lines:
                if line.change_type == ChangeType.ADDED:
                    lines.append(f"+{line.content}")
                elif line.change_type == ChangeType.REMOVED:
                    lines.append(f"-{line.content}")
                elif line.change_type == ChangeType.MODIFIED:
                    lines.append(f"-{line.content}")  # Show old version
                    lines.append(f"+{line.content}")  # Show new version
                else:
                    lines.append(f" {line.content}")
        
        return "\n".join(lines)
    
    async def format_side_by_side(self, file_diff: FileDiff, width: int = 80) -> str:
        """Format diff as side-by-side comparison"""
        lines = []
        half_width = (width - 3) // 2  # Account for separator
        
        # Header
        lines.append(f"{'OLD':<{half_width}} | {'NEW':<{half_width}}")
        lines.append("-" * width)
        
        for diff_line in file_diff.lines:
            left = ""
            right = ""
            separator = "|"
            
            if diff_line.change_type == ChangeType.UNCHANGED:
                left = diff_line.content[:half_width]
                right = diff_line.content[:half_width]
                separator = "|"
            elif diff_line.change_type == ChangeType.REMOVED:
                left = diff_line.content[:half_width]
                right = ""
                separator = "<"
            elif diff_line.change_type == ChangeType.ADDED:
                left = ""
                right = diff_line.content[:half_width]
                separator = ">"
            elif diff_line.change_type == ChangeType.MODIFIED:
                left = f"(old) {diff_line.content[:half_width-6]}"
                right = diff_line.content[:half_width]
                separator = "~"
            
            lines.append(f"{left:<{half_width}} {separator} {right:<{half_width}}")
        
        return "\n".join(lines)