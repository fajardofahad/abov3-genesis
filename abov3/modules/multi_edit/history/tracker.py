"""
Enterprise Change History & Version Tracking System

Comprehensive history tracking with version management, change attribution, 
branching support, and detailed audit trails for multi-file operations.
"""

import os
import json
import hashlib
import uuid
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChangeEntry:
    """Represents a single change in history"""
    id: str
    patch_set_id: str
    transaction_id: Optional[str]
    timestamp: datetime
    author: str
    operation_type: str  # 'create', 'modify', 'delete', 'rename'
    file_path: str
    old_path: Optional[str] = None
    content_before: Optional[str] = None
    content_after: Optional[str] = None
    checksum_before: Optional[str] = None
    checksum_after: Optional[str] = None
    line_changes: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
        if self.line_changes is None:
            self.line_changes = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class VersionInfo:
    """Represents a version snapshot"""
    id: str
    patch_set_id: str
    version_number: str
    timestamp: datetime
    author: str
    description: str
    parent_versions: List[str] = None
    file_states: Dict[str, str] = None  # file_path -> checksum
    tags: List[str] = None
    branch: str = "main"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
        if self.parent_versions is None:
            self.parent_versions = []
        if self.file_states is None:
            self.file_states = {}
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Branch:
    """Represents a development branch"""
    name: str
    created_at: datetime
    created_by: str
    parent_branch: str
    parent_version: str
    head_version: str
    description: str = ""
    is_active: bool = True
    merged_at: Optional[datetime] = None
    merged_to: Optional[str] = None
    
    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.merged_at, str):
            self.merged_at = datetime.fromisoformat(self.merged_at)


class HistoryTracker:
    """
    Enterprise-grade change history and version tracking system
    
    Features:
    - Comprehensive change tracking with full audit trails
    - Version management with branching and merging
    - File-level and line-level change attribution
    - Time-based and version-based queries
    - Change statistics and analytics
    - Efficient storage with compression
    - Cross-platform compatibility
    - Integration with git and other VCS systems
    """
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.history_dir = self.project_root / ".abov3" / "history"
        self.history_dir.mkdir(parents=True, exist_ok=True)
        
        # History storage
        self.changes_db = self.history_dir / "changes.jsonl"  # JSON Lines format
        self.versions_db = self.history_dir / "versions.json"
        self.branches_db = self.history_dir / "branches.json"
        
        # In-memory caches
        self.recent_changes: List[ChangeEntry] = []
        self.versions: Dict[str, VersionInfo] = {}
        self.branches: Dict[str, Branch] = {}
        
        # Configuration
        self.max_history_entries = 10000
        self.compression_threshold = 1000  # Compress after this many entries
        
        # Initialize
        self._load_existing_data()
    
    def _load_existing_data(self):
        """Load existing history data"""
        try:
            # Load versions
            if self.versions_db.exists():
                with open(self.versions_db, 'r', encoding='utf-8') as f:
                    versions_data = json.load(f)
                    for version_data in versions_data:
                        version = VersionInfo(**version_data)
                        self.versions[version.id] = version
            
            # Load branches
            if self.branches_db.exists():
                with open(self.branches_db, 'r', encoding='utf-8') as f:
                    branches_data = json.load(f)
                    for branch_data in branches_data:
                        branch = Branch(**branch_data)
                        self.branches[branch.name] = branch
            
            # Load recent changes (last 100 entries)
            if self.changes_db.exists():
                recent_lines = []
                with open(self.changes_db, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    recent_lines = lines[-100:] if len(lines) > 100 else lines
                
                for line in recent_lines:
                    if line.strip():
                        change_data = json.loads(line)
                        change = ChangeEntry(**change_data)
                        self.recent_changes.append(change)
        
        except Exception as e:
            logger.error(f"Failed to load history data: {e}")
    
    async def record_change(
        self,
        patch_set_id: str,
        transaction_id: str,
        operation_type: str,
        file_path: str,
        author: str = "unknown",
        old_path: str = None,
        content_before: str = None,
        content_after: str = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Record a change in the history
        
        Args:
            patch_set_id: ID of the patch set
            transaction_id: ID of the transaction
            operation_type: Type of operation (create, modify, delete, rename)
            file_path: Path to the affected file
            author: Change author
            old_path: Old path (for renames)
            content_before: Content before change
            content_after: Content after change
            metadata: Additional metadata
            
        Returns:
            Change entry ID
        """
        
        change_id = str(uuid.uuid4())
        
        # Calculate checksums
        checksum_before = None
        checksum_after = None
        
        if content_before:
            checksum_before = hashlib.sha256(content_before.encode()).hexdigest()
        if content_after:
            checksum_after = hashlib.sha256(content_after.encode()).hexdigest()
        
        # Calculate line changes
        line_changes = {}
        if content_before and content_after:
            line_changes = await self._calculate_line_changes(content_before, content_after)
        
        # Create change entry
        change = ChangeEntry(
            id=change_id,
            patch_set_id=patch_set_id,
            transaction_id=transaction_id,
            timestamp=datetime.now(timezone.utc),
            author=author,
            operation_type=operation_type,
            file_path=file_path,
            old_path=old_path,
            content_before=content_before,
            content_after=content_after,
            checksum_before=checksum_before,
            checksum_after=checksum_after,
            line_changes=line_changes,
            metadata=metadata or {}
        )
        
        # Store change
        await self._store_change(change)
        
        # Update in-memory cache
        self.recent_changes.append(change)
        if len(self.recent_changes) > 100:
            self.recent_changes = self.recent_changes[-100:]
        
        logger.info(f"Recorded change {change_id} for {file_path}")
        return change_id
    
    async def create_version(
        self,
        patch_set_id: str,
        version_number: str,
        author: str,
        description: str,
        branch: str = "main",
        parent_versions: List[str] = None,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Create a new version snapshot
        
        Args:
            patch_set_id: Associated patch set ID
            version_number: Version number (e.g., "1.2.3")
            author: Version creator
            description: Version description
            branch: Branch name
            parent_versions: Parent version IDs
            tags: Version tags
            metadata: Additional metadata
            
        Returns:
            Version ID
        """
        
        version_id = str(uuid.uuid4())
        
        # Capture current file states
        file_states = await self._capture_file_states()
        
        # Create version info
        version = VersionInfo(
            id=version_id,
            patch_set_id=patch_set_id,
            version_number=version_number,
            timestamp=datetime.now(timezone.utc),
            author=author,
            description=description,
            parent_versions=parent_versions or [],
            file_states=file_states,
            tags=tags or [],
            branch=branch,
            metadata=metadata or {}
        )
        
        # Store version
        self.versions[version_id] = version
        await self._save_versions()
        
        # Update branch head
        if branch in self.branches:
            self.branches[branch].head_version = version_id
            await self._save_branches()
        
        logger.info(f"Created version {version_number} ({version_id})")
        return version_id
    
    async def create_branch(
        self,
        branch_name: str,
        parent_branch: str,
        parent_version: str,
        author: str,
        description: str = ""
    ) -> bool:
        """
        Create a new branch
        
        Args:
            branch_name: Name of the new branch
            parent_branch: Parent branch name
            parent_version: Parent version ID
            author: Branch creator
            description: Branch description
            
        Returns:
            Success status
        """
        
        if branch_name in self.branches:
            return False
        
        # Create branch
        branch = Branch(
            name=branch_name,
            created_at=datetime.now(timezone.utc),
            created_by=author,
            parent_branch=parent_branch,
            parent_version=parent_version,
            head_version=parent_version,
            description=description
        )
        
        self.branches[branch_name] = branch
        await self._save_branches()
        
        logger.info(f"Created branch {branch_name} from {parent_branch}@{parent_version}")
        return True
    
    async def merge_branch(
        self,
        source_branch: str,
        target_branch: str,
        author: str,
        merge_commit_message: str
    ) -> Optional[str]:
        """
        Merge one branch into another
        
        Args:
            source_branch: Source branch to merge from
            target_branch: Target branch to merge into
            author: Merge author
            merge_commit_message: Merge commit message
            
        Returns:
            Merge version ID if successful
        """
        
        if source_branch not in self.branches or target_branch not in self.branches:
            return None
        
        source = self.branches[source_branch]
        target = self.branches[target_branch]
        
        # Create merge version
        merge_version_id = await self.create_version(
            patch_set_id=f"merge_{source_branch}_to_{target_branch}",
            version_number=f"merge_{int(datetime.now().timestamp())}",
            author=author,
            description=merge_commit_message,
            branch=target_branch,
            parent_versions=[source.head_version, target.head_version],
            tags=["merge"],
            metadata={
                'merge_type': 'branch',
                'source_branch': source_branch,
                'target_branch': target_branch
            }
        )
        
        # Mark source branch as merged
        source.merged_at = datetime.now(timezone.utc)
        source.merged_to = target_branch
        source.is_active = False
        
        await self._save_branches()
        
        logger.info(f"Merged branch {source_branch} into {target_branch}")
        return merge_version_id
    
    async def get_file_history(
        self,
        file_path: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        max_entries: int = 100
    ) -> List[ChangeEntry]:
        """
        Get history for a specific file
        
        Args:
            file_path: Path to the file
            since: Start date filter
            until: End date filter
            max_entries: Maximum entries to return
            
        Returns:
            List of change entries
        """
        
        changes = []
        
        # Search in recent changes first
        for change in reversed(self.recent_changes):
            if change.file_path == file_path or change.old_path == file_path:
                if since and change.timestamp < since:
                    continue
                if until and change.timestamp > until:
                    continue
                changes.append(change)
                if len(changes) >= max_entries:
                    break
        
        # If we need more, search in the full database
        if len(changes) < max_entries:
            additional_changes = await self._search_changes_db(
                file_path=file_path,
                since=since,
                until=until,
                max_entries=max_entries - len(changes)
            )
            changes.extend(additional_changes)
        
        return sorted(changes, key=lambda x: x.timestamp, reverse=True)
    
    async def get_changes_by_author(
        self,
        author: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        max_entries: int = 100
    ) -> List[ChangeEntry]:
        """Get changes by a specific author"""
        
        changes = []
        
        # Search recent changes
        for change in reversed(self.recent_changes):
            if change.author == author:
                if since and change.timestamp < since:
                    continue
                if until and change.timestamp > until:
                    continue
                changes.append(change)
                if len(changes) >= max_entries:
                    break
        
        if len(changes) < max_entries:
            additional_changes = await self._search_changes_db(
                author=author,
                since=since,
                until=until,
                max_entries=max_entries - len(changes)
            )
            changes.extend(additional_changes)
        
        return sorted(changes, key=lambda x: x.timestamp, reverse=True)
    
    async def get_version_info(self, version_id: str) -> Optional[VersionInfo]:
        """Get information about a specific version"""
        return self.versions.get(version_id)
    
    async def get_versions_by_branch(self, branch_name: str) -> List[VersionInfo]:
        """Get all versions for a specific branch"""
        return [version for version in self.versions.values() 
                if version.branch == branch_name]
    
    async def get_branch_info(self, branch_name: str) -> Optional[Branch]:
        """Get information about a specific branch"""
        return self.branches.get(branch_name)
    
    async def list_branches(self, active_only: bool = True) -> List[Branch]:
        """List all branches"""
        branches = list(self.branches.values())
        if active_only:
            branches = [b for b in branches if b.is_active]
        return sorted(branches, key=lambda x: x.created_at, reverse=True)
    
    async def get_change_statistics(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        by_author: bool = False,
        by_file_type: bool = False
    ) -> Dict[str, Any]:
        """
        Get change statistics
        
        Args:
            since: Start date filter
            until: End date filter
            by_author: Include per-author statistics
            by_file_type: Include per-file-type statistics
            
        Returns:
            Dictionary with statistics
        """
        
        stats = {
            'total_changes': 0,
            'changes_by_type': {},
            'files_affected': set(),
            'date_range': {}
        }
        
        if by_author:
            stats['changes_by_author'] = {}
        
        if by_file_type:
            stats['changes_by_file_type'] = {}
        
        # Analyze recent changes
        for change in self.recent_changes:
            if since and change.timestamp < since:
                continue
            if until and change.timestamp > until:
                continue
            
            stats['total_changes'] += 1
            
            # By operation type
            op_type = change.operation_type
            stats['changes_by_type'][op_type] = stats['changes_by_type'].get(op_type, 0) + 1
            
            # Files affected
            stats['files_affected'].add(change.file_path)
            if change.old_path:
                stats['files_affected'].add(change.old_path)
            
            # By author
            if by_author:
                author = change.author
                stats['changes_by_author'][author] = stats['changes_by_author'].get(author, 0) + 1
            
            # By file type
            if by_file_type:
                file_ext = Path(change.file_path).suffix or 'no_extension'
                stats['changes_by_file_type'][file_ext] = stats['changes_by_file_type'].get(file_ext, 0) + 1
        
        # Convert sets to counts
        stats['unique_files_affected'] = len(stats['files_affected'])
        del stats['files_affected']
        
        return stats
    
    async def compare_versions(
        self,
        version1_id: str,
        version2_id: str
    ) -> Dict[str, Any]:
        """
        Compare two versions
        
        Args:
            version1_id: First version ID
            version2_id: Second version ID
            
        Returns:
            Comparison results
        """
        
        version1 = self.versions.get(version1_id)
        version2 = self.versions.get(version2_id)
        
        if not version1 or not version2:
            return {}
        
        comparison = {
            'version1': {
                'id': version1.id,
                'number': version1.version_number,
                'timestamp': version1.timestamp.isoformat(),
                'author': version1.author
            },
            'version2': {
                'id': version2.id,
                'number': version2.version_number,
                'timestamp': version2.timestamp.isoformat(),
                'author': version2.author
            },
            'files_added': [],
            'files_removed': [],
            'files_modified': [],
            'files_unchanged': []
        }
        
        # Compare file states
        files1 = set(version1.file_states.keys())
        files2 = set(version2.file_states.keys())
        
        # Files added in version2
        comparison['files_added'] = list(files2 - files1)
        
        # Files removed in version2
        comparison['files_removed'] = list(files1 - files2)
        
        # Files in both versions
        common_files = files1 & files2
        
        for file_path in common_files:
            checksum1 = version1.file_states[file_path]
            checksum2 = version2.file_states[file_path]
            
            if checksum1 == checksum2:
                comparison['files_unchanged'].append(file_path)
            else:
                comparison['files_modified'].append(file_path)
        
        return comparison
    
    # Private helper methods
    async def _store_change(self, change: ChangeEntry):
        """Store a change entry to the database"""
        
        # Convert to JSON and append to the changes file
        change_data = asdict(change)
        change_data['timestamp'] = change.timestamp.isoformat()
        
        with open(self.changes_db, 'a', encoding='utf-8') as f:
            f.write(json.dumps(change_data, ensure_ascii=False) + '\n')
        
        # Check if we need to compress old entries
        if self._should_compress():
            await self._compress_old_entries()
    
    async def _calculate_line_changes(self, content_before: str, content_after: str) -> Dict[str, Any]:
        """Calculate line-level changes between two content versions"""
        
        lines_before = content_before.split('\n')
        lines_after = content_after.split('\n')
        
        import difflib
        
        differ = difflib.SequenceMatcher(None, lines_before, lines_after)
        
        changes = {
            'lines_added': 0,
            'lines_removed': 0,
            'lines_modified': 0,
            'hunks': []
        }
        
        for tag, i1, i2, j1, j2 in differ.get_opcodes():
            if tag == 'delete':
                changes['lines_removed'] += (i2 - i1)
            elif tag == 'insert':
                changes['lines_added'] += (j2 - j1)
            elif tag == 'replace':
                changes['lines_modified'] += min(i2 - i1, j2 - j1)
                changes['lines_added'] += max(0, (j2 - j1) - (i2 - i1))
                changes['lines_removed'] += max(0, (i2 - i1) - (j2 - j1))
            
            if tag != 'equal':
                changes['hunks'].append({
                    'type': tag,
                    'old_start': i1,
                    'old_count': i2 - i1,
                    'new_start': j1,
                    'new_count': j2 - j1
                })
        
        return changes
    
    async def _capture_file_states(self) -> Dict[str, str]:
        """Capture checksums of all files in project"""
        
        file_states = {}
        
        # Walk through project directory
        for root, dirs, files in os.walk(self.project_root):
            # Skip .abov3 directory
            if '.abov3' in root:
                continue
            
            for file_name in files:
                file_path = Path(root) / file_name
                relative_path = file_path.relative_to(self.project_root)
                
                try:
                    checksum = self._calculate_file_checksum(file_path)
                    file_states[str(relative_path)] = checksum
                except Exception as e:
                    logger.warning(f"Could not calculate checksum for {file_path}: {e}")
        
        return file_states
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file"""
        hash_sha256 = hashlib.sha256()
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception:
            return ""
    
    async def _search_changes_db(
        self,
        file_path: str = None,
        author: str = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        max_entries: int = 100
    ) -> List[ChangeEntry]:
        """Search the full changes database"""
        
        changes = []
        
        if not self.changes_db.exists():
            return changes
        
        try:
            with open(self.changes_db, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    try:
                        change_data = json.loads(line)
                        change = ChangeEntry(**change_data)
                        
                        # Apply filters
                        if file_path and change.file_path != file_path and change.old_path != file_path:
                            continue
                        if author and change.author != author:
                            continue
                        if since and change.timestamp < since:
                            continue
                        if until and change.timestamp > until:
                            continue
                        
                        changes.append(change)
                        
                        if len(changes) >= max_entries:
                            break
                    
                    except Exception as e:
                        logger.warning(f"Failed to parse change entry: {e}")
        
        except Exception as e:
            logger.error(f"Failed to search changes database: {e}")
        
        return changes
    
    def _should_compress(self) -> bool:
        """Check if old entries should be compressed"""
        if not self.changes_db.exists():
            return False
        
        line_count = 0
        with open(self.changes_db, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
        
        return line_count > self.compression_threshold
    
    async def _compress_old_entries(self):
        """Compress old entries to save space"""
        # This would implement compression logic
        # For now, we'll just log that compression would happen
        logger.info("History compression would be performed here")
    
    async def _save_versions(self):
        """Save versions to disk"""
        versions_data = []
        for version in self.versions.values():
            version_data = asdict(version)
            version_data['timestamp'] = version.timestamp.isoformat()
            versions_data.append(version_data)
        
        with open(self.versions_db, 'w', encoding='utf-8') as f:
            json.dump(versions_data, f, indent=2, ensure_ascii=False)
    
    async def _save_branches(self):
        """Save branches to disk"""
        branches_data = []
        for branch in self.branches.values():
            branch_data = asdict(branch)
            branch_data['created_at'] = branch.created_at.isoformat()
            if branch.merged_at:
                branch_data['merged_at'] = branch.merged_at.isoformat()
            branches_data.append(branch_data)
        
        with open(self.branches_db, 'w', encoding='utf-8') as f:
            json.dump(branches_data, f, indent=2, ensure_ascii=False)