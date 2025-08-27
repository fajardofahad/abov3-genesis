"""
ABOV3 Genesis Multi-file Edits & Patch Sets Module

Enterprise-grade module for managing complex multi-file edits with atomic transactions,
intelligent conflict resolution, and comprehensive review capabilities.

Components:
- Patch Set Manager: Atomic multi-file changes
- Diff Engine: Context-aware line-by-line analysis
- Review Interface: Interactive approval system
- Conflict Resolution: Intelligent merge handling
- Transaction Manager: Rollback and undo capabilities
- History Tracking: Version and change management
- Git Integration: Seamless version control
"""

from .core.patch_manager import PatchSetManager
from .diff.engine import DiffEngine
from .review.interface import ReviewInterface
from .conflict.resolver import ConflictResolver
from .transaction.manager import TransactionManager
from .history.tracker import HistoryTracker
from .git_integration.manager import GitIntegrationManager

__version__ = "1.0.0"
__author__ = "ABOV3 Genesis Team"

__all__ = [
    "PatchSetManager",
    "DiffEngine", 
    "ReviewInterface",
    "ConflictResolver",
    "TransactionManager",
    "HistoryTracker",
    "GitIntegrationManager"
]


class MultiEditOrchestrator:
    """Main orchestrator for multi-file edit operations"""
    
    def __init__(self, project_root: str = None):
        self.project_root = project_root
        self.patch_manager = PatchSetManager(project_root)
        self.diff_engine = DiffEngine()
        self.review_interface = ReviewInterface()
        self.conflict_resolver = ConflictResolver()
        self.transaction_manager = TransactionManager(project_root)
        self.history_tracker = HistoryTracker(project_root)
        self.git_manager = GitIntegrationManager(project_root)
    
    async def create_patch_set(self, changes: dict, description: str = "Multi-file edit"):
        """Create a new patch set for multi-file changes"""
        return await self.patch_manager.create_patch_set(changes, description)
    
    async def review_changes(self, patch_set_id: str):
        """Start interactive review of changes"""
        patch_set = await self.patch_manager.get_patch_set(patch_set_id)
        return await self.review_interface.start_review(patch_set)
    
    async def apply_changes(self, patch_set_id: str, approved_changes: list):
        """Apply approved changes with transaction safety"""
        async with self.transaction_manager.transaction(patch_set_id):
            return await self.patch_manager.apply_patch_set(patch_set_id, approved_changes)
    
    async def rollback_changes(self, patch_set_id: str):
        """Rollback applied changes"""
        return await self.transaction_manager.rollback(patch_set_id)