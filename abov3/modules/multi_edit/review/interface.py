"""
Interactive Review Interface

Enterprise-grade interface for comprehensive review of multi-file patch sets with
line-by-line approval, batch operations, and intelligent review assistance.
"""

import os
import json
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import logging
from datetime import datetime

from ..core.patch_manager import PatchSet, FileChange
from ..diff.engine import DiffEngine, FileDiff, ChangeType

logger = logging.getLogger(__name__)


class ReviewAction(Enum):
    """Available review actions"""
    APPROVE = "approve"
    REJECT = "reject"
    SKIP = "skip"
    APPROVE_ALL = "approve_all"
    REJECT_ALL = "reject_all"
    APPROVE_FILE = "approve_file"
    REJECT_FILE = "reject_file"
    VIEW_CONTEXT = "view_context"
    EDIT_INLINE = "edit_inline"
    ADD_COMMENT = "add_comment"


@dataclass
class ReviewDecision:
    """Represents a review decision for a specific change"""
    file_path: str
    line_number: Optional[int] = None
    action: ReviewAction = ReviewAction.APPROVE
    comment: str = ""
    reviewer: str = ""
    timestamp: datetime = None
    custom_edit: str = None  # For inline edits
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ReviewSession:
    """Represents an active review session"""
    id: str
    patch_set_id: str
    reviewer: str
    status: str  # 'active', 'completed', 'paused'
    decisions: List[ReviewDecision]
    current_file_index: int = 0
    current_line_index: int = 0
    started_at: datetime = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.started_at is None:
            self.started_at = datetime.now()


class ReviewInterface:
    """
    Interactive review interface for multi-file patch sets
    
    Features:
    - Line-by-line review with context
    - Batch approval/rejection operations
    - Inline editing capabilities
    - Comment and annotation system
    - Smart review suggestions
    - Progress tracking and resume
    - Multiple reviewer support
    """
    
    def __init__(self, project_root: str = None):
        self.project_root = project_root
        self.diff_engine = DiffEngine()
        
        # Review sessions
        self.active_sessions: Dict[str, ReviewSession] = {}
        
        # Callbacks for different output formats
        self.display_callbacks = {
            'console': self._console_display,
            'json': self._json_display,
            'html': self._html_display
        }
        
        # Review statistics
        self.review_stats = {}
    
    async def start_review(
        self, 
        patch_set: PatchSet, 
        reviewer: str = "unknown",
        output_format: str = "console",
        auto_mode: bool = False
    ) -> str:
        """
        Start a new review session
        
        Args:
            patch_set: Patch set to review
            reviewer: Name of reviewer
            output_format: Display format (console, json, html)
            auto_mode: If True, run in automated mode
            
        Returns:
            Review session ID
        """
        import uuid
        session_id = str(uuid.uuid4())
        
        # Create review session
        session = ReviewSession(
            id=session_id,
            patch_set_id=patch_set.id,
            reviewer=reviewer,
            status='active',
            decisions=[]
        )
        
        self.active_sessions[session_id] = session
        
        # Generate diffs for all files
        file_diffs = []
        for file_change in patch_set.files:
            diff = await self.diff_engine.generate_file_diff(
                file_change.old_content or "",
                file_change.new_content or "",
                file_change.file_path,
                file_change.old_path
            )
            file_diffs.append(diff)
        
        if auto_mode:
            # Run automated review
            return await self._run_automated_review(session, patch_set, file_diffs)
        else:
            # Run interactive review
            return await self._run_interactive_review(session, patch_set, file_diffs, output_format)
    
    async def _run_interactive_review(
        self,
        session: ReviewSession,
        patch_set: PatchSet,
        file_diffs: List[FileDiff],
        output_format: str
    ) -> str:
        """Run interactive review session"""
        
        display_func = self.display_callbacks.get(output_format, self._console_display)
        
        # Display patch set overview
        await display_func("patch_overview", {
            'patch_set': patch_set,
            'file_count': len(file_diffs),
            'total_changes': sum(
                diff.statistics['lines_added'] + 
                diff.statistics['lines_removed'] + 
                diff.statistics['lines_modified'] 
                for diff in file_diffs
            )
        })
        
        # Review each file
        for file_index, file_diff in enumerate(file_diffs):
            session.current_file_index = file_index
            
            # Display file header
            await display_func("file_header", {
                'file_diff': file_diff,
                'file_index': file_index + 1,
                'total_files': len(file_diffs)
            })
            
            # Review file changes
            file_decision = await self._review_file_changes(session, file_diff, display_func)
            
            if file_decision == 'quit':
                session.status = 'paused'
                break
            elif file_decision == 'skip_remaining':
                break
        
        # Complete session
        if session.status == 'active':
            session.status = 'completed'
            session.completed_at = datetime.now()
        
        # Generate review summary
        summary = await self._generate_review_summary(session, patch_set, file_diffs)
        await display_func("review_summary", summary)
        
        return session_id
    
    async def _review_file_changes(
        self,
        session: ReviewSession,
        file_diff: FileDiff,
        display_func: Callable
    ) -> str:
        """Review changes in a single file"""
        
        # Group changes by hunks for better review flow
        for hunk_index, hunk in enumerate(file_diff.hunks):
            # Display hunk header
            await display_func("hunk_header", {
                'hunk': hunk,
                'hunk_index': hunk_index + 1,
                'total_hunks': len(file_diff.hunks)
            })
            
            # Review each line in hunk
            for line_index, line in enumerate(hunk.lines):
                if line.change_type == ChangeType.UNCHANGED:
                    continue  # Skip unchanged lines unless requested
                
                session.current_line_index = line_index
                
                # Display line for review
                await display_func("line_review", {
                    'line': line,
                    'context_before': self._get_context_lines(hunk.lines, line_index, -3),
                    'context_after': self._get_context_lines(hunk.lines, line_index, 3)
                })
                
                # Get user decision
                decision = await self._get_review_decision(session, file_diff.file_path, line)
                
                if decision.action in [ReviewAction.APPROVE_ALL, ReviewAction.REJECT_ALL]:
                    return await self._handle_batch_decision(session, file_diff, decision)
                
                elif decision.action == ReviewAction.APPROVE_FILE:
                    await self._approve_entire_file(session, file_diff)
                    return 'continue'
                
                elif decision.action == ReviewAction.REJECT_FILE:
                    await self._reject_entire_file(session, file_diff)
                    return 'continue'
                
                elif decision.action == ReviewAction.VIEW_CONTEXT:
                    await self._show_extended_context(file_diff, line, display_func)
                    continue  # Re-review same line
                
                elif decision.action == ReviewAction.EDIT_INLINE:
                    edited_content = await self._handle_inline_edit(line, decision.custom_edit)
                    decision.custom_edit = edited_content
                
                # Store decision
                session.decisions.append(decision)
        
        return 'continue'
    
    def _get_context_lines(self, lines: List, center_index: int, offset: int) -> List:
        """Get context lines around a center line"""
        if offset < 0:
            start = max(0, center_index + offset)
            end = center_index
        else:
            start = center_index + 1
            end = min(len(lines), center_index + offset + 1)
        
        return lines[start:end]
    
    async def _get_review_decision(
        self,
        session: ReviewSession,
        file_path: str,
        line
    ) -> ReviewDecision:
        """Get review decision for a line (placeholder for actual UI)"""
        # In a real implementation, this would present a UI to the user
        # For now, we'll simulate automatic approval
        return ReviewDecision(
            file_path=file_path,
            line_number=line.line_number_new or line.line_number_old,
            action=ReviewAction.APPROVE,
            reviewer=session.reviewer
        )
    
    async def _handle_batch_decision(self, session: ReviewSession, file_diff: FileDiff, decision: ReviewDecision) -> str:
        """Handle batch approval/rejection decisions"""
        if decision.action == ReviewAction.APPROVE_ALL:
            # Approve all remaining changes in patch set
            for remaining_file in self.active_sessions[session.id]:
                await self._approve_entire_file(session, remaining_file)
            return 'skip_remaining'
        
        elif decision.action == ReviewAction.REJECT_ALL:
            # Reject all remaining changes
            for remaining_file in self.active_sessions[session.id]:
                await self._reject_entire_file(session, remaining_file)
            return 'skip_remaining'
        
        return 'continue'
    
    async def _approve_entire_file(self, session: ReviewSession, file_diff: FileDiff):
        """Approve all changes in a file"""
        decision = ReviewDecision(
            file_path=file_diff.file_path,
            action=ReviewAction.APPROVE_FILE,
            comment="Entire file approved",
            reviewer=session.reviewer
        )
        session.decisions.append(decision)
    
    async def _reject_entire_file(self, session: ReviewSession, file_diff: FileDiff):
        """Reject all changes in a file"""
        decision = ReviewDecision(
            file_path=file_diff.file_path,
            action=ReviewAction.REJECT_FILE,
            comment="Entire file rejected",
            reviewer=session.reviewer
        )
        session.decisions.append(decision)
    
    async def _show_extended_context(self, file_diff: FileDiff, line, display_func: Callable):
        """Show extended context around a line"""
        # Find the line in the original diff and show more context
        extended_context = await self._get_extended_file_context(file_diff, line)
        await display_func("extended_context", extended_context)
    
    async def _get_extended_file_context(self, file_diff: FileDiff, target_line) -> Dict:
        """Get extended context for a specific line"""
        return {
            'file_path': file_diff.file_path,
            'target_line': target_line,
            'context_lines': []  # Would be populated with surrounding lines
        }
    
    async def _handle_inline_edit(self, line, custom_edit: str) -> str:
        """Handle inline editing of a line"""
        if custom_edit:
            return custom_edit
        else:
            # In real implementation, would open editor
            return line.content
    
    async def _run_automated_review(
        self,
        session: ReviewSession,
        patch_set: PatchSet,
        file_diffs: List[FileDiff]
    ) -> str:
        """Run automated review with AI assistance"""
        
        # Apply automated review rules
        for file_diff in file_diffs:
            decision = await self._apply_automated_rules(file_diff)
            
            session.decisions.append(ReviewDecision(
                file_path=file_diff.file_path,
                action=decision,
                comment="Automated review decision",
                reviewer="AI Assistant"
            ))
        
        session.status = 'completed'
        session.completed_at = datetime.now()
        
        return session.id
    
    async def _apply_automated_rules(self, file_diff: FileDiff) -> ReviewAction:
        """Apply automated review rules to a file diff"""
        
        # Example rules:
        # 1. Auto-approve small changes
        total_changes = (
            file_diff.statistics['lines_added'] + 
            file_diff.statistics['lines_removed']
        )
        
        if total_changes <= 5:
            return ReviewAction.APPROVE
        
        # 2. Flag large deletions for manual review
        if file_diff.statistics['lines_removed'] > 50:
            return ReviewAction.SKIP  # Requires manual review
        
        # 3. Auto-approve documentation changes
        if file_diff.file_path.endswith(('.md', '.txt', '.rst')):
            return ReviewAction.APPROVE
        
        # Default to requiring manual review
        return ReviewAction.SKIP
    
    async def _generate_review_summary(
        self,
        session: ReviewSession,
        patch_set: PatchSet,
        file_diffs: List[FileDiff]
    ) -> Dict[str, Any]:
        """Generate comprehensive review summary"""
        
        approved_files = []
        rejected_files = []
        skipped_files = []
        inline_edits = []
        
        for decision in session.decisions:
            if decision.action in [ReviewAction.APPROVE, ReviewAction.APPROVE_FILE]:
                approved_files.append(decision.file_path)
            elif decision.action in [ReviewAction.REJECT, ReviewAction.REJECT_FILE]:
                rejected_files.append(decision.file_path)
            elif decision.action == ReviewAction.SKIP:
                skipped_files.append(decision.file_path)
            
            if decision.custom_edit:
                inline_edits.append({
                    'file_path': decision.file_path,
                    'line_number': decision.line_number,
                    'edit': decision.custom_edit
                })
        
        return {
            'session_id': session.id,
            'patch_set_id': patch_set.id,
            'reviewer': session.reviewer,
            'status': session.status,
            'duration': (session.completed_at - session.started_at).total_seconds() if session.completed_at else None,
            'statistics': {
                'total_files': len(file_diffs),
                'approved_files': len(approved_files),
                'rejected_files': len(rejected_files),
                'skipped_files': len(skipped_files),
                'inline_edits': len(inline_edits)
            },
            'approved_files': approved_files,
            'rejected_files': rejected_files,
            'skipped_files': skipped_files,
            'inline_edits': inline_edits,
            'total_changes': sum(
                diff.statistics['lines_added'] + diff.statistics['lines_removed']
                for diff in file_diffs
            )
        }
    
    # Display callback functions
    async def _console_display(self, display_type: str, data: Dict):
        """Console-based display"""
        if display_type == "patch_overview":
            print(f"\n=== PATCH SET REVIEW ===")
            print(f"Patch ID: {data['patch_set'].id}")
            print(f"Description: {data['patch_set'].description}")
            print(f"Files: {data['file_count']}")
            print(f"Total Changes: {data['total_changes']}")
            print("=" * 40)
        
        elif display_type == "file_header":
            file_diff = data['file_diff']
            print(f"\n--- File {data['file_index']}/{data['total_files']}: {file_diff.file_path} ---")
            print(f"Changes: +{file_diff.statistics['lines_added']} -{file_diff.statistics['lines_removed']}")
        
        elif display_type == "hunk_header":
            hunk = data['hunk']
            print(f"\n@@ Hunk {data['hunk_index']}/{data['total_hunks']} @@")
            if hunk.context:
                print(f"Context: {hunk.context}")
        
        elif display_type == "line_review":
            line = data['line']
            change_symbol = {
                ChangeType.ADDED: '+',
                ChangeType.REMOVED: '-',
                ChangeType.MODIFIED: '~'
            }.get(line.change_type, ' ')
            
            line_num = line.line_number_new or line.line_number_old or '?'
            print(f"{change_symbol}{line_num}: {line.content}")
        
        elif display_type == "review_summary":
            print(f"\n=== REVIEW SUMMARY ===")
            print(f"Session: {data['session_id']}")
            print(f"Status: {data['status']}")
            print(f"Approved: {data['statistics']['approved_files']}")
            print(f"Rejected: {data['statistics']['rejected_files']}")
            print(f"Skipped: {data['statistics']['skipped_files']}")
    
    async def _json_display(self, display_type: str, data: Dict):
        """JSON-based display"""
        output = {
            'type': display_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        print(json.dumps(output, indent=2, default=str))
    
    async def _html_display(self, display_type: str, data: Dict):
        """HTML-based display (placeholder)"""
        # Would generate HTML for web-based review interface
        pass
    
    async def get_review_session(self, session_id: str) -> Optional[ReviewSession]:
        """Get review session by ID"""
        return self.active_sessions.get(session_id)
    
    async def pause_review_session(self, session_id: str) -> bool:
        """Pause an active review session"""
        session = self.active_sessions.get(session_id)
        if session and session.status == 'active':
            session.status = 'paused'
            return True
        return False
    
    async def resume_review_session(self, session_id: str) -> bool:
        """Resume a paused review session"""
        session = self.active_sessions.get(session_id)
        if session and session.status == 'paused':
            session.status = 'active'
            return True
        return False
    
    async def get_approved_changes(self, session_id: str) -> List[str]:
        """Get list of approved file paths from a session"""
        session = self.active_sessions.get(session_id)
        if not session:
            return []
        
        approved = []
        for decision in session.decisions:
            if decision.action in [ReviewAction.APPROVE, ReviewAction.APPROVE_FILE]:
                if decision.file_path not in approved:
                    approved.append(decision.file_path)
        
        return approved