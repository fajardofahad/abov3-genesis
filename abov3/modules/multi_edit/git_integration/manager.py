"""
Enterprise Git Integration Manager

Seamless integration with Git version control system for multi-file patch operations,
with advanced features for enterprise workflows, conflict resolution, and automation.
"""

import os
import subprocess
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import asyncio
import tempfile
import shutil

logger = logging.getLogger(__name__)


@dataclass
class GitCommitInfo:
    """Information about a git commit"""
    hash: str
    author: str
    email: str
    timestamp: str
    message: str
    files_changed: List[str]
    insertions: int
    deletions: int
    
    @classmethod
    def from_git_log(cls, log_entry: str) -> 'GitCommitInfo':
        """Parse git log entry into GitCommitInfo"""
        lines = log_entry.strip().split('\n')
        if len(lines) < 4:
            raise ValueError("Invalid git log entry")
        
        commit_hash = lines[0].split()[1]
        author_line = lines[1].split(': ', 1)[1]
        author, email = author_line.rsplit(' ', 1)
        email = email.strip('<>')
        timestamp = lines[2].split(': ', 1)[1]
        message = lines[3].split(': ', 1)[1] if len(lines) > 3 else ""
        
        return cls(
            hash=commit_hash,
            author=author,
            email=email,
            timestamp=timestamp,
            message=message,
            files_changed=[],
            insertions=0,
            deletions=0
        )


@dataclass
class GitBranch:
    """Information about a git branch"""
    name: str
    is_current: bool
    is_remote: bool
    last_commit: str
    tracking_branch: Optional[str] = None
    
    @classmethod
    def from_git_branch(cls, branch_line: str) -> 'GitBranch':
        """Parse git branch output into GitBranch"""
        is_current = branch_line.startswith('*')
        branch_name = branch_line.strip('* ').strip()
        
        # Handle remote branches
        is_remote = branch_name.startswith('remotes/')
        if is_remote:
            branch_name = branch_name[8:]  # Remove 'remotes/'
        
        return cls(
            name=branch_name,
            is_current=is_current,
            is_remote=is_remote,
            last_commit="",
            tracking_branch=None
        )


@dataclass
class GitStatus:
    """Git repository status"""
    is_clean: bool
    modified_files: List[str]
    staged_files: List[str]
    untracked_files: List[str]
    deleted_files: List[str]
    current_branch: str
    ahead_by: int = 0
    behind_by: int = 0


class GitIntegrationManager:
    """
    Enterprise Git Integration Manager
    
    Features:
    - Automatic commit creation for patch sets
    - Branch management and merging
    - Conflict detection and resolution
    - Git hooks integration
    - Remote repository synchronization
    - Stash management for temporary changes
    - Advanced diff integration
    - Enterprise workflow automation
    """
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        
        # Validate git repository
        self.git_dir = self.project_root / ".git"
        self.is_git_repo = self.git_dir.exists()
        
        # Git configuration
        self.git_config = {
            'user_name': None,
            'user_email': None,
            'remote_url': None,
            'default_branch': 'main'
        }
        
        # Initialize git configuration
        if self.is_git_repo:
            asyncio.create_task(self._load_git_config())
        
        # Hook callbacks
        self.pre_commit_hooks = []
        self.post_commit_hooks = []
        self.pre_push_hooks = []
        self.post_push_hooks = []
    
    async def _load_git_config(self):
        """Load git configuration"""
        try:
            # Get user configuration
            result = await self._run_git_command(['config', 'user.name'])
            self.git_config['user_name'] = result.strip() if result else None
            
            result = await self._run_git_command(['config', 'user.email'])
            self.git_config['user_email'] = result.strip() if result else None
            
            # Get remote URL
            result = await self._run_git_command(['config', 'remote.origin.url'])
            self.git_config['remote_url'] = result.strip() if result else None
            
            # Get default branch
            result = await self._run_git_command(['symbolic-ref', 'refs/remotes/origin/HEAD'])
            if result:
                default_branch = result.strip().split('/')[-1]
                self.git_config['default_branch'] = default_branch
            
        except Exception as e:
            logger.warning(f"Failed to load git config: {e}")
    
    async def initialize_repository(self, remote_url: str = None) -> bool:
        """
        Initialize a new git repository
        
        Args:
            remote_url: Optional remote repository URL
            
        Returns:
            Success status
        """
        
        if self.is_git_repo:
            return True
        
        try:
            # Initialize git repository
            await self._run_git_command(['init'])
            
            # Set up initial configuration
            if self.git_config['user_name']:
                await self._run_git_command(['config', 'user.name', self.git_config['user_name']])
            
            if self.git_config['user_email']:
                await self._run_git_command(['config', 'user.email', self.git_config['user_email']])
            
            # Add remote if provided
            if remote_url:
                await self._run_git_command(['remote', 'add', 'origin', remote_url])
                self.git_config['remote_url'] = remote_url
            
            self.is_git_repo = True
            self.git_dir = self.project_root / ".git"
            
            logger.info("Git repository initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize git repository: {e}")
            return False
    
    async def get_status(self) -> GitStatus:
        """Get current git status"""
        
        if not self.is_git_repo:
            return GitStatus(
                is_clean=True,
                modified_files=[],
                staged_files=[],
                untracked_files=[],
                deleted_files=[],
                current_branch="main"
            )
        
        try:
            # Get current branch
            branch_result = await self._run_git_command(['rev-parse', '--abbrev-ref', 'HEAD'])
            current_branch = branch_result.strip() if branch_result else "main"
            
            # Get status
            status_result = await self._run_git_command(['status', '--porcelain'])
            
            modified_files = []
            staged_files = []
            untracked_files = []
            deleted_files = []
            
            if status_result:
                for line in status_result.split('\n'):
                    if not line.strip():
                        continue
                    
                    status_code = line[:2]
                    file_path = line[3:].strip()
                    
                    if status_code[0] in ['A', 'M', 'D', 'R', 'C']:
                        staged_files.append(file_path)
                    
                    if status_code[1] == 'M':
                        modified_files.append(file_path)
                    elif status_code[1] == 'D':
                        deleted_files.append(file_path)
                    elif status_code == '??':
                        untracked_files.append(file_path)
            
            # Check if repository is clean
            is_clean = not (modified_files or staged_files or untracked_files or deleted_files)
            
            # Get ahead/behind status
            ahead_by, behind_by = await self._get_ahead_behind_count(current_branch)
            
            return GitStatus(
                is_clean=is_clean,
                modified_files=modified_files,
                staged_files=staged_files,
                untracked_files=untracked_files,
                deleted_files=deleted_files,
                current_branch=current_branch,
                ahead_by=ahead_by,
                behind_by=behind_by
            )
            
        except Exception as e:
            logger.error(f"Failed to get git status: {e}")
            return GitStatus(
                is_clean=False,
                modified_files=[],
                staged_files=[],
                untracked_files=[],
                deleted_files=[],
                current_branch="main"
            )
    
    async def create_commit_for_patch_set(
        self,
        patch_set_id: str,
        description: str,
        author: str = None,
        files_to_add: List[str] = None,
        create_branch: bool = False,
        branch_name: str = None
    ) -> Optional[str]:
        """
        Create a git commit for a patch set
        
        Args:
            patch_set_id: Patch set ID
            description: Commit message
            author: Commit author (if different from config)
            files_to_add: Specific files to add (None means all)
            create_branch: Whether to create a new branch
            branch_name: Name for new branch
            
        Returns:
            Commit hash if successful
        """
        
        if not self.is_git_repo:
            return None
        
        try:
            # Create branch if requested
            if create_branch:
                if not branch_name:
                    branch_name = f"patch-set-{patch_set_id}"
                
                await self._run_git_command(['checkout', '-b', branch_name])
                logger.info(f"Created and switched to branch {branch_name}")
            
            # Add files
            if files_to_add:
                for file_path in files_to_add:
                    await self._run_git_command(['add', file_path])
            else:
                await self._run_git_command(['add', '-A'])
            
            # Create commit
            commit_args = ['commit', '-m', description]
            
            # Set author if specified
            if author:
                commit_args.extend(['--author', f'{author} <{author}@abov3.ai>'])
            
            await self._run_git_command(commit_args)
            
            # Get commit hash
            commit_hash = await self._run_git_command(['rev-parse', 'HEAD'])
            commit_hash = commit_hash.strip() if commit_hash else None
            
            # Run post-commit hooks
            for hook in self.post_commit_hooks:
                await hook(commit_hash, patch_set_id, description)
            
            logger.info(f"Created commit {commit_hash} for patch set {patch_set_id}")
            return commit_hash
            
        except Exception as e:
            logger.error(f"Failed to create commit: {e}")
            return None
    
    async def create_branch(self, branch_name: str, base_branch: str = None) -> bool:
        """Create a new git branch"""
        
        if not self.is_git_repo:
            return False
        
        try:
            if base_branch:
                await self._run_git_command(['checkout', '-b', branch_name, base_branch])
            else:
                await self._run_git_command(['checkout', '-b', branch_name])
            
            logger.info(f"Created branch {branch_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create branch {branch_name}: {e}")
            return False
    
    async def switch_branch(self, branch_name: str) -> bool:
        """Switch to a different branch"""
        
        if not self.is_git_repo:
            return False
        
        try:
            await self._run_git_command(['checkout', branch_name])
            logger.info(f"Switched to branch {branch_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to switch to branch {branch_name}: {e}")
            return False
    
    async def merge_branch(
        self,
        source_branch: str,
        target_branch: str = None,
        strategy: str = "merge",
        delete_source: bool = False
    ) -> bool:
        """
        Merge one branch into another
        
        Args:
            source_branch: Branch to merge from
            target_branch: Branch to merge into (current if None)
            strategy: Merge strategy ('merge', 'squash', 'rebase')
            delete_source: Delete source branch after merge
            
        Returns:
            Success status
        """
        
        if not self.is_git_repo:
            return False
        
        try:
            # Switch to target branch if specified
            if target_branch:
                await self.switch_branch(target_branch)
            
            # Perform merge based on strategy
            if strategy == "squash":
                await self._run_git_command(['merge', '--squash', source_branch])
                # Need to commit after squash merge
                await self._run_git_command(['commit', '-m', f'Squash merge {source_branch}'])
            elif strategy == "rebase":
                # Switch to source branch and rebase onto target
                current_branch = await self._get_current_branch()
                await self.switch_branch(source_branch)
                await self._run_git_command(['rebase', current_branch])
                await self.switch_branch(current_branch)
                await self._run_git_command(['merge', source_branch])
            else:  # Regular merge
                await self._run_git_command(['merge', source_branch])
            
            # Delete source branch if requested
            if delete_source:
                await self._run_git_command(['branch', '-d', source_branch])
            
            logger.info(f"Merged {source_branch} using {strategy} strategy")
            return True
            
        except Exception as e:
            logger.error(f"Failed to merge {source_branch}: {e}")
            return False
    
    async def push_to_remote(
        self,
        branch_name: str = None,
        remote: str = "origin",
        force: bool = False,
        set_upstream: bool = False
    ) -> bool:
        """
        Push branch to remote repository
        
        Args:
            branch_name: Branch to push (current if None)
            remote: Remote name
            force: Force push
            set_upstream: Set upstream tracking
            
        Returns:
            Success status
        """
        
        if not self.is_git_repo:
            return False
        
        try:
            # Get current branch if not specified
            if not branch_name:
                branch_name = await self._get_current_branch()
            
            # Run pre-push hooks
            for hook in self.pre_push_hooks:
                await hook(branch_name, remote)
            
            # Build push command
            push_args = ['push']
            
            if force:
                push_args.append('--force')
            
            if set_upstream:
                push_args.extend(['-u', remote, branch_name])
            else:
                push_args.extend([remote, branch_name])
            
            await self._run_git_command(push_args)
            
            # Run post-push hooks
            for hook in self.post_push_hooks:
                await hook(branch_name, remote)
            
            logger.info(f"Pushed {branch_name} to {remote}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to push {branch_name}: {e}")
            return False
    
    async def pull_from_remote(
        self,
        branch_name: str = None,
        remote: str = "origin",
        strategy: str = "merge"
    ) -> bool:
        """
        Pull changes from remote repository
        
        Args:
            branch_name: Branch to pull (current if None)
            remote: Remote name
            strategy: Pull strategy ('merge' or 'rebase')
            
        Returns:
            Success status
        """
        
        if not self.is_git_repo:
            return False
        
        try:
            if strategy == "rebase":
                await self._run_git_command(['pull', '--rebase', remote, branch_name or ''])
            else:
                await self._run_git_command(['pull', remote, branch_name or ''])
            
            logger.info(f"Pulled changes from {remote}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to pull from {remote}: {e}")
            return False
    
    async def stash_changes(self, message: str = "ABOV3 auto-stash") -> Optional[str]:
        """Stash current changes"""
        
        if not self.is_git_repo:
            return None
        
        try:
            await self._run_git_command(['stash', 'save', message])
            
            # Get stash reference
            stash_list = await self._run_git_command(['stash', 'list'])
            if stash_list:
                stash_ref = stash_list.split('\n')[0].split(':')[0]
                logger.info(f"Stashed changes as {stash_ref}")
                return stash_ref
            
        except Exception as e:
            logger.error(f"Failed to stash changes: {e}")
        
        return None
    
    async def apply_stash(self, stash_ref: str = "stash@{0}") -> bool:
        """Apply stashed changes"""
        
        if not self.is_git_repo:
            return False
        
        try:
            await self._run_git_command(['stash', 'apply', stash_ref])
            logger.info(f"Applied stash {stash_ref}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply stash {stash_ref}: {e}")
            return False
    
    async def get_commit_history(
        self,
        max_count: int = 50,
        since: str = None,
        until: str = None,
        author: str = None,
        file_path: str = None
    ) -> List[GitCommitInfo]:
        """
        Get commit history with optional filters
        
        Args:
            max_count: Maximum number of commits
            since: Since date/commit
            until: Until date/commit
            author: Author filter
            file_path: File path filter
            
        Returns:
            List of commit information
        """
        
        if not self.is_git_repo:
            return []
        
        try:
            # Build log command
            log_args = [
                'log',
                '--pretty=format:commit %H%nauthor: %an <%ae>%ndate: %ai%nmessage: %s',
                f'-{max_count}'
            ]
            
            if since:
                log_args.append(f'--since={since}')
            
            if until:
                log_args.append(f'--until={until}')
            
            if author:
                log_args.append(f'--author={author}')
            
            if file_path:
                log_args.append('--')
                log_args.append(file_path)
            
            log_output = await self._run_git_command(log_args)
            
            if not log_output:
                return []
            
            # Parse commit entries
            commits = []
            commit_entries = log_output.strip().split('\n\n')
            
            for entry in commit_entries:
                if entry.strip():
                    try:
                        commit_info = GitCommitInfo.from_git_log(entry)
                        commits.append(commit_info)
                    except Exception as e:
                        logger.warning(f"Failed to parse commit entry: {e}")
            
            return commits
            
        except Exception as e:
            logger.error(f"Failed to get commit history: {e}")
            return []
    
    async def get_branches(self, include_remote: bool = False) -> List[GitBranch]:
        """Get list of branches"""
        
        if not self.is_git_repo:
            return []
        
        try:
            branch_args = ['branch']
            if include_remote:
                branch_args.append('-a')
            
            branch_output = await self._run_git_command(branch_args)
            
            if not branch_output:
                return []
            
            branches = []
            for line in branch_output.split('\n'):
                if line.strip():
                    try:
                        branch = GitBranch.from_git_branch(line)
                        branches.append(branch)
                    except Exception as e:
                        logger.warning(f"Failed to parse branch: {e}")
            
            return branches
            
        except Exception as e:
            logger.error(f"Failed to get branches: {e}")
            return []
    
    async def generate_patch_file(
        self,
        commit_range: str,
        output_path: str = None
    ) -> Optional[str]:
        """
        Generate patch file from commit range
        
        Args:
            commit_range: Git commit range (e.g., "HEAD~3..HEAD")
            output_path: Output file path
            
        Returns:
            Path to generated patch file
        """
        
        if not self.is_git_repo:
            return None
        
        try:
            if not output_path:
                output_path = self.project_root / f"patch_{int(datetime.now().timestamp())}.patch"
            
            await self._run_git_command(['format-patch', commit_range, '--stdout'], output_file=output_path)
            
            logger.info(f"Generated patch file: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to generate patch file: {e}")
            return None
    
    async def apply_patch_file(self, patch_file_path: str) -> bool:
        """Apply a patch file"""
        
        if not self.is_git_repo:
            return False
        
        try:
            with open(patch_file_path, 'r') as f:
                patch_content = f.read()
            
            # Apply patch using git apply
            process = await asyncio.create_subprocess_exec(
                'git', 'apply', patch_file_path,
                cwd=self.project_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info(f"Applied patch file: {patch_file_path}")
                return True
            else:
                logger.error(f"Failed to apply patch: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to apply patch file: {e}")
            return False
    
    # Helper methods
    async def _run_git_command(
        self,
        args: List[str],
        output_file: str = None
    ) -> Optional[str]:
        """Run a git command and return output"""
        
        try:
            if output_file:
                with open(output_file, 'w') as f:
                    process = await asyncio.create_subprocess_exec(
                        'git', *args,
                        cwd=self.project_root,
                        stdout=f,
                        stderr=asyncio.subprocess.PIPE
                    )
                    
                    _, stderr = await process.communicate()
                    
                    if process.returncode != 0:
                        raise subprocess.CalledProcessError(
                            process.returncode,
                            f"git {' '.join(args)}",
                            stderr.decode()
                        )
                    
                    return None
            else:
                process = await asyncio.create_subprocess_exec(
                    'git', *args,
                    cwd=self.project_root,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(
                        process.returncode,
                        f"git {' '.join(args)}",
                        stderr.decode()
                    )
                
                return stdout.decode()
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to run git command: {e}")
            raise
    
    async def _get_current_branch(self) -> str:
        """Get current branch name"""
        result = await self._run_git_command(['rev-parse', '--abbrev-ref', 'HEAD'])
        return result.strip() if result else "main"
    
    async def _get_ahead_behind_count(self, branch: str) -> Tuple[int, int]:
        """Get ahead/behind count relative to upstream"""
        try:
            # Get upstream branch
            upstream = await self._run_git_command(['rev-parse', '--abbrev-ref', f'{branch}@{{upstream}}'])
            
            if not upstream:
                return 0, 0
            
            upstream = upstream.strip()
            
            # Get ahead count
            ahead_result = await self._run_git_command(['rev-list', '--count', f'{upstream}..{branch}'])
            ahead_count = int(ahead_result.strip()) if ahead_result else 0
            
            # Get behind count
            behind_result = await self._run_git_command(['rev-list', '--count', f'{branch}..{upstream}'])
            behind_count = int(behind_result.strip()) if behind_result else 0
            
            return ahead_count, behind_count
            
        except Exception:
            return 0, 0
    
    # Hook management
    def add_pre_commit_hook(self, hook: callable):
        """Add pre-commit hook"""
        self.pre_commit_hooks.append(hook)
    
    def add_post_commit_hook(self, hook: callable):
        """Add post-commit hook"""
        self.post_commit_hooks.append(hook)
    
    def add_pre_push_hook(self, hook: callable):
        """Add pre-push hook"""
        self.pre_push_hooks.append(hook)
    
    def add_post_push_hook(self, hook: callable):
        """Add post-push hook"""
        self.post_push_hooks.append(hook)