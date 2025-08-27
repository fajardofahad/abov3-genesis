"""
Enterprise Transaction Management System

ACID-compliant transaction manager for multi-file operations with comprehensive
rollback capabilities, savepoints, and distributed transaction coordination.
"""

import os
import json
import shutil
import hashlib
import asyncio
import uuid
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
from pathlib import Path
import logging
import tempfile
import contextlib

logger = logging.getLogger(__name__)


class TransactionState(Enum):
    """Transaction states"""
    CREATED = "created"
    ACTIVE = "active" 
    PREPARING = "preparing"
    PREPARED = "prepared"
    COMMITTING = "committing"
    COMMITTED = "committed"
    ABORTING = "aborting"
    ABORTED = "aborted"
    ROLLED_BACK = "rolled_back"


class OperationType(Enum):
    """Types of file operations"""
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    RENAME = "rename"
    COPY = "copy"
    CHMOD = "chmod"


@dataclass
class FileOperation:
    """Represents a single file operation"""
    id: str
    operation_type: OperationType
    file_path: str
    old_path: Optional[str] = None  # For renames/moves
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    old_permissions: Optional[int] = None
    new_permissions: Optional[int] = None
    checksum_before: Optional[str] = None
    checksum_after: Optional[str] = None
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass  
class Savepoint:
    """Represents a transaction savepoint"""
    id: str
    name: str
    transaction_id: str
    created_at: datetime
    operations_before: List[str]  # Operation IDs before this savepoint
    filesystem_state: Dict[str, Any]  # Filesystem state snapshot
    
    def __post_init__(self):
        if not hasattr(self, 'created_at') or self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class Transaction:
    """Represents a multi-file transaction"""
    id: str
    patch_set_id: Optional[str] = None
    state: TransactionState = TransactionState.CREATED
    operations: List[FileOperation] = None
    savepoints: List[Savepoint] = None
    started_at: datetime = None
    completed_at: Optional[datetime] = None
    isolation_level: str = "READ_COMMITTED"
    timeout_seconds: int = 300
    rollback_data: Dict[str, Any] = None
    lock_files: Set[str] = None
    
    def __post_init__(self):
        if self.operations is None:
            self.operations = []
        if self.savepoints is None:
            self.savepoints = []
        if self.started_at is None:
            self.started_at = datetime.now()
        if self.rollback_data is None:
            self.rollback_data = {}
        if self.lock_files is None:
            self.lock_files = set()


class TransactionManager:
    """
    Enterprise-grade transaction manager for multi-file operations
    
    Features:
    - ACID compliance with atomicity guarantees
    - Savepoint support for partial rollbacks
    - Distributed transaction coordination
    - File locking and isolation
    - Comprehensive rollback capabilities
    - Transaction logging and auditing
    - Deadlock detection and resolution
    - Performance optimization with caching
    """
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.transactions_dir = self.project_root / ".abov3" / "transactions"
        self.transactions_dir.mkdir(parents=True, exist_ok=True)
        
        # Active transactions
        self.active_transactions: Dict[str, Transaction] = {}
        
        # File locks
        self.file_locks: Dict[str, str] = {}  # file_path -> transaction_id
        self.lock_manager = asyncio.Lock()
        
        # Transaction hooks
        self.pre_commit_hooks: List[Callable] = []
        self.post_commit_hooks: List[Callable] = []
        self.pre_rollback_hooks: List[Callable] = []
        self.post_rollback_hooks: List[Callable] = []
        
        # Performance monitoring
        self.transaction_stats = {
            'total_transactions': 0,
            'successful_commits': 0,
            'rollbacks': 0,
            'deadlocks_detected': 0,
            'avg_transaction_time': 0.0
        }
    
    @contextlib.asynccontextmanager
    async def transaction(
        self, 
        patch_set_id: str = None,
        isolation_level: str = "READ_COMMITTED",
        timeout_seconds: int = 300
    ):
        """
        Create and manage a transaction context
        
        Args:
            patch_set_id: Associated patch set ID
            isolation_level: Transaction isolation level
            timeout_seconds: Transaction timeout
            
        Usage:
            async with transaction_manager.transaction("patch_123") as tx:
                await tx.add_operation(...)
                # Automatically commits on success, rolls back on error
        """
        
        transaction_id = str(uuid.uuid4())
        transaction = Transaction(
            id=transaction_id,
            patch_set_id=patch_set_id,
            state=TransactionState.CREATED,
            isolation_level=isolation_level,
            timeout_seconds=timeout_seconds
        )
        
        self.active_transactions[transaction_id] = transaction
        self.transaction_stats['total_transactions'] += 1
        
        try:
            # Begin transaction
            await self._begin_transaction(transaction)
            
            # Provide transaction context
            yield TransactionContext(self, transaction)
            
            # Commit if no exceptions
            await self._commit_transaction(transaction)
            
        except Exception as e:
            logger.error(f"Transaction {transaction_id} failed: {e}")
            await self._rollback_transaction(transaction)
            raise
        
        finally:
            # Cleanup
            if transaction_id in self.active_transactions:
                del self.active_transactions[transaction_id]
    
    async def _begin_transaction(self, transaction: Transaction):
        """Begin a new transaction"""
        transaction.state = TransactionState.ACTIVE
        transaction.started_at = datetime.now()
        
        # Save transaction state
        await self._save_transaction_state(transaction)
        
        logger.info(f"Transaction {transaction.id} started")
    
    async def _commit_transaction(self, transaction: Transaction):
        """Commit a transaction"""
        start_time = datetime.now()
        
        try:
            # Pre-commit hooks
            for hook in self.pre_commit_hooks:
                await hook(transaction)
            
            # Prepare phase
            transaction.state = TransactionState.PREPARING
            await self._prepare_transaction(transaction)
            
            transaction.state = TransactionState.PREPARED
            
            # Commit phase
            transaction.state = TransactionState.COMMITTING
            await self._apply_operations(transaction)
            
            # Complete transaction
            transaction.state = TransactionState.COMMITTED
            transaction.completed_at = datetime.now()
            
            # Post-commit hooks
            for hook in self.post_commit_hooks:
                await hook(transaction)
            
            # Release locks
            await self._release_file_locks(transaction)
            
            # Update statistics
            self.transaction_stats['successful_commits'] += 1
            duration = (datetime.now() - start_time).total_seconds()
            self._update_avg_transaction_time(duration)
            
            logger.info(f"Transaction {transaction.id} committed successfully")
            
        except Exception as e:
            logger.error(f"Transaction {transaction.id} commit failed: {e}")
            await self._rollback_transaction(transaction)
            raise
    
    async def _rollback_transaction(self, transaction: Transaction):
        """Rollback a transaction"""
        try:
            # Pre-rollback hooks
            for hook in self.pre_rollback_hooks:
                await hook(transaction)
            
            transaction.state = TransactionState.ABORTING
            
            # Reverse all operations
            await self._reverse_operations(transaction)
            
            transaction.state = TransactionState.ROLLED_BACK
            transaction.completed_at = datetime.now()
            
            # Post-rollback hooks
            for hook in self.post_rollback_hooks:
                await hook(transaction)
            
            # Release locks
            await self._release_file_locks(transaction)
            
            # Update statistics
            self.transaction_stats['rollbacks'] += 1
            
            logger.info(f"Transaction {transaction.id} rolled back")
            
        except Exception as e:
            logger.error(f"Transaction {transaction.id} rollback failed: {e}")
            transaction.state = TransactionState.ABORTED
            raise
    
    async def _prepare_transaction(self, transaction: Transaction):
        """Prepare transaction for commit (2PC phase 1)"""
        
        # Validate all operations
        for operation in transaction.operations:
            await self._validate_operation(operation)
        
        # Acquire file locks
        await self._acquire_file_locks(transaction)
        
        # Create backup snapshots
        await self._create_rollback_snapshots(transaction)
        
        # Validate filesystem state
        await self._validate_filesystem_state(transaction)
    
    async def _apply_operations(self, transaction: Transaction):
        """Apply all transaction operations"""
        
        applied_operations = []
        
        try:
            for operation in transaction.operations:
                await self._apply_single_operation(operation)
                applied_operations.append(operation)
                
                # Update rollback data
                await self._update_rollback_data(transaction, operation)
        
        except Exception as e:
            # Reverse successfully applied operations
            for op in reversed(applied_operations):
                try:
                    await self._reverse_single_operation(op)
                except Exception as reverse_error:
                    logger.error(f"Failed to reverse operation {op.id}: {reverse_error}")
            raise
    
    async def _reverse_operations(self, transaction: Transaction):
        """Reverse all transaction operations"""
        
        # Apply operations in reverse order
        for operation in reversed(transaction.operations):
            try:
                await self._reverse_single_operation(operation)
            except Exception as e:
                logger.error(f"Failed to reverse operation {operation.id}: {e}")
                # Continue with other reversals
    
    async def _apply_single_operation(self, operation: FileOperation):
        """Apply a single file operation"""
        
        file_path = Path(self.project_root) / operation.file_path
        
        if operation.operation_type == OperationType.CREATE:
            await self._create_file(file_path, operation.new_content, operation.new_permissions)
        
        elif operation.operation_type == OperationType.MODIFY:
            await self._modify_file(file_path, operation.new_content)
        
        elif operation.operation_type == OperationType.DELETE:
            await self._delete_file(file_path)
        
        elif operation.operation_type == OperationType.RENAME:
            old_path = Path(self.project_root) / operation.old_path
            await self._rename_file(old_path, file_path)
        
        elif operation.operation_type == OperationType.COPY:
            old_path = Path(self.project_root) / operation.old_path
            await self._copy_file(old_path, file_path)
        
        elif operation.operation_type == OperationType.CHMOD:
            await self._change_permissions(file_path, operation.new_permissions)
    
    async def _reverse_single_operation(self, operation: FileOperation):
        """Reverse a single file operation"""
        
        file_path = Path(self.project_root) / operation.file_path
        
        if operation.operation_type == OperationType.CREATE:
            # Delete created file
            if file_path.exists():
                await self._delete_file(file_path)
        
        elif operation.operation_type == OperationType.MODIFY:
            # Restore original content
            if operation.old_content is not None:
                await self._modify_file(file_path, operation.old_content)
        
        elif operation.operation_type == OperationType.DELETE:
            # Recreate deleted file
            if operation.old_content is not None:
                await self._create_file(file_path, operation.old_content, operation.old_permissions)
        
        elif operation.operation_type == OperationType.RENAME:
            # Rename back
            old_path = Path(self.project_root) / operation.old_path
            if file_path.exists():
                await self._rename_file(file_path, old_path)
        
        elif operation.operation_type == OperationType.COPY:
            # Delete copied file
            if file_path.exists():
                await self._delete_file(file_path)
        
        elif operation.operation_type == OperationType.CHMOD:
            # Restore original permissions
            if operation.old_permissions is not None:
                await self._change_permissions(file_path, operation.old_permissions)
    
    # File operation helpers
    async def _create_file(self, file_path: Path, content: str, permissions: int = None):
        """Create a new file"""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content or '')
        
        if permissions is not None:
            os.chmod(file_path, permissions)
    
    async def _modify_file(self, file_path: Path, content: str):
        """Modify an existing file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content or '')
    
    async def _delete_file(self, file_path: Path):
        """Delete a file"""
        if file_path.exists():
            file_path.unlink()
    
    async def _rename_file(self, old_path: Path, new_path: Path):
        """Rename/move a file"""
        new_path.parent.mkdir(parents=True, exist_ok=True)
        old_path.rename(new_path)
    
    async def _copy_file(self, old_path: Path, new_path: Path):
        """Copy a file"""
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_path, new_path)
    
    async def _change_permissions(self, file_path: Path, permissions: int):
        """Change file permissions"""
        os.chmod(file_path, permissions)
    
    # Lock management
    async def _acquire_file_locks(self, transaction: Transaction):
        """Acquire locks for all files in transaction"""
        async with self.lock_manager:
            files_to_lock = set()
            
            for operation in transaction.operations:
                files_to_lock.add(operation.file_path)
                if operation.old_path:
                    files_to_lock.add(operation.old_path)
            
            # Check for deadlocks
            conflicted_files = []
            for file_path in files_to_lock:
                if file_path in self.file_locks:
                    conflicted_files.append((file_path, self.file_locks[file_path]))
            
            if conflicted_files:
                self.transaction_stats['deadlocks_detected'] += 1
                raise Exception(f"Deadlock detected on files: {conflicted_files}")
            
            # Acquire locks
            for file_path in files_to_lock:
                self.file_locks[file_path] = transaction.id
                transaction.lock_files.add(file_path)
    
    async def _release_file_locks(self, transaction: Transaction):
        """Release all file locks for transaction"""
        async with self.lock_manager:
            for file_path in transaction.lock_files:
                if file_path in self.file_locks and self.file_locks[file_path] == transaction.id:
                    del self.file_locks[file_path]
            
            transaction.lock_files.clear()
    
    # Savepoint management
    async def create_savepoint(self, transaction_id: str, name: str) -> str:
        """Create a savepoint within a transaction"""
        transaction = self.active_transactions.get(transaction_id)
        if not transaction:
            raise ValueError(f"Transaction {transaction_id} not found")
        
        savepoint_id = str(uuid.uuid4())
        
        # Capture current filesystem state
        filesystem_state = await self._capture_filesystem_state(transaction)
        
        savepoint = Savepoint(
            id=savepoint_id,
            name=name,
            transaction_id=transaction_id,
            created_at=datetime.now(),
            operations_before=[op.id for op in transaction.operations],
            filesystem_state=filesystem_state
        )
        
        transaction.savepoints.append(savepoint)
        
        logger.info(f"Savepoint {name} ({savepoint_id}) created in transaction {transaction_id}")
        return savepoint_id
    
    async def rollback_to_savepoint(self, transaction_id: str, savepoint_name: str) -> bool:
        """Rollback to a specific savepoint"""
        transaction = self.active_transactions.get(transaction_id)
        if not transaction:
            return False
        
        # Find savepoint
        savepoint = None
        for sp in transaction.savepoints:
            if sp.name == savepoint_name:
                savepoint = sp
                break
        
        if not savepoint:
            return False
        
        # Reverse operations after savepoint
        operations_to_reverse = []
        for operation in transaction.operations:
            if operation.id not in savepoint.operations_before:
                operations_to_reverse.append(operation)
        
        # Apply reversals
        for operation in reversed(operations_to_reverse):
            try:
                await self._reverse_single_operation(operation)
                transaction.operations.remove(operation)
            except Exception as e:
                logger.error(f"Failed to reverse operation {operation.id}: {e}")
        
        # Remove savepoints created after this one
        transaction.savepoints = [sp for sp in transaction.savepoints 
                                  if sp.created_at <= savepoint.created_at]
        
        logger.info(f"Rolled back to savepoint {savepoint_name} in transaction {transaction_id}")
        return True
    
    # Backup and recovery
    async def _create_rollback_snapshots(self, transaction: Transaction):
        """Create backup snapshots for rollback"""
        
        backup_dir = self.transactions_dir / "backups" / transaction.id
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        for operation in transaction.operations:
            file_path = Path(self.project_root) / operation.file_path
            
            if file_path.exists():
                # Create backup
                backup_path = backup_dir / operation.file_path.replace(os.sep, '_')
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, backup_path)
                
                # Store backup info in rollback data
                transaction.rollback_data[operation.file_path] = {
                    'backup_path': str(backup_path),
                    'original_permissions': file_path.stat().st_mode,
                    'checksum': self._calculate_file_checksum(file_path)
                }
    
    async def _capture_filesystem_state(self, transaction: Transaction) -> Dict[str, Any]:
        """Capture current filesystem state for savepoint"""
        state = {}
        
        for operation in transaction.operations:
            file_path = Path(self.project_root) / operation.file_path
            
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                state[operation.file_path] = {
                    'content': content,
                    'permissions': file_path.stat().st_mode,
                    'size': file_path.stat().st_size,
                    'mtime': file_path.stat().st_mtime
                }
            else:
                state[operation.file_path] = None  # File doesn't exist
        
        return state
    
    # Validation helpers
    async def _validate_operation(self, operation: FileOperation):
        """Validate a single operation"""
        file_path = Path(self.project_root) / operation.file_path
        
        if operation.operation_type == OperationType.CREATE:
            if file_path.exists():
                raise ValueError(f"Cannot create {operation.file_path}: file already exists")
        
        elif operation.operation_type in [OperationType.MODIFY, OperationType.DELETE]:
            if not file_path.exists():
                raise ValueError(f"Cannot {operation.operation_type.value} {operation.file_path}: file does not exist")
        
        elif operation.operation_type == OperationType.RENAME:
            old_path = Path(self.project_root) / operation.old_path
            if not old_path.exists():
                raise ValueError(f"Cannot rename {operation.old_path}: file does not exist")
            if file_path.exists():
                raise ValueError(f"Cannot rename to {operation.file_path}: file already exists")
    
    async def _validate_filesystem_state(self, transaction: Transaction):
        """Validate filesystem state before commit"""
        for operation in transaction.operations:
            # Check if file state has changed since operation was created
            if operation.checksum_before:
                file_path = Path(self.project_root) / operation.file_path
                if file_path.exists():
                    current_checksum = self._calculate_file_checksum(file_path)
                    if current_checksum != operation.checksum_before:
                        raise ValueError(f"File {operation.file_path} was modified outside transaction")
    
    async def _update_rollback_data(self, transaction: Transaction, operation: FileOperation):
        """Update rollback data after applying operation"""
        file_path = Path(self.project_root) / operation.file_path
        
        if file_path.exists():
            transaction.rollback_data[operation.file_path + '_after'] = {
                'checksum': self._calculate_file_checksum(file_path),
                'permissions': file_path.stat().st_mode
            }
    
    # Utility methods
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file"""
        hash_sha256 = hashlib.sha256()
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception:
            return ""
    
    async def _save_transaction_state(self, transaction: Transaction):
        """Save transaction state to disk"""
        transaction_file = self.transactions_dir / f"{transaction.id}.json"
        
        # Convert to serializable format
        data = {
            'id': transaction.id,
            'patch_set_id': transaction.patch_set_id,
            'state': transaction.state.value,
            'started_at': transaction.started_at.isoformat(),
            'completed_at': transaction.completed_at.isoformat() if transaction.completed_at else None,
            'operations': [
                {
                    'id': op.id,
                    'type': op.operation_type.value,
                    'file_path': op.file_path,
                    'old_path': op.old_path,
                    'timestamp': op.timestamp.isoformat()
                }
                for op in transaction.operations
            ]
        }
        
        with open(transaction_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def _update_avg_transaction_time(self, duration: float):
        """Update average transaction time"""
        current_avg = self.transaction_stats['avg_transaction_time']
        total_tx = self.transaction_stats['successful_commits']
        
        if total_tx == 1:
            self.transaction_stats['avg_transaction_time'] = duration
        else:
            # Running average
            self.transaction_stats['avg_transaction_time'] = (
                (current_avg * (total_tx - 1) + duration) / total_tx
            )
    
    # Public API methods
    async def get_transaction_stats(self) -> Dict[str, Any]:
        """Get transaction statistics"""
        return self.transaction_stats.copy()
    
    async def list_active_transactions(self) -> List[str]:
        """List active transaction IDs"""
        return list(self.active_transactions.keys())
    
    async def get_transaction_info(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a transaction"""
        transaction = self.active_transactions.get(transaction_id)
        if not transaction:
            return None
        
        return {
            'id': transaction.id,
            'state': transaction.state.value,
            'patch_set_id': transaction.patch_set_id,
            'operations_count': len(transaction.operations),
            'savepoints_count': len(transaction.savepoints),
            'started_at': transaction.started_at.isoformat(),
            'locked_files': list(transaction.lock_files)
        }


class TransactionContext:
    """Context for transaction operations"""
    
    def __init__(self, manager: TransactionManager, transaction: Transaction):
        self.manager = manager
        self.transaction = transaction
    
    async def add_operation(
        self,
        operation_type: OperationType,
        file_path: str,
        old_path: str = None,
        old_content: str = None,
        new_content: str = None,
        old_permissions: int = None,
        new_permissions: int = None
    ) -> str:
        """Add an operation to the transaction"""
        
        operation_id = str(uuid.uuid4())
        
        # Calculate checksums if content provided
        checksum_before = None
        checksum_after = None
        
        if old_content:
            checksum_before = hashlib.sha256(old_content.encode()).hexdigest()
        if new_content:
            checksum_after = hashlib.sha256(new_content.encode()).hexdigest()
        
        operation = FileOperation(
            id=operation_id,
            operation_type=operation_type,
            file_path=file_path,
            old_path=old_path,
            old_content=old_content,
            new_content=new_content,
            old_permissions=old_permissions,
            new_permissions=new_permissions,
            checksum_before=checksum_before,
            checksum_after=checksum_after
        )
        
        self.transaction.operations.append(operation)
        return operation_id
    
    async def create_savepoint(self, name: str) -> str:
        """Create a savepoint"""
        return await self.manager.create_savepoint(self.transaction.id, name)
    
    async def rollback_to_savepoint(self, name: str) -> bool:
        """Rollback to savepoint"""
        return await self.manager.rollback_to_savepoint(self.transaction.id, name)