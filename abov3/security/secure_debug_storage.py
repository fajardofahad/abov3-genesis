"""
ABOV3 Genesis - Secure Debug Data Storage
Enterprise-grade encrypted storage and transmission for debug data
"""

import asyncio
import json
import logging
import gzip
import hashlib
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
import tempfile
import os
import struct

from .crypto_manager import CryptographyManager
from .audit_logger import SecurityAuditLogger


class DataClassification(Enum):
    """Data classification levels for debug data"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class StorageEncryptionLevel(Enum):
    """Storage encryption levels"""
    NONE = "none"
    STANDARD = "standard"      # AES-256
    HIGH = "high"             # AES-256 + key rotation
    MAXIMUM = "maximum"       # AES-256 + key rotation + additional layers


@dataclass
class DebugDataMetadata:
    """Metadata for debug data"""
    data_id: str
    session_id: str
    user_id: str
    data_type: str
    classification: DataClassification
    size_bytes: int
    created_at: datetime
    expires_at: Optional[datetime] = None
    encryption_level: StorageEncryptionLevel = StorageEncryptionLevel.STANDARD
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    checksum: Optional[str] = None
    compressed: bool = False
    location: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DebugDataRecord:
    """Complete debug data record with metadata and content"""
    metadata: DebugDataMetadata
    data: bytes
    encrypted: bool = True
    
    def __post_init__(self):
        # Calculate checksum if not provided
        if not self.metadata.checksum:
            self.metadata.checksum = hashlib.sha256(self.data).hexdigest()


class SecureDebugStorage:
    """
    Enterprise-grade secure storage system for debug data
    Provides encryption, compression, access control, and audit trails
    """
    
    def __init__(
        self,
        storage_dir: Path,
        crypto_manager: CryptographyManager,
        audit_logger: SecurityAuditLogger,
        max_storage_gb: int = 10,
        default_retention_days: int = 30
    ):
        self.storage_dir = storage_dir
        self.crypto_manager = crypto_manager
        self.audit_logger = audit_logger
        self.max_storage_bytes = max_storage_gb * 1024 * 1024 * 1024
        self.default_retention_days = default_retention_days
        
        # Storage organization
        self.data_dir = storage_dir / "debug_data"
        self.metadata_dir = storage_dir / "metadata"
        self.temp_dir = storage_dir / "temp"
        self.archive_dir = storage_dir / "archive"
        
        # Create directories
        for directory in [self.data_dir, self.metadata_dir, self.temp_dir, self.archive_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # In-memory caches
        self.metadata_cache: Dict[str, DebugDataMetadata] = {}
        self.access_cache: Dict[str, datetime] = {}
        
        # Storage metrics
        self.storage_metrics = {
            'total_records': 0,
            'total_size_bytes': 0,
            'encrypted_records': 0,
            'compressed_records': 0,
            'access_count_total': 0,
            'classification_counts': {level.value: 0 for level in DataClassification},
            'encryption_level_counts': {level.value: 0 for level in StorageEncryptionLevel}
        }
        
        # Configuration
        self.compression_threshold = 1024  # Compress data larger than 1KB
        self.auto_cleanup_interval = 3600  # 1 hour
        self.cache_ttl = 300  # 5 minutes
        
        # Setup logging
        self.logger = logging.getLogger('abov3.security.debug_storage')
        
        # Load existing metadata
        asyncio.create_task(self._load_existing_metadata())
        
        # Start background tasks
        self._cleanup_task = None
        self._start_background_tasks()
    
    async def store_debug_data(
        self,
        session_id: str,
        user_id: str,
        data_type: str,
        data: Union[str, bytes, Dict[str, Any]],
        classification: DataClassification = DataClassification.INTERNAL,
        encryption_level: StorageEncryptionLevel = StorageEncryptionLevel.STANDARD,
        retention_days: Optional[int] = None,
        tags: Optional[List[str]] = None,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store debug data with encryption and comprehensive metadata
        
        Args:
            session_id: Debug session identifier
            user_id: User identifier
            data_type: Type of debug data
            data: The data to store
            classification: Data classification level
            encryption_level: Encryption level to use
            retention_days: Days to retain data (None for default)
            tags: Optional tags for data organization
            custom_metadata: Optional custom metadata
            
        Returns:
            str: Data identifier for retrieval
        """
        try:
            # Generate unique data ID
            data_id = f"{session_id}_{data_type}_{int(time.time() * 1000000)}"
            
            # Convert data to bytes
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            elif isinstance(data, dict):
                data_bytes = json.dumps(data, ensure_ascii=False).encode('utf-8')
            elif isinstance(data, bytes):
                data_bytes = data
            else:
                data_bytes = str(data).encode('utf-8')
            
            # Compress if beneficial
            compressed = False
            if len(data_bytes) > self.compression_threshold:
                compressed_data = gzip.compress(data_bytes)
                if len(compressed_data) < len(data_bytes) * 0.9:  # Only if >10% reduction
                    data_bytes = compressed_data
                    compressed = True
            
            # Calculate expiration
            retention_days = retention_days or self.default_retention_days
            expires_at = datetime.now() + timedelta(days=retention_days)
            
            # Create metadata
            metadata = DebugDataMetadata(
                data_id=data_id,
                session_id=session_id,
                user_id=user_id,
                data_type=data_type,
                classification=classification,
                size_bytes=len(data_bytes),
                created_at=datetime.now(),
                expires_at=expires_at,
                encryption_level=encryption_level,
                compressed=compressed,
                tags=tags or [],
                custom_metadata=custom_metadata or {}
            )
            
            # Check storage limits
            if self.storage_metrics['total_size_bytes'] + len(data_bytes) > self.max_storage_bytes:
                await self._cleanup_old_data()
                if self.storage_metrics['total_size_bytes'] + len(data_bytes) > self.max_storage_bytes:
                    raise Exception("Storage limit exceeded")
            
            # Encrypt data based on level
            encrypted_data = await self._encrypt_data(data_bytes, encryption_level, data_id)
            
            # Store data to file
            data_file_path = self.data_dir / f"{data_id}.dat"
            with open(data_file_path, 'wb') as f:
                f.write(encrypted_data)
            
            metadata.location = str(data_file_path)
            
            # Store metadata
            await self._store_metadata(metadata)
            
            # Update caches and metrics
            self.metadata_cache[data_id] = metadata
            self._update_storage_metrics(metadata, added=True)
            
            # Audit storage event
            await self._audit_storage_event("debug_data_stored", {
                "data_id": data_id,
                "session_id": session_id,
                "user_id": user_id,
                "data_type": data_type,
                "classification": classification.value,
                "size_bytes": len(data_bytes),
                "encrypted": True,
                "compression": compressed,
                "retention_days": retention_days
            })
            
            self.logger.info(f"Debug data stored: {data_id} ({len(data_bytes)} bytes)")
            return data_id
            
        except Exception as e:
            self.logger.error(f"Failed to store debug data: {e}")
            await self._audit_storage_event("debug_data_store_failed", {
                "session_id": session_id,
                "user_id": user_id,
                "data_type": data_type,
                "error": str(e)
            })
            raise
    
    async def retrieve_debug_data(
        self,
        data_id: str,
        session_id: str,
        user_id: str,
        access_reason: str = "debug_operation"
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve debug data with access control and audit
        
        Args:
            data_id: Data identifier
            session_id: Current session identifier
            user_id: Current user identifier
            access_reason: Reason for accessing data
            
        Returns:
            Dict containing data and metadata, or None if not found/not allowed
        """
        try:
            # Get metadata
            metadata = await self._get_metadata(data_id)
            if not metadata:
                return None
            
            # Check expiration
            if metadata.expires_at and datetime.now() > metadata.expires_at:
                await self.delete_debug_data(data_id, "Data expired")
                return None
            
            # Basic access control - user can only access their own data or session data
            if metadata.user_id != user_id and metadata.session_id != session_id:
                await self._audit_storage_event("debug_data_access_denied", {
                    "data_id": data_id,
                    "requesting_user": user_id,
                    "requesting_session": session_id,
                    "data_owner": metadata.user_id,
                    "data_session": metadata.session_id,
                    "reason": "access_not_authorized"
                })
                return None
            
            # Load encrypted data
            data_file_path = Path(metadata.location)
            if not data_file_path.exists():
                self.logger.error(f"Data file not found: {data_file_path}")
                return None
            
            with open(data_file_path, 'rb') as f:
                encrypted_data = f.read()
            
            # Decrypt data
            decrypted_data = await self._decrypt_data(encrypted_data, metadata.encryption_level, data_id)
            
            # Decompress if needed
            if metadata.compressed:
                decrypted_data = gzip.decompress(decrypted_data)
            
            # Update access tracking
            metadata.access_count += 1
            metadata.last_accessed = datetime.now()
            await self._store_metadata(metadata)
            
            self.access_cache[data_id] = datetime.now()
            self.storage_metrics['access_count_total'] += 1
            
            # Audit access
            await self._audit_storage_event("debug_data_accessed", {
                "data_id": data_id,
                "user_id": user_id,
                "session_id": session_id,
                "access_reason": access_reason,
                "data_type": metadata.data_type,
                "classification": metadata.classification.value,
                "size_bytes": metadata.size_bytes
            })
            
            # Parse data based on type
            try:
                if metadata.data_type.endswith('_json') or 'json' in metadata.tags:
                    parsed_data = json.loads(decrypted_data.decode('utf-8'))
                else:
                    parsed_data = decrypted_data.decode('utf-8')
            except:
                parsed_data = decrypted_data  # Return raw bytes if parsing fails
            
            return {
                'data_id': data_id,
                'data': parsed_data,
                'metadata': {
                    'data_type': metadata.data_type,
                    'classification': metadata.classification.value,
                    'created_at': metadata.created_at.isoformat(),
                    'size_bytes': metadata.size_bytes,
                    'access_count': metadata.access_count,
                    'tags': metadata.tags,
                    'custom_metadata': metadata.custom_metadata
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve debug data {data_id}: {e}")
            await self._audit_storage_event("debug_data_retrieve_failed", {
                "data_id": data_id,
                "user_id": user_id,
                "session_id": session_id,
                "error": str(e)
            })
            return None
    
    async def delete_debug_data(self, data_id: str, reason: str = "User request") -> bool:
        """
        Securely delete debug data
        
        Args:
            data_id: Data identifier
            reason: Reason for deletion
            
        Returns:
            bool indicating success
        """
        try:
            # Get metadata
            metadata = await self._get_metadata(data_id)
            if not metadata:
                return False
            
            # Secure file deletion
            data_file_path = Path(metadata.location)
            if data_file_path.exists():
                # Overwrite file with random data before deletion
                file_size = data_file_path.stat().st_size
                with open(data_file_path, 'r+b') as f:
                    f.write(os.urandom(file_size))
                    f.flush()
                    os.fsync(f.fileno())
                
                # Delete file
                data_file_path.unlink()
            
            # Delete metadata
            metadata_file_path = self.metadata_dir / f"{data_id}.meta"
            if metadata_file_path.exists():
                metadata_file_path.unlink()
            
            # Update caches and metrics
            if data_id in self.metadata_cache:
                del self.metadata_cache[data_id]
            if data_id in self.access_cache:
                del self.access_cache[data_id]
            
            self._update_storage_metrics(metadata, added=False)
            
            # Audit deletion
            await self._audit_storage_event("debug_data_deleted", {
                "data_id": data_id,
                "reason": reason,
                "data_type": metadata.data_type,
                "classification": metadata.classification.value,
                "size_bytes": metadata.size_bytes
            })
            
            self.logger.info(f"Debug data deleted: {data_id}, reason: {reason}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete debug data {data_id}: {e}")
            return False
    
    async def query_debug_data(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        data_type: Optional[str] = None,
        classification: Optional[DataClassification] = None,
        tags: Optional[List[str]] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query debug data with filtering
        
        Args:
            session_id: Filter by session
            user_id: Filter by user
            data_type: Filter by data type
            classification: Filter by classification level
            tags: Filter by tags (AND operation)
            date_from: Filter by creation date (from)
            date_to: Filter by creation date (to)
            limit: Maximum number of results
            
        Returns:
            List of matching data metadata
        """
        try:
            results = []
            
            for data_id, metadata in self.metadata_cache.items():
                # Apply filters
                if session_id and metadata.session_id != session_id:
                    continue
                if user_id and metadata.user_id != user_id:
                    continue
                if data_type and metadata.data_type != data_type:
                    continue
                if classification and metadata.classification != classification:
                    continue
                if tags and not all(tag in metadata.tags for tag in tags):
                    continue
                if date_from and metadata.created_at < date_from:
                    continue
                if date_to and metadata.created_at > date_to:
                    continue
                
                # Check if data is still valid (not expired)
                if metadata.expires_at and datetime.now() > metadata.expires_at:
                    continue
                
                results.append({
                    'data_id': data_id,
                    'session_id': metadata.session_id,
                    'user_id': metadata.user_id,
                    'data_type': metadata.data_type,
                    'classification': metadata.classification.value,
                    'size_bytes': metadata.size_bytes,
                    'created_at': metadata.created_at.isoformat(),
                    'expires_at': metadata.expires_at.isoformat() if metadata.expires_at else None,
                    'access_count': metadata.access_count,
                    'last_accessed': metadata.last_accessed.isoformat() if metadata.last_accessed else None,
                    'tags': metadata.tags,
                    'compressed': metadata.compressed,
                    'encryption_level': metadata.encryption_level.value
                })
                
                if len(results) >= limit:
                    break
            
            # Sort by creation date (newest first)
            results.sort(key=lambda x: x['created_at'], reverse=True)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to query debug data: {e}")
            return []
    
    async def get_storage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics"""
        return {
            'metrics': self.storage_metrics.copy(),
            'storage_usage': {
                'total_size_gb': self.storage_metrics['total_size_bytes'] / (1024**3),
                'max_size_gb': self.max_storage_bytes / (1024**3),
                'usage_percent': (self.storage_metrics['total_size_bytes'] / self.max_storage_bytes) * 100
            },
            'data_distribution': {
                'by_classification': self.storage_metrics['classification_counts'],
                'by_encryption_level': self.storage_metrics['encryption_level_counts']
            },
            'cache_statistics': {
                'metadata_cache_size': len(self.metadata_cache),
                'access_cache_size': len(self.access_cache)
            },
            'cleanup_info': {
                'default_retention_days': self.default_retention_days,
                'auto_cleanup_interval': self.auto_cleanup_interval
            }
        }
    
    async def export_session_data(
        self,
        session_id: str,
        export_format: str = "json",
        include_metadata: bool = True
    ) -> Optional[bytes]:
        """
        Export all data for a debug session
        
        Args:
            session_id: Session to export
            export_format: Export format (json, binary)
            include_metadata: Whether to include metadata
            
        Returns:
            Exported data as bytes, or None if failed
        """
        try:
            # Query all data for session
            session_data = await self.query_debug_data(session_id=session_id, limit=1000)
            
            if not session_data:
                return None
            
            export_package = {
                'session_id': session_id,
                'export_timestamp': datetime.now().isoformat(),
                'data_count': len(session_data),
                'data': []
            }
            
            # Retrieve actual data for each record
            for record_info in session_data:
                data_record = await self.retrieve_debug_data(
                    record_info['data_id'],
                    session_id,
                    record_info['user_id'],
                    "session_export"
                )
                
                if data_record:
                    export_record = {
                        'data_id': data_record['data_id'],
                        'data': data_record['data']
                    }
                    
                    if include_metadata:
                        export_record['metadata'] = data_record['metadata']
                    
                    export_package['data'].append(export_record)
            
            # Serialize based on format
            if export_format.lower() == "json":
                export_bytes = json.dumps(export_package, ensure_ascii=False, indent=2).encode('utf-8')
            else:
                # Binary format (pickle-like, but secure)
                export_bytes = json.dumps(export_package).encode('utf-8')
            
            # Audit export
            await self._audit_storage_event("session_data_exported", {
                "session_id": session_id,
                "data_count": len(session_data),
                "export_format": export_format,
                "export_size_bytes": len(export_bytes)
            })
            
            return gzip.compress(export_bytes)  # Always compress exports
            
        except Exception as e:
            self.logger.error(f"Failed to export session data {session_id}: {e}")
            return None
    
    async def _encrypt_data(
        self,
        data: bytes,
        encryption_level: StorageEncryptionLevel,
        data_id: str
    ) -> bytes:
        """Encrypt data based on encryption level"""
        if encryption_level == StorageEncryptionLevel.NONE:
            return data
        
        # Generate data-specific key
        data_key = await self.crypto_manager.generate_data_key()
        
        # Encrypt data
        encrypted_data = await self.crypto_manager.encrypt_data(data, data_key['key'])
        
        # For higher security levels, add additional layers
        if encryption_level in [StorageEncryptionLevel.HIGH, StorageEncryptionLevel.MAXIMUM]:
            # Add timestamp-based nonce
            timestamp = struct.pack('>Q', int(time.time()))
            encrypted_data = timestamp + encrypted_data
            
            # Re-encrypt with master key
            encrypted_data = await self.crypto_manager.encrypt_data(encrypted_data)
        
        return encrypted_data
    
    async def _decrypt_data(
        self,
        encrypted_data: bytes,
        encryption_level: StorageEncryptionLevel,
        data_id: str
    ) -> bytes:
        """Decrypt data based on encryption level"""
        if encryption_level == StorageEncryptionLevel.NONE:
            return encrypted_data
        
        # For higher security levels, handle additional layers
        if encryption_level in [StorageEncryptionLevel.HIGH, StorageEncryptionLevel.MAXIMUM]:
            # Decrypt with master key first
            decrypted_data = await self.crypto_manager.decrypt_data(encrypted_data)
            
            # Remove timestamp
            if len(decrypted_data) >= 8:
                decrypted_data = decrypted_data[8:]
        else:
            decrypted_data = encrypted_data
        
        # Decrypt with data key
        # Note: In a full implementation, data keys would be stored securely
        # For now, we'll use the master key
        return await self.crypto_manager.decrypt_data(decrypted_data)
    
    async def _store_metadata(self, metadata: DebugDataMetadata):
        """Store metadata to file"""
        metadata_file = self.metadata_dir / f"{metadata.data_id}.meta"
        
        metadata_dict = {
            'data_id': metadata.data_id,
            'session_id': metadata.session_id,
            'user_id': metadata.user_id,
            'data_type': metadata.data_type,
            'classification': metadata.classification.value,
            'size_bytes': metadata.size_bytes,
            'created_at': metadata.created_at.isoformat(),
            'expires_at': metadata.expires_at.isoformat() if metadata.expires_at else None,
            'encryption_level': metadata.encryption_level.value,
            'access_count': metadata.access_count,
            'last_accessed': metadata.last_accessed.isoformat() if metadata.last_accessed else None,
            'checksum': metadata.checksum,
            'compressed': metadata.compressed,
            'location': metadata.location,
            'tags': metadata.tags,
            'custom_metadata': metadata.custom_metadata
        }
        
        # Encrypt metadata for sensitive data
        if metadata.classification in [DataClassification.RESTRICTED, DataClassification.TOP_SECRET]:
            encrypted_metadata = await self.crypto_manager.encrypt_data(
                json.dumps(metadata_dict).encode('utf-8')
            )
            with open(metadata_file, 'wb') as f:
                f.write(encrypted_metadata)
        else:
            with open(metadata_file, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
    
    async def _get_metadata(self, data_id: str) -> Optional[DebugDataMetadata]:
        """Get metadata from cache or file"""
        # Check cache first
        if data_id in self.metadata_cache:
            return self.metadata_cache[data_id]
        
        # Load from file
        metadata_file = self.metadata_dir / f"{data_id}.meta"
        if not metadata_file.exists():
            return None
        
        try:
            # Try to load as encrypted first
            with open(metadata_file, 'rb') as f:
                encrypted_data = f.read()
            
            try:
                # Try to decrypt
                decrypted_data = await self.crypto_manager.decrypt_data(encrypted_data)
                metadata_dict = json.loads(decrypted_data.decode('utf-8'))
            except:
                # Not encrypted, load as plain text
                with open(metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
            
            # Convert back to dataclass
            metadata = DebugDataMetadata(
                data_id=metadata_dict['data_id'],
                session_id=metadata_dict['session_id'],
                user_id=metadata_dict['user_id'],
                data_type=metadata_dict['data_type'],
                classification=DataClassification(metadata_dict['classification']),
                size_bytes=metadata_dict['size_bytes'],
                created_at=datetime.fromisoformat(metadata_dict['created_at']),
                expires_at=datetime.fromisoformat(metadata_dict['expires_at']) if metadata_dict['expires_at'] else None,
                encryption_level=StorageEncryptionLevel(metadata_dict['encryption_level']),
                access_count=metadata_dict.get('access_count', 0),
                last_accessed=datetime.fromisoformat(metadata_dict['last_accessed']) if metadata_dict.get('last_accessed') else None,
                checksum=metadata_dict.get('checksum'),
                compressed=metadata_dict.get('compressed', False),
                location=metadata_dict.get('location'),
                tags=metadata_dict.get('tags', []),
                custom_metadata=metadata_dict.get('custom_metadata', {})
            )
            
            # Cache for future use
            self.metadata_cache[data_id] = metadata
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load metadata for {data_id}: {e}")
            return None
    
    async def _load_existing_metadata(self):
        """Load all existing metadata into cache"""
        try:
            for metadata_file in self.metadata_dir.glob("*.meta"):
                data_id = metadata_file.stem
                metadata = await self._get_metadata(data_id)
                if metadata:
                    self._update_storage_metrics(metadata, added=True)
        except Exception as e:
            self.logger.error(f"Failed to load existing metadata: {e}")
    
    def _update_storage_metrics(self, metadata: DebugDataMetadata, added: bool):
        """Update storage metrics"""
        multiplier = 1 if added else -1
        
        self.storage_metrics['total_records'] += multiplier
        self.storage_metrics['total_size_bytes'] += multiplier * metadata.size_bytes
        
        if metadata.encryption_level != StorageEncryptionLevel.NONE:
            self.storage_metrics['encrypted_records'] += multiplier
        
        if metadata.compressed:
            self.storage_metrics['compressed_records'] += multiplier
        
        self.storage_metrics['classification_counts'][metadata.classification.value] += multiplier
        self.storage_metrics['encryption_level_counts'][metadata.encryption_level.value] += multiplier
        
        # Ensure non-negative values
        for key, value in self.storage_metrics.items():
            if isinstance(value, int) and value < 0:
                self.storage_metrics[key] = 0
    
    async def _cleanup_old_data(self):
        """Clean up expired and old data"""
        current_time = datetime.now()
        deleted_count = 0
        freed_bytes = 0
        
        # Find expired data
        expired_data_ids = []
        for data_id, metadata in self.metadata_cache.items():
            if metadata.expires_at and current_time > metadata.expires_at:
                expired_data_ids.append(data_id)
        
        # Delete expired data
        for data_id in expired_data_ids:
            metadata = self.metadata_cache.get(data_id)
            if metadata:
                freed_bytes += metadata.size_bytes
                await self.delete_debug_data(data_id, "Data expired")
                deleted_count += 1
        
        # If still over limit, delete oldest data
        if self.storage_metrics['total_size_bytes'] > self.max_storage_bytes:
            # Sort by last accessed (or created if never accessed)
            data_by_age = sorted(
                self.metadata_cache.items(),
                key=lambda x: x[1].last_accessed or x[1].created_at
            )
            
            for data_id, metadata in data_by_age:
                if self.storage_metrics['total_size_bytes'] <= self.max_storage_bytes * 0.8:
                    break
                
                freed_bytes += metadata.size_bytes
                await self.delete_debug_data(data_id, "Storage cleanup - oldest data")
                deleted_count += 1
        
        if deleted_count > 0:
            await self._audit_storage_event("storage_cleanup", {
                "deleted_count": deleted_count,
                "freed_bytes": freed_bytes,
                "reason": "automated_cleanup"
            })
            
            self.logger.info(f"Storage cleanup: deleted {deleted_count} records, freed {freed_bytes} bytes")
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        async def cleanup_loop():
            while True:
                try:
                    await self._cleanup_old_data()
                    await asyncio.sleep(self.auto_cleanup_interval)
                except Exception as e:
                    self.logger.error(f"Cleanup task error: {e}")
                    await asyncio.sleep(300)  # Wait 5 minutes before retry
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def _audit_storage_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log storage event to audit system"""
        if self.audit_logger:
            await self.audit_logger.log_event(event_type, {
                'component': 'secure_debug_storage',
                'timestamp': datetime.now().isoformat(),
                **event_data
            })
    
    async def shutdown(self):
        """Shutdown storage system"""
        try:
            # Cancel background tasks
            if self._cleanup_task:
                self._cleanup_task.cancel()
            
            # Force final cleanup
            await self._cleanup_old_data()
            
            # Clear caches
            self.metadata_cache.clear()
            self.access_cache.clear()
            
            self.logger.info("Secure debug storage shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Storage shutdown error: {e}")