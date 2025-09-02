"""
ABOV3 Genesis - Advanced Memory Management System
Claude-level context management with intelligent compression and retrieval
"""

import asyncio
import json
import hashlib
import pickle
import zlib
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import sqlite3
import os
import sys

class MemoryType(Enum):
    """Types of memory storage"""
    SESSION = "session"           # Current session memory
    PERSISTENT = "persistent"     # Long-term storage
    PROJECT = "project"          # Project-specific memory
    CODE_CONTEXT = "code_context" # Code understanding context
    CONVERSATION = "conversation" # Conversation history
    SYSTEM = "system"            # System state memory
    LEARNING = "learning"        # Learning and adaptation

class Priority(Enum):
    """Memory priority levels"""
    CRITICAL = 5    # Never compress or remove
    HIGH = 4       # Compress only when necessary  
    MEDIUM = 3     # Standard compression
    LOW = 2        # Aggressive compression
    TEMP = 1       # Can be deleted

@dataclass
class MemoryEntry:
    """Individual memory entry"""
    id: str
    content: Any
    memory_type: MemoryType
    priority: Priority
    timestamp: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    tags: Set[str] = field(default_factory=set)
    size_bytes: int = 0
    compressed: bool = False
    context_hash: str = ""
    related_entries: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContextWindow:
    """Context window for managing large token contexts"""
    entries: List[str] = field(default_factory=list)
    total_tokens: int = 0
    max_tokens: int = 128000  # Claude-3.5 context size
    compression_ratio: float = 0.0
    last_compression: datetime = field(default_factory=datetime.now)

class MemoryManager:
    """
    Advanced memory management system with Claude-level intelligence.
    Handles context windows, compression, retrieval, and session persistence.
    """
    
    def __init__(self, project_path: Path, max_memory_mb: int = 500):
        self.project_path = project_path
        self.memory_dir = project_path / '.abov3' / 'memory'
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Memory storage
        self.memory_store: Dict[str, MemoryEntry] = {}
        self.type_indices: Dict[MemoryType, Set[str]] = defaultdict(set)
        self.tag_indices: Dict[str, Set[str]] = defaultdict(set)
        self.context_windows: Dict[str, ContextWindow] = {}
        
        # Configuration
        self.max_memory_mb = max_memory_mb
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_memory_usage = 0
        
        # Context management
        self.current_context = ContextWindow()
        self.context_history: deque = deque(maxlen=100)
        
        # Performance tracking
        self.metrics = {
            'entries_stored': 0,
            'entries_retrieved': 0,
            'compressions_performed': 0,
            'memory_usage_bytes': 0,
            'average_retrieval_time': 0.0,
            'context_compressions': 0,
            'session_start': datetime.now()
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Database for persistent storage
        self.db_path = self.memory_dir / 'persistent_memory.db'
        self._initialize_database()
        
        # Load session memory
        self._load_session_memory()
        
        # Background compression thread
        self.compression_thread = threading.Thread(
            target=self._background_compression_loop,
            daemon=True
        )
        self.compression_thread.start()
    
    def _initialize_database(self):
        """Initialize SQLite database for persistent storage"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS memory_entries (
                        id TEXT PRIMARY KEY,
                        content BLOB,
                        memory_type TEXT,
                        priority INTEGER,
                        timestamp REAL,
                        access_count INTEGER,
                        last_accessed REAL,
                        tags TEXT,
                        size_bytes INTEGER,
                        compressed BOOLEAN,
                        context_hash TEXT,
                        related_entries TEXT,
                        metadata TEXT
                    )
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_entries(memory_type)
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_timestamp ON memory_entries(timestamp)
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_context_hash ON memory_entries(context_hash)
                ''')
                
                conn.commit()
        except Exception as e:
            print(f"Warning: Could not initialize memory database: {e}")
    
    def _load_session_memory(self):
        """Load session memory from previous sessions"""
        session_file = self.memory_dir / 'session_memory.json'
        if session_file.exists():
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                
                for entry_data in session_data.get('entries', []):
                    entry = MemoryEntry(
                        id=entry_data['id'],
                        content=entry_data['content'],
                        memory_type=MemoryType(entry_data['memory_type']),
                        priority=Priority(entry_data['priority']),
                        timestamp=datetime.fromisoformat(entry_data['timestamp']),
                        access_count=entry_data.get('access_count', 0),
                        last_accessed=datetime.fromisoformat(entry_data.get('last_accessed', entry_data['timestamp'])),
                        tags=set(entry_data.get('tags', [])),
                        size_bytes=entry_data.get('size_bytes', 0),
                        compressed=entry_data.get('compressed', False),
                        context_hash=entry_data.get('context_hash', ''),
                        related_entries=set(entry_data.get('related_entries', [])),
                        metadata=entry_data.get('metadata', {})
                    )
                    self.memory_store[entry.id] = entry
                    self._update_indices(entry)
                
            except Exception as e:
                print(f"Warning: Could not load session memory: {e}")
    
    def _save_session_memory(self):
        """Save session memory to disk"""
        session_file = self.memory_dir / 'session_memory.json'
        try:
            session_entries = []
            for entry in self.memory_store.values():
                if entry.memory_type == MemoryType.SESSION:
                    entry_data = {
                        'id': entry.id,
                        'content': entry.content,
                        'memory_type': entry.memory_type.value,
                        'priority': entry.priority.value,
                        'timestamp': entry.timestamp.isoformat(),
                        'access_count': entry.access_count,
                        'last_accessed': entry.last_accessed.isoformat(),
                        'tags': list(entry.tags),
                        'size_bytes': entry.size_bytes,
                        'compressed': entry.compressed,
                        'context_hash': entry.context_hash,
                        'related_entries': list(entry.related_entries),
                        'metadata': entry.metadata
                    }
                    session_entries.append(entry_data)
            
            with open(session_file, 'w') as f:
                json.dump({'entries': session_entries}, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not save session memory: {e}")
    
    def _update_indices(self, entry: MemoryEntry):
        """Update memory indices for fast lookup"""
        self.type_indices[entry.memory_type].add(entry.id)
        for tag in entry.tags:
            self.tag_indices[tag].add(entry.id)
    
    def _remove_from_indices(self, entry: MemoryEntry):
        """Remove entry from indices"""
        self.type_indices[entry.memory_type].discard(entry.id)
        for tag in entry.tags:
            self.tag_indices[tag].discard(entry.id)
    
    def _calculate_content_hash(self, content: Any) -> str:
        """Calculate hash for content to detect duplicates"""
        content_str = str(content) if not isinstance(content, (str, bytes)) else content
        if isinstance(content_str, str):
            content_str = content_str.encode('utf-8')
        return hashlib.sha256(content_str).hexdigest()[:16]
    
    def _estimate_size(self, content: Any) -> int:
        """Estimate memory size of content"""
        if isinstance(content, str):
            return len(content.encode('utf-8'))
        elif isinstance(content, (dict, list)):
            return len(json.dumps(content, default=str).encode('utf-8'))
        else:
            return sys.getsizeof(content)
    
    def _compress_content(self, content: Any) -> Tuple[bytes, bool]:
        """Compress content if beneficial"""
        try:
            # Serialize content
            serialized = pickle.dumps(content)
            
            # Compress only if content is large enough
            if len(serialized) > 1024:  # 1KB threshold
                compressed = zlib.compress(serialized, level=9)
                if len(compressed) < len(serialized) * 0.8:  # 20% compression minimum
                    return compressed, True
            
            return serialized, False
        except Exception:
            return str(content).encode('utf-8'), False
    
    def _decompress_content(self, data: bytes, compressed: bool) -> Any:
        """Decompress content"""
        try:
            if compressed:
                decompressed = zlib.decompress(data)
                return pickle.loads(decompressed)
            else:
                return pickle.loads(data)
        except Exception:
            return data.decode('utf-8')
    
    async def store(
        self,
        content: Any,
        memory_type: MemoryType,
        priority: Priority = Priority.MEDIUM,
        tags: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        entry_id: Optional[str] = None
    ) -> str:
        """Store content in memory with intelligent management"""
        
        with self.lock:
            # Generate ID if not provided
            if not entry_id:
                content_hash = self._calculate_content_hash(content)
                entry_id = f"{memory_type.value}_{content_hash}_{int(time.time())}"
            
            # Check for duplicates
            existing = self._find_similar_content(content)
            if existing:
                # Update existing entry instead of creating duplicate
                existing.access_count += 1
                existing.last_accessed = datetime.now()
                return existing.id
            
            # Estimate size
            size_bytes = self._estimate_size(content)
            
            # Compress if beneficial
            stored_content, compressed = self._compress_content(content)
            
            # Create memory entry
            entry = MemoryEntry(
                id=entry_id,
                content=stored_content if compressed else content,
                memory_type=memory_type,
                priority=priority,
                timestamp=datetime.now(),
                tags=tags or set(),
                size_bytes=size_bytes,
                compressed=compressed,
                context_hash=self._calculate_content_hash(content),
                metadata=metadata or {}
            )
            
            # Check memory limits
            await self._ensure_memory_limits(entry.size_bytes)
            
            # Store entry
            self.memory_store[entry_id] = entry
            self._update_indices(entry)
            self.current_memory_usage += entry.size_bytes
            
            # Update metrics
            self.metrics['entries_stored'] += 1
            self.metrics['memory_usage_bytes'] = self.current_memory_usage
            
            # Save persistent entries to database
            if memory_type in [MemoryType.PERSISTENT, MemoryType.PROJECT]:
                await self._save_to_database(entry)
            
            return entry_id
    
    async def retrieve(
        self,
        entry_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        tags: Optional[Set[str]] = None,
        limit: int = 10,
        content_query: Optional[str] = None
    ) -> List[Tuple[str, Any, Dict[str, Any]]]:
        """Retrieve memory entries with intelligent ranking"""
        
        start_time = time.time()
        
        with self.lock:
            candidates = set()
            
            # Direct ID lookup
            if entry_id:
                if entry_id in self.memory_store:
                    entry = self.memory_store[entry_id]
                    entry.access_count += 1
                    entry.last_accessed = datetime.now()
                    
                    content = self._decompress_content(entry.content, entry.compressed) if entry.compressed else entry.content
                    
                    return [(entry.id, content, entry.metadata)]
                else:
                    return []
            
            # Type-based lookup
            if memory_type:
                candidates.update(self.type_indices[memory_type])
            
            # Tag-based lookup
            if tags:
                tag_matches = set.intersection(*[self.tag_indices[tag] for tag in tags if tag in self.tag_indices])
                if candidates:
                    candidates &= tag_matches
                else:
                    candidates = tag_matches
            
            # If no specific criteria, get recent entries
            if not candidates and not content_query:
                candidates = set(self.memory_store.keys())
            
            # Content-based search
            if content_query:
                content_matches = self._search_content(content_query)
                if candidates:
                    candidates &= content_matches
                else:
                    candidates = content_matches
            
            # Rank candidates by relevance and recency
            ranked_entries = self._rank_entries(candidates)
            
            # Return top results
            results = []
            for entry_id in ranked_entries[:limit]:
                if entry_id in self.memory_store:
                    entry = self.memory_store[entry_id]
                    entry.access_count += 1
                    entry.last_accessed = datetime.now()
                    
                    content = self._decompress_content(entry.content, entry.compressed) if entry.compressed else entry.content
                    results.append((entry.id, content, entry.metadata))
            
            # Update metrics
            retrieval_time = time.time() - start_time
            self.metrics['entries_retrieved'] += len(results)
            self.metrics['average_retrieval_time'] = (
                (self.metrics['average_retrieval_time'] * (self.metrics['entries_retrieved'] - len(results)) + 
                 retrieval_time) / self.metrics['entries_retrieved']
            )
            
            return results
    
    def _find_similar_content(self, content: Any) -> Optional[MemoryEntry]:
        """Find similar content to avoid duplicates"""
        content_hash = self._calculate_content_hash(content)
        
        for entry in self.memory_store.values():
            if entry.context_hash == content_hash:
                return entry
        
        return None
    
    def _search_content(self, query: str) -> Set[str]:
        """Search content using simple text matching"""
        matches = set()
        query_lower = query.lower()
        
        for entry_id, entry in self.memory_store.items():
            # Search in content
            content_str = str(entry.content).lower()
            if query_lower in content_str:
                matches.add(entry_id)
                continue
            
            # Search in tags
            if any(query_lower in tag.lower() for tag in entry.tags):
                matches.add(entry_id)
                continue
            
            # Search in metadata
            metadata_str = str(entry.metadata).lower()
            if query_lower in metadata_str:
                matches.add(entry_id)
        
        return matches
    
    def _rank_entries(self, candidates: Set[str]) -> List[str]:
        """Rank entries by relevance, recency, and access frequency"""
        if not candidates:
            return []
        
        scored_entries = []
        now = datetime.now()
        
        for entry_id in candidates:
            if entry_id not in self.memory_store:
                continue
                
            entry = self.memory_store[entry_id]
            
            # Scoring factors
            recency_score = 1.0 / max((now - entry.last_accessed).total_seconds() / 3600, 0.1)  # Hours ago
            frequency_score = min(entry.access_count / 10.0, 1.0)  # Access frequency
            priority_score = entry.priority.value / 5.0  # Priority weight
            
            total_score = recency_score * 0.4 + frequency_score * 0.3 + priority_score * 0.3
            scored_entries.append((total_score, entry_id))
        
        # Sort by score descending
        scored_entries.sort(key=lambda x: x[0], reverse=True)
        return [entry_id for _, entry_id in scored_entries]
    
    async def _ensure_memory_limits(self, new_size: int):
        """Ensure memory usage stays within limits"""
        if self.current_memory_usage + new_size <= self.max_memory_bytes:
            return
        
        # Need to free up memory
        target_free = new_size + (self.max_memory_bytes * 0.1)  # 10% buffer
        
        # Get candidates for removal (lowest priority, least accessed, oldest)
        candidates = []
        for entry_id, entry in self.memory_store.items():
            if entry.priority == Priority.CRITICAL:
                continue  # Never remove critical entries
            
            score = (
                entry.priority.value * 0.4 +
                min(entry.access_count / 10.0, 1.0) * 0.3 +
                (1.0 / max((datetime.now() - entry.last_accessed).total_seconds() / 3600, 0.1)) * 0.3
            )
            candidates.append((score, entry_id, entry.size_bytes))
        
        # Sort by score (lowest first for removal)
        candidates.sort(key=lambda x: x[0])
        
        # Remove entries until we have enough space
        freed_bytes = 0
        for _, entry_id, size_bytes in candidates:
            if freed_bytes >= target_free:
                break
            
            await self._remove_entry(entry_id)
            freed_bytes += size_bytes
    
    async def _remove_entry(self, entry_id: str):
        """Remove an entry from memory"""
        if entry_id not in self.memory_store:
            return
        
        entry = self.memory_store[entry_id]
        
        # Remove from indices
        self._remove_from_indices(entry)
        
        # Update memory usage
        self.current_memory_usage -= entry.size_bytes
        
        # Remove from store
        del self.memory_store[entry_id]
    
    async def _save_to_database(self, entry: MemoryEntry):
        """Save entry to persistent database"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO memory_entries 
                    (id, content, memory_type, priority, timestamp, access_count, 
                     last_accessed, tags, size_bytes, compressed, context_hash, 
                     related_entries, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entry.id,
                    pickle.dumps(entry.content),
                    entry.memory_type.value,
                    entry.priority.value,
                    entry.timestamp.timestamp(),
                    entry.access_count,
                    entry.last_accessed.timestamp(),
                    json.dumps(list(entry.tags)),
                    entry.size_bytes,
                    entry.compressed,
                    entry.context_hash,
                    json.dumps(list(entry.related_entries)),
                    json.dumps(entry.metadata)
                ))
                conn.commit()
        except Exception as e:
            print(f"Warning: Could not save to database: {e}")
    
    def _background_compression_loop(self):
        """Background thread for intelligent compression"""
        while True:
            try:
                time.sleep(60)  # Check every minute
                
                with self.lock:
                    # Find candidates for compression
                    for entry_id, entry in self.memory_store.items():
                        if (not entry.compressed and 
                            entry.size_bytes > 2048 and  # 2KB minimum
                            entry.access_count < 5 and   # Infrequently accessed
                            (datetime.now() - entry.last_accessed).total_seconds() > 3600):  # Not accessed in 1 hour
                            
                            # Compress entry
                            compressed_content, is_compressed = self._compress_content(entry.content)
                            if is_compressed:
                                old_size = entry.size_bytes
                                entry.content = compressed_content
                                entry.compressed = True
                                entry.size_bytes = len(compressed_content)
                                
                                self.current_memory_usage += entry.size_bytes - old_size
                                self.metrics['compressions_performed'] += 1
                
            except Exception:
                pass  # Ignore compression errors
    
    async def compress_context_window(self, window_id: str = "main") -> float:
        """Compress context window to save tokens"""
        if window_id not in self.context_windows:
            return 0.0
        
        window = self.context_windows[window_id]
        if not window.entries:
            return 0.0
        
        original_tokens = window.total_tokens
        
        # Get entries to compress
        entries_to_compress = []
        for entry_id in window.entries:
            if entry_id in self.memory_store:
                entries_to_compress.append(self.memory_store[entry_id])
        
        # Compress by summarizing older entries
        if len(entries_to_compress) > 10:
            # Keep recent entries, compress older ones
            keep_recent = 5
            compress_entries = entries_to_compress[:-keep_recent]
            
            # Create compressed summary
            summary_content = {
                'type': 'compressed_context',
                'original_count': len(compress_entries),
                'summary': self._create_context_summary(compress_entries),
                'timestamp': datetime.now().isoformat()
            }
            
            # Store compressed summary
            summary_id = await self.store(
                summary_content,
                MemoryType.SYSTEM,
                Priority.HIGH,
                tags={'compressed_context', window_id}
            )
            
            # Update window
            window.entries = window.entries[-keep_recent:] + [summary_id]
            window.total_tokens = int(window.total_tokens * 0.3)  # Estimate compression
            window.compression_ratio = 1.0 - (window.total_tokens / original_tokens)
            window.last_compression = datetime.now()
            
            self.metrics['context_compressions'] += 1
        
        return window.compression_ratio
    
    def _create_context_summary(self, entries: List[MemoryEntry]) -> str:
        """Create intelligent summary of context entries"""
        summaries = []
        
        for entry in entries:
            content_str = str(entry.content)[:200] + "..." if len(str(entry.content)) > 200 else str(entry.content)
            summary = f"[{entry.memory_type.value}] {content_str}"
            summaries.append(summary)
        
        return "\n".join(summaries)
    
    async def get_project_context(self, project_name: str) -> Dict[str, Any]:
        """Get all context for a specific project"""
        project_entries = await self.retrieve(
            memory_type=MemoryType.PROJECT,
            tags={project_name},
            limit=50
        )
        
        return {
            'project_name': project_name,
            'entries': project_entries,
            'last_updated': datetime.now().isoformat(),
            'entry_count': len(project_entries)
        }
    
    async def store_conversation_turn(
        self,
        user_input: str,
        assistant_response: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """Store a conversation turn"""
        turn_data = {
            'user_input': user_input,
            'assistant_response': assistant_response,
            'context': context or {},
            'timestamp': datetime.now().isoformat()
        }
        
        await self.store(
            turn_data,
            MemoryType.CONVERSATION,
            Priority.MEDIUM,
            tags={'conversation_turn'},
            metadata={'turn_length': len(user_input) + len(assistant_response)}
        )
    
    async def get_conversation_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent conversation history"""
        history_entries = await self.retrieve(
            memory_type=MemoryType.CONVERSATION,
            tags={'conversation_turn'},
            limit=limit
        )
        
        conversations = []
        for entry_id, content, metadata in history_entries:
            conversations.append(content)
        
        return sorted(conversations, key=lambda x: x['timestamp'])
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        type_counts = {}
        for memory_type in MemoryType:
            type_counts[memory_type.value] = len(self.type_indices[memory_type])
        
        return {
            'total_entries': len(self.memory_store),
            'memory_usage_mb': self.current_memory_usage / (1024 * 1024),
            'memory_limit_mb': self.max_memory_mb,
            'memory_usage_percent': (self.current_memory_usage / self.max_memory_bytes) * 100,
            'entries_by_type': type_counts,
            'total_tags': len(self.tag_indices),
            'context_windows': len(self.context_windows),
            'session_duration_hours': (datetime.now() - self.metrics['session_start']).total_seconds() / 3600,
            'performance_metrics': self.metrics
        }
    
    async def cleanup_old_entries(self, max_age_days: int = 30):
        """Clean up old entries to free memory"""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        entries_to_remove = []
        
        for entry_id, entry in self.memory_store.items():
            if (entry.last_accessed < cutoff_date and 
                entry.priority not in [Priority.CRITICAL, Priority.HIGH] and
                entry.memory_type not in [MemoryType.PERSISTENT, MemoryType.PROJECT]):
                entries_to_remove.append(entry_id)
        
        for entry_id in entries_to_remove:
            await self._remove_entry(entry_id)
        
        return len(entries_to_remove)
    
    async def shutdown(self):
        """Gracefully shutdown memory manager"""
        # Save session memory
        self._save_session_memory()
        
        # Final database sync for persistent entries
        for entry in self.memory_store.values():
            if entry.memory_type in [MemoryType.PERSISTENT, MemoryType.PROJECT]:
                await self._save_to_database(entry)
        
        print(f"🧠 Memory Manager: Saved {len(self.memory_store)} entries")

# Global memory manager instance
_global_memory_manager: Optional[MemoryManager] = None

def get_memory_manager(project_path: Path = None, max_memory_mb: int = 500) -> MemoryManager:
    """Get the global memory manager instance"""
    global _global_memory_manager
    if _global_memory_manager is None:
        if project_path is None:
            project_path = Path.cwd()
        _global_memory_manager = MemoryManager(project_path, max_memory_mb)
    return _global_memory_manager

async def initialize_memory_system(project_path: Path, max_memory_mb: int = 500) -> MemoryManager:
    """Initialize the global memory system"""
    global _global_memory_manager
    _global_memory_manager = MemoryManager(project_path, max_memory_mb)
    return _global_memory_manager