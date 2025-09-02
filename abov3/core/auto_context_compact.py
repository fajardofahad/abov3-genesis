"""
ABOV3 Genesis - Auto Context Compact System
Claude-level intelligent context compaction with advanced semantic compression,
hierarchical summarization, and contextual relationship preservation.
"""

import asyncio
import json
import hashlib
import time
import threading
import re
import logging
import sqlite3
import zlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum, auto
import pickle
import heapq
from abc import ABC, abstractmethod
import numpy as np

# Import existing components
from .memory_manager import (
    MemoryManager,
    MemoryEntry,
    MemoryType,
    Priority,
    get_memory_manager
)
from .enhanced_debug_integration import (
    EnhancedDebugIntegration,
    get_debug_integration
)


class CompactionStrategy(Enum):
    """Context compaction strategies"""
    AGGRESSIVE = "aggressive"        # Maximum compression, minimal preservation
    BALANCED = "balanced"           # Balance compression with context preservation
    CONSERVATIVE = "conservative"   # Preserve maximum context, minimal compression
    ADAPTIVE = "adaptive"          # AI-driven strategy selection
    EMERGENCY = "emergency"        # Emergency compaction for memory limits


class ContextImportance(Enum):
    """Context importance levels"""
    CRITICAL = 5    # Never compress
    HIGH = 4       # Compress only when absolutely necessary
    MEDIUM = 3     # Standard compression eligibility
    LOW = 2        # Aggressive compression candidate
    MINIMAL = 1    # First to be compressed or removed


class CompressionMethod(Enum):
    """Compression methods"""
    SEMANTIC = "semantic"           # Semantic meaning preservation
    HIERARCHICAL = "hierarchical"   # Multi-level summarization
    TEMPORAL = "temporal"          # Time-based significance decay
    FREQUENCY = "frequency"        # Access frequency based
    STRUCTURAL = "structural"      # Code structure aware
    HYBRID = "hybrid"             # Combined approaches


@dataclass
class ContextSegment:
    """Individual context segment for analysis and compression"""
    segment_id: str
    content: Any
    content_type: str  # 'conversation', 'code', 'file', 'error', 'decision'
    timestamp: datetime
    importance: ContextImportance
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    relationships: Set[str] = field(default_factory=set)  # Related segment IDs
    semantic_hash: str = ""
    compression_ratio: float = 0.0
    tokens_estimate: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompactionEvent:
    """Event during compaction process"""
    event_id: str
    event_type: str
    timestamp: datetime
    segments_affected: List[str]
    compression_ratio: float
    tokens_saved: int
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompactionStats:
    """Statistics for compaction operations"""
    total_compactions: int = 0
    tokens_saved: int = 0
    segments_compressed: int = 0
    segments_removed: int = 0
    average_compression_ratio: float = 0.0
    last_compaction: Optional[datetime] = None
    strategy_usage: Dict[str, int] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class SemanticAnalyzer(ABC):
    """Abstract base for semantic analysis"""
    
    @abstractmethod
    async def analyze_importance(self, content: Any, context: Dict[str, Any]) -> ContextImportance:
        """Analyze content importance"""
        pass
    
    @abstractmethod
    async def extract_relationships(self, segments: List[ContextSegment]) -> Dict[str, Set[str]]:
        """Extract relationships between segments"""
        pass
    
    @abstractmethod
    async def generate_summary(self, segments: List[ContextSegment], 
                              target_length: int) -> str:
        """Generate semantic summary"""
        pass


class ClaudeStyleSemanticAnalyzer(SemanticAnalyzer):
    """Claude-style semantic analysis with advanced NLP techniques"""
    
    def __init__(self):
        self.importance_patterns = {
            # Code patterns
            'function_definition': (r'def\s+\w+\s*\(', ContextImportance.HIGH),
            'class_definition': (r'class\s+\w+', ContextImportance.HIGH),
            'import_statement': (r'import\s+|from\s+\w+\s+import', ContextImportance.MEDIUM),
            'error_handling': (r'try:|except:|finally:|raise\s+', ContextImportance.HIGH),
            
            # Decision patterns
            'user_decision': (r'(?i)(decide|choose|select|prefer|want)', ContextImportance.HIGH),
            'requirement': (r'(?i)(need|must|should|require)', ContextImportance.HIGH),
            'constraint': (r'(?i)(cannot|can\'t|won\'t|limit|restrict)', ContextImportance.HIGH),
            
            # File operations
            'file_creation': (r'(?i)(creat|generat|writ).*file', ContextImportance.MEDIUM),
            'file_modification': (r'(?i)(edit|modif|updat|chang).*file', ContextImportance.MEDIUM),
            
            # Error patterns
            'critical_error': (r'(?i)(critical|fatal|emergency|urgent)', ContextImportance.CRITICAL),
            'error_resolution': (r'(?i)(fix|solv|resolv|correct)', ContextImportance.HIGH),
            
            # Project context
            'architecture_decision': (r'(?i)(architect|design|structure|pattern)', ContextImportance.HIGH),
            'configuration': (r'(?i)(config|setting|option|parameter)', ContextImportance.MEDIUM),
        }
        
        self.relationship_indicators = [
            'related to', 'depends on', 'uses', 'calls', 'imports',
            'inherits from', 'implements', 'extends', 'overrides',
            'references', 'modifies', 'creates', 'deletes'
        ]
    
    async def analyze_importance(self, content: Any, context: Dict[str, Any]) -> ContextImportance:
        """Analyze content importance using pattern matching and heuristics"""
        try:
            content_str = str(content).lower()
            
            # Pattern-based analysis
            max_importance = ContextImportance.MINIMAL
            for pattern_name, (pattern, importance) in self.importance_patterns.items():
                if re.search(pattern, content_str):
                    if importance.value > max_importance.value:
                        max_importance = importance
            
            # Context-based adjustments
            if context.get('user_interaction', False):
                max_importance = ContextImportance(min(max_importance.value + 1, 5))
            
            if context.get('recent_access', 0) > 5:
                max_importance = ContextImportance(min(max_importance.value + 1, 5))
            
            if context.get('error_related', False):
                max_importance = ContextImportance(min(max_importance.value + 1, 5))
            
            # Length-based adjustments
            if len(content_str) > 10000:  # Very long content
                max_importance = ContextImportance(max(max_importance.value - 1, 1))
            elif len(content_str) < 50:  # Very short content
                max_importance = ContextImportance(max(max_importance.value - 1, 1))
            
            return max_importance
            
        except Exception:
            return ContextImportance.MEDIUM
    
    async def extract_relationships(self, segments: List[ContextSegment]) -> Dict[str, Set[str]]:
        """Extract semantic relationships between segments"""
        relationships = defaultdict(set)
        
        try:
            for i, segment1 in enumerate(segments):
                content1 = str(segment1.content).lower()
                
                for j, segment2 in enumerate(segments[i+1:], i+1):
                    content2 = str(segment2.content).lower()
                    
                    # Direct reference relationships
                    if any(indicator in content1 and indicator in content2 
                          for indicator in self.relationship_indicators):
                        relationships[segment1.segment_id].add(segment2.segment_id)
                        relationships[segment2.segment_id].add(segment1.segment_id)
                    
                    # Semantic similarity (simple approach)
                    similarity = self._calculate_similarity(content1, content2)
                    if similarity > 0.3:  # Threshold for relationship
                        relationships[segment1.segment_id].add(segment2.segment_id)
                        relationships[segment2.segment_id].add(segment1.segment_id)
                    
                    # Sequential relationships (temporal proximity)
                    time_diff = abs((segment1.timestamp - segment2.timestamp).total_seconds())
                    if time_diff < 300:  # Within 5 minutes
                        relationships[segment1.segment_id].add(segment2.segment_id)
        
        except Exception as e:
            logging.warning(f"Relationship extraction error: {e}")
        
        return dict(relationships)
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        try:
            # Simple word overlap similarity
            words1 = set(re.findall(r'\w+', text1.lower()))
            words2 = set(re.findall(r'\w+', text2.lower()))
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            
            return intersection / union if union > 0 else 0.0
            
        except Exception:
            return 0.0
    
    async def generate_summary(self, segments: List[ContextSegment], 
                              target_length: int) -> str:
        """Generate intelligent summary of segments"""
        try:
            if not segments:
                return ""
            
            # Sort segments by importance and recency
            sorted_segments = sorted(
                segments,
                key=lambda s: (s.importance.value, s.last_accessed),
                reverse=True
            )
            
            # Categorize segments
            categories = defaultdict(list)
            for segment in sorted_segments:
                categories[segment.content_type].append(segment)
            
            summary_parts = []
            remaining_length = target_length
            
            # Prioritized summarization by category
            category_priorities = {
                'decision': 3,
                'error': 3,
                'code': 2,
                'file': 2,
                'conversation': 1
            }
            
            for category in sorted(categories.keys(), 
                                 key=lambda x: category_priorities.get(x, 0), 
                                 reverse=True):
                if remaining_length <= 0:
                    break
                
                category_segments = categories[category]
                category_summary = self._summarize_category(
                    category, category_segments, min(remaining_length // 2, 200)
                )
                
                if category_summary:
                    summary_parts.append(f"[{category.upper()}] {category_summary}")
                    remaining_length -= len(category_summary)
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logging.warning(f"Summary generation error: {e}")
            return f"[COMPRESSED] {len(segments)} segments from {segments[0].timestamp} to {segments[-1].timestamp}"
    
    def _summarize_category(self, category: str, segments: List[ContextSegment], 
                           max_length: int) -> str:
        """Summarize segments within a category"""
        if not segments:
            return ""
        
        try:
            if category == 'decision':
                # Extract key decisions
                decisions = []
                for segment in segments[:5]:  # Top 5 most important
                    content = str(segment.content)
                    decision_match = re.search(r'(?i)(decided|chose|selected)[^.]{0,100}', content)
                    if decision_match:
                        decisions.append(decision_match.group(0))
                return "; ".join(decisions)[:max_length]
            
            elif category == 'error':
                # Summarize errors and resolutions
                error_summary = f"{len(segments)} errors handled"
                if segments:
                    recent_error = str(segments[0].content)[:100]
                    error_summary += f": {recent_error}..."
                return error_summary[:max_length]
            
            elif category == 'code':
                # Summarize code operations
                operations = []
                for segment in segments[:3]:
                    content = str(segment.content)
                    # Extract function/class names
                    functions = re.findall(r'(?:def|class)\s+(\w+)', content)
                    if functions:
                        operations.extend(functions)
                
                if operations:
                    return f"Code operations: {', '.join(operations[:5])}"[:max_length]
                else:
                    return f"{len(segments)} code segments"[:max_length]
            
            elif category == 'file':
                # Summarize file operations
                files = set()
                for segment in segments:
                    content = str(segment.content)
                    file_matches = re.findall(r'[\w/\\.-]+\.[\w]+', content)
                    files.update(file_matches[:3])  # Limit to avoid overflow
                
                if files:
                    return f"Files: {', '.join(list(files)[:5])}"[:max_length]
                else:
                    return f"{len(segments)} file operations"[:max_length]
            
            else:  # conversation and others
                # General summarization
                total_length = sum(len(str(s.content)) for s in segments)
                avg_length = total_length // len(segments) if segments else 0
                return f"{len(segments)} items (avg {avg_length} chars)"[:max_length]
                
        except Exception:
            return f"{len(segments)} {category} segments"[:max_length]


class AutoContextCompact:
    """
    Claude-level Auto Context Compact system with intelligent compression,
    semantic analysis, and contextual relationship preservation.
    """
    
    def __init__(self, project_path: Optional[Path] = None, 
                 memory_manager: Optional[MemoryManager] = None,
                 debug_integration: Optional[EnhancedDebugIntegration] = None):
        
        self.project_path = project_path or Path.cwd()
        self.memory_manager = memory_manager or get_memory_manager(self.project_path)
        self.debug_integration = debug_integration or get_debug_integration(self.project_path)
        
        # Context storage and management
        self.context_segments: Dict[str, ContextSegment] = {}
        self.active_contexts: Dict[str, List[str]] = {}  # context_id -> segment_ids
        self.compressed_contexts: Dict[str, Dict[str, Any]] = {}
        
        # Compaction configuration
        self.config = {
            'max_context_tokens': 120000,        # Claude-3.5 context size
            'compaction_threshold': 100000,      # Start compacting at 100k tokens
            'emergency_threshold': 115000,       # Emergency compaction threshold
            'preserve_recent_tokens': 20000,     # Always preserve recent context
            'min_compression_ratio': 0.3,        # Minimum 30% compression
            'max_segments_per_compaction': 100,  # Batch size limit
            'compaction_interval': 300,          # Background check interval (5 min)
            'rollback_history_size': 10,         # Number of compaction states to keep
        }
        
        # Strategy and analysis
        self.current_strategy = CompactionStrategy.ADAPTIVE
        self.semantic_analyzer = ClaudeStyleSemanticAnalyzer()
        
        # Performance tracking
        self.stats = CompactionStats()
        self.compaction_history: deque = deque(maxlen=self.config['rollback_history_size'])
        self.performance_metrics = {
            'total_tokens_processed': 0,
            'average_compression_time': 0.0,
            'success_rate': 1.0,
            'context_preservation_score': 0.9
        }
        
        # Threading and monitoring
        self.lock = threading.RLock()
        self.monitoring_enabled = False
        self.monitor_thread = None
        self._stop_monitoring = threading.Event()
        
        # Database for persistent compaction data
        self.db_path = self.project_path / '.abov3' / 'context_compact.db'
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Setup logging
        self.logger = logging.getLogger('abov3.context_compact')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Load previous state
        self._load_persistent_state()
        
        # Start background monitoring
        self.start_monitoring()
        
        self.logger.info("Auto Context Compact initialized")
    
    def _initialize_database(self):
        """Initialize SQLite database for persistent storage"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Context segments table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS context_segments (
                        segment_id TEXT PRIMARY KEY,
                        content BLOB,
                        content_type TEXT,
                        timestamp REAL,
                        importance INTEGER,
                        access_count INTEGER,
                        last_accessed REAL,
                        relationships TEXT,
                        semantic_hash TEXT,
                        compression_ratio REAL,
                        tokens_estimate INTEGER,
                        metadata TEXT
                    )
                ''')
                
                # Compaction events table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS compaction_events (
                        event_id TEXT PRIMARY KEY,
                        event_type TEXT,
                        timestamp REAL,
                        segments_affected TEXT,
                        compression_ratio REAL,
                        tokens_saved INTEGER,
                        success BOOLEAN,
                        details TEXT
                    )
                ''')
                
                # Compressed contexts table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS compressed_contexts (
                        context_id TEXT PRIMARY KEY,
                        original_segments TEXT,
                        compressed_content TEXT,
                        compression_method TEXT,
                        compression_ratio REAL,
                        timestamp REAL,
                        metadata TEXT
                    )
                ''')
                
                # Create indices
                conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON context_segments(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_importance ON context_segments(importance)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_content_type ON context_segments(content_type)')
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Database initialization error: {e}")
    
    def _load_persistent_state(self):
        """Load persistent state from database"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Load recent context segments
                cursor = conn.execute('''
                    SELECT * FROM context_segments 
                    ORDER BY timestamp DESC LIMIT 1000
                ''')
                
                for row in cursor.fetchall():
                    segment = ContextSegment(
                        segment_id=row[0],
                        content=pickle.loads(row[1]),
                        content_type=row[2],
                        timestamp=datetime.fromtimestamp(row[3]),
                        importance=ContextImportance(row[4]),
                        access_count=row[5],
                        last_accessed=datetime.fromtimestamp(row[6]),
                        relationships=set(json.loads(row[7])),
                        semantic_hash=row[8],
                        compression_ratio=row[9],
                        tokens_estimate=row[10],
                        metadata=json.loads(row[11])
                    )
                    self.context_segments[segment.segment_id] = segment
                
                # Load compressed contexts
                cursor = conn.execute('SELECT * FROM compressed_contexts')
                for row in cursor.fetchall():
                    self.compressed_contexts[row[0]] = {
                        'original_segments': json.loads(row[1]),
                        'compressed_content': row[2],
                        'compression_method': row[3],
                        'compression_ratio': row[4],
                        'timestamp': datetime.fromtimestamp(row[5]),
                        'metadata': json.loads(row[6])
                    }
                
                self.logger.info(f"Loaded {len(self.context_segments)} segments and {len(self.compressed_contexts)} compressed contexts")
                
        except Exception as e:
            self.logger.warning(f"Could not load persistent state: {e}")
    
    def _save_segment_to_db(self, segment: ContextSegment):
        """Save segment to database"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO context_segments 
                    (segment_id, content, content_type, timestamp, importance, 
                     access_count, last_accessed, relationships, semantic_hash, 
                     compression_ratio, tokens_estimate, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    segment.segment_id,
                    pickle.dumps(segment.content),
                    segment.content_type,
                    segment.timestamp.timestamp(),
                    segment.importance.value,
                    segment.access_count,
                    segment.last_accessed.timestamp(),
                    json.dumps(list(segment.relationships)),
                    segment.semantic_hash,
                    segment.compression_ratio,
                    segment.tokens_estimate,
                    json.dumps(segment.metadata)
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to save segment to database: {e}")
    
    async def add_context(self, content: Any, content_type: str = 'conversation',
                         context_id: str = 'main', 
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add new context content for management"""
        with self.lock:
            # Generate segment ID
            segment_id = self._generate_segment_id(content, content_type)
            
            # Estimate tokens
            tokens_estimate = self._estimate_tokens(content)
            
            # Calculate semantic hash
            semantic_hash = self._calculate_semantic_hash(content)
            
            # Analyze importance
            context_data = {
                'user_interaction': content_type == 'conversation',
                'recent_access': 0,
                'error_related': 'error' in content_type.lower(),
                **(metadata if metadata else {})
            }
            importance = await self.semantic_analyzer.analyze_importance(content, context_data)
            
            # Create segment
            segment = ContextSegment(
                segment_id=segment_id,
                content=content,
                content_type=content_type,
                timestamp=datetime.now(),
                importance=importance,
                semantic_hash=semantic_hash,
                tokens_estimate=tokens_estimate,
                metadata=metadata or {}
            )
            
            # Store segment
            self.context_segments[segment_id] = segment
            
            # Add to active context
            if context_id not in self.active_contexts:
                self.active_contexts[context_id] = []
            self.active_contexts[context_id].append(segment_id)
            
            # Save to database
            self._save_segment_to_db(segment)
            
            # Check if compaction is needed
            total_tokens = self._calculate_total_tokens(context_id)
            if total_tokens > self.config['compaction_threshold']:
                await self._trigger_compaction(context_id, CompactionStrategy.ADAPTIVE)
            
            self.logger.debug(f"Added context segment: {segment_id} ({tokens_estimate} tokens)")
            return segment_id
    
    async def compact_context(self, context_id: str = 'main', 
                             strategy: Optional[CompactionStrategy] = None) -> Dict[str, Any]:
        """Perform context compaction with specified strategy"""
        start_time = time.time()
        
        with self.lock:
            if context_id not in self.active_contexts:
                return {'success': False, 'error': f'Context {context_id} not found'}
            
            strategy = strategy or self.current_strategy
            
            try:
                # Get segments to process
                segment_ids = self.active_contexts[context_id]
                segments = [self.context_segments[sid] for sid in segment_ids 
                           if sid in self.context_segments]
                
                if not segments:
                    return {'success': True, 'message': 'No segments to compact'}
                
                # Calculate current state
                total_tokens = sum(s.tokens_estimate for s in segments)
                
                # Determine compaction plan
                compaction_plan = await self._create_compaction_plan(
                    segments, strategy, total_tokens
                )
                
                if not compaction_plan['segments_to_compress']:
                    return {'success': True, 'message': 'No compaction needed'}
                
                # Execute compaction
                result = await self._execute_compaction_plan(
                    context_id, compaction_plan
                )
                
                # Record performance metrics
                processing_time = time.time() - start_time
                self.performance_metrics['average_compression_time'] = (
                    (self.performance_metrics['average_compression_time'] * self.stats.total_compactions +
                     processing_time) / (self.stats.total_compactions + 1)
                )
                
                # Update statistics
                self.stats.total_compactions += 1
                self.stats.tokens_saved += result.get('tokens_saved', 0)
                self.stats.segments_compressed += len(compaction_plan['segments_to_compress'])
                self.stats.last_compaction = datetime.now()
                
                if strategy.value not in self.stats.strategy_usage:
                    self.stats.strategy_usage[strategy.value] = 0
                self.stats.strategy_usage[strategy.value] += 1
                
                # Calculate average compression ratio
                if self.stats.total_compactions > 0:
                    self.stats.average_compression_ratio = (
                        result.get('compression_ratio', 0) + 
                        self.stats.average_compression_ratio * (self.stats.total_compactions - 1)
                    ) / self.stats.total_compactions
                
                # Save compaction event
                await self._record_compaction_event(context_id, compaction_plan, result)
                
                # Trigger event handlers
                self._trigger_event('compaction_completed', {
                    'context_id': context_id,
                    'strategy': strategy.value,
                    'result': result
                })
                
                self.logger.info(f"Context compaction completed: {result.get('tokens_saved', 0)} tokens saved")
                return result
                
            except Exception as e:
                self.logger.error(f"Context compaction failed: {e}")
                return {'success': False, 'error': str(e)}
    
    async def _create_compaction_plan(self, segments: List[ContextSegment], 
                                     strategy: CompactionStrategy,
                                     total_tokens: int) -> Dict[str, Any]:
        """Create intelligent compaction plan based on strategy"""
        
        # Sort segments by various criteria
        segments_by_importance = sorted(segments, key=lambda s: s.importance.value)
        segments_by_recency = sorted(segments, key=lambda s: s.last_accessed)
        segments_by_access = sorted(segments, key=lambda s: s.access_count)
        
        # Extract relationships
        relationships = await self.semantic_analyzer.extract_relationships(segments)
        
        # Determine target compression
        if strategy == CompactionStrategy.AGGRESSIVE:
            target_tokens = int(total_tokens * 0.3)  # Compress to 30%
        elif strategy == CompactionStrategy.CONSERVATIVE:
            target_tokens = int(total_tokens * 0.8)  # Compress to 80%
        elif strategy == CompactionStrategy.EMERGENCY:
            target_tokens = int(total_tokens * 0.2)  # Emergency compression
        else:  # BALANCED or ADAPTIVE
            target_tokens = int(total_tokens * 0.5)  # Compress to 50%
        
        tokens_to_remove = total_tokens - target_tokens
        
        # Select segments for compression
        segments_to_compress = []
        segments_to_preserve = []
        tokens_removed = 0
        
        # Always preserve critical and recent segments
        recent_threshold = datetime.now() - timedelta(hours=1)
        
        for segment in segments:
            should_preserve = (
                segment.importance == ContextImportance.CRITICAL or
                segment.last_accessed > recent_threshold or
                (strategy == CompactionStrategy.CONSERVATIVE and 
                 segment.importance.value >= ContextImportance.MEDIUM.value)
            )
            
            if should_preserve:
                segments_to_preserve.append(segment.segment_id)
            else:
                if tokens_removed < tokens_to_remove:
                    segments_to_compress.append(segment.segment_id)
                    tokens_removed += segment.tokens_estimate
                else:
                    segments_to_preserve.append(segment.segment_id)
        
        # Ensure we don't remove too many related segments
        segments_to_compress = self._preserve_relationships(
            segments_to_compress, segments_to_preserve, relationships
        )
        
        return {
            'strategy': strategy,
            'total_tokens': total_tokens,
            'target_tokens': target_tokens,
            'tokens_to_remove': tokens_to_remove,
            'segments_to_compress': segments_to_compress,
            'segments_to_preserve': segments_to_preserve,
            'relationships': relationships,
            'compression_method': self._select_compression_method(strategy)
        }
    
    def _preserve_relationships(self, compress_list: List[str], 
                               preserve_list: List[str],
                               relationships: Dict[str, Set[str]]) -> List[str]:
        """Adjust compression list to preserve important relationships"""
        final_compress_list = []
        
        for segment_id in compress_list:
            related_segments = relationships.get(segment_id, set())
            
            # If segment has many relationships with preserved segments, preserve it
            preserved_relationships = len(related_segments & set(preserve_list))
            total_relationships = len(related_segments)
            
            if total_relationships > 0:
                relationship_ratio = preserved_relationships / total_relationships
                # If more than 50% of relationships are preserved, preserve this too
                if relationship_ratio > 0.5:
                    preserve_list.append(segment_id)
                else:
                    final_compress_list.append(segment_id)
            else:
                final_compress_list.append(segment_id)
        
        return final_compress_list
    
    def _select_compression_method(self, strategy: CompactionStrategy) -> CompressionMethod:
        """Select appropriate compression method based on strategy"""
        if strategy == CompactionStrategy.AGGRESSIVE:
            return CompressionMethod.TEMPORAL
        elif strategy == CompactionStrategy.CONSERVATIVE:
            return CompressionMethod.SEMANTIC
        elif strategy == CompactionStrategy.EMERGENCY:
            return CompressionMethod.FREQUENCY
        else:
            return CompressionMethod.HYBRID
    
    async def _execute_compaction_plan(self, context_id: str, 
                                      plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the compaction plan"""
        
        segments_to_compress = plan['segments_to_compress']
        compression_method = plan['compression_method']
        
        if not segments_to_compress:
            return {
                'success': True,
                'tokens_saved': 0,
                'compression_ratio': 1.0,
                'segments_compressed': 0,
                'message': 'No segments selected for compression'
            }
        
        # Get segments to compress
        compress_segments = [
            self.context_segments[sid] for sid in segments_to_compress
            if sid in self.context_segments
        ]
        
        original_tokens = sum(s.tokens_estimate for s in compress_segments)
        
        # Perform compression based on method
        if compression_method == CompressionMethod.SEMANTIC:
            compressed_content = await self.semantic_analyzer.generate_summary(
                compress_segments, target_length=original_tokens // 4
            )
        elif compression_method == CompressionMethod.HIERARCHICAL:
            compressed_content = await self._hierarchical_compress(compress_segments)
        elif compression_method == CompressionMethod.TEMPORAL:
            compressed_content = await self._temporal_compress(compress_segments)
        elif compression_method == CompressionMethod.FREQUENCY:
            compressed_content = await self._frequency_compress(compress_segments)
        else:  # HYBRID
            compressed_content = await self._hybrid_compress(compress_segments)
        
        compressed_tokens = self._estimate_tokens(compressed_content)
        tokens_saved = original_tokens - compressed_tokens
        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        
        # Create compressed context entry
        compressed_context = {
            'original_segments': segments_to_compress,
            'compressed_content': compressed_content,
            'compression_method': compression_method.value,
            'compression_ratio': compression_ratio,
            'timestamp': datetime.now(),
            'metadata': {
                'original_tokens': original_tokens,
                'compressed_tokens': compressed_tokens,
                'strategy': plan['strategy'].value,
                'segments_count': len(compress_segments)
            }
        }
        
        # Store compressed context
        compressed_id = f"{context_id}_compressed_{int(time.time())}"
        self.compressed_contexts[compressed_id] = compressed_context
        
        # Save to database
        await self._save_compressed_context(compressed_id, compressed_context)
        
        # Remove original segments from active context
        for segment_id in segments_to_compress:
            if segment_id in self.active_contexts[context_id]:
                self.active_contexts[context_id].remove(segment_id)
        
        # Add compressed context reference
        self.active_contexts[context_id].append(compressed_id)
        
        return {
            'success': True,
            'tokens_saved': tokens_saved,
            'compression_ratio': compression_ratio,
            'segments_compressed': len(compress_segments),
            'compressed_id': compressed_id,
            'original_tokens': original_tokens,
            'compressed_tokens': compressed_tokens,
            'compression_method': compression_method.value
        }
    
    async def _hierarchical_compress(self, segments: List[ContextSegment]) -> str:
        """Hierarchical compression with multi-level summarization"""
        try:
            # Group segments by type and importance
            groups = defaultdict(list)
            for segment in segments:
                key = f"{segment.content_type}_{segment.importance.name}"
                groups[key].append(segment)
            
            # Compress each group
            compressed_groups = []
            for group_key, group_segments in groups.items():
                if len(group_segments) == 1:
                    # Single segment - just truncate if needed
                    content = str(group_segments[0].content)
                    if len(content) > 200:
                        content = content[:200] + "..."
                    compressed_groups.append(f"[{group_key}] {content}")
                else:
                    # Multiple segments - summarize
                    group_summary = await self.semantic_analyzer.generate_summary(
                        group_segments, target_length=150
                    )
                    compressed_groups.append(f"[{group_key}] {group_summary}")
            
            return "\n".join(compressed_groups)
            
        except Exception as e:
            self.logger.error(f"Hierarchical compression error: {e}")
            return f"[COMPRESSED] {len(segments)} segments"
    
    async def _temporal_compress(self, segments: List[ContextSegment]) -> str:
        """Temporal compression based on time-based significance"""
        try:
            # Sort by timestamp
            sorted_segments = sorted(segments, key=lambda s: s.timestamp)
            
            # Group by time periods
            now = datetime.now()
            time_groups = {
                'recent': [],      # Last hour
                'today': [],       # Today
                'this_week': [],   # This week
                'older': []        # Older
            }
            
            for segment in sorted_segments:
                age = now - segment.timestamp
                if age < timedelta(hours=1):
                    time_groups['recent'].append(segment)
                elif age < timedelta(days=1):
                    time_groups['today'].append(segment)
                elif age < timedelta(weeks=1):
                    time_groups['this_week'].append(segment)
                else:
                    time_groups['older'].append(segment)
            
            # Compress each time group
            compressed_parts = []
            for period, period_segments in time_groups.items():
                if period_segments:
                    if period == 'recent':
                        # Preserve recent content more
                        summary = await self.semantic_analyzer.generate_summary(
                            period_segments, target_length=300
                        )
                    else:
                        # More aggressive compression for older content
                        summary = await self.semantic_analyzer.generate_summary(
                            period_segments, target_length=100
                        )
                    compressed_parts.append(f"[{period.upper()}] {summary}")
            
            return "\n".join(compressed_parts)
            
        except Exception as e:
            self.logger.error(f"Temporal compression error: {e}")
            return f"[TEMPORAL_COMPRESSED] {len(segments)} segments"
    
    async def _frequency_compress(self, segments: List[ContextSegment]) -> str:
        """Frequency-based compression prioritizing accessed content"""
        try:
            # Sort by access frequency
            sorted_segments = sorted(
                segments, 
                key=lambda s: (s.access_count, s.last_accessed), 
                reverse=True
            )
            
            # Keep frequently accessed content, compress the rest
            high_access = sorted_segments[:len(segments)//3]  # Top 1/3
            low_access = sorted_segments[len(segments)//3:]   # Rest
            
            compressed_parts = []
            
            # Preserve high-access content
            if high_access:
                high_summary = await self.semantic_analyzer.generate_summary(
                    high_access, target_length=200
                )
                compressed_parts.append(f"[HIGH_ACCESS] {high_summary}")
            
            # Compress low-access content more aggressively
            if low_access:
                low_summary = await self.semantic_analyzer.generate_summary(
                    low_access, target_length=100
                )
                compressed_parts.append(f"[LOW_ACCESS] {low_summary}")
            
            return "\n".join(compressed_parts)
            
        except Exception as e:
            self.logger.error(f"Frequency compression error: {e}")
            return f"[FREQUENCY_COMPRESSED] {len(segments)} segments"
    
    async def _hybrid_compress(self, segments: List[ContextSegment]) -> str:
        """Hybrid compression using multiple methods"""
        try:
            # Categorize segments for different compression approaches
            critical_segments = [s for s in segments if s.importance == ContextImportance.CRITICAL]
            high_segments = [s for s in segments if s.importance == ContextImportance.HIGH]
            medium_segments = [s for s in segments if s.importance == ContextImportance.MEDIUM]
            low_segments = [s for s in segments if s.importance.value <= ContextImportance.LOW.value]
            
            compressed_parts = []
            
            # Preserve critical content
            if critical_segments:
                critical_summary = await self.semantic_analyzer.generate_summary(
                    critical_segments, target_length=400
                )
                compressed_parts.append(f"[CRITICAL] {critical_summary}")
            
            # Semantic compression for high importance
            if high_segments:
                high_summary = await self.semantic_analyzer.generate_summary(
                    high_segments, target_length=200
                )
                compressed_parts.append(f"[HIGH] {high_summary}")
            
            # Temporal compression for medium importance
            if medium_segments:
                medium_compressed = await self._temporal_compress(medium_segments)
                compressed_parts.append(f"[MEDIUM] {medium_compressed}")
            
            # Aggressive compression for low importance
            if low_segments:
                low_count = len(low_segments)
                total_tokens = sum(s.tokens_estimate for s in low_segments)
                compressed_parts.append(f"[LOW] {low_count} segments ({total_tokens} tokens) - basic operations and temporary context")
            
            return "\n".join(compressed_parts)
            
        except Exception as e:
            self.logger.error(f"Hybrid compression error: {e}")
            return f"[HYBRID_COMPRESSED] {len(segments)} segments"
    
    async def _save_compressed_context(self, compressed_id: str, 
                                      compressed_context: Dict[str, Any]):
        """Save compressed context to database"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO compressed_contexts
                    (context_id, original_segments, compressed_content, 
                     compression_method, compression_ratio, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    compressed_id,
                    json.dumps(compressed_context['original_segments']),
                    compressed_context['compressed_content'],
                    compressed_context['compression_method'],
                    compressed_context['compression_ratio'],
                    compressed_context['timestamp'].timestamp(),
                    json.dumps(compressed_context['metadata'])
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to save compressed context: {e}")
    
    async def _record_compaction_event(self, context_id: str, plan: Dict[str, Any], 
                                      result: Dict[str, Any]):
        """Record compaction event for analysis"""
        try:
            event = CompactionEvent(
                event_id=f"{context_id}_{int(time.time())}",
                event_type='context_compaction',
                timestamp=datetime.now(),
                segments_affected=plan['segments_to_compress'],
                compression_ratio=result.get('compression_ratio', 0.0),
                tokens_saved=result.get('tokens_saved', 0),
                success=result.get('success', False),
                details={
                    'strategy': plan['strategy'].value,
                    'compression_method': plan['compression_method'].value,
                    'original_tokens': result.get('original_tokens', 0),
                    'compressed_tokens': result.get('compressed_tokens', 0)
                }
            )
            
            # Save to database
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    INSERT INTO compaction_events
                    (event_id, event_type, timestamp, segments_affected,
                     compression_ratio, tokens_saved, success, details)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.event_id,
                    event.event_type,
                    event.timestamp.timestamp(),
                    json.dumps(event.segments_affected),
                    event.compression_ratio,
                    event.tokens_saved,
                    event.success,
                    json.dumps(event.details)
                ))
                conn.commit()
            
            # Add to history for rollback capability
            self.compaction_history.append(event)
            
        except Exception as e:
            self.logger.error(f"Failed to record compaction event: {e}")
    
    async def _trigger_compaction(self, context_id: str, 
                                 strategy: CompactionStrategy):
        """Trigger automatic compaction"""
        try:
            result = await self.compact_context(context_id, strategy)
            if result['success']:
                self.logger.info(f"Auto-compaction successful: {result.get('tokens_saved', 0)} tokens saved")
            else:
                self.logger.warning(f"Auto-compaction failed: {result.get('error', 'Unknown error')}")
        except Exception as e:
            self.logger.error(f"Auto-compaction error: {e}")
    
    def _calculate_total_tokens(self, context_id: str) -> int:
        """Calculate total tokens in a context"""
        if context_id not in self.active_contexts:
            return 0
        
        total = 0
        for segment_id in self.active_contexts[context_id]:
            if segment_id in self.context_segments:
                total += self.context_segments[segment_id].tokens_estimate
            elif segment_id in self.compressed_contexts:
                total += self.compressed_contexts[segment_id]['metadata'].get('compressed_tokens', 0)
        
        return total
    
    def _generate_segment_id(self, content: Any, content_type: str) -> str:
        """Generate unique segment ID"""
        content_hash = hashlib.sha256(str(content).encode()).hexdigest()[:16]
        timestamp = int(time.time() * 1000)
        return f"{content_type}_{content_hash}_{timestamp}"
    
    def _estimate_tokens(self, content: Any) -> int:
        """Estimate token count for content"""
        content_str = str(content)
        # Rough estimation: ~4 characters per token for English text
        return max(len(content_str) // 4, 1)
    
    def _calculate_semantic_hash(self, content: Any) -> str:
        """Calculate semantic hash for deduplication"""
        # Simple approach - in production would use more sophisticated semantic hashing
        content_str = str(content).lower()
        # Remove common words and punctuation for semantic similarity
        words = re.findall(r'\w+', content_str)
        significant_words = [w for w in words if len(w) > 3][:50]  # Limit for performance
        semantic_content = ' '.join(sorted(significant_words))
        return hashlib.sha256(semantic_content.encode()).hexdigest()[:16]
    
    def start_monitoring(self):
        """Start background monitoring for automatic compaction"""
        if not self.monitoring_enabled:
            self.monitoring_enabled = True
            self._stop_monitoring.clear()
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.info("Context compaction monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        if self.monitoring_enabled:
            self.monitoring_enabled = False
            self._stop_monitoring.set()
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)
            self.logger.info("Context compaction monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop for automatic compaction"""
        while not self._stop_monitoring.is_set():
            try:
                for context_id in list(self.active_contexts.keys()):
                    total_tokens = self._calculate_total_tokens(context_id)
                    
                    # Check for emergency compaction
                    if total_tokens > self.config['emergency_threshold']:
                        asyncio.run(self._trigger_compaction(context_id, CompactionStrategy.EMERGENCY))
                    
                    # Check for regular compaction
                    elif total_tokens > self.config['compaction_threshold']:
                        asyncio.run(self._trigger_compaction(context_id, CompactionStrategy.ADAPTIVE))
                
                # Sleep for configured interval
                self._stop_monitoring.wait(self.config['compaction_interval'])
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                self._stop_monitoring.wait(60)  # Wait 1 minute on error
    
    async def restore_context(self, compressed_id: str) -> Dict[str, Any]:
        """Restore compressed context (rollback capability)"""
        if compressed_id not in self.compressed_contexts:
            return {'success': False, 'error': 'Compressed context not found'}
        
        try:
            compressed_context = self.compressed_contexts[compressed_id]
            
            # This is a simplified restoration - in practice would need
            # to restore original segments or provide expansion capability
            restored_content = {
                'compressed_id': compressed_id,
                'content': compressed_context['compressed_content'],
                'metadata': compressed_context['metadata'],
                'restoration_timestamp': datetime.now()
            }
            
            return {
                'success': True,
                'restored_content': restored_content,
                'compression_ratio': compressed_context['compression_ratio']
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_context_summary(self, context_id: str = 'main') -> Dict[str, Any]:
        """Get summary of current context state"""
        if context_id not in self.active_contexts:
            return {'error': f'Context {context_id} not found'}
        
        segment_ids = self.active_contexts[context_id]
        
        # Analyze segments
        total_segments = len(segment_ids)
        compressed_segments = len([sid for sid in segment_ids if sid in self.compressed_contexts])
        regular_segments = total_segments - compressed_segments
        
        total_tokens = self._calculate_total_tokens(context_id)
        
        # Categorize by type and importance
        type_counts = defaultdict(int)
        importance_counts = defaultdict(int)
        
        for segment_id in segment_ids:
            if segment_id in self.context_segments:
                segment = self.context_segments[segment_id]
                type_counts[segment.content_type] += 1
                importance_counts[segment.importance.name] += 1
        
        return {
            'context_id': context_id,
            'total_segments': total_segments,
            'regular_segments': regular_segments,
            'compressed_segments': compressed_segments,
            'total_tokens': total_tokens,
            'utilization_percent': (total_tokens / self.config['max_context_tokens']) * 100,
            'compaction_needed': total_tokens > self.config['compaction_threshold'],
            'emergency_level': total_tokens > self.config['emergency_threshold'],
            'segments_by_type': dict(type_counts),
            'segments_by_importance': dict(importance_counts),
            'last_compaction': self.stats.last_compaction.isoformat() if self.stats.last_compaction else None
        }
    
    def get_compaction_statistics(self) -> Dict[str, Any]:
        """Get comprehensive compaction statistics"""
        return {
            'stats': asdict(self.stats),
            'performance_metrics': self.performance_metrics,
            'config': self.config,
            'active_contexts': len(self.active_contexts),
            'total_segments': len(self.context_segments),
            'compressed_contexts': len(self.compressed_contexts),
            'monitoring_enabled': self.monitoring_enabled,
            'current_strategy': self.current_strategy.value
        }
    
    def set_strategy(self, strategy: CompactionStrategy):
        """Set current compaction strategy"""
        self.current_strategy = strategy
        self.logger.info(f"Compaction strategy set to: {strategy.value}")
    
    def configure(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if key in self.config:
                old_value = self.config[key]
                self.config[key] = value
                self.logger.info(f"Configuration updated: {key} = {value} (was {old_value})")
            else:
                self.logger.warning(f"Unknown configuration key: {key}")
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register event handler for compaction events"""
        self.event_handlers[event_type].append(handler)
        self.logger.debug(f"Registered handler for event: {event_type}")
    
    def _trigger_event(self, event_type: str, data: Dict[str, Any]):
        """Trigger event handlers"""
        for handler in self.event_handlers.get(event_type, []):
            try:
                handler(data)
            except Exception as e:
                self.logger.error(f"Event handler error for {event_type}: {e}")
    
    async def analyze_context_patterns(self, context_id: str = 'main') -> Dict[str, Any]:
        """Analyze patterns in context for optimization insights"""
        if context_id not in self.active_contexts:
            return {'error': f'Context {context_id} not found'}
        
        segment_ids = self.active_contexts[context_id]
        segments = [self.context_segments[sid] for sid in segment_ids 
                   if sid in self.context_segments]
        
        if not segments:
            return {'error': 'No segments to analyze'}
        
        # Pattern analysis
        patterns = {
            'temporal_distribution': self._analyze_temporal_patterns(segments),
            'content_type_distribution': self._analyze_content_types(segments),
            'importance_distribution': self._analyze_importance_patterns(segments),
            'access_patterns': self._analyze_access_patterns(segments),
            'size_distribution': self._analyze_size_patterns(segments),
            'relationship_density': self._analyze_relationship_density(segments)
        }
        
        # Optimization recommendations
        recommendations = self._generate_optimization_recommendations(patterns)
        
        return {
            'context_id': context_id,
            'analysis_timestamp': datetime.now().isoformat(),
            'segments_analyzed': len(segments),
            'patterns': patterns,
            'recommendations': recommendations
        }
    
    def _analyze_temporal_patterns(self, segments: List[ContextSegment]) -> Dict[str, Any]:
        """Analyze temporal distribution of segments"""
        now = datetime.now()
        time_buckets = {
            'last_hour': 0,
            'last_day': 0,
            'last_week': 0,
            'last_month': 0,
            'older': 0
        }
        
        for segment in segments:
            age = now - segment.timestamp
            if age < timedelta(hours=1):
                time_buckets['last_hour'] += 1
            elif age < timedelta(days=1):
                time_buckets['last_day'] += 1
            elif age < timedelta(weeks=1):
                time_buckets['last_week'] += 1
            elif age < timedelta(days=30):
                time_buckets['last_month'] += 1
            else:
                time_buckets['older'] += 1
        
        return time_buckets
    
    def _analyze_content_types(self, segments: List[ContextSegment]) -> Dict[str, Any]:
        """Analyze content type distribution"""
        type_counts = defaultdict(int)
        type_tokens = defaultdict(int)
        
        for segment in segments:
            type_counts[segment.content_type] += 1
            type_tokens[segment.content_type] += segment.tokens_estimate
        
        return {
            'counts': dict(type_counts),
            'tokens': dict(type_tokens)
        }
    
    def _analyze_importance_patterns(self, segments: List[ContextSegment]) -> Dict[str, Any]:
        """Analyze importance distribution"""
        importance_counts = defaultdict(int)
        importance_tokens = defaultdict(int)
        
        for segment in segments:
            importance_counts[segment.importance.name] += 1
            importance_tokens[segment.importance.name] += segment.tokens_estimate
        
        return {
            'counts': dict(importance_counts),
            'tokens': dict(importance_tokens)
        }
    
    def _analyze_access_patterns(self, segments: List[ContextSegment]) -> Dict[str, Any]:
        """Analyze access frequency patterns"""
        access_counts = [s.access_count for s in segments]
        
        if not access_counts:
            return {}
        
        return {
            'average_access': sum(access_counts) / len(access_counts),
            'max_access': max(access_counts),
            'min_access': min(access_counts),
            'high_access_segments': len([c for c in access_counts if c > 5]),
            'zero_access_segments': len([c for c in access_counts if c == 0])
        }
    
    def _analyze_size_patterns(self, segments: List[ContextSegment]) -> Dict[str, Any]:
        """Analyze segment size patterns"""
        sizes = [s.tokens_estimate for s in segments]
        
        if not sizes:
            return {}
        
        return {
            'total_tokens': sum(sizes),
            'average_size': sum(sizes) / len(sizes),
            'max_size': max(sizes),
            'min_size': min(sizes),
            'large_segments': len([s for s in sizes if s > 1000]),  # >1k tokens
            'small_segments': len([s for s in sizes if s < 100])    # <100 tokens
        }
    
    def _analyze_relationship_density(self, segments: List[ContextSegment]) -> Dict[str, Any]:
        """Analyze relationship density between segments"""
        total_relationships = 0
        segments_with_relationships = 0
        
        for segment in segments:
            relationship_count = len(segment.relationships)
            total_relationships += relationship_count
            if relationship_count > 0:
                segments_with_relationships += 1
        
        total_segments = len(segments)
        
        return {
            'total_relationships': total_relationships,
            'segments_with_relationships': segments_with_relationships,
            'relationship_density': total_relationships / total_segments if total_segments > 0 else 0,
            'connected_segment_ratio': segments_with_relationships / total_segments if total_segments > 0 else 0
        }
    
    def _generate_optimization_recommendations(self, patterns: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on patterns"""
        recommendations = []
        
        # Temporal recommendations
        temporal = patterns.get('temporal_distribution', {})
        if temporal.get('older', 0) > temporal.get('last_week', 0):
            recommendations.append("Consider aggressive compression of older segments")
        
        # Content type recommendations
        content_types = patterns.get('content_type_distribution', {}).get('counts', {})
        if content_types.get('conversation', 0) > 20:
            recommendations.append("High conversation volume - consider hierarchical compression")
        
        # Importance recommendations
        importance = patterns.get('importance_distribution', {}).get('counts', {})
        if importance.get('MINIMAL', 0) + importance.get('LOW', 0) > 10:
            recommendations.append("Many low-importance segments available for aggressive compression")
        
        # Access pattern recommendations
        access = patterns.get('access_patterns', {})
        if access.get('zero_access_segments', 0) > 5:
            recommendations.append("Multiple zero-access segments are prime compression candidates")
        
        # Size recommendations
        size = patterns.get('size_distribution', {})
        if size.get('large_segments', 0) > 5:
            recommendations.append("Large segments present - consider structural compression")
        
        # Relationship recommendations
        relationships = patterns.get('relationship_density', {})
        if relationships.get('relationship_density', 0) < 0.1:
            recommendations.append("Low relationship density - segments can be compressed independently")
        
        if not recommendations:
            recommendations.append("Context appears well-optimized for current strategy")
        
        return recommendations
    
    async def shutdown(self):
        """Gracefully shutdown the context compact system"""
        self.logger.info("Shutting down Auto Context Compact system")
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Save all segments to database
        for segment in self.context_segments.values():
            self._save_segment_to_db(segment)
        
        # Clear memory
        self.context_segments.clear()
        self.active_contexts.clear()
        
        self.logger.info("Auto Context Compact shutdown complete")


# Global instance
_global_context_compact: Optional[AutoContextCompact] = None

def get_context_compact(project_path: Optional[Path] = None) -> AutoContextCompact:
    """Get global context compact instance"""
    global _global_context_compact
    if _global_context_compact is None:
        _global_context_compact = AutoContextCompact(project_path)
    return _global_context_compact

# Convenience functions
async def add_context(content: Any, content_type: str = 'conversation', 
                     context_id: str = 'main', metadata: Optional[Dict[str, Any]] = None) -> str:
    """Add content to context management"""
    compact = get_context_compact()
    return await compact.add_context(content, content_type, context_id, metadata)

async def compact_context(context_id: str = 'main', 
                         strategy: Optional[CompactionStrategy] = None) -> Dict[str, Any]:
    """Trigger context compaction"""
    compact = get_context_compact()
    return await compact.compact_context(context_id, strategy)

def get_context_summary(context_id: str = 'main') -> Dict[str, Any]:
    """Get context summary"""
    compact = get_context_compact()
    return compact.get_context_summary(context_id)

def get_compaction_stats() -> Dict[str, Any]:
    """Get compaction statistics"""
    compact = get_context_compact()
    return compact.get_compaction_statistics()