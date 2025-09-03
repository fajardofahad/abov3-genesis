"""
ABOV3 Genesis - Context Intelligence System
Advanced context analysis, pattern recognition, and intelligent decision-making
for Claude-level context management and optimization.
"""

import asyncio
import json
import logging
import threading
import time
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum, auto
import pickle
import sqlite3
import hashlib
import numpy as np
from abc import ABC, abstractmethod

# Import existing components
from .auto_context_compact import (
    AutoContextCompact,
    ContextSegment,
    ContextImportance,
    CompactionStrategy,
    CompressionMethod,
    get_context_compact
)
from .memory_manager import (
    MemoryManager,
    MemoryEntry,
    MemoryType,
    Priority,
    get_memory_manager
)


class ContextPattern(Enum):
    """Types of context patterns"""
    SEQUENTIAL = "sequential"           # Sequential conversation flow
    BRANCHING = "branching"            # Multiple conversation branches
    REPETITIVE = "repetitive"          # Repeated topics/questions
    EXPLORATORY = "exploratory"        # Information gathering
    PROBLEM_SOLVING = "problem_solving" # Debugging/fixing issues
    CREATIVE = "creative"              # Content creation/generation
    ANALYTICAL = "analytical"          # Data analysis/review
    INSTRUCTIONAL = "instructional"    # Teaching/explaining
    COLLABORATIVE = "collaborative"    # Multi-turn collaboration
    MAINTENANCE = "maintenance"        # Code maintenance/updates


class ContextState(Enum):
    """Context states for intelligent management"""
    BUILDING = auto()          # Actively building context
    STABLE = auto()           # Context is stable and useful
    DEGRADING = auto()        # Context quality degrading
    FRAGMENTED = auto()       # Context becoming fragmented
    OVERLOADED = auto()       # Context approaching limits
    CRITICAL = auto()         # Context needs immediate attention
    OPTIMIZED = auto()        # Context recently optimized


class IntentType(Enum):
    """User intent types for context optimization"""
    CODE_DEVELOPMENT = "code_development"
    DEBUGGING = "debugging"
    LEARNING = "learning"
    EXPLORATION = "exploration"
    PROBLEM_SOLVING = "problem_solving"
    CREATIVE_WORK = "creative_work"
    ANALYSIS = "analysis"
    COLLABORATION = "collaboration"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"


@dataclass
class ContextAnalysis:
    """Comprehensive context analysis results"""
    context_id: str
    timestamp: datetime
    total_segments: int
    total_tokens: int
    pattern: ContextPattern
    state: ContextState
    intent: IntentType
    quality_score: float  # 0-1
    coherence_score: float  # 0-1
    relevance_score: float  # 0-1
    efficiency_score: float  # 0-1
    recommendations: List[str]
    optimization_opportunities: List[Dict[str, Any]]
    predicted_growth: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextPrediction:
    """Predictions about context evolution"""
    context_id: str
    prediction_horizon: timedelta
    predicted_tokens: int
    predicted_segments: int
    compaction_probability: float
    optimal_strategy: CompactionStrategy
    confidence: float
    triggers: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntelligenceMetrics:
    """Metrics for context intelligence system"""
    total_analyses: int = 0
    correct_predictions: int = 0
    optimization_successes: int = 0
    pattern_detections: int = 0
    intent_recognitions: int = 0
    average_quality_improvement: float = 0.0
    average_analysis_time: float = 0.0
    prediction_accuracy: float = 0.0
    user_satisfaction_score: float = 0.0


class PatternDetector(ABC):
    """Abstract base for pattern detection"""
    
    @abstractmethod
    async def detect_patterns(self, segments: List[ContextSegment]) -> List[ContextPattern]:
        """Detect patterns in context segments"""
        pass
    
    @abstractmethod
    async def predict_evolution(self, segments: List[ContextSegment], 
                               pattern: ContextPattern) -> ContextPrediction:
        """Predict how context will evolve"""
        pass


class ClaudeStylePatternDetector(PatternDetector):
    """Claude-style intelligent pattern detection with advanced heuristics"""
    
    def __init__(self):
        self.pattern_signatures = {
            ContextPattern.SEQUENTIAL: {
                'keywords': ['next', 'then', 'after', 'following', 'continue'],
                'structure_indicators': ['step', 'phase', 'part', 'section'],
                'time_indicators': ['now', 'later', 'afterwards', 'subsequently']
            },
            ContextPattern.BRANCHING: {
                'keywords': ['alternatively', 'option', 'choice', 'different', 'another'],
                'structure_indicators': ['branch', 'variation', 'approach', 'method'],
                'decision_indicators': ['if', 'unless', 'depending', 'case', 'scenario']
            },
            ContextPattern.REPETITIVE: {
                'keywords': ['again', 'repeat', 'similar', 'same', 'duplicate'],
                'structure_indicators': ['pattern', 'template', 'format', 'style'],
                'frequency_indicators': ['often', 'frequently', 'repeatedly', 'multiple']
            },
            ContextPattern.PROBLEM_SOLVING: {
                'keywords': ['fix', 'solve', 'debug', 'issue', 'problem', 'error'],
                'structure_indicators': ['solution', 'resolution', 'fix', 'workaround'],
                'urgency_indicators': ['urgent', 'critical', 'blocking', 'broken']
            },
            ContextPattern.CREATIVE: {
                'keywords': ['create', 'design', 'generate', 'build', 'make', 'new'],
                'structure_indicators': ['idea', 'concept', 'vision', 'innovation'],
                'quality_indicators': ['creative', 'original', 'unique', 'novel']
            },
            ContextPattern.ANALYTICAL: {
                'keywords': ['analyze', 'examine', 'review', 'study', 'investigate'],
                'structure_indicators': ['data', 'information', 'results', 'findings'],
                'method_indicators': ['compare', 'evaluate', 'assess', 'measure']
            }
        }
        
        self.intent_patterns = {
            IntentType.CODE_DEVELOPMENT: [
                r'(?i)(creat|writ|implement|develop|build).*(?:function|class|method|code)',
                r'(?i)(add|implement).*feature',
                r'(?i)code.*(?:generat|creat)'
            ],
            IntentType.DEBUGGING: [
                r'(?i)(debug|fix|error|bug|issue|problem)',
                r'(?i)(not work|doesn\'t work|broken|failing)',
                r'(?i)(exception|traceback|stack trace)'
            ],
            IntentType.LEARNING: [
                r'(?i)(learn|understand|explain|teach|show)',
                r'(?i)(how.*work|what.*do|why.*happen)',
                r'(?i)(tutorial|guide|documentation)'
            ],
            IntentType.EXPLORATION: [
                r'(?i)(explor|investigat|research|discover)',
                r'(?i)(what if|possibl|option|alternativ)',
                r'(?i)(experiment|try|test|attempt)'
            ]
        }
    
    async def detect_patterns(self, segments: List[ContextSegment]) -> List[ContextPattern]:
        """Detect context patterns using semantic analysis and heuristics"""
        if not segments:
            return []
        
        pattern_scores = defaultdict(float)
        total_content = ""
        
        # Collect all content for analysis
        for segment in segments:
            content = str(segment.content).lower()
            total_content += " " + content
            
            # Score patterns based on keywords and structure
            for pattern, signatures in self.pattern_signatures.items():
                score = 0.0
                
                # Keyword matching
                for keyword in signatures.get('keywords', []):
                    score += content.count(keyword) * 1.0
                
                # Structure indicator matching
                for indicator in signatures.get('structure_indicators', []):
                    score += content.count(indicator) * 1.5
                
                # Special indicator matching
                for special_key in ['time_indicators', 'decision_indicators', 
                                   'frequency_indicators', 'urgency_indicators',
                                   'quality_indicators', 'method_indicators']:
                    for indicator in signatures.get(special_key, []):
                        score += content.count(indicator) * 2.0
                
                pattern_scores[pattern] += score
        
        # Analyze temporal patterns
        await self._analyze_temporal_patterns(segments, pattern_scores)
        
        # Analyze structural patterns
        await self._analyze_structural_patterns(segments, pattern_scores)
        
        # Analyze conversation flow
        await self._analyze_conversation_flow(segments, pattern_scores)
        
        # Return patterns sorted by score
        detected_patterns = []
        for pattern, score in sorted(pattern_scores.items(), key=lambda x: x[1], reverse=True):
            if score > 0.5:  # Threshold for pattern detection
                detected_patterns.append(pattern)
        
        return detected_patterns[:3]  # Return top 3 patterns
    
    async def _analyze_temporal_patterns(self, segments: List[ContextSegment], 
                                        pattern_scores: Dict[ContextPattern, float]):
        """Analyze temporal patterns in segments"""
        if len(segments) < 2:
            return
        
        # Sort by timestamp
        sorted_segments = sorted(segments, key=lambda s: s.timestamp)
        
        # Analyze time gaps
        time_gaps = []
        for i in range(1, len(sorted_segments)):
            gap = (sorted_segments[i].timestamp - sorted_segments[i-1].timestamp).total_seconds()
            time_gaps.append(gap)
        
        if time_gaps:
            avg_gap = sum(time_gaps) / len(time_gaps) if time_gaps else 0
            
            # Sequential pattern: consistent short gaps
            if avg_gap < 300 and all(gap < 600 for gap in time_gaps):  # < 5 min avg, all < 10 min
                pattern_scores[ContextPattern.SEQUENTIAL] += 2.0
            
            # Repetitive pattern: regular intervals
            if len(set(int(gap/60) for gap in time_gaps)) <= 2:  # Similar minute intervals
                pattern_scores[ContextPattern.REPETITIVE] += 1.5
            
            # Exploratory pattern: varying gaps indicating thinking/research time
            if max(time_gaps) > avg_gap * 3:  # Some long pauses
                pattern_scores[ContextPattern.EXPLORATORY] += 1.0
    
    async def _analyze_structural_patterns(self, segments: List[ContextSegment], 
                                          pattern_scores: Dict[ContextPattern, float]):
        """Analyze structural patterns in content"""
        content_types = [s.content_type for s in segments]
        importance_levels = [s.importance for s in segments]
        
        # Analyze content type distribution
        type_counts = defaultdict(int)
        for ctype in content_types:
            type_counts[ctype] += 1
        
        # High code content suggests development pattern
        if type_counts.get('code', 0) > len(segments) * 0.3:
            pattern_scores[ContextPattern.PROBLEM_SOLVING] += 1.0
            if any('create' in str(s.content).lower() for s in segments):
                pattern_scores[ContextPattern.CREATIVE] += 1.5
        
        # High error content suggests debugging pattern
        if type_counts.get('error', 0) > 2:
            pattern_scores[ContextPattern.PROBLEM_SOLVING] += 2.0
        
        # Mixed content types suggest exploratory pattern
        if len(type_counts) > 3:
            pattern_scores[ContextPattern.EXPLORATORY] += 1.0
        
        # Analyze importance distribution
        high_importance = sum(1 for imp in importance_levels 
                             if imp.value >= ContextImportance.HIGH.value)
        
        if high_importance > len(segments) * 0.5:
            pattern_scores[ContextPattern.COLLABORATIVE] += 1.0
    
    async def _analyze_conversation_flow(self, segments: List[ContextSegment], 
                                        pattern_scores: Dict[ContextPattern, float]):
        """Analyze conversation flow patterns"""
        conversation_segments = [s for s in segments if s.content_type == 'conversation']
        
        if len(conversation_segments) < 3:
            return
        
        # Analyze question-answer patterns
        questions = 0
        answers = 0
        
        for segment in conversation_segments:
            content = str(segment.content).lower()
            if any(q in content for q in ['?', 'how', 'what', 'why', 'when', 'where']):
                questions += 1
            if any(a in content for a in ['because', 'the answer', 'here\'s', 'you can']):
                answers += 1
        
        # Question-heavy suggests learning pattern
        if questions > answers * 1.5:
            pattern_scores[ContextPattern.INSTRUCTIONAL] += 1.5
        
        # Balanced Q&A suggests collaborative pattern
        if abs(questions - answers) <= 2:
            pattern_scores[ContextPattern.COLLABORATIVE] += 1.0
    
    async def predict_evolution(self, segments: List[ContextSegment], 
                               pattern: ContextPattern) -> ContextPrediction:
        """Predict context evolution based on current pattern"""
        if not segments:
            return self._create_default_prediction()
        
        # Analyze current trajectory
        recent_segments = sorted(segments, key=lambda s: s.timestamp)[-10:]  # Last 10 segments
        current_tokens = sum(s.tokens_estimate for s in segments)
        recent_growth_rate = self._calculate_growth_rate(recent_segments)
        
        # Pattern-specific predictions
        prediction_horizon = timedelta(hours=2)  # Default 2-hour prediction
        
        if pattern == ContextPattern.SEQUENTIAL:
            # Sequential patterns tend to have steady growth
            predicted_tokens = int(current_tokens * (1 + recent_growth_rate * 0.5))
            predicted_segments = len(segments) + max(5, len(recent_segments))
            compaction_probability = 0.3
            optimal_strategy = CompactionStrategy.CONSERVATIVE
            
        elif pattern == ContextPattern.PROBLEM_SOLVING:
            # Problem-solving can have rapid growth bursts
            predicted_tokens = int(current_tokens * (1 + recent_growth_rate * 1.2))
            predicted_segments = len(segments) + max(8, len(recent_segments) * 2)
            compaction_probability = 0.6
            optimal_strategy = CompactionStrategy.BALANCED
            
        elif pattern == ContextPattern.CREATIVE:
            # Creative work tends to have variable but substantial growth
            predicted_tokens = int(current_tokens * (1 + recent_growth_rate * 0.8))
            predicted_segments = len(segments) + max(6, len(recent_segments))
            compaction_probability = 0.4
            optimal_strategy = CompactionStrategy.CONSERVATIVE
            
        elif pattern == ContextPattern.EXPLORATORY:
            # Exploratory patterns can grow rapidly and unpredictably
            predicted_tokens = int(current_tokens * (1 + recent_growth_rate * 1.5))
            predicted_segments = len(segments) + max(10, len(recent_segments) * 3)
            compaction_probability = 0.7
            optimal_strategy = CompactionStrategy.ADAPTIVE
            
        else:
            # Default prediction for other patterns
            predicted_tokens = int(current_tokens * (1 + recent_growth_rate))
            predicted_segments = len(segments) + len(recent_segments)
            compaction_probability = 0.5
            optimal_strategy = CompactionStrategy.BALANCED
        
        # Calculate confidence based on pattern consistency
        confidence = self._calculate_prediction_confidence(segments, pattern)
        
        # Identify trigger conditions
        triggers = self._identify_triggers(pattern, current_tokens)
        
        return ContextPrediction(
            context_id='prediction',
            prediction_horizon=prediction_horizon,
            predicted_tokens=predicted_tokens,
            predicted_segments=predicted_segments,
            compaction_probability=compaction_probability,
            optimal_strategy=optimal_strategy,
            confidence=confidence,
            triggers=triggers,
            metadata={
                'pattern': pattern.value,
                'current_tokens': current_tokens,
                'growth_rate': recent_growth_rate,
                'analysis_timestamp': datetime.now().isoformat()
            }
        )
    
    def _calculate_growth_rate(self, segments: List[ContextSegment]) -> float:
        """Calculate growth rate from recent segments"""
        if len(segments) < 2:
            return 0.1  # Default growth rate
        
        sorted_segments = sorted(segments, key=lambda s: s.timestamp)
        
        # Calculate token growth over time
        time_span = (sorted_segments[-1].timestamp - sorted_segments[0].timestamp).total_seconds()
        if time_span == 0:
            return 0.1
        
        token_growth = sum(s.tokens_estimate for s in sorted_segments)
        growth_rate = token_growth / time_span  # Tokens per second
        
        # Normalize to a reasonable range
        return min(max(growth_rate * 3600, 0.01), 2.0)  # Convert to hourly rate, cap at 200%
    
    def _calculate_prediction_confidence(self, segments: List[ContextSegment], 
                                        pattern: ContextPattern) -> float:
        """Calculate confidence in prediction based on pattern consistency"""
        if len(segments) < 3:
            return 0.5
        
        # Analyze consistency of pattern indicators
        recent_segments = segments[-5:]  # Last 5 segments
        pattern_score = 0.0
        
        for segment in recent_segments:
            content = str(segment.content).lower()
            
            # Check if segment content matches pattern expectations
            if pattern in self.pattern_signatures:
                signatures = self.pattern_signatures[pattern]
                segment_score = 0.0
                
                for keyword in signatures.get('keywords', []):
                    segment_score += content.count(keyword)
                
                for indicator in signatures.get('structure_indicators', []):
                    segment_score += content.count(indicator) * 1.5
                
                pattern_score += min(segment_score, 5.0)  # Cap individual segment score
        
        # Normalize confidence score
        max_possible_score = len(recent_segments) * 5.0
        confidence = min(pattern_score / max_possible_score, 1.0) if max_possible_score > 0 else 0.5
        
        return max(confidence, 0.1)  # Minimum confidence of 10%
    
    def _identify_triggers(self, pattern: ContextPattern, current_tokens: int) -> List[str]:
        """Identify trigger conditions for compaction"""
        triggers = []
        
        # Universal triggers
        if current_tokens > 80000:
            triggers.append("approaching_token_limit")
        
        if current_tokens > 100000:
            triggers.append("token_limit_exceeded")
        
        # Pattern-specific triggers
        if pattern == ContextPattern.PROBLEM_SOLVING:
            triggers.extend(["error_resolved", "solution_found", "debugging_complete"])
        
        elif pattern == ContextPattern.CREATIVE:
            triggers.extend(["creative_phase_complete", "output_generated", "iteration_finished"])
        
        elif pattern == ContextPattern.EXPLORATORY:
            triggers.extend(["exploration_complete", "information_gathered", "research_finished"])
        
        elif pattern == ContextPattern.SEQUENTIAL:
            triggers.extend(["sequence_complete", "next_phase_ready", "milestone_reached"])
        
        return triggers
    
    def _create_default_prediction(self) -> ContextPrediction:
        """Create default prediction when no data available"""
        return ContextPrediction(
            context_id='default',
            prediction_horizon=timedelta(hours=1),
            predicted_tokens=1000,
            predicted_segments=10,
            compaction_probability=0.3,
            optimal_strategy=CompactionStrategy.BALANCED,
            confidence=0.3,
            triggers=["time_based"],
            metadata={'type': 'default_prediction'}
        )


class IntentRecognizer:
    """Advanced intent recognition system"""
    
    def __init__(self):
        self.intent_classifiers = {
            IntentType.CODE_DEVELOPMENT: self._classify_code_development,
            IntentType.DEBUGGING: self._classify_debugging,
            IntentType.LEARNING: self._classify_learning,
            IntentType.EXPLORATION: self._classify_exploration,
            IntentType.PROBLEM_SOLVING: self._classify_problem_solving,
            IntentType.CREATIVE_WORK: self._classify_creative_work,
            IntentType.ANALYSIS: self._classify_analysis,
            IntentType.COLLABORATION: self._classify_collaboration,
            IntentType.MAINTENANCE: self._classify_maintenance
        }
    
    async def recognize_intent(self, segments: List[ContextSegment]) -> IntentType:
        """Recognize user intent from context segments"""
        if not segments:
            return IntentType.UNKNOWN
        
        intent_scores = defaultdict(float)
        
        # Analyze each segment for intent indicators
        for segment in segments:
            content = str(segment.content).lower()
            
            for intent_type, classifier in self.intent_classifiers.items():
                score = await classifier(content, segment)
                intent_scores[intent_type] += score
        
        # Return highest scoring intent
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            return best_intent[0] if best_intent[1] > 0.5 else IntentType.UNKNOWN
        
        return IntentType.UNKNOWN
    
    async def _classify_code_development(self, content: str, segment: ContextSegment) -> float:
        """Classify code development intent"""
        score = 0.0
        
        # Keywords
        dev_keywords = ['create', 'implement', 'develop', 'build', 'write', 'code', 
                       'function', 'class', 'method', 'feature', 'application']
        score += sum(1 for kw in dev_keywords if kw in content)
        
        # Content type boost
        if segment.content_type == 'code':
            score += 2.0
        
        # Importance boost
        if segment.importance.value >= ContextImportance.HIGH.value:
            score += 1.0
        
        return min(score / 5.0, 1.0)  # Normalize to 0-1
    
    async def _classify_debugging(self, content: str, segment: ContextSegment) -> float:
        """Classify debugging intent"""
        score = 0.0
        
        debug_keywords = ['debug', 'fix', 'error', 'bug', 'issue', 'problem', 
                         'exception', 'traceback', 'broken', 'failing']
        score += sum(2 for kw in debug_keywords if kw in content)
        
        if segment.content_type == 'error':
            score += 3.0
        
        if 'traceback' in content or 'exception' in content:
            score += 2.0
        
        return min(score / 7.0, 1.0)
    
    async def _classify_learning(self, content: str, segment: ContextSegment) -> float:
        """Classify learning intent"""
        score = 0.0
        
        learning_keywords = ['learn', 'understand', 'explain', 'teach', 'show', 
                           'how', 'what', 'why', 'tutorial', 'guide', 'example']
        score += sum(1 for kw in learning_keywords if kw in content)
        
        # Question patterns
        if '?' in content or any(q in content for q in ['how to', 'what is', 'why does']):
            score += 2.0
        
        return min(score / 5.0, 1.0)
    
    async def _classify_exploration(self, content: str, segment: ContextSegment) -> float:
        """Classify exploration intent"""
        score = 0.0
        
        explore_keywords = ['explore', 'investigate', 'research', 'discover', 
                          'what if', 'possible', 'option', 'alternative', 'try']
        score += sum(1 for kw in explore_keywords if kw in content)
        
        if segment.content_type == 'conversation' and len(content) > 500:
            score += 1.0  # Long conversations often exploratory
        
        return min(score / 4.0, 1.0)
    
    async def _classify_problem_solving(self, content: str, segment: ContextSegment) -> float:
        """Classify problem-solving intent"""
        score = 0.0
        
        problem_keywords = ['solve', 'solution', 'resolve', 'fix', 'address', 
                          'handle', 'approach', 'strategy', 'method']
        score += sum(1 for kw in problem_keywords if kw in content)
        
        if segment.importance == ContextImportance.CRITICAL:
            score += 2.0
        
        return min(score / 4.0, 1.0)
    
    async def _classify_creative_work(self, content: str, segment: ContextSegment) -> float:
        """Classify creative work intent"""
        score = 0.0
        
        creative_keywords = ['create', 'design', 'generate', 'build', 'make', 
                           'new', 'original', 'innovative', 'creative', 'idea']
        score += sum(1 for kw in creative_keywords if kw in content)
        
        if 'generate' in content or 'create' in content:
            score += 1.5
        
        return min(score / 4.0, 1.0)
    
    async def _classify_analysis(self, content: str, segment: ContextSegment) -> float:
        """Classify analysis intent"""
        score = 0.0
        
        analysis_keywords = ['analyze', 'examine', 'review', 'study', 'investigate', 
                           'compare', 'evaluate', 'assess', 'data', 'results']
        score += sum(1 for kw in analysis_keywords if kw in content)
        
        return min(score / 4.0, 1.0)
    
    async def _classify_collaboration(self, content: str, segment: ContextSegment) -> float:
        """Classify collaboration intent"""
        score = 0.0
        
        collab_keywords = ['collaborate', 'together', 'team', 'share', 'discuss', 
                         'feedback', 'review', 'input', 'suggestion']
        score += sum(1 for kw in collab_keywords if kw in content)
        
        if segment.access_count > 3:  # Frequently accessed suggests collaboration
            score += 1.0
        
        return min(score / 4.0, 1.0)
    
    async def _classify_maintenance(self, content: str, segment: ContextSegment) -> float:
        """Classify maintenance intent"""
        score = 0.0
        
        maint_keywords = ['maintain', 'update', 'upgrade', 'patch', 'modify', 
                         'improve', 'optimize', 'refactor', 'cleanup']
        score += sum(1 for kw in maint_keywords if kw in content)
        
        return min(score / 3.0, 1.0)


class ContextIntelligence:
    """
    Advanced Context Intelligence System providing Claude-level context analysis,
    pattern recognition, intent understanding, and intelligent optimization.
    """
    
    def __init__(self, project_path: Optional[Path] = None):
        self.project_path = project_path or Path.cwd()
        
        # Initialize components
        self.context_compact = get_context_compact(self.project_path)
        self.memory_manager = get_memory_manager(self.project_path)
        self.pattern_detector = ClaudeStylePatternDetector()
        self.intent_recognizer = IntentRecognizer()
        
        # Analysis cache
        self.analysis_cache: Dict[str, ContextAnalysis] = {}
        self.prediction_cache: Dict[str, ContextPrediction] = {}
        self.cache_ttl = timedelta(minutes=15)  # Cache validity period
        
        # Intelligence metrics
        self.metrics = IntelligenceMetrics()
        
        # Configuration
        self.config = {
            'analysis_interval': 300,      # Analyze every 5 minutes
            'prediction_horizon': 7200,    # 2 hours prediction
            'quality_threshold': 0.7,      # Quality threshold for recommendations
            'pattern_confidence_threshold': 0.6,
            'intent_confidence_threshold': 0.5,
            'optimization_aggressiveness': 0.5,  # 0-1 scale
            'enable_predictive_compaction': True,
            'enable_intent_optimization': True,
            'max_analysis_history': 100
        }
        
        # Threading
        self.lock = threading.RLock()
        self.monitoring_enabled = False
        self.monitor_thread = None
        self._stop_monitoring = threading.Event()
        
        # Database
        self.db_path = self.project_path / '.abov3' / 'context_intelligence.db'
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
        
        # Setup logging
        self.logger = logging.getLogger('abov3.context_intelligence')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Start monitoring
        self.start_monitoring()
        
        self.logger.info("Context Intelligence System initialized")
    
    def _initialize_database(self):
        """Initialize database for intelligence data"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Context analyses table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS context_analyses (
                        analysis_id TEXT PRIMARY KEY,
                        context_id TEXT,
                        timestamp REAL,
                        total_segments INTEGER,
                        total_tokens INTEGER,
                        pattern TEXT,
                        state TEXT,
                        intent TEXT,
                        quality_score REAL,
                        coherence_score REAL,
                        relevance_score REAL,
                        efficiency_score REAL,
                        recommendations TEXT,
                        optimization_opportunities TEXT,
                        predicted_growth TEXT,
                        metadata TEXT
                    )
                ''')
                
                # Context predictions table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS context_predictions (
                        prediction_id TEXT PRIMARY KEY,
                        context_id TEXT,
                        timestamp REAL,
                        prediction_horizon_seconds REAL,
                        predicted_tokens INTEGER,
                        predicted_segments INTEGER,
                        compaction_probability REAL,
                        optimal_strategy TEXT,
                        confidence REAL,
                        triggers TEXT,
                        metadata TEXT
                    )
                ''')
                
                # Intelligence metrics table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS intelligence_metrics (
                        metric_id TEXT PRIMARY KEY,
                        timestamp REAL,
                        total_analyses INTEGER,
                        correct_predictions INTEGER,
                        optimization_successes INTEGER,
                        pattern_detections INTEGER,
                        intent_recognitions INTEGER,
                        average_quality_improvement REAL,
                        average_analysis_time REAL,
                        prediction_accuracy REAL,
                        user_satisfaction_score REAL
                    )
                ''')
                
                # Create indices
                conn.execute('CREATE INDEX IF NOT EXISTS idx_context_id ON context_analyses(context_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON context_analyses(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_pattern ON context_analyses(pattern)')
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Database initialization error: {e}")
    
    async def analyze_context(self, context_id: str = 'main', 
                             force_refresh: bool = False) -> ContextAnalysis:
        """Perform comprehensive context analysis"""
        start_time = time.time()
        
        # Check cache
        if not force_refresh and context_id in self.analysis_cache:
            cached_analysis = self.analysis_cache[context_id]
            if datetime.now() - cached_analysis.timestamp < self.cache_ttl:
                return cached_analysis
        
        try:
            # Get context data
            context_summary = self.context_compact.get_context_summary(context_id)
            if 'error' in context_summary:
                raise ValueError(f"Context not found: {context_id}")
            
            # Get segments for analysis
            if context_id not in self.context_compact.active_contexts:
                return self._create_empty_analysis(context_id)
            
            segment_ids = self.context_compact.active_contexts[context_id]
            segments = [
                self.context_compact.context_segments[sid] 
                for sid in segment_ids 
                if sid in self.context_compact.context_segments
            ]
            
            if not segments:
                return self._create_empty_analysis(context_id)
            
            # Detect patterns
            patterns = await self.pattern_detector.detect_patterns(segments)
            primary_pattern = patterns[0] if patterns else ContextPattern.SEQUENTIAL
            
            # Recognize intent
            intent = await self.intent_recognizer.recognize_intent(segments)
            
            # Assess context state
            state = await self._assess_context_state(segments, context_summary)
            
            # Calculate quality scores
            quality_scores = await self._calculate_quality_scores(segments, primary_pattern)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                segments, primary_pattern, intent, state, quality_scores
            )
            
            # Identify optimization opportunities
            opportunities = await self._identify_optimization_opportunities(
                segments, primary_pattern, intent, quality_scores
            )
            
            # Predict growth
            predicted_growth = await self._predict_growth(segments, primary_pattern)
            
            # Create analysis
            analysis = ContextAnalysis(
                context_id=context_id,
                timestamp=datetime.now(),
                total_segments=len(segments),
                total_tokens=sum(s.tokens_estimate for s in segments),
                pattern=primary_pattern,
                state=state,
                intent=intent,
                quality_score=quality_scores['overall'],
                coherence_score=quality_scores['coherence'],
                relevance_score=quality_scores['relevance'],
                efficiency_score=quality_scores['efficiency'],
                recommendations=recommendations,
                optimization_opportunities=opportunities,
                predicted_growth=predicted_growth,
                metadata={
                    'patterns_detected': [p.value for p in patterns],
                    'analysis_duration': time.time() - start_time,
                    'segment_types': list(set(s.content_type for s in segments)),
                    'importance_distribution': {
                        imp.name: len([s for s in segments if s.importance == imp])
                        for imp in ContextImportance
                    }
                }
            )
            
            # Cache and save analysis
            self.analysis_cache[context_id] = analysis
            await self._save_analysis(analysis)
            
            # Update metrics
            self.metrics.total_analyses += 1
            analysis_time = time.time() - start_time
            self.metrics.average_analysis_time = (
                (self.metrics.average_analysis_time * (self.metrics.total_analyses - 1) +
                 analysis_time) / self.metrics.total_analyses
            )
            
            if patterns:
                self.metrics.pattern_detections += 1
            
            if intent != IntentType.UNKNOWN:
                self.metrics.intent_recognitions += 1
            
            self.logger.info(f"Context analysis completed for {context_id}: "
                           f"pattern={primary_pattern.value}, intent={intent.value}, "
                           f"quality={quality_scores['overall']:.2f}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Context analysis failed: {e}")
            return self._create_error_analysis(context_id, str(e))
    
    async def _assess_context_state(self, segments: List[ContextSegment], 
                                   context_summary: Dict[str, Any]) -> ContextState:
        """Assess current context state"""
        total_tokens = context_summary.get('total_tokens', 0)
        utilization = context_summary.get('utilization_percent', 0)
        
        # Critical state checks
        if utilization > 95 or total_tokens > 115000:
            return ContextState.CRITICAL
        
        if utilization > 85 or total_tokens > 100000:
            return ContextState.OVERLOADED
        
        # Analyze segment quality and coherence
        if len(segments) < 3:
            return ContextState.BUILDING
        
        recent_segments = sorted(segments, key=lambda s: s.timestamp)[-10:]
        
        # Check for fragmentation
        content_types = set(s.content_type for s in recent_segments)
        if len(content_types) > 4:  # Too many different types
            return ContextState.FRAGMENTED
        
        # Check for degradation (low importance segments accumulating)
        low_importance_count = sum(
            1 for s in recent_segments 
            if s.importance.value <= ContextImportance.LOW.value
        )
        
        if low_importance_count > len(recent_segments) * 0.6:
            return ContextState.DEGRADING
        
        # Check if recently optimized
        if hasattr(self.context_compact.stats, 'last_compaction'):
            last_compaction = self.context_compact.stats.last_compaction
            if last_compaction and (datetime.now() - last_compaction).total_seconds() < 3600:
                return ContextState.OPTIMIZED
        
        return ContextState.STABLE
    
    async def _calculate_quality_scores(self, segments: List[ContextSegment], 
                                       pattern: ContextPattern) -> Dict[str, float]:
        """Calculate various quality scores for context"""
        if not segments:
            return {'overall': 0.0, 'coherence': 0.0, 'relevance': 0.0, 'efficiency': 0.0}
        
        # Coherence score: how well segments relate to each other
        coherence_score = await self._calculate_coherence_score(segments)
        
        # Relevance score: how relevant segments are to current pattern/intent
        relevance_score = await self._calculate_relevance_score(segments, pattern)
        
        # Efficiency score: token/information density
        efficiency_score = await self._calculate_efficiency_score(segments)
        
        # Overall quality score (weighted average)
        overall_score = (
            coherence_score * 0.3 +
            relevance_score * 0.4 +
            efficiency_score * 0.3
        )
        
        return {
            'overall': overall_score,
            'coherence': coherence_score,
            'relevance': relevance_score,
            'efficiency': efficiency_score
        }
    
    async def _calculate_coherence_score(self, segments: List[ContextSegment]) -> float:
        """Calculate coherence score based on segment relationships"""
        if len(segments) < 2:
            return 1.0
        
        # Count segments with relationships
        connected_segments = sum(1 for s in segments if s.relationships)
        connection_ratio = connected_segments / len(segments) if segments else 0
        
        # Analyze content type consistency
        content_types = [s.content_type for s in segments]
        type_diversity = len(set(content_types)) / len(segments) if segments else 0
        
        # Lower diversity in recent segments suggests better coherence
        recent_segments = sorted(segments, key=lambda s: s.timestamp)[-5:]
        recent_types = set(s.content_type for s in recent_segments)
        recent_coherence = 1.0 - (len(recent_types) / len(recent_segments)) if recent_segments else 1.0
        
        # Weighted coherence score
        coherence = (
            connection_ratio * 0.4 +
            (1.0 - type_diversity) * 0.3 +
            recent_coherence * 0.3
        )
        
        return min(max(coherence, 0.0), 1.0)
    
    async def _calculate_relevance_score(self, segments: List[ContextSegment], 
                                        pattern: ContextPattern) -> float:
        """Calculate relevance score based on pattern alignment"""
        if not segments:
            return 0.0
        
        # Pattern-specific relevance calculation
        pattern_keywords = []
        if pattern in self.pattern_detector.pattern_signatures:
            signatures = self.pattern_detector.pattern_signatures[pattern]
            pattern_keywords = (
                signatures.get('keywords', []) +
                signatures.get('structure_indicators', [])
            )
        
        # Count segments that align with pattern
        aligned_segments = 0
        for segment in segments:
            content = str(segment.content).lower()
            if any(keyword in content for keyword in pattern_keywords):
                aligned_segments += 1
        
        pattern_alignment = aligned_segments / len(segments) if segments else 0
        
        # Importance-based relevance
        high_importance_ratio = sum(
            1 for s in segments if s.importance.value >= ContextImportance.MEDIUM.value
        ) / len(segments) if segments else 0
        
        # Recent access relevance
        recent_access_ratio = sum(
            1 for s in segments if s.access_count > 0
        ) / len(segments) if segments else 0
        
        relevance = (
            pattern_alignment * 0.4 +
            high_importance_ratio * 0.3 +
            recent_access_ratio * 0.3
        )
        
        return min(max(relevance, 0.0), 1.0)
    
    async def _calculate_efficiency_score(self, segments: List[ContextSegment]) -> float:
        """Calculate efficiency score based on information density"""
        if not segments:
            return 0.0
        
        total_tokens = sum(s.tokens_estimate for s in segments)
        if total_tokens == 0:
            return 0.0
        
        # Calculate information density metrics
        unique_content_types = len(set(s.content_type for s in segments))
        high_importance_tokens = sum(
            s.tokens_estimate for s in segments 
            if s.importance.value >= ContextImportance.HIGH.value
        )
        
        # Efficiency factors
        type_diversity = min(unique_content_types / 5.0, 1.0)  # Cap at 5 types
        importance_density = high_importance_tokens / total_tokens if total_tokens > 0 else 0
        
        # Compression potential (inverse of efficiency)
        avg_tokens_per_segment = total_tokens / len(segments) if segments else 0
        size_efficiency = min(avg_tokens_per_segment / 500.0, 1.0)  # Cap at 500 tokens
        
        efficiency = (
            type_diversity * 0.3 +
            importance_density * 0.4 +
            size_efficiency * 0.3
        )
        
        return min(max(efficiency, 0.0), 1.0)
    
    async def _generate_recommendations(self, segments: List[ContextSegment],
                                       pattern: ContextPattern,
                                       intent: IntentType,
                                       state: ContextState,
                                       quality_scores: Dict[str, float]) -> List[str]:
        """Generate intelligent recommendations for context optimization"""
        recommendations = []
        
        # State-based recommendations
        if state == ContextState.CRITICAL:
            recommendations.append("URGENT: Immediate context compaction required")
            recommendations.append("Consider emergency compaction strategy")
            
        elif state == ContextState.OVERLOADED:
            recommendations.append("Context approaching limits - schedule compaction")
            recommendations.append("Review and archive low-priority segments")
            
        elif state == ContextState.FRAGMENTED:
            recommendations.append("Context fragmentation detected - consider reorganization")
            recommendations.append("Group related segments for better coherence")
            
        elif state == ContextState.DEGRADING:
            recommendations.append("Context quality degrading - clean up low-value segments")
            recommendations.append("Focus on high-importance content preservation")
        
        # Quality-based recommendations
        if quality_scores['coherence'] < 0.5:
            recommendations.append("Low coherence - improve segment relationships")
            recommendations.append("Consider semantic compression to maintain context flow")
        
        if quality_scores['relevance'] < 0.5:
            recommendations.append("Low relevance - remove off-topic segments")
            recommendations.append("Focus context on current task/pattern")
        
        if quality_scores['efficiency'] < 0.5:
            recommendations.append("Low efficiency - compress verbose segments")
            recommendations.append("Optimize token usage with hierarchical compression")
        
        # Pattern-based recommendations
        if pattern == ContextPattern.PROBLEM_SOLVING:
            recommendations.append("Problem-solving pattern: preserve error context and solutions")
            recommendations.append("Consider creating solution summary for future reference")
            
        elif pattern == ContextPattern.CREATIVE:
            recommendations.append("Creative pattern: preserve inspiration and iteration history")
            recommendations.append("Use conservative compression to maintain creative flow")
            
        elif pattern == ContextPattern.EXPLORATORY:
            recommendations.append("Exploratory pattern: compress completed investigations")
            recommendations.append("Maintain breadcrumbs for exploration paths")
        
        # Intent-based recommendations
        if intent == IntentType.CODE_DEVELOPMENT:
            recommendations.append("Code development: preserve implementation decisions")
            recommendations.append("Maintain architecture context and design rationale")
            
        elif intent == IntentType.DEBUGGING:
            recommendations.append("Debugging: keep error traces and resolution steps")
            recommendations.append("Archive resolved issues with solution summaries")
        
        # Remove duplicates and limit
        unique_recommendations = list(dict.fromkeys(recommendations))
        return unique_recommendations[:8]  # Limit to 8 recommendations
    
    async def _identify_optimization_opportunities(self, segments: List[ContextSegment],
                                                  pattern: ContextPattern,
                                                  intent: IntentType,
                                                  quality_scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """Identify specific optimization opportunities"""
        opportunities = []
        
        # Large segment consolidation
        large_segments = [s for s in segments if s.tokens_estimate > 2000]
        if large_segments:
            opportunities.append({
                'type': 'consolidation',
                'description': f'Consolidate {len(large_segments)} large segments',
                'impact': 'high',
                'effort': 'medium',
                'tokens_affected': sum(s.tokens_estimate for s in large_segments),
                'strategy': 'hierarchical_compression'
            })
        
        # Old content archival
        old_threshold = datetime.now() - timedelta(days=1)
        old_segments = [s for s in segments if s.last_accessed < old_threshold]
        if old_segments:
            opportunities.append({
                'type': 'archival',
                'description': f'Archive {len(old_segments)} old segments',
                'impact': 'medium',
                'effort': 'low',
                'tokens_affected': sum(s.tokens_estimate for s in old_segments),
                'strategy': 'temporal_compression'
            })
        
        # Low-importance cleanup
        low_importance = [s for s in segments if s.importance.value <= ContextImportance.LOW.value]
        if len(low_importance) > 5:
            opportunities.append({
                'type': 'cleanup',
                'description': f'Clean up {len(low_importance)} low-importance segments',
                'impact': 'medium',
                'effort': 'low',
                'tokens_affected': sum(s.tokens_estimate for s in low_importance),
                'strategy': 'aggressive_compression'
            })
        
        # Duplicate content removal
        content_hashes = defaultdict(list)
        for segment in segments:
            content_hashes[segment.semantic_hash].append(segment)
        
        duplicates = [segments for segments in content_hashes.values() if len(segments) > 1]
        if duplicates:
            duplicate_count = sum(len(dups) - 1 for dups in duplicates)  # Keep one of each
            opportunities.append({
                'type': 'deduplication',
                'description': f'Remove {duplicate_count} duplicate segments',
                'impact': 'high',
                'effort': 'low',
                'tokens_affected': sum(
                    sum(s.tokens_estimate for s in dups[1:]) for dups in duplicates
                ),
                'strategy': 'deduplication'
            })
        
        # Relationship-based grouping
        isolated_segments = [s for s in segments if not s.relationships]
        if len(isolated_segments) > 10:
            opportunities.append({
                'type': 'grouping',
                'description': f'Group {len(isolated_segments)} isolated segments',
                'impact': 'medium',
                'effort': 'high',
                'tokens_affected': sum(s.tokens_estimate for s in isolated_segments),
                'strategy': 'semantic_compression'
            })
        
        return opportunities
    
    async def _predict_growth(self, segments: List[ContextSegment], 
                             pattern: ContextPattern) -> Dict[str, float]:
        """Predict context growth patterns"""
        if len(segments) < 2:
            return {
                'hourly_token_growth': 100.0,
                'hourly_segment_growth': 2.0,
                'compaction_probability_1h': 0.1,
                'compaction_probability_6h': 0.3
            }
        
        # Analyze recent growth
        recent_segments = sorted(segments, key=lambda s: s.timestamp)[-10:]
        
        if len(recent_segments) >= 2:
            time_span = (recent_segments[-1].timestamp - recent_segments[0].timestamp).total_seconds()
            if time_span > 0:
                token_growth_rate = sum(s.tokens_estimate for s in recent_segments) / (time_span / 3600)
                segment_growth_rate = len(recent_segments) / (time_span / 3600)
            else:
                token_growth_rate = 100.0
                segment_growth_rate = 2.0
        else:
            token_growth_rate = 100.0
            segment_growth_rate = 2.0
        
        # Pattern-specific adjustments
        if pattern == ContextPattern.PROBLEM_SOLVING:
            token_growth_rate *= 1.5  # Problem-solving can be intensive
            segment_growth_rate *= 1.3
        elif pattern == ContextPattern.EXPLORATORY:
            token_growth_rate *= 2.0  # Exploration generates lots of content
            segment_growth_rate *= 1.8
        elif pattern == ContextPattern.CREATIVE:
            token_growth_rate *= 1.2  # Creative work steady growth
            segment_growth_rate *= 1.1
        
        # Calculate compaction probabilities
        current_tokens = sum(s.tokens_estimate for s in segments)
        
        # 1-hour prediction
        predicted_tokens_1h = current_tokens + token_growth_rate
        compaction_prob_1h = min(max((predicted_tokens_1h - 90000) / 30000, 0.0), 1.0)
        
        # 6-hour prediction
        predicted_tokens_6h = current_tokens + (token_growth_rate * 6)
        compaction_prob_6h = min(max((predicted_tokens_6h - 80000) / 40000, 0.0), 1.0)
        
        return {
            'hourly_token_growth': token_growth_rate,
            'hourly_segment_growth': segment_growth_rate,
            'compaction_probability_1h': compaction_prob_1h,
            'compaction_probability_6h': compaction_prob_6h
        }
    
    def _create_empty_analysis(self, context_id: str) -> ContextAnalysis:
        """Create empty analysis for contexts with no segments"""
        return ContextAnalysis(
            context_id=context_id,
            timestamp=datetime.now(),
            total_segments=0,
            total_tokens=0,
            pattern=ContextPattern.SEQUENTIAL,
            state=ContextState.BUILDING,
            intent=IntentType.UNKNOWN,
            quality_score=0.0,
            coherence_score=0.0,
            relevance_score=0.0,
            efficiency_score=0.0,
            recommendations=["Context is empty - add content to begin analysis"],
            optimization_opportunities=[],
            predicted_growth={'hourly_token_growth': 0.0, 'hourly_segment_growth': 0.0}
        )
    
    def _create_error_analysis(self, context_id: str, error: str) -> ContextAnalysis:
        """Create error analysis when analysis fails"""
        return ContextAnalysis(
            context_id=context_id,
            timestamp=datetime.now(),
            total_segments=0,
            total_tokens=0,
            pattern=ContextPattern.SEQUENTIAL,
            state=ContextState.BUILDING,
            intent=IntentType.UNKNOWN,
            quality_score=0.0,
            coherence_score=0.0,
            relevance_score=0.0,
            efficiency_score=0.0,
            recommendations=[f"Analysis failed: {error}"],
            optimization_opportunities=[],
            predicted_growth={'hourly_token_growth': 0.0, 'hourly_segment_growth': 0.0},
            metadata={'error': error}
        )
    
    async def _save_analysis(self, analysis: ContextAnalysis):
        """Save analysis to database"""
        try:
            analysis_id = f"{analysis.context_id}_{int(analysis.timestamp.timestamp())}"
            
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO context_analyses
                    (analysis_id, context_id, timestamp, total_segments, total_tokens,
                     pattern, state, intent, quality_score, coherence_score, 
                     relevance_score, efficiency_score, recommendations,
                     optimization_opportunities, predicted_growth, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    analysis_id,
                    analysis.context_id,
                    analysis.timestamp.timestamp(),
                    analysis.total_segments,
                    analysis.total_tokens,
                    analysis.pattern.value,
                    analysis.state.value,
                    analysis.intent.value,
                    analysis.quality_score,
                    analysis.coherence_score,
                    analysis.relevance_score,
                    analysis.efficiency_score,
                    json.dumps(analysis.recommendations),
                    json.dumps(analysis.optimization_opportunities),
                    json.dumps(analysis.predicted_growth),
                    json.dumps(analysis.metadata)
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to save analysis: {e}")
    
    async def predict_context_evolution(self, context_id: str = 'main',
                                       horizon_hours: int = 2) -> ContextPrediction:
        """Predict how context will evolve"""
        try:
            # Get current analysis
            analysis = await self.analyze_context(context_id)
            
            # Get segments for prediction
            if context_id not in self.context_compact.active_contexts:
                return self.pattern_detector._create_default_prediction()
            
            segment_ids = self.context_compact.active_contexts[context_id]
            segments = [
                self.context_compact.context_segments[sid] 
                for sid in segment_ids 
                if sid in self.context_compact.context_segments
            ]
            
            # Use pattern detector for detailed prediction
            prediction = await self.pattern_detector.predict_evolution(segments, analysis.pattern)
            
            # Adjust prediction based on analysis insights
            prediction.prediction_horizon = timedelta(hours=horizon_hours)
            
            # Cache prediction
            self.prediction_cache[context_id] = prediction
            
            # Save to database
            await self._save_prediction(prediction)
            
            self.logger.info(f"Context prediction generated for {context_id}: "
                           f"{prediction.predicted_tokens} tokens predicted")
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Context prediction failed: {e}")
            return self.pattern_detector._create_default_prediction()
    
    async def _save_prediction(self, prediction: ContextPrediction):
        """Save prediction to database"""
        try:
            prediction_id = f"{prediction.context_id}_{int(time.time())}"
            
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    INSERT INTO context_predictions
                    (prediction_id, context_id, timestamp, prediction_horizon_seconds,
                     predicted_tokens, predicted_segments, compaction_probability,
                     optimal_strategy, confidence, triggers, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    prediction_id,
                    prediction.context_id,
                    time.time(),
                    prediction.prediction_horizon.total_seconds(),
                    prediction.predicted_tokens,
                    prediction.predicted_segments,
                    prediction.compaction_probability,
                    prediction.optimal_strategy.value,
                    prediction.confidence,
                    json.dumps(prediction.triggers),
                    json.dumps(prediction.metadata)
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to save prediction: {e}")
    
    async def optimize_context(self, context_id: str = 'main',
                              apply_recommendations: bool = False) -> Dict[str, Any]:
        """Optimize context based on intelligence analysis"""
        try:
            analysis = await self.analyze_context(context_id)
            
            optimization_result = {
                'context_id': context_id,
                'analysis_timestamp': analysis.timestamp,
                'current_quality': analysis.quality_score,
                'recommendations_provided': len(analysis.recommendations),
                'opportunities_identified': len(analysis.optimization_opportunities),
                'optimizations_applied': [],
                'quality_improvement': 0.0,
                'success': True
            }
            
            if apply_recommendations:
                # Apply automatic optimizations based on opportunities
                for opportunity in analysis.optimization_opportunities:
                    if opportunity.get('effort') == 'low' and opportunity.get('impact') in ['medium', 'high']:
                        result = await self._apply_optimization(context_id, opportunity)
                        if result['success']:
                            optimization_result['optimizations_applied'].append({
                                'type': opportunity['type'],
                                'description': opportunity['description'],
                                'tokens_saved': result.get('tokens_saved', 0)
                            })
                
                # Re-analyze after optimization
                if optimization_result['optimizations_applied']:
                    new_analysis = await self.analyze_context(context_id, force_refresh=True)
                    optimization_result['quality_improvement'] = (
                        new_analysis.quality_score - analysis.quality_score
                    )
                    self.metrics.optimization_successes += 1
                    
                    # Update average quality improvement
                    self.metrics.average_quality_improvement = (
                        (self.metrics.average_quality_improvement * (self.metrics.optimization_successes - 1) +
                         optimization_result['quality_improvement']) / self.metrics.optimization_successes
                    )
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"Context optimization failed: {e}")
            return {
                'context_id': context_id,
                'success': False,
                'error': str(e)
            }
    
    async def _apply_optimization(self, context_id: str, 
                                 opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a specific optimization opportunity"""
        try:
            optimization_type = opportunity['type']
            strategy = opportunity.get('strategy', 'balanced')
            
            if optimization_type == 'archival':
                # Trigger temporal compression
                result = await self.context_compact.compact_context(
                    context_id, 
                    CompactionStrategy.BALANCED
                )
                
            elif optimization_type == 'cleanup':
                # Trigger aggressive compression for low-importance content
                result = await self.context_compact.compact_context(
                    context_id,
                    CompactionStrategy.AGGRESSIVE
                )
                
            elif optimization_type == 'deduplication':
                # This would require custom deduplication logic
                # For now, use semantic compression
                result = await self.context_compact.compact_context(
                    context_id,
                    CompactionStrategy.ADAPTIVE
                )
                
            elif optimization_type == 'consolidation':
                # Use hierarchical compression
                result = await self.context_compact.compact_context(
                    context_id,
                    CompactionStrategy.CONSERVATIVE
                )
                
            else:
                result = {'success': False, 'error': f'Unknown optimization type: {optimization_type}'}
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def start_monitoring(self):
        """Start background monitoring for intelligent context management"""
        if not self.monitoring_enabled:
            self.monitoring_enabled = True
            self._stop_monitoring.clear()
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.info("Context intelligence monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        if self.monitoring_enabled:
            self.monitoring_enabled = False
            self._stop_monitoring.set()
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)
            self.logger.info("Context intelligence monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop for intelligent context management"""
        while not self._stop_monitoring.is_set():
            try:
                # Analyze all active contexts
                for context_id in list(self.context_compact.active_contexts.keys()):
                    asyncio.run(self._monitor_context(context_id))
                
                # Sleep for configured interval
                self._stop_monitoring.wait(self.config['analysis_interval'])
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                self._stop_monitoring.wait(60)  # Wait 1 minute on error
    
    async def _monitor_context(self, context_id: str):
        """Monitor a specific context for optimization opportunities"""
        try:
            # Perform analysis
            analysis = await self.analyze_context(context_id)
            
            # Check for automatic optimization triggers
            if self.config['enable_predictive_compaction']:
                # Check compaction probability
                growth = analysis.predicted_growth
                if growth.get('compaction_probability_1h', 0) > 0.7:
                    await self.context_compact.compact_context(
                        context_id,
                        CompactionStrategy.ADAPTIVE
                    )
                    self.logger.info(f"Predictive compaction triggered for {context_id}")
            
            # Check for quality degradation
            if analysis.quality_score < self.config['quality_threshold']:
                if analysis.state in [ContextState.DEGRADING, ContextState.FRAGMENTED]:
                    # Apply automatic optimization
                    await self.optimize_context(context_id, apply_recommendations=True)
                    self.logger.info(f"Automatic optimization applied to {context_id}")
            
            # Check for critical state
            if analysis.state == ContextState.CRITICAL:
                await self.context_compact.compact_context(
                    context_id,
                    CompactionStrategy.EMERGENCY
                )
                self.logger.warning(f"Emergency compaction applied to {context_id}")
            
        except Exception as e:
            self.logger.error(f"Context monitoring error for {context_id}: {e}")
    
    def get_intelligence_report(self) -> Dict[str, Any]:
        """Generate comprehensive intelligence report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'metrics': asdict(self.metrics),
            'active_contexts': len(self.context_compact.active_contexts),
            'cached_analyses': len(self.analysis_cache),
            'cached_predictions': len(self.prediction_cache),
            'configuration': self.config,
            'monitoring_enabled': self.monitoring_enabled,
            'recent_analyses': [
                {
                    'context_id': analysis.context_id,
                    'timestamp': analysis.timestamp.isoformat(),
                    'pattern': analysis.pattern.value,
                    'intent': analysis.intent.value,
                    'quality_score': analysis.quality_score,
                    'state': analysis.state.value
                }
                for analysis in list(self.analysis_cache.values())[-5:]
            ]
        }
    
    async def shutdown(self):
        """Gracefully shutdown the intelligence system"""
        self.logger.info("Shutting down Context Intelligence System")
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Save metrics to database
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                metric_id = f"shutdown_{int(time.time())}"
                conn.execute('''
                    INSERT INTO intelligence_metrics
                    (metric_id, timestamp, total_analyses, correct_predictions,
                     optimization_successes, pattern_detections, intent_recognitions,
                     average_quality_improvement, average_analysis_time,
                     prediction_accuracy, user_satisfaction_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metric_id,
                    time.time(),
                    self.metrics.total_analyses,
                    self.metrics.correct_predictions,
                    self.metrics.optimization_successes,
                    self.metrics.pattern_detections,
                    self.metrics.intent_recognitions,
                    self.metrics.average_quality_improvement,
                    self.metrics.average_analysis_time,
                    self.metrics.prediction_accuracy,
                    self.metrics.user_satisfaction_score
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to save final metrics: {e}")
        
        # Clear caches
        self.analysis_cache.clear()
        self.prediction_cache.clear()
        
        self.logger.info("Context Intelligence System shutdown complete")


# Global instance
_global_intelligence: Optional[ContextIntelligence] = None

def get_context_intelligence(project_path: Optional[Path] = None) -> ContextIntelligence:
    """Get global context intelligence instance"""
    global _global_intelligence
    if _global_intelligence is None:
        _global_intelligence = ContextIntelligence(project_path)
    return _global_intelligence

# Convenience functions
async def analyze_context(context_id: str = 'main') -> ContextAnalysis:
    """Analyze context with intelligence system"""
    intelligence = get_context_intelligence()
    return await intelligence.analyze_context(context_id)

async def predict_context_evolution(context_id: str = 'main', 
                                   horizon_hours: int = 2) -> ContextPrediction:
    """Predict context evolution"""
    intelligence = get_context_intelligence()
    return await intelligence.predict_context_evolution(context_id, horizon_hours)

async def optimize_context(context_id: str = 'main', 
                          apply_recommendations: bool = False) -> Dict[str, Any]:
    """Optimize context intelligently"""
    intelligence = get_context_intelligence()
    return await intelligence.optimize_context(context_id, apply_recommendations)

def get_intelligence_report() -> Dict[str, Any]:
    """Get intelligence system report"""
    intelligence = get_context_intelligence()
    return intelligence.get_intelligence_report()