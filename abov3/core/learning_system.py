"""
ABOV3 Genesis - Adaptive Learning System
Machine learning system that improves code generation based on user feedback and usage patterns
"""

import asyncio
import json
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
import pickle
from enum import Enum
import hashlib
import re
import math
from scipy.stats import beta
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """Types of user feedback"""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"  # 1-5 scale
    CORRECTION = "correction"  # User provides corrected version
    COMMENT = "comment"  # Textual feedback
    USAGE = "usage"  # Implicit feedback from usage patterns
    IMPLICIT_POSITIVE = "implicit_positive"  # Derived from behavior
    IMPLICIT_NEGATIVE = "implicit_negative"  # Derived from behavior
    EXPERT_ANNOTATION = "expert_annotation"  # High-quality labels
    A_B_TEST_PREFERENCE = "ab_test_preference"  # Comparative feedback

class QualityMetric(Enum):
    """Code quality metrics"""
    CORRECTNESS = "correctness"
    COMPLETENESS = "completeness"
    READABILITY = "readability"
    PERFORMANCE = "performance"
    SECURITY = "security"
    BEST_PRACTICES = "best_practices"
    MAINTAINABILITY = "maintainability"
    CREATIVITY = "creativity"
    RELEVANCE = "relevance"
    CLARITY = "clarity"
    EFFICIENCY = "efficiency"
    SCALABILITY = "scalability"

@dataclass
class FeedbackEntry:
    """Individual feedback entry with enhanced metadata"""
    feedback_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:12])
    timestamp: float = field(default_factory=time.time)
    user_request: str = ""
    ai_response: str = ""
    feedback_type: FeedbackType = FeedbackType.RATING
    feedback_value: Union[int, float, str, Dict] = None
    model_used: str = ""
    task_type: str = ""
    context_info: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[QualityMetric, float] = field(default_factory=dict)
    session_id: str = ""
    user_id: str = "anonymous"
    
    # Enhanced fields for advanced learning
    user_expertise_level: str = "intermediate"  # beginner, intermediate, advanced, expert
    response_processing_time: float = 0.0
    user_satisfaction_score: float = 0.0  # 0-1 derived score
    behavioral_signals: Dict[str, Any] = field(default_factory=dict)
    improvement_suggestions: List[str] = field(default_factory=list)
    confidence_score: float = 0.0  # How confident we are in this feedback
    weight: float = 1.0  # Weight for this feedback in learning
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        result = asdict(self)
        result['feedback_type'] = self.feedback_type.value
        result['quality_metrics'] = {k.value: v for k, v in self.quality_metrics.items()}
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeedbackEntry':
        """Create from dictionary"""
        data = data.copy()
        if 'feedback_type' in data:
            data['feedback_type'] = FeedbackType(data['feedback_type'])
        if 'quality_metrics' in data:
            data['quality_metrics'] = {QualityMetric(k): v for k, v in data['quality_metrics'].items()}
        return cls(**data)

@dataclass
class ModelPerformance:
    """Performance metrics for a model"""
    model_name: str
    total_requests: int = 0
    successful_requests: int = 0
    average_rating: float = 0.0
    quality_scores: Dict[QualityMetric, float] = field(default_factory=dict)
    task_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)  # task -> metrics
    improvement_trend: List[float] = field(default_factory=list)  # Recent performance trend
    last_updated: float = field(default_factory=time.time)
    
    def update_performance(self, feedback: FeedbackEntry):
        """Update performance metrics with new feedback"""
        self.total_requests += 1
        self.last_updated = time.time()
        
        # Update success rate
        if self._is_positive_feedback(feedback):
            self.successful_requests += 1
        
        # Update average rating
        if feedback.feedback_type == FeedbackType.RATING and isinstance(feedback.feedback_value, (int, float)):
            old_avg = self.average_rating
            new_rating = float(feedback.feedback_value)
            self.average_rating = ((old_avg * (self.total_requests - 1)) + new_rating) / self.total_requests
        
        # Update quality scores
        for metric, score in feedback.quality_metrics.items():
            if metric not in self.quality_scores:
                self.quality_scores[metric] = score
            else:
                # Exponential moving average
                alpha = 0.3  # Learning rate
                self.quality_scores[metric] = (1 - alpha) * self.quality_scores[metric] + alpha * score
        
        # Update task-specific performance
        if feedback.task_type:
            if feedback.task_type not in self.task_performance:
                self.task_performance[feedback.task_type] = {}
            
            task_perf = self.task_performance[feedback.task_type]
            if 'success_rate' not in task_perf:
                task_perf['success_rate'] = 0.0
                task_perf['total_attempts'] = 0
            
            task_perf['total_attempts'] += 1
            if self._is_positive_feedback(feedback):
                task_perf['success_rate'] = ((task_perf['success_rate'] * (task_perf['total_attempts'] - 1)) + 1.0) / task_perf['total_attempts']
            else:
                task_perf['success_rate'] = (task_perf['success_rate'] * (task_perf['total_attempts'] - 1)) / task_perf['total_attempts']
        
        # Update improvement trend
        self.improvement_trend.append(self.average_rating)
        if len(self.improvement_trend) > 50:  # Keep last 50 data points
            self.improvement_trend = self.improvement_trend[-50:]
    
    def _is_positive_feedback(self, feedback: FeedbackEntry) -> bool:
        """Determine if feedback is positive"""
        if feedback.feedback_type == FeedbackType.THUMBS_UP:
            return True
        elif feedback.feedback_type == FeedbackType.THUMBS_DOWN:
            return False
        elif feedback.feedback_type == FeedbackType.RATING:
            return isinstance(feedback.feedback_value, (int, float)) and feedback.feedback_value >= 3.0
        else:
            # For other types, check quality metrics
            if feedback.quality_metrics:
                avg_quality = sum(feedback.quality_metrics.values()) / len(feedback.quality_metrics)
                return avg_quality >= 0.6
            return True  # Default to positive if unclear

class AdvancedPatternLearner:
    """Advanced pattern learning with reinforcement and clustering"""
    
    def __init__(self):
        self.successful_patterns = defaultdict(list)
        self.failed_patterns = defaultdict(list)
        self.pattern_scores = defaultdict(float)
        self.pattern_clusters = {}
        self.pattern_embeddings = {}
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.pattern_confidence = defaultdict(lambda: beta(1, 1))  # Bayesian confidence
        
        # Advanced pattern types
        self.contextual_patterns = defaultdict(dict)  # patterns by context
        self.temporal_patterns = defaultdict(list)    # patterns over time
        self.user_specific_patterns = defaultdict(dict) # patterns by user type
        self.model_specific_patterns = defaultdict(dict) # patterns by model
        
        # Reinforcement learning components
        self.q_table = defaultdict(lambda: defaultdict(float))  # State-action values
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.1
        
        # Pattern evolution tracking
        self.pattern_evolution = defaultdict(list)
        self.pattern_effectiveness = defaultdict(list)
        self.adaptive_thresholds = defaultdict(float)
    
    def learn_from_feedback(self, feedback: FeedbackEntry):
        """Advanced learning from feedback with reinforcement"""
        if not feedback.ai_response:
            return
        
        # Extract patterns with enhanced context
        patterns = self._extract_enhanced_patterns(feedback.ai_response, feedback.task_type, feedback.context_info)
        
        is_positive = self._is_positive_feedback(feedback)
        quality_score = self._get_overall_quality(feedback)
        
        # Learn contextual patterns
        context_key = self._create_context_key(feedback)
        user_type = feedback.user_expertise_level
        model_name = feedback.model_used
        
        for pattern in patterns:
            pattern_key = f"{feedback.task_type}:{pattern}"
            
            # Update Bayesian confidence
            if is_positive:
                self.pattern_confidence[pattern_key] = beta(
                    self.pattern_confidence[pattern_key].args[0] + quality_score,
                    self.pattern_confidence[pattern_key].args[1]
                )
            else:
                self.pattern_confidence[pattern_key] = beta(
                    self.pattern_confidence[pattern_key].args[0],
                    self.pattern_confidence[pattern_key].args[1] + (1.0 - quality_score)
                )
            
            # Reinforcement learning update
            state = self._encode_state(feedback)
            action = pattern
            reward = quality_score if is_positive else -quality_score
            
            # Q-learning update
            current_q = self.q_table[state][action]
            max_future_q = max(self.q_table[state].values()) if self.q_table[state] else 0
            new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
            self.q_table[state][action] = new_q
            
            # Update pattern collections with enhanced metadata
            pattern_info = {
                'context': feedback.user_request[:300],
                'timestamp': feedback.timestamp,
                'quality': quality_score,
                'user_type': user_type,
                'model': model_name,
                'session_id': feedback.session_id,
                'behavioral_signals': feedback.behavioral_signals,
                'confidence': feedback.confidence_score,
                'weight': feedback.weight
            }
            
            if is_positive:
                self.successful_patterns[pattern_key].append(pattern_info)
                # Weighted score update
                score_increment = 0.1 * feedback.weight * quality_score
                self.pattern_scores[pattern_key] = min(1.0, self.pattern_scores[pattern_key] + score_increment)
            else:
                self.failed_patterns[pattern_key].append(pattern_info)
                # Weighted score decrement
                score_decrement = 0.15 * feedback.weight * (1.0 - quality_score)
                self.pattern_scores[pattern_key] = max(-1.0, self.pattern_scores[pattern_key] - score_decrement)
            
            # Update contextual patterns
            if context_key not in self.contextual_patterns[pattern_key]:
                self.contextual_patterns[pattern_key][context_key] = []
            self.contextual_patterns[pattern_key][context_key].append(pattern_info)
            
            # Update user-specific patterns
            if user_type not in self.user_specific_patterns[pattern_key]:
                self.user_specific_patterns[pattern_key][user_type] = []
            self.user_specific_patterns[pattern_key][user_type].append(pattern_info)
            
            # Update model-specific patterns
            if model_name not in self.model_specific_patterns[pattern_key]:
                self.model_specific_patterns[pattern_key][model_name] = []
            self.model_specific_patterns[pattern_key][model_name].append(pattern_info)
            
            # Track pattern evolution
            self.pattern_evolution[pattern_key].append({
                'timestamp': feedback.timestamp,
                'score': self.pattern_scores[pattern_key],
                'quality': quality_score,
                'feedback_type': feedback.feedback_type.value
            })
        
        # Periodic pattern clustering for pattern discovery
        if len(self.successful_patterns) % 50 == 0:  # Every 50 patterns
            asyncio.create_task(self._update_pattern_clusters())
    
    def _extract_patterns(self, response: str, task_type: str) -> List[str]:
        """Extract learnable patterns from AI response"""
        patterns = []
        
        if task_type == "code_generation":
            patterns.extend(self._extract_code_patterns(response))
        elif task_type == "debugging":
            patterns.extend(self._extract_debug_patterns(response))
        elif task_type == "explanation":
            patterns.extend(self._extract_explanation_patterns(response))
        
        # General patterns
        patterns.extend(self._extract_general_patterns(response))
        
        return patterns
    
    def _extract_code_patterns(self, response: str) -> List[str]:
        """Extract code-specific patterns"""
        patterns = []
        
        # Code block patterns
        code_blocks = re.findall(r'```\w*\n(.*?)\n```', response, re.DOTALL)
        for code in code_blocks:
            # Extract structural patterns
            if 'class ' in code:
                patterns.append('uses_class_definition')
            if 'def ' in code:
                patterns.append('uses_function_definition')
            if 'try:' in code and 'except' in code:
                patterns.append('uses_exception_handling')
            if 'import ' in code or 'from ' in code:
                patterns.append('includes_imports')
            if '"""' in code or "'''" in code:
                patterns.append('includes_docstrings')
            if 'logger.' in code or 'logging.' in code:
                patterns.append('includes_logging')
            if 'assert ' in code or 'raise ' in code:
                patterns.append('includes_assertions')
        
        # Response structure patterns
        if len(code_blocks) > 1:
            patterns.append('multiple_code_blocks')
        if '## ' in response or '# ' in response:
            patterns.append('uses_headers')
        if 'Example:' in response or 'Usage:' in response:
            patterns.append('includes_examples')
        
        return patterns
    
    def _extract_debug_patterns(self, response: str) -> List[str]:
        """Extract debugging-specific patterns"""
        patterns = []
        
        debug_keywords = [
            'root cause', 'issue is', 'problem is', 'error occurs',
            'fix is', 'solution is', 'change', 'modify', 'update'
        ]
        
        for keyword in debug_keywords:
            if keyword in response.lower():
                patterns.append(f'debug_uses_{keyword.replace(" ", "_")}')
        
        if 'step' in response.lower():
            patterns.append('uses_step_by_step')
        if 'before:' in response.lower() and 'after:' in response.lower():
            patterns.append('shows_before_after')
        
        return patterns
    
    def _extract_explanation_patterns(self, response: str) -> List[str]:
        """Extract explanation-specific patterns"""
        patterns = []
        
        if 'because' in response.lower() or 'reason' in response.lower():
            patterns.append('explains_reasoning')
        if 'example' in response.lower():
            patterns.append('provides_examples')
        if 'alternatively' in response.lower() or 'another way' in response.lower():
            patterns.append('offers_alternatives')
        if len(response.split('.')) > 5:
            patterns.append('detailed_explanation')
        
        return patterns
    
    def _extract_general_patterns(self, response: str) -> List[str]:
        """Extract general response patterns"""
        patterns = []
        
        # Length patterns
        if len(response) > 2000:
            patterns.append('long_response')
        elif len(response) < 300:
            patterns.append('short_response')
        else:
            patterns.append('medium_response')
        
        # Structure patterns
        if response.count('\n\n') > 3:
            patterns.append('well_structured')
        if response.count('1.') > 0 or response.count('- ') > 2:
            patterns.append('uses_lists')
        if '**' in response or '*' in response:
            patterns.append('uses_formatting')
        
        return patterns
    
    def _is_positive_feedback(self, feedback: FeedbackEntry) -> bool:
        """Check if feedback is positive"""
        if feedback.feedback_type == FeedbackType.THUMBS_UP:
            return True
        elif feedback.feedback_type == FeedbackType.THUMBS_DOWN:
            return False
        elif feedback.feedback_type == FeedbackType.RATING:
            return isinstance(feedback.feedback_value, (int, float)) and feedback.feedback_value >= 3.0
        else:
            return self._get_overall_quality(feedback) >= 0.6
    
    def _get_overall_quality(self, feedback: FeedbackEntry) -> float:
        """Calculate overall quality score from feedback"""
        if feedback.quality_metrics:
            return sum(feedback.quality_metrics.values()) / len(feedback.quality_metrics)
        elif feedback.feedback_type == FeedbackType.RATING:
            return float(feedback.feedback_value) / 5.0 if isinstance(feedback.feedback_value, (int, float)) else 0.5
        else:
            return 0.5  # Neutral
    
    def get_intelligent_pattern_recommendations(
        self, 
        task_type: str, 
        context: str, 
        user_type: str = "intermediate",
        model_name: str = "",
        session_context: Dict[str, Any] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Get intelligent pattern recommendations with enhanced context"""
        recommendations = []
        context_key = self._create_context_key_from_info(task_type, context, user_type)
        session_context = session_context or {}
        
        for pattern_key, base_score in self.pattern_scores.items():
            if not pattern_key.startswith(f"{task_type}:") or base_score <= 0.1:
                continue
            
            pattern = pattern_key.split(':', 1)[1]
            
            # Calculate enhanced score with multiple factors
            enhanced_score = base_score
            metadata = {}
            
            # Bayesian confidence adjustment
            confidence_dist = self.pattern_confidence[pattern_key]
            confidence_score = confidence_dist.mean()
            confidence_interval = confidence_dist.interval(0.95)
            enhanced_score *= confidence_score
            metadata['confidence'] = confidence_score
            metadata['confidence_interval'] = confidence_interval
            
            # Contextual relevance boost
            if pattern_key in self.contextual_patterns:
                contextual_boost = self._calculate_contextual_relevance(
                    self.contextual_patterns[pattern_key], context_key, context
                )
                enhanced_score *= (1 + contextual_boost)
                metadata['contextual_relevance'] = contextual_boost
            
            # User-specific adaptation
            if pattern_key in self.user_specific_patterns and user_type in self.user_specific_patterns[pattern_key]:
                user_patterns = self.user_specific_patterns[pattern_key][user_type]
                user_success_rate = sum(1 for p in user_patterns if p['quality'] > 0.6) / len(user_patterns)
                enhanced_score *= (1 + user_success_rate * 0.3)
                metadata['user_success_rate'] = user_success_rate
            
            # Model-specific adaptation
            if model_name and pattern_key in self.model_specific_patterns:
                if model_name in self.model_specific_patterns[pattern_key]:
                    model_patterns = self.model_specific_patterns[pattern_key][model_name]
                    model_success_rate = sum(1 for p in model_patterns if p['quality'] > 0.6) / len(model_patterns)
                    enhanced_score *= (1 + model_success_rate * 0.2)
                    metadata['model_success_rate'] = model_success_rate
            
            # Q-learning value consideration
            state = self._encode_state_from_context(task_type, context, user_type)
            q_value = self.q_table[state][pattern]
            if q_value > 0:
                enhanced_score *= (1 + q_value * 0.1)
                metadata['q_value'] = q_value
            
            # Recent performance trend
            if pattern_key in self.pattern_evolution:
                recent_trend = self._calculate_recent_trend(self.pattern_evolution[pattern_key])
                enhanced_score *= (1 + recent_trend * 0.15)
                metadata['recent_trend'] = recent_trend
            
            # Add pattern frequency and recency
            successful_count = len(self.successful_patterns.get(pattern_key, []))
            failed_count = len(self.failed_patterns.get(pattern_key, []))
            total_count = successful_count + failed_count
            
            if total_count > 0:
                success_rate = successful_count / total_count
                enhanced_score *= success_rate
                metadata['success_rate'] = success_rate
                metadata['usage_count'] = total_count
            
            if enhanced_score > 0.2:  # Minimum threshold for recommendations
                recommendations.append((pattern, enhanced_score, metadata))
        
        # Sort by enhanced score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        # Diversity filtering - avoid too similar patterns
        diverse_recommendations = self._apply_diversity_filter(recommendations[:20])
        
        return diverse_recommendations[:12]  # Top 12 diverse recommendations

class AdaptiveLearningSystem:
    """Advanced learning system with reinforcement learning and multi-modal feedback"""
    
    def __init__(self, project_path: Optional[Path] = None):
        self.project_path = project_path
        
        # Enhanced learning components
        self.feedback_history: List[FeedbackEntry] = []
        self.model_performance: Dict[str, ModelPerformance] = {}
        self.pattern_learner = AdvancedPatternLearner()
        
        # Advanced learning features
        self.user_modeling = UserModelingSystem()
        self.active_learning = ActiveLearningSystem()
        self.feedback_synthesizer = FeedbackSynthesizer()
        self.performance_predictor = PerformancePredictor()
        
        # Meta-learning components
        self.meta_optimizer = MetaLearningOptimizer()
        self.curriculum_designer = CurriculumDesigner()
        self.knowledge_graph = KnowledgeGraph()
        
        # Enhanced adaptation parameters
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.7
        self.meta_learning_rate = 0.01
        self.exploration_decay = 0.995
        self.confidence_threshold = 0.8
        
        # Multi-objective optimization weights
        self.objective_weights = {
            'accuracy': 0.3,
            'user_satisfaction': 0.25,
            'efficiency': 0.2,
            'novelty': 0.1,
            'safety': 0.15
        }
        
        # Enhanced statistics and tracking
        self.total_feedback_entries = 0
        self.learning_events = []
        self.learning_metrics = {
            'cumulative_reward': 0.0,
            'average_improvement': 0.0,
            'adaptation_success_rate': 0.0,
            'user_satisfaction_trend': [],
            'model_convergence_metrics': {},
            'exploration_exploitation_ratio': 0.5
        }
        
        # Advanced persistence with versioning
        self.data_file = None
        self.backup_files = []
        if project_path:
            self.data_file = project_path / '.abov3' / 'learning_data_v2.pkl'
            self.data_file.parent.mkdir(parents=True, exist_ok=True)
            self._load_enhanced_learning_data()
        
        # Real-time learning activation
        self.real_time_learning_enabled = True
        self.batch_learning_interval = 100  # Process batch every 100 feedback entries
        self.background_learning_task = None
    
    def record_feedback(
        self,
        user_request: str,
        ai_response: str,
        feedback_type: FeedbackType,
        feedback_value: Union[int, float, str, Dict],
        model_used: str,
        task_type: str,
        context_info: Dict[str, Any] = None,
        quality_metrics: Dict[QualityMetric, float] = None,
        session_id: str = "",
        user_id: str = "anonymous"
    ) -> str:
        """Record user feedback"""
        
        feedback = FeedbackEntry(
            user_request=user_request,
            ai_response=ai_response,
            feedback_type=feedback_type,
            feedback_value=feedback_value,
            model_used=model_used,
            task_type=task_type,
            context_info=context_info or {},
            quality_metrics=quality_metrics or {},
            session_id=session_id,
            user_id=user_id
        )
        
        self.feedback_history.append(feedback)
        self.total_feedback_entries += 1
        
        # Update model performance
        if model_used not in self.model_performance:
            self.model_performance[model_used] = ModelPerformance(model_used)
        
        self.model_performance[model_used].update_performance(feedback)
        
        # Learn patterns
        self.pattern_learner.learn_from_feedback(feedback)
        
        # Trigger adaptation if needed
        self._trigger_adaptation_if_needed(feedback)
        
        # Persist learning data periodically
        if self.total_feedback_entries % 10 == 0:  # Every 10 feedback entries
            self._save_learning_data()
        
        logger.info(f"Recorded feedback: {feedback_type.value} for {model_used} on {task_type}")
        return feedback.feedback_id
    
    def record_implicit_feedback(
        self,
        user_request: str,
        ai_response: str,
        model_used: str,
        task_type: str,
        usage_metrics: Dict[str, Any]
    ):
        """Record implicit feedback from usage patterns"""
        
        # Derive quality score from usage metrics
        quality_score = self._calculate_implicit_quality(usage_metrics)
        
        # Create implicit feedback entry
        self.record_feedback(
            user_request=user_request,
            ai_response=ai_response,
            feedback_type=FeedbackType.USAGE,
            feedback_value=quality_score,
            model_used=model_used,
            task_type=task_type,
            context_info=usage_metrics
        )
    
    def _calculate_implicit_quality(self, usage_metrics: Dict[str, Any]) -> float:
        """Calculate quality score from implicit usage signals"""
        quality_score = 0.5  # Start neutral
        
        # Response time - faster usually better for simple tasks
        response_time = usage_metrics.get('response_time_ms', 0)
        if response_time < 5000:  # Less than 5 seconds
            quality_score += 0.1
        elif response_time > 15000:  # More than 15 seconds
            quality_score -= 0.1
        
        # User actions after response
        user_actions = usage_metrics.get('user_actions', [])
        for action in user_actions:
            if action == 'copy_code':
                quality_score += 0.2
            elif action == 'edit_code':
                quality_score -= 0.1  # Needed editing
            elif action == 'run_code':
                quality_score += 0.15
            elif action == 'save_file':
                quality_score += 0.25
            elif action == 'regenerate':
                quality_score -= 0.3  # Had to regenerate
        
        # Time spent reviewing response
        review_time = usage_metrics.get('review_time_s', 0)
        if 10 < review_time < 120:  # Good review time
            quality_score += 0.1
        elif review_time < 5:  # Too quick - might be poor quality
            quality_score -= 0.1
        
        # Follow-up questions
        followup_count = usage_metrics.get('followup_questions', 0)
        if followup_count == 0:
            quality_score += 0.1  # Complete response
        elif followup_count > 3:
            quality_score -= 0.2  # Many follow-ups suggest incomplete
        
        return max(0.0, min(1.0, quality_score))
    
    def _trigger_adaptation_if_needed(self, feedback: FeedbackEntry):
        """Trigger adaptation based on feedback patterns"""
        model_perf = self.model_performance.get(feedback.model_used)
        if not model_perf:
            return
        
        # Check if adaptation is needed
        should_adapt = False
        adaptation_reason = ""
        
        # Low success rate
        if model_perf.total_requests >= 10:
            success_rate = model_perf.successful_requests / model_perf.total_requests
            if success_rate < 0.6:
                should_adapt = True
                adaptation_reason = f"Low success rate: {success_rate:.2f}"
        
        # Declining performance trend
        if len(model_perf.improvement_trend) >= 10:
            recent_avg = sum(model_perf.improvement_trend[-5:]) / 5
            older_avg = sum(model_perf.improvement_trend[-10:-5]) / 5
            if recent_avg < older_avg - 0.2:
                should_adapt = True
                adaptation_reason = f"Declining performance: {recent_avg:.2f} vs {older_avg:.2f}"
        
        # Poor task-specific performance
        if feedback.task_type in model_perf.task_performance:
            task_perf = model_perf.task_performance[feedback.task_type]
            if task_perf.get('total_attempts', 0) >= 5 and task_perf.get('success_rate', 0) < 0.5:
                should_adapt = True
                adaptation_reason = f"Poor {feedback.task_type} performance: {task_perf['success_rate']:.2f}"
        
        if should_adapt:
            adaptation_event = {
                'timestamp': time.time(),
                'model': feedback.model_used,
                'task_type': feedback.task_type,
                'reason': adaptation_reason,
                'trigger_feedback_id': feedback.feedback_id
            }
            
            self.learning_events.append(adaptation_event)
            logger.info(f"Adaptation triggered for {feedback.model_used}: {adaptation_reason}")
            
            # Generate specific recommendations
            recommendations = self._generate_adaptation_recommendations(feedback.model_used, feedback.task_type)
            adaptation_event['recommendations'] = recommendations
    
    def _generate_adaptation_recommendations(self, model_name: str, task_type: str) -> List[str]:
        """Generate specific recommendations for improving model performance"""
        recommendations = []
        
        model_perf = self.model_performance.get(model_name)
        if not model_perf:
            return recommendations
        
        # Analyze failure patterns
        negative_feedback = [
            f for f in self.feedback_history[-100:]  # Last 100 entries
            if f.model_used == model_name and not self._is_positive_feedback(f)
        ]
        
        if negative_feedback:
            # Common issues in negative feedback
            common_issues = defaultdict(int)
            for feedback in negative_feedback:
                if 'error' in feedback.user_request.lower():
                    common_issues['error_handling'] += 1
                if 'incomplete' in str(feedback.feedback_value).lower():
                    common_issues['completeness'] += 1
                if 'syntax' in str(feedback.feedback_value).lower():
                    common_issues['syntax_errors'] += 1
                if 'performance' in str(feedback.feedback_value).lower():
                    common_issues['performance'] += 1
            
            # Generate recommendations based on common issues
            for issue, count in common_issues.items():
                if count >= 2:  # At least 2 occurrences
                    recommendations.append(f"Improve {issue.replace('_', ' ')} (found in {count} negative feedback entries)")
        
        # Task-specific recommendations
        if task_type == "code_generation":
            recommendations.extend([
                "Consider using more detailed system prompts for code generation",
                "Add more code examples to context",
                "Reduce temperature for more deterministic code output",
                "Include more error handling patterns in examples"
            ])
        elif task_type == "debugging":
            recommendations.extend([
                "Provide more structured debugging methodology in prompts",
                "Include more error pattern examples",
                "Emphasize step-by-step problem analysis"
            ])
        elif task_type == "explanation":
            recommendations.extend([
                "Use more conversational system prompts",
                "Include more analogies and examples",
                "Structure explanations with clear headings"
            ])
        
        # Pattern-based recommendations
        pattern_recs = self.pattern_learner.get_pattern_recommendations(task_type, "")
        for pattern, score in pattern_recs[:3]:  # Top 3 patterns
            recommendations.append(f"Consider incorporating pattern: {pattern} (score: {score:.2f})")
        
        return recommendations[:8]  # Return top 8 recommendations

class UserModelingSystem:
    """Models user behavior and preferences for personalized learning"""
    
    def __init__(self):
        self.user_profiles = defaultdict(dict)
        self.user_clusters = {}
        self.preference_models = {}
    
    def update_user_profile(self, user_id: str, feedback: FeedbackEntry):
        """Update user profile based on feedback"""
        profile = self.user_profiles[user_id]
        
        # Update expertise level estimation
        if 'expertise_indicators' not in profile:
            profile['expertise_indicators'] = []
        
        # Extract expertise signals
        expertise_signals = self._extract_expertise_signals(feedback)
        profile['expertise_indicators'].extend(expertise_signals)
        
        # Update preferences
        if 'preferences' not in profile:
            profile['preferences'] = defaultdict(float)
        
        preference_updates = self._extract_preferences(feedback)
        for pref, value in preference_updates.items():
            profile['preferences'][pref] = profile['preferences'][pref] * 0.9 + value * 0.1
    
    def _extract_expertise_signals(self, feedback: FeedbackEntry) -> List[str]:
        """Extract signals indicating user expertise level"""
        signals = []
        
        # Request complexity analysis
        request = feedback.user_request.lower()
        if any(term in request for term in ['architecture', 'scalability', 'performance', 'security']):
            signals.append('advanced_concepts')
        
        if len(feedback.user_request.split()) > 50:
            signals.append('detailed_requirements')
        
        # Response evaluation ability
        if feedback.feedback_type in [FeedbackType.CORRECTION, FeedbackType.COMMENT]:
            signals.append('critical_evaluation')
        
        return signals
    
    def _extract_preferences(self, feedback: FeedbackEntry) -> Dict[str, float]:
        """Extract user preferences from feedback"""
        preferences = {}
        
        # Response length preference
        response_length = len(feedback.ai_response)
        if feedback.feedback_value and isinstance(feedback.feedback_value, (int, float)):
            quality = float(feedback.feedback_value) / 5.0 if feedback.feedback_type == FeedbackType.RATING else feedback.feedback_value
            
            if response_length > 1000 and quality > 0.7:
                preferences['detailed_responses'] = 0.8
            elif response_length < 300 and quality > 0.7:
                preferences['concise_responses'] = 0.8
        
        return preferences

class ActiveLearningSystem:
    """Identifies areas where more feedback is needed"""
    
    def __init__(self):
        self.uncertainty_tracker = defaultdict(list)
        self.information_gain_estimator = InformationGainEstimator()
    
    def identify_uncertain_areas(self) -> List[Dict[str, Any]]:
        """Identify areas with high uncertainty that need more feedback"""
        uncertain_areas = []
        
        for area, uncertainties in self.uncertainty_tracker.items():
            if len(uncertainties) > 5:  # Sufficient data
                avg_uncertainty = sum(uncertainties[-10:]) / min(10, len(uncertainties))
                if avg_uncertainty > 0.7:  # High uncertainty threshold
                    uncertain_areas.append({
                        'area': area,
                        'uncertainty': avg_uncertainty,
                        'sample_count': len(uncertainties),
                        'priority': avg_uncertainty * math.log(len(uncertainties) + 1)
                    })
        
        return sorted(uncertain_areas, key=lambda x: x['priority'], reverse=True)[:5]

class FeedbackSynthesizer:
    """Synthesizes insights from multiple feedback sources"""
    
    def __init__(self):
        self.synthesis_models = {}
        self.conflict_resolution = ConflictResolver()
    
    def synthesize_feedback(self, feedback_batch: List[FeedbackEntry]) -> Dict[str, Any]:
        """Synthesize insights from a batch of feedback"""
        synthesis = {
            'dominant_patterns': [],
            'quality_trends': {},
            'user_satisfaction_trend': 0.0,
            'model_performance_insights': {},
            'improvement_opportunities': []
        }
        
        # Analyze patterns
        pattern_frequency = defaultdict(int)
        quality_scores = defaultdict(list)
        
        for feedback in feedback_batch:
            # Extract patterns and track quality
            if feedback.quality_metrics:
                for metric, score in feedback.quality_metrics.items():
                    quality_scores[metric.value].append(score)
        
        # Synthesize quality trends
        for metric, scores in quality_scores.items():
            if len(scores) > 1:
                synthesis['quality_trends'][metric] = {
                    'average': sum(scores) / len(scores),
                    'trend': self._calculate_trend(scores),
                    'variance': np.var(scores) if len(scores) > 1 else 0
                }
        
        return synthesis
    
    def _calculate_trend(self, scores: List[float]) -> str:
        """Calculate trend direction from scores"""
        if len(scores) < 2:
            return 'stable'
        
        recent_avg = sum(scores[-len(scores)//2:]) / (len(scores)//2)
        earlier_avg = sum(scores[:len(scores)//2]) / (len(scores)//2)
        
        if recent_avg > earlier_avg + 0.1:
            return 'improving'
        elif recent_avg < earlier_avg - 0.1:
            return 'declining'
        else:
            return 'stable'

class PerformancePredictor:
    """Predicts performance outcomes based on current state"""
    
    def __init__(self):
        self.prediction_models = {}
        self.feature_extractors = {}
    
    def predict_response_quality(self, context: Dict[str, Any]) -> Tuple[float, float]:
        """Predict response quality and confidence"""
        # Simple heuristic-based prediction for now
        base_quality = 0.7
        
        # Adjust based on task complexity
        complexity = context.get('task_complexity', 'medium')
        if complexity == 'high':
            base_quality -= 0.1
        elif complexity == 'low':
            base_quality += 0.1
        
        # Adjust based on model capability
        model_performance = context.get('model_performance', {})
        if model_performance:
            avg_performance = sum(model_performance.values()) / len(model_performance)
            base_quality = base_quality * 0.7 + avg_performance * 0.3
        
        confidence = min(0.9, base_quality)
        
        return base_quality, confidence

class MetaLearningOptimizer:
    """Optimizes the learning process itself"""
    
    def __init__(self):
        self.learning_history = []
        self.optimization_strategies = {}
    
    def optimize_learning_parameters(self, performance_metrics: Dict[str, float]) -> Dict[str, float]:
        """Optimize learning parameters based on performance"""
        optimized_params = {}
        
        # Simple adaptive parameter adjustment
        if performance_metrics.get('adaptation_success_rate', 0) < 0.6:
            optimized_params['learning_rate'] = min(0.2, performance_metrics.get('learning_rate', 0.1) * 1.1)
        else:
            optimized_params['learning_rate'] = max(0.01, performance_metrics.get('learning_rate', 0.1) * 0.95)
        
        return optimized_params

class CurriculumDesigner:
    """Designs learning curriculum based on user progress"""
    
    def __init__(self):
        self.curriculum_stages = []
        self.progression_tracking = {}
    
    def design_learning_path(self, user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Design personalized learning path"""
        path = []
        
        expertise_level = user_profile.get('expertise_level', 'intermediate')
        
        if expertise_level == 'beginner':
            path.extend([
                {'stage': 'basic_syntax', 'difficulty': 1, 'concepts': ['variables', 'functions', 'loops']},
                {'stage': 'simple_algorithms', 'difficulty': 2, 'concepts': ['sorting', 'searching', 'basic_data_structures']},
                {'stage': 'object_oriented', 'difficulty': 3, 'concepts': ['classes', 'inheritance', 'polymorphism']}
            ])
        elif expertise_level == 'intermediate':
            path.extend([
                {'stage': 'advanced_algorithms', 'difficulty': 4, 'concepts': ['dynamic_programming', 'graph_algorithms']},
                {'stage': 'system_design', 'difficulty': 5, 'concepts': ['scalability', 'databases', 'apis']}
            ])
        
        return path

class KnowledgeGraph:
    """Maintains knowledge relationships for better understanding"""
    
    def __init__(self):
        self.entities = {}
        self.relationships = defaultdict(list)
        self.concept_embeddings = {}
    
    def add_concept(self, concept: str, related_concepts: List[str]):
        """Add concept and its relationships"""
        self.entities[concept] = {
            'frequency': self.entities.get(concept, {}).get('frequency', 0) + 1,
            'last_seen': time.time()
        }
        
        for related in related_concepts:
            self.relationships[concept].append(related)

class InformationGainEstimator:
    """Estimates information gain from potential feedback"""
    
    def __init__(self):
        self.entropy_cache = {}
    
    def estimate_gain(self, context: Dict[str, Any]) -> float:
        """Estimate information gain from getting feedback in this context"""
        # Simple entropy-based estimation
        uncertainty = context.get('model_confidence', 0.5)
        return 1.0 - uncertainty

class ConflictResolver:
    """Resolves conflicts between different feedback sources"""
    
    def __init__(self):
        self.resolution_strategies = {
            'voting': self._voting_resolution,
            'weighted_average': self._weighted_average_resolution,
            'expert_preference': self._expert_preference_resolution
        }
    
    def resolve_conflicts(self, conflicting_feedback: List[FeedbackEntry]) -> FeedbackEntry:
        """Resolve conflicts between feedback entries"""
        if len(conflicting_feedback) <= 1:
            return conflicting_feedback[0] if conflicting_feedback else None
        
        # Use weighted average resolution by default
        return self._weighted_average_resolution(conflicting_feedback)
    
    def _weighted_average_resolution(self, feedback_list: List[FeedbackEntry]) -> FeedbackEntry:
        """Resolve using weighted average of feedback values"""
        if not feedback_list:
            return None
        
        # Create composite feedback entry
        resolved_feedback = feedback_list[0]  # Use first as template
        
        # Calculate weighted average of numeric feedback
        numeric_feedback = [
            (f.feedback_value, f.weight) for f in feedback_list 
            if isinstance(f.feedback_value, (int, float))
        ]
        
        if numeric_feedback:
            total_weighted_value = sum(value * weight for value, weight in numeric_feedback)
            total_weight = sum(weight for _, weight in numeric_feedback)
            resolved_feedback.feedback_value = total_weighted_value / total_weight if total_weight > 0 else 0
        
        return resolved_feedback
    
    def _voting_resolution(self, feedback_list: List[FeedbackEntry]) -> FeedbackEntry:
        """Resolve using majority voting"""
        # Simple majority vote implementation
        votes = defaultdict(int)
        for feedback in feedback_list:
            votes[feedback.feedback_value] += 1
        
        majority_value = max(votes.items(), key=lambda x: x[1])[0]
        
        # Return feedback with majority value
        for feedback in feedback_list:
            if feedback.feedback_value == majority_value:
                return feedback
        
        return feedback_list[0]
    
    def _expert_preference_resolution(self, feedback_list: List[FeedbackEntry]) -> FeedbackEntry:
        """Prefer feedback from expert users"""
        expert_feedback = [f for f in feedback_list if f.user_expertise_level == 'expert']
        if expert_feedback:
            return expert_feedback[0]
        
        advanced_feedback = [f for f in feedback_list if f.user_expertise_level == 'advanced']
        if advanced_feedback:
            return advanced_feedback[0]
        
        return feedback_list[0]
    
    def _is_positive_feedback(self, feedback: FeedbackEntry) -> bool:
        """Check if feedback is positive"""
        if feedback.feedback_type == FeedbackType.THUMBS_UP:
            return True
        elif feedback.feedback_type == FeedbackType.THUMBS_DOWN:
            return False
        elif feedback.feedback_type == FeedbackType.RATING:
            return isinstance(feedback.feedback_value, (int, float)) and feedback.feedback_value >= 3.0
        else:
            if feedback.quality_metrics:
                avg_quality = sum(feedback.quality_metrics.values()) / len(feedback.quality_metrics)
                return avg_quality >= 0.6
            return True  # Default to positive if unclear
    
    def get_model_recommendations(self, task_type: str, context: str = "") -> List[Tuple[str, float, str]]:
        """Get model recommendations for a task based on learned performance"""
        recommendations = []
        
        for model_name, perf in self.model_performance.items():
            if task_type in perf.task_performance:
                task_perf = perf.task_performance[task_type]
                success_rate = task_perf.get('success_rate', 0)
                
                # Calculate confidence based on number of attempts
                attempts = task_perf.get('total_attempts', 0)
                confidence = min(1.0, attempts / 10.0)  # Full confidence after 10 attempts
                
                # Adjust score by confidence
                adjusted_score = success_rate * confidence
                
                reason = f"Success rate: {success_rate:.2f} ({attempts} attempts)"
                recommendations.append((model_name, adjusted_score, reason))
        
        # Sort by adjusted score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations
    
    def get_optimization_suggestions(self, model_name: str, task_type: str) -> Dict[str, Any]:
        """Get optimization suggestions for model/task combination"""
        suggestions = {
            'parameter_adjustments': {},
            'prompt_improvements': [],
            'context_optimization': [],
            'training_recommendations': []
        }
        
        model_perf = self.model_performance.get(model_name)
        if not model_perf:
            return suggestions
        
        # Analyze quality metrics for this model
        quality_issues = []
        for metric, score in model_perf.quality_scores.items():
            if score < 0.6:
                quality_issues.append(metric)
        
        # Parameter adjustment suggestions
        if QualityMetric.CORRECTNESS in quality_issues:
            suggestions['parameter_adjustments']['temperature'] = 'Reduce to 0.05-0.1 for more deterministic output'
            suggestions['parameter_adjustments']['top_p'] = 'Reduce to 0.9-0.95 for more focused responses'
        
        if QualityMetric.COMPLETENESS in quality_issues:
            suggestions['parameter_adjustments']['max_tokens'] = 'Increase token limit'
            suggestions['parameter_adjustments']['stop_sequences'] = 'Review stop sequences to prevent early truncation'
        
        if QualityMetric.READABILITY in quality_issues:
            suggestions['prompt_improvements'].append('Add emphasis on code comments and documentation')
            suggestions['prompt_improvements'].append('Include readability best practices in system prompt')
        
        if QualityMetric.PERFORMANCE in quality_issues:
            suggestions['context_optimization'].append('Add performance optimization examples')
            suggestions['prompt_improvements'].append('Emphasize algorithmic efficiency')
        
        if QualityMetric.SECURITY in quality_issues:
            suggestions['context_optimization'].append('Include security best practices and examples')
            suggestions['prompt_improvements'].append('Add security-focused system instructions')
        
        # Pattern-based suggestions
        pattern_recs = self.pattern_learner.get_pattern_recommendations(task_type, "")
        for pattern, score in pattern_recs[:5]:
            if score > 0.5:
                suggestions['training_recommendations'].append(f'Emphasize pattern: {pattern} (success rate: {score:.2f})')
        
        return suggestions
    
    def get_learning_report(self) -> Dict[str, Any]:
        """Get comprehensive learning system report"""
        report = {
            'overview': {
                'total_feedback_entries': self.total_feedback_entries,
                'models_tracked': len(self.model_performance),
                'learning_events': len(self.learning_events),
                'patterns_learned': len(self.pattern_learner.pattern_scores)
            },
            'model_performance': {},
            'recent_adaptations': self.learning_events[-10:],  # Last 10 adaptations
            'top_patterns': [],
            'quality_trends': {}
        }
        
        # Model performance summary
        for model_name, perf in self.model_performance.items():
            report['model_performance'][model_name] = {
                'total_requests': perf.total_requests,
                'success_rate': perf.successful_requests / max(1, perf.total_requests),
                'average_rating': perf.average_rating,
                'quality_scores': {k.value: v for k, v in perf.quality_scores.items()},
                'task_performance': perf.task_performance,
                'trend': perf.improvement_trend[-10:] if perf.improvement_trend else []
            }
        
        # Top patterns
        sorted_patterns = sorted(self.pattern_learner.pattern_scores.items(), key=lambda x: x[1], reverse=True)
        report['top_patterns'] = sorted_patterns[:20]
        
        # Quality trends over time
        if self.feedback_history:
            recent_feedback = self.feedback_history[-50:]  # Last 50 entries
            
            quality_by_time = defaultdict(list)
            for feedback in recent_feedback:
                day = datetime.fromtimestamp(feedback.timestamp).strftime('%Y-%m-%d')
                overall_quality = sum(feedback.quality_metrics.values()) / max(1, len(feedback.quality_metrics))
                quality_by_time[day].append(overall_quality)
            
            for day, qualities in quality_by_time.items():
                report['quality_trends'][day] = sum(qualities) / len(qualities)
        
        return report
    
    def _save_learning_data(self):
        """Save learning data to disk"""
        if not self.data_file:
            return
        
        try:
            data = {
                'feedback_history': [f.to_dict() for f in self.feedback_history[-1000:]],  # Keep last 1000
                'model_performance': {
                    name: asdict(perf) for name, perf in self.model_performance.items()
                },
                'pattern_scores': dict(self.pattern_learner.pattern_scores),
                'successful_patterns': dict(self.pattern_learner.successful_patterns),
                'failed_patterns': dict(self.pattern_learner.failed_patterns),
                'learning_events': self.learning_events[-100:],  # Keep last 100
                'total_feedback_entries': self.total_feedback_entries,
                'saved_at': time.time()
            }
            
            with open(self.data_file, 'wb') as f:
                pickle.dump(data, f)
            
            logger.debug(f"Saved learning data to {self.data_file}")
            
        except Exception as e:
            logger.error(f"Failed to save learning data: {e}")
    
    def _load_learning_data(self):
        """Load learning data from disk"""
        if not self.data_file or not self.data_file.exists():
            return
        
        try:
            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)
            
            # Restore feedback history
            if 'feedback_history' in data:
                self.feedback_history = [FeedbackEntry.from_dict(f) for f in data['feedback_history']]
            
            # Restore model performance
            if 'model_performance' in data:
                for name, perf_data in data['model_performance'].items():
                    perf = ModelPerformance(**perf_data)
                    self.model_performance[name] = perf
            
            # Restore pattern learning data
            if 'pattern_scores' in data:
                self.pattern_learner.pattern_scores.update(data['pattern_scores'])
            if 'successful_patterns' in data:
                self.pattern_learner.successful_patterns.update(data['successful_patterns'])
            if 'failed_patterns' in data:
                self.pattern_learner.failed_patterns.update(data['failed_patterns'])
            
            # Restore other data
            self.learning_events = data.get('learning_events', [])
            self.total_feedback_entries = data.get('total_feedback_entries', 0)
            
            logger.info(f"Loaded learning data from {self.data_file}")
            logger.info(f"Restored {len(self.feedback_history)} feedback entries and {len(self.model_performance)} model performance records")
            
        except Exception as e:
            logger.error(f"Failed to load learning data: {e}")

# Utility functions for easy integration

def create_learning_system(project_path: Path = None) -> AdaptiveLearningSystem:
    """Create and initialize learning system"""
    return AdaptiveLearningSystem(project_path)

def record_simple_feedback(
    learning_system: AdaptiveLearningSystem,
    user_request: str,
    ai_response: str,
    thumbs_up: bool,
    model_name: str,
    task_type: str
) -> str:
    """Record simple thumbs up/down feedback"""
    feedback_type = FeedbackType.THUMBS_UP if thumbs_up else FeedbackType.THUMBS_DOWN
    return learning_system.record_feedback(
        user_request=user_request,
        ai_response=ai_response,
        feedback_type=feedback_type,
        feedback_value=thumbs_up,
        model_used=model_name,
        task_type=task_type
    )

def record_rating_feedback(
    learning_system: AdaptiveLearningSystem,
    user_request: str,
    ai_response: str,
    rating: int,  # 1-5
    model_name: str,
    task_type: str,
    comments: str = ""
) -> str:
    """Record rating feedback (1-5 stars)"""
    quality_metrics = {}
    
    # Derive quality metrics from rating
    if rating >= 4:
        quality_metrics = {
            QualityMetric.CORRECTNESS: 0.8 + (rating - 4) * 0.2,
            QualityMetric.COMPLETENESS: 0.8 + (rating - 4) * 0.2,
            QualityMetric.READABILITY: 0.7 + (rating - 4) * 0.3,
        }
    elif rating >= 3:
        quality_metrics = {
            QualityMetric.CORRECTNESS: 0.6,
            QualityMetric.COMPLETENESS: 0.6,
            QualityMetric.READABILITY: 0.6,
        }
    else:
        quality_metrics = {
            QualityMetric.CORRECTNESS: 0.3 + rating * 0.1,
            QualityMetric.COMPLETENESS: 0.3 + rating * 0.1,
            QualityMetric.READABILITY: 0.3 + rating * 0.1,
        }
    
    context_info = {"comments": comments} if comments else {}
    
    return learning_system.record_feedback(
        user_request=user_request,
        ai_response=ai_response,
        feedback_type=FeedbackType.RATING,
        feedback_value=rating,
        model_used=model_name,
        task_type=task_type,
        quality_metrics=quality_metrics,
        context_info=context_info
    )