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

logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """Types of user feedback"""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"  # 1-5 scale
    CORRECTION = "correction"  # User provides corrected version
    COMMENT = "comment"  # Textual feedback
    USAGE = "usage"  # Implicit feedback from usage patterns

class QualityMetric(Enum):
    """Code quality metrics"""
    CORRECTNESS = "correctness"
    COMPLETENESS = "completeness"
    READABILITY = "readability"
    PERFORMANCE = "performance"
    SECURITY = "security"
    BEST_PRACTICES = "best_practices"
    MAINTAINABILITY = "maintainability"

@dataclass
class FeedbackEntry:
    """Individual feedback entry"""
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

class PatternLearner:
    """Learns patterns from successful and unsuccessful generations"""
    
    def __init__(self):
        self.successful_patterns = defaultdict(list)
        self.failed_patterns = defaultdict(list)
        self.pattern_scores = defaultdict(float)
    
    def learn_from_feedback(self, feedback: FeedbackEntry):
        """Learn patterns from feedback"""
        if not feedback.ai_response:
            return
        
        # Extract patterns from the response
        patterns = self._extract_patterns(feedback.ai_response, feedback.task_type)
        
        is_positive = self._is_positive_feedback(feedback)
        
        for pattern in patterns:
            pattern_key = f"{feedback.task_type}:{pattern}"
            
            if is_positive:
                self.successful_patterns[pattern_key].append({
                    'context': feedback.user_request[:200],
                    'timestamp': feedback.timestamp,
                    'quality': self._get_overall_quality(feedback)
                })
                # Increase pattern score
                self.pattern_scores[pattern_key] = min(1.0, self.pattern_scores[pattern_key] + 0.1)
            else:
                self.failed_patterns[pattern_key].append({
                    'context': feedback.user_request[:200],
                    'timestamp': feedback.timestamp,
                    'quality': self._get_overall_quality(feedback)
                })
                # Decrease pattern score
                self.pattern_scores[pattern_key] = max(-1.0, self.pattern_scores[pattern_key] - 0.15)
    
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
    
    def get_pattern_recommendations(self, task_type: str, context: str) -> List[Tuple[str, float]]:
        """Get pattern recommendations for a task"""
        recommendations = []
        
        for pattern_key, score in self.pattern_scores.items():
            if pattern_key.startswith(f"{task_type}:") and score > 0.3:
                pattern = pattern_key.split(':', 1)[1]
                recommendations.append((pattern, score))
        
        # Sort by score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:10]  # Top 10 recommendations

class AdaptiveLearningSystem:
    """Main learning system that coordinates all learning components"""
    
    def __init__(self, project_path: Optional[Path] = None):
        self.project_path = project_path
        
        # Learning components
        self.feedback_history: List[FeedbackEntry] = []
        self.model_performance: Dict[str, ModelPerformance] = {}
        self.pattern_learner = PatternLearner()
        
        # Adaptation parameters
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.7  # Quality threshold for considering adaptation
        
        # Statistics
        self.total_feedback_entries = 0
        self.learning_events = []
        
        # Persistence
        self.data_file = None
        if project_path:
            self.data_file = project_path / '.abov3' / 'learning_data.pkl'
            self.data_file.parent.mkdir(parents=True, exist_ok=True)
            self._load_learning_data()
    
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