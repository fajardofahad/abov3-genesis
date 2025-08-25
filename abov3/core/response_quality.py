"""
ABOV3 Genesis - Advanced Response Quality Scoring and Retry System
Implements Claude-level quality assessment and automatic retries
"""

import asyncio
import json
import re
import time
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import logging
from enum import Enum
from collections import defaultdict
import ast

logger = logging.getLogger(__name__)

class QualityDimension(Enum):
    """Different dimensions of response quality"""
    CORRECTNESS = "correctness"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"
    RELEVANCE = "relevance"
    CODE_QUALITY = "code_quality"
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    BEST_PRACTICES = "best_practices"
    CREATIVITY = "creativity"

class RetryReason(Enum):
    """Reasons for retry"""
    LOW_QUALITY = "low_quality"
    INCOMPLETE_RESPONSE = "incomplete_response"
    SYNTAX_ERROR = "syntax_error"
    SECURITY_ISSUE = "security_issue"
    OFF_TOPIC = "off_topic"
    TIMEOUT = "timeout"
    MODEL_ERROR = "model_error"

@dataclass
class QualityScore:
    """Quality score for a response"""
    overall_score: float = 0.0
    dimension_scores: Dict[QualityDimension, float] = field(default_factory=dict)
    confidence: float = 0.0
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class RetryAttempt:
    """Information about a retry attempt"""
    attempt_number: int
    reason: RetryReason
    previous_score: float
    model_used: str
    timestamp: float = field(default_factory=time.time)
    changes_made: List[str] = field(default_factory=list)

class AdvancedQualityScorer:
    """Advanced quality scoring system with multiple algorithms"""
    
    def __init__(self):
        self.quality_thresholds = {
            QualityDimension.CORRECTNESS: 0.7,
            QualityDimension.COMPLETENESS: 0.6,
            QualityDimension.CLARITY: 0.6,
            QualityDimension.RELEVANCE: 0.7,
            QualityDimension.CODE_QUALITY: 0.7,
            QualityDimension.SECURITY: 0.8,
            QualityDimension.PERFORMANCE: 0.6,
            QualityDimension.MAINTAINABILITY: 0.6,
            QualityDimension.BEST_PRACTICES: 0.7,
            QualityDimension.CREATIVITY: 0.5
        }
        
        # Task-specific quality weights
        self.task_quality_weights = {
            "code_generation": {
                QualityDimension.CORRECTNESS: 0.25,
                QualityDimension.CODE_QUALITY: 0.20,
                QualityDimension.COMPLETENESS: 0.15,
                QualityDimension.SECURITY: 0.15,
                QualityDimension.BEST_PRACTICES: 0.15,
                QualityDimension.CLARITY: 0.10
            },
            "debugging": {
                QualityDimension.CORRECTNESS: 0.30,
                QualityDimension.RELEVANCE: 0.25,
                QualityDimension.COMPLETENESS: 0.20,
                QualityDimension.CLARITY: 0.15,
                QualityDimension.CODE_QUALITY: 0.10
            },
            "explanation": {
                QualityDimension.CLARITY: 0.30,
                QualityDimension.COMPLETENESS: 0.25,
                QualityDimension.RELEVANCE: 0.20,
                QualityDimension.CORRECTNESS: 0.15,
                QualityDimension.CREATIVITY: 0.10
            },
            "code_review": {
                QualityDimension.CORRECTNESS: 0.25,
                QualityDimension.CODE_QUALITY: 0.20,
                QualityDimension.SECURITY: 0.20,
                QualityDimension.BEST_PRACTICES: 0.15,
                QualityDimension.COMPLETENESS: 0.10,
                QualityDimension.CLARITY: 0.10
            },
            "architecture": {
                QualityDimension.COMPLETENESS: 0.25,
                QualityDimension.CORRECTNESS: 0.20,
                QualityDimension.BEST_PRACTICES: 0.20,
                QualityDimension.CLARITY: 0.15,
                QualityDimension.CREATIVITY: 0.10,
                QualityDimension.MAINTAINABILITY: 0.10
            }
        }
    
    async def score_response(
        self,
        user_request: str,
        ai_response: str,
        task_type: str,
        context_info: Dict[str, Any] = None,
        model_name: str = "unknown"
    ) -> QualityScore:
        """Score response quality across multiple dimensions"""
        
        context_info = context_info or {}
        quality_score = QualityScore()
        
        # Get task-specific weights
        weights = self.task_quality_weights.get(task_type, self._get_default_weights())
        
        # Score each dimension
        for dimension in QualityDimension:
            if dimension in weights:
                score = await self._score_dimension(
                    dimension, user_request, ai_response, task_type, context_info, model_name
                )
                quality_score.dimension_scores[dimension] = score
        
        # Calculate overall score
        quality_score.overall_score = sum(
            score * weights.get(dimension, 0.0) 
            for dimension, score in quality_score.dimension_scores.items()
        )
        
        # Calculate confidence based on consistency of scores
        score_values = list(quality_score.dimension_scores.values())
        if score_values:
            avg_score = sum(score_values) / len(score_values)
            variance = sum((s - avg_score) ** 2 for s in score_values) / len(score_values)
            quality_score.confidence = max(0.0, 1.0 - (variance / 0.25))  # Normalize variance
        
        # Identify issues and suggestions
        await self._analyze_quality_issues(quality_score, user_request, ai_response, task_type)
        
        # Add metadata
        quality_score.metadata = {
            "task_type": task_type,
            "model_name": model_name,
            "response_length": len(ai_response),
            "request_length": len(user_request),
            "code_blocks_count": len(re.findall(r'```', ai_response)) // 2,
            "has_examples": "example" in ai_response.lower(),
            "has_explanations": any(word in ai_response.lower() for word in ["because", "therefore", "this means", "in other words"])
        }
        
        return quality_score
    
    async def _score_dimension(
        self,
        dimension: QualityDimension,
        user_request: str,
        ai_response: str,
        task_type: str,
        context_info: Dict[str, Any],
        model_name: str
    ) -> float:
        """Score a specific quality dimension"""
        
        if dimension == QualityDimension.CORRECTNESS:
            return await self._score_correctness(user_request, ai_response, task_type)
        elif dimension == QualityDimension.COMPLETENESS:
            return await self._score_completeness(user_request, ai_response, task_type)
        elif dimension == QualityDimension.CLARITY:
            return await self._score_clarity(ai_response, task_type)
        elif dimension == QualityDimension.RELEVANCE:
            return await self._score_relevance(user_request, ai_response)
        elif dimension == QualityDimension.CODE_QUALITY:
            return await self._score_code_quality(ai_response, task_type)
        elif dimension == QualityDimension.SECURITY:
            return await self._score_security(ai_response, task_type)
        elif dimension == QualityDimension.PERFORMANCE:
            return await self._score_performance(ai_response, task_type)
        elif dimension == QualityDimension.MAINTAINABILITY:
            return await self._score_maintainability(ai_response, task_type)
        elif dimension == QualityDimension.BEST_PRACTICES:
            return await self._score_best_practices(ai_response, task_type)
        elif dimension == QualityDimension.CREATIVITY:
            return await self._score_creativity(user_request, ai_response, task_type)
        else:
            return 0.5  # Default neutral score
    
    async def _score_correctness(self, user_request: str, ai_response: str, task_type: str) -> float:
        """Score response correctness"""
        score = 0.5  # Base score
        
        # Check for obvious errors
        if "error" in ai_response.lower() and task_type != "debugging":
            score -= 0.3
        
        # Check for code syntax if applicable
        if task_type in ["code_generation", "debugging", "code_review"]:
            code_blocks = re.findall(r'```(?:\w+)?\n(.*?)\n```', ai_response, re.DOTALL)
            if code_blocks:
                syntax_score = self._check_code_syntax(code_blocks)
                score += syntax_score * 0.4
        
        # Check for logical consistency
        if self._has_logical_consistency(ai_response):
            score += 0.2
        
        # Check if response addresses the request
        if self._addresses_request(user_request, ai_response):
            score += 0.3
        else:
            score -= 0.4
        
        return max(0.0, min(1.0, score))
    
    async def _score_completeness(self, user_request: str, ai_response: str, task_type: str) -> float:
        """Score response completeness"""
        score = 0.3  # Base score
        
        # Length-based scoring
        response_length = len(ai_response)
        if task_type == "code_generation":
            if response_length > 200:
                score += 0.3
            if response_length > 500:
                score += 0.2
        elif task_type == "explanation":
            if response_length > 100:
                score += 0.2
            if response_length > 300:
                score += 0.2
        
        # Check for required elements
        required_elements = self._get_required_elements(task_type, user_request)
        present_elements = sum(1 for element in required_elements if element in ai_response.lower())
        if required_elements:
            score += 0.3 * (present_elements / len(required_elements))
        
        # Check for examples and explanations
        if "example" in ai_response.lower() or "```" in ai_response:
            score += 0.1
        
        if any(word in ai_response.lower() for word in ["note:", "important:", "remember:", "tip:"]):
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    async def _score_clarity(self, ai_response: str, task_type: str) -> float:
        """Score response clarity"""
        score = 0.4  # Base score
        
        # Structure scoring
        if ai_response.count('\n\n') >= 2:  # Good paragraph separation
            score += 0.2
        
        # Numbered lists or bullet points
        if re.search(r'\d+\.|\*|\-', ai_response):
            score += 0.1
        
        # Code formatting
        if '```' in ai_response:
            score += 0.1
        
        # Clear explanations
        explanation_words = ["this means", "in other words", "for example", "specifically", "that is"]
        if any(phrase in ai_response.lower() for phrase in explanation_words):
            score += 0.1
        
        # Readability (simple metric)
        sentences = ai_response.split('.')
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            if 10 <= avg_sentence_length <= 25:  # Good sentence length
                score += 0.1
        
        return max(0.0, min(1.0, score))
    
    async def _score_relevance(self, user_request: str, ai_response: str) -> float:
        """Score response relevance to request"""
        score = 0.3  # Base score
        
        # Keyword overlap
        request_words = set(user_request.lower().split())
        response_words = set(ai_response.lower().split())
        
        # Remove common words
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "can", "may", "might"}
        request_words -= common_words
        response_words -= common_words
        
        if request_words:
            overlap = len(request_words.intersection(response_words))
            relevance_ratio = overlap / len(request_words)
            score += relevance_ratio * 0.4
        
        # Topic consistency
        if self._maintains_topic_consistency(user_request, ai_response):
            score += 0.3
        
        return max(0.0, min(1.0, score))
    
    async def _score_code_quality(self, ai_response: str, task_type: str) -> float:
        """Score code quality in response"""
        if task_type not in ["code_generation", "debugging", "code_review", "refactoring"]:
            return 0.8  # Not applicable, assume good
        
        score = 0.4  # Base score
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)\n```', ai_response, re.DOTALL)
        
        if not code_blocks:
            return 0.2  # No code when expected
        
        for code in code_blocks:
            # Check for good practices
            if 'def ' in code or 'function' in code:
                score += 0.1  # Function definitions
            
            if any(keyword in code for keyword in ['import', 'from', 'require', '#include']):
                score += 0.1  # Proper imports
            
            if any(keyword in code for keyword in ['try:', 'catch', 'except:', 'finally:']):
                score += 0.1  # Error handling
            
            if '"""' in code or "'''" in code or '//' in code or '#' in code:
                score += 0.1  # Documentation/comments
            
            if re.search(r'\bclass\s+\w+', code):
                score += 0.1  # Class definitions
            
            # Check for bad practices
            if 'global ' in code:
                score -= 0.1  # Global variables
            
            if len(code.splitlines()) > 50 and code.count('def ') < 2:
                score -= 0.1  # Long functions
        
        return max(0.0, min(1.0, score))
    
    async def _score_security(self, ai_response: str, task_type: str) -> float:
        """Score security considerations in response"""
        if task_type not in ["code_generation", "code_review", "architecture"]:
            return 0.9  # Not applicable, assume good
        
        score = 0.7  # Base score (assume secure by default)
        
        # Security red flags
        security_issues = [
            'eval(', 'exec(', 'system(', 'shell_exec', 'passthru',
            'md5(', 'sha1(', 'plain text password', 'hardcoded password',
            'sql injection', 'xss', 'csrf', 'session hijacking'
        ]
        
        for issue in security_issues:
            if issue in ai_response.lower():
                score -= 0.2
        
        # Security best practices
        security_goods = [
            'bcrypt', 'scrypt', 'argon2', 'pbkdf2', 'csrf token',
            'prepared statement', 'parameterized query', 'input validation',
            'sanitization', 'authentication', 'authorization', 'https',
            'tls', 'ssl', 'encryption', 'secure', 'validate'
        ]
        
        for good in security_goods:
            if good in ai_response.lower():
                score += 0.05
        
        return max(0.0, min(1.0, score))
    
    async def _score_performance(self, ai_response: str, task_type: str) -> float:
        """Score performance considerations"""
        score = 0.6  # Base score
        
        # Performance indicators
        perf_indicators = [
            'o(n', 'time complexity', 'space complexity', 'algorithm',
            'optimization', 'cache', 'index', 'lazy loading', 'pagination',
            'async', 'parallel', 'concurrent', 'batch', 'bulk', 'efficient'
        ]
        
        for indicator in perf_indicators:
            if indicator in ai_response.lower():
                score += 0.1
        
        # Performance anti-patterns
        anti_patterns = [
            'nested loop', 'n+1 query', 'memory leak', 'blocking',
            'synchronous', 'inefficient', 'slow query'
        ]
        
        for pattern in anti_patterns:
            if pattern in ai_response.lower():
                score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    async def _score_maintainability(self, ai_response: str, task_type: str) -> float:
        """Score code maintainability"""
        if task_type not in ["code_generation", "architecture", "refactoring"]:
            return 0.8  # Not applicable
        
        score = 0.5  # Base score
        
        # Maintainability indicators
        maintainability_indicators = [
            'modular', 'reusable', 'testable', 'readable', 'clean',
            'single responsibility', 'separation of concerns', 'dry',
            'solid principles', 'design pattern', 'documentation',
            'comment', 'type hint', 'interface', 'abstract'
        ]
        
        for indicator in maintainability_indicators:
            if indicator in ai_response.lower():
                score += 0.08
        
        return max(0.0, min(1.0, score))
    
    async def _score_best_practices(self, ai_response: str, task_type: str) -> float:
        """Score adherence to best practices"""
        score = 0.5  # Base score
        
        # General best practices
        best_practices = [
            'pep 8', 'eslint', 'prettier', 'linting', 'formatting',
            'testing', 'unit test', 'integration test', 'git',
            'version control', 'code review', 'continuous integration',
            'documentation', 'readme', 'api documentation'
        ]
        
        for practice in best_practices:
            if practice in ai_response.lower():
                score += 0.08
        
        return max(0.0, min(1.0, score))
    
    async def _score_creativity(self, user_request: str, ai_response: str, task_type: str) -> float:
        """Score creativity and innovation in response"""
        score = 0.5  # Base score
        
        # Creativity indicators
        creativity_indicators = [
            'alternative', 'another approach', 'different way', 'creative',
            'innovative', 'novel', 'unique', 'interesting', 'elegant',
            'clever', 'optimization', 'improvement', 'enhancement'
        ]
        
        for indicator in creativity_indicators:
            if indicator in ai_response.lower():
                score += 0.1
        
        # Multiple solutions/approaches
        if ai_response.count('option') > 1 or ai_response.count('approach') > 1:
            score += 0.2
        
        return max(0.0, min(1.0, score))
    
    def _check_code_syntax(self, code_blocks: List[str]) -> float:
        """Check syntax validity of code blocks"""
        if not code_blocks:
            return 0.0
        
        valid_blocks = 0
        for code in code_blocks:
            try:
                # Try to parse as Python (basic check)
                if self._looks_like_python(code):
                    ast.parse(code)
                    valid_blocks += 1
                else:
                    # For non-Python code, do basic checks
                    if self._basic_syntax_check(code):
                        valid_blocks += 1
            except:
                # Syntax error
                continue
        
        return valid_blocks / len(code_blocks) if code_blocks else 0.0
    
    def _looks_like_python(self, code: str) -> bool:
        """Check if code looks like Python"""
        python_indicators = ['def ', 'class ', 'import ', 'from ', 'if __name__', 'print(']
        return any(indicator in code for indicator in python_indicators)
    
    def _basic_syntax_check(self, code: str) -> bool:
        """Basic syntax check for non-Python code"""
        # Check for balanced brackets
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        
        for char in code:
            if char in brackets:
                stack.append(char)
            elif char in brackets.values():
                if not stack:
                    return False
                if brackets[stack.pop()] != char:
                    return False
        
        return len(stack) == 0
    
    def _has_logical_consistency(self, response: str) -> bool:
        """Check for logical consistency in response"""
        # Simple check for contradictory statements
        contradictions = [
            ('always', 'never'),
            ('should', 'should not'),
            ('must', 'must not'),
            ('recommended', 'not recommended'),
            ('secure', 'insecure'),
            ('fast', 'slow'),
            ('efficient', 'inefficient')
        ]
        
        response_lower = response.lower()
        for pos, neg in contradictions:
            if pos in response_lower and neg in response_lower:
                # Check if they're in close proximity (potential contradiction)
                pos_index = response_lower.find(pos)
                neg_index = response_lower.find(neg)
                if abs(pos_index - neg_index) < 200:  # Within 200 characters
                    return False
        
        return True
    
    def _addresses_request(self, request: str, response: str) -> bool:
        """Check if response addresses the request"""
        request_words = set(request.lower().split())
        response_words = set(response.lower().split())
        
        # Remove common words
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        request_words -= common_words
        response_words -= common_words
        
        if not request_words:
            return True
        
        overlap = len(request_words.intersection(response_words))
        return overlap / len(request_words) >= 0.3  # At least 30% overlap
    
    def _get_required_elements(self, task_type: str, request: str) -> List[str]:
        """Get required elements for a task type"""
        elements_map = {
            "code_generation": ["def", "function", "class", "import", "return"],
            "debugging": ["error", "issue", "problem", "fix", "solution"],
            "explanation": ["because", "reason", "how", "why", "means"],
            "code_review": ["improve", "suggest", "recommend", "issue", "good"],
            "architecture": ["component", "module", "layer", "design", "pattern"],
            "testing": ["test", "assert", "expect", "verify", "validate"]
        }
        
        base_elements = elements_map.get(task_type, [])
        
        # Add request-specific elements
        if "example" in request.lower():
            base_elements.append("example")
        if "step" in request.lower():
            base_elements.extend(["step", "first", "then"])
        
        return base_elements
    
    def _maintains_topic_consistency(self, request: str, response: str) -> bool:
        """Check if response maintains topic consistency"""
        # Extract main topics from request
        topics = []
        tech_keywords = [
            'python', 'javascript', 'java', 'cpp', 'c++', 'go', 'rust',
            'react', 'vue', 'angular', 'django', 'flask', 'spring',
            'database', 'sql', 'nosql', 'api', 'rest', 'graphql',
            'machine learning', 'ai', 'data science', 'web development'
        ]
        
        for keyword in tech_keywords:
            if keyword in request.lower():
                topics.append(keyword)
        
        if not topics:
            return True  # No specific topics identified
        
        # Check if topics are mentioned in response
        mentioned_topics = sum(1 for topic in topics if topic in response.lower())
        return mentioned_topics / len(topics) >= 0.5
    
    def _get_default_weights(self) -> Dict[QualityDimension, float]:
        """Get default quality dimension weights"""
        return {
            QualityDimension.CORRECTNESS: 0.25,
            QualityDimension.COMPLETENESS: 0.20,
            QualityDimension.CLARITY: 0.15,
            QualityDimension.RELEVANCE: 0.15,
            QualityDimension.CODE_QUALITY: 0.10,
            QualityDimension.SECURITY: 0.05,
            QualityDimension.PERFORMANCE: 0.05,
            QualityDimension.BEST_PRACTICES: 0.05
        }
    
    async def _analyze_quality_issues(
        self,
        quality_score: QualityScore,
        user_request: str,
        ai_response: str,
        task_type: str
    ):
        """Analyze quality issues and provide suggestions"""
        
        # Identify issues based on low scores
        for dimension, score in quality_score.dimension_scores.items():
            threshold = self.quality_thresholds.get(dimension, 0.6)
            if score < threshold:
                issue_msg, suggestion_msg = self._get_dimension_feedback(dimension, score, task_type)
                quality_score.issues.append(issue_msg)
                quality_score.suggestions.append(suggestion_msg)
        
        # Global issues
        if len(ai_response) < 50:
            quality_score.issues.append("Response is too brief")
            quality_score.suggestions.append("Provide more detailed explanation or examples")
        
        if task_type in ["code_generation", "debugging"] and "```" not in ai_response:
            quality_score.issues.append("No code blocks found in response")
            quality_score.suggestions.append("Include properly formatted code examples")
    
    def _get_dimension_feedback(self, dimension: QualityDimension, score: float, task_type: str) -> Tuple[str, str]:
        """Get feedback for specific quality dimension"""
        feedback_map = {
            QualityDimension.CORRECTNESS: (
                "Response may contain factual errors or incorrect information",
                "Verify facts and ensure all information provided is accurate"
            ),
            QualityDimension.COMPLETENESS: (
                "Response appears incomplete or missing important details",
                "Provide more comprehensive coverage of the topic with additional details and examples"
            ),
            QualityDimension.CLARITY: (
                "Response lacks clarity and may be difficult to understand",
                "Use clearer language, better structure, and more explanations"
            ),
            QualityDimension.RELEVANCE: (
                "Response doesn't fully address the user's specific request",
                "Focus more directly on what the user asked and provide more relevant information"
            ),
            QualityDimension.CODE_QUALITY: (
                "Code quality could be improved with better practices",
                "Follow coding standards, add proper error handling, and include documentation"
            ),
            QualityDimension.SECURITY: (
                "Response may have security vulnerabilities or concerns",
                "Address security best practices and potential vulnerabilities"
            ),
            QualityDimension.PERFORMANCE: (
                "Response doesn't adequately address performance considerations",
                "Include performance optimization suggestions and complexity analysis"
            ),
            QualityDimension.MAINTAINABILITY: (
                "Code maintainability could be improved",
                "Focus on modular design, clear naming, and proper documentation"
            ),
            QualityDimension.BEST_PRACTICES: (
                "Response doesn't follow industry best practices",
                "Include best practices and standard conventions for the given technology"
            ),
            QualityDimension.CREATIVITY: (
                "Response lacks creative solutions or alternative approaches",
                "Consider providing multiple approaches or more innovative solutions"
            )
        }
        
        return feedback_map.get(dimension, ("Quality issue identified", "Improve response quality"))

class AutoRetrySystem:
    """Automatic retry system with intelligent decision making"""
    
    def __init__(self, max_retries: int = 3, quality_threshold: float = 0.7):
        self.max_retries = max_retries
        self.quality_threshold = quality_threshold
        self.scorer = AdvancedQualityScorer()
        self.retry_history: Dict[str, List[RetryAttempt]] = defaultdict(list)
        
        # Retry strategies for different issues
        self.retry_strategies = {
            RetryReason.LOW_QUALITY: self._strategy_improve_quality,
            RetryReason.INCOMPLETE_RESPONSE: self._strategy_complete_response,
            RetryReason.SYNTAX_ERROR: self._strategy_fix_syntax,
            RetryReason.SECURITY_ISSUE: self._strategy_improve_security,
            RetryReason.OFF_TOPIC: self._strategy_stay_on_topic,
            RetryReason.TIMEOUT: self._strategy_handle_timeout,
            RetryReason.MODEL_ERROR: self._strategy_handle_model_error
        }
    
    async def should_retry(
        self,
        request_id: str,
        user_request: str,
        ai_response: str,
        task_type: str,
        model_name: str,
        context_info: Dict[str, Any] = None,
        processing_time: float = 0.0
    ) -> Tuple[bool, Optional[RetryReason], Optional[Dict[str, Any]]]:
        """Determine if response should be retried"""
        
        # Check retry limit
        if len(self.retry_history[request_id]) >= self.max_retries:
            return False, None, None
        
        # Score the response
        quality_score = await self.scorer.score_response(
            user_request, ai_response, task_type, context_info, model_name
        )
        
        # Determine if retry is needed
        retry_reason = None
        retry_context = {}
        
        # Check overall quality
        if quality_score.overall_score < self.quality_threshold:
            retry_reason = RetryReason.LOW_QUALITY
            retry_context["low_dimensions"] = [
                dim.value for dim, score in quality_score.dimension_scores.items()
                if score < self.scorer.quality_thresholds.get(dim, 0.6)
            ]
        
        # Check for specific issues
        elif quality_score.dimension_scores.get(QualityDimension.COMPLETENESS, 1.0) < 0.4:
            retry_reason = RetryReason.INCOMPLETE_RESPONSE
        
        elif task_type in ["code_generation", "debugging"] and "```" not in ai_response:
            retry_reason = RetryReason.INCOMPLETE_RESPONSE
            retry_context["missing_code"] = True
        
        elif quality_score.dimension_scores.get(QualityDimension.SECURITY, 1.0) < 0.5:
            retry_reason = RetryReason.SECURITY_ISSUE
        
        elif quality_score.dimension_scores.get(QualityDimension.RELEVANCE, 1.0) < 0.4:
            retry_reason = RetryReason.OFF_TOPIC
        
        elif processing_time > 30.0:  # 30 second timeout
            retry_reason = RetryReason.TIMEOUT
        
        elif "error" in ai_response.lower() and "error handling" not in ai_response.lower():
            retry_reason = RetryReason.MODEL_ERROR
        
        # Check for syntax errors in code
        if retry_reason is None and task_type in ["code_generation", "debugging"]:
            code_blocks = re.findall(r'```(?:\w+)?\n(.*?)\n```', ai_response, re.DOTALL)
            if code_blocks and self.scorer._check_code_syntax(code_blocks) < 0.5:
                retry_reason = RetryReason.SYNTAX_ERROR
        
        should_retry_decision = retry_reason is not None
        
        if should_retry_decision:
            # Record retry attempt
            attempt = RetryAttempt(
                attempt_number=len(self.retry_history[request_id]) + 1,
                reason=retry_reason,
                previous_score=quality_score.overall_score,
                model_used=model_name
            )
            self.retry_history[request_id].append(attempt)
        
        return should_retry_decision, retry_reason, retry_context
    
    async def get_retry_strategy(
        self,
        request_id: str,
        retry_reason: RetryReason,
        retry_context: Dict[str, Any],
        user_request: str,
        previous_response: str,
        task_type: str
    ) -> Dict[str, Any]:
        """Get strategy for retry attempt"""
        
        strategy_func = self.retry_strategies.get(retry_reason, self._strategy_default)
        return await strategy_func(request_id, retry_context, user_request, previous_response, task_type)
    
    async def _strategy_improve_quality(
        self,
        request_id: str,
        retry_context: Dict[str, Any],
        user_request: str,
        previous_response: str,
        task_type: str
    ) -> Dict[str, Any]:
        """Strategy to improve overall quality"""
        
        low_dimensions = retry_context.get("low_dimensions", [])
        
        improvements = []
        if "correctness" in low_dimensions:
            improvements.append("Focus on accuracy and fact-checking")
        if "completeness" in low_dimensions:
            improvements.append("Provide more comprehensive and detailed information")
        if "clarity" in low_dimensions:
            improvements.append("Use clearer explanations and better structure")
        if "code_quality" in low_dimensions:
            improvements.append("Follow best coding practices and standards")
        
        return {
            "system_prompt_addition": f"Previous response had quality issues. Please improve by: {'; '.join(improvements)}",
            "temperature_adjustment": -0.1,  # Lower temperature for more consistent quality
            "use_different_model": len(self.retry_history[request_id]) >= 2,
            "additional_context": "Focus on providing the highest quality response possible."
        }
    
    async def _strategy_complete_response(
        self,
        request_id: str,
        retry_context: Dict[str, Any],
        user_request: str,
        previous_response: str,
        task_type: str
    ) -> Dict[str, Any]:
        """Strategy to complete incomplete response"""
        
        additions = []
        if retry_context.get("missing_code"):
            additions.append("Include properly formatted code examples")
        
        additions.append("Ensure all parts of the request are fully addressed")
        
        return {
            "system_prompt_addition": f"Previous response was incomplete. Please provide a complete response that: {'; '.join(additions)}",
            "temperature_adjustment": 0.0,
            "additional_context": f"Make sure to fully address: {user_request}",
            "require_minimum_length": True
        }
    
    async def _strategy_fix_syntax(
        self,
        request_id: str,
        retry_context: Dict[str, Any],
        user_request: str,
        previous_response: str,
        task_type: str
    ) -> Dict[str, Any]:
        """Strategy to fix syntax errors"""
        
        return {
            "system_prompt_addition": "Previous response contained syntax errors. Ensure all code is syntactically correct and properly formatted.",
            "temperature_adjustment": -0.2,  # Much lower temperature for syntax accuracy
            "additional_context": "Double-check all code syntax before responding.",
            "validation_required": True
        }
    
    async def _strategy_improve_security(
        self,
        request_id: str,
        retry_context: Dict[str, Any],
        user_request: str,
        previous_response: str,
        task_type: str
    ) -> Dict[str, Any]:
        """Strategy to improve security"""
        
        return {
            "system_prompt_addition": "Previous response had security concerns. Focus on security best practices and avoid vulnerable patterns.",
            "temperature_adjustment": -0.1,
            "additional_context": "Prioritize security in your response and explain security considerations.",
            "security_focus": True
        }
    
    async def _strategy_stay_on_topic(
        self,
        request_id: str,
        retry_context: Dict[str, Any],
        user_request: str,
        previous_response: str,
        task_type: str
    ) -> Dict[str, Any]:
        """Strategy to stay on topic"""
        
        return {
            "system_prompt_addition": f"Previous response was off-topic. Focus specifically on: {user_request}",
            "temperature_adjustment": -0.1,
            "additional_context": "Stay focused on the user's specific request and avoid tangential information.",
            "relevance_check": True
        }
    
    async def _strategy_handle_timeout(
        self,
        request_id: str,
        retry_context: Dict[str, Any],
        user_request: str,
        previous_response: str,
        task_type: str
    ) -> Dict[str, Any]:
        """Strategy to handle timeouts"""
        
        return {
            "use_different_model": True,  # Try faster model
            "temperature_adjustment": 0.1,  # Slightly higher for faster generation
            "max_tokens_reduction": 0.7,   # Reduce max tokens for faster response
            "prioritize_speed": True
        }
    
    async def _strategy_handle_model_error(
        self,
        request_id: str,
        retry_context: Dict[str, Any],
        user_request: str,
        previous_response: str,
        task_type: str
    ) -> Dict[str, Any]:
        """Strategy to handle model errors"""
        
        return {
            "use_different_model": True,  # Definitely try different model
            "system_prompt_addition": "Provide a clear, error-free response to the user's request.",
            "temperature_adjustment": -0.1,
            "fallback_response": True
        }
    
    async def _strategy_default(
        self,
        request_id: str,
        retry_context: Dict[str, Any],
        user_request: str,
        previous_response: str,
        task_type: str
    ) -> Dict[str, Any]:
        """Default retry strategy"""
        
        return {
            "system_prompt_addition": "Previous response needs improvement. Please provide a better, more accurate response.",
            "temperature_adjustment": -0.05,
            "use_different_model": len(self.retry_history[request_id]) >= 2
        }
    
    def get_retry_statistics(self, request_id: str = None) -> Dict[str, Any]:
        """Get retry statistics"""
        
        if request_id:
            attempts = self.retry_history[request_id]
            return {
                "total_attempts": len(attempts),
                "retry_reasons": [attempt.reason.value for attempt in attempts],
                "models_used": [attempt.model_used for attempt in attempts],
                "improvements": [
                    attempts[i].previous_score - attempts[i-1].previous_score 
                    for i in range(1, len(attempts))
                ] if len(attempts) > 1 else []
            }
        else:
            # Global statistics
            total_requests = len(self.retry_history)
            total_retries = sum(len(attempts) for attempts in self.retry_history.values())
            
            return {
                "total_requests": total_requests,
                "total_retries": total_retries,
                "retry_rate": total_retries / max(1, total_requests),
                "average_retries_per_request": total_retries / max(1, total_requests),
                "most_common_retry_reasons": self._get_common_retry_reasons()
            }
    
    def _get_common_retry_reasons(self) -> Dict[str, int]:
        """Get most common retry reasons"""
        reason_counts = defaultdict(int)
        
        for attempts in self.retry_history.values():
            for attempt in attempts:
                reason_counts[attempt.reason.value] += 1
        
        return dict(sorted(reason_counts.items(), key=lambda x: x[1], reverse=True))

# Factory function for easy integration
def create_quality_retry_system(max_retries: int = 3, quality_threshold: float = 0.7) -> AutoRetrySystem:
    """Create configured auto-retry system"""
    return AutoRetrySystem(max_retries=max_retries, quality_threshold=quality_threshold)