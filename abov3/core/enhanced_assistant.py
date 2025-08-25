"""
ABOV3 Genesis - Enhanced AI Assistant with Optimized Ollama Integration
Next-generation AI assistant that achieves Claude-level performance using local Ollama models
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import logging
import re
import uuid

# Import our optimization systems
from .ollama_optimization import OllamaModelOptimizer, PromptTemplateEngine
from .context_manager import SmartContextManager, ContextPriority, ContentType
from .learning_system import AdaptiveLearningSystem, FeedbackType, QualityMetric
from .multi_model_manager import MultiModelManager
from .project_intelligence import ProjectIntelligence
from ..infrastructure.performance import PerformanceOptimizer, performance_context
from ..infrastructure.resilience import ErrorRecoveryManager, resilience_context

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Individual conversation turn"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])

@dataclass
class AssistantSession:
    """Assistant session with conversation history and context"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    user_id: str = "anonymous"
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    conversation_history: List[ConversationTurn] = field(default_factory=list)
    context_data: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    total_interactions: int = 0
    quality_scores: List[float] = field(default_factory=list)

class EnhancedAIAssistant:
    """
    Enhanced AI Assistant that orchestrates all optimization systems
    to deliver Claude-level performance using local Ollama models
    """
    
    def __init__(self, project_path: Optional[Path] = None):
        self.project_path = project_path
        self.abov3_dir = project_path / '.abov3' if project_path else None
        
        # Core systems
        self.model_manager: Optional[MultiModelManager] = None
        self.optimizer: Optional[OllamaModelOptimizer] = None
        self.context_manager: Optional[SmartContextManager] = None
        self.learning_system: Optional[AdaptiveLearningSystem] = None
        self.project_intelligence: Optional[ProjectIntelligence] = None
        self.performance_optimizer: Optional[PerformanceOptimizer] = None
        self.error_manager: Optional[ErrorRecoveryManager] = None
        
        # Session management
        self.active_sessions: Dict[str, AssistantSession] = {}
        self.session_timeout_hours = 24
        
        # Quality assurance
        self.quality_threshold = 0.7
        self.max_retries = 2
        self.enable_quality_checks = True
        
        # Performance tracking
        self.performance_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "average_quality": 0.0,
            "average_response_time": 0.0,
            "cache_hit_rate": 0.0,
            "initialization_time": 0.0
        }
        
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """Initialize all systems and prepare for operation"""
        start_time = time.time()
        
        try:
            logger.info("Initializing Enhanced AI Assistant...")
            
            # Initialize core systems in parallel for faster startup
            init_tasks = [
                self._init_performance_optimizer(),
                self._init_error_manager(),
                self._init_project_intelligence(),
                self._init_learning_system()
            ]
            
            await asyncio.gather(*init_tasks)
            
            # Initialize systems that depend on others
            await self._init_context_manager()
            await self._init_optimizer()
            await self._init_model_manager()
            
            # Warm up the system
            await self._warm_up_system()
            
            self.is_initialized = True
            self.performance_stats["initialization_time"] = time.time() - start_time
            
            logger.info(f"Enhanced AI Assistant initialized in {self.performance_stats['initialization_time']:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced AI Assistant: {e}")
            return False
    
    async def _init_performance_optimizer(self):
        """Initialize performance optimizer"""
        try:
            from ..infrastructure.performance import PerformanceOptimizer, PerformanceLevel
            self.performance_optimizer = PerformanceOptimizer(
                PerformanceLevel.PRODUCTION,
                self.project_path
            )
            logger.debug("Performance optimizer initialized")
        except Exception as e:
            logger.warning(f"Performance optimizer initialization failed: {e}")
    
    async def _init_error_manager(self):
        """Initialize error recovery manager"""
        try:
            self.error_manager = ErrorRecoveryManager(self.project_path)
            
            # Register fallback handlers
            self.error_manager.register_fallback(
                "AIModelNotAvailable", 
                self._fallback_model_unavailable
            )
            self.error_manager.register_fallback(
                "TimeoutError",
                self._fallback_timeout_recovery
            )
            self.error_manager.register_fallback(
                "ConnectionError",
                self._fallback_connection_error
            )
            
            logger.debug("Error recovery manager initialized")
        except Exception as e:
            logger.warning(f"Error manager initialization failed: {e}")
    
    async def _init_project_intelligence(self):
        """Initialize project intelligence"""
        try:
            if self.project_path:
                self.project_intelligence = ProjectIntelligence(self.project_path)
                # Perform initial project analysis in background
                asyncio.create_task(self.project_intelligence.analyze_project())
                logger.debug("Project intelligence initialized")
        except Exception as e:
            logger.warning(f"Project intelligence initialization failed: {e}")
    
    async def _init_learning_system(self):
        """Initialize adaptive learning system"""
        try:
            self.learning_system = AdaptiveLearningSystem(self.project_path)
            logger.debug("Learning system initialized")
        except Exception as e:
            logger.warning(f"Learning system initialization failed: {e}")
    
    async def _init_context_manager(self):
        """Initialize smart context manager"""
        try:
            self.context_manager = SmartContextManager(
                max_context_tokens=16384,  # Start with large context
                model_name="general"
            )
            logger.debug("Context manager initialized")
        except Exception as e:
            logger.warning(f"Context manager initialization failed: {e}")
    
    async def _init_optimizer(self):
        """Initialize Ollama optimizer"""
        try:
            from .ollama_optimization import optimize_ollama_for_genesis
            self.optimizer = await optimize_ollama_for_genesis(self.project_path)
            logger.debug("Ollama optimizer initialized")
        except Exception as e:
            logger.warning(f"Ollama optimizer initialization failed: {e}")
    
    async def _init_model_manager(self):
        """Initialize multi-model manager"""
        try:
            self.model_manager = MultiModelManager(self.project_path)
            await self.model_manager.initialize_systems(
                self.optimizer,
                self.context_manager,
                self.learning_system
            )
            
            # Optimize context manager for available models
            if self.model_manager.model_clients:
                first_model = next(iter(self.model_manager.model_clients.keys()))
                if self.context_manager:
                    self.context_manager.optimize_for_model(first_model)
                    
            logger.debug("Multi-model manager initialized")
        except Exception as e:
            logger.warning(f"Model manager initialization failed: {e}")
    
    async def _warm_up_system(self):
        """Warm up the system with a test request"""
        try:
            if self.model_manager and self.model_manager.model_clients:
                # Simple warm-up request
                await self.model_manager.process_request(
                    "Hello",
                    task_type="conversation",
                    context_info={"warm_up": True}
                )
                logger.debug("System warm-up completed")
        except Exception as e:
            logger.debug(f"System warm-up failed (non-critical): {e}")
    
    async def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        user_id: str = "anonymous",
        task_type: str = "auto_detect",
        preferences: Dict[str, Any] = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Main chat interface with intelligent task detection and optimization
        """
        if not self.is_initialized:
            return {
                "success": False,
                "error": "Assistant not initialized. Call initialize() first.",
                "response": ""
            }
        
        start_time = time.time()
        preferences = preferences or {}
        context = context or {}
        
        # Performance and resilience context
        async def _process_with_optimization():
            async with performance_context(self.performance_optimizer) if self.performance_optimizer else self._dummy_context():
                async with resilience_context(self.error_manager) if self.error_manager else self._dummy_context():
                    return await self._process_chat_request(
                        message, session_id, user_id, task_type, preferences, context, start_time
                    )
        
        return await _process_with_optimization()
    
    async def _dummy_context(self):
        """Dummy context manager for when systems are unavailable"""
        class DummyContext:
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
        return DummyContext()
    
    async def _process_chat_request(
        self,
        message: str,
        session_id: Optional[str],
        user_id: str,
        task_type: str,
        preferences: Dict[str, Any],
        context: Dict[str, Any],
        start_time: float
    ) -> Dict[str, Any]:
        """Process the chat request with full optimization"""
        
        try:
            # Get or create session
            session = self._get_or_create_session(session_id, user_id)
            session.preferences.update(preferences)
            
            # Auto-detect task type if needed
            if task_type == "auto_detect":
                task_type = self._detect_task_type(message, session)
            
            # Build enhanced context
            enhanced_context = await self._build_enhanced_context(
                message, session, task_type, context
            )
            
            # Cache check for similar requests
            cache_key = self._create_cache_key(message, task_type, enhanced_context)
            cached_response = None
            
            if self.performance_optimizer:
                cached_response = await self.performance_optimizer.get_cached_response(cache_key)
                if cached_response:
                    self.performance_stats["cache_hit_rate"] += 1
                    return self._format_cached_response(cached_response, session, start_time)
            
            # Process request with quality assurance
            response_result = await self._process_with_quality_assurance(
                message, task_type, enhanced_context, session
            )
            
            if response_result["success"]:
                # Record conversation turn
                self._record_conversation_turn(session, "user", message)
                self._record_conversation_turn(session, "assistant", response_result["response"])
                
                # Update context manager with new conversation
                if self.context_manager:
                    self.context_manager.add_conversation_turn("user", message)
                    self.context_manager.add_conversation_turn("assistant", response_result["response"])
                
                # Cache successful response
                if self.performance_optimizer:
                    await self.performance_optimizer.cache_response(
                        cache_key, response_result, ttl=1800  # 30 minutes
                    )
                
                # Update session
                session.total_interactions += 1
                session.last_activity = time.time()
                
                # Update stats
                self.performance_stats["total_requests"] += 1
                self.performance_stats["successful_requests"] += 1
                
                processing_time = time.time() - start_time
                self.performance_stats["average_response_time"] = (
                    (self.performance_stats["average_response_time"] * (self.performance_stats["total_requests"] - 1) + processing_time) /
                    self.performance_stats["total_requests"]
                )
                
                # Format final response
                return self._format_successful_response(
                    response_result, session, processing_time, task_type, enhanced_context
                )
            
            else:
                # Handle failure
                self.performance_stats["total_requests"] += 1
                return self._format_error_response(
                    response_result.get("error", "Unknown error"),
                    session, time.time() - start_time
                )
                
        except Exception as e:
            logger.error(f"Chat processing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "I apologize, but I encountered an error while processing your request. Please try again.",
                "session_id": session_id,
                "processing_time_ms": (time.time() - start_time) * 1000
            }
    
    def _get_or_create_session(self, session_id: Optional[str], user_id: str) -> AssistantSession:
        """Get existing session or create new one"""
        if session_id and session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.last_activity = time.time()
            return session
        
        # Create new session
        session = AssistantSession(user_id=user_id)
        self.active_sessions[session.session_id] = session
        
        # Clean up old sessions
        self._cleanup_old_sessions()
        
        return session
    
    def _cleanup_old_sessions(self):
        """Clean up expired sessions"""
        current_time = time.time()
        timeout_seconds = self.session_timeout_hours * 3600
        
        expired_sessions = [
            sid for sid, session in self.active_sessions.items()
            if current_time - session.last_activity > timeout_seconds
        ]
        
        for sid in expired_sessions:
            del self.active_sessions[sid]
        
        if expired_sessions:
            logger.debug(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def _detect_task_type(self, message: str, session: AssistantSession) -> str:
        """Intelligently detect the task type from the message"""
        message_lower = message.lower()
        
        # Code generation indicators
        code_indicators = [
            "write", "create", "generate", "build", "implement", "code",
            "function", "class", "script", "program", "app", "website"
        ]
        
        if any(indicator in message_lower for indicator in code_indicators):
            # Check for specific code tasks
            if any(word in message_lower for word in ["bug", "error", "fix", "debug", "broken"]):
                return "debugging"
            elif any(word in message_lower for word in ["review", "check", "analyze", "improve"]):
                return "code_review"
            elif any(word in message_lower for word in ["test", "testing", "unit test", "pytest"]):
                return "testing"
            elif any(word in message_lower for word in ["architecture", "design", "structure", "system"]):
                return "architecture"
            elif any(word in message_lower for word in ["optimize", "performance", "speed up", "faster"]):
                return "optimization"
            else:
                return "code_generation"
        
        # Explanation requests
        explanation_indicators = ["explain", "how does", "what is", "why", "describe", "tell me about"]
        if any(indicator in message_lower for indicator in explanation_indicators):
            return "explanation"
        
        # Documentation requests
        doc_indicators = ["document", "documentation", "readme", "comments", "docstring"]
        if any(indicator in message_lower for indicator in doc_indicators):
            return "documentation"
        
        # Check conversation history for context
        if session.conversation_history:
            recent_turns = session.conversation_history[-6:]  # Last 3 exchanges
            recent_content = " ".join(turn.content.lower() for turn in recent_turns)
            
            if "code" in recent_content or "function" in recent_content:
                return "code_generation"
        
        # Default to conversation
        return "conversation"
    
    async def _build_enhanced_context(
        self,
        message: str,
        session: AssistantSession,
        task_type: str,
        additional_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build comprehensive context for the request"""
        
        enhanced_context = {
            "user_message": message,
            "task_type": task_type,
            "session_info": {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "total_interactions": session.total_interactions,
                "conversation_length": len(session.conversation_history)
            },
            "user_preferences": session.preferences.copy(),
            "timestamp": time.time()
        }
        
        # Add project intelligence
        if self.project_intelligence:
            try:
                project_context = self.project_intelligence.get_context_for_ai()
                enhanced_context["project_context"] = project_context
                
                # Analyze project if needed
                if self.project_intelligence.project_knowledge.get('confidence_score', 0) < 0.5:
                    asyncio.create_task(self.project_intelligence.analyze_project())
            except Exception as e:
                logger.debug(f"Project intelligence context error: {e}")
        
        # Add conversation history
        if session.conversation_history:
            recent_history = session.conversation_history[-10:]  # Last 5 exchanges
            enhanced_context["conversation_history"] = [
                {"role": turn.role, "content": turn.content, "timestamp": turn.timestamp}
                for turn in recent_history
            ]
        
        # Add code context if relevant
        if task_type in ["code_generation", "debugging", "code_review", "testing"]:
            enhanced_context["code_context"] = await self._extract_code_context(message, session)
        
        # Add learning context
        if self.learning_system:
            try:
                model_recommendations = self.learning_system.get_model_recommendations(task_type, message)
                enhanced_context["model_recommendations"] = model_recommendations
            except Exception as e:
                logger.debug(f"Learning system context error: {e}")
        
        # Merge additional context
        enhanced_context.update(additional_context)
        
        return enhanced_context
    
    async def _extract_code_context(self, message: str, session: AssistantSession) -> Dict[str, Any]:
        """Extract code-related context from message and session"""
        code_context = {}
        
        # Extract code blocks from message
        code_blocks = re.findall(r'```(\w*)\n(.*?)\n```', message, re.DOTALL)
        if code_blocks:
            code_context["code_blocks"] = [
                {"language": lang or "unknown", "code": code}
                for lang, code in code_blocks
            ]
        
        # Look for file references
        file_extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.html', '.css']
        mentioned_files = []
        for ext in file_extensions:
            files = re.findall(r'\S+' + ext, message)
            mentioned_files.extend(files)
        
        if mentioned_files:
            code_context["mentioned_files"] = mentioned_files
        
        # Extract programming language hints
        language_hints = []
        languages = ['python', 'javascript', 'typescript', 'java', 'c++', 'html', 'css', 'react', 'vue', 'angular']
        for lang in languages:
            if lang in message.lower():
                language_hints.append(lang)
        
        if language_hints:
            code_context["language_hints"] = language_hints
        
        # Look for error messages or stack traces
        if re.search(r'error|exception|traceback|stack trace', message, re.IGNORECASE):
            code_context["contains_error"] = True
            
            # Extract error patterns
            error_patterns = re.findall(r'(Error|Exception): (.+)', message)
            if error_patterns:
                code_context["error_messages"] = error_patterns
        
        return code_context
    
    async def _process_with_quality_assurance(
        self,
        message: str,
        task_type: str,
        context: Dict[str, Any],
        session: AssistantSession
    ) -> Dict[str, Any]:
        """Process request with quality assurance and retries"""
        
        best_response = None
        best_quality = 0.0
        attempts = 0
        
        while attempts < self.max_retries + 1:
            try:
                # Select and use model
                if self.model_manager:
                    response_result = await self.model_manager.process_request(
                        user_request=message,
                        task_type=task_type,
                        context_info=context
                    )
                else:
                    # Fallback to basic processing
                    response_result = await self._fallback_processing(message, task_type, context)
                
                if response_result.get("success"):
                    # Assess quality
                    quality_score = await self._assess_response_quality(
                        message, response_result["response"], task_type, context
                    )
                    
                    # Keep track of best response
                    if quality_score > best_quality:
                        best_quality = quality_score
                        best_response = response_result
                        best_response["quality_score"] = quality_score
                    
                    # If quality is good enough, use this response
                    if self.enable_quality_checks and quality_score >= self.quality_threshold:
                        logger.debug(f"Quality check passed: {quality_score:.3f}")
                        return best_response
                    elif not self.enable_quality_checks:
                        # Quality checks disabled, use first successful response
                        return response_result
                
                attempts += 1
                
                # If we have retries left and quality is poor, try again
                if attempts <= self.max_retries and best_quality < self.quality_threshold:
                    logger.debug(f"Quality below threshold ({best_quality:.3f}), retrying ({attempts}/{self.max_retries})")
                    await asyncio.sleep(0.5)  # Brief delay before retry
                    
            except Exception as e:
                logger.warning(f"Request processing attempt {attempts + 1} failed: {e}")
                attempts += 1
                if attempts <= self.max_retries:
                    await asyncio.sleep(1.0)  # Longer delay after error
        
        # Return best response we got, even if quality is below threshold
        if best_response:
            logger.info(f"Returning best available response with quality {best_quality:.3f}")
            return best_response
        else:
            return {"success": False, "error": "All processing attempts failed"}
    
    async def _assess_response_quality(
        self,
        user_request: str,
        ai_response: str,
        task_type: str,
        context: Dict[str, Any]
    ) -> float:
        """Assess the quality of an AI response"""
        
        quality_score = 0.5  # Start with neutral score
        
        # Basic completeness check
        if len(ai_response.strip()) < 10:
            return 0.1  # Very poor quality for extremely short responses
        
        # Length appropriateness (different expectations for different tasks)
        expected_lengths = {
            "code_generation": (200, 2000),
            "debugging": (150, 1500),
            "explanation": (100, 1000),
            "conversation": (50, 500)
        }
        
        min_len, max_len = expected_lengths.get(task_type, (50, 1000))
        response_len = len(ai_response)
        
        if min_len <= response_len <= max_len:
            quality_score += 0.2
        elif response_len < min_len * 0.5:
            quality_score -= 0.2
        elif response_len > max_len * 2:
            quality_score -= 0.1
        
        # Task-specific quality checks
        if task_type == "code_generation":
            quality_score += self._assess_code_quality(ai_response)
        elif task_type == "debugging":
            quality_score += self._assess_debug_quality(ai_response, user_request)
        elif task_type == "explanation":
            quality_score += self._assess_explanation_quality(ai_response, user_request)
        
        # General quality indicators
        if "```" in ai_response and task_type in ["code_generation", "debugging"]:
            quality_score += 0.1  # Proper code formatting
        
        if any(word in ai_response.lower() for word in ["error", "exception", "apologi"]):
            if task_type != "debugging":
                quality_score -= 0.1  # Unexpected errors
        
        # Coherence check (simple)
        sentences = ai_response.split('.')
        if len(sentences) > 2:
            quality_score += 0.1  # Well-structured response
        
        # Check for helpful elements
        helpful_indicators = ["example", "note:", "tip:", "important:", "remember:"]
        if any(indicator in ai_response.lower() for indicator in helpful_indicators):
            quality_score += 0.1
        
        return max(0.0, min(1.0, quality_score))  # Clamp to [0, 1]
    
    def _assess_code_quality(self, response: str) -> float:
        """Assess quality of code generation response"""
        quality_bonus = 0.0
        
        # Check for code blocks
        code_blocks = re.findall(r'```\w*\n(.*?)\n```', response, re.DOTALL)
        if code_blocks:
            quality_bonus += 0.2
            
            for code in code_blocks:
                # Check for good practices
                if 'def ' in code or 'function' in code:
                    quality_bonus += 0.1  # Function definition
                if 'class ' in code:
                    quality_bonus += 0.1  # Class definition
                if 'import ' in code or 'from ' in code:
                    quality_bonus += 0.05  # Imports
                if '"""' in code or "'''" in code:
                    quality_bonus += 0.1  # Documentation
                if 'try:' in code and 'except' in code:
                    quality_bonus += 0.1  # Error handling
        
        # Check for explanations
        if 'this code' in response.lower() or 'function' in response.lower():
            quality_bonus += 0.1
        
        return min(0.3, quality_bonus)  # Cap bonus at 0.3
    
    def _assess_debug_quality(self, response: str, user_request: str) -> float:
        """Assess quality of debugging response"""
        quality_bonus = 0.0
        
        debug_indicators = [
            'issue', 'problem', 'error', 'bug', 'fix', 'solution',
            'cause', 'reason', 'because', 'change', 'modify'
        ]
        
        found_indicators = sum(1 for indicator in debug_indicators if indicator in response.lower())
        quality_bonus += min(0.2, found_indicators * 0.05)
        
        # Check for structured approach
        numbered_list = re.search(r'\d+\.', response)
        if 'step' in response.lower() or numbered_list:
            quality_bonus += 0.1
        
        return min(0.3, quality_bonus)
    
    def _assess_explanation_quality(self, response: str, user_request: str) -> float:
        """Assess quality of explanation response"""
        quality_bonus = 0.0
        
        # Check for clear structure
        if any(word in response.lower() for word in ['first', 'second', 'then', 'next', 'finally']):
            quality_bonus += 0.1
        
        # Check for examples
        if 'example' in response.lower() or 'for instance' in response.lower():
            quality_bonus += 0.1
        
        # Check for comprehensive coverage
        if len(response.split('.')) >= 5:  # At least 5 sentences
            quality_bonus += 0.1
        
        return min(0.3, quality_bonus)
    
    async def _fallback_processing(
        self, message: str, task_type: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback processing when model manager is unavailable"""
        
        # Simple template-based responses
        fallback_responses = {
            "code_generation": "I'd be happy to help you with code generation. However, I'm currently experiencing technical difficulties with my AI models. Please try again in a moment, or consider using a simpler request.",
            "debugging": "I can help with debugging, but I'm currently having issues accessing my analysis tools. Please try again shortly.",
            "explanation": "I'd love to explain that concept, but I'm experiencing some technical issues. Please try your request again in a moment.",
            "conversation": "I'm here to help! However, I'm experiencing some technical difficulties right now. Please try again in a moment."
        }
        
        response = fallback_responses.get(task_type, fallback_responses["conversation"])
        
        return {
            "success": True,
            "response": response,
            "model_used": "fallback",
            "processing_time_ms": 10,
            "fallback": True
        }
    
    def _create_cache_key(self, message: str, task_type: str, context: Dict[str, Any]) -> str:
        """Create cache key for request"""
        import hashlib
        
        # Create key from essential components
        key_components = [
            message[:200],  # Truncated message
            task_type,
            str(context.get("project_context", "")),
            str(len(context.get("conversation_history", [])))
        ]
        
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
    
    def _record_conversation_turn(self, session: AssistantSession, role: str, content: str):
        """Record a conversation turn in the session"""
        turn = ConversationTurn(
            role=role,
            content=content,
            session_id=session.session_id
        )
        
        session.conversation_history.append(turn)
        
        # Limit conversation history length
        if len(session.conversation_history) > 50:  # Keep last 25 exchanges
            session.conversation_history = session.conversation_history[-50:]
    
    def _format_successful_response(
        self,
        response_result: Dict[str, Any],
        session: AssistantSession,
        processing_time: float,
        task_type: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format successful response with metadata"""
        
        quality_score = response_result.get("quality_score", 0.8)
        
        # Update session quality tracking
        session.quality_scores.append(quality_score)
        if len(session.quality_scores) > 20:  # Keep last 20 scores
            session.quality_scores = session.quality_scores[-20:]
        
        # Update system stats
        self.performance_stats["average_quality"] = (
            (self.performance_stats["average_quality"] * (self.performance_stats["successful_requests"] - 1) + quality_score) /
            self.performance_stats["successful_requests"]
        )
        
        return {
            "success": True,
            "response": response_result["response"],
            "session_id": session.session_id,
            "processing_time_ms": processing_time * 1000,
            "model_used": response_result.get("model_used", "unknown"),
            "model_confidence": response_result.get("model_confidence", 0.8),
            "quality_score": quality_score,
            "task_type": task_type,
            "optimization_applied": response_result.get("optimization_applied", False),
            "metadata": {
                "session_interactions": session.total_interactions,
                "average_session_quality": sum(session.quality_scores) / len(session.quality_scores) if session.quality_scores else 0.5,
                "context_size": len(str(context)),
                "response_length": len(response_result["response"])
            }
        }
    
    def _format_cached_response(
        self, cached_response: Dict[str, Any], session: AssistantSession, start_time: float
    ) -> Dict[str, Any]:
        """Format cached response"""
        processing_time = time.time() - start_time
        
        result = cached_response.copy()
        result.update({
            "session_id": session.session_id,
            "processing_time_ms": processing_time * 1000,
            "from_cache": True,
            "metadata": {
                "session_interactions": session.total_interactions,
                "cache_hit": True
            }
        })
        
        return result
    
    def _format_error_response(
        self, error_message: str, session: AssistantSession, processing_time: float
    ) -> Dict[str, Any]:
        """Format error response"""
        return {
            "success": False,
            "error": error_message,
            "response": "I apologize, but I encountered an error while processing your request. Please try again or rephrase your question.",
            "session_id": session.session_id,
            "processing_time_ms": processing_time * 1000,
            "suggestions": [
                "Try rephrasing your question",
                "Break down complex requests into smaller parts",
                "Check if your request contains any unusual characters or formatting"
            ]
        }
    
    # Fallback handlers for error recovery
    async def _fallback_model_unavailable(self, error_context) -> Dict[str, Any]:
        """Fallback when AI model is unavailable"""
        return {
            "response": "I'm currently unable to access my AI models. This might be due to high demand or technical maintenance. Please try again in a few minutes.",
            "suggestions": ["Wait a few minutes and try again", "Try a simpler request", "Check your internet connection"]
        }
    
    async def _fallback_timeout_recovery(self, error_context) -> Dict[str, Any]:
        """Fallback for timeout errors"""
        return {
            "response": "Your request is taking longer than expected to process. This might be due to the complexity of your request or high system load.",
            "suggestions": ["Try breaking your request into smaller parts", "Simplify your request", "Try again later"]
        }
    
    async def _fallback_connection_error(self, error_context) -> Dict[str, Any]:
        """Fallback for connection errors"""
        return {
            "response": "I'm experiencing connectivity issues. Please check your connection and try again.",
            "suggestions": ["Check your internet connection", "Try again in a moment", "Contact support if the issue persists"]
        }
    
    # Feedback and learning interface
    
    async def record_feedback(
        self,
        session_id: str,
        message_id: str,
        feedback_type: str,
        feedback_value: Union[int, str, bool, Dict],
        comments: str = ""
    ) -> bool:
        """Record user feedback for learning"""
        
        if not self.learning_system or session_id not in self.active_sessions:
            return False
        
        try:
            session = self.active_sessions[session_id]
            
            # Find the relevant conversation turn
            relevant_turns = [turn for turn in session.conversation_history if turn.turn_id == message_id]
            if not relevant_turns:
                # Use the last assistant turn if specific turn not found
                assistant_turns = [turn for turn in session.conversation_history if turn.role == "assistant"]
                if not assistant_turns:
                    return False
                relevant_turn = assistant_turns[-1]
            else:
                relevant_turn = relevant_turns[0]
            
            # Get the user request (previous turn)
            turn_index = session.conversation_history.index(relevant_turn)
            user_request = ""
            if turn_index > 0:
                user_request = session.conversation_history[turn_index - 1].content
            
            # Convert feedback type
            feedback_type_mapping = {
                "thumbs_up": FeedbackType.THUMBS_UP,
                "thumbs_down": FeedbackType.THUMBS_DOWN,
                "rating": FeedbackType.RATING,
                "comment": FeedbackType.COMMENT
            }
            
            fb_type = feedback_type_mapping.get(feedback_type, FeedbackType.COMMENT)
            
            # Determine task type from turn metadata or guess
            task_type = relevant_turn.metadata.get("task_type", "conversation")
            
            # Record feedback
            self.learning_system.record_feedback(
                user_request=user_request,
                ai_response=relevant_turn.content,
                feedback_type=fb_type,
                feedback_value=feedback_value,
                model_used=relevant_turn.metadata.get("model_used", "unknown"),
                task_type=task_type,
                context_info={"comments": comments, "session_id": session_id},
                session_id=session_id,
                user_id=session.user_id
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")
            return False
    
    # Status and monitoring interface
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "initialized": self.is_initialized,
            "active_sessions": len(self.active_sessions),
            "performance_stats": self.performance_stats.copy(),
            "components": {
                "model_manager": self.model_manager is not None,
                "optimizer": self.optimizer is not None,
                "context_manager": self.context_manager is not None,
                "learning_system": self.learning_system is not None,
                "project_intelligence": self.project_intelligence is not None,
                "performance_optimizer": self.performance_optimizer is not None,
                "error_manager": self.error_manager is not None
            }
        }
        
        # Add component-specific status
        if self.model_manager:
            status["model_manager_status"] = self.model_manager.get_system_status()
        
        if self.performance_optimizer:
            asyncio.create_task(self._add_performance_stats(status))
        
        if self.learning_system:
            status["learning_stats"] = self.learning_system.get_learning_report()
        
        return status
    
    async def _add_performance_stats(self, status: Dict[str, Any]):
        """Add performance statistics to status"""
        try:
            perf_report = await self.performance_optimizer.get_performance_report()
            status["performance_report"] = perf_report
        except Exception as e:
            logger.debug(f"Could not get performance report: {e}")
    
    async def cleanup(self):
        """Clean up all resources"""
        logger.info("Cleaning up Enhanced AI Assistant...")
        
        # Cleanup all components
        cleanup_tasks = []
        
        if self.model_manager:
            cleanup_tasks.append(self.model_manager.cleanup())
        
        if self.performance_optimizer:
            cleanup_tasks.append(self.performance_optimizer.cleanup())
        
        if self.error_manager:
            cleanup_tasks.append(self.error_manager.cleanup())
        
        # Run cleanup tasks
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        # Clear sessions
        self.active_sessions.clear()
        
        logger.info("Enhanced AI Assistant cleanup completed")

# Utility functions for easy usage

async def create_enhanced_assistant(project_path: Path = None) -> EnhancedAIAssistant:
    """Create and initialize enhanced AI assistant"""
    assistant = EnhancedAIAssistant(project_path)
    
    if await assistant.initialize():
        logger.info("Enhanced AI Assistant ready for use")
        return assistant
    else:
        raise RuntimeError("Failed to initialize Enhanced AI Assistant")

def create_simple_chat_interface(assistant: EnhancedAIAssistant):
    """Create simple chat interface for testing"""
    
    async def chat_loop():
        print("🚀 ABOV3 Genesis Enhanced AI Assistant")
        print("Type 'quit' to exit, 'status' for system status")
        print("-" * 50)
        
        session_id = None
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'status':
                    status = assistant.get_system_status()
                    print(f"📊 System Status: {json.dumps(status, indent=2)}")
                    continue
                
                if not user_input:
                    continue
                
                # Process chat request
                response = await assistant.chat(
                    message=user_input,
                    session_id=session_id,
                    user_id="test_user"
                )
                
                if response["success"]:
                    print(f"\n🤖 Assistant: {response['response']}")
                    print(f"⚡ Model: {response.get('model_used', 'unknown')} | "
                          f"Time: {response['processing_time_ms']:.0f}ms | "
                          f"Quality: {response.get('quality_score', 0):.2f}")
                    
                    session_id = response["session_id"]
                else:
                    print(f"\n❌ Error: {response['error']}")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
    
    return chat_loop