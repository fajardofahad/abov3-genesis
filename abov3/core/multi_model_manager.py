"""
ABOV3 Genesis - Multi-Model Management System
Advanced system for managing multiple Ollama models with intelligent selection and load balancing
"""

import asyncio
import time
import json
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import logging
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import hashlib

from .ollama_client import OllamaClient
from .ollama_optimization import OllamaModelOptimizer
from .context_manager import SmartContextManager
from .learning_system import AdaptiveLearningSystem, FeedbackType

logger = logging.getLogger(__name__)

class ModelCapability(Enum):
    """Model capabilities"""
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    DEBUGGING = "debugging"
    EXPLANATION = "explanation"
    ARCHITECTURE = "architecture"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    CONVERSATION = "conversation"
    TRANSLATION = "translation"
    OPTIMIZATION = "optimization"

class ModelStatus(Enum):
    """Model status"""
    AVAILABLE = "available"
    BUSY = "busy"
    ERROR = "error"
    LOADING = "loading"
    UNAVAILABLE = "unavailable"

@dataclass
class ModelInfo:
    """Information about a model"""
    name: str
    full_name: str
    size_gb: float
    context_window: int
    capabilities: List[ModelCapability]
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    optimal_temperature: float = 0.7
    optimal_top_p: float = 0.9
    status: ModelStatus = ModelStatus.UNAVAILABLE
    last_used: float = 0.0
    total_uses: int = 0
    success_rate: float = 0.0
    average_response_time: float = 0.0
    memory_usage_mb: int = 0
    specialized_for: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelPerformanceMetrics:
    """Real-time performance metrics for a model"""
    model_name: str
    current_load: float = 0.0  # 0.0 to 1.0
    queue_length: int = 0
    active_requests: int = 0
    avg_response_time_ms: float = 0.0
    success_rate_24h: float = 0.0
    error_count_1h: int = 0
    last_error_time: float = 0.0
    last_success_time: float = 0.0
    total_tokens_processed: int = 0
    health_score: float = 1.0  # 0.0 to 1.0

class MultiModelManager:
    """
    Advanced multi-model management system with intelligent selection,
    load balancing, and performance optimization
    """
    
    def __init__(self, project_path: Optional[Path] = None):
        self.project_path = project_path
        
        # Model registry and clients
        self.available_models: Dict[str, ModelInfo] = {}
        self.model_clients: Dict[str, OllamaClient] = {}
        self.performance_metrics: Dict[str, ModelPerformanceMetrics] = {}
        
        # Request queue and load balancing
        self.request_queues: Dict[str, asyncio.Queue] = {}
        self.request_history: deque = deque(maxlen=10000)  # Keep last 10k requests
        
        # Integration with other systems
        self.optimizer: Optional[OllamaModelOptimizer] = None
        self.context_manager: Optional[SmartContextManager] = None
        self.learning_system: Optional[AdaptiveLearningSystem] = None
        
        # Configuration
        self.max_concurrent_requests_per_model = 3
        self.model_timeout_seconds = 120
        self.health_check_interval = 30
        self.auto_scale_enabled = True
        
        # Monitoring and statistics
        self.system_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_selection_time_ms": 0.0,
            "models_loaded": 0,
            "uptime_start": time.time()
        }
        
        # Background tasks
        self.health_check_task: Optional[asyncio.Task] = None
        self.performance_monitor_task: Optional[asyncio.Task] = None
        self.auto_scale_task: Optional[asyncio.Task] = None
        
        # Initialize model definitions
        self._initialize_model_definitions()
        
        # Start background tasks
        asyncio.create_task(self._start_background_tasks())
    
    async def _start_background_tasks(self):
        """Start background monitoring and management tasks"""
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        self.performance_monitor_task = asyncio.create_task(self._performance_monitor_loop())
        if self.auto_scale_enabled:
            self.auto_scale_task = asyncio.create_task(self._auto_scale_loop())
    
    def _initialize_model_definitions(self):
        """Initialize model definitions with capabilities and characteristics"""
        
        # Code-focused models
        self.available_models["codellama:7b"] = ModelInfo(
            name="codellama:7b",
            full_name="Code Llama 7B",
            size_gb=3.8,
            context_window=16384,
            capabilities=[
                ModelCapability.CODE_GENERATION,
                ModelCapability.CODE_REVIEW,
                ModelCapability.DEBUGGING,
                ModelCapability.EXPLANATION,
                ModelCapability.TESTING
            ],
            strengths=["Python", "JavaScript", "C++", "code completion", "bug fixing"],
            weaknesses=["creative writing", "general conversation", "non-code tasks"],
            optimal_temperature=0.05,
            optimal_top_p=0.95,
            specialized_for=["coding", "programming", "software development"]
        )
        
        self.available_models["codellama:13b"] = ModelInfo(
            name="codellama:13b",
            full_name="Code Llama 13B",
            size_gb=7.4,
            context_window=16384,
            capabilities=[
                ModelCapability.CODE_GENERATION,
                ModelCapability.CODE_REVIEW,
                ModelCapability.DEBUGGING,
                ModelCapability.ARCHITECTURE,
                ModelCapability.OPTIMIZATION,
                ModelCapability.TESTING
            ],
            strengths=["complex code", "system design", "performance optimization", "multiple languages"],
            weaknesses=["speed", "resource usage"],
            optimal_temperature=0.1,
            optimal_top_p=0.95,
            specialized_for=["complex coding", "architecture", "enterprise development"]
        )
        
        self.available_models["deepseek-coder:6.7b"] = ModelInfo(
            name="deepseek-coder:6.7b",
            full_name="DeepSeek Coder 6.7B",
            size_gb=3.7,
            context_window=16384,
            capabilities=[
                ModelCapability.CODE_GENERATION,
                ModelCapability.CODE_REVIEW,
                ModelCapability.DEBUGGING,
                ModelCapability.EXPLANATION,
                ModelCapability.OPTIMIZATION,
                ModelCapability.TESTING
            ],
            strengths=["code quality", "best practices", "modern frameworks", "clean code"],
            weaknesses=["speed for large tasks"],
            optimal_temperature=0.1,
            optimal_top_p=0.98,
            specialized_for=["high-quality code", "modern development", "best practices"]
        )
        
        self.available_models["deepseek-coder:33b"] = ModelInfo(
            name="deepseek-coder:33b",
            full_name="DeepSeek Coder 33B",
            size_gb=18.8,
            context_window=16384,
            capabilities=[
                ModelCapability.CODE_GENERATION,
                ModelCapability.CODE_REVIEW,
                ModelCapability.DEBUGGING,
                ModelCapability.ARCHITECTURE,
                ModelCapability.OPTIMIZATION,
                ModelCapability.TESTING,
                ModelCapability.DOCUMENTATION
            ],
            strengths=["enterprise code", "complex systems", "multiple languages", "architecture"],
            weaknesses=["resource intensive", "slower responses"],
            optimal_temperature=0.1,
            optimal_top_p=0.98,
            specialized_for=["enterprise development", "complex systems", "team projects"]
        )
        
        # General-purpose models
        self.available_models["llama3:8b"] = ModelInfo(
            name="llama3:8b",
            full_name="Llama 3 8B",
            size_gb=4.7,
            context_window=8192,
            capabilities=[
                ModelCapability.CODE_GENERATION,
                ModelCapability.EXPLANATION,
                ModelCapability.CONVERSATION,
                ModelCapability.DEBUGGING,
                ModelCapability.DOCUMENTATION
            ],
            strengths=["balanced performance", "explanations", "reasoning", "conversation"],
            weaknesses=["specialized coding tasks", "very large codebases"],
            optimal_temperature=0.2,
            optimal_top_p=0.9,
            specialized_for=["general purpose", "explanations", "learning"]
        )
        
        self.available_models["llama3:70b"] = ModelInfo(
            name="llama3:70b",
            full_name="Llama 3 70B",
            size_gb=39.5,
            context_window=8192,
            capabilities=[
                ModelCapability.CODE_GENERATION,
                ModelCapability.CODE_REVIEW,
                ModelCapability.DEBUGGING,
                ModelCapability.ARCHITECTURE,
                ModelCapability.EXPLANATION,
                ModelCapability.CONVERSATION,
                ModelCapability.DOCUMENTATION,
                ModelCapability.OPTIMIZATION
            ],
            strengths=["high quality", "complex reasoning", "comprehensive responses", "versatility"],
            weaknesses=["very resource intensive", "slow"],
            optimal_temperature=0.15,
            optimal_top_p=0.9,
            specialized_for=["high-quality work", "complex projects", "comprehensive solutions"]
        )
        
        self.available_models["qwen2:7b"] = ModelInfo(
            name="qwen2:7b",
            full_name="Qwen 2 7B",
            size_gb=4.4,
            context_window=32768,
            capabilities=[
                ModelCapability.CODE_GENERATION,
                ModelCapability.CODE_REVIEW,
                ModelCapability.EXPLANATION,
                ModelCapability.CONVERSATION,
                ModelCapability.TRANSLATION
            ],
            strengths=["large context", "multilingual", "fast responses", "good balance"],
            weaknesses=["specialized domains", "complex architecture"],
            optimal_temperature=0.15,
            optimal_top_p=0.95,
            specialized_for=["large context tasks", "multilingual projects", "balanced work"]
        )
        
        self.available_models["mistral:7b"] = ModelInfo(
            name="mistral:7b",
            full_name="Mistral 7B",
            size_gb=4.1,
            context_window=8192,
            capabilities=[
                ModelCapability.CODE_GENERATION,
                ModelCapability.EXPLANATION,
                ModelCapability.CONVERSATION,
                ModelCapability.DEBUGGING
            ],
            strengths=["fast responses", "efficient", "good reasoning", "balanced"],
            weaknesses=["context window", "specialized coding"],
            optimal_temperature=0.2,
            optimal_top_p=0.9,
            specialized_for=["fast responses", "general coding", "efficient processing"]
        )
        
        # Initialize performance metrics for all models
        for model_name in self.available_models:
            self.performance_metrics[model_name] = ModelPerformanceMetrics(model_name)
    
    async def initialize_systems(
        self,
        optimizer: Optional[OllamaModelOptimizer] = None,
        context_manager: Optional[SmartContextManager] = None,
        learning_system: Optional[AdaptiveLearningSystem] = None
    ):
        """Initialize integration with other systems"""
        self.optimizer = optimizer
        self.context_manager = context_manager
        self.learning_system = learning_system
        
        # Detect available models on the system
        await self._detect_available_models()
        
        logger.info(f"Multi-model manager initialized with {len(self.model_clients)} available models")
    
    async def _detect_available_models(self):
        """Detect which models are actually available on the system"""
        # Create a base client to check available models
        base_client = OllamaClient()
        
        try:
            # Get list of installed models
            installed_models = await base_client.list_models()
            
            # Handle case where list_models returns list directly or dict with 'models' key
            if isinstance(installed_models, list):
                models_list = installed_models
            else:
                models_list = installed_models.get('models', [])
            
            for model_info in models_list:
                model_name = model_info['name']
                
                if model_name in self.available_models:
                    # Create client for this model
                    client = OllamaClient()
                    self.model_clients[model_name] = client
                    self.request_queues[model_name] = asyncio.Queue(maxsize=10)
                    
                    # Update model status
                    self.available_models[model_name].status = ModelStatus.AVAILABLE
                    
                    # Update system stats
                    self.system_stats["models_loaded"] += 1
                    
                    logger.info(f"Model {model_name} is available")
                else:
                    # Unknown model, add basic info
                    self.available_models[model_name] = ModelInfo(
                        name=model_name,
                        full_name=model_name,
                        size_gb=model_info.get('size', 0) / (1024**3),  # Convert to GB
                        context_window=8192,  # Default
                        capabilities=[ModelCapability.CODE_GENERATION],  # Default
                        status=ModelStatus.AVAILABLE
                    )
                    
                    client = OllamaClient()
                    self.model_clients[model_name] = client
                    self.request_queues[model_name] = asyncio.Queue(maxsize=10)
                    self.performance_metrics[model_name] = ModelPerformanceMetrics(model_name)
                    
                    self.system_stats["models_loaded"] += 1
                    logger.info(f"Unknown model {model_name} detected and added")
        
        except Exception as e:
            logger.error(f"Failed to detect available models: {e}")
        finally:
            # Clean up base client
            try:
                await base_client.close()
            except:
                pass
            
            # Fallback: try to use any available model as default
            try:
                fallback_client = OllamaClient()
                if await fallback_client.is_available():
                    await fallback_client.connect()
                    models_list = await fallback_client.list_models()
                    if models_list:
                        # Use first available model as fallback
                        first_model = models_list[0]
                        fallback_name = first_model.get('name', 'unknown')
                        
                        self.model_clients[fallback_name] = fallback_client
                        self.request_queues[fallback_name] = asyncio.Queue(maxsize=10)
                        self.available_models[fallback_name] = ModelInfo(
                            name=fallback_name,
                            full_name=fallback_name,
                            size_gb=first_model.get('size', 0) / (1024**3),
                            context_window=8192,
                            capabilities=[ModelCapability.CODE_GENERATION],
                            status=ModelStatus.AVAILABLE
                        )
                        logger.info(f"Using {fallback_name} as fallback model")
                    else:
                        await fallback_client.close()
                        logger.warning("No models available - system will use fallback responses")
                else:
                    logger.warning("Ollama server not available - system will use fallback responses")
            except Exception as fallback_error:
                logger.error(f"Failed to initialize fallback model: {fallback_error}")
    
    async def select_best_model(
        self,
        task_type: str,
        user_request: str,
        context_info: Dict[str, Any] = None,
        constraints: Dict[str, Any] = None,
        performance_priority: str = "balanced"  # "speed", "quality", "balanced"
    ) -> Tuple[str, float]:
        """
        Select the best model for a given task using advanced intelligent selection algorithm
        Returns: (model_name, confidence_score)
        """
        start_time = time.time()
        context_info = context_info or {}
        constraints = constraints or {}
        
        # Advanced task classification
        task_complexity = self._analyze_task_complexity(user_request, context_info)
        task_domain = self._identify_task_domain(user_request, context_info)
        urgency_level = self._assess_urgency(constraints)
        
        # Enhanced task capability mapping with sub-categories
        task_capability_map = {
            "code_generation": ModelCapability.CODE_GENERATION,
            "code_completion": ModelCapability.CODE_GENERATION,
            "code_review": ModelCapability.CODE_REVIEW,
            "debugging": ModelCapability.DEBUGGING,
            "explanation": ModelCapability.EXPLANATION,
            "architecture": ModelCapability.ARCHITECTURE,
            "testing": ModelCapability.TESTING,
            "documentation": ModelCapability.DOCUMENTATION,
            "conversation": ModelCapability.CONVERSATION,
            "optimization": ModelCapability.OPTIMIZATION,
            "refactoring": ModelCapability.OPTIMIZATION,
            "api_design": ModelCapability.ARCHITECTURE,
            "performance_analysis": ModelCapability.OPTIMIZATION
        }
        
        required_capability = task_capability_map.get(task_type, ModelCapability.CODE_GENERATION)
        
        # Score all available models with advanced algorithms
        model_scores = {}
        for model_name, model_info in self.available_models.items():
            if model_name not in self.model_clients:
                continue  # Model not available
            
            score = await self._calculate_advanced_model_score(
                model_name, model_info, required_capability,
                user_request, context_info, constraints,
                task_complexity, task_domain, urgency_level, performance_priority
            )
            
            if score > 0:
                model_scores[model_name] = score
        
        if not model_scores:
            # No suitable models found, use first available
            fallback_model = next(iter(self.model_clients.keys()))
            logger.warning(f"No optimal model found, using fallback: {fallback_model}")
            return fallback_model, 0.5
        
        # Select best model
        best_model = max(model_scores.items(), key=lambda x: x[1])
        
        # Record selection time
        selection_time_ms = (time.time() - start_time) * 1000
        self.system_stats["avg_selection_time_ms"] = (
            (self.system_stats["avg_selection_time_ms"] * self.system_stats["total_requests"] + selection_time_ms) /
            (self.system_stats["total_requests"] + 1)
        )
        
        logger.debug(f"Selected {best_model[0]} with score {best_model[1]:.3f} for {task_type}")
        return best_model[0], best_model[1]
    
    async def _calculate_advanced_model_score(
        self,
        model_name: str,
        model_info: ModelInfo,
        required_capability: ModelCapability,
        user_request: str,
        context_info: Dict[str, Any],
        constraints: Dict[str, Any],
        task_complexity: str,
        task_domain: str,
        urgency_level: str,
        performance_priority: str
    ) -> float:
        """Calculate advanced suitability score for a model"""
        
        score = 0.0
        
        # Base capability score (higher weight)
        if required_capability in model_info.capabilities:
            score += 40.0  # Increased base score
        else:
            # Check for related capabilities
            related_capabilities = self._get_related_capabilities(required_capability)
            capability_match = any(cap in model_info.capabilities for cap in related_capabilities)
            if capability_match:
                score += 20.0  # Partial credit for related capabilities
            else:
                return 0.0  # Not suitable if no relevant capability
        
        # Enhanced performance metrics score (35% weight)
        metrics = self.performance_metrics.get(model_name)
        if metrics:
            # Health and success rate with improved weighting
            score += metrics.health_score * 18.0
            score += min(metrics.success_rate_24h, 1.0) * 17.0
            
            # Advanced load balancing based on priority
            if performance_priority == "speed":
                load_penalty = metrics.current_load * 15.0  # Heavily penalize load
                response_time_bonus = max(0, 10.0 - (metrics.avg_response_time_ms / 1000))  # Bonus for fast models
                score += response_time_bonus
            elif performance_priority == "quality":
                load_penalty = metrics.current_load * 5.0  # Less concerned with load
                quality_bonus = metrics.health_score * 5.0  # Extra bonus for healthy models
                score += quality_bonus
            else:  # balanced
                load_penalty = metrics.current_load * 10.0
            
            score -= load_penalty
        
        # Learning system score (20% weight)
        if self.learning_system:
            task_type = context_info.get('task_type', 'general')
            model_recs = self.learning_system.get_model_recommendations(task_type, user_request)
            for rec_model, rec_score, _ in model_recs:
                if rec_model == model_name:
                    score += rec_score * 20.0
                    break
        
        # Advanced context suitability (15% weight)
        request_length = len(user_request)
        context_length = sum(len(str(v)) for v in context_info.values())
        total_length = request_length + context_length
        
        # Improved context window scoring
        context_ratio = total_length / model_info.context_window if model_info.context_window > 0 else 0
        
        if context_ratio < 0.3:  # Very comfortable fit
            score += 15.0
        elif context_ratio < 0.6:  # Good fit
            score += 12.0
        elif context_ratio < 0.8:  # Acceptable fit
            score += 8.0
        elif context_ratio < 0.95:  # Tight fit but workable
            score += 3.0
        else:  # Too large
            score -= 25.0
        
        # Bonus for large context models handling complex tasks
        if task_complexity == "high" and model_info.context_window >= 16384:
            score += 8.0
        
        # Enhanced specialization and domain matching (15% weight)
        specialization_bonus = 0.0
        
        if model_info.specialized_for:
            for specialization in model_info.specialized_for:
                # Direct keyword matching
                if any(keyword in user_request.lower() for keyword in specialization.split()):
                    specialization_bonus += 12.0
                
                # Domain-specific bonuses
                if task_domain in specialization.lower():
                    specialization_bonus += 8.0
        
        # Task complexity matching
        if task_complexity == "high":
            # Prefer larger, more capable models for complex tasks
            if "33b" in model_name or "70b" in model_name:
                specialization_bonus += 10.0
            elif "13b" in model_name or "15b" in model_name:
                specialization_bonus += 5.0
        elif task_complexity == "low":
            # Smaller models can handle simple tasks efficiently
            if "7b" in model_name or "6.7b" in model_name:
                specialization_bonus += 5.0
        
        score += min(20.0, specialization_bonus)  # Cap the specialization bonus
        
        # Enhanced constraint handling and penalties
        constraint_penalty = 0.0
        
        if constraints.get('max_response_time_ms'):
            if metrics and metrics.avg_response_time_ms > constraints['max_response_time_ms']:
                constraint_penalty += 20.0
        
        if constraints.get('min_quality_score'):
            if metrics and metrics.success_rate_24h < constraints['min_quality_score']:
                constraint_penalty += 30.0
        
        if constraints.get('preferred_languages'):
            preferred_langs = constraints['preferred_languages']
            if not any(lang.lower() in user_request.lower() for lang in preferred_langs):
                # Check if model excels at any preferred language
                lang_support = any(lang.lower() in ' '.join(model_info.strengths).lower() 
                                 for lang in preferred_langs)
                if lang_support:
                    score += 5.0
        
        score -= constraint_penalty
        
        # Advanced resource availability scoring
        availability_penalty = 0.0
        if model_info.status != ModelStatus.AVAILABLE:
            availability_penalty += 60.0
        elif model_info.status == ModelStatus.BUSY:
            if urgency_level == "high":
                availability_penalty += 30.0  # Heavy penalty for urgent tasks
            else:
                availability_penalty += 15.0  # Moderate penalty for non-urgent
        
        score -= availability_penalty
        
        # Final score adjustments based on priority
        if performance_priority == "quality":
            # Boost score for known high-quality models
            if any(indicator in model_name.lower() for indicator in ["deepseek", "claude", "gpt-4"]):
                score *= 1.1
        elif performance_priority == "speed":
            # Boost score for fast models (typically smaller ones)
            if any(indicator in model_name.lower() for indicator in ["7b", "8b", "phi", "gemma"]):
                score *= 1.1
        
        return max(0.0, min(100.0, score))  # Normalize to 0-100
    
    async def process_request(
        self,
        user_request: str,
        task_type: str = "code_generation",
        context_info: Dict[str, Any] = None,
        model_preference: Optional[str] = None,
        constraints: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process a request using the best available model
        """
        start_time = time.time()
        request_id = hashlib.md5(f"{user_request[:100]}{start_time}".encode()).hexdigest()[:12]
        
        context_info = context_info or {}
        context_info['task_type'] = task_type
        
        try:
            # Select model
            if model_preference and model_preference in self.model_clients:
                selected_model = model_preference
                confidence = 0.8  # User preference
            else:
                selected_model, confidence = await self.select_best_model(
                    task_type, user_request, context_info, constraints
                )
            
            # Update request statistics
            self.system_stats["total_requests"] += 1
            
            # Get model client
            client = self.model_clients[selected_model]
            model_info = self.available_models[selected_model]
            metrics = self.performance_metrics[selected_model]
            
            # Update model status
            metrics.current_load = min(1.0, metrics.current_load + (1.0 / self.max_concurrent_requests_per_model))
            metrics.queue_length += 1
            metrics.active_requests += 1
            
            # Build optimized request
            if self.optimizer:
                optimization_result = self.optimizer.optimize_for_task(
                    task_type, selected_model, user_request, context_info
                )
                
                prompt = optimization_result["prompt"]
                options = optimization_result["options"]
            else:
                # Fallback optimization
                prompt = user_request
                options = {
                    "temperature": model_info.optimal_temperature,
                    "top_p": model_info.optimal_top_p
                }
            
            # Process request with timeout
            try:
                # Collect response from async generator
                response_parts = []
                async def collect_response():
                    async for chunk in client.generate(selected_model, prompt, options=options):
                        if 'response' in chunk:
                            response_parts.append(chunk['response'])
                        if chunk.get('done', False):
                            break
                    return ''.join(response_parts)
                
                response = await asyncio.wait_for(
                    collect_response(),
                    timeout=self.model_timeout_seconds
                )
                
                # Record success
                processing_time = time.time() - start_time
                await self._record_success(selected_model, processing_time, len(response))
                
                # Update learning system
                if self.learning_system:
                    self.learning_system.record_implicit_feedback(
                        user_request=user_request,
                        ai_response=response,
                        model_used=selected_model,
                        task_type=task_type,
                        usage_metrics={
                            'response_time_ms': processing_time * 1000,
                            'request_id': request_id,
                            'model_confidence': confidence
                        }
                    )
                
                result = {
                    'success': True,
                    'response': response,
                    'model_used': selected_model,
                    'model_confidence': confidence,
                    'processing_time_ms': processing_time * 1000,
                    'request_id': request_id,
                    'optimization_applied': self.optimizer is not None
                }
                
                self.system_stats["successful_requests"] += 1
                return result
                
            except asyncio.TimeoutError:
                await self._record_error(selected_model, "timeout")
                raise TimeoutError(f"Request timed out after {self.model_timeout_seconds}s")
            
            except Exception as e:
                await self._record_error(selected_model, str(e))
                raise
        
        except Exception as e:
            self.system_stats["failed_requests"] += 1
            logger.error(f"Request {request_id} failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'request_id': request_id,
                'processing_time_ms': (time.time() - start_time) * 1000
            }
        
        finally:
            # Update model status
            if 'selected_model' in locals():
                metrics = self.performance_metrics[selected_model]
                metrics.current_load = max(0.0, metrics.current_load - (1.0 / self.max_concurrent_requests_per_model))
                metrics.queue_length = max(0, metrics.queue_length - 1)
                metrics.active_requests = max(0, metrics.active_requests - 1)
    
    async def _record_success(self, model_name: str, processing_time: float, response_length: int):
        """Record successful request"""
        metrics = self.performance_metrics[model_name]
        model_info = self.available_models[model_name]
        
        # Update response time (exponential moving average)
        new_time_ms = processing_time * 1000
        if metrics.avg_response_time_ms == 0:
            metrics.avg_response_time_ms = new_time_ms
        else:
            alpha = 0.1  # Learning rate
            metrics.avg_response_time_ms = (1 - alpha) * metrics.avg_response_time_ms + alpha * new_time_ms
        
        # Update success tracking
        metrics.last_success_time = time.time()
        metrics.total_tokens_processed += response_length // 4  # Rough token estimate
        
        # Update model usage
        model_info.last_used = time.time()
        model_info.total_uses += 1
        model_info.average_response_time = (
            (model_info.average_response_time * (model_info.total_uses - 1) + processing_time) /
            model_info.total_uses
        )
        
        # Update health score
        metrics.health_score = min(1.0, metrics.health_score + 0.01)
    
    async def _record_error(self, model_name: str, error_message: str):
        """Record failed request"""
        metrics = self.performance_metrics[model_name]
        
        # Update error tracking
        metrics.error_count_1h += 1
        metrics.last_error_time = time.time()
        
        # Decrease health score
        metrics.health_score = max(0.0, metrics.health_score - 0.05)
        
        logger.warning(f"Model {model_name} error: {error_message}")
    
    async def _health_check_loop(self):
        """Background health check for all models"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def _perform_health_checks(self):
        """Perform health checks on all models"""
        current_time = time.time()
        
        for model_name, client in self.model_clients.items():
            try:
                # Simple health check
                start_time = time.time()
                await asyncio.wait_for(
                    client.generate(model_name, "Hello", options={"temperature": 0.1}),
                    timeout=10.0
                )
                
                response_time = time.time() - start_time
                
                # Update status
                self.available_models[model_name].status = ModelStatus.AVAILABLE
                
                # Update health metrics if response was fast
                if response_time < 5.0:
                    metrics = self.performance_metrics[model_name]
                    metrics.health_score = min(1.0, metrics.health_score + 0.02)
                
            except Exception as e:
                logger.warning(f"Health check failed for {model_name}: {e}")
                self.available_models[model_name].status = ModelStatus.ERROR
                
                # Decrease health score
                metrics = self.performance_metrics[model_name]
                metrics.health_score = max(0.0, metrics.health_score - 0.1)
        
        # Clean up old error counts
        for metrics in self.performance_metrics.values():
            if current_time - metrics.last_error_time > 3600:  # 1 hour
                metrics.error_count_1h = max(0, metrics.error_count_1h - 1)
    
    async def _performance_monitor_loop(self):
        """Background performance monitoring"""
        while True:
            try:
                await asyncio.sleep(60)  # Every minute
                await self._update_performance_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
    
    async def _update_performance_metrics(self):
        """Update performance metrics for all models"""
        current_time = time.time()
        
        for model_name in self.model_clients:
            metrics = self.performance_metrics[model_name]
            
            # Calculate 24h success rate from request history
            recent_requests = [
                req for req in self.request_history
                if req.get('model') == model_name and current_time - req.get('timestamp', 0) < 86400
            ]
            
            if recent_requests:
                successful = sum(1 for req in recent_requests if req.get('success', False))
                metrics.success_rate_24h = successful / len(recent_requests)
            else:
                metrics.success_rate_24h = 1.0  # No recent data, assume good
            
            # Update overall model success rate
            model_info = self.available_models[model_name]
            if model_info.total_uses > 0:
                model_info.success_rate = metrics.success_rate_24h
    
    async def _auto_scale_loop(self):
        """Background auto-scaling based on load"""
        while True:
            try:
                await asyncio.sleep(120)  # Every 2 minutes
                await self._check_auto_scale()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-scaling error: {e}")
    
    async def _check_auto_scale(self):
        """Check if auto-scaling is needed"""
        # Check overall system load
        total_active_requests = sum(
            metrics.active_requests for metrics in self.performance_metrics.values()
        )
        
        total_queue_length = sum(
            metrics.queue_length for metrics in self.performance_metrics.values()
        )
        
        # Simple auto-scaling logic
        if total_queue_length > len(self.model_clients) * 5:
            logger.info("High queue length detected, consider adding more model instances")
        
        if total_active_requests > len(self.model_clients) * self.max_concurrent_requests_per_model * 0.8:
            logger.info("High system load detected, consider optimizing request distribution")
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a model"""
        if model_name not in self.available_models:
            return None
        
        model_info = self.available_models[model_name]
        metrics = self.performance_metrics.get(model_name)
        
        return {
            'name': model_info.name,
            'full_name': model_info.full_name,
            'size_gb': model_info.size_gb,
            'context_window': model_info.context_window,
            'capabilities': [cap.value for cap in model_info.capabilities],
            'strengths': model_info.strengths,
            'weaknesses': model_info.weaknesses,
            'status': model_info.status.value,
            'last_used': model_info.last_used,
            'total_uses': model_info.total_uses,
            'success_rate': model_info.success_rate,
            'average_response_time': model_info.average_response_time,
            'specialized_for': model_info.specialized_for,
            'current_metrics': {
                'current_load': metrics.current_load if metrics else 0,
                'queue_length': metrics.queue_length if metrics else 0,
                'active_requests': metrics.active_requests if metrics else 0,
                'avg_response_time_ms': metrics.avg_response_time_ms if metrics else 0,
                'success_rate_24h': metrics.success_rate_24h if metrics else 0,
                'health_score': metrics.health_score if metrics else 1.0
            }
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status and statistics"""
        current_time = time.time()
        uptime_hours = (current_time - self.system_stats["uptime_start"]) / 3600
        
        # Calculate success rate
        total_requests = self.system_stats["total_requests"]
        success_rate = (self.system_stats["successful_requests"] / max(1, total_requests))
        
        # Get model statuses
        model_statuses = {}
        for model_name, model_info in self.available_models.items():
            if model_name in self.model_clients:
                model_statuses[model_name] = model_info.status.value
        
        # System load
        total_active_requests = sum(
            metrics.active_requests for metrics in self.performance_metrics.values()
        )
        
        average_health = sum(
            metrics.health_score for metrics in self.performance_metrics.values()
        ) / max(1, len(self.performance_metrics))
        
        return {
            'uptime_hours': uptime_hours,
            'models_loaded': self.system_stats["models_loaded"],
            'models_available': len([m for m in model_statuses.values() if m == "available"]),
            'total_requests': total_requests,
            'success_rate': success_rate,
            'avg_selection_time_ms': self.system_stats["avg_selection_time_ms"],
            'active_requests': total_active_requests,
            'average_system_health': average_health,
            'model_statuses': model_statuses,
            'integration_status': {
                'optimizer': self.optimizer is not None,
                'context_manager': self.context_manager is not None,
                'learning_system': self.learning_system is not None
            }
        }
    
    async def get_model_recommendations(self, task_type: str, context: str = "") -> List[Dict[str, Any]]:
        """Get model recommendations for a specific task"""
        recommendations = []
        
        # Use learning system if available
        if self.learning_system:
            learned_recs = self.learning_system.get_model_recommendations(task_type, context)
            for model_name, score, reason in learned_recs[:3]:  # Top 3 from learning
                if model_name in self.model_clients:
                    model_info = self.get_model_info(model_name)
                    recommendations.append({
                        'model_name': model_name,
                        'confidence': score,
                        'reason': f"Learning system: {reason}",
                        'model_info': model_info
                    })
        
        # Add rule-based recommendations
        task_capability_map = {
            "code_generation": ModelCapability.CODE_GENERATION,
            "code_review": ModelCapability.CODE_REVIEW,
            "debugging": ModelCapability.DEBUGGING,
            "explanation": ModelCapability.EXPLANATION,
            "architecture": ModelCapability.ARCHITECTURE,
        }
        
        required_capability = task_capability_map.get(task_type, ModelCapability.CODE_GENERATION)
        
        for model_name, model_info in self.available_models.items():
            if (model_name in self.model_clients and 
                required_capability in model_info.capabilities and
                not any(rec['model_name'] == model_name for rec in recommendations)):
                
                # Calculate rule-based score
                score = 0.5  # Base score
                if task_type.lower() in [s.lower() for s in model_info.specialized_for]:
                    score += 0.3
                
                metrics = self.performance_metrics.get(model_name)
                if metrics:
                    score += metrics.health_score * 0.2
                
                reason = f"Capable of {task_type}, specialized for {', '.join(model_info.specialized_for)}"
                
                recommendations.append({
                    'model_name': model_name,
                    'confidence': score,
                    'reason': reason,
                    'model_info': self.get_model_info(model_name)
                })
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        return recommendations[:8]  # More recommendations for better selection
    
    async def cleanup(self):
        """Clean up resources"""
        # Cancel background tasks
        for task in [self.health_check_task, self.performance_monitor_task, self.auto_scale_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close all model clients
        for client in self.model_clients.values():
            if hasattr(client, 'cleanup'):
                try:
                    await client.cleanup()
                except:
                    pass
            elif hasattr(client, 'close'):
                try:
                    await client.close()
                except:
                    pass
        
        logger.info("Multi-model manager cleanup completed")
    
    def _analyze_task_complexity(self, user_request: str, context_info: Dict[str, Any]) -> str:
        """Analyze the complexity level of the task"""
        request_lower = user_request.lower()
        
        # High complexity indicators
        high_complexity_indicators = [
            'distributed', 'microservice', 'scalable', 'enterprise', 'architecture',
            'multi-threaded', 'concurrent', 'async', 'performance optimization',
            'machine learning', 'ai model', 'algorithm design', 'system design',
            'database design', 'security audit', 'cryptography', 'blockchain',
            'large scale', 'production deployment', 'load balancing'
        ]
        
        # Medium complexity indicators  
        medium_complexity_indicators = [
            'api design', 'framework', 'library integration', 'testing strategy',
            'error handling', 'logging', 'monitoring', 'configuration',
            'deployment', 'docker', 'kubernetes', 'ci/cd', 'automation',
            'refactoring', 'optimization', 'performance', 'debugging complex'
        ]
        
        # Check request content
        high_count = sum(1 for indicator in high_complexity_indicators if indicator in request_lower)
        medium_count = sum(1 for indicator in medium_complexity_indicators if indicator in request_lower)
        
        # Consider context factors
        context_complexity = 0
        if context_info:
            code_blocks = len(context_info.get('code_blocks', []))
            if code_blocks > 5:
                context_complexity += 1
            
            project_size = context_info.get('project_size', 'small')
            if project_size in ['large', 'enterprise']:
                context_complexity += 1
            
            if len(user_request) > 500:  # Long, detailed requests
                context_complexity += 1
        
        # Determine complexity
        if high_count >= 2 or (high_count >= 1 and context_complexity >= 2):
            return "high"
        elif medium_count >= 2 or high_count >= 1 or context_complexity >= 1:
            return "medium"
        else:
            return "low"
    
    def _identify_task_domain(self, user_request: str, context_info: Dict[str, Any]) -> str:
        """Identify the primary domain of the task"""
        request_lower = user_request.lower()
        
        domain_indicators = {
            'web_development': ['web', 'html', 'css', 'javascript', 'react', 'vue', 'angular', 'frontend', 'backend', 'api', 'rest', 'graphql'],
            'mobile_development': ['mobile', 'ios', 'android', 'swift', 'kotlin', 'react native', 'flutter', 'app store'],
            'data_science': ['data', 'pandas', 'numpy', 'analysis', 'visualization', 'machine learning', 'ml', 'ai', 'statistics', 'dataset'],
            'devops': ['docker', 'kubernetes', 'ci/cd', 'deployment', 'infrastructure', 'aws', 'azure', 'gcp', 'terraform', 'ansible'],
            'systems_programming': ['c++', 'rust', 'go', 'system', 'performance', 'memory', 'low level', 'embedded', 'kernel'],
            'database': ['sql', 'database', 'postgresql', 'mysql', 'mongodb', 'redis', 'query', 'schema', 'migration'],
            'security': ['security', 'authentication', 'encryption', 'vulnerability', 'penetration', 'audit', 'cryptography'],
            'game_development': ['game', 'unity', 'unreal', 'graphics', 'shader', 'rendering', '3d', 'physics'],
            'enterprise': ['enterprise', 'business', 'erp', 'crm', 'workflow', 'integration', 'scalability'],
            'general_programming': []  # Default fallback
        }
        
        domain_scores = {}
        for domain, indicators in domain_indicators.items():
            if domain == 'general_programming':
                continue
            score = sum(1 for indicator in indicators if indicator in request_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        else:
            return 'general_programming'
    
    def _assess_urgency(self, constraints: Dict[str, Any]) -> str:
        """Assess the urgency level of the request"""
        if not constraints:
            return "normal"
        
        # Check for urgency indicators
        max_response_time = constraints.get('max_response_time_ms', 0)
        if max_response_time > 0:
            if max_response_time < 5000:  # Less than 5 seconds
                return "high"
            elif max_response_time < 15000:  # Less than 15 seconds
                return "medium"
        
        # Check for explicit urgency
        priority = constraints.get('priority', 'normal').lower()
        if priority in ['urgent', 'high', 'critical']:
            return "high"
        elif priority in ['medium', 'elevated']:
            return "medium"
        
        return "normal"
    
    def _get_related_capabilities(self, capability: ModelCapability) -> List[ModelCapability]:
        """Get capabilities related to the required one"""
        related_map = {
            ModelCapability.CODE_GENERATION: [ModelCapability.ARCHITECTURE, ModelCapability.TESTING],
            ModelCapability.DEBUGGING: [ModelCapability.CODE_REVIEW, ModelCapability.TESTING, ModelCapability.OPTIMIZATION],
            ModelCapability.CODE_REVIEW: [ModelCapability.DEBUGGING, ModelCapability.OPTIMIZATION],
            ModelCapability.ARCHITECTURE: [ModelCapability.CODE_GENERATION, ModelCapability.OPTIMIZATION],
            ModelCapability.TESTING: [ModelCapability.CODE_GENERATION, ModelCapability.DEBUGGING],
            ModelCapability.OPTIMIZATION: [ModelCapability.CODE_REVIEW, ModelCapability.DEBUGGING],
            ModelCapability.DOCUMENTATION: [ModelCapability.EXPLANATION, ModelCapability.CODE_REVIEW],
            ModelCapability.EXPLANATION: [ModelCapability.DOCUMENTATION, ModelCapability.CONVERSATION],
            ModelCapability.CONVERSATION: [ModelCapability.EXPLANATION],
            ModelCapability.TRANSLATION: []
        }
        
        return related_map.get(capability, [])

# Utility functions

async def create_multi_model_manager(project_path: Path = None) -> MultiModelManager:
    """Create and initialize multi-model manager"""
    manager = MultiModelManager(project_path)
    
    # Initialize with integrated systems
    from .ollama_optimization import optimize_ollama_for_genesis
    from .context_manager import SmartContextManager
    from .learning_system import create_learning_system
    
    optimizer = await optimize_ollama_for_genesis(project_path)
    context_manager = SmartContextManager(max_context_tokens=16384)
    learning_system = create_learning_system(project_path)
    
    await manager.initialize_systems(optimizer, context_manager, learning_system)
    
    return manager

def select_models_for_task(task_type: str, available_models: List[str]) -> List[str]:
    """Simple utility to select suitable models for a task type"""
    task_preferences = {
        "code_generation": ["codellama", "deepseek-coder", "llama3"],
        "debugging": ["codellama", "deepseek-coder", "llama3"],
        "explanation": ["llama3", "qwen", "mistral"],
        "architecture": ["deepseek-coder", "llama3"],
        "conversation": ["llama3", "mistral", "qwen"]
    }
    
    preferred = task_preferences.get(task_type, ["llama3", "codellama"])
    
    suitable_models = []
    for preference in preferred:
        for model in available_models:
            if preference in model.lower():
                suitable_models.append(model)
    
    return suitable_models if suitable_models else available_models