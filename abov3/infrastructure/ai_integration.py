"""
ABOV3 Genesis - AI Integration Infrastructure
Enterprise-grade AI model integration with Ollama, fallback systems, and health monitoring
"""

import asyncio
import aiohttp
import time
import json
import hashlib
from typing import Dict, Any, Optional, List, Union, AsyncGenerator, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
import random
from collections import deque, defaultdict
import threading

from ..core.ollama_client import OllamaClient
from .performance import CacheManager, PerformanceOptimizer
from .resilience import CircuitBreaker, CircuitBreakerConfig, ErrorRecoveryManager

logger = logging.getLogger(__name__)

class ModelProvider(Enum):
    """AI model providers"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic" 
    HUGGINGFACE = "huggingface"
    LOCAL_INFERENCE = "local_inference"
    MOCK = "mock"  # For testing/fallback

class ModelCapability(Enum):
    """Model capabilities"""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    CONVERSATION = "conversation"
    FUNCTION_CALLING = "function_calling"
    EMBEDDING = "embedding"
    IMAGE_ANALYSIS = "image_analysis"

class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class ModelConfig:
    """Configuration for AI model"""
    model_id: str
    provider: ModelProvider
    endpoint_url: str
    capabilities: List[ModelCapability]
    priority: int = 1  # Lower is higher priority
    timeout: float = 30.0
    max_tokens: int = 4096
    temperature: float = 0.7
    context_window: int = 8192
    cost_per_token: float = 0.0  # For cost tracking
    rate_limit_rpm: int = 60  # Requests per minute
    weight: float = 1.0  # For load balancing
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelHealth:
    """Model health tracking"""
    model_id: str
    provider: ModelProvider
    status: HealthStatus = HealthStatus.UNKNOWN
    last_check: float = field(default_factory=time.time)
    response_time: float = 0.0
    success_rate: float = 0.0
    error_count: int = 0
    consecutive_errors: int = 0
    uptime_percentage: float = 0.0
    total_requests: int = 0
    failed_requests: int = 0
    last_error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InferenceRequest:
    """AI inference request"""
    request_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:12])
    prompt: str = ""
    model_preference: Optional[str] = None
    capabilities_required: List[ModelCapability] = field(default_factory=list)
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    timeout: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    created_at: float = field(default_factory=time.time)

@dataclass 
class InferenceResponse:
    """AI inference response"""
    request_id: str
    model_id: str
    provider: ModelProvider
    response_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    usage: Dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    cached: bool = False
    error: Optional[str] = None
    fallback_used: bool = False

class AIHealthMonitor:
    """
    Comprehensive health monitoring for AI models and providers
    """

    def __init__(self, check_interval: float = 60.0):
        self.check_interval = check_interval
        self._model_health: Dict[str, ModelHealth] = {}
        self._health_history = deque(maxlen=1440)  # 24 hours of data (1min intervals)
        self._monitor_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
        # Health check methods
        self._health_checks: Dict[ModelProvider, Callable] = {
            ModelProvider.OLLAMA: self._check_ollama_health,
            ModelProvider.MOCK: self._check_mock_health
        }

    async def start_monitoring(self, models: List[ModelConfig]):
        """Start health monitoring for models"""
        # Initialize health tracking
        async with self._lock:
            for model in models:
                self._model_health[model.model_id] = ModelHealth(
                    model_id=model.model_id,
                    provider=model.provider
                )
        
        # Start monitoring task
        if not self._monitor_task or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(self._health_monitor_loop())

    async def _health_monitor_loop(self):
        """Main health monitoring loop"""
        while True:
            try:
                await asyncio.sleep(self.check_interval)
                await self._perform_health_checks()
                await self._analyze_health_trends()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")

    async def _perform_health_checks(self):
        """Perform health checks on all models"""
        check_tasks = []
        
        async with self._lock:
            for model_id, health in self._model_health.items():
                check_task = asyncio.create_task(
                    self._check_model_health(model_id, health)
                )
                check_tasks.append(check_task)
        
        if check_tasks:
            await asyncio.gather(*check_tasks, return_exceptions=True)

    async def _check_model_health(self, model_id: str, health: ModelHealth):
        """Check health of a specific model"""
        start_time = time.time()
        
        try:
            # Get appropriate health check function
            health_check_func = self._health_checks.get(health.provider)
            
            if health_check_func:
                is_healthy, error_msg = await health_check_func(model_id)
                
                # Update health status
                response_time = (time.time() - start_time) * 1000  # ms
                
                async with self._lock:
                    health.last_check = time.time()
                    health.response_time = response_time
                    
                    if is_healthy:
                        health.status = HealthStatus.HEALTHY
                        health.consecutive_errors = 0
                        health.last_error = None
                    else:
                        health.consecutive_errors += 1
                        health.error_count += 1
                        health.last_error = error_msg
                        
                        # Determine status based on error count
                        if health.consecutive_errors >= 5:
                            health.status = HealthStatus.UNHEALTHY
                        elif health.consecutive_errors >= 2:
                            health.status = HealthStatus.DEGRADED
                        else:
                            health.status = HealthStatus.HEALTHY
                    
                    # Update success rate
                    health.total_requests += 1
                    if not is_healthy:
                        health.failed_requests += 1
                    
                    health.success_rate = 1.0 - (health.failed_requests / max(1, health.total_requests))
                    
        except Exception as e:
            logger.error(f"Health check failed for {model_id}: {e}")
            
            async with self._lock:
                health.status = HealthStatus.UNKNOWN
                health.last_error = str(e)
                health.consecutive_errors += 1

    async def _check_ollama_health(self, model_id: str) -> tuple[bool, Optional[str]]:
        """Health check for Ollama models"""
        try:
            async with OllamaClient() as client:
                # Test basic connectivity
                if not await client.is_available():
                    return False, "Ollama service not available"
                
                # Test model availability
                models = await client.list_models()
                available_models = [m.get('name', '') for m in models]
                
                if model_id not in available_models:
                    return False, f"Model {model_id} not available in Ollama"
                
                # Test inference with simple prompt
                test_prompt = "Hello, respond with 'OK' if you're working properly."
                
                async for response_chunk in client.generate(
                    model=model_id,
                    prompt=test_prompt,
                    options={'max_tokens': 10}
                ):
                    if response_chunk.get('done'):
                        response_text = response_chunk.get('response', '').strip()
                        if response_text:
                            return True, None
                        else:
                            return False, "Empty response from model"
                
                return False, "No response received"
                
        except Exception as e:
            return False, str(e)

    async def _check_mock_health(self, model_id: str) -> tuple[bool, Optional[str]]:
        """Health check for mock models (always healthy)"""
        return True, None

    async def _analyze_health_trends(self):
        """Analyze health trends and detect patterns"""
        current_time = time.time()
        
        # Create health snapshot
        snapshot = {
            'timestamp': current_time,
            'models': {}
        }
        
        async with self._lock:
            for model_id, health in self._model_health.items():
                snapshot['models'][model_id] = {
                    'status': health.status.value,
                    'response_time': health.response_time,
                    'success_rate': health.success_rate,
                    'consecutive_errors': health.consecutive_errors
                }
        
        self._health_history.append(snapshot)
        
        # Analyze trends (could implement trend detection, alerts, etc.)
        await self._detect_health_anomalies()

    async def _detect_health_anomalies(self):
        """Detect health anomalies and patterns"""
        if len(self._health_history) < 10:
            return
        
        # Check for degrading performance trends
        recent_snapshots = list(self._health_history)[-10:]
        
        for model_id in self._model_health.keys():
            response_times = []
            success_rates = []
            
            for snapshot in recent_snapshots:
                model_data = snapshot.get('models', {}).get(model_id, {})
                if model_data:
                    response_times.append(model_data.get('response_time', 0))
                    success_rates.append(model_data.get('success_rate', 1.0))
            
            # Check for increasing response times
            if len(response_times) >= 5:
                recent_avg = sum(response_times[-5:]) / 5
                older_avg = sum(response_times[:5]) / 5
                
                if recent_avg > older_avg * 1.5:  # 50% increase
                    logger.warning(f"Model {model_id} showing degraded response times: {recent_avg:.1f}ms vs {older_avg:.1f}ms")
            
            # Check for declining success rates
            if len(success_rates) >= 5:
                recent_avg = sum(success_rates[-5:]) / 5
                older_avg = sum(success_rates[:5]) / 5
                
                if recent_avg < older_avg * 0.9:  # 10% decrease
                    logger.warning(f"Model {model_id} showing declining success rate: {recent_avg:.1%} vs {older_avg:.1%}")

    async def get_model_health(self, model_id: str) -> Optional[ModelHealth]:
        """Get health status for a specific model"""
        async with self._lock:
            return self._model_health.get(model_id)

    async def get_healthy_models(self) -> List[str]:
        """Get list of healthy model IDs"""
        healthy = []
        
        async with self._lock:
            for model_id, health in self._model_health.items():
                if health.status == HealthStatus.HEALTHY:
                    healthy.append(model_id)
        
        return healthy

    async def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report"""
        async with self._lock:
            total_models = len(self._model_health)
            healthy_count = sum(1 for h in self._model_health.values() if h.status == HealthStatus.HEALTHY)
            degraded_count = sum(1 for h in self._model_health.values() if h.status == HealthStatus.DEGRADED)
            unhealthy_count = sum(1 for h in self._model_health.values() if h.status == HealthStatus.UNHEALTHY)
            
            model_details = {}
            for model_id, health in self._model_health.items():
                model_details[model_id] = {
                    'status': health.status.value,
                    'response_time_ms': health.response_time,
                    'success_rate': health.success_rate,
                    'uptime_percentage': health.uptime_percentage,
                    'total_requests': health.total_requests,
                    'failed_requests': health.failed_requests,
                    'consecutive_errors': health.consecutive_errors,
                    'last_check': health.last_check,
                    'last_error': health.last_error
                }
            
            overall_health = "healthy" if unhealthy_count == 0 and degraded_count == 0 else "degraded" if unhealthy_count == 0 else "unhealthy"
            
            return {
                'overall_health': overall_health,
                'total_models': total_models,
                'healthy_models': healthy_count,
                'degraded_models': degraded_count,
                'unhealthy_models': unhealthy_count,
                'model_details': model_details,
                'last_check': max((h.last_check for h in self._model_health.values()), default=0)
            }

    async def stop_monitoring(self):
        """Stop health monitoring"""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

class ModelFallbackChain:
    """
    Intelligent model fallback chain with automatic failover
    """

    def __init__(self, models: List[ModelConfig], health_monitor: AIHealthMonitor):
        self.models = {m.model_id: m for m in models}
        self.health_monitor = health_monitor
        
        # Sort models by priority
        self.priority_order = sorted(models, key=lambda m: m.priority)
        
        # Fallback strategies
        self._fallback_history = deque(maxlen=1000)

    async def get_best_model(
        self,
        capabilities: List[ModelCapability] = None,
        exclude_models: List[str] = None
    ) -> Optional[ModelConfig]:
        """Get the best available model for the request"""
        
        capabilities = capabilities or []
        exclude_models = exclude_models or []
        
        # Get healthy models
        healthy_models = await self.health_monitor.get_healthy_models()
        
        # Filter candidates
        candidates = []
        for model in self.priority_order:
            # Skip excluded models
            if model.model_id in exclude_models:
                continue
            
            # Skip unhealthy models
            if model.model_id not in healthy_models:
                continue
            
            # Check capabilities
            if capabilities:
                model_capabilities = set(model.capabilities)
                required_capabilities = set(capabilities)
                
                if not required_capabilities.issubset(model_capabilities):
                    continue
            
            candidates.append(model)
        
        if candidates:
            return candidates[0]  # Highest priority (lowest priority value)
        
        # If no healthy models, try degraded models
        all_models = self.health_monitor._model_health.keys()
        degraded_candidates = []
        
        for model in self.priority_order:
            if model.model_id in exclude_models:
                continue
            
            health = await self.health_monitor.get_model_health(model.model_id)
            if health and health.status == HealthStatus.DEGRADED:
                # Check capabilities
                if capabilities:
                    model_capabilities = set(model.capabilities)
                    required_capabilities = set(capabilities)
                    
                    if not required_capabilities.issubset(model_capabilities):
                        continue
                
                degraded_candidates.append(model)
        
        if degraded_candidates:
            logger.warning(f"Using degraded model: {degraded_candidates[0].model_id}")
            return degraded_candidates[0]
        
        # Last resort: return any model that matches capabilities
        for model in self.priority_order:
            if model.model_id in exclude_models:
                continue
                
            if capabilities:
                model_capabilities = set(model.capabilities)
                required_capabilities = set(capabilities)
                
                if not required_capabilities.issubset(model_capabilities):
                    continue
            
            logger.error(f"Using potentially unhealthy model as last resort: {model.model_id}")
            return model
        
        return None

    async def execute_with_fallback(
        self,
        request: InferenceRequest,
        execution_func: Callable[[ModelConfig, InferenceRequest], Any]
    ) -> InferenceResponse:
        """Execute request with automatic fallback on failure"""
        
        attempted_models = []
        last_error = None
        
        for attempt in range(len(self.priority_order)):
            # Get best available model
            best_model = await self.get_best_model(
                request.capabilities_required,
                attempted_models
            )
            
            if not best_model:
                break
            
            attempted_models.append(best_model.model_id)
            
            try:
                # Execute with selected model
                response = await execution_func(best_model, request)
                
                # Record successful fallback if not first choice
                if attempt > 0:
                    self._fallback_history.append({
                        'timestamp': time.time(),
                        'request_id': request.request_id,
                        'original_model': self.priority_order[0].model_id if self.priority_order else None,
                        'fallback_model': best_model.model_id,
                        'attempt_number': attempt + 1,
                        'success': True
                    })
                    
                    response.fallback_used = True
                
                return response
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Model {best_model.model_id} failed (attempt {attempt + 1}): {e}")
                
                # Record failed attempt
                self._fallback_history.append({
                    'timestamp': time.time(),
                    'request_id': request.request_id,
                    'model': best_model.model_id,
                    'attempt_number': attempt + 1,
                    'success': False,
                    'error': str(e)
                })
                
                # Continue to next model
                continue
        
        # All models failed
        error_response = InferenceResponse(
            request_id=request.request_id,
            model_id="none",
            provider=ModelProvider.MOCK,
            error=f"All models failed. Last error: {last_error}",
            fallback_used=True
        )
        
        return error_response

    async def get_fallback_stats(self) -> Dict[str, Any]:
        """Get fallback statistics"""
        if not self._fallback_history:
            return {'no_data': True}
        
        total_fallbacks = len(self._fallback_history)
        successful_fallbacks = sum(1 for f in self._fallback_history if f.get('success', False))
        
        # Model failure counts
        model_failures = defaultdict(int)
        for fallback in self._fallback_history:
            if not fallback.get('success', False):
                model_failures[fallback.get('model', 'unknown')] += 1
        
        return {
            'total_fallback_attempts': total_fallbacks,
            'successful_fallbacks': successful_fallbacks,
            'fallback_success_rate': successful_fallbacks / max(1, total_fallbacks),
            'model_failure_counts': dict(model_failures),
            'recent_fallbacks': list(self._fallback_history)[-10:]  # Last 10 fallbacks
        }

class EnhancedOllamaIntegration:
    """
    Production-grade Ollama integration with enterprise features
    """

    def __init__(
        self,
        project_path: Optional[Path] = None,
        performance_optimizer: Optional[PerformanceOptimizer] = None,
        error_manager: Optional[ErrorRecoveryManager] = None
    ):
        self.project_path = project_path
        self.performance_optimizer = performance_optimizer
        self.error_manager = error_manager
        
        # Model configuration
        self._models: List[ModelConfig] = []
        self._model_clients: Dict[str, OllamaClient] = {}
        
        # Health monitoring
        self.health_monitor = AIHealthMonitor(check_interval=60.0)
        
        # Fallback chain
        self.fallback_chain = None
        
        # Circuit breakers per model
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Request tracking
        self._request_history = deque(maxlen=10000)
        self._active_requests: Dict[str, InferenceRequest] = {}
        
        # Statistics
        self._stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cached_responses': 0,
            'fallback_responses': 0,
            'average_latency': 0.0
        }
        
        self._lock = asyncio.Lock()

    async def initialize(self):
        """Initialize the enhanced Ollama integration"""
        # Discover available Ollama models
        await self._discover_ollama_models()
        
        # Initialize health monitoring
        if self._models:
            await self.health_monitor.start_monitoring(self._models)
            
            # Create fallback chain
            self.fallback_chain = ModelFallbackChain(self._models, self.health_monitor)
        
        # Initialize circuit breakers
        for model in self._models:
            config = CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=60.0,
                timeout=30.0
            )
            self._circuit_breakers[model.model_id] = CircuitBreaker(
                f"model_{model.model_id}",
                config
            )
        
        logger.info(f"Enhanced Ollama integration initialized with {len(self._models)} models")

    async def _discover_ollama_models(self):
        """Discover available Ollama models"""
        try:
            async with OllamaClient() as client:
                if not await client.is_available():
                    logger.warning("Ollama not available, creating mock models for fallback")
                    self._create_mock_models()
                    return
                
                ollama_models = await client.list_models()
                
                for model_data in ollama_models:
                    model_name = model_data.get('name', '')
                    if not model_name:
                        continue
                    
                    # Determine capabilities based on model name
                    capabilities = self._determine_model_capabilities(model_name)
                    
                    # Create model configuration
                    model_config = ModelConfig(
                        model_id=model_name,
                        provider=ModelProvider.OLLAMA,
                        endpoint_url="http://localhost:11434",
                        capabilities=capabilities,
                        priority=self._get_model_priority(model_name),
                        timeout=30.0,
                        max_tokens=4096,
                        context_window=self._get_context_window(model_name),
                        metadata=model_data
                    )
                    
                    self._models.append(model_config)
                    
                    # Create dedicated client for this model
                    self._model_clients[model_name] = OllamaClient()
        
        except Exception as e:
            logger.error(f"Failed to discover Ollama models: {e}")
            self._create_mock_models()

    def _determine_model_capabilities(self, model_name: str) -> List[ModelCapability]:
        """Determine model capabilities based on name"""
        capabilities = [ModelCapability.TEXT_GENERATION, ModelCapability.CONVERSATION]
        
        # Code models
        if any(keyword in model_name.lower() for keyword in ['code', 'llama', 'deepseek', 'starcoder']):
            capabilities.extend([
                ModelCapability.CODE_GENERATION,
                ModelCapability.CODE_ANALYSIS
            ])
        
        # Function calling models
        if any(keyword in model_name.lower() for keyword in ['function', 'tool', 'agent']):
            capabilities.append(ModelCapability.FUNCTION_CALLING)
        
        return capabilities

    def _get_model_priority(self, model_name: str) -> int:
        """Get model priority (lower is better)"""
        # Priority based on model quality/size
        priority_map = {
            'llama3:70b': 1,
            'llama3:13b': 2,
            'llama3:8b': 3,
            'llama3:latest': 4,
            'codellama:34b': 2,
            'codellama:13b': 3,
            'codellama:7b': 4,
            'codellama:latest': 4,
            'deepseek-coder': 2,
            'mistral:latest': 5,
            'gemma:7b': 6
        }
        
        for pattern, priority in priority_map.items():
            if pattern in model_name.lower():
                return priority
        
        return 10  # Default priority

    def _get_context_window(self, model_name: str) -> int:
        """Get model context window size"""
        # Context window estimates
        if 'llama3' in model_name.lower():
            return 8192
        elif 'codellama' in model_name.lower():
            return 16384
        elif 'mistral' in model_name.lower():
            return 8192
        else:
            return 4096

    def _create_mock_models(self):
        """Create mock models for fallback when Ollama is unavailable"""
        mock_models = [
            ModelConfig(
                model_id="mock-general",
                provider=ModelProvider.MOCK,
                endpoint_url="mock://localhost",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CONVERSATION
                ],
                priority=100  # Lowest priority
            ),
            ModelConfig(
                model_id="mock-code",
                provider=ModelProvider.MOCK,
                endpoint_url="mock://localhost",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.CODE_ANALYSIS
                ],
                priority=101
            )
        ]
        
        self._models.extend(mock_models)

    async def generate_response(self, request: InferenceRequest) -> InferenceResponse:
        """Generate AI response with full enterprise features"""
        start_time = time.time()
        
        # Update statistics
        async with self._lock:
            self._stats['total_requests'] += 1
            self._active_requests[request.request_id] = request
        
        try:
            # Check cache first
            cached_response = await self._check_cache(request)
            if cached_response:
                async with self._lock:
                    self._stats['cached_responses'] += 1
                return cached_response
            
            # Execute with fallback
            response = await self.fallback_chain.execute_with_fallback(
                request,
                self._execute_model_request
            )
            
            # Cache successful responses
            if not response.error:
                await self._cache_response(request, response)
                
                async with self._lock:
                    self._stats['successful_requests'] += 1
                    if response.fallback_used:
                        self._stats['fallback_responses'] += 1
            else:
                async with self._lock:
                    self._stats['failed_requests'] += 1
            
            # Update latency
            response.latency_ms = (time.time() - start_time) * 1000
            
            # Update average latency
            async with self._lock:
                total_requests = self._stats['total_requests']
                current_avg = self._stats['average_latency']
                self._stats['average_latency'] = (
                    (current_avg * (total_requests - 1) + response.latency_ms) / total_requests
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Request {request.request_id} failed: {e}")
            
            async with self._lock:
                self._stats['failed_requests'] += 1
            
            return InferenceResponse(
                request_id=request.request_id,
                model_id="error",
                provider=ModelProvider.MOCK,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000
            )
        
        finally:
            # Clean up
            async with self._lock:
                self._active_requests.pop(request.request_id, None)

    async def _execute_model_request(self, model: ModelConfig, request: InferenceRequest) -> InferenceResponse:
        """Execute request on specific model with circuit breaker protection"""
        
        # Get circuit breaker for this model
        circuit_breaker = self._circuit_breakers.get(model.model_id)
        
        if model.provider == ModelProvider.OLLAMA:
            # Execute Ollama request with circuit breaker
            if circuit_breaker:
                return await circuit_breaker.call(
                    self._execute_ollama_request, model, request
                )
            else:
                return await self._execute_ollama_request(model, request)
        
        elif model.provider == ModelProvider.MOCK:
            return await self._execute_mock_request(model, request)
        
        else:
            raise ValueError(f"Unsupported provider: {model.provider}")

    async def _execute_ollama_request(self, model: ModelConfig, request: InferenceRequest) -> InferenceResponse:
        """Execute Ollama API request"""
        client = self._model_clients.get(model.model_id)
        if not client:
            raise ValueError(f"No client available for model {model.model_id}")
        
        try:
            # Prepare options
            options = {
                'temperature': request.temperature or model.temperature,
                'max_tokens': request.max_tokens or model.max_tokens
            }
            
            response_text = ""
            usage = {'prompt_tokens': 0, 'completion_tokens': 0}
            
            # Stream response
            async for chunk in client.generate(
                model=model.model_id,
                prompt=request.prompt,
                options=options
            ):
                if chunk.get('done'):
                    # Final chunk with metadata
                    usage = {
                        'prompt_tokens': chunk.get('prompt_eval_count', 0),
                        'completion_tokens': chunk.get('eval_count', 0)
                    }
                    break
                else:
                    # Content chunk
                    response_text += chunk.get('response', '')
            
            return InferenceResponse(
                request_id=request.request_id,
                model_id=model.model_id,
                provider=model.provider,
                response_text=response_text,
                usage=usage,
                metadata={'model_config': model.__dict__}
            )
            
        except Exception as e:
            raise Exception(f"Ollama request failed: {e}")

    async def _execute_mock_request(self, model: ModelConfig, request: InferenceRequest) -> InferenceResponse:
        """Execute mock request for testing/fallback"""
        
        # Simulate processing delay
        await asyncio.sleep(0.1)
        
        # Generate mock response based on capabilities
        if ModelCapability.CODE_GENERATION in model.capabilities:
            response_text = f"""# Mock Code Response
# This is a fallback response when AI services are unavailable
# Request: {request.prompt[:50]}...

def mock_function():
    \"\"\"
    This is a placeholder function generated by the mock provider.
    In a real scenario, this would be generated by the AI model.
    \"\"\"
    return "Mock implementation"

# Note: This is a fallback response. Please check your AI service configuration.
"""
        else:
            response_text = f"""I apologize, but I'm currently operating in fallback mode as the primary AI services are temporarily unavailable. 

Your request was: "{request.prompt[:100]}..."

This is a mock response to ensure system reliability. Please try again later when the AI services are restored, or check your Ollama configuration.

For assistance with setup, please refer to the ABOV3 Genesis documentation."""
        
        return InferenceResponse(
            request_id=request.request_id,
            model_id=model.model_id,
            provider=model.provider,
            response_text=response_text,
            usage={'prompt_tokens': len(request.prompt) // 4, 'completion_tokens': len(response_text) // 4},
            fallback_used=True,
            metadata={'mock': True, 'reason': 'fallback_mode'}
        )

    async def _check_cache(self, request: InferenceRequest) -> Optional[InferenceResponse]:
        """Check performance cache for existing response"""
        if not self.performance_optimizer:
            return None
        
        # Create cache key from request
        cache_key = self.performance_optimizer.create_cache_key(
            'ai_response',
            request.prompt,
            request.max_tokens,
            request.temperature,
            str(request.capabilities_required)
        )
        
        cached_data = await self.performance_optimizer.get_cached_response(cache_key)
        
        if cached_data:
            cached_response = InferenceResponse(**cached_data)
            cached_response.cached = True
            return cached_response
        
        return None

    async def _cache_response(self, request: InferenceRequest, response: InferenceResponse):
        """Cache successful response"""
        if not self.performance_optimizer or response.error:
            return
        
        cache_key = self.performance_optimizer.create_cache_key(
            'ai_response',
            request.prompt,
            request.max_tokens,
            request.temperature,
            str(request.capabilities_required)
        )
        
        # Cache response data (excluding request_id)
        cache_data = {
            'request_id': response.request_id,  # Keep original for debugging
            'model_id': response.model_id,
            'provider': response.provider.value,
            'response_text': response.response_text,
            'metadata': response.metadata,
            'usage': response.usage,
            'cached': False  # Will be set to True when retrieved
        }
        
        # Cache for 30 minutes by default
        await self.performance_optimizer.cache_response(cache_key, cache_data, ttl=1800)

    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models with health status"""
        model_list = []
        
        for model in self._models:
            health = await self.health_monitor.get_model_health(model.model_id)
            
            model_info = {
                'model_id': model.model_id,
                'provider': model.provider.value,
                'capabilities': [c.value for c in model.capabilities],
                'priority': model.priority,
                'max_tokens': model.max_tokens,
                'context_window': model.context_window,
                'health_status': health.status.value if health else 'unknown',
                'response_time_ms': health.response_time if health else 0,
                'success_rate': health.success_rate if health else 0
            }
            
            model_list.append(model_info)
        
        return sorted(model_list, key=lambda m: m['priority'])

    async def get_integration_stats(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics"""
        health_report = await self.health_monitor.get_health_report()
        fallback_stats = await self.fallback_chain.get_fallback_stats() if self.fallback_chain else {}
        
        # Circuit breaker stats
        circuit_breaker_stats = {}
        for model_id, cb in self._circuit_breakers.items():
            circuit_breaker_stats[model_id] = cb.get_stats()
        
        async with self._lock:
            stats = self._stats.copy()
        
        return {
            'request_statistics': stats,
            'health_monitoring': health_report,
            'fallback_chain': fallback_stats,
            'circuit_breakers': circuit_breaker_stats,
            'active_requests': len(self._active_requests),
            'total_models': len(self._models),
            'integration_uptime': time.time() - getattr(self, '_start_time', time.time())
        }

    async def cleanup(self):
        """Cleanup resources"""
        # Stop health monitoring
        await self.health_monitor.stop_monitoring()
        
        # Close model clients
        for client in self._model_clients.values():
            try:
                await client.close()
            except:
                pass
        
        self._model_clients.clear()

# Context manager for AI integration
class ai_integration_context:
    """Context manager for AI integration with performance and resilience features"""
    
    def __init__(
        self,
        project_path: Optional[Path] = None,
        performance_optimizer: Optional[PerformanceOptimizer] = None,
        error_manager: Optional[ErrorRecoveryManager] = None
    ):
        self.integration = EnhancedOllamaIntegration(
            project_path=project_path,
            performance_optimizer=performance_optimizer,
            error_manager=error_manager
        )

    async def __aenter__(self):
        await self.integration.initialize()
        return self.integration

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.integration.cleanup()