"""
ABOV3 Genesis Infrastructure Package
Enterprise-grade infrastructure for AI-powered coding assistants
"""

__version__ = "1.0.0"
__author__ = "ABOV3 Enterprise DevOps Team"

# Core orchestration
from .orchestrator import (
    InfrastructureOrchestrator,
    InfrastructureConfig,
    infrastructure_context,
    setup_abov3_infrastructure
)

# Performance optimization
from .performance import (
    PerformanceOptimizer,
    CacheManager,
    ConnectionPool,
    PerformanceLevel
)

# Error handling and resilience
from .resilience import (
    ErrorRecoveryManager,
    CircuitBreaker,
    CircuitBreakerConfig,
    RetryPolicy,
    with_circuit_breaker,
    with_retry
)

# Scalability features
from .scalability import (
    LoadBalancer,
    ResourceManager,
    AutoScaler,
    DistributedTaskQueue,
    LoadBalancingStrategy
)

# AI integration enhancements
from .ai_integration import (
    EnhancedOllamaIntegration,
    AIHealthMonitor,
    ModelFallbackChain,
    InferenceRequest,
    ModelCapability
)

# Environment setup
from .environment import (
    EnvironmentSetup,
    DependencyManager,
    SystemDetector
)

# Monitoring and observability
from .monitoring import (
    ObservabilityManager,
    StructuredLogger,
    MetricsCollector,
    AlertManager
)

# Deployment tools
from .deployment import (
    DeploymentManager,
    DockerfileGenerator,
    KubernetesManifestGenerator,
    CIPipelineGenerator,
    EnvironmentTier
)

__all__ = [
    # Orchestration
    'InfrastructureOrchestrator',
    'InfrastructureConfig', 
    'infrastructure_context',
    'setup_abov3_infrastructure',
    
    # Performance
    'PerformanceOptimizer',
    'CacheManager',
    'ConnectionPool',
    'PerformanceLevel',
    
    # Resilience
    'ErrorRecoveryManager',
    'CircuitBreaker',
    'CircuitBreakerConfig',
    'RetryPolicy',
    'with_circuit_breaker',
    'with_retry',
    
    # Scalability
    'LoadBalancer',
    'ResourceManager',
    'AutoScaler',
    'DistributedTaskQueue',
    'LoadBalancingStrategy',
    
    # AI Integration
    'EnhancedOllamaIntegration',
    'AIHealthMonitor',
    'ModelFallbackChain',
    'InferenceRequest',
    'ModelCapability',
    
    # Environment
    'EnvironmentSetup',
    'DependencyManager',
    'SystemDetector',
    
    # Monitoring
    'ObservabilityManager',
    'StructuredLogger',
    'MetricsCollector',
    'AlertManager',
    
    # Deployment
    'DeploymentManager',
    'DockerfileGenerator',
    'KubernetesManifestGenerator',
    'CIPipelineGenerator',
    'EnvironmentTier',
]