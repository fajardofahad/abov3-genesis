#!/usr/bin/env python3
"""
ABOV3 Genesis - Enterprise Infrastructure Demo
Demonstrates the production-ready infrastructure components
"""

import asyncio
import time
from pathlib import Path

# Import infrastructure components
from abov3.infrastructure.orchestrator import (
    InfrastructureOrchestrator, 
    InfrastructureConfig,
    infrastructure_context,
    setup_abov3_infrastructure
)
from abov3.infrastructure.performance import PerformanceLevel
from abov3.infrastructure.deployment import EnvironmentTier
from abov3.infrastructure.ai_integration import InferenceRequest, ModelCapability


async def demonstrate_infrastructure():
    """Demonstrate enterprise infrastructure capabilities"""
    print("ABOV3 Genesis Enterprise Infrastructure Demo")
    print("=" * 60)
    
    # Setup project path
    project_path = Path(__file__).parent.parent
    
    # Option 1: Quick setup (recommended for most use cases)
    print("\n[SETUP] Setting up infrastructure with quick setup...")
    
    try:
        orchestrator = await setup_abov3_infrastructure(
            project_path=project_path,
            performance_level=PerformanceLevel.PRODUCTION,
            environment_tier=EnvironmentTier.DEVELOPMENT
        )
        
        print("[SUCCESS] Infrastructure setup completed!")
        
        # Demonstrate infrastructure status
        await demonstrate_status_monitoring(orchestrator)
        
        # Demonstrate AI integration
        await demonstrate_ai_integration(orchestrator)
        
        # Demonstrate performance optimization
        await demonstrate_performance_features(orchestrator)
        
        # Demonstrate deployment package creation
        await demonstrate_deployment_tools(orchestrator)
        
        # Clean shutdown
        await orchestrator.shutdown()
        
    except Exception as e:
        print(f"[ERROR] Infrastructure demo failed: {e}")
        return

    # Option 2: Using context manager (automatic cleanup)
    print("\n[RELOAD] Demonstrating context manager approach...")
    
    config = InfrastructureConfig(
        project_path=project_path,
        performance_level=PerformanceLevel.PRODUCTION,
        environment_tier=EnvironmentTier.DEVELOPMENT,
        enable_monitoring=True,
        enable_ai_integration=True,
        enable_auto_scaling=True,
        enable_deployment_tools=True
    )
    
    try:
        async with infrastructure_context(config) as orchestrator:
            print("[SUCCESS] Infrastructure context established!")
            
            # Get comprehensive status
            status = await orchestrator.get_infrastructure_status()
            print(f"[MONITOR] Infrastructure Health: {status['overall_health']} ({status['health_score']:.1f}%)")
            print(f"[TIME] Uptime: {status['uptime_seconds']:.1f} seconds")
            print(f"[NEXT] Components: {status['total_components']}")
            
    except Exception as e:
        print(f"[ERROR] Context manager demo failed: {e}")


async def demonstrate_status_monitoring(orchestrator: InfrastructureOrchestrator):
    """Demonstrate infrastructure monitoring capabilities"""
    print("\n[MONITOR] Infrastructure Status & Monitoring")
    print("-" * 40)
    
    # Get overall infrastructure status
    status = await orchestrator.get_infrastructure_status()
    
    print(f"Overall Health: {status['overall_health']}")
    print(f"Health Score: {status['health_score']:.1f}%")
    print(f"Total Components: {status['total_components']}")
    print(f"Uptime: {status['uptime_seconds']:.1f} seconds")
    
    # Show component health
    print("\nComponent Health Status:")
    for component, health in status['component_health'].items():
        status_emoji = "[SUCCESS]" if health['status'] == 'healthy' else "⚠️" if health['status'] == 'degraded' else "[ERROR]"
        print(f"  {status_emoji} {component}: {health['status']}")
    
    # Get performance metrics
    print("\nPerformance Metrics:")
    metrics = await orchestrator.get_performance_metrics()
    
    if 'performance' in metrics:
        perf = metrics['performance']
        print(f"  Cache Hit Rate: {perf.get('cache', {}).get('hit_rate', 0):.1%}")
        print(f"  Connection Pool: {perf.get('connection_pool', {}).get('success_rate', 0):.1%}")


async def demonstrate_ai_integration(orchestrator: InfrastructureOrchestrator):
    """Demonstrate AI integration capabilities"""
    print("\n[AI] AI Integration Features")
    print("-" * 40)
    
    ai_integration = orchestrator.get_component('ai_integration')
    if not ai_integration:
        print("[ERROR] AI integration not available")
        return
    
    # Get available models
    models = await ai_integration.get_available_models()
    print(f"Available Models: {len(models)}")
    
    for model in models[:3]:  # Show first 3 models
        print(f"  [LIST] {model['model_id']} ({model['provider']}) - {model['health_status']}")
    
    # Demonstrate AI request with fallback
    print("\nTesting AI Request with Enterprise Features:")
    
    request = InferenceRequest(
        prompt="Write a simple Python function that adds two numbers",
        capabilities_required=[ModelCapability.CODE_GENERATION],
        max_tokens=200,
        temperature=0.1
    )
    
    start_time = time.time()
    response = await ai_integration.generate_response(request)
    duration = (time.time() - start_time) * 1000
    
    print(f"  [FAST] Response Time: {duration:.0f}ms")
    print(f"  [NEXT] Model Used: {response.model_id}")
    print(f"  [PERF] Cached: {response.cached}")
    print(f"  [SECURE] Fallback Used: {response.fallback_used}")
    
    if response.error:
        print(f"  [ERROR] Error: {response.error}")
    else:
        print(f"  [TEXT] Response Length: {len(response.response_text)} characters")
        print(f"  [TARGET] First Line: {response.response_text.split(chr(10))[0][:60]}...")


async def demonstrate_performance_features(orchestrator: InfrastructureOrchestrator):
    """Demonstrate performance optimization features"""
    print("\n[PERF] Performance Optimization Features")
    print("-" * 40)
    
    performance = orchestrator.get_component('performance_optimizer')
    if not performance:
        print("[ERROR] Performance optimizer not available")
        return
    
    # Get performance report
    report = await performance.get_performance_report()
    
    print("Cache Performance:")
    cache_stats = report.get('cache', {})
    print(f"  Hit Rate: {cache_stats.get('hit_rate', 0):.1%}")
    print(f"  Size: {cache_stats.get('size', 0)} items")
    print(f"  Memory Usage: {cache_stats.get('memory_usage_mb', 0):.1f} MB")
    
    print("\nConnection Pool Performance:")
    pool_stats = report.get('connection_pool', {})
    print(f"  Success Rate: {pool_stats.get('success_rate', 0):.1%}")
    print(f"  Average Response Time: {pool_stats.get('avg_response_time', 0):.0f}ms")
    print(f"  Active Connections: {pool_stats.get('active_connections', 0)}")
    
    print("\nSystem Metrics:")
    system_metrics = report.get('system_metrics', {})
    print(f"  CPU Usage: {system_metrics.get('cpu_usage', 0):.1f}%")
    print(f"  Memory Usage: {system_metrics.get('memory_usage', 0):.1f}%")
    print(f"  Active Tasks: {system_metrics.get('active_tasks', 0)}")


async def demonstrate_deployment_tools(orchestrator: InfrastructureOrchestrator):
    """Demonstrate deployment capabilities"""
    print("\n[DEPLOY] Deployment Tools")
    print("-" * 40)
    
    deployment = orchestrator.get_component('deployment')
    if not deployment:
        print("[ERROR] Deployment manager not available")
        return
    
    # Get deployment status
    status = await deployment.get_deployment_status()
    
    print("Deployment Environment:")
    print(f"  Docker Available: {'[SUCCESS]' if status['docker_available'] else '[ERROR]'}")
    print(f"  Kubectl Available: {'[SUCCESS]' if status['kubectl_available'] else '[ERROR]'}")
    print(f"  Available Targets: {', '.join(status['available_targets'])}")
    print(f"  Available Tiers: {', '.join(status['available_tiers'])}")
    
    # Create a deployment package for demonstration
    print("\nCreating Development Deployment Package...")
    
    package_result = await orchestrator.create_deployment_package(
        tier=EnvironmentTier.DEVELOPMENT,
        include_ci=True
    )
    
    if package_result.get('success'):
        files_created = package_result.get('created_files', [])
        print(f"  [SUCCESS] Created {len(files_created)} deployment files")
        print("  [FOLDER] Files created:")
        for file_path in files_created[:5]:  # Show first 5 files
            print(f"    - {file_path}")
        if len(files_created) > 5:
            print(f"    ... and {len(files_created) - 5} more files")
    else:
        print(f"  [ERROR] Package creation failed: {package_result.get('error', 'Unknown error')}")


async def demonstrate_resilience_features():
    """Demonstrate error handling and resilience"""
    print("\n[SECURE] Resilience & Error Handling Demo")
    print("-" * 40)
    
    from abov3.infrastructure.resilience import (
        ErrorRecoveryManager,
        CircuitBreakerConfig,
        with_circuit_breaker,
        with_retry,
        RetryPolicy
    )
    
    # Create error manager
    error_manager = ErrorRecoveryManager()
    
    # Demonstrate circuit breaker
    print("Circuit Breaker Demo:")
    
    @with_circuit_breaker("demo_service", CircuitBreakerConfig(failure_threshold=2))
    async def flaky_service():
        import random
        if random.random() < 0.7:  # 70% failure rate
            raise Exception("Service temporarily unavailable")
        return "Success!"
    
    # Try the service multiple times
    success_count = 0
    for i in range(5):
        try:
            result = await flaky_service()
            print(f"  Attempt {i+1}: {result}")
            success_count += 1
        except Exception as e:
            print(f"  Attempt {i+1}: Failed - {e}")
    
    print(f"  Success Rate: {success_count}/5")
    
    # Demonstrate retry policy
    print("\nRetry Policy Demo:")
    
    @with_retry(RetryPolicy(max_attempts=3, base_delay=0.1))
    async def unreliable_service():
        import random
        if random.random() < 0.5:  # 50% failure rate
            raise Exception("Temporary failure")
        return "Service recovered!"
    
    try:
        result = await unreliable_service()
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Final failure: {e}")
    
    # Get error statistics
    stats = await error_manager.get_error_statistics()
    print(f"\nError Statistics:")
    print(f"  Total Errors: {stats.get('total_errors', 0)}")
    print(f"  Recovery Rate: {stats.get('recovery_rate', 0):.1%}")


async def main():
    """Main demo function"""
    try:
        print("[TARGET] Starting ABOV3 Genesis Enterprise Infrastructure Demo...")
        
        await demonstrate_infrastructure()
        await demonstrate_resilience_features()
        
        print("\n[DONE] Demo completed successfully!")
        print("\n[INFO] Key Features Demonstrated:")
        print("  [SUCCESS] Enterprise-grade infrastructure orchestration")
        print("  [SUCCESS] AI integration with fallback systems")
        print("  [SUCCESS] Performance optimization and caching")
        print("  [SUCCESS] Health monitoring and observability")
        print("  [SUCCESS] Deployment tools and CI/CD pipeline generation")
        print("  [SUCCESS] Error handling and resilience patterns")
        print("  [SUCCESS] Production-ready scalability features")
        
        print("\n[NEXT] Next Steps:")
        print("  1. Customize InfrastructureConfig for your needs")
        print("  2. Integrate with your existing codebase")
        print("  3. Configure deployment targets and CI/CD pipelines")
        print("  4. Set up monitoring and alerting")
        print("  5. Scale to production workloads")
        
    except Exception as e:
        print(f"\n[ERROR] Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("ABOV3 Genesis - Enterprise Infrastructure Demo")
    print("Production-ready infrastructure for AI coding assistants")
    print("=" * 60)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[PAUSE] Demo interrupted by user")
    except Exception as e:
        print(f"\n[CRASH] Unexpected error: {e}")
        import traceback
        traceback.print_exc()