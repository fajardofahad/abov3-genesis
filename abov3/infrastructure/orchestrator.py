"""
ABOV3 Genesis - Infrastructure Orchestration
Main coordinator for all infrastructure components
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field

from .performance import PerformanceOptimizer, PerformanceLevel
from .resilience import ErrorRecoveryManager, setup_global_exception_handler
from .scalability import ScalabilityManager
from .ai_integration import EnhancedOllamaIntegration
from .environment import EnvironmentManager
from .monitoring import ObservabilityManager
from .deployment import DeploymentManager, DeploymentConfig, EnvironmentTier, DeploymentTarget

logger = logging.getLogger(__name__)

@dataclass
class InfrastructureConfig:
    """Configuration for infrastructure components"""
    project_path: Path
    performance_level: PerformanceLevel = PerformanceLevel.PRODUCTION
    enable_monitoring: bool = True
    enable_ai_integration: bool = True
    enable_auto_scaling: bool = True
    enable_environment_setup: bool = True
    enable_deployment_tools: bool = True
    environment_tier: EnvironmentTier = EnvironmentTier.PRODUCTION

class InfrastructureOrchestrator:
    """
    Main infrastructure orchestrator for ABOV3 Genesis
    Coordinates all enterprise infrastructure components
    """

    def __init__(self, config: InfrastructureConfig):
        self.config = config
        self.project_path = config.project_path
        
        # Initialize components
        self._components = {}
        self._startup_order = []
        self._shutdown_order = []
        
        # Health tracking
        self._health_status = {}
        self._startup_time = None
        self._ready = False
        
        # Background tasks
        self._health_monitor_task = None

    async def initialize(self) -> Dict[str, Any]:
        """Initialize all infrastructure components"""
        logger.info("🚀 Initializing ABOV3 Genesis Infrastructure")
        self._startup_time = time.time()
        
        initialization_results = {
            'started_at': self._startup_time,
            'components': {},
            'startup_order': [],
            'errors': []
        }
        
        try:
            # 1. Performance Optimization (foundation layer)
            if True:  # Always enable performance optimization
                logger.info("📈 Initializing Performance Optimizer...")
                perf_optimizer = PerformanceOptimizer(
                    self.config.performance_level,
                    self.config.project_path
                )
                self._components['performance_optimizer'] = perf_optimizer
                self._startup_order.append('performance_optimizer')
                initialization_results['components']['performance_optimizer'] = {'status': 'initialized'}
                initialization_results['startup_order'].append('performance_optimizer')

            # 2. Error Recovery & Resilience (foundation layer)
            logger.info("🛡️ Initializing Error Recovery Manager...")
            error_manager = ErrorRecoveryManager(self.config.project_path)
            self._components['error_recovery'] = error_manager
            self._startup_order.append('error_recovery')
            
            # Setup global exception handling
            setup_global_exception_handler(error_manager)
            
            initialization_results['components']['error_recovery'] = {'status': 'initialized'}
            initialization_results['startup_order'].append('error_recovery')

            # 3. Monitoring & Observability (critical for operations)
            if self.config.enable_monitoring:
                logger.info("📊 Initializing Observability Manager...")
                observability = ObservabilityManager(self.config.project_path)
                self._components['observability'] = observability
                self._startup_order.append('observability')
                initialization_results['components']['observability'] = {'status': 'initialized'}
                initialization_results['startup_order'].append('observability')

            # 4. Environment Management
            if self.config.enable_environment_setup:
                logger.info("🔧 Initializing Environment Manager...")
                env_manager = EnvironmentManager(self.config.project_path)
                self._components['environment'] = env_manager
                self._startup_order.append('environment')
                
                # Validate environment
                env_validation = await env_manager.validate_environment()
                initialization_results['components']['environment'] = {
                    'status': 'initialized',
                    'validation': env_validation
                }
                initialization_results['startup_order'].append('environment')

            # 5. AI Integration (core functionality)
            if self.config.enable_ai_integration:
                logger.info("🤖 Initializing AI Integration...")
                ai_integration = EnhancedOllamaIntegration(
                    project_path=self.config.project_path,
                    performance_optimizer=self._components.get('performance_optimizer'),
                    error_manager=error_manager
                )
                await ai_integration.initialize()
                self._components['ai_integration'] = ai_integration
                self._startup_order.append('ai_integration')
                
                # Get AI integration stats
                ai_stats = await ai_integration.get_integration_stats()
                initialization_results['components']['ai_integration'] = {
                    'status': 'initialized',
                    'stats': ai_stats
                }
                initialization_results['startup_order'].append('ai_integration')

            # 6. Scalability Management
            if self.config.enable_auto_scaling:
                logger.info("⚖️ Initializing Scalability Manager...")
                scalability = ScalabilityManager(self.config.project_path)
                
                # Add local worker node
                await scalability.add_worker_node("localhost", 8000, capabilities=["ai_inference", "code_generation"])
                
                self._components['scalability'] = scalability
                self._startup_order.append('scalability')
                
                # Get scalability report
                scalability_report = await scalability.get_comprehensive_report()
                initialization_results['components']['scalability'] = {
                    'status': 'initialized',
                    'report': scalability_report
                }
                initialization_results['startup_order'].append('scalability')

            # 7. Deployment Tools
            if self.config.enable_deployment_tools:
                logger.info("🚢 Initializing Deployment Manager...")
                deployment = DeploymentManager(self.config.project_path)
                self._components['deployment'] = deployment
                self._startup_order.append('deployment')
                
                # Get deployment status
                deployment_status = await deployment.get_deployment_status()
                initialization_results['components']['deployment'] = {
                    'status': 'initialized',
                    'deployment_status': deployment_status
                }
                initialization_results['startup_order'].append('deployment')

            # Start health monitoring
            self._start_health_monitoring()

            # Mark as ready
            self._ready = True
            
            initialization_results['success'] = True
            initialization_results['completed_at'] = time.time()
            initialization_results['duration'] = initialization_results['completed_at'] - self._startup_time
            initialization_results['total_components'] = len(self._components)
            
            # Reverse order for shutdown
            self._shutdown_order = list(reversed(self._startup_order))
            
            logger.info(f"✅ Infrastructure initialized successfully in {initialization_results['duration']:.2f}s")
            logger.info(f"📦 Components loaded: {list(self._components.keys())}")
            
            return initialization_results
            
        except Exception as e:
            logger.error(f"❌ Infrastructure initialization failed: {e}")
            initialization_results['success'] = False
            initialization_results['error'] = str(e)
            initialization_results['errors'].append(str(e))
            
            # Cleanup any partially initialized components
            await self._cleanup_partial_initialization()
            
            return initialization_results

    def _start_health_monitoring(self):
        """Start background health monitoring"""
        if self._health_monitor_task is None:
            self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())

    async def _health_monitor_loop(self):
        """Background health monitoring loop"""
        while self._ready:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                await self._update_component_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")

    async def _update_component_health(self):
        """Update health status of all components"""
        for component_name, component in self._components.items():
            try:
                # Check component health based on type
                health_status = await self._check_component_health(component_name, component)
                self._health_status[component_name] = {
                    'status': health_status,
                    'last_check': time.time()
                }
            except Exception as e:
                self._health_status[component_name] = {
                    'status': 'unhealthy',
                    'last_check': time.time(),
                    'error': str(e)
                }

    async def _check_component_health(self, component_name: str, component) -> str:
        """Check health of individual component"""
        try:
            if component_name == 'performance_optimizer':
                # Check if performance optimizer is responding
                report = await component.get_performance_report()
                return 'healthy' if report else 'degraded'
            
            elif component_name == 'error_recovery':
                # Check error manager statistics
                stats = await component.get_error_statistics()
                error_rate = stats.get('errors_last_hour', 0)
                return 'healthy' if error_rate < 10 else 'degraded'
            
            elif component_name == 'observability':
                # Check monitoring dashboard
                dashboard = await component.get_observability_dashboard()
                return 'healthy' if dashboard.get('uptime_seconds', 0) > 0 else 'unhealthy'
            
            elif component_name == 'ai_integration':
                # Check AI integration stats
                stats = await component.get_integration_stats()
                healthy_models = stats.get('health_monitoring', {}).get('healthy_models', 0)
                return 'healthy' if healthy_models > 0 else 'degraded'
            
            elif component_name == 'scalability':
                # Check scalability report
                report = await component.get_comprehensive_report()
                healthy_nodes = report.get('load_balancing', {}).get('healthy_nodes', 0)
                return 'healthy' if healthy_nodes > 0 else 'degraded'
            
            else:
                return 'healthy'  # Default for components without specific health checks
                
        except Exception as e:
            logger.warning(f"Health check failed for {component_name}: {e}")
            return 'unhealthy'

    async def _cleanup_partial_initialization(self):
        """Cleanup partially initialized components during failed startup"""
        logger.info("🧹 Cleaning up partially initialized components...")
        
        for component_name in reversed(self._startup_order):
            if component_name in self._components:
                try:
                    component = self._components[component_name]
                    if hasattr(component, 'cleanup'):
                        await component.cleanup()
                    logger.info(f"✅ Cleaned up {component_name}")
                except Exception as e:
                    logger.error(f"❌ Failed to cleanup {component_name}: {e}")

    async def get_infrastructure_status(self) -> Dict[str, Any]:
        """Get comprehensive infrastructure status"""
        if not self._ready:
            return {
                'status': 'not_ready',
                'message': 'Infrastructure not fully initialized'
            }
        
        uptime = time.time() - self._startup_time if self._startup_time else 0
        
        status = {
            'status': 'ready',
            'uptime_seconds': uptime,
            'startup_time': self._startup_time,
            'total_components': len(self._components),
            'component_health': self._health_status.copy(),
            'components': {}
        }
        
        # Get detailed status from each component
        for component_name, component in self._components.items():
            try:
                if component_name == 'performance_optimizer':
                    status['components'][component_name] = await component.get_performance_report()
                elif component_name == 'error_recovery':
                    status['components'][component_name] = await component.get_error_statistics()
                elif component_name == 'observability':
                    status['components'][component_name] = await component.get_observability_dashboard()
                elif component_name == 'ai_integration':
                    status['components'][component_name] = await component.get_integration_stats()
                elif component_name == 'scalability':
                    status['components'][component_name] = await component.get_comprehensive_report()
                elif component_name == 'deployment':
                    status['components'][component_name] = await component.get_deployment_status()
                elif component_name == 'environment':
                    status['components'][component_name] = await component.get_environment_status()
                else:
                    status['components'][component_name] = {'status': 'active'}
                    
            except Exception as e:
                status['components'][component_name] = {'error': str(e)}
        
        # Overall health assessment
        healthy_components = sum(1 for h in self._health_status.values() if h.get('status') == 'healthy')
        total_components = len(self._components)
        
        if healthy_components == total_components:
            status['overall_health'] = 'healthy'
        elif healthy_components > total_components * 0.7:
            status['overall_health'] = 'degraded'
        else:
            status['overall_health'] = 'unhealthy'
        
        status['health_score'] = (healthy_components / max(1, total_components)) * 100
        
        return status

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from all components"""
        metrics = {}
        
        if 'performance_optimizer' in self._components:
            perf_report = await self._components['performance_optimizer'].get_performance_report()
            metrics['performance'] = perf_report
        
        if 'observability' in self._components:
            dashboard = await self._components['observability'].get_observability_dashboard()
            metrics['observability'] = dashboard
        
        if 'ai_integration' in self._components:
            ai_stats = await self._components['ai_integration'].get_integration_stats()
            metrics['ai_integration'] = ai_stats
        
        if 'scalability' in self._components:
            scale_report = await self._components['scalability'].get_comprehensive_report()
            metrics['scalability'] = scale_report
        
        return metrics

    def get_component(self, name: str):
        """Get a specific infrastructure component"""
        return self._components.get(name)

    async def restart_component(self, component_name: str) -> bool:
        """Restart a specific component"""
        if component_name not in self._components:
            return False
        
        try:
            logger.info(f"🔄 Restarting component: {component_name}")
            
            component = self._components[component_name]
            
            # Cleanup existing component
            if hasattr(component, 'cleanup'):
                await component.cleanup()
            
            # Reinitialize component
            if component_name == 'ai_integration':
                new_component = EnhancedOllamaIntegration(
                    project_path=self.config.project_path,
                    performance_optimizer=self._components.get('performance_optimizer'),
                    error_manager=self._components.get('error_recovery')
                )
                await new_component.initialize()
                self._components[component_name] = new_component
            
            # Add other component restart logic as needed
            
            logger.info(f"✅ Component restarted: {component_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to restart {component_name}: {e}")
            return False

    async def shutdown(self) -> Dict[str, Any]:
        """Graceful shutdown of all infrastructure components"""
        logger.info("[SHUTDOWN] Shutting down ABOV3 Genesis Infrastructure")
        
        shutdown_results = {
            'started_at': time.time(),
            'components_shutdown': [],
            'errors': []
        }
        
        # Stop health monitoring
        self._ready = False
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown components in reverse order
        for component_name in self._shutdown_order:
            if component_name in self._components:
                try:
                    logger.info(f"[SHUTDOWN] Shutting down {component_name}...")
                    component = self._components[component_name]
                    
                    if hasattr(component, 'cleanup'):
                        await component.cleanup()
                    
                    shutdown_results['components_shutdown'].append(component_name)
                    logger.info(f"[SUCCESS] {component_name} shutdown complete")
                    
                except Exception as e:
                    error_msg = f"Error shutting down {component_name}: {e}"
                    logger.error(f"[ERROR] {error_msg}")
                    shutdown_results['errors'].append(error_msg)
        
        # Clear components
        self._components.clear()
        self._health_status.clear()
        
        shutdown_results['completed_at'] = time.time()
        shutdown_results['duration'] = shutdown_results['completed_at'] - shutdown_results['started_at']
        shutdown_results['success'] = len(shutdown_results['errors']) == 0
        
        logger.info(f"[COMPLETE] Infrastructure shutdown completed in {shutdown_results['duration']:.2f}s")
        
        return shutdown_results

    async def create_deployment_package(
        self,
        tier: EnvironmentTier = EnvironmentTier.PRODUCTION,
        include_ci: bool = True
    ) -> Dict[str, Any]:
        """Create a complete deployment package"""
        logger.info(f"📦 Creating deployment package for {tier.value}")
        
        if 'deployment' not in self._components:
            return {'error': 'Deployment manager not available'}
        
        deployment_manager = self._components['deployment']
        
        # Create deployment configuration
        config = DeploymentConfig(
            name="abov3-genesis",
            target=DeploymentTarget.KUBERNETES,
            tier=tier,
            environment_variables={
                'ABOV3_ENV': tier.value,
                'ABOV3_LOG_LEVEL': 'INFO' if tier == EnvironmentTier.PRODUCTION else 'DEBUG'
            }
        )
        
        # Setup deployment environment
        result = await deployment_manager.setup_deployment_environment(
            config,
            include_ci=include_ci,
            ci_platforms=['github', 'gitlab']
        )
        
        return result

# Context manager for complete infrastructure
class infrastructure_context:
    """Context manager for complete infrastructure orchestration"""
    
    def __init__(self, config: InfrastructureConfig):
        self.orchestrator = InfrastructureOrchestrator(config)
        self.initialization_result = None

    async def __aenter__(self):
        self.initialization_result = await self.orchestrator.initialize()
        
        if not self.initialization_result.get('success'):
            raise RuntimeError(f"Infrastructure initialization failed: {self.initialization_result.get('error')}")
        
        return self.orchestrator

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        shutdown_result = await self.orchestrator.shutdown()
        
        if not shutdown_result.get('success'):
            logger.warning(f"Infrastructure shutdown had errors: {shutdown_result.get('errors')}")

# Convenience function for quick setup
async def setup_abov3_infrastructure(
    project_path: Path,
    performance_level: PerformanceLevel = PerformanceLevel.PRODUCTION,
    environment_tier: EnvironmentTier = EnvironmentTier.PRODUCTION
) -> InfrastructureOrchestrator:
    """Quick setup function for ABOV3 Genesis infrastructure"""
    
    config = InfrastructureConfig(
        project_path=project_path,
        performance_level=performance_level,
        environment_tier=environment_tier,
        enable_monitoring=True,
        enable_ai_integration=True,
        enable_auto_scaling=True,
        enable_environment_setup=True,
        enable_deployment_tools=True
    )
    
    orchestrator = InfrastructureOrchestrator(config)
    
    init_result = await orchestrator.initialize()
    
    if not init_result.get('success'):
        raise RuntimeError(f"Infrastructure setup failed: {init_result.get('error')}")
    
    return orchestrator