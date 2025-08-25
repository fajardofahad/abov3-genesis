#!/usr/bin/env python3
"""
ABOV3 Genesis - Enterprise Deployment Automation
Production-ready deployment system for enterprise environments
"""

import os
import sys
import json
import yaml
import shutil
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    name: str
    environment: str  # dev, staging, production
    target_platform: str  # docker, kubernetes, systemd, windows_service
    ollama_models: List[str] = field(default_factory=list)
    security_level: str = "standard"  # minimal, standard, high, airgapped
    monitoring_enabled: bool = True
    load_balancing: bool = False
    auto_scaling: bool = False
    backup_enabled: bool = True
    ssl_enabled: bool = True
    log_level: str = "INFO"
    max_concurrent_requests: int = 100
    cache_size_mb: int = 512
    custom_settings: Dict[str, Any] = field(default_factory=dict)

class EnterpriseDeploymentManager:
    """
    Enterprise deployment manager for ABOV3 Genesis
    Handles deployment to various environments and platforms
    """
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.deploy_dir = project_root / "deploy"
        self.templates_dir = self.deploy_dir / "templates"
        self.configs_dir = self.deploy_dir / "configs"
        
        # Create deployment directories
        self._create_deployment_structure()
        
        # Supported deployment targets
        self.supported_platforms = [
            "docker",
            "kubernetes", 
            "systemd",
            "windows_service",
            "docker_swarm",
            "standalone"
        ]
        
        self.supported_environments = ["dev", "staging", "production"]
        
    def _create_deployment_structure(self):
        """Create deployment directory structure"""
        directories = [
            self.deploy_dir,
            self.templates_dir,
            self.configs_dir,
            self.deploy_dir / "scripts",
            self.deploy_dir / "docker",
            self.deploy_dir / "kubernetes",
            self.deploy_dir / "systemd",
            self.deploy_dir / "monitoring",
            self.deploy_dir / "security"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def create_deployment_package(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Create complete deployment package"""
        print(f"Creating deployment package for {config.name} ({config.environment})")
        
        try:
            deployment_id = self._generate_deployment_id(config)
            package_dir = self.deploy_dir / f"{config.name}-{deployment_id}"
            package_dir.mkdir(exist_ok=True)
            
            # Generate deployment artifacts
            artifacts = {}
            
            if config.target_platform == "docker":
                artifacts.update(self._create_docker_deployment(config, package_dir))
            elif config.target_platform == "kubernetes":
                artifacts.update(self._create_kubernetes_deployment(config, package_dir))
            elif config.target_platform == "systemd":
                artifacts.update(self._create_systemd_deployment(config, package_dir))
            elif config.target_platform == "windows_service":
                artifacts.update(self._create_windows_service_deployment(config, package_dir))
            else:
                artifacts.update(self._create_standalone_deployment(config, package_dir))
            
            # Create configuration files
            artifacts.update(self._create_configuration_files(config, package_dir))
            
            # Create deployment scripts
            artifacts.update(self._create_deployment_scripts(config, package_dir))
            
            # Create monitoring setup
            if config.monitoring_enabled:
                artifacts.update(self._create_monitoring_setup(config, package_dir))
            
            # Create security configurations
            artifacts.update(self._create_security_setup(config, package_dir))
            
            # Create documentation
            artifacts.update(self._create_deployment_documentation(config, package_dir))
            
            return {
                "status": "success",
                "deployment_id": deployment_id,
                "package_path": str(package_dir),
                "artifacts": artifacts,
                "config": config.__dict__
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _generate_deployment_id(self, config: DeploymentConfig) -> str:
        """Generate unique deployment ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_hash = hashlib.md5(
            f"{config.name}{config.environment}{config.target_platform}".encode()
        ).hexdigest()[:8]
        return f"{timestamp}_{config_hash}"
    
    def _create_docker_deployment(self, config: DeploymentConfig, package_dir: Path) -> Dict[str, str]:
        """Create Docker deployment artifacts"""
        artifacts = {}
        
        # Dockerfile
        dockerfile_content = self._generate_dockerfile(config)
        dockerfile_path = package_dir / "Dockerfile"
        dockerfile_path.write_text(dockerfile_content)
        artifacts["dockerfile"] = str(dockerfile_path)
        
        # Docker Compose
        compose_content = self._generate_docker_compose(config)
        compose_path = package_dir / "docker-compose.yml"
        compose_path.write_text(compose_content)
        artifacts["docker_compose"] = str(compose_path)
        
        # Environment file
        env_content = self._generate_env_file(config)
        env_path = package_dir / ".env"
        env_path.write_text(env_content)
        artifacts["env_file"] = str(env_path)
        
        return artifacts
    
    def _create_kubernetes_deployment(self, config: DeploymentConfig, package_dir: Path) -> Dict[str, str]:
        """Create Kubernetes deployment artifacts"""
        artifacts = {}
        
        # Deployment manifest
        deployment_manifest = self._generate_k8s_deployment(config)
        deployment_path = package_dir / "deployment.yaml"
        deployment_path.write_text(deployment_manifest)
        artifacts["k8s_deployment"] = str(deployment_path)
        
        # Service manifest
        service_manifest = self._generate_k8s_service(config)
        service_path = package_dir / "service.yaml"
        service_path.write_text(service_manifest)
        artifacts["k8s_service"] = str(service_path)
        
        # ConfigMap
        configmap_manifest = self._generate_k8s_configmap(config)
        configmap_path = package_dir / "configmap.yaml"
        configmap_path.write_text(configmap_manifest)
        artifacts["k8s_configmap"] = str(configmap_path)
        
        # Ingress (if SSL enabled)
        if config.ssl_enabled:
            ingress_manifest = self._generate_k8s_ingress(config)
            ingress_path = package_dir / "ingress.yaml"
            ingress_path.write_text(ingress_manifest)
            artifacts["k8s_ingress"] = str(ingress_path)
        
        return artifacts
    
    def _create_systemd_deployment(self, config: DeploymentConfig, package_dir: Path) -> Dict[str, str]:
        """Create systemd deployment artifacts"""
        artifacts = {}
        
        # Service file
        service_content = self._generate_systemd_service(config)
        service_path = package_dir / f"abov3-{config.name}.service"
        service_path.write_text(service_content)
        artifacts["systemd_service"] = str(service_path)
        
        # Installation script
        install_script = self._generate_systemd_install_script(config)
        install_path = package_dir / "install_systemd.sh"
        install_path.write_text(install_script)
        install_path.chmod(0o755)
        artifacts["systemd_install"] = str(install_path)
        
        return artifacts
    
    def _create_windows_service_deployment(self, config: DeploymentConfig, package_dir: Path) -> Dict[str, str]:
        """Create Windows service deployment artifacts"""
        artifacts = {}
        
        # Windows service wrapper script
        service_script = self._generate_windows_service_script(config)
        service_path = package_dir / "abov3_service.py"
        service_path.write_text(service_script)
        artifacts["windows_service"] = str(service_path)
        
        # Installation batch file
        install_batch = self._generate_windows_install_batch(config)
        batch_path = package_dir / "install_service.bat"
        batch_path.write_text(install_batch)
        artifacts["windows_install"] = str(batch_path)
        
        return artifacts
    
    def _create_standalone_deployment(self, config: DeploymentConfig, package_dir: Path) -> Dict[str, str]:
        """Create standalone deployment artifacts"""
        artifacts = {}
        
        # Startup script
        startup_script = self._generate_startup_script(config)
        script_path = package_dir / "start_abov3.sh"
        script_path.write_text(startup_script)
        script_path.chmod(0o755)
        artifacts["startup_script"] = str(script_path)
        
        # Windows startup batch
        windows_batch = self._generate_windows_startup_batch(config)
        batch_path = package_dir / "start_abov3.bat"
        batch_path.write_text(windows_batch)
        artifacts["windows_startup"] = str(batch_path)
        
        return artifacts
    
    def _create_configuration_files(self, config: DeploymentConfig, package_dir: Path) -> Dict[str, str]:
        """Create configuration files"""
        artifacts = {}
        
        # Main application config
        app_config = self._generate_app_config(config)
        app_config_path = package_dir / "abov3_config.yaml"
        app_config_path.write_text(app_config)
        artifacts["app_config"] = str(app_config_path)
        
        # Logging configuration
        log_config = self._generate_logging_config(config)
        log_config_path = package_dir / "logging_config.yaml"
        log_config_path.write_text(log_config)
        artifacts["logging_config"] = str(log_config_path)
        
        # Security configuration
        security_config = self._generate_security_config(config)
        security_config_path = package_dir / "security_config.yaml"
        security_config_path.write_text(security_config)
        artifacts["security_config"] = str(security_config_path)
        
        return artifacts
    
    def _create_deployment_scripts(self, config: DeploymentConfig, package_dir: Path) -> Dict[str, str]:
        """Create deployment scripts"""
        artifacts = {}
        
        # Deployment script
        deploy_script = self._generate_deploy_script(config)
        deploy_path = package_dir / "deploy.sh"
        deploy_path.write_text(deploy_script)
        deploy_path.chmod(0o755)
        artifacts["deploy_script"] = str(deploy_path)
        
        # Health check script
        health_script = self._generate_health_check_script(config)
        health_path = package_dir / "health_check.py"
        health_path.write_text(health_script)
        health_path.chmod(0o755)
        artifacts["health_check"] = str(health_path)
        
        # Backup script
        if config.backup_enabled:
            backup_script = self._generate_backup_script(config)
            backup_path = package_dir / "backup.sh"
            backup_path.write_text(backup_script)
            backup_path.chmod(0o755)
            artifacts["backup_script"] = str(backup_path)
        
        return artifacts
    
    def _create_monitoring_setup(self, config: DeploymentConfig, package_dir: Path) -> Dict[str, str]:
        """Create monitoring setup"""
        artifacts = {}
        
        # Prometheus configuration
        prometheus_config = self._generate_prometheus_config(config)
        prometheus_path = package_dir / "prometheus.yml"
        prometheus_path.write_text(prometheus_config)
        artifacts["prometheus_config"] = str(prometheus_path)
        
        # Grafana dashboard
        grafana_dashboard = self._generate_grafana_dashboard(config)
        dashboard_path = package_dir / "abov3_dashboard.json"
        dashboard_path.write_text(grafana_dashboard)
        artifacts["grafana_dashboard"] = str(dashboard_path)
        
        return artifacts
    
    def _create_security_setup(self, config: DeploymentConfig, package_dir: Path) -> Dict[str, str]:
        """Create security configurations"""
        artifacts = {}
        
        # TLS/SSL configuration
        if config.ssl_enabled:
            ssl_config = self._generate_ssl_config(config)
            ssl_path = package_dir / "ssl_config.yaml"
            ssl_path.write_text(ssl_config)
            artifacts["ssl_config"] = str(ssl_path)
        
        # Firewall rules
        firewall_rules = self._generate_firewall_rules(config)
        firewall_path = package_dir / "firewall_rules.sh"
        firewall_path.write_text(firewall_rules)
        firewall_path.chmod(0o755)
        artifacts["firewall_rules"] = str(firewall_path)
        
        return artifacts
    
    def _create_deployment_documentation(self, config: DeploymentConfig, package_dir: Path) -> Dict[str, str]:
        """Create deployment documentation"""
        artifacts = {}
        
        # Deployment README
        readme_content = self._generate_deployment_readme(config)
        readme_path = package_dir / "README.md"
        readme_path.write_text(readme_content)
        artifacts["readme"] = str(readme_path)
        
        # Troubleshooting guide
        troubleshooting = self._generate_troubleshooting_guide(config)
        troubleshooting_path = package_dir / "TROUBLESHOOTING.md"
        troubleshooting_path.write_text(troubleshooting)
        artifacts["troubleshooting"] = str(troubleshooting_path)
        
        return artifacts
    
    # Template generation methods
    
    def _generate_dockerfile(self, config: DeploymentConfig) -> str:
        """Generate Dockerfile"""
        return f"""# ABOV3 Genesis - Production Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Install Ollama (if needed)
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 abov3user && chown -R abov3user:abov3user /app
USER abov3user

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD python health_check.py

# Expose port
EXPOSE 8000

# Start application
CMD ["python", "-m", "abov3.main"]
"""
    
    def _generate_docker_compose(self, config: DeploymentConfig) -> str:
        """Generate Docker Compose file"""
        compose = {
            'version': '3.8',
            'services': {
                'abov3-genesis': {
                    'build': '.',
                    'ports': ['8000:8000'],
                    'environment': {
                        'ABOV3_ENV': config.environment,
                        'ABOV3_LOG_LEVEL': config.log_level,
                        'ABOV3_MAX_REQUESTS': config.max_concurrent_requests,
                        'ABOV3_CACHE_SIZE': config.cache_size_mb
                    },
                    'volumes': [
                        './logs:/app/logs',
                        './data:/app/data'
                    ],
                    'restart': 'unless-stopped',
                    'healthcheck': {
                        'test': ['CMD', 'python', 'health_check.py'],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3,
                        'start_period': '60s'
                    }
                }
            }
        }
        
        if config.monitoring_enabled:
            compose['services']['prometheus'] = {
                'image': 'prom/prometheus:latest',
                'ports': ['9090:9090'],
                'volumes': ['./prometheus.yml:/etc/prometheus/prometheus.yml'],
                'restart': 'unless-stopped'
            }
            compose['services']['grafana'] = {
                'image': 'grafana/grafana:latest',
                'ports': ['3000:3000'],
                'environment': {
                    'GF_SECURITY_ADMIN_PASSWORD': 'admin123'
                },
                'volumes': ['./abov3_dashboard.json:/var/lib/grafana/dashboards/abov3.json'],
                'restart': 'unless-stopped'
            }
        
        return yaml.dump(compose, default_flow_style=False)
    
    def _generate_app_config(self, config: DeploymentConfig) -> str:
        """Generate application configuration"""
        app_config = {
            'application': {
                'name': config.name,
                'environment': config.environment,
                'log_level': config.log_level,
                'max_concurrent_requests': config.max_concurrent_requests,
                'cache_size_mb': config.cache_size_mb
            },
            'ai_models': {
                'ollama_models': config.ollama_models,
                'model_timeout_seconds': 30,
                'fallback_enabled': True
            },
            'security': {
                'level': config.security_level,
                'ssl_enabled': config.ssl_enabled,
                'cors_enabled': config.environment != 'production'
            },
            'monitoring': {
                'enabled': config.monitoring_enabled,
                'metrics_endpoint': '/metrics',
                'health_endpoint': '/health'
            },
            'performance': {
                'load_balancing': config.load_balancing,
                'auto_scaling': config.auto_scaling,
                'backup_enabled': config.backup_enabled
            }
        }
        
        # Add custom settings
        app_config.update(config.custom_settings)
        
        return yaml.dump(app_config, default_flow_style=False)
    
    def _generate_deployment_readme(self, config: DeploymentConfig) -> str:
        """Generate deployment README"""
        return f"""# ABOV3 Genesis Deployment: {config.name}

## Overview
This deployment package contains everything needed to deploy ABOV3 Genesis in the {config.environment} environment using {config.target_platform}.

## Configuration
- Environment: {config.environment}
- Platform: {config.target_platform}
- Security Level: {config.security_level}
- Monitoring: {"Enabled" if config.monitoring_enabled else "Disabled"}
- SSL: {"Enabled" if config.ssl_enabled else "Disabled"}

## Quick Start

### Prerequisites
- Python 3.11 or higher
- Ollama installed and running
- Required AI models: {', '.join(config.ollama_models) if config.ollama_models else 'Auto-detected'}

### Deployment Steps

1. **Prepare Environment**
   ```bash
   # Set up required directories
   mkdir -p logs data backups
   chmod 755 *.sh
   ```

2. **Configure Application**
   ```bash
   # Review and customize configuration files
   nano abov3_config.yaml
   nano security_config.yaml
   ```

3. **Deploy Application**
   ```bash
   # Run deployment script
   ./deploy.sh
   ```

4. **Verify Deployment**
   ```bash
   # Check health status
   python health_check.py
   ```

## Monitoring
{f'''
- Prometheus metrics: http://localhost:9090
- Grafana dashboard: http://localhost:3000 (admin/admin123)
- Health check: http://localhost:8000/health
''' if config.monitoring_enabled else '- Monitoring is disabled for this deployment'}

## Security
- Security level: {config.security_level}
{f'- SSL/TLS encryption enabled' if config.ssl_enabled else '- SSL/TLS encryption disabled'}
- Firewall rules: See firewall_rules.sh

## Maintenance

### Backup
{f'''
```bash
./backup.sh
```
''' if config.backup_enabled else 'Backup is disabled for this deployment'}

### Logs
- Application logs: ./logs/
- System logs: Check systemd/Docker logs

### Updates
1. Stop the service
2. Replace application files
3. Restart the service
4. Verify health check

## Support
For issues and troubleshooting, see TROUBLESHOOTING.md

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    def _generate_health_check_script(self, config: DeploymentConfig) -> str:
        """Generate health check script"""
        return f"""#!/usr/bin/env python3
\"\"\"
ABOV3 Genesis Health Check
Verifies system health and readiness
\"\"\"

import requests
import sys
import json

def check_health():
    \"\"\"Check application health\"\"\"
    try:
        # Check main application
        response = requests.get('http://localhost:8000/health', timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            print(f"Status: {{health_data.get('status', 'unknown')}}")
            print(f"Uptime: {{health_data.get('uptime', 0):.2f}} seconds")
            print(f"Version: {{health_data.get('version', 'unknown')}}")
            
            if health_data.get('status') == 'healthy':
                print("Health check PASSED")
                return 0
            else:
                print("Health check FAILED - Service unhealthy")
                return 1
        else:
            print(f"Health check FAILED - HTTP {{response.status_code}}")
            return 1
            
    except requests.exceptions.RequestException as e:
        print(f"Health check FAILED - Connection error: {{e}}")
        return 1
    except Exception as e:
        print(f"Health check FAILED - Unexpected error: {{e}}")
        return 1

if __name__ == "__main__":
    sys.exit(check_health())
"""
    
    def _generate_troubleshooting_guide(self, config: DeploymentConfig) -> str:
        """Generate troubleshooting guide"""
        return f"""# ABOV3 Genesis Troubleshooting Guide

## Common Issues

### 1. Service Won't Start
**Symptoms:** Application fails to start or exits immediately

**Possible Causes:**
- Ollama not running or not accessible
- Port 8000 already in use
- Insufficient permissions
- Missing dependencies

**Solutions:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# Check port availability
netstat -tuln | grep 8000

# Check logs
tail -f logs/application.log

# Verify permissions
ls -la . 
```

### 2. AI Models Not Available
**Symptoms:** "No models available" errors

**Possible Causes:**
- Ollama models not installed
- Ollama service not running
- Network connectivity issues

**Solutions:**
```bash
# List installed models
ollama list

# Install required models
{chr(10).join(f"ollama pull {model}" for model in config.ollama_models) if config.ollama_models else "# No specific models configured"}

# Test Ollama connection
curl -X POST http://localhost:11434/api/generate -d '{{"model": "llama3", "prompt": "test"}}'
```

### 3. High Response Times
**Symptoms:** Slow API responses

**Possible Causes:**
- High system load
- Insufficient resources
- Model loading delays

**Solutions:**
```bash
# Check system resources
top
free -h
df -h

# Monitor application metrics
{f"curl http://localhost:8000/metrics" if config.monitoring_enabled else "# Monitoring not enabled"}

# Optimize configuration
# Edit abov3_config.yaml:
# - Reduce max_concurrent_requests
# - Increase cache_size_mb
```

### 4. SSL/TLS Issues
{f'''**Symptoms:** Certificate errors, HTTPS not working

**Solutions:**
```bash
# Check certificate files
ls -la ssl/

# Test SSL configuration
openssl s_client -connect localhost:443 -servername localhost

# Regenerate certificates if needed
./generate_ssl_cert.sh
```
''' if config.ssl_enabled else '**Note:** SSL/TLS is disabled for this deployment'}

### 5. Memory Issues
**Symptoms:** Out of memory errors, system slowdown

**Solutions:**
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head

# Reduce memory usage:
# - Limit concurrent requests
# - Reduce cache size
# - Monitor model memory usage
```

## Log Analysis

### Application Logs
```bash
# View recent logs
tail -f logs/application.log

# Search for errors
grep -i error logs/application.log

# Filter by component
grep "component=ai_integration" logs/application.log
```

### System Logs
{f'''
```bash
# For systemd deployments
journalctl -u abov3-{config.name} -f

# For Docker deployments
docker logs -f abov3-genesis

# For Kubernetes deployments
kubectl logs -f deployment/abov3-genesis
```
''' if config.target_platform in ['systemd', 'docker', 'kubernetes'] else '# System logs depend on deployment method'}

## Performance Tuning

### Configuration Optimization
```yaml
# abov3_config.yaml optimizations for {config.environment}
application:
  max_concurrent_requests: {config.max_concurrent_requests}
  cache_size_mb: {config.cache_size_mb}
  
performance:
  {'load_balancing: true' if config.load_balancing else '# Load balancing disabled'}
  {'auto_scaling: true' if config.auto_scaling else '# Auto scaling disabled'}
```

### Resource Recommendations
- **CPU:** Minimum 2 cores, recommended 4+ cores
- **Memory:** Minimum 4GB, recommended 8GB+ 
- **Storage:** Minimum 10GB free space
- **Network:** Stable internet connection for model downloads

## Getting Help
1. Check this troubleshooting guide
2. Review application logs
3. Check system resources
4. Verify configuration files
5. Test individual components

For additional support, contact your system administrator.

Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

def create_sample_configs():
    """Create sample deployment configurations"""
    configs = [
        DeploymentConfig(
            name="development",
            environment="dev",
            target_platform="standalone",
            ollama_models=["llama3", "codellama"],
            security_level="minimal",
            monitoring_enabled=True,
            ssl_enabled=False,
            log_level="DEBUG"
        ),
        DeploymentConfig(
            name="production",
            environment="production",
            target_platform="docker",
            ollama_models=["llama3", "codellama", "mistral"],
            security_level="high",
            monitoring_enabled=True,
            ssl_enabled=True,
            load_balancing=True,
            auto_scaling=True,
            log_level="INFO",
            max_concurrent_requests=200
        ),
        DeploymentConfig(
            name="airgapped",
            environment="production",
            target_platform="kubernetes",
            ollama_models=["llama3"],
            security_level="airgapped",
            monitoring_enabled=True,
            ssl_enabled=True,
            load_balancing=True,
            backup_enabled=True,
            custom_settings={
                "network_access": False,
                "offline_mode": True
            }
        )
    ]
    return configs

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="ABOV3 Genesis Enterprise Deployment")
    parser.add_argument("--config", help="Deployment configuration file")
    parser.add_argument("--create-samples", action="store_true", help="Create sample configurations")
    parser.add_argument("--list-platforms", action="store_true", help="List supported platforms")
    
    args = parser.parse_args()
    
    project_root = Path.cwd()
    deployment_manager = EnterpriseDeploymentManager(project_root)
    
    if args.list_platforms:
        print("Supported deployment platforms:")
        for platform in deployment_manager.supported_platforms:
            print(f"  - {platform}")
        return 0
    
    if args.create_samples:
        print("Creating sample deployment configurations...")
        configs = create_sample_configs()
        
        for config in configs:
            config_file = deployment_manager.configs_dir / f"{config.name}_{config.environment}_config.yaml"
            config_data = {
                'name': config.name,
                'environment': config.environment,
                'target_platform': config.target_platform,
                'ollama_models': config.ollama_models,
                'security_level': config.security_level,
                'monitoring_enabled': config.monitoring_enabled,
                'load_balancing': config.load_balancing,
                'auto_scaling': config.auto_scaling,
                'backup_enabled': config.backup_enabled,
                'ssl_enabled': config.ssl_enabled,
                'log_level': config.log_level,
                'max_concurrent_requests': config.max_concurrent_requests,
                'cache_size_mb': config.cache_size_mb,
                'custom_settings': config.custom_settings
            }
            
            config_file.write_text(yaml.dump(config_data, default_flow_style=False))
            print(f"Created: {config_file}")
        
        print(f"Sample configurations created in: {deployment_manager.configs_dir}")
        return 0
    
    if args.config:
        # Load configuration and create deployment package
        config_file = Path(args.config)
        if not config_file.exists():
            print(f"Configuration file not found: {config_file}")
            return 1
        
        try:
            with open(config_file) as f:
                config_data = yaml.safe_load(f)
            
            config = DeploymentConfig(**config_data)
            result = deployment_manager.create_deployment_package(config)
            
            if result["status"] == "success":
                print(f"Deployment package created successfully!")
                print(f"Package location: {result['package_path']}")
                print(f"Deployment ID: {result['deployment_id']}")
                print("Artifacts created:")
                for artifact_type, path in result["artifacts"].items():
                    print(f"  - {artifact_type}: {path}")
            else:
                print(f"Deployment package creation failed: {result['error']}")
                return 1
                
        except Exception as e:
            print(f"Error processing configuration: {e}")
            return 1
    else:
        print("No configuration specified. Use --config to specify deployment configuration.")
        print("Use --create-samples to create sample configurations.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())