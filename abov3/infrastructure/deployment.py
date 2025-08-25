"""
ABOV3 Genesis - Deployment Infrastructure
Production deployment configurations, CI/CD pipelines, and containerization
"""

import asyncio
import json
import yaml
import shutil
import tempfile
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
import time
import subprocess
import platform
import os
import aiofiles

logger = logging.getLogger(__name__)

class DeploymentTarget(Enum):
    """Deployment targets"""
    LOCAL = "local"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    CLOUD_RUN = "cloud_run"
    AWS_ECS = "aws_ecs"
    AZURE_CONTAINER = "azure_container"
    HEROKU = "heroku"
    VERCEL = "vercel"

class EnvironmentTier(Enum):
    """Environment tiers"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    name: str
    target: DeploymentTarget
    tier: EnvironmentTier
    image_name: str = "abov3-genesis"
    image_tag: str = "latest"
    port: int = 8000
    replicas: int = 1
    memory_limit: str = "1Gi"
    cpu_limit: str = "500m"
    memory_request: str = "512Mi"
    cpu_request: str = "250m"
    environment_variables: Dict[str, str] = field(default_factory=dict)
    secrets: Dict[str, str] = field(default_factory=dict)
    volumes: List[Dict[str, str]] = field(default_factory=list)
    health_check_path: str = "/health"
    readiness_check_path: str = "/ready"
    startup_timeout: int = 300  # seconds
    registry: Optional[str] = None
    namespace: str = "default"
    domain: Optional[str] = None
    ssl_enabled: bool = True
    auto_scaling: bool = False
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: int = 70

class DockerfileGenerator:
    """
    Generate optimized Dockerfiles for different deployment scenarios
    """

    def __init__(self, project_path: Path):
        self.project_path = project_path

    def generate_dockerfile(
        self,
        tier: EnvironmentTier = EnvironmentTier.PRODUCTION,
        python_version: str = "3.11",
        base_image: str = "python",
        optimize_size: bool = True
    ) -> str:
        """Generate optimized Dockerfile"""
        
        if optimize_size:
            base_image_tag = f"{python_version}-slim"
        else:
            base_image_tag = f"{python_version}"
        
        dockerfile_content = f"""# ABOV3 Genesis - Production Dockerfile
# Generated for {tier.value} environment
# Optimized: {optimize_size}

FROM {base_image}:{base_image_tag}

# Metadata
LABEL maintainer="ABOV3 Genesis Team"
LABEL version="1.0.0"
LABEL description="ABOV3 Genesis - AI-Powered Coding Assistant"
LABEL tier="{tier.value}"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PYTHONPATH=/app \\
    PIP_NO_CACHE_DIR=1 \\
    PIP_DISABLE_PIP_VERSION_CHECK=1 \\
    ABOV3_ENV={tier.value}

# Create non-root user for security
RUN groupadd -r abov3 && useradd -r -g abov3 abov3

# Install system dependencies
"""

        if optimize_size:
            dockerfile_content += """RUN apt-get update && apt-get install -y --no-install-recommends \\
    git \\
    curl \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/* \\
    && apt-get clean
"""
        else:
            dockerfile_content += """RUN apt-get update && apt-get install -y \\
    git \\
    curl \\
    build-essential \\
    vim \\
    htop \\
    && rm -rf /var/lib/apt/lists/*
"""

        dockerfile_content += """
# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-prod.txt* ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \\
    pip install --no-cache-dir -r requirements.txt"""

        if tier == EnvironmentTier.PRODUCTION:
            dockerfile_content += """ && \\
    pip install --no-cache-dir gunicorn uvicorn[standard]"""

        dockerfile_content += """

# Copy application code
COPY . .

# Remove development files in production
"""
        if tier == EnvironmentTier.PRODUCTION:
            dockerfile_content += """RUN rm -rf tests/ *.md docs/ .git/ .pytest_cache/ __pycache__/ *.egg-info/

"""

        dockerfile_content += """# Create necessary directories
RUN mkdir -p /app/.abov3/logs /app/.abov3/cache /app/.abov3/data && \\
    chown -R abov3:abov3 /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Switch to non-root user
USER abov3

# Expose port
EXPOSE 8000

# Set default command
"""
        if tier == EnvironmentTier.DEVELOPMENT:
            dockerfile_content += """CMD ["python", "-m", "abov3.main"]"""
        else:
            dockerfile_content += """CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "abov3.main:app"]"""

        return dockerfile_content

    def generate_dockerignore(self) -> str:
        """Generate .dockerignore file"""
        return """# ABOV3 Genesis - Docker ignore file

# Git
.git/
.gitignore

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Documentation
docs/_build/
*.md
README*

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Temporary files
.tmp/
temp/
*.tmp
*.temp

# Environment files (should be injected)
.env
.env.*

# Development files
Dockerfile.dev
docker-compose.dev.yml
"""

    async def create_dockerfile(
        self,
        output_path: Path,
        tier: EnvironmentTier = EnvironmentTier.PRODUCTION,
        **kwargs
    ):
        """Create Dockerfile"""
        dockerfile_content = self.generate_dockerfile(tier, **kwargs)
        dockerignore_content = self.generate_dockerignore()
        
        # Write Dockerfile
        dockerfile_path = output_path / "Dockerfile"
        async with aiofiles.open(dockerfile_path, 'w') as f:
            await f.write(dockerfile_content)
        
        # Write .dockerignore
        dockerignore_path = output_path / ".dockerignore"
        async with aiofiles.open(dockerignore_path, 'w') as f:
            await f.write(dockerignore_content)
        
        logger.info(f"Created Dockerfile for {tier.value} at {dockerfile_path}")

class KubernetesManifestGenerator:
    """
    Generate Kubernetes deployment manifests
    """

    def __init__(self, project_path: Path):
        self.project_path = project_path

    def generate_deployment_manifest(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Generate Kubernetes Deployment manifest"""
        
        manifest = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': f"{config.name}-deployment",
                'namespace': config.namespace,
                'labels': {
                    'app': config.name,
                    'tier': config.tier.value,
                    'component': 'backend'
                }
            },
            'spec': {
                'replicas': config.replicas,
                'selector': {
                    'matchLabels': {
                        'app': config.name,
                        'tier': config.tier.value
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': config.name,
                            'tier': config.tier.value,
                            'component': 'backend'
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': config.name,
                            'image': f"{config.image_name}:{config.image_tag}",
                            'ports': [{
                                'containerPort': config.port,
                                'protocol': 'TCP'
                            }],
                            'env': [
                                {'name': k, 'value': v} 
                                for k, v in config.environment_variables.items()
                            ],
                            'resources': {
                                'limits': {
                                    'memory': config.memory_limit,
                                    'cpu': config.cpu_limit
                                },
                                'requests': {
                                    'memory': config.memory_request,
                                    'cpu': config.cpu_request
                                }
                            },
                            'livenessProbe': {
                                'httpGet': {
                                    'path': config.health_check_path,
                                    'port': config.port
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10,
                                'timeoutSeconds': 5,
                                'failureThreshold': 3
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': config.readiness_check_path,
                                    'port': config.port
                                },
                                'initialDelaySeconds': 5,
                                'periodSeconds': 5,
                                'timeoutSeconds': 3,
                                'failureThreshold': 3
                            },
                            'startupProbe': {
                                'httpGet': {
                                    'path': config.health_check_path,
                                    'port': config.port
                                },
                                'initialDelaySeconds': 10,
                                'periodSeconds': 10,
                                'timeoutSeconds': 5,
                                'failureThreshold': config.startup_timeout // 10
                            }
                        }],
                        'restartPolicy': 'Always',
                        'terminationGracePeriodSeconds': 30
                    }
                }
            }
        }
        
        # Add secrets if specified
        if config.secrets:
            secrets_env = [
                {
                    'name': k,
                    'valueFrom': {
                        'secretKeyRef': {
                            'name': f"{config.name}-secrets",
                            'key': k
                        }
                    }
                }
                for k in config.secrets.keys()
            ]
            manifest['spec']['template']['spec']['containers'][0]['env'].extend(secrets_env)
        
        # Add volumes if specified
        if config.volumes:
            manifest['spec']['template']['spec']['volumes'] = config.volumes
            manifest['spec']['template']['spec']['containers'][0]['volumeMounts'] = [
                {
                    'name': vol['name'],
                    'mountPath': vol['mountPath']
                }
                for vol in config.volumes
            ]
        
        return manifest

    def generate_service_manifest(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Generate Kubernetes Service manifest"""
        
        return {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': f"{config.name}-service",
                'namespace': config.namespace,
                'labels': {
                    'app': config.name,
                    'tier': config.tier.value
                }
            },
            'spec': {
                'selector': {
                    'app': config.name,
                    'tier': config.tier.value
                },
                'ports': [{
                    'protocol': 'TCP',
                    'port': 80,
                    'targetPort': config.port
                }],
                'type': 'ClusterIP'
            }
        }

    def generate_ingress_manifest(self, config: DeploymentConfig) -> Optional[Dict[str, Any]]:
        """Generate Kubernetes Ingress manifest"""
        
        if not config.domain:
            return None
        
        manifest = {
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'Ingress',
            'metadata': {
                'name': f"{config.name}-ingress",
                'namespace': config.namespace,
                'labels': {
                    'app': config.name,
                    'tier': config.tier.value
                },
                'annotations': {
                    'nginx.ingress.kubernetes.io/rewrite-target': '/',
                    'kubernetes.io/ingress.class': 'nginx'
                }
            },
            'spec': {
                'rules': [{
                    'host': config.domain,
                    'http': {
                        'paths': [{
                            'path': '/',
                            'pathType': 'Prefix',
                            'backend': {
                                'service': {
                                    'name': f"{config.name}-service",
                                    'port': {
                                        'number': 80
                                    }
                                }
                            }
                        }]
                    }
                }]
            }
        }
        
        # Add TLS if SSL is enabled
        if config.ssl_enabled:
            manifest['metadata']['annotations']['cert-manager.io/cluster-issuer'] = 'letsencrypt-prod'
            manifest['spec']['tls'] = [{
                'hosts': [config.domain],
                'secretName': f"{config.name}-tls"
            }]
        
        return manifest

    def generate_hpa_manifest(self, config: DeploymentConfig) -> Optional[Dict[str, Any]]:
        """Generate Horizontal Pod Autoscaler manifest"""
        
        if not config.auto_scaling:
            return None
        
        return {
            'apiVersion': 'autoscaling/v2',
            'kind': 'HorizontalPodAutoscaler',
            'metadata': {
                'name': f"{config.name}-hpa",
                'namespace': config.namespace,
                'labels': {
                    'app': config.name,
                    'tier': config.tier.value
                }
            },
            'spec': {
                'scaleTargetRef': {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'name': f"{config.name}-deployment"
                },
                'minReplicas': config.min_replicas,
                'maxReplicas': config.max_replicas,
                'metrics': [{
                    'type': 'Resource',
                    'resource': {
                        'name': 'cpu',
                        'target': {
                            'type': 'Utilization',
                            'averageUtilization': config.target_cpu_utilization
                        }
                    }
                }]
            }
        }

    def generate_secret_manifest(self, config: DeploymentConfig) -> Optional[Dict[str, Any]]:
        """Generate Kubernetes Secret manifest"""
        
        if not config.secrets:
            return None
        
        return {
            'apiVersion': 'v1',
            'kind': 'Secret',
            'metadata': {
                'name': f"{config.name}-secrets",
                'namespace': config.namespace,
                'labels': {
                    'app': config.name,
                    'tier': config.tier.value
                }
            },
            'type': 'Opaque',
            'stringData': config.secrets
        }

    async def generate_all_manifests(
        self,
        config: DeploymentConfig,
        output_dir: Path
    ) -> List[Path]:
        """Generate all Kubernetes manifests"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_files = []
        
        # Deployment
        deployment = self.generate_deployment_manifest(config)
        deployment_path = output_dir / f"{config.name}-deployment.yaml"
        async with aiofiles.open(deployment_path, 'w') as f:
            await f.write(yaml.dump(deployment, default_flow_style=False))
        generated_files.append(deployment_path)
        
        # Service
        service = self.generate_service_manifest(config)
        service_path = output_dir / f"{config.name}-service.yaml"
        async with aiofiles.open(service_path, 'w') as f:
            await f.write(yaml.dump(service, default_flow_style=False))
        generated_files.append(service_path)
        
        # Ingress
        ingress = self.generate_ingress_manifest(config)
        if ingress:
            ingress_path = output_dir / f"{config.name}-ingress.yaml"
            async with aiofiles.open(ingress_path, 'w') as f:
                await f.write(yaml.dump(ingress, default_flow_style=False))
            generated_files.append(ingress_path)
        
        # HPA
        hpa = self.generate_hpa_manifest(config)
        if hpa:
            hpa_path = output_dir / f"{config.name}-hpa.yaml"
            async with aiofiles.open(hpa_path, 'w') as f:
                await f.write(yaml.dump(hpa, default_flow_style=False))
            generated_files.append(hpa_path)
        
        # Secrets
        secret = self.generate_secret_manifest(config)
        if secret:
            secret_path = output_dir / f"{config.name}-secrets.yaml"
            async with aiofiles.open(secret_path, 'w') as f:
                await f.write(yaml.dump(secret, default_flow_style=False))
            generated_files.append(secret_path)
        
        logger.info(f"Generated {len(generated_files)} Kubernetes manifests in {output_dir}")
        return generated_files

class CIPipelineGenerator:
    """
    Generate CI/CD pipeline configurations
    """

    def __init__(self, project_path: Path):
        self.project_path = project_path

    def generate_github_actions_workflow(
        self,
        config: DeploymentConfig,
        run_tests: bool = True,
        build_docker: bool = True,
        deploy_k8s: bool = False
    ) -> str:
        """Generate GitHub Actions workflow"""
        
        workflow = f"""# ABOV3 Genesis - CI/CD Pipeline
name: Build and Deploy

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  IMAGE_NAME: {config.image_name}
  IMAGE_TAG: ${{{{ github.sha }}}}
  REGISTRY: ghcr.io
  NAMESPACE: {config.namespace}

jobs:
  test:
    runs-on: ubuntu-latest
    if: {str(run_tests).lower()}
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run linters
      run: |
        flake8 abov3/
        black --check abov3/
        mypy abov3/
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=abov3 --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build:
    runs-on: ubuntu-latest
    if: {str(build_docker).lower()}
    needs: [test]
    
    outputs:
      image: ${{{{ steps.build.outputs.image }}}}
      
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{{{ env.REGISTRY }}}}
        username: ${{{{ github.actor }}}}
        password: ${{{{ secrets.GITHUB_TOKEN }}}}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{{{ env.REGISTRY }}}}/${{{{ github.repository }}}}/${{{{ env.IMAGE_NAME }}}}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix=${{{{github.ref_name}}}}-
          type=raw,value=latest,enable=${{{{github.ref == 'refs/heads/main'}}}}
    
    - name: Build and push
      id: build
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{{{ steps.meta.outputs.tags }}}}
        labels: ${{{{ steps.meta.outputs.labels }}}}
        cache-from: type=gha
        cache-to: type=gha,mode=max
        build-args: |
          BUILDKIT_INLINE_CACHE=1
          TIER={config.tier.value}
"""

        if deploy_k8s:
            workflow += f"""
  deploy:
    runs-on: ubuntu-latest
    if: {str(deploy_k8s).lower()} && github.ref == 'refs/heads/main'
    needs: [build]
    environment: {config.tier.value}
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Configure kubectl
      uses: azure/k8s-set-context@v3
      with:
        method: kubeconfig
        kubeconfig: ${{{{ secrets.KUBECONFIG }}}}
    
    - name: Deploy to Kubernetes
      run: |
        # Update image tag in manifests
        sed -i 's|{config.image_name}:latest|{config.image_name}:${{{{ github.sha }}}}|g' k8s/*.yaml
        
        # Apply manifests
        kubectl apply -f k8s/ -n {config.namespace}
        
        # Wait for rollout
        kubectl rollout status deployment/{config.name}-deployment -n {config.namespace} --timeout=300s
    
    - name: Verify deployment
      run: |
        kubectl get pods -n {config.namespace} -l app={config.name}
        kubectl get svc -n {config.namespace} -l app={config.name}
"""

        workflow += """
  security:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Run security scan
      uses: securecodewarrior/github-action-add@v1
      with:
        sarif-file: 'security-scan-results.sarif'
    
    - name: Upload security scan results
      uses: github/codeql-action/upload-sarif@v3
      if: always()
      with:
        sarif_file: 'security-scan-results.sarif'
"""

        return workflow

    def generate_gitlab_ci_pipeline(self, config: DeploymentConfig) -> str:
        """Generate GitLab CI pipeline"""
        
        return f"""# ABOV3 Genesis - GitLab CI/CD Pipeline
image: docker:latest

variables:
  DOCKER_IMAGE_NAME: {config.image_name}
  DOCKER_TAG: $CI_COMMIT_SHA
  KUBERNETES_NAMESPACE: {config.namespace}

stages:
  - test
  - build
  - deploy
  - verify

before_script:
  - docker info

# Test stage
test:
  stage: test
  image: python:3.11
  before_script:
    - pip install --upgrade pip
    - pip install -r requirements.txt -r requirements-dev.txt
  script:
    - flake8 abov3/
    - black --check abov3/
    - mypy abov3/
    - pytest tests/ -v --cov=abov3
  coverage: '/TOTAL.+?(\\d+%)/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

# Build stage
build:
  stage: build
  services:
    - docker:dind
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - docker build -t $CI_REGISTRY_IMAGE/$DOCKER_IMAGE_NAME:$DOCKER_TAG .
    - docker push $CI_REGISTRY_IMAGE/$DOCKER_IMAGE_NAME:$DOCKER_TAG
  only:
    - main
    - develop

# Deploy to staging
deploy_staging:
  stage: deploy
  image: bitnami/kubectl:latest
  environment:
    name: staging
    url: https://staging.{config.domain or 'example.com'}
  before_script:
    - kubectl config use-context staging
  script:
    - sed -i "s|{config.image_name}:latest|$CI_REGISTRY_IMAGE/$DOCKER_IMAGE_NAME:$DOCKER_TAG|g" k8s/*.yaml
    - kubectl apply -f k8s/ -n $KUBERNETES_NAMESPACE
    - kubectl rollout status deployment/{config.name}-deployment -n $KUBERNETES_NAMESPACE --timeout=300s
  only:
    - develop

# Deploy to production
deploy_production:
  stage: deploy
  image: bitnami/kubectl:latest
  environment:
    name: production
    url: https://{config.domain or 'example.com'}
  before_script:
    - kubectl config use-context production
  script:
    - sed -i "s|{config.image_name}:latest|$CI_REGISTRY_IMAGE/$DOCKER_IMAGE_NAME:$DOCKER_TAG|g" k8s/*.yaml
    - kubectl apply -f k8s/ -n $KUBERNETES_NAMESPACE
    - kubectl rollout status deployment/{config.name}-deployment -n $KUBERNETES_NAMESPACE --timeout=300s
  when: manual
  only:
    - main

# Verify deployment
verify_deployment:
  stage: verify
  script:
    - |
      echo "Verifying deployment health..."
      sleep 30  # Wait for startup
      curl -f https://{config.domain or 'example.com'}/health || exit 1
      echo "Deployment verified successfully!"
  only:
    - main
    - develop
"""

    def generate_jenkins_pipeline(self, config: DeploymentConfig) -> str:
        """Generate Jenkins pipeline (Groovy)"""
        
        return f"""// ABOV3 Genesis - Jenkins Pipeline
pipeline {{
    agent any
    
    environment {{
        IMAGE_NAME = '{config.image_name}'
        IMAGE_TAG = "${{BUILD_NUMBER}}"
        REGISTRY = 'your-registry.com'
        NAMESPACE = '{config.namespace}'
        DEPLOYMENT_NAME = '{config.name}-deployment'
    }}
    
    stages {{
        stage('Checkout') {{
            steps {{
                checkout scm
                script {{
                    env.GIT_COMMIT_SHORT = sh(
                        script: "git rev-parse --short HEAD",
                        returnStdout: true
                    ).trim()
                }}
            }}
        }}
        
        stage('Test') {{
            agent {{
                docker {{
                    image 'python:3.11'
                    args '-v /var/run/docker.sock:/var/run/docker.sock'
                }}
            }}
            steps {{
                sh '''
                    pip install --upgrade pip
                    pip install -r requirements.txt -r requirements-dev.txt
                    
                    # Run linting
                    flake8 abov3/
                    black --check abov3/
                    mypy abov3/
                    
                    # Run tests
                    pytest tests/ -v --cov=abov3 --junit-xml=test-results.xml
                '''
            }}
            post {{
                always {{
                    junit 'test-results.xml'
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: 'htmlcov',
                        reportFiles: 'index.html',
                        reportName: 'Coverage Report'
                    ])
                }}
            }}
        }}
        
        stage('Build') {{
            when {{
                anyOf {{
                    branch 'main'
                    branch 'develop'
                }}
            }}
            steps {{
                script {{
                    def image = docker.build("${{REGISTRY}}/${{IMAGE_NAME}}:${{IMAGE_TAG}}")
                    docker.withRegistry("https://${{REGISTRY}}", 'registry-credentials') {{
                        image.push()
                        image.push('latest')
                    }}
                }}
            }}
        }}
        
        stage('Deploy to Staging') {{
            when {{
                branch 'develop'
            }}
            environment {{
                KUBECONFIG = credentials('staging-kubeconfig')
            }}
            steps {{
                sh '''
                    # Update image tag in manifests
                    sed -i "s|{config.image_name}:latest|$REGISTRY/$IMAGE_NAME:$IMAGE_TAG|g" k8s/*.yaml
                    
                    # Deploy to staging
                    kubectl apply -f k8s/ -n $NAMESPACE
                    kubectl rollout status deployment/$DEPLOYMENT_NAME -n $NAMESPACE --timeout=300s
                '''
            }}
        }}
        
        stage('Deploy to Production') {{
            when {{
                branch 'main'
            }}
            environment {{
                KUBECONFIG = credentials('production-kubeconfig')
            }}
            input {{
                message "Deploy to production?"
                ok "Deploy"
                parameters {{
                    choice(
                        name: 'DEPLOYMENT_STRATEGY',
                        choices: ['rolling', 'blue-green'],
                        description: 'Deployment strategy'
                    )
                }}
            }}
            steps {{
                sh '''
                    # Update image tag in manifests
                    sed -i "s|{config.image_name}:latest|$REGISTRY/$IMAGE_NAME:$IMAGE_TAG|g" k8s/*.yaml
                    
                    # Deploy to production
                    kubectl apply -f k8s/ -n $NAMESPACE
                    kubectl rollout status deployment/$DEPLOYMENT_NAME -n $NAMESPACE --timeout=600s
                '''
            }}
        }}
        
        stage('Verify Deployment') {{
            when {{
                anyOf {{
                    branch 'main'
                    branch 'develop'
                }}
            }}
            steps {{
                script {{
                    def healthUrl = branch == 'main' ? 
                        'https://{config.domain or "example.com"}/health' : 
                        'https://staging.{config.domain or "example.com"}/health'
                    
                    sh '''
                        echo "Waiting for deployment to be ready..."
                        sleep 30
                        
                        echo "Verifying health endpoint..."
                        curl -f $healthUrl || exit 1
                        
                        echo "Deployment verified successfully!"
                    '''
                }}
            }}
        }}
    }}
    
    post {{
        always {{
            cleanWs()
        }}
        failure {{
            emailext (
                subject: "Pipeline Failed: ${{env.JOB_NAME}} - ${{env.BUILD_NUMBER}}",
                body: "Build failed. Check console output at ${{env.BUILD_URL}}",
                to: "${{env.CHANGE_AUTHOR_EMAIL}}"
            )
        }}
        success {{
            when {{
                branch 'main'
            }}
            slackSend (
                color: 'good',
                message: "Successfully deployed ABOV3 Genesis to production! :rocket:"
            )
        }}
    }}
}}
"""

    async def create_pipeline_files(
        self,
        config: DeploymentConfig,
        output_dir: Path,
        platforms: List[str] = None
    ):
        """Create CI/CD pipeline files"""
        
        if platforms is None:
            platforms = ['github', 'gitlab', 'jenkins']
        
        created_files = []
        
        if 'github' in platforms:
            # GitHub Actions workflow
            workflow_dir = output_dir / '.github' / 'workflows'
            workflow_dir.mkdir(parents=True, exist_ok=True)
            
            workflow_content = self.generate_github_actions_workflow(config)
            workflow_path = workflow_dir / 'ci-cd.yml'
            
            async with aiofiles.open(workflow_path, 'w') as f:
                await f.write(workflow_content)
            created_files.append(workflow_path)
        
        if 'gitlab' in platforms:
            # GitLab CI pipeline
            gitlab_ci_content = self.generate_gitlab_ci_pipeline(config)
            gitlab_ci_path = output_dir / '.gitlab-ci.yml'
            
            async with aiofiles.open(gitlab_ci_path, 'w') as f:
                await f.write(gitlab_ci_content)
            created_files.append(gitlab_ci_path)
        
        if 'jenkins' in platforms:
            # Jenkins pipeline
            jenkins_content = self.generate_jenkins_pipeline(config)
            jenkins_path = output_dir / 'Jenkinsfile'
            
            async with aiofiles.open(jenkins_path, 'w') as f:
                await f.write(jenkins_content)
            created_files.append(jenkins_path)
        
        logger.info(f"Created {len(created_files)} CI/CD pipeline files")
        return created_files

class DeploymentManager:
    """
    Main deployment management coordinator
    """

    def __init__(self, project_path: Path):
        self.project_path = project_path
        
        # Initialize generators
        self.dockerfile_generator = DockerfileGenerator(project_path)
        self.k8s_generator = KubernetesManifestGenerator(project_path)
        self.ci_generator = CIPipelineGenerator(project_path)
        
        # Deployment history
        self._deployment_history = []

    async def setup_deployment_environment(
        self,
        config: DeploymentConfig,
        include_ci: bool = True,
        ci_platforms: List[str] = None
    ) -> Dict[str, Any]:
        """Setup complete deployment environment"""
        
        logger.info(f"Setting up deployment environment for {config.name} ({config.tier.value})")
        
        setup_results = {
            'config': config,
            'started_at': time.time(),
            'created_files': []
        }
        
        try:
            # Create deployment directory
            deploy_dir = self.project_path / 'deploy'
            deploy_dir.mkdir(exist_ok=True)
            
            # Create tier-specific directory
            tier_dir = deploy_dir / config.tier.value
            tier_dir.mkdir(exist_ok=True)
            
            # Generate Dockerfile
            await self.dockerfile_generator.create_dockerfile(
                self.project_path,
                tier=config.tier
            )
            setup_results['created_files'].extend(['Dockerfile', '.dockerignore'])
            
            # Generate Kubernetes manifests
            k8s_dir = tier_dir / 'k8s'
            k8s_files = await self.k8s_generator.generate_all_manifests(config, k8s_dir)
            setup_results['created_files'].extend([str(f.relative_to(self.project_path)) for f in k8s_files])
            
            # Generate CI/CD pipelines
            if include_ci:
                ci_files = await self.ci_generator.create_pipeline_files(
                    config,
                    self.project_path,
                    ci_platforms
                )
                setup_results['created_files'].extend([str(f.relative_to(self.project_path)) for f in ci_files])
            
            # Create deployment scripts
            await self._create_deployment_scripts(config, tier_dir)
            setup_results['created_files'].extend([
                f'deploy/{config.tier.value}/deploy.sh',
                f'deploy/{config.tier.value}/undeploy.sh'
            ])
            
            # Create README
            await self._create_deployment_readme(config, tier_dir)
            setup_results['created_files'].append(f'deploy/{config.tier.value}/README.md')
            
            setup_results['success'] = True
            setup_results['completed_at'] = time.time()
            setup_results['duration'] = setup_results['completed_at'] - setup_results['started_at']
            
            # Record deployment setup
            self._deployment_history.append({
                'action': 'setup',
                'config': config.name,
                'tier': config.tier.value,
                'timestamp': time.time(),
                'files_created': len(setup_results['created_files'])
            })
            
            logger.info(f"Deployment environment setup completed: {len(setup_results['created_files'])} files created")
            
            return setup_results
            
        except Exception as e:
            logger.error(f"Deployment environment setup failed: {e}")
            setup_results['success'] = False
            setup_results['error'] = str(e)
            return setup_results

    async def _create_deployment_scripts(self, config: DeploymentConfig, output_dir: Path):
        """Create deployment and cleanup scripts"""
        
        # Deploy script
        deploy_script = f"""#!/bin/bash
# ABOV3 Genesis - Deployment Script
# Environment: {config.tier.value}
# Target: {config.target.value}

set -e

echo "🚀 Deploying ABOV3 Genesis - {config.tier.value}"

# Build and push Docker image (if needed)
if command -v docker &> /dev/null; then
    echo "📦 Building Docker image..."
    docker build -t {config.image_name}:{config.image_tag} .
    
    if [ ! -z "${{REGISTRY}}" ]; then
        echo "📤 Pushing to registry..."
        docker tag {config.image_name}:{config.image_tag} $REGISTRY/{config.image_name}:{config.image_tag}
        docker push $REGISTRY/{config.image_name}:{config.image_tag}
    fi
fi

# Deploy to Kubernetes (if manifests exist)
if [ -d "k8s" ] && command -v kubectl &> /dev/null; then
    echo "☸️  Deploying to Kubernetes..."
    
    # Create namespace if it doesn't exist
    kubectl create namespace {config.namespace} --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply manifests
    kubectl apply -f k8s/ -n {config.namespace}
    
    # Wait for rollout
    echo "⏳ Waiting for deployment to complete..."
    kubectl rollout status deployment/{config.name}-deployment -n {config.namespace} --timeout=300s
    
    # Show status
    echo "📊 Deployment status:"
    kubectl get pods -n {config.namespace} -l app={config.name}
    kubectl get svc -n {config.namespace} -l app={config.name}
    
    # Show access information
    if kubectl get ingress -n {config.namespace} | grep -q {config.name}; then
        echo "🌐 Access URL: https://{config.domain or 'your-domain.com'}"
    fi
fi

echo "✅ Deployment completed successfully!"
"""

        # Undeploy script
        undeploy_script = f"""#!/bin/bash
# ABOV3 Genesis - Cleanup Script
# Environment: {config.tier.value}

set -e

echo "🧹 Cleaning up ABOV3 Genesis - {config.tier.value}"

# Remove from Kubernetes
if [ -d "k8s" ] && command -v kubectl &> /dev/null; then
    echo "☸️  Removing from Kubernetes..."
    kubectl delete -f k8s/ -n {config.namespace} --ignore-not-found=true
    
    # Optionally remove namespace (uncomment if desired)
    # kubectl delete namespace {config.namespace} --ignore-not-found=true
fi

# Remove Docker images (optional)
if command -v docker &> /dev/null; then
    echo "🗑️  Removing Docker images..."
    docker rmi {config.image_name}:{config.image_tag} --force || true
    if [ ! -z "${{REGISTRY}}" ]; then
        docker rmi $REGISTRY/{config.image_name}:{config.image_tag} --force || true
    fi
fi

echo "✅ Cleanup completed!"
"""

        # Write scripts
        deploy_path = output_dir / 'deploy.sh'
        undeploy_path = output_dir / 'undeploy.sh'
        
        async with aiofiles.open(deploy_path, 'w') as f:
            await f.write(deploy_script)
        
        async with aiofiles.open(undeploy_path, 'w') as f:
            await f.write(undeploy_script)
        
        # Make scripts executable (Unix only)
        if platform.system() != 'Windows':
            deploy_path.chmod(0o755)
            undeploy_path.chmod(0o755)

    async def _create_deployment_readme(self, config: DeploymentConfig, output_dir: Path):
        """Create deployment README"""
        
        readme_content = f"""# ABOV3 Genesis - {config.tier.value.title()} Deployment

This directory contains deployment configurations for ABOV3 Genesis {config.tier.value} environment.

## Overview

- **Target**: {config.target.value}
- **Environment**: {config.tier.value}
- **Image**: {config.image_name}:{config.image_tag}
- **Namespace**: {config.namespace}
- **Port**: {config.port}

## Quick Start

### Prerequisites

- Docker installed and configured
- kubectl configured with cluster access
- Sufficient cluster resources:
  - CPU: {config.cpu_request} (request) / {config.cpu_limit} (limit)
  - Memory: {config.memory_request} (request) / {config.memory_limit} (limit)

### Deploy

```bash
# Make sure you're in the {config.tier.value} deployment directory
cd deploy/{config.tier.value}

# Set registry if using external registry
export REGISTRY=your-registry.com

# Run deployment script
./deploy.sh
```

### Verify Deployment

```bash
# Check pod status
kubectl get pods -n {config.namespace} -l app={config.name}

# Check service
kubectl get svc -n {config.namespace} -l app={config.name}

# Check ingress (if configured)
kubectl get ingress -n {config.namespace}

# View logs
kubectl logs -f deployment/{config.name}-deployment -n {config.namespace}

# Health check
curl https://{config.domain or 'your-domain.com'}/health
```

### Access Application

"""

        if config.domain:
            readme_content += f"""- **URL**: https://{config.domain}
- **Health Check**: https://{config.domain}/health
- **Ready Check**: https://{config.domain}/ready
"""
        else:
            readme_content += """- **URL**: Configure ingress with your domain
- **Health Check**: /health endpoint
- **Ready Check**: /ready endpoint
"""

        readme_content += f"""
## Configuration

### Environment Variables

"""
        for key, value in config.environment_variables.items():
            readme_content += f"- `{key}`: {value}\n"

        if config.secrets:
            readme_content += f"""
### Secrets

The following secrets need to be configured:
"""
            for key in config.secrets.keys():
                readme_content += f"- `{key}`: Configure in Kubernetes secret\n"

        readme_content += f"""
### Scaling

- **Replicas**: {config.replicas}"""

        if config.auto_scaling:
            readme_content += f"""
- **Auto-scaling**: Enabled
  - Min replicas: {config.min_replicas}
  - Max replicas: {config.max_replicas}
  - CPU target: {config.target_cpu_utilization}%"""
        else:
            readme_content += """
- **Auto-scaling**: Disabled"""

        readme_content += f"""

## Monitoring

### Health Checks

- **Liveness Probe**: `GET {config.health_check_path}`
- **Readiness Probe**: `GET {config.readiness_check_path}`
- **Startup Probe**: `GET {config.health_check_path}` (timeout: {config.startup_timeout}s)

### Metrics

Application metrics are available at `/metrics` endpoint (if enabled).

### Logs

Application logs are structured JSON and can be collected using:

```bash
# Real-time logs
kubectl logs -f deployment/{config.name}-deployment -n {config.namespace}

# Export logs
kubectl logs deployment/{config.name}-deployment -n {config.namespace} > app.log
```

## Troubleshooting

### Common Issues

1. **Pod not starting**:
   ```bash
   kubectl describe pod -n {config.namespace} -l app={config.name}
   ```

2. **Image pull errors**:
   ```bash
   kubectl get events -n {config.namespace} --sort-by=.metadata.creationTimestamp
   ```

3. **Resource constraints**:
   ```bash
   kubectl top nodes
   kubectl describe node <node-name>
   ```

### Cleanup

To remove the deployment:

```bash
./undeploy.sh
```

## Files

- `k8s/`: Kubernetes manifests
- `deploy.sh`: Deployment script
- `undeploy.sh`: Cleanup script
- `README.md`: This file

## Support

For issues and support, please refer to the main ABOV3 Genesis documentation.
"""

        readme_path = output_dir / 'README.md'
        async with aiofiles.open(readme_path, 'w') as f:
            await f.write(readme_content)

    async def get_deployment_status(self) -> Dict[str, Any]:
        """Get deployment system status"""
        
        return {
            'deployment_history': self._deployment_history,
            'available_targets': [target.value for target in DeploymentTarget],
            'available_tiers': [tier.value for tier in EnvironmentTier],
            'docker_available': shutil.which('docker') is not None,
            'kubectl_available': shutil.which('kubectl') is not None,
            'project_path': str(self.project_path)
        }

# Context manager for deployment
class deployment_context:
    """Context manager for deployment operations"""
    
    def __init__(self, project_path: Path):
        self.manager = DeploymentManager(project_path)

    async def __aenter__(self):
        return self.manager

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup if needed
        pass