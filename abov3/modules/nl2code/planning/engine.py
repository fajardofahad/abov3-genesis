"""
ABOV3 Genesis - Planning Engine
Advanced planning engine for feature decomposition, milestone generation, and implementation strategy
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging

from ..core.processor import FeatureRequirement, TechnicalContext

logger = logging.getLogger(__name__)


@dataclass
class TaskStep:
    """Represents a single implementation step"""
    id: str
    name: str
    description: str
    category: str  # 'setup', 'frontend', 'backend', 'database', 'testing', 'deployment'
    estimated_hours: int
    dependencies: List[str]
    files_to_create: List[str]
    files_to_modify: List[str]
    technical_requirements: Dict[str, Any]
    validation_criteria: List[str]
    priority: int  # 1-10, lower is higher priority


@dataclass
class Milestone:
    """Represents a project milestone"""
    id: str
    name: str
    description: str
    tasks: List[TaskStep]
    estimated_completion: datetime
    deliverables: List[str]
    dependencies: List[str]
    success_criteria: List[str]
    risk_factors: List[str]


@dataclass
class ImplementationPlan:
    """Complete implementation plan"""
    project_name: str
    description: str
    tech_context: TechnicalContext
    milestones: List[Milestone]
    total_estimated_hours: int
    estimated_completion_date: datetime
    critical_path: List[str]
    resource_requirements: Dict[str, Any]
    risk_assessment: Dict[str, Any]


class PlanningEngine:
    """
    Advanced planning engine that converts feature requirements into detailed
    implementation plans with milestones, tasks, and dependencies
    """
    
    def __init__(self, ollama_client=None):
        self.ollama_client = ollama_client
        self.task_templates = self._initialize_task_templates()
        self.milestone_patterns = self._initialize_milestone_patterns()
        
    def _initialize_task_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize task templates for different categories"""
        return {
            'project_setup': {
                'base_hours': 2,
                'files': ['package.json', 'requirements.txt', 'README.md', '.env.example'],
                'validation': ['Project structure is created', 'Dependencies are installable']
            },
            'database_setup': {
                'base_hours': 4,
                'files': ['models.py', 'database.py', 'migrations/', 'schema.sql'],
                'validation': ['Database connects successfully', 'Tables are created']
            },
            'authentication': {
                'base_hours': 8,
                'files': ['auth.py', 'login.html', 'register.html', 'middleware.py'],
                'validation': ['Users can register', 'Users can login', 'Sessions work']
            },
            'api_endpoints': {
                'base_hours': 6,
                'files': ['routes.py', 'controllers.py', 'serializers.py'],
                'validation': ['Endpoints return correct responses', 'Error handling works']
            },
            'frontend_components': {
                'base_hours': 5,
                'files': ['components/', 'pages/', 'styles/'],
                'validation': ['Components render correctly', 'User interactions work']
            },
            'business_logic': {
                'base_hours': 8,
                'files': ['services.py', 'utils.py', 'validators.py'],
                'validation': ['Business rules are enforced', 'Logic works as expected']
            },
            'integration': {
                'base_hours': 12,
                'files': ['integrations/', 'external_apis.py', 'webhooks.py'],
                'validation': ['External APIs connect', 'Data flows correctly']
            },
            'testing': {
                'base_hours': 6,
                'files': ['tests/', 'test_*.py', 'conftest.py'],
                'validation': ['All tests pass', 'Coverage is adequate']
            },
            'deployment': {
                'base_hours': 4,
                'files': ['Dockerfile', 'docker-compose.yml', 'deploy.yml'],
                'validation': ['Application deploys successfully', 'Health checks pass']
            }
        }
    
    def _initialize_milestone_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize milestone patterns for different project types"""
        return {
            'web_application': {
                'milestones': [
                    'Project Foundation',
                    'Core Backend',
                    'Frontend Interface',
                    'Integration & Testing',
                    'Deployment & Launch'
                ],
                'typical_duration': 80  # hours
            },
            'api_service': {
                'milestones': [
                    'API Foundation',
                    'Core Endpoints',
                    'Authentication & Security',
                    'Testing & Documentation',
                    'Deployment'
                ],
                'typical_duration': 50
            },
            'mobile_app': {
                'milestones': [
                    'App Foundation',
                    'Core Features',
                    'UI/UX Implementation',
                    'Backend Integration',
                    'Testing & Deployment'
                ],
                'typical_duration': 120
            }
        }
    
    async def create_implementation_plan(
        self, 
        features: List[FeatureRequirement], 
        tech_context: TechnicalContext,
        project_name: str = "Generated Project",
        target_completion_weeks: int = 4
    ) -> ImplementationPlan:
        """
        Create a comprehensive implementation plan from feature requirements
        """
        logger.info(f"Creating implementation plan for {len(features)} features")
        
        # Analyze project characteristics
        project_characteristics = self._analyze_project_characteristics(features, tech_context)
        
        # Generate task breakdown
        tasks = await self._generate_task_breakdown(features, tech_context)
        
        # Create milestones
        milestones = self._organize_tasks_into_milestones(tasks, tech_context, project_characteristics)
        
        # Calculate timelines
        timelines = self._calculate_timelines(milestones, target_completion_weeks)
        
        # Assess risks and dependencies
        risk_assessment = self._assess_project_risks(features, tech_context, tasks)
        
        # Determine resource requirements
        resources = self._calculate_resource_requirements(tasks, tech_context)
        
        # Build critical path
        critical_path = self._build_critical_path(milestones)
        
        # Enhance with AI analysis if available
        if self.ollama_client:
            plan_data = await self._enhance_plan_with_ai(
                features, tech_context, milestones, tasks
            )
            if plan_data:
                milestones = plan_data.get('milestones', milestones)
                risk_assessment.update(plan_data.get('risks', {}))
        
        return ImplementationPlan(
            project_name=project_name,
            description=self._generate_project_description(features, tech_context),
            tech_context=tech_context,
            milestones=milestones,
            total_estimated_hours=sum(len(m.tasks) for m in milestones) * 6,  # Rough estimate
            estimated_completion_date=datetime.now() + timedelta(weeks=target_completion_weeks),
            critical_path=critical_path,
            resource_requirements=resources,
            risk_assessment=risk_assessment
        )
    
    def _analyze_project_characteristics(
        self, 
        features: List[FeatureRequirement], 
        tech_context: TechnicalContext
    ) -> Dict[str, Any]:
        """Analyze project to determine characteristics and patterns"""
        
        characteristics = {
            'project_type': tech_context.project_type,
            'complexity_level': self._assess_overall_complexity(features),
            'feature_categories': self._categorize_features(features),
            'integration_complexity': self._assess_integration_complexity(features),
            'ui_complexity': self._assess_ui_complexity(features),
            'data_complexity': self._assess_data_complexity(features),
            'authentication_needed': any(f.category == 'authentication' for f in features),
            'external_integrations': len([f for f in features if f.category == 'integration']),
            'estimated_team_size': self._estimate_team_size(features, tech_context)
        }
        
        return characteristics
    
    def _assess_overall_complexity(self, features: List[FeatureRequirement]) -> str:
        """Assess the overall complexity of the project"""
        
        complexity_scores = {'simple': 1, 'moderate': 2, 'complex': 3}
        total_score = sum(complexity_scores.get(f.complexity, 2) for f in features)
        average_complexity = total_score / len(features) if features else 1
        
        if average_complexity <= 1.3:
            return 'simple'
        elif average_complexity <= 2.3:
            return 'moderate'
        else:
            return 'complex'
    
    def _categorize_features(self, features: List[FeatureRequirement]) -> Dict[str, int]:
        """Categorize features by type"""
        categories = {}
        for feature in features:
            categories[feature.category] = categories.get(feature.category, 0) + 1
        return categories
    
    def _assess_integration_complexity(self, features: List[FeatureRequirement]) -> str:
        """Assess integration complexity"""
        integration_features = [f for f in features if f.category == 'integration']
        
        if not integration_features:
            return 'none'
        elif len(integration_features) <= 2:
            return 'simple'
        elif len(integration_features) <= 5:
            return 'moderate'
        else:
            return 'complex'
    
    def _assess_ui_complexity(self, features: List[FeatureRequirement]) -> str:
        """Assess UI complexity"""
        ui_features = [f for f in features if f.category == 'ui']
        
        if not ui_features:
            return 'minimal'
        
        # Check for complex UI patterns
        complex_indicators = ['dashboard', 'admin', 'chart', 'visualization', 'responsive']
        complex_count = sum(1 for f in ui_features 
                           for indicator in complex_indicators 
                           if indicator in f.description.lower())
        
        if complex_count >= 3:
            return 'complex'
        elif complex_count >= 1:
            return 'moderate'
        else:
            return 'simple'
    
    def _assess_data_complexity(self, features: List[FeatureRequirement]) -> str:
        """Assess data model complexity"""
        db_features = [f for f in features if f.category == 'database']
        
        if not db_features:
            return 'minimal'
        
        # Look for relationship indicators
        relationship_indicators = ['foreign key', 'relationship', 'join', 'reference']
        relationship_count = sum(1 for f in db_features 
                               for indicator in relationship_indicators 
                               if indicator in f.description.lower())
        
        if relationship_count >= 3:
            return 'complex'
        elif relationship_count >= 1:
            return 'moderate'
        else:
            return 'simple'
    
    def _estimate_team_size(
        self, 
        features: List[FeatureRequirement], 
        tech_context: TechnicalContext
    ) -> int:
        """Estimate required team size"""
        
        total_hours = sum(f.estimated_effort for f in features)
        
        # Base team size estimation
        if total_hours <= 40:
            return 1  # Solo developer
        elif total_hours <= 120:
            return 2  # Small team
        elif total_hours <= 300:
            return 3  # Medium team
        else:
            return 4  # Larger team
    
    async def _generate_task_breakdown(
        self, 
        features: List[FeatureRequirement], 
        tech_context: TechnicalContext
    ) -> List[TaskStep]:
        """Generate detailed task breakdown from features"""
        
        tasks = []
        task_counter = 1
        
        # Always start with project setup
        setup_task = self._create_project_setup_task(task_counter, tech_context)
        tasks.append(setup_task)
        task_counter += 1
        
        # Generate tasks for each feature
        for feature in features:
            feature_tasks = await self._generate_tasks_for_feature(
                feature, tech_context, task_counter
            )
            tasks.extend(feature_tasks)
            task_counter += len(feature_tasks)
        
        # Add integration and testing tasks
        integration_tasks = self._create_integration_tasks(features, tech_context, task_counter)
        tasks.extend(integration_tasks)
        task_counter += len(integration_tasks)
        
        # Add deployment tasks
        deployment_task = self._create_deployment_task(tech_context, task_counter)
        tasks.append(deployment_task)
        
        # Set up dependencies between tasks
        tasks = self._establish_task_dependencies(tasks, features)
        
        return tasks
    
    def _create_project_setup_task(self, task_id: int, tech_context: TechnicalContext) -> TaskStep:
        """Create project setup task"""
        
        template = self.task_templates['project_setup']
        
        files_to_create = []
        if tech_context.project_type == 'web':
            if 'react' in tech_context.tech_stack or 'vue' in tech_context.tech_stack:
                files_to_create.extend(['package.json', 'src/index.js', 'public/index.html'])
            if 'python' in tech_context.tech_stack:
                files_to_create.extend(['requirements.txt', 'main.py', 'app.py'])
        
        return TaskStep(
            id=f"task_{task_id:03d}",
            name="Project Setup",
            description=f"Initialize {tech_context.project_type} project with {tech_context.framework or 'standard'} framework",
            category="setup",
            estimated_hours=template['base_hours'],
            dependencies=[],
            files_to_create=files_to_create + template['files'],
            files_to_modify=[],
            technical_requirements={
                'framework': tech_context.framework,
                'tech_stack': tech_context.tech_stack,
                'package_manager': self._determine_package_manager(tech_context)
            },
            validation_criteria=template['validation'] + [
                f"{tech_context.framework} is properly configured",
                "Development server can start"
            ],
            priority=1
        )
    
    async def _generate_tasks_for_feature(
        self, 
        feature: FeatureRequirement, 
        tech_context: TechnicalContext, 
        start_id: int
    ) -> List[TaskStep]:
        """Generate tasks for a specific feature"""
        
        tasks = []
        current_id = start_id
        
        # Get template for feature category
        template = self.task_templates.get(feature.category, {
            'base_hours': feature.estimated_effort,
            'files': [],
            'validation': feature.acceptance_criteria
        })
        
        # Create main implementation task
        main_task = TaskStep(
            id=f"task_{current_id:03d}",
            name=f"Implement {feature.name}",
            description=feature.description,
            category=feature.category,
            estimated_hours=feature.estimated_effort,
            dependencies=self._convert_feature_dependencies_to_task_ids(feature.dependencies),
            files_to_create=self._generate_files_for_feature(feature, tech_context),
            files_to_modify=self._generate_files_to_modify_for_feature(feature, tech_context),
            technical_requirements=feature.technical_specs,
            validation_criteria=feature.acceptance_criteria,
            priority=self._convert_priority_to_number(feature.priority)
        )
        tasks.append(main_task)
        current_id += 1
        
        # Create additional tasks for complex features
        if feature.complexity == 'complex':
            # Add testing task
            test_task = TaskStep(
                id=f"task_{current_id:03d}",
                name=f"Test {feature.name}",
                description=f"Create comprehensive tests for {feature.name}",
                category="testing",
                estimated_hours=max(2, feature.estimated_effort // 3),
                dependencies=[main_task.id],
                files_to_create=[f"tests/test_{feature.name.lower().replace(' ', '_')}.py"],
                files_to_modify=[],
                technical_requirements={'testing_framework': self._suggest_testing_framework(tech_context)},
                validation_criteria=[
                    "Unit tests cover main functionality",
                    "Integration tests verify feature works",
                    "All tests pass"
                ],
                priority=main_task.priority + 1
            )
            tasks.append(test_task)
        
        return tasks
    
    def _generate_files_for_feature(
        self, 
        feature: FeatureRequirement, 
        tech_context: TechnicalContext
    ) -> List[str]:
        """Generate list of files to create for a feature"""
        
        files = []
        base_name = feature.name.lower().replace(' ', '_')
        
        if feature.category == 'authentication':
            if 'python' in tech_context.tech_stack:
                files.extend([
                    f"models/user.py",
                    f"auth/{base_name}.py",
                    f"routes/auth.py"
                ])
            elif 'javascript' in tech_context.tech_stack:
                files.extend([
                    f"models/User.js",
                    f"middleware/auth.js",
                    f"routes/auth.js"
                ])
        
        elif feature.category == 'api':
            if 'python' in tech_context.tech_stack:
                files.extend([
                    f"routes/{base_name}.py",
                    f"controllers/{base_name}_controller.py",
                    f"serializers/{base_name}_serializer.py"
                ])
            elif 'javascript' in tech_context.tech_stack:
                files.extend([
                    f"routes/{base_name}.js",
                    f"controllers/{base_name}Controller.js"
                ])
        
        elif feature.category == 'ui':
            if tech_context.framework == 'react':
                files.extend([
                    f"components/{feature.name.replace(' ', '')}.jsx",
                    f"components/{feature.name.replace(' ', '')}.module.css"
                ])
            elif tech_context.framework == 'vue':
                files.extend([
                    f"components/{feature.name.replace(' ', '')}.vue"
                ])
            else:
                files.extend([
                    f"templates/{base_name}.html",
                    f"static/css/{base_name}.css",
                    f"static/js/{base_name}.js"
                ])
        
        elif feature.category == 'database':
            if 'python' in tech_context.tech_stack:
                files.extend([
                    f"models/{base_name}.py",
                    f"migrations/create_{base_name}.py"
                ])
            elif 'javascript' in tech_context.tech_stack:
                files.extend([
                    f"models/{base_name.title()}.js",
                    f"migrations/{datetime.now().strftime('%Y%m%d%H%M%S')}_create_{base_name}.js"
                ])
        
        return files
    
    def _generate_files_to_modify_for_feature(
        self, 
        feature: FeatureRequirement, 
        tech_context: TechnicalContext
    ) -> List[str]:
        """Generate list of files to modify for a feature"""
        
        files = []
        
        # Common files that often need modification
        if feature.category == 'api':
            files.extend(['app.py', 'main.py', 'routes/__init__.py'])
        
        elif feature.category == 'ui':
            files.extend(['App.js', 'App.vue', 'index.html'])
        
        elif feature.category == 'database':
            files.extend(['models/__init__.py', 'database.py'])
        
        return files
    
    def _create_integration_tasks(
        self, 
        features: List[FeatureRequirement], 
        tech_context: TechnicalContext, 
        start_id: int
    ) -> List[TaskStep]:
        """Create integration and testing tasks"""
        
        tasks = []
        current_id = start_id
        
        # Integration testing task
        integration_task = TaskStep(
            id=f"task_{current_id:03d}",
            name="Integration Testing",
            description="Test all features working together",
            category="testing",
            estimated_hours=8,
            dependencies=[f"task_{i:03d}" for i in range(2, start_id)],  # Depends on all feature tasks
            files_to_create=[
                "tests/test_integration.py",
                "tests/conftest.py",
                "tests/fixtures.py"
            ],
            files_to_modify=[],
            technical_requirements={
                'testing_framework': self._suggest_testing_framework(tech_context),
                'test_database': True,
                'mock_external_apis': True
            },
            validation_criteria=[
                "All integration tests pass",
                "End-to-end workflows work correctly",
                "Error handling is comprehensive"
            ],
            priority=8
        )
        tasks.append(integration_task)
        
        return tasks
    
    def _create_deployment_task(self, tech_context: TechnicalContext, task_id: int) -> TaskStep:
        """Create deployment task"""
        
        template = self.task_templates['deployment']
        
        return TaskStep(
            id=f"task_{task_id:03d}",
            name="Production Deployment",
            description=f"Deploy to {tech_context.deployment_target or 'production'}",
            category="deployment",
            estimated_hours=template['base_hours'],
            dependencies=[f"task_{task_id-1:03d}"],  # Depends on integration testing
            files_to_create=template['files'] + [
                ".github/workflows/deploy.yml",
                "docker-compose.prod.yml"
            ],
            files_to_modify=[],
            technical_requirements={
                'deployment_target': tech_context.deployment_target,
                'containerization': True,
                'ci_cd': True,
                'monitoring': True
            },
            validation_criteria=template['validation'] + [
                "Application is accessible from internet",
                "All services are running correctly",
                "Monitoring and logging are active"
            ],
            priority=9
        )
    
    def _establish_task_dependencies(
        self, 
        tasks: List[TaskStep], 
        features: List[FeatureRequirement]
    ) -> List[TaskStep]:
        """Establish dependencies between tasks"""
        
        # Create lookup maps
        task_by_id = {task.id: task for task in tasks}
        
        # Set up logical dependencies
        for task in tasks:
            if task.category == 'api' and any(t.category == 'database' for t in tasks):
                # API tasks depend on database tasks
                db_tasks = [t.id for t in tasks if t.category == 'database']
                task.dependencies.extend(db_tasks)
            
            elif task.category == 'ui' and any(t.category == 'api' for t in tasks):
                # UI tasks depend on API tasks
                api_tasks = [t.id for t in tasks if t.category == 'api']
                task.dependencies.extend(api_tasks)
            
            elif task.category == 'testing':
                # Testing tasks depend on implementation tasks
                impl_tasks = [t.id for t in tasks if t.category not in ['testing', 'deployment']]
                task.dependencies.extend(impl_tasks)
            
            elif task.category == 'deployment':
                # Deployment depends on everything else
                other_tasks = [t.id for t in tasks if t.category != 'deployment']
                task.dependencies.extend(other_tasks)
        
        # Remove duplicates and self-references
        for task in tasks:
            task.dependencies = list(set(task.dependencies))
            if task.id in task.dependencies:
                task.dependencies.remove(task.id)
        
        return tasks
    
    def _organize_tasks_into_milestones(
        self, 
        tasks: List[TaskStep], 
        tech_context: TechnicalContext,
        project_characteristics: Dict[str, Any]
    ) -> List[Milestone]:
        """Organize tasks into logical milestones"""
        
        milestones = []
        
        # Get milestone pattern for project type
        pattern = self.milestone_patterns.get(
            tech_context.project_type + '_application',
            self.milestone_patterns['web_application']
        )
        
        milestone_names = pattern['milestones']
        
        # Group tasks by category and priority
        task_groups = {
            'foundation': [t for t in tasks if t.category in ['setup', 'database']],
            'backend': [t for t in tasks if t.category in ['api', 'business_logic', 'authentication']],
            'frontend': [t for t in tasks if t.category in ['ui']],
            'integration': [t for t in tasks if t.category in ['integration', 'testing']],
            'deployment': [t for t in tasks if t.category == 'deployment']
        }
        
        # Create milestones
        milestone_id = 1
        for i, milestone_name in enumerate(milestone_names):
            milestone_tasks = []
            
            if i == 0:  # Foundation
                milestone_tasks = task_groups['foundation']
            elif i == 1:  # Core Backend/API
                milestone_tasks = task_groups['backend']
            elif i == 2:  # Frontend/Interface
                milestone_tasks = task_groups['frontend']
            elif i == 3:  # Integration & Testing
                milestone_tasks = task_groups['integration']
            elif i == 4:  # Deployment
                milestone_tasks = task_groups['deployment']
            
            if milestone_tasks:
                milestone = Milestone(
                    id=f"milestone_{milestone_id:02d}",
                    name=milestone_name,
                    description=self._generate_milestone_description(milestone_name, milestone_tasks),
                    tasks=milestone_tasks,
                    estimated_completion=datetime.now() + timedelta(weeks=milestone_id),
                    deliverables=self._generate_milestone_deliverables(milestone_tasks),
                    dependencies=[f"milestone_{milestone_id-1:02d}"] if milestone_id > 1 else [],
                    success_criteria=self._generate_milestone_success_criteria(milestone_tasks),
                    risk_factors=self._identify_milestone_risks(milestone_tasks, tech_context)
                )
                milestones.append(milestone)
                milestone_id += 1
        
        return milestones
    
    def _generate_milestone_description(self, name: str, tasks: List[TaskStep]) -> str:
        """Generate milestone description"""
        
        descriptions = {
            'Project Foundation': "Set up the project structure, dependencies, and core infrastructure",
            'Core Backend': "Implement core backend functionality including APIs and business logic", 
            'Frontend Interface': "Build user interface components and user experience",
            'Integration & Testing': "Integrate all components and perform comprehensive testing",
            'Deployment & Launch': "Deploy to production and ensure system is operational"
        }
        
        return descriptions.get(name, f"Complete {len(tasks)} tasks for {name}")
    
    def _generate_milestone_deliverables(self, tasks: List[TaskStep]) -> List[str]:
        """Generate milestone deliverables"""
        
        deliverables = []
        categories = set(task.category for task in tasks)
        
        if 'setup' in categories:
            deliverables.append("Project structure and configuration")
        if 'database' in categories:
            deliverables.append("Database schema and models")
        if 'api' in categories:
            deliverables.append("RESTful API endpoints")
        if 'ui' in categories:
            deliverables.append("User interface components")
        if 'authentication' in categories:
            deliverables.append("Authentication system")
        if 'testing' in categories:
            deliverables.append("Test suite and coverage report")
        if 'deployment' in categories:
            deliverables.append("Production deployment")
        
        return deliverables
    
    def _generate_milestone_success_criteria(self, tasks: List[TaskStep]) -> List[str]:
        """Generate milestone success criteria"""
        
        criteria = []
        
        # Collect all validation criteria from tasks
        all_validations = []
        for task in tasks:
            all_validations.extend(task.validation_criteria)
        
        # Group similar criteria
        if any('test' in v.lower() for v in all_validations):
            criteria.append("All tests pass successfully")
        
        if any('deploy' in v.lower() for v in all_validations):
            criteria.append("System deploys without errors")
        
        if any('user' in v.lower() for v in all_validations):
            criteria.append("User functionality works correctly")
        
        if any('api' in v.lower() for v in all_validations):
            criteria.append("API endpoints respond correctly")
        
        # Add generic criteria
        criteria.extend([
            "All milestone tasks are completed",
            "Quality standards are met",
            "Documentation is updated"
        ])
        
        return list(set(criteria))  # Remove duplicates
    
    def _identify_milestone_risks(
        self, 
        tasks: List[TaskStep], 
        tech_context: TechnicalContext
    ) -> List[str]:
        """Identify potential risks for milestone"""
        
        risks = []
        categories = set(task.category for task in tasks)
        
        if 'integration' in categories:
            risks.append("Integration complexity may cause delays")
            risks.append("External API dependencies may be unreliable")
        
        if 'database' in categories:
            risks.append("Database migration issues")
            risks.append("Data model changes may require refactoring")
        
        if 'authentication' in categories:
            risks.append("Security vulnerabilities if not implemented correctly")
            risks.append("User experience issues with auth flow")
        
        if 'ui' in categories:
            risks.append("Browser compatibility issues")
            risks.append("Responsive design challenges")
        
        if 'deployment' in categories:
            risks.append("Production environment configuration issues")
            risks.append("Performance issues under load")
        
        # Add risks based on tech context
        if tech_context.architecture == 'microservices':
            risks.append("Service communication complexity")
        
        return risks
    
    def _calculate_timelines(
        self, 
        milestones: List[Milestone], 
        target_weeks: int
    ) -> Dict[str, Any]:
        """Calculate realistic timelines for milestones"""
        
        total_hours = sum(sum(task.estimated_hours for task in m.tasks) for m in milestones)
        hours_per_week = 40  # Assuming full-time development
        
        current_date = datetime.now()
        
        for i, milestone in enumerate(milestones):
            milestone_hours = sum(task.estimated_hours for task in milestone.tasks)
            weeks_needed = max(1, milestone_hours / hours_per_week)
            
            milestone.estimated_completion = current_date + timedelta(weeks=weeks_needed)
            current_date = milestone.estimated_completion
        
        return {
            'total_hours': total_hours,
            'total_weeks': (current_date - datetime.now()).days / 7,
            'hours_per_week': hours_per_week
        }
    
    def _assess_project_risks(
        self, 
        features: List[FeatureRequirement], 
        tech_context: TechnicalContext,
        tasks: List[TaskStep]
    ) -> Dict[str, Any]:
        """Assess overall project risks"""
        
        risks = {
            'technical': [],
            'schedule': [],
            'resource': [],
            'external': []
        }
        
        # Technical risks
        if tech_context.architecture == 'microservices':
            risks['technical'].append("Microservice complexity and communication overhead")
        
        complex_features = [f for f in features if f.complexity == 'complex']
        if len(complex_features) > 3:
            risks['technical'].append("High number of complex features increases technical risk")
        
        # Schedule risks
        total_hours = sum(task.estimated_hours for task in tasks)
        if total_hours > 200:
            risks['schedule'].append("Large scope increases schedule risk")
        
        # Resource risks
        if len(set(task.category for task in tasks)) > 5:
            risks['resource'].append("Diverse skill requirements may strain resources")
        
        # External risks
        integration_tasks = [t for t in tasks if t.category == 'integration']
        if integration_tasks:
            risks['external'].append("External API dependencies create delivery risk")
        
        return risks
    
    def _calculate_resource_requirements(
        self, 
        tasks: List[TaskStep], 
        tech_context: TechnicalContext
    ) -> Dict[str, Any]:
        """Calculate resource requirements"""
        
        categories = set(task.category for task in tasks)
        
        skills_needed = []
        if 'ui' in categories:
            skills_needed.append('Frontend Development')
        if any(cat in categories for cat in ['api', 'database', 'business_logic']):
            skills_needed.append('Backend Development')
        if 'database' in categories:
            skills_needed.append('Database Design')
        if 'testing' in categories:
            skills_needed.append('Quality Assurance')
        if 'deployment' in categories:
            skills_needed.append('DevOps/Deployment')
        
        tools_needed = list(set(tech_context.tech_stack))
        if tech_context.database:
            tools_needed.append(tech_context.database)
        if tech_context.deployment_target:
            tools_needed.append(tech_context.deployment_target)
        
        return {
            'skills_needed': skills_needed,
            'tools_needed': tools_needed,
            'estimated_team_size': self._estimate_team_size_from_tasks(tasks),
            'development_environment': self._suggest_dev_environment(tech_context)
        }
    
    def _build_critical_path(self, milestones: List[Milestone]) -> List[str]:
        """Build critical path through milestones and tasks"""
        
        critical_path = []
        
        # Simple critical path - assumes sequential milestone completion
        for milestone in milestones:
            critical_path.append(milestone.id)
            
            # Add critical tasks within milestone
            critical_tasks = sorted(milestone.tasks, key=lambda t: t.priority)[:2]
            critical_path.extend([task.id for task in critical_tasks])
        
        return critical_path
    
    async def _enhance_plan_with_ai(
        self,
        features: List[FeatureRequirement],
        tech_context: TechnicalContext, 
        milestones: List[Milestone],
        tasks: List[TaskStep]
    ) -> Optional[Dict[str, Any]]:
        """Enhance plan with AI analysis"""
        
        if not self.ollama_client:
            return None
        
        try:
            # Prepare prompt for AI enhancement
            plan_summary = {
                'features': [asdict(f) for f in features],
                'tech_context': asdict(tech_context),
                'milestones': [asdict(m) for m in milestones[:2]],  # Limit for token constraints
                'task_count': len(tasks)
            }
            
            prompt = f"""
            Analyze this implementation plan and suggest improvements:
            
            {json.dumps(plan_summary, indent=2)}
            
            Please suggest:
            1. Additional risks we should consider
            2. Missing tasks or dependencies
            3. Timeline adjustments
            4. Resource optimization opportunities
            
            Respond with JSON format suggestions.
            """
            
            messages = [{'role': 'user', 'content': prompt}]
            
            async with self.ollama_client:
                options = self.ollama_client.get_genesis_optimized_options('analysis')
                ai_response = ""
                
                async for chunk in self.ollama_client.chat('llama3:latest', messages, options=options):
                    if 'message' in chunk and chunk['message']['content']:
                        ai_response += chunk['message']['content']
                    elif 'response' in chunk:
                        ai_response += chunk['response']
                    
                    if chunk.get('done', False):
                        break
            
            # Try to parse AI suggestions
            import re
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
        except Exception as e:
            logger.warning(f"AI enhancement failed: {e}")
        
        return None
    
    # Helper methods
    def _convert_feature_dependencies_to_task_ids(self, dependencies: List[str]) -> List[str]:
        """Convert feature dependencies to task IDs"""
        # This would need a more sophisticated mapping in a real implementation
        return []
    
    def _convert_priority_to_number(self, priority: str) -> int:
        """Convert priority string to number"""
        priority_map = {'high': 1, 'medium': 5, 'low': 9}
        return priority_map.get(priority, 5)
    
    def _suggest_testing_framework(self, tech_context: TechnicalContext) -> str:
        """Suggest testing framework based on tech stack"""
        if 'python' in tech_context.tech_stack:
            return 'pytest'
        elif 'javascript' in tech_context.tech_stack:
            return 'jest'
        else:
            return 'unittest'
    
    def _determine_package_manager(self, tech_context: TechnicalContext) -> str:
        """Determine appropriate package manager"""
        if 'javascript' in tech_context.tech_stack:
            return 'npm'
        elif 'python' in tech_context.tech_stack:
            return 'pip'
        else:
            return 'default'
    
    def _estimate_team_size_from_tasks(self, tasks: List[TaskStep]) -> int:
        """Estimate team size from task analysis"""
        total_hours = sum(task.estimated_hours for task in tasks)
        parallel_categories = len(set(task.category for task in tasks))
        
        # Simple estimation
        if total_hours <= 80 and parallel_categories <= 3:
            return 1
        elif total_hours <= 200 and parallel_categories <= 5:
            return 2
        elif total_hours <= 400:
            return 3
        else:
            return 4
    
    def _suggest_dev_environment(self, tech_context: TechnicalContext) -> Dict[str, Any]:
        """Suggest development environment setup"""
        env = {
            'version_control': 'git',
            'ide_recommendations': [],
            'required_tools': []
        }
        
        if 'python' in tech_context.tech_stack:
            env['ide_recommendations'].extend(['VS Code', 'PyCharm'])
            env['required_tools'].extend(['Python 3.8+', 'pip', 'virtualenv'])
        
        if 'javascript' in tech_context.tech_stack:
            env['ide_recommendations'].extend(['VS Code', 'WebStorm'])
            env['required_tools'].extend(['Node.js', 'npm'])
        
        return env
    
    def _generate_project_description(
        self, 
        features: List[FeatureRequirement], 
        tech_context: TechnicalContext
    ) -> str:
        """Generate project description"""
        
        feature_summary = ", ".join([f.name for f in features[:3]])
        if len(features) > 3:
            feature_summary += f" and {len(features) - 3} more features"
        
        return f"A {tech_context.project_type} application with {feature_summary}, built using {tech_context.framework or 'modern'} technology stack."
    
    def export_plan(
        self, 
        plan: ImplementationPlan, 
        output_path: Optional[Path] = None,
        format: str = 'json'
    ) -> Dict[str, Any]:
        """Export implementation plan to file"""
        
        plan_data = asdict(plan)
        
        # Convert datetime objects to strings
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(item) for item in obj]
            return obj
        
        plan_data = convert_datetime(plan_data)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(plan_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Implementation plan exported to: {output_path}")
        
        return plan_data