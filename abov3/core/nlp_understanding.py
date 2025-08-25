"""
ABOV3 Genesis - Advanced Natural Language Understanding for Complex Coding Requests
Implements sophisticated NLP to understand and decompose complex coding tasks
"""

import asyncio
import json
import re
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import logging
from enum import Enum
from collections import defaultdict, Counter
import hashlib
import spacy
from transformers import pipeline, AutoTokenizer, AutoModel
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class IntentType(Enum):
    """Types of coding intents"""
    CODE_GENERATION = "code_generation"
    CODE_EXPLANATION = "code_explanation"
    CODE_DEBUG = "code_debug"
    CODE_OPTIMIZATION = "code_optimization"
    CODE_REFACTORING = "code_refactoring"
    ARCHITECTURE_DESIGN = "architecture_design"
    API_DESIGN = "api_design"
    DATABASE_DESIGN = "database_design"
    TESTING_STRATEGY = "testing_strategy"
    DEPLOYMENT_SETUP = "deployment_setup"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    SECURITY_REVIEW = "security_review"
    DOCUMENTATION = "documentation"
    TUTORIAL_CREATION = "tutorial_creation"
    PROBLEM_SOLVING = "problem_solving"

class ComplexityLevel(Enum):
    """Complexity levels for requests"""
    SIMPLE = "simple"          # Single function, basic task
    MODERATE = "moderate"      # Multiple components, some complexity
    COMPLEX = "complex"        # System-level, multiple technologies
    ENTERPRISE = "enterprise"  # Large-scale, enterprise patterns

class TechnologyDomain(Enum):
    """Technology domains"""
    WEB_FRONTEND = "web_frontend"
    WEB_BACKEND = "web_backend"
    MOBILE_DEVELOPMENT = "mobile_development"
    DATA_SCIENCE = "data_science"
    MACHINE_LEARNING = "machine_learning"
    DEVOPS = "devops"
    SYSTEM_PROGRAMMING = "system_programming"
    DATABASE = "database"
    NETWORKING = "networking"
    SECURITY = "security"
    GAME_DEVELOPMENT = "game_development"
    EMBEDDED = "embedded"
    BLOCKCHAIN = "blockchain"
    CLOUD_COMPUTING = "cloud_computing"

@dataclass
class EntityExtraction:
    """Extracted entities from request"""
    technologies: Set[str] = field(default_factory=set)
    programming_languages: Set[str] = field(default_factory=set)
    frameworks: Set[str] = field(default_factory=set)
    libraries: Set[str] = field(default_factory=set)
    platforms: Set[str] = field(default_factory=set)
    concepts: Set[str] = field(default_factory=set)
    file_types: Set[str] = field(default_factory=set)
    design_patterns: Set[str] = field(default_factory=set)
    architectural_patterns: Set[str] = field(default_factory=set)

@dataclass
class RequestDecomposition:
    """Decomposition of a complex request into subtasks"""
    main_task: str
    subtasks: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    estimated_complexity: Dict[str, str] = field(default_factory=dict)
    required_expertise: Dict[str, str] = field(default_factory=dict)

@dataclass
class NLUResult:
    """Complete NLU analysis result"""
    original_request: str
    intent: IntentType
    complexity_level: ComplexityLevel
    confidence: float
    entities: EntityExtraction
    decomposition: RequestDecomposition
    context_requirements: Dict[str, Any] = field(default_factory=dict)
    clarifying_questions: List[str] = field(default_factory=list)
    suggested_approach: List[str] = field(default_factory=list)
    technology_recommendations: Dict[str, List[str]] = field(default_factory=dict)
    estimated_time: Dict[str, str] = field(default_factory=dict)
    risk_factors: List[str] = field(default_factory=list)

class AdvancedNLPEngine:
    """Advanced NLP engine for understanding complex coding requests"""
    
    def __init__(self):
        # Initialize NLP models
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, using basic tokenization")
            self.nlp = None
        
        # Technology knowledge base
        self.tech_knowledge = TechnologyKnowledgeBase()
        
        # Intent classification
        self.intent_classifier = IntentClassifier()
        
        # Entity extraction
        self.entity_extractor = TechnicalEntityExtractor()
        
        # Task decomposition
        self.task_decomposer = TaskDecomposer()
        
        # Complexity analyzer
        self.complexity_analyzer = ComplexityAnalyzer()
        
        # Context analyzer
        self.context_analyzer = ContextAnalyzer()
        
        # Pattern recognition
        self.pattern_recognizer = CodingPatternRecognizer()
        
        # Cache for performance
        self.analysis_cache = {}
        self.cache_ttl = 3600  # 1 hour
    
    async def analyze_request(self, request: str, context: Dict[str, Any] = None) -> NLUResult:
        """Comprehensive analysis of a coding request"""
        
        start_time = time.time()
        context = context or {}
        
        # Check cache first
        cache_key = self._create_cache_key(request, context)
        if cache_key in self.analysis_cache:
            cache_entry = self.analysis_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                logger.debug("Returning cached NLU analysis")
                return cache_entry['result']
        
        # Preprocess request
        preprocessed_request = await self._preprocess_request(request)
        
        # Parallel analysis for better performance
        analysis_tasks = [
            self.intent_classifier.classify_intent(preprocessed_request, context),
            self.entity_extractor.extract_entities(preprocessed_request, context),
            self.complexity_analyzer.analyze_complexity(preprocessed_request, context),
            self.context_analyzer.analyze_context_requirements(preprocessed_request, context),
            self.pattern_recognizer.identify_patterns(preprocessed_request, context)
        ]
        
        results = await asyncio.gather(*analysis_tasks)
        intent_result, entities, complexity_info, context_requirements, patterns = results
        
        # Decompose task based on analysis
        decomposition = await self.task_decomposer.decompose_task(
            preprocessed_request, intent_result, entities, complexity_info
        )
        
        # Generate clarifying questions
        clarifying_questions = await self._generate_clarifying_questions(
            preprocessed_request, intent_result, entities, decomposition
        )
        
        # Suggest approach
        suggested_approach = await self._suggest_approach(
            intent_result, entities, complexity_info, decomposition
        )
        
        # Technology recommendations
        tech_recommendations = await self._recommend_technologies(
            intent_result, entities, complexity_info
        )
        
        # Risk assessment
        risk_factors = await self._assess_risks(
            intent_result, entities, complexity_info, decomposition
        )
        
        # Estimate time requirements
        time_estimates = await self._estimate_time_requirements(
            decomposition, complexity_info
        )
        
        # Create comprehensive result
        result = NLUResult(
            original_request=request,
            intent=intent_result['intent'],
            complexity_level=complexity_info['level'],
            confidence=intent_result['confidence'],
            entities=entities,
            decomposition=decomposition,
            context_requirements=context_requirements,
            clarifying_questions=clarifying_questions,
            suggested_approach=suggested_approach,
            technology_recommendations=tech_recommendations,
            estimated_time=time_estimates,
            risk_factors=risk_factors
        )
        
        # Cache result
        self.analysis_cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
        
        logger.debug(f"NLU analysis completed in {time.time() - start_time:.3f}s")
        return result
    
    async def _preprocess_request(self, request: str) -> str:
        """Preprocess the request for better analysis"""
        
        # Clean up the request
        preprocessed = request.strip()
        
        # Normalize whitespace
        preprocessed = re.sub(r'\s+', ' ', preprocessed)
        
        # Expand common abbreviations
        abbreviations = {
            'API': 'Application Programming Interface',
            'UI': 'User Interface',
            'UX': 'User Experience',
            'DB': 'Database',
            'AI': 'Artificial Intelligence',
            'ML': 'Machine Learning',
            'CI/CD': 'Continuous Integration Continuous Deployment',
            'CRUD': 'Create Read Update Delete',
            'REST': 'Representational State Transfer',
            'JWT': 'JSON Web Token',
            'OAuth': 'Open Authorization',
            'SQL': 'Structured Query Language',
            'NoSQL': 'Not Only SQL',
            'ORM': 'Object Relational Mapping',
            'MVC': 'Model View Controller',
            'MVP': 'Model View Presenter',
            'MVVM': 'Model View ViewModel'
        }
        
        for abbr, expansion in abbreviations.items():
            preprocessed = re.sub(r'\b' + abbr + r'\b', expansion, preprocessed, flags=re.IGNORECASE)
        
        return preprocessed
    
    def _create_cache_key(self, request: str, context: Dict[str, Any]) -> str:
        """Create cache key for request and context"""
        context_str = json.dumps(context, sort_keys=True) if context else ""
        combined = f"{request}|{context_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    async def _generate_clarifying_questions(
        self,
        request: str,
        intent_result: Dict[str, Any],
        entities: EntityExtraction,
        decomposition: RequestDecomposition
    ) -> List[str]:
        """Generate clarifying questions for ambiguous requests"""
        
        questions = []
        
        # Check for missing technology specifications
        if intent_result['intent'] == IntentType.CODE_GENERATION and not entities.programming_languages:
            questions.append("Which programming language would you prefer for this implementation?")
        
        # Check for missing architectural details
        if intent_result['intent'] == IntentType.ARCHITECTURE_DESIGN and not entities.architectural_patterns:
            questions.append("Do you have any specific architectural patterns in mind (e.g., microservices, monolith, event-driven)?")
        
        # Check for scale requirements
        if "scale" not in request.lower() and "user" not in request.lower():
            questions.append("What scale are you expecting (number of users, data volume, etc.)?")
        
        # Check for performance requirements
        if intent_result['intent'] in [IntentType.CODE_OPTIMIZATION, IntentType.PERFORMANCE_ANALYSIS]:
            if "performance" not in request.lower() and "speed" not in request.lower():
                questions.append("What are your specific performance requirements or bottlenecks?")
        
        # Check for deployment context
        if intent_result['intent'] == IntentType.DEPLOYMENT_SETUP and not entities.platforms:
            questions.append("Which deployment platform or environment are you targeting?")
        
        # Check for testing requirements
        if len(decomposition.subtasks) > 3 and not any("test" in task.get('description', '').lower() for task in decomposition.subtasks):
            questions.append("Do you need automated testing as part of this implementation?")
        
        return questions[:5]  # Limit to 5 most important questions
    
    async def _suggest_approach(
        self,
        intent_result: Dict[str, Any],
        entities: EntityExtraction,
        complexity_info: Dict[str, Any],
        decomposition: RequestDecomposition
    ) -> List[str]:
        """Suggest implementation approach"""
        
        approach = []
        
        intent = intent_result['intent']
        complexity = complexity_info['level']
        
        if intent == IntentType.CODE_GENERATION:
            if complexity == ComplexityLevel.SIMPLE:
                approach.extend([
                    "Start with a simple, single-file implementation",
                    "Focus on core functionality first",
                    "Add error handling and validation",
                    "Include basic testing"
                ])
            elif complexity == ComplexityLevel.COMPLEX:
                approach.extend([
                    "Design the overall architecture first",
                    "Break down into modular components",
                    "Implement core business logic",
                    "Add infrastructure and configuration",
                    "Implement comprehensive testing strategy",
                    "Plan for deployment and monitoring"
                ])
        
        elif intent == IntentType.ARCHITECTURE_DESIGN:
            approach.extend([
                "Analyze requirements and constraints",
                "Design high-level system architecture",
                "Define component interfaces and APIs",
                "Plan data flow and storage strategy",
                "Consider scalability and performance",
                "Document architecture decisions"
            ])
        
        elif intent == IntentType.CODE_DEBUG:
            approach.extend([
                "Reproduce the issue consistently",
                "Add comprehensive logging",
                "Use debugging tools and profilers",
                "Test hypotheses systematically",
                "Fix root cause, not symptoms",
                "Add regression tests"
            ])
        
        return approach
    
    async def _recommend_technologies(
        self,
        intent_result: Dict[str, Any],
        entities: EntityExtraction,
        complexity_info: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Recommend appropriate technologies"""
        
        recommendations = defaultdict(list)
        intent = intent_result['intent']
        complexity = complexity_info['level']
        
        # Programming language recommendations
        if not entities.programming_languages:
            if intent == IntentType.WEB_FRONTEND:
                recommendations['languages'] = ['JavaScript', 'TypeScript']
            elif intent == IntentType.DATA_SCIENCE:
                recommendations['languages'] = ['Python', 'R']
            elif intent == IntentType.SYSTEM_PROGRAMMING:
                recommendations['languages'] = ['Rust', 'Go', 'C++']
            else:
                recommendations['languages'] = ['Python', 'JavaScript', 'Java']
        
        # Framework recommendations
        if intent == IntentType.CODE_GENERATION:
            if 'web' in intent_result.get('context', {}).get('domain', ''):
                if 'Python' in entities.programming_languages:
                    recommendations['frameworks'] = ['FastAPI', 'Django', 'Flask']
                elif 'JavaScript' in entities.programming_languages:
                    recommendations['frameworks'] = ['React', 'Vue.js', 'Express.js']
        
        # Database recommendations
        if intent in [IntentType.DATABASE_DESIGN, IntentType.API_DESIGN]:
            if complexity in [ComplexityLevel.SIMPLE, ComplexityLevel.MODERATE]:
                recommendations['databases'] = ['PostgreSQL', 'SQLite']
            else:
                recommendations['databases'] = ['PostgreSQL', 'MongoDB', 'Redis']
        
        # Tool recommendations
        if intent == IntentType.TESTING_STRATEGY:
            if 'Python' in entities.programming_languages:
                recommendations['testing_tools'] = ['pytest', 'unittest', 'coverage.py']
            elif 'JavaScript' in entities.programming_languages:
                recommendations['testing_tools'] = ['Jest', 'Mocha', 'Cypress']
        
        return dict(recommendations)
    
    async def _assess_risks(
        self,
        intent_result: Dict[str, Any],
        entities: EntityExtraction,
        complexity_info: Dict[str, Any],
        decomposition: RequestDecomposition
    ) -> List[str]:
        """Assess potential risks and challenges"""
        
        risks = []
        
        complexity = complexity_info['level']
        intent = intent_result['intent']
        
        # Complexity-based risks
        if complexity == ComplexityLevel.ENTERPRISE:
            risks.extend([
                "High complexity may lead to longer development time",
                "Integration challenges with existing systems",
                "Scalability bottlenecks if not properly designed",
                "Maintenance complexity over time"
            ])
        
        # Technology-specific risks
        if 'microservices' in entities.architectural_patterns:
            risks.append("Microservices complexity: distributed system challenges, network latency, data consistency")
        
        if not entities.programming_languages and intent == IntentType.CODE_GENERATION:
            risks.append("Technology choice ambiguity may lead to suboptimal selection")
        
        # Security risks
        if intent in [IntentType.API_DESIGN, IntentType.WEB_BACKEND]:
            risks.extend([
                "Security vulnerabilities if authentication/authorization not properly implemented",
                "Data exposure risks without proper input validation"
            ])
        
        # Performance risks
        if complexity in [ComplexityLevel.COMPLEX, ComplexityLevel.ENTERPRISE]:
            risks.extend([
                "Performance bottlenecks under high load",
                "Database query optimization challenges"
            ])
        
        return risks[:8]  # Limit to most critical risks
    
    async def _estimate_time_requirements(
        self,
        decomposition: RequestDecomposition,
        complexity_info: Dict[str, Any]
    ) -> Dict[str, str]:
        """Estimate time requirements for different phases"""
        
        estimates = {}
        complexity = complexity_info['level']
        subtask_count = len(decomposition.subtasks)
        
        # Base estimates by complexity
        base_estimates = {
            ComplexityLevel.SIMPLE: {"planning": "0.5-1 day", "development": "1-3 days", "testing": "0.5-1 day"},
            ComplexityLevel.MODERATE: {"planning": "1-2 days", "development": "3-7 days", "testing": "1-2 days"},
            ComplexityLevel.COMPLEX: {"planning": "2-5 days", "development": "1-3 weeks", "testing": "3-7 days"},
            ComplexityLevel.ENTERPRISE: {"planning": "1-2 weeks", "development": "1-3 months", "testing": "1-2 weeks"}
        }
        
        estimates = base_estimates.get(complexity, base_estimates[ComplexityLevel.MODERATE])
        
        # Adjust based on subtask count
        if subtask_count > 10:
            estimates["integration"] = "2-5 days"
            estimates["deployment"] = "1-3 days"
        
        return estimates

class TechnologyKnowledgeBase:
    """Knowledge base of technologies, frameworks, and their relationships"""
    
    def __init__(self):
        self.programming_languages = {
            'python', 'javascript', 'typescript', 'java', 'c++', 'c#', 'go', 'rust',
            'php', 'ruby', 'swift', 'kotlin', 'dart', 'scala', 'clojure', 'elixir'
        }
        
        self.frameworks = {
            'react', 'vue', 'angular', 'svelte', 'django', 'flask', 'fastapi',
            'spring', 'express', 'nest.js', 'rails', 'laravel', 'asp.net'
        }
        
        self.databases = {
            'postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch', 'cassandra',
            'dynamodb', 'sqlite', 'oracle', 'sql server', 'neo4j', 'couchdb'
        }
        
        self.cloud_platforms = {
            'aws', 'azure', 'gcp', 'google cloud', 'heroku', 'vercel', 'netlify',
            'digitalocean', 'linode', 'cloudflare'
        }
        
        self.architectural_patterns = {
            'microservices', 'monolith', 'serverless', 'event-driven', 'mvc', 'mvp',
            'mvvm', 'layered architecture', 'hexagonal architecture', 'clean architecture'
        }
        
        self.design_patterns = {
            'singleton', 'factory', 'observer', 'strategy', 'command', 'adapter',
            'decorator', 'facade', 'proxy', 'chain of responsibility'
        }

class IntentClassifier:
    """Classifies the intent of coding requests"""
    
    def __init__(self):
        self.intent_patterns = {
            IntentType.CODE_GENERATION: [
                r'\b(create|build|implement|develop|write|generate|make)\b',
                r'\b(function|class|module|component|service|api)\b',
                r'\b(application|app|system|program|script)\b'
            ],
            IntentType.CODE_EXPLANATION: [
                r'\b(explain|describe|understand|how does|what does)\b',
                r'\b(documentation|guide|tutorial)\b'
            ],
            IntentType.CODE_DEBUG: [
                r'\b(debug|fix|error|bug|issue|problem|troubleshoot)\b',
                r'\b(not working|failing|broken|crash)\b'
            ],
            IntentType.CODE_OPTIMIZATION: [
                r'\b(optimize|improve|performance|faster|efficient)\b',
                r'\b(slow|bottleneck|memory|cpu)\b'
            ],
            IntentType.ARCHITECTURE_DESIGN: [
                r'\b(architecture|design|structure|organize|plan)\b',
                r'\b(system design|high level|overview)\b'
            ],
            IntentType.TESTING_STRATEGY: [
                r'\b(test|testing|unit test|integration test|e2e)\b',
                r'\b(quality assurance|qa|coverage)\b'
            ]
        }
    
    async def classify_intent(self, request: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Classify the intent of the request"""
        
        request_lower = request.lower()
        intent_scores = {}
        
        # Pattern-based classification
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, request_lower))
                score += matches
            
            if score > 0:
                intent_scores[intent] = score
        
        # Determine primary intent
        if intent_scores:
            primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
            confidence = min(0.95, intent_scores[primary_intent] / 5.0)
        else:
            primary_intent = IntentType.CODE_GENERATION  # Default
            confidence = 0.3
        
        return {
            'intent': primary_intent,
            'confidence': confidence,
            'all_scores': intent_scores
        }

class TechnicalEntityExtractor:
    """Extracts technical entities from requests"""
    
    def __init__(self):
        self.tech_kb = TechnologyKnowledgeBase()
    
    async def extract_entities(self, request: str, context: Dict[str, Any]) -> EntityExtraction:
        """Extract technical entities from request"""
        
        request_lower = request.lower()
        entities = EntityExtraction()
        
        # Extract programming languages
        for lang in self.tech_kb.programming_languages:
            if lang in request_lower:
                entities.programming_languages.add(lang.title())
        
        # Extract frameworks
        for framework in self.tech_kb.frameworks:
            if framework in request_lower:
                entities.frameworks.add(framework.title())
        
        # Extract databases
        for db in self.tech_kb.databases:
            if db in request_lower:
                entities.technologies.add(db.title())
        
        # Extract cloud platforms
        for platform in self.tech_kb.cloud_platforms:
            if platform in request_lower:
                entities.platforms.add(platform.upper())
        
        # Extract architectural patterns
        for pattern in self.tech_kb.architectural_patterns:
            if pattern in request_lower:
                entities.architectural_patterns.add(pattern)
        
        # Extract design patterns
        for pattern in self.tech_kb.design_patterns:
            if pattern in request_lower:
                entities.design_patterns.add(pattern)
        
        # Extract file types
        file_extensions = re.findall(r'\.(\w+)', request)
        entities.file_types.update(file_extensions)
        
        # Extract concepts using NLP if available
        concepts = self._extract_concepts(request)
        entities.concepts.update(concepts)
        
        return entities
    
    def _extract_concepts(self, request: str) -> Set[str]:
        """Extract programming concepts from request"""
        
        concepts = set()
        
        # Common programming concepts
        concept_patterns = {
            'authentication': r'\b(auth|login|signin|signup|register)\b',
            'authorization': r'\b(permission|role|access control)\b',
            'database': r'\b(crud|query|schema|migration)\b',
            'api': r'\b(rest|graphql|endpoint|route)\b',
            'frontend': r'\b(ui|ux|interface|component)\b',
            'backend': r'\b(server|service|logic|processing)\b',
            'testing': r'\b(test|mock|stub|assertion)\b',
            'deployment': r'\b(deploy|ci/cd|docker|container)\b'
        }
        
        request_lower = request.lower()
        for concept, pattern in concept_patterns.items():
            if re.search(pattern, request_lower):
                concepts.add(concept)
        
        return concepts

class TaskDecomposer:
    """Decomposes complex tasks into manageable subtasks"""
    
    async def decompose_task(
        self,
        request: str,
        intent_result: Dict[str, Any],
        entities: EntityExtraction,
        complexity_info: Dict[str, Any]
    ) -> RequestDecomposition:
        """Decompose a complex request into subtasks"""
        
        intent = intent_result['intent']
        complexity = complexity_info['level']
        
        decomposition = RequestDecomposition(main_task=request)
        
        if intent == IntentType.CODE_GENERATION:
            if complexity == ComplexityLevel.SIMPLE:
                decomposition.subtasks = [
                    {"id": "implement", "description": "Implement core functionality", "priority": "high"},
                    {"id": "test", "description": "Add basic tests", "priority": "medium"},
                    {"id": "document", "description": "Add documentation", "priority": "low"}
                ]
            elif complexity in [ComplexityLevel.COMPLEX, ComplexityLevel.ENTERPRISE]:
                decomposition.subtasks = [
                    {"id": "design", "description": "Design system architecture", "priority": "critical"},
                    {"id": "setup", "description": "Set up project structure", "priority": "high"},
                    {"id": "core", "description": "Implement core business logic", "priority": "critical"},
                    {"id": "data", "description": "Implement data layer", "priority": "high"},
                    {"id": "api", "description": "Create API endpoints", "priority": "high"},
                    {"id": "frontend", "description": "Develop user interface", "priority": "medium"},
                    {"id": "integration", "description": "Integrate components", "priority": "high"},
                    {"id": "testing", "description": "Comprehensive testing", "priority": "high"},
                    {"id": "deployment", "description": "Deployment setup", "priority": "medium"},
                    {"id": "monitoring", "description": "Add monitoring and logging", "priority": "low"}
                ]
        
        elif intent == IntentType.ARCHITECTURE_DESIGN:
            decomposition.subtasks = [
                {"id": "requirements", "description": "Analyze requirements", "priority": "critical"},
                {"id": "constraints", "description": "Identify constraints", "priority": "high"},
                {"id": "components", "description": "Define system components", "priority": "critical"},
                {"id": "interfaces", "description": "Design component interfaces", "priority": "high"},
                {"id": "data_flow", "description": "Plan data flow", "priority": "high"},
                {"id": "scalability", "description": "Address scalability concerns", "priority": "medium"},
                {"id": "security", "description": "Security considerations", "priority": "high"},
                {"id": "documentation", "description": "Document architecture", "priority": "medium"}
            ]
        
        # Generate dependencies and execution order
        decomposition.dependencies = self._generate_dependencies(decomposition.subtasks)
        decomposition.execution_order = self._generate_execution_order(
            decomposition.subtasks, decomposition.dependencies
        )
        
        # Estimate complexity for each subtask
        decomposition.estimated_complexity = {
            task['id']: self._estimate_subtask_complexity(task, entities, complexity)
            for task in decomposition.subtasks
        }
        
        return decomposition
    
    def _generate_dependencies(self, subtasks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Generate dependencies between subtasks"""
        
        dependencies = {}
        task_ids = [task['id'] for task in subtasks]
        
        # Common dependency patterns
        dependency_rules = {
            'core': [],
            'design': [],
            'setup': ['design'],
            'data': ['setup'],
            'api': ['core', 'data'],
            'frontend': ['api'],
            'integration': ['frontend', 'api'],
            'testing': ['integration'],
            'deployment': ['testing'],
            'monitoring': ['deployment']
        }
        
        for task_id in task_ids:
            dependencies[task_id] = [
                dep for dep in dependency_rules.get(task_id, [])
                if dep in task_ids
            ]
        
        return dependencies
    
    def _generate_execution_order(
        self, subtasks: List[Dict[str, Any]], dependencies: Dict[str, List[str]]
    ) -> List[str]:
        """Generate optimal execution order considering dependencies"""
        
        # Topological sort
        in_degree = {task['id']: 0 for task in subtasks}
        for task_id, deps in dependencies.items():
            for dep in deps:
                in_degree[task_id] += 1
        
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        execution_order = []
        
        while queue:
            current = queue.pop(0)
            execution_order.append(current)
            
            for task_id, deps in dependencies.items():
                if current in deps:
                    in_degree[task_id] -= 1
                    if in_degree[task_id] == 0:
                        queue.append(task_id)
        
        return execution_order
    
    def _estimate_subtask_complexity(
        self, task: Dict[str, Any], entities: EntityExtraction, overall_complexity: ComplexityLevel
    ) -> str:
        """Estimate complexity of individual subtask"""
        
        task_id = task['id']
        
        # Base complexity mapping
        complexity_mapping = {
            'design': 'high',
            'setup': 'low',
            'core': 'high',
            'data': 'medium',
            'api': 'medium',
            'frontend': 'medium',
            'integration': 'high',
            'testing': 'medium',
            'deployment': 'medium',
            'monitoring': 'low'
        }
        
        base_complexity = complexity_mapping.get(task_id, 'medium')
        
        # Adjust based on overall complexity
        if overall_complexity == ComplexityLevel.ENTERPRISE:
            if base_complexity == 'low':
                base_complexity = 'medium'
            elif base_complexity == 'medium':
                base_complexity = 'high'
        
        return base_complexity

class ComplexityAnalyzer:
    """Analyzes the complexity of coding requests"""
    
    async def analyze_complexity(self, request: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze complexity of the request"""
        
        complexity_indicators = {
            ComplexityLevel.SIMPLE: [
                r'\b(simple|basic|small|quick|single)\b',
                r'\b(function|method|script)\b'
            ],
            ComplexityLevel.MODERATE: [
                r'\b(application|app|system|service)\b',
                r'\b(multiple|several|few)\b'
            ],
            ComplexityLevel.COMPLEX: [
                r'\b(complex|advanced|sophisticated|enterprise)\b',
                r'\b(distributed|microservice|architecture)\b',
                r'\b(scalable|high performance|production)\b'
            ],
            ComplexityLevel.ENTERPRISE: [
                r'\b(enterprise|large scale|mission critical)\b',
                r'\b(thousands|millions|billion)\b',
                r'\b(highly available|fault tolerant)\b'
            ]
        }
        
        request_lower = request.lower()
        scores = {}
        
        for level, patterns in complexity_indicators.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, request_lower))
                score += matches
            scores[level] = score
        
        # Determine complexity level
        if scores[ComplexityLevel.ENTERPRISE] > 0:
            level = ComplexityLevel.ENTERPRISE
        elif scores[ComplexityLevel.COMPLEX] > 0:
            level = ComplexityLevel.COMPLEX
        elif scores[ComplexityLevel.MODERATE] > 0:
            level = ComplexityLevel.MODERATE
        else:
            level = ComplexityLevel.SIMPLE
        
        # Additional complexity factors
        factors = []
        if len(request.split()) > 100:
            factors.append("Long detailed requirements")
        if request.count('and') > 5:
            factors.append("Multiple requirements")
        if any(word in request_lower for word in ['integrate', 'connect', 'sync']):
            factors.append("Integration requirements")
        
        return {
            'level': level,
            'scores': scores,
            'factors': factors,
            'confidence': max(scores.values()) / 5.0 if max(scores.values()) > 0 else 0.5
        }

class ContextAnalyzer:
    """Analyzes context requirements for requests"""
    
    async def analyze_context_requirements(
        self, request: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze what context information is needed"""
        
        requirements = {
            'project_info': False,
            'existing_code': False,
            'environment_details': False,
            'user_requirements': False,
            'performance_constraints': False,
            'security_requirements': False,
            'integration_points': False
        }
        
        request_lower = request.lower()
        
        # Check for different context needs
        if any(word in request_lower for word in ['existing', 'current', 'modify', 'update']):
            requirements['existing_code'] = True
        
        if any(word in request_lower for word in ['deploy', 'environment', 'production', 'staging']):
            requirements['environment_details'] = True
        
        if any(word in request_lower for word in ['user', 'customer', 'client', 'stakeholder']):
            requirements['user_requirements'] = True
        
        if any(word in request_lower for word in ['performance', 'speed', 'latency', 'throughput']):
            requirements['performance_constraints'] = True
        
        if any(word in request_lower for word in ['secure', 'security', 'auth', 'permission']):
            requirements['security_requirements'] = True
        
        if any(word in request_lower for word in ['integrate', 'api', 'service', 'external']):
            requirements['integration_points'] = True
        
        return requirements

class CodingPatternRecognizer:
    """Recognizes common coding patterns and requirements"""
    
    async def identify_patterns(self, request: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Identify coding patterns and architectural needs"""
        
        patterns = {
            'design_patterns': [],
            'architectural_patterns': [],
            'common_requirements': [],
            'anti_patterns_to_avoid': []
        }
        
        request_lower = request.lower()
        
        # Identify design patterns
        design_pattern_indicators = {
            'singleton': ['single instance', 'one instance', 'global access'],
            'factory': ['create objects', 'object creation', 'instantiate'],
            'observer': ['notify', 'event', 'listener', 'subscribe'],
            'strategy': ['algorithm', 'different ways', 'interchangeable'],
            'decorator': ['extend functionality', 'wrap', 'enhance']
        }
        
        for pattern, indicators in design_pattern_indicators.items():
            if any(indicator in request_lower for indicator in indicators):
                patterns['design_patterns'].append(pattern)
        
        # Identify architectural patterns
        if any(word in request_lower for word in ['microservice', 'distributed', 'service']):
            patterns['architectural_patterns'].append('microservices')
        
        if any(word in request_lower for word in ['event', 'message', 'queue', 'async']):
            patterns['architectural_patterns'].append('event-driven')
        
        if any(word in request_lower for word in ['layer', 'tier', 'separation']):
            patterns['architectural_patterns'].append('layered')
        
        # Common requirements
        if any(word in request_lower for word in ['auth', 'login', 'user']):
            patterns['common_requirements'].append('authentication')
        
        if any(word in request_lower for word in ['permission', 'role', 'access']):
            patterns['common_requirements'].append('authorization')
        
        if any(word in request_lower for word in ['database', 'data', 'persist']):
            patterns['common_requirements'].append('data_persistence')
        
        if any(word in request_lower for word in ['api', 'endpoint', 'rest']):
            patterns['common_requirements'].append('api_design')
        
        return patterns

# Factory function
def create_nlp_engine() -> AdvancedNLPEngine:
    """Create and configure advanced NLP engine"""
    return AdvancedNLPEngine()