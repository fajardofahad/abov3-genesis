"""
ABOV3 Genesis - Natural Language Processor
Core processor for analyzing and understanding natural language feature descriptions
"""

import asyncio
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class FeatureRequirement:
    """Represents a parsed feature requirement"""
    name: str
    description: str
    priority: str  # 'high', 'medium', 'low'
    category: str  # 'ui', 'backend', 'database', 'api', 'auth', 'business_logic'
    complexity: str  # 'simple', 'moderate', 'complex'
    dependencies: List[str]
    estimated_effort: int  # in hours
    technical_specs: Dict[str, Any]
    acceptance_criteria: List[str]


@dataclass
class TechnicalContext:
    """Technical context for the project"""
    project_type: str  # 'web', 'mobile', 'api', 'desktop', 'cli'
    tech_stack: List[str]
    architecture: str  # 'monolith', 'microservices', 'serverless'
    framework: Optional[str]
    database: Optional[str]
    deployment_target: Optional[str]
    performance_requirements: Dict[str, Any]
    security_requirements: List[str]


class NaturalLanguageProcessor:
    """
    Advanced natural language processor for converting feature descriptions
    into structured technical requirements and implementation plans
    """
    
    def __init__(self, ollama_client=None):
        self.ollama_client = ollama_client
        self.feature_patterns = self._initialize_patterns()
        self.tech_stack_mapping = self._initialize_tech_mapping()
        self.complexity_indicators = self._initialize_complexity_indicators()
        
    def _initialize_patterns(self) -> Dict[str, List[str]]:
        """Initialize regex patterns for feature detection"""
        return {
            'authentication': [
                r'(?:user\s+)?(?:login|signin|authentication|auth)',
                r'register|signup|registration',
                r'password\s+reset|forgot\s+password',
                r'user\s+management|user\s+accounts'
            ],
            'database': [
                r'(?:store|save|persist)\s+(?:data|information)',
                r'database|db|storage',
                r'crud\s+operations',
                r'data\s+model|entity|table'
            ],
            'api': [
                r'api|rest|endpoint|service',
                r'http\s+(?:get|post|put|delete)',
                r'json\s+response|xml\s+response',
                r'microservice|backend\s+service'
            ],
            'ui': [
                r'user\s+interface|ui|frontend|webpage|website',
                r'form|button|menu|navigation|navbar',
                r'responsive|mobile\s+friendly',
                r'dashboard|admin\s+panel'
            ],
            'business_logic': [
                r'business\s+(?:logic|rules|process)',
                r'workflow|automation',
                r'calculation|algorithm|processing',
                r'validation|verification'
            ],
            'integration': [
                r'integrate\s+with|connect\s+to|third\s+party',
                r'external\s+api|webhook',
                r'payment\s+(?:gateway|processing)|stripe|paypal',
                r'email\s+sending|notifications'
            ],
            'security': [
                r'security|secure|encryption|hash',
                r'authorization|permissions|roles',
                r'ssl|https|certificate',
                r'data\s+protection|privacy'
            ],
            'performance': [
                r'performance|optimization|speed|fast',
                r'caching|cache|redis|memcached',
                r'load\s+balancing|scaling|scale',
                r'database\s+indexing|query\s+optimization'
            ]
        }
    
    def _initialize_tech_mapping(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize technology stack mappings"""
        return {
            'web': {
                'frontend': ['react', 'vue', 'angular', 'html', 'css', 'javascript'],
                'backend': ['node.js', 'express', 'python', 'django', 'flask', 'fastapi'],
                'database': ['mongodb', 'postgresql', 'mysql', 'sqlite'],
                'deployment': ['vercel', 'netlify', 'heroku', 'aws', 'docker']
            },
            'mobile': {
                'native': ['swift', 'kotlin', 'java'],
                'cross_platform': ['react native', 'flutter', 'xamarin'],
                'backend': ['firebase', 'supabase', 'aws amplify']
            },
            'api': {
                'frameworks': ['fastapi', 'express', 'flask', 'django-rest'],
                'database': ['postgresql', 'mongodb', 'redis'],
                'documentation': ['swagger', 'openapi'],
                'testing': ['pytest', 'jest', 'postman']
            },
            'desktop': {
                'frameworks': ['electron', 'tauri', 'tkinter', 'qt', 'javafx'],
                'languages': ['python', 'javascript', 'java', 'c++', 'rust']
            }
        }
    
    def _initialize_complexity_indicators(self) -> Dict[str, Dict[str, int]]:
        """Initialize complexity scoring indicators"""
        return {
            'simple': {
                'keywords': ['simple', 'basic', 'minimal', 'quick', 'easy'],
                'score_range': (1, 3),
                'effort_hours': (2, 8)
            },
            'moderate': {
                'keywords': ['moderate', 'standard', 'typical', 'normal', 'complete'],
                'score_range': (4, 6),
                'effort_hours': (8, 24)
            },
            'complex': {
                'keywords': ['complex', 'advanced', 'enterprise', 'scalable', 'production'],
                'score_range': (7, 10),
                'effort_hours': (24, 80)
            }
        }
    
    async def analyze_requirements(
        self, 
        description: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[FeatureRequirement], TechnicalContext]:
        """
        Analyze natural language description and extract structured requirements
        """
        logger.info(f"Analyzing requirements for: {description[:100]}...")
        
        # Extract technical context
        tech_context = await self._extract_technical_context(description, context)
        
        # Parse feature requirements
        features = await self._parse_features(description, tech_context)
        
        # Enhance with AI analysis if available
        if self.ollama_client:
            features = await self._enhance_with_ai_analysis(description, features, tech_context)
        
        # Validate and refine requirements
        features = self._validate_and_refine_requirements(features, tech_context)
        
        logger.info(f"Extracted {len(features)} feature requirements")
        return features, tech_context
    
    async def _extract_technical_context(
        self, 
        description: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> TechnicalContext:
        """Extract technical context from description"""
        
        # Determine project type
        project_type = self._determine_project_type(description)
        
        # Extract technology preferences
        tech_stack = self._extract_tech_stack(description, project_type)
        
        # Determine architecture
        architecture = self._determine_architecture(description)
        
        # Extract performance and security requirements
        performance_reqs = self._extract_performance_requirements(description)
        security_reqs = self._extract_security_requirements(description)
        
        return TechnicalContext(
            project_type=project_type,
            tech_stack=tech_stack,
            architecture=architecture,
            framework=self._suggest_framework(project_type, tech_stack),
            database=self._suggest_database(description, tech_stack),
            deployment_target=self._suggest_deployment(description, project_type),
            performance_requirements=performance_reqs,
            security_requirements=security_reqs
        )
    
    def _determine_project_type(self, description: str) -> str:
        """Determine the type of project from description"""
        description_lower = description.lower()
        
        # Check for specific project type indicators
        if any(word in description_lower for word in ['website', 'web app', 'web application', 'html', 'css', 'browser']):
            return 'web'
        elif any(word in description_lower for word in ['mobile app', 'ios', 'android', 'phone', 'mobile']):
            return 'mobile'
        elif any(word in description_lower for word in ['api', 'rest', 'microservice', 'backend', 'service']):
            return 'api'
        elif any(word in description_lower for word in ['desktop', 'gui', 'window', 'application']):
            return 'desktop'
        elif any(word in description_lower for word in ['cli', 'command line', 'terminal', 'script']):
            return 'cli'
        else:
            # Default to web for general descriptions
            return 'web'
    
    def _extract_tech_stack(self, description: str, project_type: str) -> List[str]:
        """Extract mentioned technologies from description"""
        tech_stack = []
        description_lower = description.lower()
        
        # Common technology patterns
        tech_patterns = {
            'python': r'\bpython\b',
            'javascript': r'\b(?:javascript|js)\b',
            'typescript': r'\btypescript\b',
            'react': r'\breact\b',
            'vue': r'\bvue(?:\.js)?\b',
            'angular': r'\bangular\b',
            'node.js': r'\bnode(?:\.js)?\b',
            'express': r'\bexpress\b',
            'django': r'\bdjango\b',
            'flask': r'\bflask\b',
            'fastapi': r'\bfastapi\b',
            'mongodb': r'\bmongodb?\b',
            'postgresql': r'\bpostgres(?:ql)?\b',
            'mysql': r'\bmysql\b',
            'redis': r'\bredis\b'
        }
        
        for tech, pattern in tech_patterns.items():
            if re.search(pattern, description_lower):
                tech_stack.append(tech)
        
        # If no specific tech mentioned, suggest defaults based on project type
        if not tech_stack:
            defaults = self.tech_stack_mapping.get(project_type, {})
            if defaults:
                # Add default technologies
                tech_stack.extend(defaults.get('frontend', [])[:1])  # One frontend
                tech_stack.extend(defaults.get('backend', [])[:1])   # One backend
                tech_stack.extend(defaults.get('database', [])[:1])  # One database
        
        return tech_stack
    
    def _determine_architecture(self, description: str) -> str:
        """Determine architecture pattern from description"""
        description_lower = description.lower()
        
        if any(word in description_lower for word in ['microservice', 'micro-service', 'distributed', 'scalable']):
            return 'microservices'
        elif any(word in description_lower for word in ['serverless', 'lambda', 'function']):
            return 'serverless'
        else:
            return 'monolith'
    
    async def _parse_features(
        self, 
        description: str, 
        tech_context: TechnicalContext
    ) -> List[FeatureRequirement]:
        """Parse individual features from description"""
        
        features = []
        
        # Split description into sentences for feature extraction
        sentences = self._split_into_logical_units(description)
        
        for sentence in sentences:
            feature = self._extract_feature_from_sentence(sentence, tech_context)
            if feature:
                features.append(feature)
        
        # Add implicit features based on project type
        implicit_features = self._add_implicit_features(description, tech_context)
        features.extend(implicit_features)
        
        # Deduplicate and merge similar features
        features = self._deduplicate_features(features)
        
        return features
    
    def _split_into_logical_units(self, description: str) -> List[str]:
        """Split description into logical units for feature extraction"""
        
        # Split by common delimiters
        units = re.split(r'[.!?]\s+|[,;]\s+(?:and|also|plus|additionally|furthermore)\s+', description)
        
        # Clean and filter
        units = [unit.strip() for unit in units if unit.strip() and len(unit.strip()) > 10]
        
        return units
    
    def _extract_feature_from_sentence(
        self, 
        sentence: str, 
        tech_context: TechnicalContext
    ) -> Optional[FeatureRequirement]:
        """Extract a feature requirement from a sentence"""
        
        sentence_lower = sentence.lower()
        
        # Try to match against known patterns
        for category, patterns in self.feature_patterns.items():
            for pattern in patterns:
                if re.search(pattern, sentence_lower):
                    return self._create_feature_requirement(
                        sentence, category, tech_context
                    )
        
        # Check for action verbs that might indicate features
        action_verbs = ['create', 'build', 'make', 'add', 'implement', 'develop', 'design']
        if any(verb in sentence_lower for verb in action_verbs):
            return self._create_feature_requirement(
                sentence, 'business_logic', tech_context
            )
        
        return None
    
    def _create_feature_requirement(
        self, 
        sentence: str, 
        category: str, 
        tech_context: TechnicalContext
    ) -> FeatureRequirement:
        """Create a structured feature requirement"""
        
        # Generate feature name
        name = self._generate_feature_name(sentence, category)
        
        # Determine complexity and effort
        complexity = self._assess_complexity(sentence)
        effort = self._estimate_effort(sentence, complexity, category)
        
        # Extract dependencies
        dependencies = self._extract_dependencies(sentence, category)
        
        # Generate acceptance criteria
        acceptance_criteria = self._generate_acceptance_criteria(sentence, category)
        
        # Build technical specifications
        technical_specs = self._build_technical_specs(sentence, category, tech_context)
        
        return FeatureRequirement(
            name=name,
            description=sentence.strip(),
            priority=self._determine_priority(sentence),
            category=category,
            complexity=complexity,
            dependencies=dependencies,
            estimated_effort=effort,
            technical_specs=technical_specs,
            acceptance_criteria=acceptance_criteria
        )
    
    def _generate_feature_name(self, sentence: str, category: str) -> str:
        """Generate a concise feature name"""
        
        # Extract key nouns and verbs
        words = sentence.lower().split()
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        key_words = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Take first few meaningful words
        name_words = key_words[:3]
        
        # Capitalize and join
        name = ' '.join(word.capitalize() for word in name_words)
        
        # Add category prefix if helpful
        if category == 'authentication':
            name = f"Auth: {name}"
        elif category == 'api':
            name = f"API: {name}"
        elif category == 'ui':
            name = f"UI: {name}"
        
        return name or f"{category.capitalize()} Feature"
    
    def _assess_complexity(self, sentence: str) -> str:
        """Assess the complexity of a feature from its description"""
        
        sentence_lower = sentence.lower()
        
        # Check for complexity indicators
        for complexity, indicators in self.complexity_indicators.items():
            if any(keyword in sentence_lower for keyword in indicators['keywords']):
                return complexity
        
        # Heuristic based on sentence length and technical terms
        tech_terms = ['database', 'api', 'integration', 'algorithm', 'security', 'performance']
        tech_count = sum(1 for term in tech_terms if term in sentence_lower)
        
        if len(sentence) > 100 or tech_count >= 2:
            return 'complex'
        elif len(sentence) > 50 or tech_count >= 1:
            return 'moderate'
        else:
            return 'simple'
    
    def _estimate_effort(self, sentence: str, complexity: str, category: str) -> int:
        """Estimate effort in hours for a feature"""
        
        base_efforts = {
            'authentication': {'simple': 4, 'moderate': 12, 'complex': 24},
            'database': {'simple': 3, 'moderate': 8, 'complex': 16},
            'api': {'simple': 2, 'moderate': 6, 'complex': 12},
            'ui': {'simple': 3, 'moderate': 8, 'complex': 20},
            'business_logic': {'simple': 4, 'moderate': 10, 'complex': 24},
            'integration': {'simple': 6, 'moderate': 16, 'complex': 32},
            'security': {'simple': 4, 'moderate': 12, 'complex': 24},
            'performance': {'simple': 3, 'moderate': 8, 'complex': 20}
        }
        
        return base_efforts.get(category, {}).get(complexity, 8)
    
    def _determine_priority(self, sentence: str) -> str:
        """Determine feature priority from description"""
        
        sentence_lower = sentence.lower()
        
        high_priority_indicators = ['critical', 'essential', 'must', 'required', 'important']
        low_priority_indicators = ['nice to have', 'optional', 'future', 'maybe', 'possibly']
        
        if any(indicator in sentence_lower for indicator in high_priority_indicators):
            return 'high'
        elif any(indicator in sentence_lower for indicator in low_priority_indicators):
            return 'low'
        else:
            return 'medium'
    
    def _extract_dependencies(self, sentence: str, category: str) -> List[str]:
        """Extract feature dependencies"""
        
        dependencies = []
        
        # Category-based dependencies
        if category == 'api' and 'database' in sentence.lower():
            dependencies.append('database_setup')
        
        if category == 'ui' and any(term in sentence.lower() for term in ['login', 'auth', 'user']):
            dependencies.append('authentication')
        
        if 'payment' in sentence.lower():
            dependencies.extend(['authentication', 'database_setup'])
        
        return dependencies
    
    def _generate_acceptance_criteria(self, sentence: str, category: str) -> List[str]:
        """Generate acceptance criteria for the feature"""
        
        criteria = []
        
        # Category-based criteria templates
        if category == 'authentication':
            criteria = [
                "User can register with valid credentials",
                "User can login with correct username/password",
                "Invalid login attempts are handled gracefully",
                "User sessions are managed securely"
            ]
        elif category == 'api':
            criteria = [
                "API endpoints return correct status codes",
                "Request/response format matches specification",
                "Error handling is implemented",
                "API is properly documented"
            ]
        elif category == 'ui':
            criteria = [
                "UI is responsive across different screen sizes",
                "User interactions provide appropriate feedback",
                "Accessibility standards are met",
                "Design matches requirements"
            ]
        elif category == 'database':
            criteria = [
                "Data is stored and retrieved correctly",
                "Database schema supports required operations",
                "Data integrity is maintained",
                "Performance is acceptable"
            ]
        else:
            # Generic criteria
            criteria = [
                "Feature works as described",
                "Error cases are handled appropriately",
                "Performance meets requirements",
                "Code is well-tested"
            ]
        
        return criteria
    
    def _build_technical_specs(
        self, 
        sentence: str, 
        category: str, 
        tech_context: TechnicalContext
    ) -> Dict[str, Any]:
        """Build technical specifications for the feature"""
        
        specs = {
            'category': category,
            'tech_stack': tech_context.tech_stack,
            'framework': tech_context.framework,
            'database': tech_context.database
        }
        
        # Category-specific technical details
        if category == 'api':
            specs.update({
                'endpoints': self._extract_api_endpoints(sentence),
                'methods': ['GET', 'POST', 'PUT', 'DELETE'],
                'authentication_required': 'auth' in sentence.lower() or 'user' in sentence.lower(),
                'data_format': 'json'
            })
        
        elif category == 'database':
            specs.update({
                'tables': self._extract_data_entities(sentence),
                'relationships': self._infer_relationships(sentence),
                'indexing_needed': 'search' in sentence.lower() or 'query' in sentence.lower()
            })
        
        elif category == 'ui':
            specs.update({
                'components': self._extract_ui_components(sentence),
                'responsive': True,
                'accessibility': True,
                'framework': tech_context.framework or 'react'
            })
        
        return specs
    
    def _extract_api_endpoints(self, sentence: str) -> List[str]:
        """Extract potential API endpoints from sentence"""
        endpoints = []
        
        # Look for resource names
        resources = re.findall(r'\b(?:user|product|order|item|post|comment|message)s?\b', sentence.lower())
        
        for resource in set(resources):
            base = resource.rstrip('s')  # Remove plural
            endpoints.extend([
                f"/{base}s",
                f"/{base}s/{{id}}",
                f"/{base}s/create",
                f"/{base}s/{{id}}/update"
            ])
        
        return endpoints
    
    def _extract_data_entities(self, sentence: str) -> List[str]:
        """Extract data entities/tables from sentence"""
        entities = []
        
        # Common entity patterns
        entity_patterns = [
            r'\b(?:user|customer|client)s?\b',
            r'\b(?:product|item|article)s?\b',
            r'\b(?:order|purchase|transaction)s?\b',
            r'\b(?:post|article|blog)s?\b',
            r'\b(?:comment|review|rating)s?\b'
        ]
        
        for pattern in entity_patterns:
            matches = re.findall(pattern, sentence.lower())
            entities.extend([match.rstrip('s') for match in matches])
        
        return list(set(entities))
    
    def _infer_relationships(self, sentence: str) -> List[Dict[str, str]]:
        """Infer data relationships from sentence"""
        relationships = []
        
        # Simple relationship patterns
        if 'user' in sentence.lower() and 'order' in sentence.lower():
            relationships.append({
                'from': 'user',
                'to': 'order',
                'type': 'one_to_many'
            })
        
        if 'product' in sentence.lower() and 'category' in sentence.lower():
            relationships.append({
                'from': 'category',
                'to': 'product',
                'type': 'one_to_many'
            })
        
        return relationships
    
    def _extract_ui_components(self, sentence: str) -> List[str]:
        """Extract UI components from sentence"""
        components = []
        
        component_patterns = {
            'form': r'\b(?:form|input|register|login|signup)\b',
            'table': r'\b(?:table|list|grid|data)\b',
            'button': r'\b(?:button|click|submit|save)\b',
            'navigation': r'\b(?:menu|nav|navigation|link)\b',
            'modal': r'\b(?:modal|popup|dialog)\b',
            'chart': r'\b(?:chart|graph|visualization)\b'
        }
        
        for component, pattern in component_patterns.items():
            if re.search(pattern, sentence.lower()):
                components.append(component)
        
        return components
    
    def _add_implicit_features(
        self, 
        description: str, 
        tech_context: TechnicalContext
    ) -> List[FeatureRequirement]:
        """Add implicit features based on project requirements"""
        
        implicit_features = []
        description_lower = description.lower()
        
        # Add basic project setup features
        if tech_context.project_type == 'web':
            if any(term in description_lower for term in ['user', 'login', 'account', 'register']):
                implicit_features.append(
                    FeatureRequirement(
                        name="Project Setup",
                        description="Initialize project structure and dependencies",
                        priority="high",
                        category="setup",
                        complexity="simple",
                        dependencies=[],
                        estimated_effort=2,
                        technical_specs={
                            'framework': tech_context.framework,
                            'structure': 'standard_web_app'
                        },
                        acceptance_criteria=[
                            "Project structure is created",
                            "Dependencies are installed",
                            "Basic configuration is in place"
                        ]
                    )
                )
        
        # Add database setup if data storage is needed
        if any(term in description_lower for term in ['store', 'save', 'database', 'data', 'user']):
            implicit_features.append(
                FeatureRequirement(
                    name="Database Setup",
                    description="Set up database and basic schema",
                    priority="high",
                    category="database",
                    complexity="moderate",
                    dependencies=["Project Setup"],
                    estimated_effort=4,
                    technical_specs={
                        'database': tech_context.database,
                        'orm': True,
                        'migrations': True
                    },
                    acceptance_criteria=[
                        "Database is configured and connected",
                        "Basic schema is created",
                        "ORM is set up and working"
                    ]
                )
            )
        
        return implicit_features
    
    def _deduplicate_features(self, features: List[FeatureRequirement]) -> List[FeatureRequirement]:
        """Remove duplicate features and merge similar ones"""
        
        unique_features = []
        seen_names = set()
        
        for feature in features:
            # Simple deduplication by name similarity
            similar_exists = False
            for existing_name in seen_names:
                similarity = self._calculate_similarity(feature.name.lower(), existing_name.lower())
                if similarity > 0.8:  # 80% similarity threshold
                    similar_exists = True
                    break
            
            if not similar_exists:
                unique_features.append(feature)
                seen_names.add(feature.name.lower())
        
        return unique_features
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _validate_and_refine_requirements(
        self, 
        features: List[FeatureRequirement], 
        tech_context: TechnicalContext
    ) -> List[FeatureRequirement]:
        """Validate and refine the extracted requirements"""
        
        refined_features = []
        
        for feature in features:
            # Validate technical specifications match tech context
            if tech_context.tech_stack:
                feature.technical_specs['validated_tech_stack'] = tech_context.tech_stack
            
            # Adjust effort estimates based on complexity and category
            if feature.complexity == 'complex' and feature.category in ['integration', 'security']:
                feature.estimated_effort = int(feature.estimated_effort * 1.5)
            
            # Add security considerations if needed
            if any(term in feature.description.lower() for term in ['user', 'login', 'data', 'payment']):
                feature.technical_specs['security_considerations'] = [
                    'Input validation',
                    'Data encryption',
                    'Secure authentication'
                ]
            
            refined_features.append(feature)
        
        # Sort by priority and dependencies
        return self._sort_features_by_priority_and_dependencies(refined_features)
    
    def _sort_features_by_priority_and_dependencies(
        self, 
        features: List[FeatureRequirement]
    ) -> List[FeatureRequirement]:
        """Sort features by priority and dependency order"""
        
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        
        # Simple topological sort based on dependencies
        sorted_features = []
        remaining = features.copy()
        
        while remaining:
            # Find features with no unmet dependencies
            ready_features = []
            for feature in remaining:
                dependencies_met = all(
                    any(dep.lower() in existing.name.lower() or dep.lower() in existing.category.lower() 
                        for existing in sorted_features)
                    for dep in feature.dependencies
                )
                if dependencies_met or not feature.dependencies:
                    ready_features.append(feature)
            
            if not ready_features:
                # If no features are ready, just take the first one to avoid infinite loop
                ready_features = [remaining[0]]
            
            # Sort ready features by priority
            ready_features.sort(key=lambda x: priority_order.get(x.priority, 1))
            
            # Add to sorted list and remove from remaining
            sorted_features.extend(ready_features)
            for feature in ready_features:
                remaining.remove(feature)
        
        return sorted_features
    
    async def _enhance_with_ai_analysis(
        self, 
        description: str, 
        features: List[FeatureRequirement], 
        tech_context: TechnicalContext
    ) -> List[FeatureRequirement]:
        """Enhance requirements analysis using AI"""
        
        if not self.ollama_client:
            return features
        
        try:
            # Prepare AI prompt for requirement analysis
            features_json = [asdict(f) for f in features]
            
            prompt = f"""
            Analyze this project description and refine the extracted requirements:
            
            Project Description: {description}
            
            Current Requirements: {json.dumps(features_json, indent=2)}
            
            Technical Context: {asdict(tech_context)}
            
            Please:
            1. Identify any missing critical features
            2. Refine effort estimates based on complexity
            3. Suggest improvements to technical specifications
            4. Add any missing dependencies
            5. Improve acceptance criteria
            
            Respond with refined requirements in JSON format.
            """
            
            # Get AI analysis
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
            
            # Parse AI response and enhance features
            enhanced_features = self._parse_ai_enhanced_requirements(ai_response, features)
            return enhanced_features
            
        except Exception as e:
            logger.warning(f"AI enhancement failed: {e}")
            return features
    
    def _parse_ai_enhanced_requirements(
        self, 
        ai_response: str, 
        original_features: List[FeatureRequirement]
    ) -> List[FeatureRequirement]:
        """Parse AI-enhanced requirements"""
        
        try:
            # Try to extract JSON from AI response
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                enhanced_data = json.loads(json_match.group())
                
                # Convert back to FeatureRequirement objects
                enhanced_features = []
                for item in enhanced_data.get('features', enhanced_data):
                    if isinstance(item, dict):
                        enhanced_features.append(FeatureRequirement(**item))
                
                if enhanced_features:
                    return enhanced_features
            
        except Exception as e:
            logger.warning(f"Failed to parse AI enhanced requirements: {e}")
        
        return original_features
    
    def _suggest_framework(self, project_type: str, tech_stack: List[str]) -> Optional[str]:
        """Suggest appropriate framework based on context"""
        
        if 'react' in tech_stack:
            return 'react'
        elif 'vue' in tech_stack:
            return 'vue'
        elif 'angular' in tech_stack:
            return 'angular'
        elif 'django' in tech_stack:
            return 'django'
        elif 'flask' in tech_stack:
            return 'flask'
        elif 'fastapi' in tech_stack:
            return 'fastapi'
        elif 'express' in tech_stack:
            return 'express'
        
        # Default suggestions by project type
        defaults = {
            'web': 'react',
            'api': 'fastapi',
            'mobile': 'react-native',
            'desktop': 'electron'
        }
        
        return defaults.get(project_type)
    
    def _suggest_database(self, description: str, tech_stack: List[str]) -> Optional[str]:
        """Suggest appropriate database"""
        
        description_lower = description.lower()
        
        # Check for explicit database mentions in tech stack
        for tech in tech_stack:
            if tech in ['mongodb', 'postgresql', 'mysql', 'sqlite', 'redis']:
                return tech
        
        # Suggest based on project characteristics
        if any(term in description_lower for term in ['document', 'json', 'flexible', 'nosql']):
            return 'mongodb'
        elif any(term in description_lower for term in ['relational', 'sql', 'complex queries']):
            return 'postgresql'
        elif 'simple' in description_lower or 'prototype' in description_lower:
            return 'sqlite'
        else:
            return 'postgresql'  # Safe default
    
    def _suggest_deployment(self, description: str, project_type: str) -> Optional[str]:
        """Suggest deployment target"""
        
        description_lower = description.lower()
        
        if any(term in description_lower for term in ['vercel', 'netlify', 'heroku', 'aws', 'cloud']):
            # Extract mentioned platform
            for platform in ['vercel', 'netlify', 'heroku', 'aws']:
                if platform in description_lower:
                    return platform
        
        # Default suggestions
        if project_type == 'web':
            return 'vercel'
        elif project_type == 'api':
            return 'heroku'
        else:
            return 'docker'
    
    def _extract_performance_requirements(self, description: str) -> Dict[str, Any]:
        """Extract performance requirements from description"""
        
        requirements = {}
        description_lower = description.lower()
        
        if 'fast' in description_lower or 'performance' in description_lower:
            requirements['response_time'] = 'fast'
            requirements['optimization_needed'] = True
        
        if 'scale' in description_lower or 'many users' in description_lower:
            requirements['scalability'] = 'high'
            requirements['load_balancing'] = True
        
        if 'real-time' in description_lower or 'live' in description_lower:
            requirements['real_time'] = True
            requirements['websockets'] = True
        
        return requirements
    
    def _extract_security_requirements(self, description: str) -> List[str]:
        """Extract security requirements from description"""
        
        requirements = []
        description_lower = description.lower()
        
        if any(term in description_lower for term in ['user', 'login', 'account', 'auth']):
            requirements.extend(['authentication', 'authorization', 'session_management'])
        
        if any(term in description_lower for term in ['payment', 'credit card', 'billing']):
            requirements.extend(['pci_compliance', 'data_encryption', 'secure_transmission'])
        
        if any(term in description_lower for term in ['personal data', 'privacy', 'gdpr']):
            requirements.extend(['data_privacy', 'gdpr_compliance', 'data_anonymization'])
        
        if 'admin' in description_lower or 'management' in description_lower:
            requirements.extend(['role_based_access', 'audit_logging'])
        
        return list(set(requirements))  # Remove duplicates
    
    def export_requirements(
        self, 
        features: List[FeatureRequirement], 
        tech_context: TechnicalContext,
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Export requirements to structured format"""
        
        export_data = {
            'project_analysis': {
                'timestamp': datetime.now().isoformat(),
                'technical_context': asdict(tech_context),
                'total_features': len(features),
                'total_estimated_hours': sum(f.estimated_effort for f in features),
                'complexity_breakdown': {
                    'simple': len([f for f in features if f.complexity == 'simple']),
                    'moderate': len([f for f in features if f.complexity == 'moderate']),
                    'complex': len([f for f in features if f.complexity == 'complex'])
                }
            },
            'features': [asdict(f) for f in features]
        }
        
        # Save to file if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Requirements exported to: {output_path}")
        
        return export_data