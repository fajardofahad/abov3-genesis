"""
ABOV3 Genesis - NL2Code Ollama Integration
Integration with existing Ollama optimization system for enhanced AI-powered code generation
"""

import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator
from pathlib import Path
import logging
import json

from ...core.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class NL2CodeOllamaIntegration:
    """
    Integration layer between NL2Code module and ABOV3's Ollama optimization system
    Provides specialized AI capabilities for natural language to code conversion
    """
    
    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        self.ollama_client = ollama_client or OllamaClient()
        self.specialized_models = self._initialize_specialized_models()
        self.prompt_templates = self._initialize_prompt_templates()
        self.optimization_configs = self._initialize_optimization_configs()
        
    def _initialize_specialized_models(self) -> Dict[str, Dict[str, Any]]:
        """Initialize specialized model configurations for different NL2Code tasks"""
        return {
            'requirement_analysis': {
                'preferred_models': [
                    'llama3:latest',
                    'gemma:7b',
                    'mistral:latest'
                ],
                'task_type': 'analysis',
                'context_requirements': {
                    'complexity_level': 'high',
                    'performance_critical': True,
                    'creative_task': False
                }
            },
            'code_generation': {
                'preferred_models': [
                    'codellama:latest',
                    'deepseek-coder:6.7b',
                    'starcoder:latest'
                ],
                'task_type': 'code_generation',
                'context_requirements': {
                    'complexity_level': 'high',
                    'performance_critical': True,
                    'is_completion': False
                }
            },
            'architecture_planning': {
                'preferred_models': [
                    'llama3:latest',
                    'mistral:latest',
                    'qwen:latest'
                ],
                'task_type': 'architecture',
                'context_requirements': {
                    'complexity_level': 'high',
                    'creative_task': True,
                    'performance_critical': True
                }
            },
            'test_generation': {
                'preferred_models': [
                    'codellama:latest',
                    'deepseek-coder:6.7b',
                    'llama3:latest'
                ],
                'task_type': 'testing',
                'context_requirements': {
                    'complexity_level': 'medium',
                    'performance_critical': True,
                    'is_completion': False
                }
            },
            'documentation': {
                'preferred_models': [
                    'llama3:latest',
                    'gemma:7b',
                    'mistral:latest'
                ],
                'task_type': 'documentation_generation',
                'context_requirements': {
                    'complexity_level': 'medium',
                    'creative_task': True,
                    'performance_critical': False
                }
            }
        }
    
    def _initialize_prompt_templates(self) -> Dict[str, str]:
        """Initialize specialized prompt templates for NL2Code tasks"""
        return {
            'requirement_analysis': '''You are ABOV3 Genesis, an expert AI software architect specializing in requirement analysis.

Analyze the following feature description and extract structured requirements:

Feature Description: {description}

Project Context:
{project_context}

Instructions:
1. Identify core features and their priorities
2. Determine technical complexity and dependencies
3. Estimate development effort in hours
4. Generate acceptance criteria
5. Identify potential risks and constraints

Respond with a JSON structure containing:
- features: Array of feature objects with name, description, priority, complexity, dependencies, estimated_hours, acceptance_criteria
- technical_context: Project type, recommended tech stack, architecture patterns
- risk_assessment: Potential challenges and mitigation strategies

Be extremely precise and provide production-ready analysis.''',

            'code_generation': '''You are ABOV3 Genesis, an expert AI software engineer specializing in production-ready code generation.

Generate complete, production-ready code for:

Task: {task_description}
Technical Context: {tech_context}
File Requirements: {file_requirements}

Requirements:
1. Generate complete, runnable code with no placeholders
2. Include proper error handling and validation
3. Follow best practices and design patterns
4. Add comprehensive comments and documentation
5. Include necessary imports and dependencies
6. Ensure code is secure and performant

For each file, provide:
```{language}
// File: {file_path}
{complete_code}
```

Generate only the code requested. Be extremely precise and production-ready.''',

            'architecture_planning': '''You are ABOV3 Genesis, an expert AI software architect specializing in system design.

Create a comprehensive implementation plan for:

Project: {project_description}
Features: {features_summary}
Technical Context: {tech_context}

Generate a detailed implementation plan with:

1. **Architecture Overview**
   - System architecture pattern (monolith/microservices/serverless)
   - Technology stack recommendations
   - Database design approach
   - API design patterns

2. **Implementation Milestones**
   - Milestone breakdown with dependencies
   - Estimated timeline for each phase
   - Risk assessment and mitigation strategies

3. **Technical Specifications**
   - File structure and organization
   - Component interactions and data flow
   - Security considerations
   - Performance optimization strategies

4. **Development Workflow**
   - Recommended development sequence
   - Testing strategy
   - Deployment approach

Respond with detailed, actionable planning information in JSON format.''',

            'test_generation': '''You are ABOV3 Genesis, an expert AI QA engineer specializing in comprehensive test generation.

Generate comprehensive tests for:

Code Component: {component_description}
Testing Framework: {testing_framework}
Test Types Needed: {test_types}

Requirements:
1. Generate complete, runnable test code
2. Include unit, integration, and E2E tests as appropriate
3. Cover positive, negative, and edge cases
4. Include proper test setup and teardown
5. Add meaningful test descriptions and comments
6. Ensure high test coverage (>85%)

For each test file, provide:
```{language}
// File: {test_file_path}
{complete_test_code}
```

Focus on production-quality tests that catch real issues.''',

            'documentation': '''You are ABOV3 Genesis, an expert technical writer specializing in software documentation.

Generate comprehensive documentation for:

Component: {component_description}
Documentation Type: {doc_type}
Target Audience: {audience}

Requirements:
1. Clear, concise, and accurate documentation
2. Include code examples where appropriate
3. Cover installation, configuration, and usage
4. Add troubleshooting and FAQ sections
5. Follow markdown best practices
6. Include API references if applicable

Generate complete, professional documentation that developers can immediately use.'''
        }
    
    def _initialize_optimization_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize optimization configurations for different NL2Code tasks"""
        return {
            'requirement_analysis': {
                'temperature': 0.15,
                'top_p': 0.92,
                'top_k': 30,
                'repeat_penalty': 1.08,
                'mirostat': 1,
                'mirostat_tau': 3.5,
                'num_predict': 2048,
                'stop': ['\n\n---', '<|end|>', 'Human:']
            },
            'code_generation': {
                'temperature': 0.05,
                'top_p': 0.98,
                'top_k': 20,
                'repeat_penalty': 0.95,
                'mirostat': 2,
                'mirostat_tau': 3.0,
                'num_predict': 4096,
                'stop': ['```\n\n', '<|end|>', 'Human:', '\n\n\n\n']
            },
            'architecture_planning': {
                'temperature': 0.25,
                'top_p': 0.90,
                'top_k': 45,
                'repeat_penalty': 1.12,
                'mirostat': 1,
                'mirostat_tau': 5.0,
                'num_predict': 3072,
                'stop': ['\n\n---', '<|end|>', 'Human:']
            },
            'test_generation': {
                'temperature': 0.08,
                'top_p': 0.96,
                'top_k': 25,
                'repeat_penalty': 1.02,
                'mirostat': 2,
                'mirostat_tau': 3.5,
                'num_predict': 3072,
                'stop': ['```\n\n', '<|end|>', 'Human:']
            },
            'documentation': {
                'temperature': 0.3,
                'top_p': 0.92,
                'top_k': 50,
                'repeat_penalty': 1.15,
                'mirostat': 0,
                'num_predict': 2048,
                'stop': ['\n\n---', '<|end|>', 'Human:', '# ']
            }
        }
    
    async def analyze_requirements_with_ai(
        self,
        description: str,
        project_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Use AI to analyze natural language requirements and extract structured information
        """
        logger.info("Analyzing requirements with AI assistance")
        
        try:
            # Select best model for requirement analysis
            model = await self._select_optimal_model('requirement_analysis')
            
            # Prepare prompt
            prompt = self.prompt_templates['requirement_analysis'].format(
                description=description,
                project_context=json.dumps(project_context or {}, indent=2)
            )
            
            # Get optimized configuration
            options = self._get_optimized_options('requirement_analysis', model)
            
            # Generate AI response
            messages = [{'role': 'user', 'content': prompt}]
            response = await self._generate_ai_response(model, messages, options)
            
            # Parse and validate response
            analysis_result = self._parse_requirements_analysis(response)
            
            logger.info("Requirements analysis completed successfully")
            return analysis_result
            
        except Exception as e:
            logger.error(f"AI requirements analysis failed: {e}")
            raise
    
    async def generate_code_with_ai(
        self,
        task_description: str,
        tech_context: Dict[str, Any],
        file_requirements: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Use AI to generate production-ready code based on specifications
        """
        logger.info("Generating code with AI assistance")
        
        try:
            # Select best model for code generation
            model = await self._select_optimal_model('code_generation')
            
            # Prepare prompt with technical context
            prompt = self.prompt_templates['code_generation'].format(
                task_description=task_description,
                tech_context=json.dumps(tech_context, indent=2),
                file_requirements=json.dumps(file_requirements, indent=2),
                language=tech_context.get('primary_language', 'python'),
                file_path='auto-generated'
            )
            
            # Get optimized configuration for code generation
            options = self._get_optimized_options('code_generation', model)
            
            # Generate code with AI
            messages = [{'role': 'user', 'content': prompt}]
            response = await self._generate_ai_response(model, messages, options)
            
            # Extract and validate code blocks
            code_blocks = self._extract_code_blocks(response)
            
            logger.info(f"Generated {len(code_blocks)} code files")
            return {
                'success': True,
                'code_blocks': code_blocks,
                'raw_response': response
            }
            
        except Exception as e:
            logger.error(f"AI code generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'code_blocks': []
            }
    
    async def create_implementation_plan_with_ai(
        self,
        project_description: str,
        features_summary: str,
        tech_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use AI to create comprehensive implementation plans
        """
        logger.info("Creating implementation plan with AI assistance")
        
        try:
            # Select best model for architecture planning
            model = await self._select_optimal_model('architecture_planning')
            
            # Prepare architectural planning prompt
            prompt = self.prompt_templates['architecture_planning'].format(
                project_description=project_description,
                features_summary=features_summary,
                tech_context=json.dumps(tech_context, indent=2)
            )
            
            # Get optimized configuration
            options = self._get_optimized_options('architecture_planning', model)
            
            # Generate implementation plan
            messages = [{'role': 'user', 'content': prompt}]
            response = await self._generate_ai_response(model, messages, options)
            
            # Parse implementation plan
            plan_data = self._parse_implementation_plan(response)
            
            logger.info("Implementation plan created successfully")
            return plan_data
            
        except Exception as e:
            logger.error(f"AI implementation planning failed: {e}")
            raise
    
    async def generate_tests_with_ai(
        self,
        component_description: str,
        testing_framework: str,
        test_types: List[str]
    ) -> Dict[str, Any]:
        """
        Use AI to generate comprehensive test suites
        """
        logger.info("Generating tests with AI assistance")
        
        try:
            # Select best model for test generation
            model = await self._select_optimal_model('test_generation')
            
            # Prepare test generation prompt
            prompt = self.prompt_templates['test_generation'].format(
                component_description=component_description,
                testing_framework=testing_framework,
                test_types=', '.join(test_types),
                language=self._determine_language_from_framework(testing_framework),
                test_file_path='auto-generated'
            )
            
            # Get optimized configuration
            options = self._get_optimized_options('test_generation', model)
            
            # Generate tests
            messages = [{'role': 'user', 'content': prompt}]
            response = await self._generate_ai_response(model, messages, options)
            
            # Extract test code blocks
            test_blocks = self._extract_code_blocks(response)
            
            logger.info(f"Generated {len(test_blocks)} test files")
            return {
                'success': True,
                'test_blocks': test_blocks,
                'raw_response': response
            }
            
        except Exception as e:
            logger.error(f"AI test generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'test_blocks': []
            }
    
    async def generate_documentation_with_ai(
        self,
        component_description: str,
        doc_type: str,
        audience: str = 'developers'
    ) -> Dict[str, Any]:
        """
        Use AI to generate comprehensive documentation
        """
        logger.info("Generating documentation with AI assistance")
        
        try:
            # Select best model for documentation
            model = await self._select_optimal_model('documentation')
            
            # Prepare documentation prompt
            prompt = self.prompt_templates['documentation'].format(
                component_description=component_description,
                doc_type=doc_type,
                audience=audience
            )
            
            # Get optimized configuration
            options = self._get_optimized_options('documentation', model)
            
            # Generate documentation
            messages = [{'role': 'user', 'content': prompt}]
            response = await self._generate_ai_response(model, messages, options)
            
            logger.info("Documentation generated successfully")
            return {
                'success': True,
                'documentation': response,
                'format': 'markdown'
            }
            
        except Exception as e:
            logger.error(f"AI documentation generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'documentation': ''
            }
    
    async def _select_optimal_model(self, task_type: str) -> str:
        """
        Select the optimal model for a specific task type based on availability and performance
        """
        task_config = self.specialized_models.get(task_type, {})
        preferred_models = task_config.get('preferred_models', ['llama3:latest'])
        
        # Check which models are available
        async with self.ollama_client:
            available_models = await self.ollama_client.list_models()
            available_model_names = [model.get('name', '') for model in available_models]
        
        # Select first available preferred model
        for model in preferred_models:
            if model in available_model_names:
                logger.debug(f"Selected model {model} for task {task_type}")
                return model
        
        # Fallback to any available model
        if available_model_names:
            fallback_model = available_model_names[0]
            logger.warning(f"Using fallback model {fallback_model} for task {task_type}")
            return fallback_model
        
        # Default fallback
        logger.error(f"No models available for task {task_type}, using default")
        return 'llama3:latest'
    
    def _get_optimized_options(self, task_type: str, model_name: str) -> Dict[str, Any]:
        """
        Get optimized options for a specific task type and model
        """
        # Start with base task optimization
        base_options = self.optimization_configs.get(task_type, {})
        
        # Apply model-specific optimizations using existing Ollama client method
        context_info = self.specialized_models.get(task_type, {}).get('context_requirements', {})
        
        # Use the existing optimization method from OllamaClient
        optimized_options = self.ollama_client.get_genesis_optimized_options(
            task_type, model_name, context_info
        )
        
        # Merge with our specialized configurations
        optimized_options.update(base_options)
        
        return optimized_options
    
    async def _generate_ai_response(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Dict[str, Any]
    ) -> str:
        """
        Generate AI response using optimized settings
        """
        response_parts = []
        
        try:
            async with self.ollama_client:
                # Ensure model exists
                if not await self.ollama_client.check_model_exists(model):
                    raise ValueError(f"Model {model} not available")
                
                # Generate response
                async for chunk in self.ollama_client.chat(
                    model, messages, options=options, stream=False
                ):
                    if 'error' in chunk:
                        raise Exception(f"AI Error: {chunk['error']}")
                    
                    if 'message' in chunk and chunk['message'].get('content'):
                        response_parts.append(chunk['message']['content'])
                    elif 'response' in chunk:
                        response_parts.append(chunk['response'])
                    
                    if chunk.get('done', False):
                        break
            
            return ''.join(response_parts).strip()
            
        except Exception as e:
            logger.error(f"AI response generation failed: {e}")
            raise
    
    def _parse_requirements_analysis(self, response: str) -> Dict[str, Any]:
        """
        Parse and validate requirements analysis response
        """
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            # If no JSON found, create structured response from text
            return self._create_structured_requirements_from_text(response)
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON requirements: {e}")
            return self._create_structured_requirements_from_text(response)
    
    def _create_structured_requirements_from_text(self, text: str) -> Dict[str, Any]:
        """
        Create structured requirements from unstructured text response
        """
        return {
            'features': [
                {
                    'name': 'Extracted Feature',
                    'description': text[:200] + '...' if len(text) > 200 else text,
                    'priority': 'medium',
                    'complexity': 'moderate',
                    'dependencies': [],
                    'estimated_hours': 8,
                    'acceptance_criteria': ['Feature works as described']
                }
            ],
            'technical_context': {
                'project_type': 'web',
                'recommended_tech_stack': ['python', 'fastapi'],
                'architecture_pattern': 'monolith'
            },
            'risk_assessment': {
                'potential_challenges': ['Implementation complexity'],
                'mitigation_strategies': ['Incremental development']
            }
        }
    
    def _extract_code_blocks(self, response: str) -> List[Dict[str, Any]]:
        """
        Extract code blocks from AI response
        """
        code_blocks = []
        
        # Pattern for markdown code blocks with file paths
        import re
        pattern = r'```(\w+)\s*(?://\s*File:\s*([^\n]+))?\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        for language, file_path, code in matches:
            if code.strip():
                code_blocks.append({
                    'language': language,
                    'file_path': file_path.strip() if file_path else f'generated.{language}',
                    'code': code.strip(),
                    'size': len(code.strip())
                })
        
        # Fallback: look for any code blocks without file paths
        if not code_blocks:
            simple_pattern = r'```(\w+)?\n(.*?)```'
            simple_matches = re.findall(simple_pattern, response, re.DOTALL)
            
            for language, code in simple_matches:
                if code.strip():
                    ext = language if language else 'txt'
                    code_blocks.append({
                        'language': language or 'text',
                        'file_path': f'generated.{ext}',
                        'code': code.strip(),
                        'size': len(code.strip())
                    })
        
        return code_blocks
    
    def _parse_implementation_plan(self, response: str) -> Dict[str, Any]:
        """
        Parse implementation plan from AI response
        """
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            # Create basic plan structure from text
            return self._create_basic_plan_from_text(response)
            
        except json.JSONDecodeError:
            return self._create_basic_plan_from_text(response)
    
    def _create_basic_plan_from_text(self, text: str) -> Dict[str, Any]:
        """
        Create basic implementation plan from text
        """
        return {
            'architecture_overview': {
                'pattern': 'monolith',
                'tech_stack': ['python', 'fastapi', 'postgresql'],
                'api_design': 'REST'
            },
            'milestones': [
                {
                    'name': 'Foundation Setup',
                    'description': 'Set up project structure and core dependencies',
                    'estimated_hours': 8,
                    'dependencies': []
                },
                {
                    'name': 'Core Implementation',
                    'description': 'Implement main features and functionality',
                    'estimated_hours': 24,
                    'dependencies': ['Foundation Setup']
                },
                {
                    'name': 'Testing & Deployment',
                    'description': 'Add tests and deploy to production',
                    'estimated_hours': 16,
                    'dependencies': ['Core Implementation']
                }
            ],
            'technical_specifications': {
                'file_structure': 'Standard web application structure',
                'security_considerations': 'Authentication and input validation',
                'performance_strategy': 'Database indexing and caching'
            },
            'development_workflow': {
                'sequence': 'Setup -> Implementation -> Testing -> Deployment',
                'testing_strategy': 'Unit and integration tests',
                'deployment_approach': 'Containerized deployment'
            }
        }
    
    def _determine_language_from_framework(self, framework: str) -> str:
        """
        Determine programming language from testing framework
        """
        framework_lower = framework.lower()
        
        if 'pytest' in framework_lower:
            return 'python'
        elif 'jest' in framework_lower or 'mocha' in framework_lower:
            return 'javascript'
        elif 'junit' in framework_lower:
            return 'java'
        elif 'rspec' in framework_lower:
            return 'ruby'
        else:
            return 'python'  # Default
    
    async def enhance_feature_understanding(
        self,
        feature_description: str,
        existing_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhance feature understanding with AI analysis
        """
        logger.info("Enhancing feature understanding with AI")
        
        try:
            model = await self._select_optimal_model('requirement_analysis')
            
            prompt = f"""Analyze this feature in context and provide enhanced understanding:

Feature: {feature_description}

Existing Context: {json.dumps(existing_context, indent=2)}

Provide detailed analysis including:
1. Technical complexity assessment
2. Implementation approach recommendations
3. Potential challenges and solutions
4. Integration points with existing systems
5. Performance considerations
6. Security implications

Respond with structured JSON format."""
            
            messages = [{'role': 'user', 'content': prompt}]
            options = self._get_optimized_options('requirement_analysis', model)
            
            response = await self._generate_ai_response(model, messages, options)
            return self._parse_requirements_analysis(response)
            
        except Exception as e:
            logger.error(f"Feature understanding enhancement failed: {e}")
            return existing_context
    
    async def optimize_code_quality(
        self,
        code: str,
        language: str,
        optimization_goals: List[str]
    ) -> Dict[str, Any]:
        """
        Use AI to optimize code quality and performance
        """
        logger.info("Optimizing code quality with AI")
        
        try:
            model = await self._select_optimal_model('code_generation')
            
            prompt = f"""Optimize this {language} code for production use:

```{language}
{code}
```

Optimization Goals: {', '.join(optimization_goals)}

Requirements:
1. Improve code quality and readability
2. Optimize for performance where possible
3. Add proper error handling
4. Ensure security best practices
5. Add comprehensive comments
6. Follow language-specific conventions

Provide the optimized code with explanations of changes made."""
            
            messages = [{'role': 'user', 'content': prompt}]
            options = self._get_optimized_options('code_generation', model)
            
            response = await self._generate_ai_response(model, messages, options)
            code_blocks = self._extract_code_blocks(response)
            
            return {
                'success': True,
                'optimized_code': code_blocks[0]['code'] if code_blocks else code,
                'improvements': response,
                'original_size': len(code),
                'optimized_size': len(code_blocks[0]['code']) if code_blocks else len(code)
            }
            
        except Exception as e:
            logger.error(f"Code optimization failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'optimized_code': code
            }
    
    async def validate_implementation_approach(
        self,
        approach_description: str,
        requirements: Dict[str, Any],
        constraints: List[str]
    ) -> Dict[str, Any]:
        """
        Use AI to validate and improve implementation approaches
        """
        logger.info("Validating implementation approach with AI")
        
        try:
            model = await self._select_optimal_model('architecture_planning')
            
            prompt = f"""Validate and improve this implementation approach:

Approach: {approach_description}

Requirements: {json.dumps(requirements, indent=2)}

Constraints: {', '.join(constraints)}

Provide:
1. Validation of the approach against requirements
2. Identification of potential issues or gaps
3. Suggestions for improvements
4. Alternative approaches if applicable
5. Risk assessment and mitigation strategies

Respond with structured analysis."""
            
            messages = [{'role': 'user', 'content': prompt}]
            options = self._get_optimized_options('architecture_planning', model)
            
            response = await self._generate_ai_response(model, messages, options)
            
            return {
                'success': True,
                'validation_result': response,
                'is_valid': 'valid' in response.lower() or 'good' in response.lower(),
                'suggestions': self._extract_suggestions_from_response(response)
            }
            
        except Exception as e:
            logger.error(f"Implementation validation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'is_valid': False
            }
    
    def _extract_suggestions_from_response(self, response: str) -> List[str]:
        """
        Extract actionable suggestions from AI response
        """
        suggestions = []
        
        # Look for numbered lists, bullet points, or suggestion markers
        import re
        patterns = [
            r'\d+\.\s+([^\n]+)',  # Numbered lists
            r'[-*]\s+([^\n]+)',   # Bullet points
            r'Suggestion:\s*([^\n]+)',  # Explicit suggestions
            r'Recommend:\s*([^\n]+)',   # Recommendations
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.MULTILINE)
            suggestions.extend(matches)
        
        return suggestions[:10]  # Limit to top 10 suggestions
    
    async def cleanup(self):
        """
        Cleanup resources used by the integration
        """
        if self.ollama_client:
            await self.ollama_client.close()
        
        logger.info("NL2Code Ollama integration cleaned up")


# Utility functions for easy integration
async def get_ai_enhanced_requirements(
    description: str,
    project_context: Optional[Dict[str, Any]] = None,
    ollama_client: Optional[OllamaClient] = None
) -> Dict[str, Any]:
    """
    Convenience function to get AI-enhanced requirements analysis
    """
    integration = NL2CodeOllamaIntegration(ollama_client)
    
    try:
        return await integration.analyze_requirements_with_ai(description, project_context)
    finally:
        await integration.cleanup()


async def generate_ai_code(
    task_description: str,
    tech_context: Dict[str, Any],
    file_requirements: List[Dict[str, Any]],
    ollama_client: Optional[OllamaClient] = None
) -> Dict[str, Any]:
    """
    Convenience function to generate code with AI
    """
    integration = NL2CodeOllamaIntegration(ollama_client)
    
    try:
        return await integration.generate_code_with_ai(task_description, tech_context, file_requirements)
    finally:
        await integration.cleanup()


async def create_ai_implementation_plan(
    project_description: str,
    features_summary: str,
    tech_context: Dict[str, Any],
    ollama_client: Optional[OllamaClient] = None
) -> Dict[str, Any]:
    """
    Convenience function to create implementation plan with AI
    """
    integration = NL2CodeOllamaIntegration(ollama_client)
    
    try:
        return await integration.create_implementation_plan_with_ai(
            project_description, features_summary, tech_context
        )
    finally:
        await integration.cleanup()


async def generate_ai_tests(
    component_description: str,
    testing_framework: str,
    test_types: List[str],
    ollama_client: Optional[OllamaClient] = None
) -> Dict[str, Any]:
    """
    Convenience function to generate tests with AI
    """
    integration = NL2CodeOllamaIntegration(ollama_client)
    
    try:
        return await integration.generate_tests_with_ai(
            component_description, testing_framework, test_types
        )
    finally:
        await integration.cleanup()