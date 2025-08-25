"""
ABOV3 Genesis - Enhanced Ollama AI Optimization System
Advanced prompt engineering, context management, and model optimization for Claude-level performance
"""

import asyncio
import json
import re
import time
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import logging
from collections import deque, defaultdict

from .ollama_client import OllamaClient
from .project_intelligence import ProjectIntelligence

logger = logging.getLogger(__name__)

@dataclass
class ContextWindow:
    """Smart context window management"""
    max_tokens: int
    current_tokens: int = 0
    priority_content: List[str] = field(default_factory=list)
    context_content: List[str] = field(default_factory=list)
    code_examples: List[str] = field(default_factory=list)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    
    def add_priority_content(self, content: str, priority: int = 1):
        """Add high-priority content that should always be included"""
        self.priority_content.insert(0 if priority > 5 else -1, content)
    
    def add_context(self, content: str):
        """Add context information"""
        self.context_content.append(content)
    
    def add_code_example(self, code: str, description: str = ""):
        """Add code example with description"""
        example = f"Example: {description}\n```\n{code}\n```" if description else f"```\n{code}\n```"
        self.code_examples.append(example)
    
    def optimize_for_tokens(self, target_tokens: int) -> str:
        """Build optimized context within token limit"""
        result_parts = []
        current_tokens = 0
        
        # Estimate tokens (rough: 1 token ≈ 4 characters)
        def estimate_tokens(text: str) -> int:
            return len(text) // 4
        
        # Always include priority content first
        for content in self.priority_content:
            tokens = estimate_tokens(content)
            if current_tokens + tokens <= target_tokens:
                result_parts.append(content)
                current_tokens += tokens
            else:
                # Truncate if necessary
                remaining_chars = (target_tokens - current_tokens) * 4
                if remaining_chars > 100:  # Only if meaningful amount left
                    result_parts.append(content[:remaining_chars] + "...")
                break
        
        # Add code examples (high value for code generation)
        for example in self.code_examples:
            tokens = estimate_tokens(example)
            if current_tokens + tokens <= target_tokens:
                result_parts.append(example)
                current_tokens += tokens
            else:
                break
        
        # Add recent conversation history
        for msg in reversed(self.conversation_history[-10:]):  # Last 10 messages
            msg_text = f"{msg['role']}: {msg['content']}"
            tokens = estimate_tokens(msg_text)
            if current_tokens + tokens <= target_tokens:
                result_parts.insert(-len(self.code_examples) if self.code_examples else -1, msg_text)
                current_tokens += tokens
            else:
                break
        
        # Add remaining context
        for content in self.context_content:
            tokens = estimate_tokens(content)
            if current_tokens + tokens <= target_tokens:
                result_parts.append(content)
                current_tokens += tokens
            else:
                break
        
        return "\n\n".join(result_parts)

class PromptTemplateEngine:
    """Advanced prompt template system optimized for different code generation tasks"""
    
    def __init__(self):
        self.templates = {
            "code_generation": self._get_code_generation_template(),
            "code_review": self._get_code_review_template(), 
            "debugging": self._get_debugging_template(),
            "architecture": self._get_architecture_template(),
            "explanation": self._get_explanation_template(),
            "optimization": self._get_optimization_template(),
            "testing": self._get_testing_template()
        }
        
        self.system_prompts = {
            "general": self._get_general_system_prompt(),
            "coding_expert": self._get_coding_expert_system_prompt(),
            "architect": self._get_architect_system_prompt(),
            "debugger": self._get_debugger_system_prompt()
        }
    
    def _get_general_system_prompt(self) -> str:
        return """You are ABOV3 Genesis AI, an advanced coding assistant that generates production-ready code with Claude-level quality and precision.

CORE PRINCIPLES:
1. Generate complete, working, production-ready code
2. Always use proper error handling and best practices
3. Include comprehensive comments and documentation
4. Optimize for readability, maintainability, and performance
5. Follow industry standards and conventions
6. Provide working examples with complete implementations

RESPONSE FORMAT:
- Always use markdown code blocks with language specification
- Include detailed explanations for complex logic
- Provide alternative approaches when relevant
- Suggest improvements and optimizations
- Include necessary imports and dependencies

QUALITY STANDARDS:
- Code must be syntactically correct and runnable
- Follow language-specific best practices
- Include proper type hints where applicable
- Implement comprehensive error handling
- Use descriptive variable and function names
- Add inline comments for complex logic"""

    def _get_coding_expert_system_prompt(self) -> str:
        return """You are an elite software engineer with 20+ years of experience across all programming languages and paradigms. You generate code that exceeds industry standards.

EXPERTISE AREAS:
- Full-stack development (Frontend, Backend, Database, DevOps)
- System architecture and design patterns
- Performance optimization and scalability
- Security best practices and vulnerability prevention
- Modern frameworks and libraries
- Testing strategies and automation
- Code review and quality assurance

CODE GENERATION STANDARDS:
1. ALWAYS generate complete, production-ready code
2. Include ALL necessary imports and dependencies
3. Implement proper error handling and edge cases
4. Add comprehensive documentation and comments
5. Follow SOLID principles and clean code practices
6. Optimize for performance and scalability
7. Include security considerations
8. Use modern language features and best practices

RESPONSE STRUCTURE:
1. Brief explanation of approach
2. Complete implementation with code blocks
3. Usage examples and test cases
4. Performance and security considerations
5. Potential improvements or alternatives

Remember: Your code should be ready for immediate use in production environments."""

    def _get_architect_system_prompt(self) -> str:
        return """You are a Principal Software Architect specializing in system design and enterprise architecture. You design scalable, maintainable, and robust software systems.

ARCHITECTURAL EXPERTISE:
- Microservices and distributed systems
- Event-driven and reactive architectures
- Database design and data modeling
- API design and integration patterns
- Cloud-native and container orchestration
- Security architecture and compliance
- Performance and scalability planning
- DevOps and CI/CD pipeline design

DESIGN PRINCIPLES:
- Scalability and high availability
- Security by design
- Maintainability and extensibility
- Performance optimization
- Cost-effectiveness
- Technology stack selection
- Risk assessment and mitigation

OUTPUT FORMAT:
1. System overview and requirements analysis
2. Architecture diagrams (in text/ASCII format)
3. Component breakdown and responsibilities
4. Data flow and interaction patterns
5. Technology recommendations
6. Implementation roadmap
7. Potential risks and mitigation strategies"""

    def _get_debugger_system_prompt(self) -> str:
        return """You are an expert debugging specialist with exceptional skills in identifying, analyzing, and fixing software issues across all programming languages and platforms.

DEBUGGING EXPERTISE:
- Static and dynamic code analysis
- Performance profiling and optimization
- Memory leak detection and resolution
- Concurrency and threading issues
- Database query optimization
- Network and API debugging
- Security vulnerability assessment
- Test-driven debugging approaches

DEBUGGING METHODOLOGY:
1. Systematic issue identification
2. Root cause analysis
3. Impact assessment
4. Solution design and implementation
5. Prevention strategies
6. Testing and validation
7. Documentation and knowledge sharing

RESPONSE APPROACH:
1. Issue analysis and classification
2. Potential causes identification
3. Step-by-step debugging process
4. Complete fix implementation
5. Testing recommendations
6. Prevention strategies
7. Code quality improvements

Always provide working, tested solutions with explanations."""

    def _get_code_generation_template(self) -> str:
        return """Generate high-quality, production-ready code for: {request}

REQUIREMENTS:
{requirements}

CONTEXT:
{context}

SPECIFICATIONS:
- Language: {language}
- Framework: {framework}
- Target: {target_platform}

Please provide:
1. Complete implementation with all necessary imports
2. Comprehensive error handling
3. Detailed comments and documentation
4. Usage examples
5. Testing recommendations

Focus on:
- Code quality and best practices
- Performance and scalability
- Security considerations
- Maintainability and readability"""

    def _get_code_review_template(self) -> str:
        return """Perform a comprehensive code review for the following code:

CODE TO REVIEW:
```{language}
{code}
```

REVIEW FOCUS AREAS:
1. Code quality and best practices
2. Security vulnerabilities
3. Performance optimization opportunities
4. Bug identification
5. Maintainability improvements
6. Testing coverage
7. Documentation quality

Please provide:
1. Overall assessment and rating
2. Specific issues with line numbers
3. Improvement recommendations
4. Refactored code examples
5. Security considerations
6. Performance optimization suggestions"""

    def _get_debugging_template(self) -> str:
        return """Debug and fix the following issue:

PROBLEM DESCRIPTION:
{problem}

CODE WITH ISSUES:
```{language}
{code}
```

ERROR MESSAGES:
{error_messages}

EXPECTED BEHAVIOR:
{expected_behavior}

Please provide:
1. Issue identification and root cause analysis
2. Step-by-step debugging approach
3. Complete fixed code implementation
4. Explanation of the fix
5. Testing recommendations
6. Prevention strategies for similar issues"""

    def _get_architecture_template(self) -> str:
        return """Design a software architecture for: {request}

REQUIREMENTS:
{requirements}

CONSTRAINTS:
{constraints}

SCALE REQUIREMENTS:
- Users: {user_scale}
- Data: {data_scale}
- Performance: {performance_requirements}

Please provide:
1. High-level architecture overview
2. Component breakdown and responsibilities
3. Data flow and interaction patterns
4. Technology stack recommendations
5. Scalability considerations
6. Security architecture
7. Implementation roadmap"""

    def _get_explanation_template(self) -> str:
        return """Explain the following code/concept in detail: {topic}

CODE (if applicable):
```{language}
{code}
```

EXPLANATION LEVEL: {level}

Please provide:
1. Clear, comprehensive explanation
2. Step-by-step breakdown
3. Key concepts and terminology
4. Practical examples
5. Common use cases
6. Best practices
7. Related concepts and further reading"""

    def _get_optimization_template(self) -> str:
        return """Optimize the following code for better performance and quality:

CURRENT CODE:
```{language}
{code}
```

OPTIMIZATION GOALS:
{goals}

CONSTRAINTS:
{constraints}

Please provide:
1. Performance analysis of current code
2. Optimization opportunities identification
3. Optimized implementation
4. Performance comparison
5. Trade-offs and considerations
6. Alternative approaches
7. Monitoring and profiling recommendations"""

    def _get_testing_template(self) -> str:
        return """Generate comprehensive tests for the following code:

CODE TO TEST:
```{language}
{code}
```

TESTING REQUIREMENTS:
- Test types: {test_types}
- Coverage goal: {coverage_goal}
- Framework: {test_framework}

Please provide:
1. Test strategy and approach
2. Unit tests with edge cases
3. Integration tests
4. Mock implementations
5. Test data and fixtures
6. Performance tests (if applicable)
7. Test execution and reporting setup"""

    def build_prompt(self, template_type: str, **kwargs) -> str:
        """Build optimized prompt from template"""
        if template_type not in self.templates:
            template_type = "code_generation"
        
        template = self.templates[template_type]
        
        # Fill in template variables
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Missing template variable: {e}")
            # Fill missing variables with placeholders
            for key in re.findall(r'\{(\w+)\}', template):
                if key not in kwargs:
                    kwargs[key] = f"[{key.upper()}]"
            return template.format(**kwargs)

    def get_system_prompt(self, prompt_type: str = "general") -> str:
        """Get optimized system prompt"""
        return self.system_prompts.get(prompt_type, self.system_prompts["general"])

class OllamaModelOptimizer:
    """Advanced Ollama model optimizer with learning capabilities"""
    
    def __init__(self, project_path: Optional[Path] = None):
        self.project_path = project_path
        self.template_engine = PromptTemplateEngine()
        
        # Performance tracking
        self.response_quality_scores = deque(maxlen=1000)
        self.model_performance = defaultdict(lambda: {"success": 0, "failure": 0, "avg_quality": 0.0})
        
        # Learning system
        self.feedback_history = []
        self.optimization_history = []
        
        # Context management
        self.context_cache = {}
        
        # Model-specific optimizations learned over time
        self.model_optimizations = {
            "codellama": {
                "strengths": ["code_generation", "debugging", "code_review"],
                "weaknesses": ["creative_writing", "general_conversation"],
                "optimal_temperature": 0.05,
                "optimal_top_p": 0.95,
                "context_preference": "code_focused"
            },
            "deepseek-coder": {
                "strengths": ["code_generation", "architecture", "optimization"],
                "weaknesses": ["creative_writing"],
                "optimal_temperature": 0.1,
                "optimal_top_p": 0.98,
                "context_preference": "detailed_requirements"
            },
            "llama3": {
                "strengths": ["general_reasoning", "explanation", "conversation"],
                "weaknesses": ["complex_code_generation"],
                "optimal_temperature": 0.2,
                "optimal_top_p": 0.9,
                "context_preference": "conversational"
            },
            "qwen": {
                "strengths": ["multilingual", "code_generation", "analysis"],
                "weaknesses": ["domain_specific"],
                "optimal_temperature": 0.15,
                "optimal_top_p": 0.95,
                "context_preference": "structured"
            }
        }
    
    def get_model_strengths(self, model_name: str) -> List[str]:
        """Get model's known strengths"""
        for model_key, info in self.model_optimizations.items():
            if model_key.lower() in model_name.lower():
                return info["strengths"]
        return ["general"]
    
    def optimize_for_task(self, task_type: str, model_name: str, user_request: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Optimize model parameters and prompt for specific task"""
        context = context or {}
        
        # Determine best model for task if not specified
        if not model_name:
            model_name = self._select_best_model_for_task(task_type)
        
        # Get model-specific optimizations
        model_opts = self._get_model_optimizations(model_name, task_type)
        
        # Build optimized context window
        context_window = self._build_context_window(task_type, user_request, context, model_name)
        
        # Select appropriate system prompt
        system_prompt_type = self._get_system_prompt_type(task_type)
        system_prompt = self.template_engine.get_system_prompt(system_prompt_type)
        
        # Build task-specific prompt
        optimized_prompt = self._build_optimized_prompt(task_type, user_request, context, context_window)
        
        # Combine everything
        final_prompt = f"{system_prompt}\n\n{context_window.optimize_for_tokens(model_opts.get('max_context_tokens', 8000))}\n\n{optimized_prompt}"
        
        return {
            "model_name": model_name,
            "prompt": final_prompt,
            "options": model_opts,
            "task_type": task_type,
            "estimated_quality": self._estimate_response_quality(model_name, task_type)
        }
    
    def _select_best_model_for_task(self, task_type: str) -> str:
        """Select the best available model for the task"""
        task_model_preferences = {
            "code_generation": ["codellama", "deepseek-coder", "starcoder", "llama3"],
            "debugging": ["codellama", "deepseek-coder", "llama3"],
            "code_review": ["codellama", "deepseek-coder", "llama3"],
            "architecture": ["deepseek-coder", "llama3", "codellama"],
            "explanation": ["llama3", "qwen", "codellama"],
            "optimization": ["deepseek-coder", "codellama", "llama3"],
            "testing": ["codellama", "deepseek-coder", "llama3"],
            "conversation": ["llama3", "qwen", "mistral"]
        }
        
        preferred_models = task_model_preferences.get(task_type, ["llama3", "codellama"])
        
        # For now, return the first preference (in real implementation, check availability)
        return preferred_models[0] + ":latest"
    
    def _get_model_optimizations(self, model_name: str, task_type: str) -> Dict[str, Any]:
        """Get optimized parameters for model and task combination"""
        base_options = {
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 40,
            "repeat_penalty": 1.02,
            "num_predict": -1,
            "max_context_tokens": 8192
        }
        
        # Get model-specific optimizations
        for model_key, opts in self.model_optimizations.items():
            if model_key.lower() in model_name.lower():
                base_options.update({
                    "temperature": opts.get("optimal_temperature", base_options["temperature"]),
                    "top_p": opts.get("optimal_top_p", base_options["top_p"])
                })
                break
        
        # Task-specific adjustments
        task_adjustments = {
            "code_generation": {"temperature": 0.05, "repeat_penalty": 1.0, "mirostat": 2},
            "debugging": {"temperature": 0.1, "top_k": 30, "repeat_penalty": 1.0},
            "creative_writing": {"temperature": 0.8, "top_p": 0.9, "repeat_penalty": 1.3},
            "explanation": {"temperature": 0.4, "top_p": 0.9},
            "conversation": {"temperature": 0.6, "top_p": 0.9}
        }
        
        if task_type in task_adjustments:
            base_options.update(task_adjustments[task_type])
        
        return base_options
    
    def _build_context_window(self, task_type: str, user_request: str, context: Dict[str, Any], model_name: str) -> ContextWindow:
        """Build optimized context window for the request"""
        # Determine context window size based on model
        max_tokens = 8192  # Default
        if "codellama" in model_name.lower():
            max_tokens = 16384
        elif "deepseek" in model_name.lower():
            max_tokens = 16384
        elif "llama3" in model_name.lower():
            max_tokens = 8192
        
        window = ContextWindow(max_tokens=max_tokens)
        
        # Add task-specific context
        if task_type == "code_generation":
            self._add_code_generation_context(window, user_request, context)
        elif task_type == "debugging":
            self._add_debugging_context(window, user_request, context)
        elif task_type == "code_review":
            self._add_code_review_context(window, user_request, context)
        elif task_type == "architecture":
            self._add_architecture_context(window, user_request, context)
        
        # Add project-specific context
        if context.get("project_intelligence"):
            project_context = context["project_intelligence"].get_context_for_ai()
            window.add_context(project_context)
        
        # Add conversation history
        if context.get("conversation_history"):
            window.conversation_history = context["conversation_history"][-5:]  # Last 5 exchanges
        
        return window
    
    def _add_code_generation_context(self, window: ContextWindow, user_request: str, context: Dict[str, Any]):
        """Add code generation specific context"""
        # Add high-priority coding standards
        window.add_priority_content("""
CODING STANDARDS:
- Use clear, descriptive variable and function names
- Include comprehensive error handling
- Add type hints and documentation
- Follow language-specific best practices
- Implement proper logging and debugging support
- Include input validation and edge case handling
- Use modern language features and idioms
""", priority=10)
        
        # Add language-specific examples if detected
        detected_language = self._detect_language_from_request(user_request)
        if detected_language:
            examples = self._get_language_examples(detected_language)
            for example in examples:
                window.add_code_example(example["code"], example["description"])
        
        # Add project-specific patterns
        if context.get("project_patterns"):
            window.add_context(f"Project patterns and conventions:\n{context['project_patterns']}")
    
    def _add_debugging_context(self, window: ContextWindow, user_request: str, context: Dict[str, Any]):
        """Add debugging specific context"""
        window.add_priority_content("""
DEBUGGING METHODOLOGY:
1. Identify the exact error and symptoms
2. Reproduce the issue consistently
3. Analyze the root cause systematically
4. Implement the minimal fix required
5. Add preventive measures and tests
6. Document the solution and learning
""", priority=9)
        
        # Add debugging examples
        debugging_examples = [
            {
                "code": "try:\n    result = risky_operation()\nexcept SpecificError as e:\n    logger.error(f'Operation failed: {e}')\n    return default_value",
                "description": "Proper exception handling pattern"
            }
        ]
        
        for example in debugging_examples:
            window.add_code_example(example["code"], example["description"])
    
    def _add_code_review_context(self, window: ContextWindow, user_request: str, context: Dict[str, Any]):
        """Add code review specific context"""
        window.add_priority_content("""
CODE REVIEW CHECKLIST:
□ Functionality: Does the code work as intended?
□ Readability: Is the code easy to understand?
□ Performance: Are there obvious optimization opportunities?
□ Security: Are there potential vulnerabilities?
□ Maintainability: Is the code easy to modify and extend?
□ Testing: Is the code properly tested?
□ Documentation: Is the code adequately documented?
""", priority=8)
    
    def _add_architecture_context(self, window: ContextWindow, user_request: str, context: Dict[str, Any]):
        """Add architecture design specific context"""
        window.add_priority_content("""
ARCHITECTURAL PRINCIPLES:
- Single Responsibility Principle
- Open/Closed Principle
- Dependency Inversion
- Separation of Concerns
- Scalability and Performance
- Security by Design
- Maintainability and Testability
""", priority=8)
        
        # Add architecture pattern examples
        arch_examples = [
            {
                "code": "# MVC Pattern\nclass Controller:\n    def __init__(self, model, view):\n        self.model = model\n        self.view = view\n    \n    def handle_request(self, request):\n        data = self.model.process(request)\n        return self.view.render(data)",
                "description": "MVC Architecture Pattern"
            }
        ]
        
        for example in arch_examples:
            window.add_code_example(example["code"], example["description"])
    
    def _get_system_prompt_type(self, task_type: str) -> str:
        """Get appropriate system prompt type for task"""
        prompt_mapping = {
            "code_generation": "coding_expert",
            "debugging": "debugger", 
            "code_review": "coding_expert",
            "architecture": "architect",
            "explanation": "coding_expert",
            "optimization": "coding_expert",
            "testing": "coding_expert"
        }
        
        return prompt_mapping.get(task_type, "general")
    
    def _build_optimized_prompt(self, task_type: str, user_request: str, context: Dict[str, Any], context_window: ContextWindow) -> str:
        """Build optimized prompt using templates"""
        # Extract relevant information from context
        language = context.get("language", self._detect_language_from_request(user_request))
        framework = context.get("framework", "")
        target_platform = context.get("target_platform", "general")
        
        # Build template parameters
        template_params = {
            "request": user_request,
            "requirements": context.get("requirements", "Generate high-quality, production-ready code"),
            "context": context.get("additional_context", ""),
            "language": language,
            "framework": framework,
            "target_platform": target_platform,
            "level": context.get("explanation_level", "intermediate"),
            "test_types": context.get("test_types", "unit, integration"),
            "coverage_goal": context.get("coverage_goal", "90%"),
            "test_framework": context.get("test_framework", "pytest")
        }
        
        # Build prompt from template
        return self.template_engine.build_prompt(task_type, **template_params)
    
    def _detect_language_from_request(self, user_request: str) -> str:
        """Detect programming language from user request"""
        language_keywords = {
            "python": ["python", "py", "django", "flask", "fastapi", "pandas", "numpy"],
            "javascript": ["javascript", "js", "node", "react", "vue", "angular", "express"],
            "typescript": ["typescript", "ts", "tsx"],
            "java": ["java", "spring", "maven", "gradle"],
            "cpp": ["c++", "cpp", "cmake"],
            "c": ["c language", " c "],
            "csharp": ["c#", "csharp", ".net", "dotnet"],
            "go": ["golang", "go lang", " go "],
            "rust": ["rust", "cargo"],
            "php": ["php", "laravel", "symfony"],
            "ruby": ["ruby", "rails"],
            "swift": ["swift", "ios"],
            "kotlin": ["kotlin", "android"],
            "html": ["html", "html5", "web page"],
            "css": ["css", "scss", "sass", "styling"],
            "sql": ["sql", "database", "query", "mysql", "postgresql"]
        }
        
        user_request_lower = user_request.lower()
        
        for language, keywords in language_keywords.items():
            if any(keyword in user_request_lower for keyword in keywords):
                return language
        
        # Check for file extensions
        extensions = {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".java": "java", ".cpp": "cpp", ".c": "c", ".cs": "csharp",
            ".go": "go", ".rs": "rust", ".php": "php", ".rb": "ruby",
            ".swift": "swift", ".kt": "kotlin", ".html": "html",
            ".css": "css", ".sql": "sql"
        }
        
        for ext, lang in extensions.items():
            if ext in user_request_lower:
                return lang
        
        return "python"  # Default fallback
    
    def _get_language_examples(self, language: str) -> List[Dict[str, str]]:
        """Get high-quality examples for specific language"""
        examples = {
            "python": [
                {
                    "code": '''def process_data(data: List[Dict[str, Any]], validate: bool = True) -> List[Dict[str, Any]]:
    """Process and validate data with comprehensive error handling."""
    if not data:
        raise ValueError("Data cannot be empty")
    
    processed_data = []
    for item in data:
        try:
            if validate:
                validate_item(item)
            processed_item = transform_item(item)
            processed_data.append(processed_item)
        except ValidationError as e:
            logger.warning(f"Skipping invalid item: {e}")
        except Exception as e:
            logger.error(f"Unexpected error processing item: {e}")
            raise
    
    return processed_data''',
                    "description": "Python function with proper error handling and type hints"
                }
            ],
            "javascript": [
                {
                    "code": '''async function processData(data, options = {}) {
    const { validate = true, timeout = 5000 } = options;
    
    if (!Array.isArray(data) || data.length === 0) {
        throw new Error('Data must be a non-empty array');
    }
    
    const results = [];
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    
    try {
        for (const item of data) {
            if (validate && !isValidItem(item)) {
                console.warn('Skipping invalid item:', item);
                continue;
            }
            
            const processed = await processItem(item, { 
                signal: controller.signal 
            });
            results.push(processed);
        }
        
        return results;
    } finally {
        clearTimeout(timeoutId);
    }
}''',
                    "description": "JavaScript async function with error handling and timeout"
                }
            ]
        }
        
        return examples.get(language, [])
    
    def _estimate_response_quality(self, model_name: str, task_type: str) -> float:
        """Estimate expected response quality based on historical data"""
        model_key = next((key for key in self.model_optimizations.keys() if key.lower() in model_name.lower()), None)
        
        if not model_key:
            return 0.7  # Default estimate
        
        model_info = self.model_optimizations[model_key]
        
        # Check if task is in model's strengths
        if task_type in model_info["strengths"]:
            base_quality = 0.9
        elif task_type in model_info.get("weaknesses", []):
            base_quality = 0.6
        else:
            base_quality = 0.75
        
        # Adjust based on historical performance
        if model_name in self.model_performance:
            perf = self.model_performance[model_name]
            success_rate = perf["success"] / max(1, perf["success"] + perf["failure"])
            historical_quality = perf["avg_quality"]
            
            # Weight recent performance
            final_quality = (base_quality * 0.3) + (success_rate * 0.3) + (historical_quality * 0.4)
        else:
            final_quality = base_quality
        
        return min(1.0, max(0.0, final_quality))
    
    def record_feedback(self, model_name: str, task_type: str, user_feedback: str, quality_score: float):
        """Record user feedback for learning"""
        feedback_entry = {
            "timestamp": time.time(),
            "model_name": model_name,
            "task_type": task_type,
            "user_feedback": user_feedback,
            "quality_score": quality_score
        }
        
        self.feedback_history.append(feedback_entry)
        
        # Update model performance tracking
        if quality_score >= 0.7:
            self.model_performance[model_name]["success"] += 1
        else:
            self.model_performance[model_name]["failure"] += 1
        
        # Update average quality
        current_avg = self.model_performance[model_name]["avg_quality"]
        total_attempts = self.model_performance[model_name]["success"] + self.model_performance[model_name]["failure"]
        
        new_avg = ((current_avg * (total_attempts - 1)) + quality_score) / total_attempts
        self.model_performance[model_name]["avg_quality"] = new_avg
        
        # Learn from feedback
        self._learn_from_feedback(feedback_entry)
    
    def _learn_from_feedback(self, feedback_entry: Dict[str, Any]):
        """Learn and adapt from user feedback"""
        model_name = feedback_entry["model_name"]
        task_type = feedback_entry["task_type"]
        quality_score = feedback_entry["quality_score"]
        
        # Find model key
        model_key = next((key for key in self.model_optimizations.keys() if key.lower() in model_name.lower()), None)
        
        if not model_key:
            return
        
        model_info = self.model_optimizations[model_key]
        
        # Adapt strengths and weaknesses based on feedback
        if quality_score >= 0.8 and task_type not in model_info["strengths"]:
            # Task performed well, consider it a strength
            if len(model_info["strengths"]) < 5:  # Limit strengths
                model_info["strengths"].append(task_type)
                if task_type in model_info.get("weaknesses", []):
                    model_info["weaknesses"].remove(task_type)
        
        elif quality_score < 0.5 and task_type not in model_info.get("weaknesses", []):
            # Task performed poorly, consider it a weakness
            if "weaknesses" not in model_info:
                model_info["weaknesses"] = []
            if len(model_info["weaknesses"]) < 3:  # Limit weaknesses
                model_info["weaknesses"].append(task_type)
                if task_type in model_info["strengths"]:
                    model_info["strengths"].remove(task_type)
        
        logger.info(f"Learned from feedback: {model_name} on {task_type} -> quality {quality_score}")
    
    def get_optimization_suggestions(self, model_name: str) -> List[str]:
        """Get optimization suggestions for improving model performance"""
        suggestions = []
        
        if model_name in self.model_performance:
            perf = self.model_performance[model_name]
            success_rate = perf["success"] / max(1, perf["success"] + perf["failure"])
            avg_quality = perf["avg_quality"]
            
            if success_rate < 0.7:
                suggestions.append("Consider using different system prompts or adjusting temperature settings")
            
            if avg_quality < 0.6:
                suggestions.append("Try providing more detailed context and examples")
                suggestions.append("Consider using a different model for this task type")
            
            # Analyze recent feedback
            recent_feedback = [f for f in self.feedback_history if f["model_name"] == model_name][-10:]
            if recent_feedback:
                avg_recent_quality = sum(f["quality_score"] for f in recent_feedback) / len(recent_feedback)
                if avg_recent_quality < avg_quality:
                    suggestions.append("Recent performance has declined, consider model fine-tuning")
        
        return suggestions

async def optimize_ollama_for_genesis(project_path: Optional[Path] = None) -> OllamaModelOptimizer:
    """Initialize and return optimized Ollama system for Genesis"""
    optimizer = OllamaModelOptimizer(project_path)
    
    # Load any saved optimizations
    if project_path:
        optimization_file = project_path / '.abov3' / 'ollama_optimizations.json'
        if optimization_file.exists():
            try:
                with open(optimization_file, 'r') as f:
                    saved_data = json.load(f)
                    optimizer.model_performance.update(saved_data.get('model_performance', {}))
                    optimizer.feedback_history.extend(saved_data.get('feedback_history', []))
                logger.info(f"Loaded optimization data from {optimization_file}")
            except Exception as e:
                logger.warning(f"Could not load optimization data: {e}")
    
    return optimizer