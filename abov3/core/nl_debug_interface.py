"""
ABOV3 Genesis - Natural Language Debug Interface
Advanced NLP system for conversational debugging with Claude-level understanding
"""

import re
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from collections import defaultdict, Counter
from dataclasses import dataclass
from enum import Enum
import ast

# NLP imports with fallbacks
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class QueryIntent(Enum):
    """Debug query intent categories"""
    ERROR_ANALYSIS = "analyze_error"
    FIX_SUGGESTION = "suggest_fix"
    CODE_EXPLANATION = "explain_code"
    PERFORMANCE_ANALYSIS = "analyze_performance"
    SECURITY_CHECK = "check_security"
    BEST_PRACTICES = "check_best_practices"
    REFACTORING = "suggest_refactoring"
    TESTING = "suggest_tests"
    DEBUGGING_HELP = "debug_help"
    CODE_GENERATION = "generate_code"
    UNKNOWN = "unknown"


@dataclass
class ParsedQuery:
    """Parsed natural language query"""
    original_text: str
    intent: QueryIntent
    entities: Dict[str, List[str]]
    confidence: float
    query_type: str
    parameters: Dict[str, Any]
    context_references: List[str]


@dataclass
class DebugResponse:
    """Structured debug response"""
    response_text: str
    intent: QueryIntent
    confidence: float
    code_examples: List[str]
    recommendations: List[str]
    follow_up_questions: List[str]
    metadata: Dict[str, Any]


class IntentClassifier:
    """ML-powered intent classification for debug queries"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.intent_patterns = {}
        self.confidence_threshold = 0.5
        self.initialized = False
        
        self._load_intent_patterns()
        
        if HAS_TRANSFORMERS:
            self._initialize_model()
    
    def _load_intent_patterns(self):
        """Load intent recognition patterns"""
        self.intent_patterns = {
            QueryIntent.ERROR_ANALYSIS: [
                r"what.*wrong|why.*error|why.*fail|analyze.*error|debug.*error",
                r"error.*mean|exception.*mean|what.*cause|root.*cause",
                r"fix.*error|solve.*error|resolve.*issue|troubleshoot"
            ],
            QueryIntent.FIX_SUGGESTION: [
                r"how.*fix|how.*solve|how.*resolve|suggest.*fix",
                r"repair.*code|correct.*code|improve.*code",
                r"solution.*for|way.*to.*fix|fix.*this"
            ],
            QueryIntent.CODE_EXPLANATION: [
                r"explain.*code|what.*does.*code|how.*works|understand.*code",
                r"walk.*through|step.*by.*step|breakdown|analyze.*logic",
                r"meaning.*of|purpose.*of|intent.*of"
            ],
            QueryIntent.PERFORMANCE_ANALYSIS: [
                r"slow|performance|speed|optimize|faster|bottleneck",
                r"memory.*usage|cpu.*usage|resource.*usage|efficiency",
                r"improve.*performance|make.*faster|reduce.*time"
            ],
            QueryIntent.SECURITY_CHECK: [
                r"security|secure|vulnerability|exploit|attack|safe",
                r"injection|xss|csrf|authentication|authorization",
                r"security.*issue|security.*flaw|insecure"
            ],
            QueryIntent.BEST_PRACTICES: [
                r"best.*practice|coding.*standard|convention|guideline",
                r"clean.*code|code.*quality|maintainable|readable",
                r"pythonic|idiomatic|proper.*way|recommended.*way"
            ],
            QueryIntent.REFACTORING: [
                r"refactor|restructure|reorganize|improve.*structure",
                r"simplify.*code|reduce.*complexity|make.*cleaner",
                r"better.*design|improve.*architecture"
            ],
            QueryIntent.TESTING: [
                r"test|testing|unit.*test|integration.*test|test.*case",
                r"mock|assert|pytest|unittest|coverage",
                r"how.*to.*test|write.*test|test.*strategy"
            ],
            QueryIntent.DEBUGGING_HELP: [
                r"debug|debugging|debugger|breakpoint|step.*through",
                r"trace|stack.*trace|call.*stack|execution.*flow",
                r"variable.*value|inspect.*variable|watch"
            ],
            QueryIntent.CODE_GENERATION: [
                r"generate.*code|write.*code|create.*function|implement",
                r"code.*for|example.*of|template.*for|scaffold",
                r"build.*function|create.*class|implement.*method"
            ]
        }
    
    def _initialize_model(self):
        """Initialize transformer model for intent classification"""
        try:
            # Use a lightweight model for intent classification
            model_name = "microsoft/DialoGPT-medium"  # Can be replaced with domain-specific model
            self.model = pipeline("text-classification", 
                                 model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                                 device=0 if torch.cuda.is_available() else -1)
            self.initialized = True
            logging.info("Intent classification model initialized")
        except Exception as e:
            logging.warning(f"Failed to initialize intent model: {e}")
            self.initialized = False
    
    def classify_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """Classify the intent of a debug query"""
        # First try pattern-based classification
        pattern_intent, pattern_confidence = self._classify_by_patterns(query)
        
        if pattern_confidence > 0.7:
            return pattern_intent, pattern_confidence
        
        # Try ML-based classification if available
        if self.initialized and HAS_TRANSFORMERS:
            ml_intent, ml_confidence = self._classify_by_ml(query)
            
            # Combine pattern and ML results
            if ml_confidence > pattern_confidence:
                return ml_intent, ml_confidence
        
        return pattern_intent, pattern_confidence
    
    def _classify_by_patterns(self, query: str) -> Tuple[QueryIntent, float]:
        """Classify intent using regex patterns"""
        query_lower = query.lower()
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower))
                score += matches * (len(pattern) / 100.0)  # Weight by pattern complexity
            
            if score > 0:
                intent_scores[intent] = score
        
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            confidence = min(best_intent[1] / 2.0, 1.0)  # Normalize confidence
            return best_intent[0], confidence
        
        return QueryIntent.UNKNOWN, 0.0
    
    def _classify_by_ml(self, query: str) -> Tuple[QueryIntent, float]:
        """Classify intent using ML model"""
        try:
            # This is a placeholder - would need a proper intent classification model
            # For now, use sentiment as a proxy for confidence
            result = self.model(query)
            
            # Map sentiment to debug intent (this is a simplification)
            # In practice, you'd train a specific model for debug intents
            if result[0]['label'] == 'LABEL_0':  # Negative sentiment might indicate error
                return QueryIntent.ERROR_ANALYSIS, result[0]['score']
            elif result[0]['label'] == 'LABEL_2':  # Positive might indicate seeking help
                return QueryIntent.DEBUGGING_HELP, result[0]['score']
            else:
                return QueryIntent.CODE_EXPLANATION, result[0]['score']
        
        except Exception as e:
            logging.warning(f"ML intent classification failed: {e}")
            return QueryIntent.UNKNOWN, 0.0


class EntityExtractor:
    """Extract entities from debug queries"""
    
    def __init__(self):
        self.nlp = None
        self.custom_entities = {}
        
        if HAS_SPACY:
            self._load_spacy_model()
        
        self._define_custom_entities()
    
    def _load_spacy_model(self):
        """Load spaCy model for entity extraction"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            try:
                # Try to download the model
                spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
            except:
                logging.warning("Could not load spaCy model. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
    
    def _define_custom_entities(self):
        """Define custom entities for debugging context"""
        self.custom_entities = {
            'programming_languages': [
                'python', 'javascript', 'java', 'c++', 'c#', 'go', 'rust', 
                'typescript', 'php', 'ruby', 'kotlin', 'swift', 'scala'
            ],
            'error_types': [
                'attributeerror', 'keyerror', 'indexerror', 'typeerror', 'valueerror',
                'nameerror', 'syntaxerror', 'runtimeerror', 'memoryerror', 'recursionerror'
            ],
            'code_constructs': [
                'function', 'method', 'class', 'variable', 'loop', 'condition',
                'exception', 'import', 'module', 'package', 'decorator', 'generator'
            ],
            'tools_frameworks': [
                'django', 'flask', 'fastapi', 'react', 'vue', 'angular',
                'pandas', 'numpy', 'sklearn', 'tensorflow', 'pytorch'
            ],
            'file_types': [
                'py', 'js', 'html', 'css', 'json', 'xml', 'csv', 'txt',
                'md', 'yaml', 'yml', 'sql', 'log'
            ]
        }
    
    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities from query"""
        entities = {
            'programming_languages': [],
            'error_types': [],
            'code_constructs': [],
            'tools_frameworks': [],
            'file_types': [],
            'file_names': [],
            'function_names': [],
            'variable_names': [],
            'line_numbers': [],
            'general_entities': []
        }
        
        # Extract using spaCy if available
        if self.nlp:
            doc = self.nlp(query)
            for ent in doc.ents:
                entities['general_entities'].append({
                    'text': ent.text,
                    'label': ent.label_,
                    'description': spacy.explain(ent.label_)
                })
        
        # Extract custom entities
        query_lower = query.lower()
        
        for category, items in self.custom_entities.items():
            found_items = []
            for item in items:
                if item.lower() in query_lower:
                    found_items.append(item)
            entities[category] = found_items
        
        # Extract specific patterns
        entities['line_numbers'] = re.findall(r'line\s+(\d+)', query_lower)
        entities['file_names'] = re.findall(r'(\w+\.\w+)', query)
        entities['function_names'] = re.findall(r'function\s+(\w+)|def\s+(\w+)', query_lower)
        entities['variable_names'] = re.findall(r'variable\s+(\w+)|var\s+(\w+)', query_lower)
        
        # Clean up empty lists and flatten tuples
        cleaned_entities = {}
        for key, value in entities.items():
            if value:
                if key in ['function_names', 'variable_names']:
                    # Flatten tuples from regex groups
                    flattened = [item for sublist in value for item in sublist if item]
                    cleaned_entities[key] = flattened
                else:
                    cleaned_entities[key] = value
        
        return cleaned_entities


class ContextManager:
    """Manage conversational context for debug sessions"""
    
    def __init__(self):
        self.current_context = {}
        self.conversation_history = []
        self.context_stack = []
        self.active_files = set()
        self.active_errors = {}
        self.session_metadata = {}
    
    def update_context(self, query: str, parsed_query: ParsedQuery, 
                      response: DebugResponse):
        """Update conversation context"""
        context_update = {
            'timestamp': datetime.now(),
            'query': query,
            'intent': parsed_query.intent.value,
            'entities': parsed_query.entities,
            'response_summary': response.response_text[:200] + "..." if len(response.response_text) > 200 else response.response_text
        }
        
        self.conversation_history.append(context_update)
        
        # Update active entities
        entities = parsed_query.entities
        
        if 'file_names' in entities:
            self.active_files.update(entities['file_names'])
        
        if 'error_types' in entities:
            for error_type in entities['error_types']:
                if error_type not in self.active_errors:
                    self.active_errors[error_type] = []
                self.active_errors[error_type].append(context_update['timestamp'])
        
        # Maintain context window (keep last 20 interactions)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    def get_relevant_context(self, current_query: str) -> Dict[str, Any]:
        """Get relevant context for current query"""
        context = {
            'recent_queries': self.conversation_history[-5:] if self.conversation_history else [],
            'active_files': list(self.active_files),
            'active_errors': dict(self.active_errors),
            'session_metadata': self.session_metadata,
            'context_references': []
        }
        
        # Find context references in current query
        query_lower = current_query.lower()
        
        # References to previous interactions
        if any(word in query_lower for word in ['that', 'this', 'it', 'previous', 'last', 'earlier']):
            context['context_references'].extend(['previous_interaction'])
        
        # References to files
        for file_name in self.active_files:
            if file_name.lower() in query_lower:
                context['context_references'].append(f"file:{file_name}")
        
        # References to errors
        for error_type in self.active_errors:
            if error_type.lower() in query_lower:
                context['context_references'].append(f"error:{error_type}")
        
        return context
    
    def push_context(self, context_type: str, context_data: Dict[str, Any]):
        """Push a new context layer (e.g., entering function debug)"""
        self.context_stack.append({
            'type': context_type,
            'data': context_data,
            'timestamp': datetime.now()
        })
    
    def pop_context(self) -> Optional[Dict[str, Any]]:
        """Pop the current context layer"""
        return self.context_stack.pop() if self.context_stack else None
    
    def clear_context(self):
        """Clear all context"""
        self.current_context = {}
        self.conversation_history = []
        self.context_stack = []
        self.active_files = set()
        self.active_errors = {}


class ResponseGenerator:
    """Generate natural language responses for debug queries"""
    
    def __init__(self):
        self.response_templates = {}
        self.code_examples = {}
        self.follow_up_templates = {}
        
        self._load_response_templates()
        self._load_code_examples()
        self._load_follow_up_templates()
    
    def _load_response_templates(self):
        """Load response templates for different intents"""
        self.response_templates = {
            QueryIntent.ERROR_ANALYSIS: [
                "I've analyzed the {error_type} and found that {analysis}. The root cause appears to be {root_cause}.",
                "This {error_type} typically occurs when {common_cause}. In your case, {specific_analysis}.",
                "Looking at the error pattern, I can see that {pattern_analysis}. Here's what's happening: {explanation}."
            ],
            QueryIntent.FIX_SUGGESTION: [
                "To fix this issue, I recommend the following approach: {fix_steps}. This should resolve the {problem_description}.",
                "Here are several ways to solve this problem: {fix_options}. I'd recommend starting with {preferred_fix}.",
                "The most effective solution would be to {primary_fix}. Additionally, consider {additional_suggestions}."
            ],
            QueryIntent.CODE_EXPLANATION: [
                "This code works by {main_explanation}. Let me break it down step by step: {step_breakdown}.",
                "The purpose of this code is to {purpose}. Here's how each part contributes: {part_explanations}.",
                "This implementation uses {techniques} to achieve {goal}. The key logic is: {key_logic}."
            ],
            QueryIntent.PERFORMANCE_ANALYSIS: [
                "I've identified several performance considerations: {performance_issues}. The main bottleneck appears to be {main_bottleneck}.",
                "Your code's performance could be improved in these areas: {improvement_areas}. The biggest impact would come from {biggest_improvement}.",
                "From a performance perspective, {performance_assessment}. I recommend {optimization_suggestions}."
            ],
            QueryIntent.SECURITY_CHECK: [
                "I've found {security_issue_count} potential security concerns: {security_issues}. The most critical is {critical_issue}.",
                "From a security standpoint, {security_assessment}. Pay particular attention to {security_focus_areas}.",
                "The security analysis reveals {security_findings}. I recommend addressing {priority_fixes} first."
            ],
            QueryIntent.BEST_PRACTICES: [
                "Based on best practices, here are some recommendations: {best_practice_suggestions}. The most important change would be {top_priority}.",
                "Your code could be improved by following these conventions: {conventions}. This would enhance {benefits}.",
                "To align with best practices, consider {practice_recommendations}. These changes will improve {quality_aspects}."
            ],
            QueryIntent.REFACTORING: [
                "For refactoring, I suggest {refactoring_approach}. This would improve {refactoring_benefits}.",
                "The code could be restructured by {restructuring_plan}. The main advantages would be {advantages}.",
                "Consider these refactoring opportunities: {refactoring_opportunities}. Start with {starting_point}."
            ],
            QueryIntent.TESTING: [
                "For testing this code, I recommend {testing_strategy}. Here are the key test cases: {test_cases}.",
                "Your testing approach should cover {testing_areas}. The most critical tests are {critical_tests}.",
                "To ensure code quality, implement these tests: {test_recommendations}. Focus especially on {test_focus}."
            ],
            QueryIntent.DEBUGGING_HELP: [
                "For debugging, try this approach: {debugging_strategy}. Set breakpoints at {breakpoint_locations}.",
                "To debug this issue, {debugging_steps}. Pay attention to {debugging_focus_areas}.",
                "The debugging process should involve {debugging_process}. Key variables to monitor: {key_variables}."
            ],
            QueryIntent.CODE_GENERATION: [
                "Here's a code implementation for your requirements: {generated_code}. This approach {approach_explanation}.",
                "I've created a solution that {solution_description}. The implementation includes {implementation_features}.",
                "This code template should meet your needs: {code_template}. You can customize it by {customization_options}."
            ]
        }
    
    def _load_code_examples(self):
        """Load code examples for different scenarios"""
        self.code_examples = {
            'error_handling': '''
# Robust error handling
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    result = fallback_value
except Exception as e:
    logger.exception("Unexpected error occurred")
    raise
''',
            'null_check': '''
# Safe null checking
if obj is not None:
    value = obj.attribute
else:
    value = default_value

# Or using getattr
value = getattr(obj, 'attribute', default_value)
''',
            'performance_optimization': '''
# Performance optimization example
# Before: Inefficient loop
result = []
for item in large_list:
    if condition(item):
        result.append(transform(item))

# After: List comprehension
result = [transform(item) for item in large_list if condition(item)]
''',
            'security_fix': '''
# Security improvement
# Before: Vulnerable to injection
query = f"SELECT * FROM users WHERE id = {user_id}"

# After: Parameterized query
query = "SELECT * FROM users WHERE id = %s"
cursor.execute(query, (user_id,))
'''
        }
    
    def _load_follow_up_templates(self):
        """Load follow-up question templates"""
        self.follow_up_templates = {
            QueryIntent.ERROR_ANALYSIS: [
                "Would you like me to show you how to prevent this error in the future?",
                "Do you need help implementing the suggested fix?",
                "Are there other similar errors in your codebase I should check?"
            ],
            QueryIntent.FIX_SUGGESTION: [
                "Would you like me to explain why this fix works?",
                "Do you need help implementing any of these solutions?",
                "Should I check for similar issues in other parts of your code?"
            ],
            QueryIntent.CODE_EXPLANATION: [
                "Would you like me to explain any specific part in more detail?",
                "Do you have questions about how this relates to other parts of your code?",
                "Should I suggest improvements to this code?"
            ],
            QueryIntent.PERFORMANCE_ANALYSIS: [
                "Would you like specific optimization recommendations?",
                "Should I analyze other functions for performance issues?",
                "Do you want to see benchmarking code for these optimizations?"
            ]
        }
    
    def generate_response(self, parsed_query: ParsedQuery, analysis_results: Dict[str, Any],
                         context: Dict[str, Any]) -> DebugResponse:
        """Generate a natural language response"""
        
        intent = parsed_query.intent
        
        # Select appropriate template
        templates = self.response_templates.get(intent, ["I understand you're asking about {query}. Let me help you with that."])
        template = templates[0]  # Use first template for now
        
        # Generate response text
        response_text = self._fill_template(template, parsed_query, analysis_results, context)
        
        # Select relevant code examples
        code_examples = self._select_code_examples(intent, analysis_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(intent, analysis_results)
        
        # Generate follow-up questions
        follow_up_questions = self._generate_follow_ups(intent, context)
        
        return DebugResponse(
            response_text=response_text,
            intent=intent,
            confidence=parsed_query.confidence,
            code_examples=code_examples,
            recommendations=recommendations,
            follow_up_questions=follow_up_questions,
            metadata={
                'analysis_results': analysis_results,
                'context_used': context,
                'template_used': template
            }
        )
    
    def _fill_template(self, template: str, parsed_query: ParsedQuery, 
                      analysis_results: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Fill template with actual data"""
        
        # Extract relevant information for template filling
        template_vars = {}
        
        entities = parsed_query.entities
        
        # Basic template variables
        template_vars['query'] = parsed_query.original_text
        template_vars['error_type'] = entities.get('error_types', ['issue'])[0] if entities.get('error_types') else 'issue'
        
        # Analysis-based variables
        if 'error_analysis' in analysis_results:
            error_analysis = analysis_results['error_analysis']
            template_vars['analysis'] = error_analysis.get('message', 'an issue with your code')
            template_vars['root_cause'] = error_analysis.get('root_cause', {}).get('description', 'a logical error')
            template_vars['common_cause'] = self._get_common_cause(template_vars['error_type'])
            template_vars['specific_analysis'] = error_analysis.get('message', 'there is an issue')
        
        if 'fix_suggestions' in analysis_results:
            fixes = analysis_results['fix_suggestions']
            if fixes:
                template_vars['fix_steps'] = self._format_fix_steps(fixes[:3])
                template_vars['fix_options'] = self._format_fix_options(fixes)
                template_vars['preferred_fix'] = fixes[0].explanation if fixes else 'the first option'
                template_vars['primary_fix'] = fixes[0].explanation if fixes else 'apply appropriate error handling'
                template_vars['additional_suggestions'] = ', '.join([f.explanation for f in fixes[1:3]]) if len(fixes) > 1 else 'adding defensive programming practices'
        
        # Performance-related variables
        if 'performance_analysis' in analysis_results:
            perf = analysis_results['performance_analysis']
            template_vars['performance_issues'] = ', '.join(perf.get('issues', ['no significant issues found']))
            template_vars['main_bottleneck'] = perf.get('bottleneck', 'inefficient algorithms')
            template_vars['improvement_areas'] = ', '.join(perf.get('improvement_areas', ['algorithm optimization']))
            template_vars['biggest_improvement'] = perf.get('biggest_improvement', 'optimizing the main algorithm')
        
        # Security-related variables
        if 'security_analysis' in analysis_results:
            security = analysis_results['security_analysis']
            template_vars['security_issue_count'] = len(security.get('issues', []))
            template_vars['security_issues'] = ', '.join(security.get('issues', ['no issues found']))
            template_vars['critical_issue'] = security.get('critical_issue', 'input validation')
        
        # Default values for missing template variables
        default_values = {
            'analysis': 'your code structure',
            'root_cause': 'a logical issue in the implementation',
            'common_cause': 'incorrect usage of the API or syntax',
            'specific_analysis': 'the code needs attention',
            'fix_steps': 'review the error message and apply appropriate fixes',
            'fix_options': 'several approaches to resolve this',
            'preferred_fix': 'the most straightforward solution',
            'primary_fix': 'address the immediate issue',
            'additional_suggestions': 'improve error handling and add validation',
            'performance_issues': 'potential optimization opportunities',
            'main_bottleneck': 'algorithm complexity',
            'improvement_areas': 'code efficiency',
            'biggest_improvement': 'algorithm optimization',
            'security_issue_count': 0,
            'security_issues': 'no critical issues found',
            'critical_issue': 'input validation'
        }
        
        # Fill in any missing variables
        for key, default_value in default_values.items():
            if key not in template_vars:
                template_vars[key] = default_value
        
        # Replace template placeholders
        try:
            formatted_response = template.format(**template_vars)
        except KeyError as e:
            # Fallback if template variable is missing
            formatted_response = f"I've analyzed your query about {parsed_query.original_text}. Let me help you with this issue."
        
        return formatted_response
    
    def _get_common_cause(self, error_type: str) -> str:
        """Get common cause description for error type"""
        common_causes = {
            'attributeerror': "trying to access an attribute that doesn't exist or on a None object",
            'keyerror': "accessing a dictionary key that doesn't exist",
            'indexerror': "trying to access a list index that's out of bounds",
            'typeerror': "using an object of the wrong type for an operation",
            'valueerror': "passing a value of the correct type but inappropriate value",
            'nameerror': "trying to use a variable that hasn't been defined",
            'syntaxerror': "incorrect Python syntax in the code",
            'indentationerror': "incorrect indentation in Python code"
        }
        
        return common_causes.get(error_type.lower(), "an issue with how the code is structured or used")
    
    def _format_fix_steps(self, fixes: List) -> str:
        """Format fix suggestions as numbered steps"""
        if not fixes:
            return "review the code and apply appropriate corrections"
        
        steps = []
        for i, fix in enumerate(fixes[:3], 1):
            explanation = getattr(fix, 'explanation', str(fix))
            steps.append(f"{i}. {explanation}")
        
        return "; ".join(steps)
    
    def _format_fix_options(self, fixes: List) -> str:
        """Format fix suggestions as options"""
        if not fixes:
            return "multiple approaches to resolve this issue"
        
        options = []
        for i, fix in enumerate(fixes[:3], 1):
            explanation = getattr(fix, 'explanation', str(fix))
            options.append(f"Option {i}: {explanation}")
        
        return "; ".join(options)
    
    def _select_code_examples(self, intent: QueryIntent, analysis_results: Dict[str, Any]) -> List[str]:
        """Select relevant code examples"""
        examples = []
        
        if intent == QueryIntent.ERROR_ANALYSIS or intent == QueryIntent.FIX_SUGGESTION:
            if 'error_analysis' in analysis_results:
                error_type = analysis_results['error_analysis'].get('error_type', '').lower()
                
                if 'attributeerror' in error_type or 'none' in error_type:
                    examples.append(self.code_examples['null_check'])
                else:
                    examples.append(self.code_examples['error_handling'])
        
        elif intent == QueryIntent.PERFORMANCE_ANALYSIS:
            examples.append(self.code_examples['performance_optimization'])
        
        elif intent == QueryIntent.SECURITY_CHECK:
            examples.append(self.code_examples['security_fix'])
        
        return examples
    
    def _generate_recommendations(self, intent: QueryIntent, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if intent == QueryIntent.ERROR_ANALYSIS:
            recommendations.extend([
                "Add proper error handling with try-except blocks",
                "Validate input parameters before processing",
                "Use defensive programming techniques"
            ])
        
        elif intent == QueryIntent.PERFORMANCE_ANALYSIS:
            recommendations.extend([
                "Profile your code to identify actual bottlenecks",
                "Use appropriate data structures for your use case",
                "Consider caching for expensive operations"
            ])
        
        elif intent == QueryIntent.SECURITY_CHECK:
            recommendations.extend([
                "Validate and sanitize all user inputs",
                "Use parameterized queries for database operations",
                "Implement proper authentication and authorization"
            ])
        
        # Add analysis-specific recommendations
        if 'recommendations' in analysis_results:
            recommendations.extend(analysis_results['recommendations'][:5])
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _generate_follow_ups(self, intent: QueryIntent, context: Dict[str, Any]) -> List[str]:
        """Generate relevant follow-up questions"""
        templates = self.follow_up_templates.get(intent, [])
        
        # Add context-specific follow-ups
        follow_ups = templates.copy()
        
        if context.get('active_files'):
            follow_ups.append("Should I analyze other files in your project?")
        
        if context.get('active_errors'):
            follow_ups.append("Would you like to see patterns across your error history?")
        
        return follow_ups[:3]  # Limit to top 3 follow-ups


class NaturalLanguageDebugInterface:
    """Main interface for natural language debugging"""
    
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.context_manager = ContextManager()
        self.response_generator = ResponseGenerator()
        self.query_history = []
        
    def process_debug_query(self, query: str, code_context: Optional[str] = None,
                           error_context: Optional[Dict[str, Any]] = None) -> DebugResponse:
        """Process a natural language debug query"""
        
        # Parse the query
        parsed_query = self._parse_query(query)
        
        # Get relevant context
        context = self.context_manager.get_relevant_context(query)
        
        # Add provided context
        if code_context:
            context['code_context'] = code_context
        if error_context:
            context['error_context'] = error_context
        
        # Analyze based on intent
        analysis_results = self._perform_analysis(parsed_query, context)
        
        # Generate response
        response = self.response_generator.generate_response(parsed_query, analysis_results, context)
        
        # Update conversation context
        self.context_manager.update_context(query, parsed_query, response)
        
        # Store in query history
        self.query_history.append({
            'timestamp': datetime.now(),
            'query': query,
            'parsed_query': parsed_query,
            'response': response
        })
        
        return response
    
    def _parse_query(self, query: str) -> ParsedQuery:
        """Parse natural language query"""
        
        # Classify intent
        intent, confidence = self.intent_classifier.classify_intent(query)
        
        # Extract entities
        entities = self.entity_extractor.extract_entities(query)
        
        # Determine query type
        query_type = self._determine_query_type(query, intent)
        
        # Extract parameters
        parameters = self._extract_parameters(query, entities)
        
        # Find context references
        context_references = self._find_context_references(query)
        
        return ParsedQuery(
            original_text=query,
            intent=intent,
            entities=entities,
            confidence=confidence,
            query_type=query_type,
            parameters=parameters,
            context_references=context_references
        )
    
    def _determine_query_type(self, query: str, intent: QueryIntent) -> str:
        """Determine the type of query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['why', 'what', 'how']):
            return 'question'
        elif any(word in query_lower for word in ['fix', 'solve', 'help']):
            return 'request'
        elif any(word in query_lower for word in ['check', 'analyze', 'review']):
            return 'analysis'
        else:
            return 'general'
    
    def _extract_parameters(self, query: str, entities: Dict[str, List[str]]) -> Dict[str, Any]:
        """Extract parameters from query"""
        parameters = {}
        
        # Extract file references
        if 'file_names' in entities and entities['file_names']:
            parameters['target_files'] = entities['file_names']
        
        # Extract line numbers
        if 'line_numbers' in entities and entities['line_numbers']:
            parameters['line_numbers'] = [int(n) for n in entities['line_numbers']]
        
        # Extract specific error types
        if 'error_types' in entities and entities['error_types']:
            parameters['error_types'] = entities['error_types']
        
        # Extract function/method names
        if 'function_names' in entities and entities['function_names']:
            parameters['function_names'] = entities['function_names']
        
        return parameters
    
    def _find_context_references(self, query: str) -> List[str]:
        """Find references to previous context"""
        references = []
        query_lower = query.lower()
        
        # Pronoun references
        if any(word in query_lower for word in ['this', 'that', 'it']):
            references.append('pronoun_reference')
        
        # Temporal references
        if any(word in query_lower for word in ['previous', 'last', 'earlier', 'before']):
            references.append('temporal_reference')
        
        # Same/similar references
        if any(word in query_lower for word in ['same', 'similar', 'like']):
            references.append('similarity_reference')
        
        return references
    
    def _perform_analysis(self, parsed_query: ParsedQuery, 
                         context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform analysis based on query intent"""
        analysis_results = {}
        
        intent = parsed_query.intent
        
        if intent == QueryIntent.ERROR_ANALYSIS:
            analysis_results['error_analysis'] = self._analyze_error(parsed_query, context)
        
        elif intent == QueryIntent.FIX_SUGGESTION:
            analysis_results['fix_suggestions'] = self._suggest_fixes(parsed_query, context)
        
        elif intent == QueryIntent.CODE_EXPLANATION:
            analysis_results['code_explanation'] = self._explain_code(parsed_query, context)
        
        elif intent == QueryIntent.PERFORMANCE_ANALYSIS:
            analysis_results['performance_analysis'] = self._analyze_performance(parsed_query, context)
        
        elif intent == QueryIntent.SECURITY_CHECK:
            analysis_results['security_analysis'] = self._check_security(parsed_query, context)
        
        elif intent == QueryIntent.BEST_PRACTICES:
            analysis_results['best_practices'] = self._check_best_practices(parsed_query, context)
        
        elif intent == QueryIntent.REFACTORING:
            analysis_results['refactoring_suggestions'] = self._suggest_refactoring(parsed_query, context)
        
        elif intent == QueryIntent.TESTING:
            analysis_results['testing_suggestions'] = self._suggest_testing(parsed_query, context)
        
        return analysis_results
    
    def _analyze_error(self, parsed_query: ParsedQuery, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze error from query"""
        error_analysis = {
            'error_type': 'unknown',
            'message': 'Error analysis requested',
            'root_cause': {'description': 'Unable to determine without more context'},
            'suggestions': []
        }
        
        # Extract error information from entities
        entities = parsed_query.entities
        
        if 'error_types' in entities and entities['error_types']:
            error_analysis['error_type'] = entities['error_types'][0]
        
        # Use error context if provided
        if 'error_context' in context:
            error_ctx = context['error_context']
            error_analysis['error_type'] = error_ctx.get('error_type', error_analysis['error_type'])
            error_analysis['message'] = error_ctx.get('message', error_analysis['message'])
        
        return error_analysis
    
    def _suggest_fixes(self, parsed_query: ParsedQuery, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Suggest fixes based on query"""
        # This would integrate with the IntelligentFixGenerator
        # For now, return placeholder suggestions
        fixes = [
            {
                'explanation': 'Add proper error handling to catch and manage exceptions',
                'confidence': 0.8,
                'code_example': 'try:\n    # your code\nexcept Exception as e:\n    handle_error(e)'
            },
            {
                'explanation': 'Validate input parameters before processing',
                'confidence': 0.7,
                'code_example': 'if not isinstance(input_value, expected_type):\n    raise ValueError("Invalid input")'
            }
        ]
        
        return fixes
    
    def _explain_code(self, parsed_query: ParsedQuery, context: Dict[str, Any]) -> Dict[str, Any]:
        """Explain code functionality"""
        explanation = {
            'summary': 'Code explanation requested',
            'detailed_explanation': 'Without specific code context, I can provide general guidance on code structure and best practices.',
            'key_concepts': [],
            'complexity_assessment': 'moderate'
        }
        
        if 'code_context' in context:
            # Analyze the provided code
            code = context['code_context']
            explanation['summary'] = f'This code appears to be a {self._identify_code_type(code)} implementation'
            explanation['detailed_explanation'] = self._generate_code_explanation(code)
        
        return explanation
    
    def _analyze_performance(self, parsed_query: ParsedQuery, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance aspects"""
        performance_analysis = {
            'issues': ['No specific performance issues identified without code context'],
            'bottleneck': 'Unable to determine without profiling data',
            'improvement_areas': ['General optimization recommendations'],
            'biggest_improvement': 'Code profiling and optimization'
        }
        
        return performance_analysis
    
    def _check_security(self, parsed_query: ParsedQuery, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check security issues"""
        security_analysis = {
            'issues': [],
            'critical_issue': 'No critical issues found without code context',
            'recommendations': ['Implement input validation', 'Use parameterized queries', 'Add authentication']
        }
        
        return security_analysis
    
    def _check_best_practices(self, parsed_query: ParsedQuery, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check code against best practices"""
        best_practices = {
            'violations': [],
            'recommendations': ['Follow PEP 8 style guidelines', 'Add docstrings', 'Use meaningful variable names'],
            'score': 0.7
        }
        
        return best_practices
    
    def _suggest_refactoring(self, parsed_query: ParsedQuery, context: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest refactoring opportunities"""
        refactoring = {
            'opportunities': ['Extract complex functions', 'Reduce code duplication', 'Improve naming'],
            'priority': 'medium',
            'benefits': ['Improved maintainability', 'Better readability', 'Easier testing']
        }
        
        return refactoring
    
    def _suggest_testing(self, parsed_query: ParsedQuery, context: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest testing strategies"""
        testing = {
            'test_types': ['unit tests', 'integration tests', 'edge case tests'],
            'test_cases': ['Normal input', 'Invalid input', 'Boundary conditions'],
            'framework_recommendations': ['pytest', 'unittest', 'mock']
        }
        
        return testing
    
    def _identify_code_type(self, code: str) -> str:
        """Identify the type of code"""
        if 'class' in code:
            return 'class definition'
        elif 'def' in code:
            return 'function definition'
        elif 'import' in code:
            return 'module import'
        else:
            return 'code snippet'
    
    def _generate_code_explanation(self, code: str) -> str:
        """Generate explanation for code"""
        # This would use more sophisticated analysis
        # For now, provide a basic explanation
        lines = code.splitlines()
        explanation = f"This code consists of {len(lines)} lines. "
        
        if 'def' in code:
            explanation += "It defines functions for specific operations. "
        if 'class' in code:
            explanation += "It includes class definitions for object-oriented programming. "
        if 'try:' in code:
            explanation += "It includes error handling with try-except blocks. "
        if 'import' in code:
            explanation += "It imports external modules or libraries. "
        
        return explanation
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of the conversation"""
        summary = {
            'total_queries': len(self.query_history),
            'intent_distribution': Counter(),
            'most_common_entities': Counter(),
            'session_duration': 0,
            'user_satisfaction_indicators': []
        }
        
        if not self.query_history:
            return summary
        
        # Calculate session duration
        first_query = self.query_history[0]['timestamp']
        last_query = self.query_history[-1]['timestamp']
        summary['session_duration'] = (last_query - first_query).total_seconds()
        
        # Analyze intent distribution
        for query_record in self.query_history:
            intent = query_record['parsed_query'].intent
            summary['intent_distribution'][intent.value] += 1
        
        # Analyze entity usage
        for query_record in self.query_history:
            entities = query_record['parsed_query'].entities
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    summary['most_common_entities'][entity] += 1
        
        return summary


# Export main classes
__all__ = [
    'NaturalLanguageDebugInterface',
    'IntentClassifier',
    'EntityExtractor',
    'ContextManager',
    'ResponseGenerator',
    'QueryIntent',
    'ParsedQuery',
    'DebugResponse'
]