"""
ABOV3 Genesis - Enhanced ML-Powered Debugger
Integration of all ML/AI debugging capabilities with existing debug engine
"""

import sys
import os
import json
import logging
import traceback
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

# Import existing debug engine
try:
    from .enterprise_debugger import EnterpriseDebugEngine, get_debug_engine
    from .ml_debug_engine import (
        TransformerErrorAnalyzer, SemanticCodeAnalyzer, 
        IntelligentFixGenerator, PredictiveDebugger, AutoLearningSystem
    )
    from .nl_debug_interface import NaturalLanguageDebugInterface, QueryIntent
    from .automated_test_generator import TestSuiteGenerator, generate_tests_for_code
except ImportError:
    # Fallback imports for standalone usage
    pass

# ML imports with fallbacks
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class MLDebugSession:
    """Enhanced debug session with ML capabilities"""
    session_id: str
    created_at: datetime
    code_context: str
    error_history: List[Dict[str, Any]] = field(default_factory=list)
    fix_applications: List[Dict[str, Any]] = field(default_factory=list)
    user_interactions: List[Dict[str, Any]] = field(default_factory=list)
    ml_insights: Dict[str, Any] = field(default_factory=dict)
    performance_data: Dict[str, Any] = field(default_factory=dict)
    test_results: Dict[str, Any] = field(default_factory=dict)


class EnhancedMLDebugger:
    """
    Claude-level intelligent debugger integrating all ML/AI capabilities
    """
    
    def __init__(self):
        # Core debugging engine
        self.base_debugger = get_debug_engine() if 'get_debug_engine' in globals() else None
        
        # ML/AI components
        self.error_analyzer = TransformerErrorAnalyzer() if 'TransformerErrorAnalyzer' in globals() else None
        self.semantic_analyzer = SemanticCodeAnalyzer() if 'SemanticCodeAnalyzer' in globals() else None
        self.fix_generator = IntelligentFixGenerator() if 'IntelligentFixGenerator' in globals() else None
        self.predictive_debugger = PredictiveDebugger() if 'PredictiveDebugger' in globals() else None
        self.learning_system = AutoLearningSystem() if 'AutoLearningSystem' in globals() else None
        self.nl_interface = NaturalLanguageDebugInterface() if 'NaturalLanguageDebugInterface' in globals() else None
        self.test_generator = TestSuiteGenerator() if 'TestSuiteGenerator' in globals() else None
        
        # Session management
        self.active_sessions = {}
        self.current_session_id = None
        
        # Configuration
        self.config = {
            'enable_ml_analysis': True,
            'enable_predictive_debugging': True,
            'enable_auto_learning': True,
            'enable_nl_interface': True,
            'enable_test_generation': True,
            'confidence_threshold': 0.5,
            'max_fix_suggestions': 5,
            'auto_apply_high_confidence_fixes': False
        }
        
        # Initialize logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup enhanced logging for ML debugging"""
        self.logger = logging.getLogger('enhanced_ml_debugger')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def create_debug_session(self, code: str, file_path: str = "", 
                           session_name: str = "") -> str:
        """Create a new enhanced debug session"""
        import uuid
        session_id = str(uuid.uuid4())
        
        session = MLDebugSession(
            session_id=session_id,
            created_at=datetime.now(),
            code_context=code
        )
        
        self.active_sessions[session_id] = session
        self.current_session_id = session_id
        
        # Initialize session with ML analysis
        if self.config['enable_ml_analysis']:
            self._initialize_session_analysis(session, code, file_path)
        
        self.logger.info(f"Created enhanced debug session: {session_id}")
        return session_id
    
    def _initialize_session_analysis(self, session: MLDebugSession, 
                                   code: str, file_path: str):
        """Initialize session with comprehensive ML analysis"""
        try:
            # Semantic code analysis
            if self.semantic_analyzer:
                semantic_analysis = self.semantic_analyzer.analyze_code_semantics(code, file_path)
                session.ml_insights['semantic_analysis'] = semantic_analysis
            
            # Predictive analysis
            if self.predictive_debugger:
                health_analysis = self.predictive_debugger.analyze_code_health(code, {})
                session.ml_insights['health_analysis'] = health_analysis
            
            # Test generation analysis
            if self.test_generator and self.config['enable_test_generation']:
                test_analysis = self.test_generator.code_analyzer.analyze_code(code, file_path)
                session.ml_insights['test_analysis'] = test_analysis
            
        except Exception as e:
            self.logger.warning(f"Session analysis failed: {e}")
    
    def analyze_error_with_ml(self, exception: Exception, 
                             code_context: str = "", **kwargs) -> Dict[str, Any]:
        """Analyze error using all ML capabilities"""
        analysis_result = {
            'timestamp': datetime.now(),
            'error_type': type(exception).__name__,
            'error_message': str(exception),
            'basic_analysis': {},
            'ml_analysis': {},
            'fix_suggestions': [],
            'predictions': {},
            'confidence': 0.0
        }
        
        try:
            # Basic analysis using existing debugger
            if self.base_debugger:
                basic_analysis = self.base_debugger.analyze_exception(exception, **kwargs)
                analysis_result['basic_analysis'] = basic_analysis
            
            # Enhanced ML analysis
            if self.error_analyzer:
                # Create error embedding
                error_context = f"{code_context}\n{traceback.format_exc()}"
                error_embedding = self.error_analyzer.encode_error(str(exception), error_context)
                
                # Find similar errors
                similar_errors = self.error_analyzer.find_similar_errors(error_embedding)
                
                analysis_result['ml_analysis'] = {
                    'similar_errors': similar_errors,
                    'error_patterns': self._identify_error_patterns(exception),
                    'context_analysis': self._analyze_error_context(exception, code_context)
                }
            
            # Generate intelligent fix suggestions
            if self.fix_generator:
                fix_suggestions = self.fix_generator.generate_fix_suggestions(
                    str(exception), code_context, type(exception).__name__
                )
                analysis_result['fix_suggestions'] = [
                    {
                        'fix_id': fix.fix_id,
                        'explanation': fix.explanation,
                        'code': fix.fixed_code,
                        'confidence': fix.confidence
                    }
                    for fix in fix_suggestions
                ]
            
            # Calculate overall confidence
            analysis_result['confidence'] = self._calculate_analysis_confidence(analysis_result)
            
            # Store in current session
            if self.current_session_id:
                session = self.active_sessions.get(self.current_session_id)
                if session:
                    session.error_history.append(analysis_result)
            
        except Exception as e:
            self.logger.error(f"ML error analysis failed: {e}")
            analysis_result['error'] = str(e)
        
        return analysis_result
    
    def _identify_error_patterns(self, exception: Exception) -> List[Dict[str, Any]]:
        """Identify patterns in the error"""
        patterns = []
        
        error_msg = str(exception)
        error_type = type(exception).__name__
        
        # Common error patterns
        pattern_definitions = [
            {
                'name': 'null_reference',
                'pattern': r"'NoneType'.*has no attribute",
                'description': 'Null reference error - object is None'
            },
            {
                'name': 'missing_key',
                'pattern': r'KeyError.*',
                'description': 'Dictionary key not found'
            },
            {
                'name': 'index_bounds',
                'pattern': r'list index out of range',
                'description': 'List index exceeds bounds'
            },
            {
                'name': 'type_mismatch',
                'pattern': r'TypeError.*',
                'description': 'Incorrect type used for operation'
            }
        ]
        
        for pattern_def in pattern_definitions:
            if re.search(pattern_def['pattern'], error_msg, re.IGNORECASE):
                patterns.append({
                    'pattern_name': pattern_def['name'],
                    'description': pattern_def['description'],
                    'confidence': 0.8
                })
        
        return patterns
    
    def _analyze_error_context(self, exception: Exception, 
                             code_context: str) -> Dict[str, Any]:
        """Analyze the context around the error"""
        context_analysis = {
            'code_complexity': 0,
            'error_location': {},
            'surrounding_patterns': [],
            'risk_factors': []
        }
        
        # Analyze code complexity around error
        if code_context:
            lines = code_context.split('\n')
            context_analysis['code_complexity'] = len([line for line in lines if line.strip()])
            
            # Look for common risk patterns
            risk_patterns = ['eval(', 'exec(', 'import *', 'global ', '__import__']
            for pattern in risk_patterns:
                if pattern in code_context:
                    context_analysis['risk_factors'].append(f"Risky pattern found: {pattern}")
        
        # Analyze traceback if available
        if exception.__traceback__:
            tb = traceback.extract_tb(exception.__traceback__)
            if tb:
                last_frame = tb[-1]
                context_analysis['error_location'] = {
                    'file': last_frame.filename,
                    'line': last_frame.lineno,
                    'function': last_frame.name,
                    'code': last_frame.line
                }
        
        return context_analysis
    
    def _calculate_analysis_confidence(self, analysis_result: Dict[str, Any]) -> float:
        """Calculate overall confidence in the analysis"""
        confidence_factors = []
        
        # Basic analysis confidence
        if 'basic_analysis' in analysis_result and analysis_result['basic_analysis']:
            confidence_factors.append(0.6)
        
        # ML analysis confidence
        ml_analysis = analysis_result.get('ml_analysis', {})
        if ml_analysis.get('similar_errors'):
            confidence_factors.append(0.8)
        
        if ml_analysis.get('error_patterns'):
            confidence_factors.append(0.7)
        
        # Fix suggestions confidence
        fix_suggestions = analysis_result.get('fix_suggestions', [])
        if fix_suggestions and len(fix_suggestions) > 0:
            avg_fix_confidence = sum(fix.get('confidence', 0) for fix in fix_suggestions) / len(fix_suggestions) if fix_suggestions else 0
            confidence_factors.append(avg_fix_confidence)
        
        # Calculate weighted average
        if confidence_factors and len(confidence_factors) > 0:
            return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0
        else:
            return 0.3  # Low confidence without ML data
    
    def ask_natural_language(self, query: str, **context) -> Dict[str, Any]:
        """Process natural language debug query"""
        if not self.nl_interface:
            return {
                'error': 'Natural language interface not available',
                'fallback_response': 'Please rephrase your query or provide specific error details.'
            }
        
        try:
            # Add session context
            if self.current_session_id:
                session = self.active_sessions.get(self.current_session_id)
                if session:
                    context.update({
                        'code_context': session.code_context,
                        'error_history': session.error_history,
                        'ml_insights': session.ml_insights
                    })
            
            # Process the query
            response = self.nl_interface.process_debug_query(
                query,
                code_context=context.get('code_context'),
                error_context=context.get('error_context')
            )
            
            # Store interaction in session
            if self.current_session_id:
                session = self.active_sessions.get(self.current_session_id)
                if session:
                    session.user_interactions.append({
                        'timestamp': datetime.now(),
                        'query': query,
                        'response': response.response_text,
                        'intent': response.intent.value if response.intent else 'unknown',
                        'confidence': response.confidence
                    })
            
            return {
                'response': response.response_text,
                'intent': response.intent.value if response.intent else 'unknown',
                'confidence': response.confidence,
                'code_examples': response.code_examples,
                'recommendations': response.recommendations,
                'follow_up_questions': response.follow_up_questions
            }
        
        except Exception as e:
            self.logger.error(f"Natural language processing failed: {e}")
            return {
                'error': str(e),
                'fallback_response': 'I encountered an error processing your query. Please try rephrasing or provide more specific details.'
            }
    
    def generate_tests_for_session(self, session_id: str = None) -> Dict[str, Any]:
        """Generate tests for the current session code"""
        session_id = session_id or self.current_session_id
        
        if not session_id or session_id not in self.active_sessions:
            return {'error': 'No active session found'}
        
        session = self.active_sessions[session_id]
        
        if not self.test_generator:
            return {'error': 'Test generator not available'}
        
        try:
            # Generate comprehensive test suite
            test_suite = self.test_generator.generate_test_suite(
                session.code_context,
                file_path="",  # Session doesn't have file path
                test_types=None  # Use default test types
            )
            
            # Store in session
            session.test_results = test_suite
            
            return {
                'test_suite': test_suite,
                'recommendations': self.test_generator.generate_test_recommendations(test_suite),
                'statistics': test_suite.get('statistics', {}),
                'coverage_report': test_suite.get('coverage_report', {})
            }
        
        except Exception as e:
            self.logger.error(f"Test generation failed: {e}")
            return {'error': str(e)}
    
    def apply_fix_suggestion(self, fix_id: str, user_feedback: float = 0.0) -> Dict[str, Any]:
        """Apply a fix suggestion and learn from the outcome"""
        if not self.current_session_id:
            return {'error': 'No active session'}
        
        session = self.active_sessions.get(self.current_session_id)
        if not session:
            return {'error': 'Session not found'}
        
        # Find the fix suggestion
        fix_suggestion = None
        for error_entry in session.error_history:
            for fix in error_entry.get('fix_suggestions', []):
                if fix.get('fix_id') == fix_id:
                    fix_suggestion = fix
                    break
        
        if not fix_suggestion:
            return {'error': 'Fix suggestion not found'}
        
        try:
            # Record fix application
            fix_application = {
                'timestamp': datetime.now(),
                'fix_id': fix_id,
                'fix_code': fix_suggestion.get('code', ''),
                'user_feedback': user_feedback,
                'applied': True
            }
            
            session.fix_applications.append(fix_application)
            
            # Learn from the fix application
            if self.fix_generator and self.learning_system:
                # This would integrate with actual code application
                success = user_feedback > 0.5  # Simple success metric
                
                self.fix_generator.learn_from_fix(
                    error_msg="",  # Would get from context
                    error_type="",  # Would get from context
                    original_code="",  # Would get from context
                    fixed_code=fix_suggestion.get('code', ''),
                    success=success,
                    user_feedback=user_feedback
                )
            
            return {
                'success': True,
                'fix_applied': fix_suggestion,
                'learning_recorded': True
            }
        
        except Exception as e:
            self.logger.error(f"Fix application failed: {e}")
            return {'error': str(e)}
    
    def get_predictive_insights(self, code: str) -> Dict[str, Any]:
        """Get predictive insights for code"""
        if not self.predictive_debugger:
            return {'error': 'Predictive debugger not available'}
        
        try:
            # Analyze code health
            health_analysis = self.predictive_debugger.analyze_code_health(code, {})
            
            # Generate insights
            insights = {
                'health_score': health_analysis.get('overall_score', 0.0),
                'risk_factors': health_analysis.get('risk_factors', []),
                'anomalies': health_analysis.get('anomalies', []),
                'predictions': health_analysis.get('predictions', {}),
                'recommendations': health_analysis.get('recommendations', []),
                'confidence': health_analysis.get('confidence', 0.0)
            }
            
            return insights
        
        except Exception as e:
            self.logger.error(f"Predictive analysis failed: {e}")
            return {'error': str(e)}
    
    def get_session_summary(self, session_id: str = None) -> Dict[str, Any]:
        """Get comprehensive session summary"""
        session_id = session_id or self.current_session_id
        
        if not session_id or session_id not in self.active_sessions:
            return {'error': 'Session not found'}
        
        session = self.active_sessions[session_id]
        
        summary = {
            'session_id': session_id,
            'created_at': session.created_at,
            'duration': (datetime.now() - session.created_at).total_seconds(),
            'error_count': len(session.error_history),
            'fix_applications': len(session.fix_applications),
            'user_interactions': len(session.user_interactions),
            'ml_insights_available': bool(session.ml_insights),
            'test_results_available': bool(session.test_results)
        }
        
        # Add error analysis summary
        if session.error_history:
            error_types = [error['error_type'] for error in session.error_history]
            summary['error_type_distribution'] = {
                error_type: error_types.count(error_type)
                for error_type in set(error_types)
            }
            
            # Average confidence
            confidences = [error.get('confidence', 0) for error in session.error_history]
            summary['average_analysis_confidence'] = sum(confidences) / len(confidences) if confidences else 0
        
        # Add interaction analysis
        if session.user_interactions:
            intents = [interaction.get('intent', 'unknown') for interaction in session.user_interactions]
            summary['interaction_intent_distribution'] = {
                intent: intents.count(intent)
                for intent in set(intents)
            }
        
        # Add ML insights summary
        if session.ml_insights:
            summary['ml_insights_summary'] = {
                'semantic_analysis': 'semantic_analysis' in session.ml_insights,
                'health_analysis': 'health_analysis' in session.ml_insights,
                'test_analysis': 'test_analysis' in session.ml_insights
            }
            
            # Health score if available
            health_analysis = session.ml_insights.get('health_analysis', {})
            if health_analysis:
                summary['code_health_score'] = health_analysis.get('overall_score', 0.0)
        
        return summary
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from the learning system"""
        if not self.learning_system:
            return {'error': 'Learning system not available'}
        
        try:
            insights = self.learning_system.get_learning_insights()
            statistics = self.learning_system.get_learning_statistics()
            
            return {
                'insights': insights,
                'statistics': statistics,
                'model_update_suggestions': self.learning_system.suggest_model_updates()
            }
        
        except Exception as e:
            self.logger.error(f"Learning insights failed: {e}")
            return {'error': str(e)}
    
    def configure_ml_features(self, **config_updates):
        """Configure ML feature settings"""
        valid_keys = set(self.config.keys())
        provided_keys = set(config_updates.keys())
        
        invalid_keys = provided_keys - valid_keys
        if invalid_keys:
            return {'error': f'Invalid configuration keys: {invalid_keys}'}
        
        # Update configuration
        self.config.update(config_updates)
        
        self.logger.info(f"Configuration updated: {config_updates}")
        
        return {
            'success': True,
            'updated_config': self.config,
            'message': 'ML features configured successfully'
        }
    
    def export_session_data(self, session_id: str = None, format: str = 'json') -> Dict[str, Any]:
        """Export session data for analysis or backup"""
        session_id = session_id or self.current_session_id
        
        if not session_id or session_id not in self.active_sessions:
            return {'error': 'Session not found'}
        
        session = self.active_sessions[session_id]
        
        try:
            # Prepare export data
            export_data = {
                'session_metadata': {
                    'session_id': session.session_id,
                    'created_at': session.created_at.isoformat(),
                    'exported_at': datetime.now().isoformat()
                },
                'code_context': session.code_context,
                'error_history': session.error_history,
                'fix_applications': session.fix_applications,
                'user_interactions': session.user_interactions,
                'ml_insights': session.ml_insights,
                'test_results': session.test_results
            }
            
            if format == 'json':
                # Convert datetime objects to strings for JSON serialization
                def convert_datetime(obj):
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    elif isinstance(obj, dict):
                        return {k: convert_datetime(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_datetime(item) for item in obj]
                    return obj
                
                export_data = convert_datetime(export_data)
                
                return {
                    'format': 'json',
                    'data': export_data,
                    'size_kb': len(json.dumps(export_data)) / 1024
                }
            
            else:
                return {'error': f'Unsupported export format: {format}'}
        
        except Exception as e:
            self.logger.error(f"Session export failed: {e}")
            return {'error': str(e)}


# Global instance
_enhanced_debugger = None

def get_enhanced_debugger() -> EnhancedMLDebugger:
    """Get global enhanced debugger instance"""
    global _enhanced_debugger
    if _enhanced_debugger is None:
        _enhanced_debugger = EnhancedMLDebugger()
    return _enhanced_debugger


# Convenience functions for easy usage
def debug_with_ml(code: str, error: Exception = None) -> Dict[str, Any]:
    """Quick ML-enhanced debugging function"""
    debugger = get_enhanced_debugger()
    session_id = debugger.create_debug_session(code)
    
    if error:
        return debugger.analyze_error_with_ml(error, code)
    else:
        return debugger.get_predictive_insights(code)


def ask_debug_question(question: str, code: str = "", error_context: Dict[str, Any] = None) -> str:
    """Ask a natural language debug question"""
    debugger = get_enhanced_debugger()
    
    if code:
        debugger.create_debug_session(code)
    
    result = debugger.ask_natural_language(question, error_context=error_context)
    return result.get('response', 'Unable to process question')


def generate_tests(code: str) -> Dict[str, Any]:
    """Generate tests for code using ML"""
    debugger = get_enhanced_debugger()
    session_id = debugger.create_debug_session(code)
    return debugger.generate_tests_for_session(session_id)


# Export main classes and functions
__all__ = [
    'EnhancedMLDebugger',
    'MLDebugSession',
    'get_enhanced_debugger',
    'debug_with_ml',
    'ask_debug_question',
    'generate_tests'
]