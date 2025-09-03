"""
ABOV3 Debug Integration Module
Integrates enterprise debugger with ABOV3 core systems
"""

import os
import sys
import json
import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path
from datetime import datetime

# Import core ABOV3 modules
from abov3.core.assistant import Assistant
from abov3.core.ollama_client import OllamaClient
from abov3.core.error_handler import ErrorHandler
from abov3.core.context_manager import ContextManager

# Import enterprise debugger
from abov3.core.enterprise_debugger import (
    EnterpriseDebugEngine,
    IntelligentErrorAnalyzer,
    InteractiveDebugger,
    NaturalLanguageDebugger,
    get_debug_engine
)


class DebugIntegration:
    """Integrates enterprise debugger with ABOV3 systems"""
    
    def __init__(self, assistant: Optional[Assistant] = None):
        self.assistant = assistant
        self.debug_engine = get_debug_engine()
        self.error_handler = ErrorHandler()
        self.context_manager = ContextManager() if not assistant else assistant.context_manager
        self.logger = self._setup_logger()
        self.debug_mode = False
        self.auto_debug = True
        
        # Create debug session
        self.session_id = self.debug_engine.create_debug_session("ABOV3_Integration")
        
        # Hook into error handler
        self._setup_error_hooks()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup integration logger"""
        logger = logging.getLogger('ABOV3.DebugIntegration')
        logger.setLevel(logging.DEBUG)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _setup_error_hooks(self):
        """Setup hooks to capture errors automatically"""
        original_excepthook = sys.excepthook
        
        def debug_excepthook(exc_type, exc_value, exc_traceback):
            """Custom exception hook for debugging"""
            if self.auto_debug:
                self.logger.error(f"Uncaught exception: {exc_type.__name__}: {exc_value}")
                
                # Analyze with enterprise debugger
                analysis = self.debug_engine.analyze_exception(
                    exc_value,
                    context=self.get_current_context()
                )
                
                # Log analysis results
                self.logger.info(f"Root cause: {analysis.get('root_cause', {}).get('description', 'Unknown')}")
                
                if analysis.get('solutions'):
                    self.logger.info("Suggested solutions:")
                    for solution in analysis['solutions'][:3]:
                        self.logger.info(f"  - {solution}")
                
                # Store in context for assistant
                if self.context_manager:
                    self.context_manager.add_context({
                        'error_analysis': analysis,
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Call original hook
            original_excepthook(exc_type, exc_value, exc_traceback)
        
        sys.excepthook = debug_excepthook
    
    def enable_debug_mode(self):
        """Enable comprehensive debug mode"""
        self.debug_mode = True
        self.logger.info("Debug mode enabled")
        
        # Enable detailed logging
        logging.getLogger().setLevel(logging.DEBUG)
        
        # Start performance monitoring
        if hasattr(self.debug_engine, 'performance_profiler'):
            self.debug_engine.performance_profiler.start_profiling()
    
    def disable_debug_mode(self):
        """Disable debug mode"""
        self.debug_mode = False
        self.logger.info("Debug mode disabled")
        
        # Reset logging
        logging.getLogger().setLevel(logging.INFO)
        
        # Stop performance monitoring
        if hasattr(self.debug_engine, 'performance_profiler'):
            self.debug_engine.performance_profiler.stop_profiling()
    
    def get_current_context(self) -> Dict[str, Any]:
        """Get current debugging context"""
        context = {
            'timestamp': datetime.now().isoformat(),
            'debug_mode': self.debug_mode,
            'session_id': self.session_id
        }
        
        # Add assistant context if available
        if self.assistant:
            context['assistant_state'] = {
                'model': getattr(self.assistant, 'model', 'unknown'),
                'context_size': len(self.assistant.context_manager.context_history)
                if hasattr(self.assistant, 'context_manager') else 0
            }
        
        # Add performance data
        if self.session_id in self.debug_engine.debug_sessions:
            session = self.debug_engine.debug_sessions[self.session_id]
            context['performance_data'] = session.get('performance_data', {})
            context['error_count'] = session.get('error_count', 0)
        
        return context
    
    def debug_assistant_response(self, prompt: str, response: str) -> Dict[str, Any]:
        """Debug assistant responses for quality and issues"""
        analysis = {
            'prompt_analysis': self._analyze_prompt(prompt),
            'response_analysis': self._analyze_response(response),
            'quality_score': 0.0,
            'issues': [],
            'suggestions': []
        }
        
        # Check for common response issues
        if not response or response.strip() == "":
            analysis['issues'].append("Empty response")
            analysis['suggestions'].append("Check model connectivity and prompt formatting")
        
        if len(response) < 50:
            analysis['issues'].append("Response too short")
            analysis['suggestions'].append("Consider more detailed prompts or check model parameters")
        
        if response.count('```') % 2 != 0:
            analysis['issues'].append("Unclosed code blocks")
            analysis['suggestions'].append("Ensure proper markdown formatting in responses")
        
        # Calculate quality score
        quality_factors = {
            'length': min(len(response) / 500, 1.0) * 0.3,
            'structure': (1.0 if '```' in response else 0.5) * 0.2,
            'relevance': self._calculate_relevance(prompt, response) * 0.5
        }
        
        analysis['quality_score'] = sum(quality_factors.values())
        
        # Log if quality is low
        if analysis['quality_score'] < 0.5:
            self.logger.warning(f"Low quality response detected: {analysis['quality_score']:.2f}")
            for issue in analysis['issues']:
                self.logger.warning(f"  Issue: {issue}")
        
        return analysis
    
    def _analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt for potential issues"""
        analysis = {
            'length': len(prompt),
            'complexity': self._calculate_prompt_complexity(prompt),
            'clarity_score': 0.0,
            'suggestions': []
        }
        
        # Check prompt quality
        if len(prompt) < 10:
            analysis['suggestions'].append("Prompt too short - add more context")
        
        if len(prompt) > 2000:
            analysis['suggestions'].append("Prompt very long - consider breaking into smaller parts")
        
        if prompt.isupper():
            analysis['suggestions'].append("Avoid all caps - use normal case")
        
        # Calculate clarity score
        clarity_factors = {
            'punctuation': (1.0 if any(p in prompt for p in '.?!') else 0.5) * 0.3,
            'length_appropriate': (1.0 if 20 < len(prompt) < 500 else 0.5) * 0.3,
            'structure': (1.0 if '\n' in prompt or len(prompt.split()) > 5 else 0.5) * 0.4
        }
        
        analysis['clarity_score'] = sum(clarity_factors.values())
        
        return analysis
    
    def _analyze_response(self, response: str) -> Dict[str, Any]:
        """Analyze response for quality and issues"""
        analysis = {
            'length': len(response),
            'code_blocks': response.count('```') // 2,
            'lists': response.count('\n-') + response.count('\n*') + response.count('\n1.'),
            'headings': response.count('\n#'),
            'completeness': 1.0
        }
        
        # Check for incomplete responses
        if response.endswith('...') or response.endswith('etc'):
            analysis['completeness'] = 0.7
        
        if any(phrase in response.lower() for phrase in ['i cannot', 'i am unable', 'i don\'t']):
            analysis['completeness'] = 0.5
        
        return analysis
    
    def _calculate_prompt_complexity(self, prompt: str) -> float:
        """Calculate prompt complexity score"""
        factors = {
            'length': min(len(prompt) / 200, 1.0) * 0.3,
            'questions': min(prompt.count('?') / 3, 1.0) * 0.2,
            'technical_terms': self._count_technical_terms(prompt) * 0.3,
            'structure': (1.0 if '\n' in prompt else 0.5) * 0.2
        }
        
        return sum(factors.values())
    
    def _count_technical_terms(self, text: str) -> float:
        """Count technical terms in text"""
        technical_terms = [
            'function', 'class', 'method', 'variable', 'debug', 'error',
            'exception', 'performance', 'memory', 'optimize', 'algorithm',
            'database', 'api', 'framework', 'library', 'module'
        ]
        
        text_lower = text.lower()
        count = sum(1 for term in technical_terms if term in text_lower)
        
        return min(count / 5, 1.0)
    
    def _calculate_relevance(self, prompt: str, response: str) -> float:
        """Calculate relevance between prompt and response"""
        # Simple keyword matching for relevance
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                       'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                       'could', 'should', 'may', 'might', 'can', 'could', 'to',
                       'of', 'in', 'on', 'at', 'by', 'for', 'with', 'from'}
        
        prompt_words = prompt_words - common_words
        response_words = response_words - common_words
        
        if not prompt_words:
            return 0.5
        
        # Calculate overlap
        overlap = len(prompt_words & response_words)
        relevance = overlap / len(prompt_words) if prompt_words else 0
        
        return min(relevance * 2, 1.0)  # Scale up and cap at 1.0
    
    def debug_code_generation(self, generated_code: str, language: str = 'python') -> Dict[str, Any]:
        """Debug generated code for issues"""
        self.logger.info(f"Debugging generated {language} code...")
        
        # Use enterprise debugger for analysis
        analysis = self.debug_engine.debug_code(generated_code, language)
        
        # Enhance with additional checks
        enhanced_analysis = {
            **analysis,
            'generation_quality': self._assess_code_quality(generated_code, analysis),
            'ai_specific_issues': self._check_ai_generation_issues(generated_code)
        }
        
        # Log issues
        if not analysis['syntax_check']['valid']:
            self.logger.error("Generated code has syntax errors")
            for error in analysis['syntax_check']['errors']:
                self.logger.error(f"  Line {error['line']}: {error['message']}")
        
        if analysis['security_analysis']['high_risk']:
            self.logger.warning("Security issues in generated code:")
            for risk in analysis['security_analysis']['high_risk']:
                self.logger.warning(f"  - {risk}")
        
        return enhanced_analysis
    
    def _assess_code_quality(self, code: str, analysis: Dict[str, Any]) -> float:
        """Assess overall code quality"""
        quality_factors = {
            'syntax_valid': 1.0 if analysis['syntax_check']['valid'] else 0.0,
            'no_security_issues': 1.0 if not analysis['security_analysis']['high_risk'] else 0.0,
            'low_complexity': 1.0 if analysis['complexity_analysis']['cyclomatic_complexity'] < 10 else 0.5,
            'proper_structure': 1.0 if analysis['complexity_analysis']['function_count'] > 0 else 0.5,
            'no_unused_vars': 1.0 if not analysis['static_analysis']['unused_variables'] else 0.7
        }
        
        weights = {
            'syntax_valid': 0.3,
            'no_security_issues': 0.2,
            'low_complexity': 0.2,
            'proper_structure': 0.15,
            'no_unused_vars': 0.15
        }
        
        quality_score = sum(quality_factors[key] * weights[key] for key in quality_factors)
        
        return quality_score
    
    def _check_ai_generation_issues(self, code: str) -> List[str]:
        """Check for common AI code generation issues"""
        issues = []
        
        # Check for placeholder comments
        if 'TODO' in code or 'FIXME' in code or '...' in code:
            issues.append("Contains incomplete sections or placeholders")
        
        # Check for repetitive patterns (common in AI generation)
        lines = code.splitlines()
        if len(lines) > 10:
            for i in range(len(lines) - 3):
                if lines[i] == lines[i+1] == lines[i+2]:
                    issues.append("Repetitive code detected - possible generation artifact")
                    break
        
        # Check for nonsensical variable names
        import re
        var_pattern = r'\b([a-z]{15,}|[A-Z]{15,})\b'  # Very long single-case names
        if re.search(var_pattern, code):
            issues.append("Unusual variable names detected")
        
        # Check for incomplete functions
        if 'def ' in code and 'pass' in code:
            issues.append("Contains stub functions with 'pass'")
        
        return issues
    
    def process_debug_command(self, command: str) -> str:
        """Process debug commands from user"""
        command_lower = command.lower().strip()
        
        # Parse command
        if command_lower.startswith('debug'):
            parts = command_lower.split()
            
            if len(parts) == 1 or parts[1] == 'status':
                return self._get_debug_status()
            
            elif parts[1] == 'enable':
                self.enable_debug_mode()
                return "Debug mode enabled. All operations will be monitored."
            
            elif parts[1] == 'disable':
                self.disable_debug_mode()
                return "Debug mode disabled."
            
            elif parts[1] == 'report':
                report = self.debug_engine.get_debug_report()
                return self._format_debug_report(report)
            
            elif parts[1] == 'analyze' and len(parts) > 2:
                # Analyze specific aspect
                aspect = parts[2]
                if aspect == 'performance':
                    return self._analyze_performance()
                elif aspect == 'errors':
                    return self._analyze_errors()
                elif aspect == 'memory':
                    return self._analyze_memory()
            
            elif parts[1] == 'query' and len(parts) > 2:
                # Natural language debug query
                query = ' '.join(parts[2:])
                return self.debug_engine.query(query, **self.get_current_context())
        
        # If not a debug command, try natural language processing
        return self.debug_engine.query(command, **self.get_current_context())
    
    def _get_debug_status(self) -> str:
        """Get current debug status"""
        status = f"""
Debug Status:
- Mode: {'Enabled' if self.debug_mode else 'Disabled'}
- Session: {self.session_id[:8]}...
- Auto-debug: {'On' if self.auto_debug else 'Off'}
"""
        
        if self.session_id in self.debug_engine.debug_sessions:
            session = self.debug_engine.debug_sessions[self.session_id]
            status += f"""- Errors captured: {session.get('error_count', 0)}
- Performance profiles: {len(session.get('performance_data', {}))}
- Session duration: {(datetime.now() - session['created']).total_seconds():.1f}s
"""
        
        return status.strip()
    
    def _format_debug_report(self, report: Dict[str, Any]) -> str:
        """Format debug report for display"""
        formatted = f"""
Debug Report - {report.get('session_name', 'Unknown')}
{'='*50}

Duration: {report.get('duration', 0):.1f} seconds

Error Summary:
- Total errors: {report.get('error_summary', {}).get('total_errors', 0)}
"""
        
        error_types = report.get('error_summary', {}).get('error_types', {})
        if error_types:
            formatted += "- Error types:\n"
            for error_type, count in error_types.items():
                formatted += f"  - {error_type}: {count}\n"
        
        perf_summary = report.get('performance_summary', {})
        if perf_summary:
            formatted += f"""
Performance Summary:
- Total execution time: {perf_summary.get('total_execution_time', 0):.3f}s
- Total function calls: {perf_summary.get('total_function_calls', 0)}
"""
            
            slowest = perf_summary.get('slowest_functions', [])
            if slowest:
                formatted += "\nSlowest functions:\n"
                for func in slowest[:3]:
                    formatted += f"  - {func['name']}: {func['total_time']:.3f}s ({func['calls']} calls)\n"
        
        recommendations = report.get('recommendations', [])
        if recommendations:
            formatted += "\nRecommendations:\n"
            for rec in recommendations:
                formatted += f"  - {rec}\n"
        
        return formatted.strip()
    
    def _analyze_performance(self) -> str:
        """Analyze performance metrics"""
        context = self.get_current_context()
        return self.debug_engine.query("What are the performance bottlenecks?", **context)
    
    def _analyze_errors(self) -> str:
        """Analyze captured errors"""
        context = self.get_current_context()
        return self.debug_engine.query("What errors have occurred and how to fix them?", **context)
    
    def _analyze_memory(self) -> str:
        """Analyze memory usage"""
        context = self.get_current_context()
        return self.debug_engine.query("Is there a memory leak or high memory usage?", **context)
    
    async def debug_async_operation(self, coroutine: Any, description: str = "") -> Any:
        """Debug async operations"""
        self.logger.debug(f"Debugging async operation: {description}")
        
        start_time = datetime.now()
        
        try:
            result = await coroutine
            
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.debug(f"Async operation completed in {duration:.3f}s")
            
            return result
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Async operation failed after {duration:.3f}s: {e}")
            
            # Analyze error
            analysis = self.debug_engine.analyze_exception(e, context={
                'operation': description,
                'duration': duration,
                'async': True
            })
            
            # Log solutions
            if analysis.get('solutions'):
                self.logger.info("Suggested fixes:")
                for solution in analysis['solutions'][:3]:
                    self.logger.info(f"  - {solution}")
            
            raise
    
    def export_debug_data(self, filepath: str = None) -> str:
        """Export debug data for analysis"""
        if not filepath:
            filepath = f"debug_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'debug_mode': self.debug_mode,
            'report': self.debug_engine.get_debug_report()
        }
        
        # Add session data
        if self.session_id in self.debug_engine.debug_sessions:
            session = self.debug_engine.debug_sessions[self.session_id]
            export_data['session_data'] = {
                'errors': session.get('errors', []),
                'performance_data': {
                    k: {
                        'function_name': v.function_name,
                        'call_count': v.call_count,
                        'total_time': v.total_time,
                        'avg_time': v.avg_time
                    }
                    for k, v in session.get('performance_data', {}).items()
                }
            }
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Debug data exported to {filepath}")
        return filepath


# Global integration instance
_debug_integration = None

def get_debug_integration(assistant: Optional[Assistant] = None) -> DebugIntegration:
    """Get global debug integration instance"""
    global _debug_integration
    if _debug_integration is None:
        _debug_integration = DebugIntegration(assistant)
    return _debug_integration


# Convenience functions for easy integration
def enable_abov3_debugging():
    """Enable ABOV3 debugging globally"""
    integration = get_debug_integration()
    integration.enable_debug_mode()
    return integration

def debug_command(command: str) -> str:
    """Process debug command"""
    integration = get_debug_integration()
    return integration.process_debug_command(command)

def debug_response(prompt: str, response: str) -> Dict[str, Any]:
    """Debug assistant response"""
    integration = get_debug_integration()
    return integration.debug_assistant_response(prompt, response)

def debug_generated_code(code: str, language: str = 'python') -> Dict[str, Any]:
    """Debug generated code"""
    integration = get_debug_integration()
    return integration.debug_code_generation(code, language)