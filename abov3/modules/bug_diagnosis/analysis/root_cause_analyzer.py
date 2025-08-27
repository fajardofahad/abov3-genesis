"""
Root cause analysis engine for identifying the fundamental cause of bugs
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import re
import json

from ..parsers.error_parser import ParsedError, ErrorType
from ..tracing.execution_tracer import TraceResult

logger = logging.getLogger(__name__)

@dataclass
class RootCauseResult:
    """Result of root cause analysis"""
    root_cause: str
    confidence_score: float
    contributing_factors: List[str]
    similar_patterns: List[Dict[str, Any]]
    suggested_investigation: List[str]
    patterns_matched: List[str]
    causality_chain: List[str]
    
class RootCauseAnalyzer:
    """
    Advanced root cause analysis using pattern matching and AI
    """
    
    def __init__(self):
        self.causality_patterns = self._init_causality_patterns()
        self.investigation_templates = self._init_investigation_templates()
        
    def _init_causality_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize causality patterns for different error types"""
        return {
            ErrorType.NULL_REFERENCE: [
                {
                    "cause": "Variable not initialized",
                    "indicators": ["undefined", "null", "None", "not defined"],
                    "factors": ["Missing initialization", "Async operation not awaited"]
                },
                {
                    "cause": "Optional value not checked",
                    "indicators": ["optional", "maybe", "nullable"],
                    "factors": ["Missing null check", "Unsafe unwrapping"]
                },
                {
                    "cause": "API returned null unexpectedly",
                    "indicators": ["fetch", "request", "response", "api"],
                    "factors": ["API contract change", "Network failure"]
                }
            ],
            ErrorType.INDEX_ERROR: [
                {
                    "cause": "Array/List boundary exceeded",
                    "indicators": ["index", "length", "size", "bounds"],
                    "factors": ["Off-by-one error", "Dynamic size change"]
                },
                {
                    "cause": "Empty collection accessed",
                    "indicators": ["empty", "no elements", "zero length"],
                    "factors": ["Missing empty check", "Race condition"]
                }
            ],
            ErrorType.TYPE_ERROR: [
                {
                    "cause": "Incorrect type assumption",
                    "indicators": ["expected", "got", "cannot convert"],
                    "factors": ["API change", "Type inference failure"]
                },
                {
                    "cause": "Missing type conversion",
                    "indicators": ["cast", "convert", "coerce"],
                    "factors": ["Implicit conversion expected", "Type mismatch"]
                }
            ],
            ErrorType.MEMORY_ERROR: [
                {
                    "cause": "Memory leak",
                    "indicators": ["heap", "allocation", "gc"],
                    "factors": ["Unclosed resources", "Circular references"]
                },
                {
                    "cause": "Stack overflow",
                    "indicators": ["stack", "recursion", "depth"],
                    "factors": ["Infinite recursion", "Deep call stack"]
                }
            ],
            ErrorType.RACE_CONDITION: [
                {
                    "cause": "Concurrent access to shared state",
                    "indicators": ["concurrent", "thread", "async", "parallel"],
                    "factors": ["Missing synchronization", "Incorrect locking"]
                },
                {
                    "cause": "Operation order dependency",
                    "indicators": ["before", "after", "sequence", "order"],
                    "factors": ["Async timing issue", "Event race"]
                }
            ],
            ErrorType.CONNECTION_ERROR: [
                {
                    "cause": "Network connectivity issue",
                    "indicators": ["connection", "refused", "timeout", "unreachable"],
                    "factors": ["Service down", "Network configuration"]
                },
                {
                    "cause": "Authentication/Authorization failure",
                    "indicators": ["unauthorized", "forbidden", "credentials"],
                    "factors": ["Invalid credentials", "Token expired"]
                }
            ],
            ErrorType.CONFIGURATION: [
                {
                    "cause": "Missing or invalid configuration",
                    "indicators": ["config", "setting", "environment", "variable"],
                    "factors": ["Environment mismatch", "Missing config file"]
                },
                {
                    "cause": "Dependency version conflict",
                    "indicators": ["version", "dependency", "conflict", "incompatible"],
                    "factors": ["Package update", "Breaking change"]
                }
            ]
        }
    
    def _init_investigation_templates(self) -> Dict[ErrorType, List[str]]:
        """Initialize investigation suggestions for each error type"""
        return {
            ErrorType.NULL_REFERENCE: [
                "Check variable initialization path",
                "Verify async operations are properly awaited",
                "Add null/undefined checks before access",
                "Review API response handling"
            ],
            ErrorType.INDEX_ERROR: [
                "Verify array/list bounds before access",
                "Check for empty collections",
                "Review loop boundary conditions",
                "Examine dynamic collection modifications"
            ],
            ErrorType.TYPE_ERROR: [
                "Verify type annotations match usage",
                "Check for implicit type conversions",
                "Review API/library documentation for type changes",
                "Add explicit type casting where needed"
            ],
            ErrorType.MEMORY_ERROR: [
                "Profile memory usage patterns",
                "Check for unclosed resources/connections",
                "Review recursive function base cases",
                "Examine data structure growth patterns"
            ],
            ErrorType.RACE_CONDITION: [
                "Add proper synchronization mechanisms",
                "Review shared state access patterns",
                "Check async operation ordering",
                "Consider using immutable data structures"
            ],
            ErrorType.CONNECTION_ERROR: [
                "Verify network connectivity and firewall rules",
                "Check service availability and health",
                "Review authentication credentials and tokens",
                "Examine timeout and retry configurations"
            ],
            ErrorType.CONFIGURATION: [
                "Verify all required environment variables are set",
                "Check configuration file syntax and values",
                "Review dependency versions and compatibility",
                "Validate configuration against schema/documentation"
            ]
        }
    
    async def analyze(
        self,
        parsed_error: ParsedError,
        code_context: Dict[str, Any],
        execution_trace: Optional[TraceResult],
        pattern_database: Dict[str, Any],
        use_ai: bool = True
    ) -> RootCauseResult:
        """
        Perform root cause analysis
        """
        # Start with pattern-based analysis
        root_cause, confidence, patterns_matched = self._pattern_analysis(
            parsed_error, pattern_database
        )
        
        # Enhance with causality analysis
        causality_chain = self._build_causality_chain(
            parsed_error, code_context, execution_trace
        )
        
        # Identify contributing factors
        contributing_factors = self._identify_contributing_factors(
            parsed_error, code_context, execution_trace
        )
        
        # Find similar patterns in history
        similar_patterns = self._find_similar_patterns(
            parsed_error, pattern_database
        )
        
        # Generate investigation suggestions
        suggested_investigation = self._generate_investigation_suggestions(
            parsed_error, root_cause, contributing_factors
        )
        
        # AI-enhanced analysis if enabled
        if use_ai and confidence < 0.7:
            ai_result = await self._ai_enhanced_analysis(
                parsed_error, code_context, execution_trace
            )
            if ai_result:
                root_cause = ai_result.get("root_cause", root_cause)
                confidence = max(confidence, ai_result.get("confidence", 0))
                contributing_factors.extend(ai_result.get("factors", []))
        
        return RootCauseResult(
            root_cause=root_cause,
            confidence_score=confidence,
            contributing_factors=contributing_factors,
            similar_patterns=similar_patterns,
            suggested_investigation=suggested_investigation,
            patterns_matched=patterns_matched,
            causality_chain=causality_chain
        )
    
    def _pattern_analysis(
        self, parsed_error: ParsedError, pattern_database: Dict[str, Any]
    ) -> tuple[str, float, List[str]]:
        """Analyze error against known patterns"""
        best_match = None
        best_confidence = 0.0
        patterns_matched = []
        
        # Check against pattern database
        for pattern_name, pattern_info in pattern_database.items():
            for pattern in pattern_info.get("patterns", []):
                if re.search(pattern, parsed_error.raw_error, re.IGNORECASE):
                    patterns_matched.append(pattern_name)
                    
                    # Find the most likely cause
                    causes = pattern_info.get("common_causes", [])
                    if causes and not best_match:
                        best_match = causes[0]
                        best_confidence = 0.8
        
        # Check against causality patterns
        if parsed_error.error_type in self.causality_patterns:
            for pattern in self.causality_patterns[parsed_error.error_type]:
                indicators_found = sum(
                    1 for indicator in pattern["indicators"]
                    if indicator.lower() in parsed_error.raw_error.lower()
                )
                
                if indicators_found > 0:
                    match_confidence = indicators_found / len(pattern["indicators"])
                    if match_confidence > best_confidence:
                        best_match = pattern["cause"]
                        best_confidence = match_confidence
                        patterns_matched.append(pattern["cause"])
        
        if not best_match:
            best_match = f"Unidentified root cause for {parsed_error.error_type.value}"
            best_confidence = 0.3
        
        return best_match, best_confidence, patterns_matched
    
    def _build_causality_chain(
        self,
        parsed_error: ParsedError,
        code_context: Dict[str, Any],
        execution_trace: Optional[TraceResult]
    ) -> List[str]:
        """Build a chain of causality leading to the error"""
        chain = []
        
        # Add immediate cause
        chain.append(f"Error occurred: {parsed_error.primary_message}")
        
        # Add execution context if available
        if execution_trace and execution_trace.failure_point:
            chain.append(f"At: {execution_trace.failure_point}")
            
            # Add state at failure
            if execution_trace.state_changes:
                last_state = list(execution_trace.state_changes.values())[-1] if execution_trace.state_changes else None
                if last_state:
                    chain.append(f"State at failure: {last_state}")
        
        # Add stack context
        if parsed_error.stack_frames:
            top_frame = parsed_error.stack_frames[0]
            chain.append(f"In function: {top_frame.get('function', 'unknown')}")
        
        # Add code context
        if code_context.get("variables"):
            suspicious_vars = [
                var for var, value in code_context["variables"].items()
                if value in [None, "null", "undefined", ""]
            ]
            if suspicious_vars:
                chain.append(f"Suspicious variables: {', '.join(suspicious_vars)}")
        
        return chain
    
    def _identify_contributing_factors(
        self,
        parsed_error: ParsedError,
        code_context: Dict[str, Any],
        execution_trace: Optional[TraceResult]
    ) -> List[str]:
        """Identify factors that contributed to the error"""
        factors = []
        
        # Check for common contributing factors based on error type
        if parsed_error.error_type in self.causality_patterns:
            for pattern in self.causality_patterns[parsed_error.error_type]:
                # Check if indicators match
                indicators_found = any(
                    indicator.lower() in parsed_error.raw_error.lower()
                    for indicator in pattern["indicators"]
                )
                
                if indicators_found:
                    factors.extend(pattern["factors"])
        
        # Add execution-based factors
        if execution_trace:
            if execution_trace.loops_detected:
                factors.append("Loop execution involved")
            if execution_trace.recursion_depth > 10:
                factors.append(f"Deep recursion (depth: {execution_trace.recursion_depth})")
        
        # Add context-based factors
        if code_context:
            if code_context.get("dependencies"):
                factors.append(f"External dependencies: {len(code_context['dependencies'])}")
            if code_context.get("async_operations"):
                factors.append("Asynchronous operations detected")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_factors = []
        for factor in factors:
            if factor not in seen:
                seen.add(factor)
                unique_factors.append(factor)
        
        return unique_factors
    
    def _find_similar_patterns(
        self, parsed_error: ParsedError, pattern_database: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find similar error patterns from database"""
        similar = []
        
        for pattern_name, pattern_info in pattern_database.items():
            similarity_score = 0.0
            
            # Check pattern matches
            for pattern in pattern_info.get("patterns", []):
                if re.search(pattern, parsed_error.raw_error, re.IGNORECASE):
                    similarity_score += 0.5
                    break
            
            # Check error type similarity
            if parsed_error.error_type.value in pattern_name.lower():
                similarity_score += 0.3
            
            # Check common words
            common_words = set(parsed_error.primary_message.lower().split()) & \
                          set(pattern_name.lower().split("_"))
            if common_words:
                similarity_score += 0.2 * len(common_words)
            
            if similarity_score > 0.3:
                similar.append({
                    "pattern": pattern_name,
                    "similarity": min(1.0, similarity_score),
                    "fixes": pattern_info.get("fix_patterns", [])
                })
        
        # Sort by similarity
        similar.sort(key=lambda x: x["similarity"], reverse=True)
        
        return similar[:5]  # Return top 5 similar patterns
    
    def _generate_investigation_suggestions(
        self,
        parsed_error: ParsedError,
        root_cause: str,
        contributing_factors: List[str]
    ) -> List[str]:
        """Generate specific investigation suggestions"""
        suggestions = []
        
        # Add template-based suggestions
        if parsed_error.error_type in self.investigation_templates:
            suggestions.extend(self.investigation_templates[parsed_error.error_type])
        
        # Add root-cause-specific suggestions
        if "initialization" in root_cause.lower():
            suggestions.append("Trace variable initialization from declaration to usage")
        if "async" in root_cause.lower():
            suggestions.append("Review async/await usage and promise handling")
        if "type" in root_cause.lower():
            suggestions.append("Enable strict type checking in development")
        if "memory" in root_cause.lower():
            suggestions.append("Run memory profiler to identify leaks")
        if "race" in root_cause.lower():
            suggestions.append("Add logging to track operation sequence")
        
        # Add factor-specific suggestions
        for factor in contributing_factors:
            if "loop" in factor.lower():
                suggestions.append("Add loop invariant checks and bounds validation")
            if "recursion" in factor.lower():
                suggestions.append("Add recursion depth limit and base case validation")
            if "dependency" in factor.lower():
                suggestions.append("Audit dependency versions and compatibility")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion not in seen:
                seen.add(suggestion)
                unique_suggestions.append(suggestion)
        
        return unique_suggestions[:7]  # Limit to 7 suggestions
    
    async def _ai_enhanced_analysis(
        self,
        parsed_error: ParsedError,
        code_context: Dict[str, Any],
        execution_trace: Optional[TraceResult]
    ) -> Optional[Dict[str, Any]]:
        """Use AI to enhance root cause analysis"""
        # This would integrate with the Ollama client for advanced analysis
        # For now, return a placeholder
        return {
            "root_cause": parsed_error.primary_message,
            "confidence": 0.6,
            "factors": ["AI-detected pattern"]
        }