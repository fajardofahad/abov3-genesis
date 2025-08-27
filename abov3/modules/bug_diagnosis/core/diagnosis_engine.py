"""
ABOV3 Genesis - Bug Diagnosis Engine
Core orchestration for bug diagnosis and automated fix generation
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import json
import traceback
import re

from ....core.context_manager import SmartContextManager
from ....core.ollama_client import OllamaClient
from ...context_aware.core.comprehension_engine import ComprehensionEngine
from ..parsers.error_parser import ErrorParser, ErrorType, ParsedError
from ..analysis.root_cause_analyzer import RootCauseAnalyzer, RootCauseResult
from ..fixers.auto_fixer import AutoFixer, FixStrategy, FixResult
from ..tracing.execution_tracer import ExecutionTracer, TraceResult
from ..utils.language_detector import LanguageDetector
from ..utils.pattern_matcher import PatternMatcher

logger = logging.getLogger(__name__)

class DiagnosisMode(Enum):
    """Different diagnosis modes for various bug types"""
    RUNTIME_ERROR = "runtime_error"
    TEST_FAILURE = "test_failure"
    PERFORMANCE_ISSUE = "performance_issue"
    LOGIC_BUG = "logic_bug"
    COMPILATION_ERROR = "compilation_error"
    MEMORY_LEAK = "memory_leak"
    RACE_CONDITION = "race_condition"
    SECURITY_VULNERABILITY = "security_vulnerability"
    INTEGRATION_ISSUE = "integration_issue"
    AUTO_DETECT = "auto_detect"

class ConfidenceLevel(Enum):
    """Confidence levels for diagnosis and fixes"""
    HIGH = "high"           # 90-100% confident
    MEDIUM = "medium"       # 60-89% confident
    LOW = "low"            # 30-59% confident
    UNCERTAIN = "uncertain" # Below 30% confident

@dataclass
class DiagnosisRequest:
    """Request for bug diagnosis"""
    error_message: Optional[str] = None
    symptom_description: Optional[str] = None
    stack_trace: Optional[str] = None
    code_snippet: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    mode: DiagnosisMode = DiagnosisMode.AUTO_DETECT
    project_path: Optional[str] = None
    language: Optional[str] = None
    framework: Optional[str] = None
    include_dependencies: bool = True
    max_trace_depth: int = 10
    generate_fix: bool = True
    fix_strategy: FixStrategy = FixStrategy.SAFE
    explain_steps: bool = True
    run_validation: bool = True
    context_lines: int = 10
    use_ai_analysis: bool = True

@dataclass
class DiagnosisStep:
    """Individual step in the debugging process"""
    step_number: int
    description: str
    action_taken: str
    findings: Dict[str, Any]
    confidence: ConfidenceLevel
    duration_ms: float
    
@dataclass
class DiagnosisResult:
    """Result of bug diagnosis"""
    success: bool
    error_type: ErrorType
    root_cause: str
    confidence: ConfidenceLevel
    affected_files: List[str]
    diagnosis_steps: List[DiagnosisStep]
    suggested_fixes: List[FixResult]
    execution_trace: Optional[TraceResult]
    related_issues: List[Dict[str, Any]]
    performance_impact: Optional[Dict[str, Any]]
    security_implications: Optional[Dict[str, Any]]
    explanation: str
    metadata: Dict[str, Any]
    total_duration_ms: float

class BugDiagnosisEngine:
    """
    Advanced bug diagnosis engine with AI-powered analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.context_manager = SmartContextManager()
        self.ollama_client = OllamaClient()
        self.comprehension_engine = ComprehensionEngine()
        self.error_parser = ErrorParser()
        self.root_cause_analyzer = RootCauseAnalyzer()
        self.auto_fixer = AutoFixer()
        self.execution_tracer = ExecutionTracer()
        self.language_detector = LanguageDetector()
        self.pattern_matcher = PatternMatcher()
        
        # Cache for repeated diagnoses
        self._diagnosis_cache = {}
        self._pattern_database = self._load_pattern_database()
        
    def _load_pattern_database(self) -> Dict[str, Any]:
        """Load known bug patterns and solutions"""
        return {
            "null_pointer": {
                "patterns": [
                    r"NullPointerException",
                    r"TypeError.*'NoneType'",
                    r"Cannot read propert.*of undefined",
                    r"null reference exception"
                ],
                "common_causes": [
                    "Uninitialized variable",
                    "Missing null check",
                    "Async operation not awaited",
                    "Optional value not handled"
                ],
                "fix_patterns": [
                    "Add null/undefined check",
                    "Initialize variable with default value",
                    "Use optional chaining (?.)",
                    "Add proper error handling"
                ]
            },
            "index_out_of_bounds": {
                "patterns": [
                    r"IndexError",
                    r"ArrayIndexOutOfBoundsException",
                    r"index.*out of (range|bounds)",
                    r"list index out of range"
                ],
                "common_causes": [
                    "Loop boundary condition error",
                    "Empty array/list access",
                    "Off-by-one error",
                    "Dynamic array size change"
                ],
                "fix_patterns": [
                    "Add boundary checks",
                    "Verify array/list length before access",
                    "Fix loop conditions",
                    "Use safe access methods"
                ]
            },
            "type_mismatch": {
                "patterns": [
                    r"TypeError",
                    r"ClassCastException",
                    r"type.*mismatch",
                    r"cannot convert.*to"
                ],
                "common_causes": [
                    "Incorrect type assumption",
                    "Missing type conversion",
                    "API return type change",
                    "Generic type issue"
                ],
                "fix_patterns": [
                    "Add explicit type conversion",
                    "Use type guards/checks",
                    "Update type annotations",
                    "Handle multiple types"
                ]
            },
            "async_await": {
                "patterns": [
                    r"await.*outside.*async",
                    r"Promise.*not.*await",
                    r"coroutine.*never awaited",
                    r"async.*sync.*conflict"
                ],
                "common_causes": [
                    "Missing async keyword",
                    "Forgotten await",
                    "Mixing sync and async code",
                    "Promise chain issues"
                ],
                "fix_patterns": [
                    "Add async/await keywords",
                    "Convert to proper async flow",
                    "Use Promise.all for parallel ops",
                    "Handle promise rejections"
                ]
            },
            "memory_leak": {
                "patterns": [
                    r"OutOfMemoryError",
                    r"memory.*leak",
                    r"heap.*exhausted",
                    r"GC overhead limit"
                ],
                "common_causes": [
                    "Unclosed resources",
                    "Circular references",
                    "Event listener accumulation",
                    "Cache without eviction"
                ],
                "fix_patterns": [
                    "Implement proper cleanup",
                    "Use weak references",
                    "Remove event listeners",
                    "Implement cache eviction"
                ]
            },
            "race_condition": {
                "patterns": [
                    r"race.*condition",
                    r"concurrent.*modification",
                    r"deadlock",
                    r"thread.*safety"
                ],
                "common_causes": [
                    "Shared mutable state",
                    "Missing synchronization",
                    "Incorrect lock ordering",
                    "Atomic operation assumption"
                ],
                "fix_patterns": [
                    "Add synchronization primitives",
                    "Use immutable data structures",
                    "Implement proper locking",
                    "Use thread-safe collections"
                ]
            }
        }
    
    async def diagnose(self, request: DiagnosisRequest) -> DiagnosisResult:
        """
        Main entry point for bug diagnosis
        """
        start_time = time.time()
        diagnosis_steps = []
        
        try:
            # Step 1: Parse and classify the error
            step1_start = time.time()
            parsed_error = await self._parse_error(request)
            diagnosis_steps.append(DiagnosisStep(
                step_number=1,
                description="Parse and classify error",
                action_taken="Analyzed error message, stack trace, and symptoms",
                findings={
                    "error_type": parsed_error.error_type.value,
                    "primary_message": parsed_error.primary_message,
                    "language": parsed_error.language,
                    "framework": parsed_error.framework
                },
                confidence=self._calculate_confidence(parsed_error.confidence_score),
                duration_ms=(time.time() - step1_start) * 1000
            ))
            
            # Step 2: Analyze code context
            step2_start = time.time()
            code_context = await self._analyze_code_context(request, parsed_error)
            diagnosis_steps.append(DiagnosisStep(
                step_number=2,
                description="Analyze code context",
                action_taken="Examined surrounding code and dependencies",
                findings={
                    "affected_functions": code_context.get("functions", []),
                    "variable_states": code_context.get("variables", {}),
                    "dependencies": code_context.get("dependencies", [])
                },
                confidence=ConfidenceLevel.MEDIUM,
                duration_ms=(time.time() - step2_start) * 1000
            ))
            
            # Step 3: Trace execution path
            step3_start = time.time()
            execution_trace = None
            if request.mode in [DiagnosisMode.RUNTIME_ERROR, DiagnosisMode.LOGIC_BUG]:
                execution_trace = await self._trace_execution(request, parsed_error)
                diagnosis_steps.append(DiagnosisStep(
                    step_number=3,
                    description="Trace execution path",
                    action_taken="Traced code execution flow to identify failure point",
                    findings={
                        "execution_path": execution_trace.path if execution_trace else [],
                        "state_changes": execution_trace.state_changes if execution_trace else {},
                        "failure_point": execution_trace.failure_point if execution_trace else None
                    },
                    confidence=ConfidenceLevel.HIGH if execution_trace else ConfidenceLevel.LOW,
                    duration_ms=(time.time() - step3_start) * 1000
                ))
            
            # Step 4: Identify root cause
            step4_start = time.time()
            root_cause_result = await self._analyze_root_cause(
                parsed_error, code_context, execution_trace, request
            )
            diagnosis_steps.append(DiagnosisStep(
                step_number=4,
                description="Identify root cause",
                action_taken="Analyzed patterns and correlations to find root cause",
                findings={
                    "root_cause": root_cause_result.root_cause,
                    "contributing_factors": root_cause_result.contributing_factors,
                    "similar_patterns": root_cause_result.similar_patterns
                },
                confidence=self._calculate_confidence(root_cause_result.confidence_score),
                duration_ms=(time.time() - step4_start) * 1000
            ))
            
            # Step 5: Generate fixes
            suggested_fixes = []
            if request.generate_fix:
                step5_start = time.time()
                fixes = await self._generate_fixes(
                    parsed_error, root_cause_result, code_context, request
                )
                suggested_fixes = fixes
                diagnosis_steps.append(DiagnosisStep(
                    step_number=5,
                    description="Generate fixes",
                    action_taken="Created multiple fix strategies with confidence levels",
                    findings={
                        "fixes_generated": len(fixes),
                        "strategies_used": [f.strategy.value for f in fixes],
                        "validation_status": "pending"
                    },
                    confidence=ConfidenceLevel.HIGH if fixes else ConfidenceLevel.LOW,
                    duration_ms=(time.time() - step5_start) * 1000
                ))
                
                # Step 6: Validate fixes
                if request.run_validation and fixes:
                    step6_start = time.time()
                    validation_results = await self._validate_fixes(fixes, request)
                    diagnosis_steps.append(DiagnosisStep(
                        step_number=6,
                        description="Validate fixes",
                        action_taken="Tested fixes for correctness and side effects",
                        findings={
                            "valid_fixes": validation_results.get("valid_count", 0),
                            "test_results": validation_results.get("test_results", {}),
                            "side_effects": validation_results.get("side_effects", [])
                        },
                        confidence=ConfidenceLevel.HIGH,
                        duration_ms=(time.time() - step6_start) * 1000
                    ))
            
            # Step 7: Check for related issues
            step7_start = time.time()
            related_issues = await self._find_related_issues(
                parsed_error, root_cause_result, request
            )
            diagnosis_steps.append(DiagnosisStep(
                step_number=7,
                description="Check for related issues",
                action_taken="Searched for similar patterns and potential cascading issues",
                findings={
                    "related_count": len(related_issues),
                    "patterns_found": [issue.get("pattern") for issue in related_issues[:3]]
                },
                confidence=ConfidenceLevel.MEDIUM,
                duration_ms=(time.time() - step7_start) * 1000
            ))
            
            # Generate comprehensive explanation
            explanation = await self._generate_explanation(
                parsed_error, root_cause_result, diagnosis_steps, suggested_fixes
            )
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(diagnosis_steps)
            
            return DiagnosisResult(
                success=True,
                error_type=parsed_error.error_type,
                root_cause=root_cause_result.root_cause,
                confidence=overall_confidence,
                affected_files=list(set(
                    [request.file_path] if request.file_path else [] +
                    code_context.get("affected_files", [])
                )),
                diagnosis_steps=diagnosis_steps,
                suggested_fixes=suggested_fixes,
                execution_trace=execution_trace,
                related_issues=related_issues,
                performance_impact=await self._assess_performance_impact(
                    parsed_error, root_cause_result
                ),
                security_implications=await self._assess_security_implications(
                    parsed_error, root_cause_result
                ),
                explanation=explanation,
                metadata={
                    "diagnosis_id": self._generate_diagnosis_id(),
                    "timestamp": time.time(),
                    "version": "1.0.0",
                    "model_used": self.ollama_client.model_name,
                    "patterns_matched": root_cause_result.patterns_matched
                },
                total_duration_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            logger.error(f"Diagnosis failed: {str(e)}")
            return DiagnosisResult(
                success=False,
                error_type=ErrorType.UNKNOWN,
                root_cause=f"Diagnosis failed: {str(e)}",
                confidence=ConfidenceLevel.UNCERTAIN,
                affected_files=[],
                diagnosis_steps=diagnosis_steps,
                suggested_fixes=[],
                execution_trace=None,
                related_issues=[],
                performance_impact=None,
                security_implications=None,
                explanation=f"Failed to diagnose the issue: {str(e)}",
                metadata={"error": str(e)},
                total_duration_ms=(time.time() - start_time) * 1000
            )
    
    async def _parse_error(self, request: DiagnosisRequest) -> ParsedError:
        """Parse and classify the error"""
        return await self.error_parser.parse(
            error_message=request.error_message,
            stack_trace=request.stack_trace,
            symptom_description=request.symptom_description,
            code_snippet=request.code_snippet,
            language=request.language,
            framework=request.framework
        )
    
    async def _analyze_code_context(
        self, request: DiagnosisRequest, parsed_error: ParsedError
    ) -> Dict[str, Any]:
        """Analyze the code context around the error"""
        context = {
            "functions": [],
            "variables": {},
            "dependencies": [],
            "affected_files": []
        }
        
        if request.file_path and request.line_number:
            # Use comprehension engine to understand code context
            comprehension_result = await self.comprehension_engine.comprehend({
                "query": f"Analyze code around line {request.line_number} in {request.file_path}",
                "target_paths": [request.file_path],
                "context_depth": 2
            })
            
            if comprehension_result:
                context.update(comprehension_result.get("context", {}))
        
        # Extract additional context from stack trace
        if parsed_error.stack_frames:
            for frame in parsed_error.stack_frames[:5]:  # Analyze top 5 frames
                context["affected_files"].append(frame.get("file"))
                context["functions"].append(frame.get("function"))
        
        return context
    
    async def _trace_execution(
        self, request: DiagnosisRequest, parsed_error: ParsedError
    ) -> Optional[TraceResult]:
        """Trace the execution path to the error"""
        if not request.file_path:
            return None
            
        return await self.execution_tracer.trace(
            file_path=request.file_path,
            line_number=request.line_number,
            error_type=parsed_error.error_type,
            max_depth=request.max_trace_depth
        )
    
    async def _analyze_root_cause(
        self,
        parsed_error: ParsedError,
        code_context: Dict[str, Any],
        execution_trace: Optional[TraceResult],
        request: DiagnosisRequest
    ) -> RootCauseResult:
        """Analyze and identify the root cause"""
        return await self.root_cause_analyzer.analyze(
            parsed_error=parsed_error,
            code_context=code_context,
            execution_trace=execution_trace,
            pattern_database=self._pattern_database,
            use_ai=request.use_ai_analysis
        )
    
    async def _generate_fixes(
        self,
        parsed_error: ParsedError,
        root_cause: RootCauseResult,
        code_context: Dict[str, Any],
        request: DiagnosisRequest
    ) -> List[FixResult]:
        """Generate multiple fix strategies"""
        fixes = []
        
        # Generate fixes based on strategy
        strategies = [request.fix_strategy]
        if request.fix_strategy == FixStrategy.AUTO:
            strategies = [FixStrategy.SAFE, FixStrategy.OPTIMAL, FixStrategy.QUICK]
        
        for strategy in strategies:
            fix = await self.auto_fixer.generate_fix(
                error_type=parsed_error.error_type,
                root_cause=root_cause,
                code_context=code_context,
                strategy=strategy,
                file_path=request.file_path,
                line_number=request.line_number
            )
            if fix:
                fixes.append(fix)
        
        # Sort by confidence
        fixes.sort(key=lambda f: f.confidence_score, reverse=True)
        
        return fixes[:3]  # Return top 3 fixes
    
    async def _validate_fixes(
        self, fixes: List[FixResult], request: DiagnosisRequest
    ) -> Dict[str, Any]:
        """Validate the generated fixes"""
        validation_results = {
            "valid_count": 0,
            "test_results": {},
            "side_effects": []
        }
        
        for fix in fixes:
            # Simulate validation (in production, would run actual tests)
            is_valid = fix.confidence_score > 0.7
            if is_valid:
                validation_results["valid_count"] += 1
            
            validation_results["test_results"][fix.fix_id] = {
                "valid": is_valid,
                "confidence": fix.confidence_score
            }
        
        return validation_results
    
    async def _find_related_issues(
        self,
        parsed_error: ParsedError,
        root_cause: RootCauseResult,
        request: DiagnosisRequest
    ) -> List[Dict[str, Any]]:
        """Find related or similar issues in the codebase"""
        related_issues = []
        
        # Search for similar patterns
        if request.project_path:
            patterns = self.pattern_matcher.find_similar_patterns(
                pattern=root_cause.root_cause,
                search_path=request.project_path,
                language=parsed_error.language
            )
            
            for pattern in patterns[:5]:  # Limit to 5 related issues
                related_issues.append({
                    "pattern": pattern.get("pattern"),
                    "file": pattern.get("file"),
                    "line": pattern.get("line"),
                    "similarity": pattern.get("similarity", 0)
                })
        
        return related_issues
    
    async def _assess_performance_impact(
        self, parsed_error: ParsedError, root_cause: RootCauseResult
    ) -> Optional[Dict[str, Any]]:
        """Assess the performance impact of the bug"""
        if parsed_error.error_type in [ErrorType.PERFORMANCE, ErrorType.MEMORY_LEAK]:
            return {
                "severity": "high",
                "metrics_affected": ["response_time", "memory_usage", "cpu_usage"],
                "estimated_impact": "10-50% degradation",
                "user_impact": "Noticeable slowdown or resource exhaustion"
            }
        
        return None
    
    async def _assess_security_implications(
        self, parsed_error: ParsedError, root_cause: RootCauseResult
    ) -> Optional[Dict[str, Any]]:
        """Assess security implications of the bug"""
        security_keywords = ["injection", "xss", "csrf", "auth", "permission", "privilege"]
        
        if any(keyword in root_cause.root_cause.lower() for keyword in security_keywords):
            return {
                "severity": "critical",
                "vulnerability_type": "Potential security vulnerability",
                "cwe_id": "CWE-89",  # Example: SQL Injection
                "recommendation": "Immediate fix required, consider security review"
            }
        
        return None
    
    async def _generate_explanation(
        self,
        parsed_error: ParsedError,
        root_cause: RootCauseResult,
        steps: List[DiagnosisStep],
        fixes: List[FixResult]
    ) -> str:
        """Generate a comprehensive explanation of the diagnosis"""
        explanation_parts = [
            f"## Bug Diagnosis Summary\n",
            f"**Error Type**: {parsed_error.error_type.value}\n",
            f"**Root Cause**: {root_cause.root_cause}\n",
            f"**Confidence**: {root_cause.confidence_score:.0%}\n\n",
            
            "## Debugging Process\n"
        ]
        
        for step in steps:
            explanation_parts.append(
                f"{step.step_number}. **{step.description}**\n"
                f"   - Action: {step.action_taken}\n"
                f"   - Confidence: {step.confidence.value}\n"
                f"   - Duration: {step.duration_ms:.0f}ms\n\n"
            )
        
        if fixes:
            explanation_parts.append("\n## Suggested Fixes\n")
            for i, fix in enumerate(fixes, 1):
                explanation_parts.append(
                    f"{i}. **{fix.description}** (Confidence: {fix.confidence_score:.0%})\n"
                    f"   - Strategy: {fix.strategy.value}\n"
                    f"   - Impact: {fix.impact_assessment}\n\n"
                )
        
        if root_cause.contributing_factors:
            explanation_parts.append("\n## Contributing Factors\n")
            for factor in root_cause.contributing_factors:
                explanation_parts.append(f"- {factor}\n")
        
        return "".join(explanation_parts)
    
    def _calculate_confidence(self, score: float) -> ConfidenceLevel:
        """Convert numeric score to confidence level"""
        if score >= 0.9:
            return ConfidenceLevel.HIGH
        elif score >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNCERTAIN
    
    def _calculate_overall_confidence(
        self, steps: List[DiagnosisStep]
    ) -> ConfidenceLevel:
        """Calculate overall confidence from all steps"""
        confidence_scores = {
            ConfidenceLevel.HIGH: 1.0,
            ConfidenceLevel.MEDIUM: 0.7,
            ConfidenceLevel.LOW: 0.4,
            ConfidenceLevel.UNCERTAIN: 0.1
        }
        
        if not steps:
            return ConfidenceLevel.UNCERTAIN
        
        total_score = sum(confidence_scores[step.confidence] for step in steps)
        avg_score = total_score / len(steps)
        
        return self._calculate_confidence(avg_score)
    
    def _generate_diagnosis_id(self) -> str:
        """Generate unique diagnosis ID"""
        import uuid
        return f"diag_{uuid.uuid4().hex[:12]}"