"""
Automated fix generation engine with multiple strategies
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import re
import json

from ..parsers.error_parser import ErrorType
from ..analysis.root_cause_analyzer import RootCauseResult

logger = logging.getLogger(__name__)

class FixStrategy(Enum):
    """Different strategies for generating fixes"""
    SAFE = "safe"              # Conservative, minimal changes
    OPTIMAL = "optimal"        # Best practice solution
    QUICK = "quick"           # Fast but possibly temporary
    COMPREHENSIVE = "comprehensive"  # Complete refactoring
    AUTO = "auto"             # Automatically choose strategy

@dataclass
class FixResult:
    """Result of fix generation"""
    fix_id: str
    strategy: FixStrategy
    description: str
    code_changes: List[Dict[str, Any]]  # List of file changes
    confidence_score: float
    impact_assessment: str
    testing_required: List[str]
    rollback_plan: Optional[str]
    implementation_steps: List[str]
    warnings: List[str]
    estimated_time: str  # e.g., "5 minutes", "1 hour"

class AutoFixer:
    """
    Intelligent automated fix generator
    """
    
    def __init__(self):
        self.fix_templates = self._init_fix_templates()
        self.language_specific_fixes = self._init_language_fixes()
        
    def _init_fix_templates(self) -> Dict[ErrorType, List[Dict[str, Any]]]:
        """Initialize fix templates for different error types"""
        return {
            ErrorType.NULL_REFERENCE: [
                {
                    "strategy": FixStrategy.SAFE,
                    "template": "if ({var} != null) {{ {original_code} }}",
                    "description": "Add null check before access"
                },
                {
                    "strategy": FixStrategy.OPTIMAL,
                    "template": "{var} = {var} ?? {default_value}",
                    "description": "Use null coalescing with default value"
                },
                {
                    "strategy": FixStrategy.COMPREHENSIVE,
                    "template": "Optional<{type}> {var} = Optional.ofNullable({value})",
                    "description": "Use Optional pattern for null safety"
                }
            ],
            ErrorType.INDEX_ERROR: [
                {
                    "strategy": FixStrategy.SAFE,
                    "template": "if ({index} >= 0 && {index} < {array}.length) {{ {original_code} }}",
                    "description": "Add boundary check"
                },
                {
                    "strategy": FixStrategy.OPTIMAL,
                    "template": "{array}.get(Math.min({index}, {array}.length - 1))",
                    "description": "Use safe access with bounds clamping"
                }
            ],
            ErrorType.TYPE_ERROR: [
                {
                    "strategy": FixStrategy.SAFE,
                    "template": "{type}({value})",
                    "description": "Add explicit type conversion"
                },
                {
                    "strategy": FixStrategy.OPTIMAL,
                    "template": "isinstance({var}, {type}) ? {var} : {convert}({var})",
                    "description": "Type check with conversion"
                }
            ],
            ErrorType.MEMORY_ERROR: [
                {
                    "strategy": FixStrategy.SAFE,
                    "template": "try {{ {code} }} finally {{ {cleanup} }}",
                    "description": "Add resource cleanup in finally block"
                },
                {
                    "strategy": FixStrategy.OPTIMAL,
                    "template": "with {resource} as {var}: {code}",
                    "description": "Use context manager for automatic cleanup"
                }
            ],
            ErrorType.RACE_CONDITION: [
                {
                    "strategy": FixStrategy.SAFE,
                    "template": "synchronized({lock}) {{ {code} }}",
                    "description": "Add synchronization block"
                },
                {
                    "strategy": FixStrategy.OPTIMAL,
                    "template": "const {var} = Object.freeze({value})",
                    "description": "Use immutable data structure"
                }
            ]
        }
    
    def _init_language_fixes(self) -> Dict[str, Dict[str, Any]]:
        """Initialize language-specific fix patterns"""
        return {
            "python": {
                "null_check": "if {var} is not None:",
                "try_except": "try:\n    {code}\nexcept {exception} as e:\n    {handler}",
                "type_check": "isinstance({var}, {type})",
                "default_value": "{var} = {var} or {default}",
                "context_manager": "with {resource} as {var}:\n    {code}"
            },
            "javascript": {
                "null_check": "if ({var} != null) {{",
                "try_catch": "try {{\n    {code}\n}} catch ({exception}) {{\n    {handler}\n}}",
                "type_check": "typeof {var} === '{type}'",
                "default_value": "{var} = {var} || {default}",
                "optional_chain": "{var}?.{property}"
            },
            "typescript": {
                "null_check": "if ({var} !== null && {var} !== undefined) {{",
                "try_catch": "try {{\n    {code}\n}} catch (error: unknown) {{\n    {handler}\n}}",
                "type_guard": "function is{Type}(val: unknown): val is {Type} {{",
                "default_value": "{var} = {var} ?? {default}",
                "optional_chain": "{var}?.{property}"
            },
            "java": {
                "null_check": "if ({var} != null) {{",
                "try_catch": "try {{\n    {code}\n}} catch ({exception} e) {{\n    {handler}\n}}",
                "type_check": "{var} instanceof {type}",
                "optional": "Optional.ofNullable({var})",
                "synchronized": "synchronized({lock}) {{\n    {code}\n}}"
            },
            "go": {
                "null_check": "if {var} != nil {{",
                "error_check": "if err != nil {{\n    return err\n}}",
                "type_assertion": "{var}, ok := {value}.({type})",
                "defer": "defer {cleanup}()",
                "mutex": "mu.Lock()\ndefer mu.Unlock()"
            }
        }
    
    async def generate_fix(
        self,
        error_type: ErrorType,
        root_cause: RootCauseResult,
        code_context: Dict[str, Any],
        strategy: FixStrategy,
        file_path: Optional[str],
        line_number: Optional[int]
    ) -> Optional[FixResult]:
        """
        Generate a fix for the identified issue
        """
        import uuid
        fix_id = f"fix_{uuid.uuid4().hex[:8]}"
        
        # Get appropriate fix templates
        templates = self.fix_templates.get(error_type, [])
        if not templates:
            return self._generate_generic_fix(
                fix_id, error_type, root_cause, strategy, file_path, line_number
            )
        
        # Select template based on strategy
        selected_template = None
        for template in templates:
            if template["strategy"] == strategy or strategy == FixStrategy.AUTO:
                selected_template = template
                break
        
        if not selected_template:
            selected_template = templates[0]  # Fallback to first template
        
        # Generate code changes
        code_changes = self._generate_code_changes(
            selected_template, code_context, file_path, line_number
        )
        
        # Assess impact
        impact = self._assess_impact(error_type, strategy, code_changes)
        
        # Generate implementation steps
        steps = self._generate_implementation_steps(
            selected_template, error_type, file_path
        )
        
        # Identify required testing
        testing = self._identify_testing_requirements(
            error_type, strategy, code_changes
        )
        
        # Generate warnings
        warnings = self._generate_warnings(error_type, strategy, root_cause)
        
        # Estimate time
        estimated_time = self._estimate_time(strategy, code_changes)
        
        return FixResult(
            fix_id=fix_id,
            strategy=strategy if strategy != FixStrategy.AUTO else selected_template["strategy"],
            description=selected_template["description"],
            code_changes=code_changes,
            confidence_score=self._calculate_confidence(
                error_type, root_cause, strategy
            ),
            impact_assessment=impact,
            testing_required=testing,
            rollback_plan=self._generate_rollback_plan(code_changes),
            implementation_steps=steps,
            warnings=warnings,
            estimated_time=estimated_time
        )
    
    def _generate_generic_fix(
        self,
        fix_id: str,
        error_type: ErrorType,
        root_cause: RootCauseResult,
        strategy: FixStrategy,
        file_path: Optional[str],
        line_number: Optional[int]
    ) -> FixResult:
        """Generate a generic fix when no specific template exists"""
        return FixResult(
            fix_id=fix_id,
            strategy=strategy if strategy != FixStrategy.AUTO else FixStrategy.SAFE,
            description=f"Generic fix for {error_type.value}",
            code_changes=[{
                "file": file_path or "unknown",
                "line": line_number or 0,
                "type": "suggestion",
                "original": "",
                "fixed": f"// TODO: Fix {error_type.value} - {root_cause.root_cause}",
                "explanation": "Manual fix required"
            }],
            confidence_score=0.3,
            impact_assessment="Low confidence generic fix",
            testing_required=["Manual testing required"],
            rollback_plan="Revert changes if issue persists",
            implementation_steps=["Review root cause", "Implement appropriate fix", "Test thoroughly"],
            warnings=["This is a generic fix suggestion - manual review required"],
            estimated_time="30 minutes"
        )
    
    def _generate_code_changes(
        self,
        template: Dict[str, Any],
        code_context: Dict[str, Any],
        file_path: Optional[str],
        line_number: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Generate specific code changes"""
        changes = []
        
        if file_path and line_number:
            # Extract original code (simplified - would need actual file reading)
            original_code = code_context.get("code_snippet", "// Original code here")
            
            # Apply template (simplified - would need proper parsing)
            fixed_code = template["template"].format(
                var="variable",
                original_code=original_code,
                default_value="defaultValue",
                array="array",
                index="index",
                type="Type",
                value="value",
                code="code",
                cleanup="cleanup()",
                lock="lock"
            )
            
            changes.append({
                "file": file_path,
                "line": line_number,
                "type": "replace",
                "original": original_code,
                "fixed": fixed_code,
                "explanation": template["description"]
            })
        
        return changes
    
    def _assess_impact(
        self,
        error_type: ErrorType,
        strategy: FixStrategy,
        code_changes: List[Dict[str, Any]]
    ) -> str:
        """Assess the impact of the fix"""
        if strategy == FixStrategy.SAFE:
            return "Minimal impact - adds defensive checks without changing logic"
        elif strategy == FixStrategy.OPTIMAL:
            return "Moderate impact - improves code quality and reliability"
        elif strategy == FixStrategy.QUICK:
            return "Quick fix - may need refinement later"
        elif strategy == FixStrategy.COMPREHENSIVE:
            return "Significant impact - refactors code for better architecture"
        else:
            return "Impact assessment pending"
    
    def _generate_implementation_steps(
        self,
        template: Dict[str, Any],
        error_type: ErrorType,
        file_path: Optional[str]
    ) -> List[str]:
        """Generate step-by-step implementation guide"""
        steps = []
        
        if file_path:
            steps.append(f"Open file: {file_path}")
        
        steps.extend([
            f"Locate the error site ({error_type.value})",
            f"Apply fix: {template['description']}",
            "Verify syntax correctness",
            "Run unit tests",
            "Test the specific error scenario",
            "Monitor for side effects"
        ])
        
        return steps
    
    def _identify_testing_requirements(
        self,
        error_type: ErrorType,
        strategy: FixStrategy,
        code_changes: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify what testing is required"""
        tests = ["Unit test for the fixed function"]
        
        if error_type == ErrorType.NULL_REFERENCE:
            tests.extend([
                "Test with null/undefined values",
                "Test with valid values",
                "Test edge cases"
            ])
        elif error_type == ErrorType.INDEX_ERROR:
            tests.extend([
                "Test with empty arrays/lists",
                "Test boundary conditions",
                "Test with negative indices"
            ])
        elif error_type == ErrorType.RACE_CONDITION:
            tests.extend([
                "Concurrent access testing",
                "Load testing",
                "Thread safety verification"
            ])
        elif error_type == ErrorType.MEMORY_ERROR:
            tests.extend([
                "Memory leak testing",
                "Resource cleanup verification",
                "Performance profiling"
            ])
        
        if strategy == FixStrategy.COMPREHENSIVE:
            tests.append("Full regression testing")
        
        return tests
    
    def _generate_warnings(
        self,
        error_type: ErrorType,
        strategy: FixStrategy,
        root_cause: RootCauseResult
    ) -> List[str]:
        """Generate warnings about the fix"""
        warnings = []
        
        if root_cause.confidence_score < 0.6:
            warnings.append("Low confidence in root cause - fix may not address the issue")
        
        if strategy == FixStrategy.QUICK:
            warnings.append("This is a quick fix - consider implementing a more robust solution")
        
        if error_type == ErrorType.RACE_CONDITION:
            warnings.append("Race conditions are complex - thorough testing required")
        
        if error_type == ErrorType.MEMORY_ERROR:
            warnings.append("Memory issues may have multiple causes - monitor after fix")
        
        return warnings
    
    def _calculate_confidence(
        self,
        error_type: ErrorType,
        root_cause: RootCauseResult,
        strategy: FixStrategy
    ) -> float:
        """Calculate confidence in the fix"""
        base_confidence = root_cause.confidence_score
        
        # Adjust based on error type complexity
        complexity_factor = {
            ErrorType.SYNTAX_ERROR: 0.95,
            ErrorType.NULL_REFERENCE: 0.85,
            ErrorType.INDEX_ERROR: 0.85,
            ErrorType.TYPE_ERROR: 0.80,
            ErrorType.RACE_CONDITION: 0.60,
            ErrorType.MEMORY_ERROR: 0.65,
            ErrorType.UNKNOWN: 0.40
        }.get(error_type, 0.70)
        
        # Adjust based on strategy
        strategy_factor = {
            FixStrategy.SAFE: 0.90,
            FixStrategy.OPTIMAL: 0.85,
            FixStrategy.QUICK: 0.70,
            FixStrategy.COMPREHENSIVE: 0.95
        }.get(strategy, 0.75)
        
        return min(1.0, base_confidence * complexity_factor * strategy_factor)
    
    def _generate_rollback_plan(self, code_changes: List[Dict[str, Any]]) -> str:
        """Generate a rollback plan"""
        if not code_changes:
            return "No changes to rollback"
        
        steps = [
            "1. Save current (fixed) version for reference",
            "2. Restore original code from version control",
            f"3. Revert changes in {len(code_changes)} location(s)",
            "4. Run tests to verify rollback",
            "5. Document why fix was unsuccessful"
        ]
        
        return "\n".join(steps)
    
    def _estimate_time(
        self, strategy: FixStrategy, code_changes: List[Dict[str, Any]]
    ) -> str:
        """Estimate implementation time"""
        base_time = len(code_changes) * 5  # 5 minutes per change
        
        multiplier = {
            FixStrategy.SAFE: 1.0,
            FixStrategy.QUICK: 0.5,
            FixStrategy.OPTIMAL: 1.5,
            FixStrategy.COMPREHENSIVE: 3.0
        }.get(strategy, 1.0)
        
        total_minutes = int(base_time * multiplier)
        
        if total_minutes < 15:
            return "5-15 minutes"
        elif total_minutes < 60:
            return f"{total_minutes} minutes"
        else:
            hours = total_minutes // 60
            return f"{hours} hour(s)"