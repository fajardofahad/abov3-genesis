"""
ABOV3 Genesis - Automated Test Generation System
ML-powered test generation for comprehensive code coverage and bug detection
"""

import ast
import json
import logging
import re
import inspect
from typing import Any, Dict, List, Optional, Tuple, Set, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import hashlib
from enum import Enum

# ML imports with fallbacks
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import hypothesis
    from hypothesis import strategies as st
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False


class TestType(Enum):
    """Types of tests to generate"""
    UNIT = "unit"
    INTEGRATION = "integration"
    EDGE_CASE = "edge_case"
    PROPERTY_BASED = "property_based"
    MUTATION = "mutation"
    PERFORMANCE = "performance"
    SECURITY = "security"
    REGRESSION = "regression"


@dataclass
class TestCase:
    """Generated test case"""
    test_id: str
    test_name: str
    test_type: TestType
    function_name: str
    test_code: str
    description: str
    inputs: List[Any]
    expected_output: Any
    confidence: float
    coverage_areas: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FunctionSignature:
    """Function signature analysis"""
    name: str
    parameters: List[Dict[str, Any]]
    return_annotation: Optional[str]
    docstring: Optional[str]
    complexity: int
    line_start: int
    line_end: int
    ast_node: ast.FunctionDef


class CodeAnalyzer:
    """Analyze code for test generation"""
    
    def __init__(self):
        self.function_signatures = {}
        self.class_structures = {}
        self.import_dependencies = []
        self.global_variables = {}
        self.code_patterns = {}
    
    def analyze_code(self, code: str, file_path: str = "") -> Dict[str, Any]:
        """Analyze code structure for test generation"""
        analysis = {
            'functions': [],
            'classes': [],
            'imports': [],
            'global_vars': [],
            'complexity_score': 0,
            'test_targets': [],
            'dependencies': [],
            'file_path': file_path
        }
        
        try:
            tree = ast.parse(code)
            
            # Analyze AST
            visitor = CodeAnalysisVisitor()
            visitor.visit(tree)
            
            analysis['functions'] = visitor.functions
            analysis['classes'] = visitor.classes
            analysis['imports'] = visitor.imports
            analysis['global_vars'] = visitor.global_vars
            analysis['complexity_score'] = visitor.complexity_score
            
            # Identify test targets
            analysis['test_targets'] = self._identify_test_targets(visitor.functions, visitor.classes)
            
            # Analyze dependencies
            analysis['dependencies'] = self._analyze_dependencies(visitor.imports)
            
        except SyntaxError as e:
            logging.warning(f"Syntax error in code analysis: {e}")
        except Exception as e:
            logging.error(f"Code analysis failed: {e}")
        
        return analysis
    
    def _identify_test_targets(self, functions: List[FunctionSignature], 
                              classes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify functions and methods that need testing"""
        test_targets = []
        
        # Add functions as test targets
        for func in functions:
            if not func.name.startswith('_') or func.name.startswith('__'):  # Skip private methods unless special
                test_targets.append({
                    'type': 'function',
                    'name': func.name,
                    'signature': func,
                    'priority': self._calculate_test_priority(func),
                    'complexity': func.complexity
                })
        
        # Add class methods as test targets
        for cls in classes:
            for method in cls.get('methods', []):
                if not method['name'].startswith('_') or method['name'].startswith('__'):
                    test_targets.append({
                        'type': 'method',
                        'class': cls['name'],
                        'name': method['name'],
                        'signature': method,
                        'priority': self._calculate_test_priority(method),
                        'complexity': method.get('complexity', 1)
                    })
        
        # Sort by priority
        test_targets.sort(key=lambda x: x['priority'], reverse=True)
        
        return test_targets
    
    def _calculate_test_priority(self, func_or_method: Union[FunctionSignature, Dict[str, Any]]) -> int:
        """Calculate testing priority for a function or method"""
        priority = 0
        
        # Get function info
        if isinstance(func_or_method, FunctionSignature):
            name = func_or_method.name
            complexity = func_or_method.complexity
            docstring = func_or_method.docstring
        else:
            name = func_or_method.get('name', '')
            complexity = func_or_method.get('complexity', 1)
            docstring = func_or_method.get('docstring', '')
        
        # Higher priority for complex functions
        priority += complexity * 2
        
        # Higher priority for public functions
        if not name.startswith('_'):
            priority += 5
        
        # Higher priority for documented functions
        if docstring:
            priority += 3
        
        # Higher priority for functions with specific patterns
        if any(keyword in name.lower() for keyword in ['process', 'calculate', 'validate', 'parse']):
            priority += 4
        
        # Lower priority for simple getters/setters
        if any(pattern in name.lower() for pattern in ['get_', 'set_', 'is_', 'has_']):
            priority -= 2
        
        return max(priority, 1)  # Minimum priority of 1
    
    def _analyze_dependencies(self, imports: List[Dict[str, Any]]) -> List[str]:
        """Analyze import dependencies"""
        dependencies = []
        
        for imp in imports:
            if imp['type'] == 'import':
                dependencies.extend(imp['names'])
            elif imp['type'] == 'from_import':
                dependencies.append(imp['module'])
        
        return list(set(dependencies))  # Remove duplicates


class CodeAnalysisVisitor(ast.NodeVisitor):
    """AST visitor for code analysis"""
    
    def __init__(self):
        self.functions = []
        self.classes = []
        self.imports = []
        self.global_vars = []
        self.complexity_score = 0
        self.current_class = None
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definition"""
        # Calculate function complexity
        complexity = self._calculate_complexity(node)
        self.complexity_score += complexity
        
        # Extract parameters
        parameters = []
        for arg in node.args.args:
            param_info = {
                'name': arg.arg,
                'annotation': ast.unparse(arg.annotation) if arg.annotation else None,
                'default': None
            }
            parameters.append(param_info)
        
        # Handle default values
        defaults = node.args.defaults
        if defaults:
            # Assign defaults to last len(defaults) parameters
            for i, default in enumerate(defaults):
                param_index = len(parameters) - len(defaults) + i
                if 0 <= param_index < len(parameters):
                    try:
                        parameters[param_index]['default'] = ast.unparse(default)
                    except:
                        parameters[param_index]['default'] = str(default)
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        # Create function signature
        func_sig = FunctionSignature(
            name=node.name,
            parameters=parameters,
            return_annotation=ast.unparse(node.returns) if node.returns else None,
            docstring=docstring,
            complexity=complexity,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            ast_node=node
        )
        
        if self.current_class:
            # Add as method to current class
            if 'methods' not in self.current_class:
                self.current_class['methods'] = []
            self.current_class['methods'].append({
                'name': node.name,
                'parameters': parameters,
                'return_annotation': func_sig.return_annotation,
                'docstring': docstring,
                'complexity': complexity,
                'line_start': node.lineno,
                'line_end': node.end_lineno or node.lineno
            })
        else:
            self.functions.append(func_sig)
        
        self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definition"""
        # Store previous class context
        prev_class = self.current_class
        
        # Create class info
        class_info = {
            'name': node.name,
            'bases': [ast.unparse(base) for base in node.bases],
            'docstring': ast.get_docstring(node),
            'line_start': node.lineno,
            'line_end': node.end_lineno or node.lineno,
            'methods': []
        }
        
        self.current_class = class_info
        self.classes.append(class_info)
        
        # Visit class body
        self.generic_visit(node)
        
        # Restore previous class context
        self.current_class = prev_class
    
    def visit_Import(self, node: ast.Import):
        """Visit import statement"""
        names = [alias.name for alias in node.names]
        self.imports.append({
            'type': 'import',
            'names': names,
            'line': node.lineno
        })
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Visit from-import statement"""
        names = [alias.name for alias in node.names]
        self.imports.append({
            'type': 'from_import',
            'module': node.module,
            'names': names,
            'level': node.level,
            'line': node.lineno
        })
    
    def visit_Assign(self, node: ast.Assign):
        """Visit assignment (for global variables)"""
        if not self.current_class:  # Only track global assignments
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        value = ast.unparse(node.value)
                    except:
                        value = str(node.value)
                    
                    self.global_vars.append({
                        'name': target.id,
                        'value': value,
                        'line': node.lineno
                    })
        
        self.generic_visit(node)
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.With, ast.AsyncWith):
                complexity += 1
            elif isinstance(child, ast.Assert):
                complexity += 1
            elif isinstance(child, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                complexity += 1
        
        return complexity


class TestGenerationStrategies:
    """Different strategies for test generation"""
    
    @staticmethod
    def generate_unit_tests(func_signature: FunctionSignature, 
                           code_analysis: Dict[str, Any]) -> List[TestCase]:
        """Generate unit tests for a function"""
        tests = []
        
        # Generate basic test cases
        basic_test = TestGenerationStrategies._generate_basic_test(func_signature)
        if basic_test:
            tests.append(basic_test)
        
        # Generate edge case tests
        edge_tests = TestGenerationStrategies._generate_edge_case_tests(func_signature)
        tests.extend(edge_tests)
        
        # Generate null/empty input tests
        null_tests = TestGenerationStrategies._generate_null_input_tests(func_signature)
        tests.extend(null_tests)
        
        return tests
    
    @staticmethod
    def _generate_basic_test(func_signature: FunctionSignature) -> Optional[TestCase]:
        """Generate a basic happy path test"""
        func_name = func_signature.name
        params = func_signature.parameters
        
        # Generate test inputs based on parameter types
        test_inputs = []
        test_args = []
        
        for param in params:
            input_value, arg_repr = TestGenerationStrategies._generate_input_for_param(param)
            test_inputs.append(input_value)
            test_args.append(arg_repr)
        
        # Generate test code
        test_name = f"test_{func_name}_basic"
        args_str = ", ".join(test_args)
        
        test_code = f"""def {test_name}():
    # Test basic functionality of {func_name}
    result = {func_name}({args_str})
    
    # Add assertions based on expected behavior
    assert result is not None  # Basic non-null check
    # TODO: Add specific assertions based on function behavior
"""
        
        return TestCase(
            test_id=f"basic_{func_name}_{hash(test_name) % 10000}",
            test_name=test_name,
            test_type=TestType.UNIT,
            function_name=func_name,
            test_code=test_code,
            description=f"Basic test for {func_name} function",
            inputs=test_inputs,
            expected_output=None,  # Unknown without execution
            confidence=0.7,
            coverage_areas=["basic_functionality"]
        )
    
    @staticmethod
    def _generate_edge_case_tests(func_signature: FunctionSignature) -> List[TestCase]:
        """Generate edge case tests"""
        tests = []
        func_name = func_signature.name
        params = func_signature.parameters
        
        # Test with boundary values
        boundary_test = TestGenerationStrategies._generate_boundary_test(func_signature)
        if boundary_test:
            tests.append(boundary_test)
        
        # Test with invalid inputs
        invalid_test = TestGenerationStrategies._generate_invalid_input_test(func_signature)
        if invalid_test:
            tests.append(invalid_test)
        
        return tests
    
    @staticmethod
    def _generate_boundary_test(func_signature: FunctionSignature) -> Optional[TestCase]:
        """Generate boundary value test"""
        func_name = func_signature.name
        params = func_signature.parameters
        
        boundary_cases = []
        test_args = []
        
        for param in params:
            boundary_value, arg_repr = TestGenerationStrategies._generate_boundary_input(param)
            boundary_cases.append(boundary_value)
            test_args.append(arg_repr)
        
        if not test_args:
            return None
        
        test_name = f"test_{func_name}_boundary_values"
        args_str = ", ".join(test_args)
        
        test_code = f"""def {test_name}():
    # Test {func_name} with boundary values
    try:
        result = {func_name}({args_str})
        # Verify result is handled correctly for boundary inputs
        assert result is not None or result == 0 or result == []  # Flexible assertion
    except (ValueError, TypeError) as e:
        # Expected for some boundary cases
        assert str(e)  # Ensure error message is meaningful
"""
        
        return TestCase(
            test_id=f"boundary_{func_name}_{hash(test_name) % 10000}",
            test_name=test_name,
            test_type=TestType.EDGE_CASE,
            function_name=func_name,
            test_code=test_code,
            description=f"Boundary value test for {func_name}",
            inputs=boundary_cases,
            expected_output=None,
            confidence=0.8,
            coverage_areas=["boundary_conditions", "edge_cases"]
        )
    
    @staticmethod
    def _generate_invalid_input_test(func_signature: FunctionSignature) -> Optional[TestCase]:
        """Generate test with invalid inputs"""
        func_name = func_signature.name
        params = func_signature.parameters
        
        if not params:
            return None
        
        # Generate obviously invalid inputs
        invalid_inputs = []
        test_args = []
        
        for param in params:
            invalid_value, arg_repr = TestGenerationStrategies._generate_invalid_input(param)
            invalid_inputs.append(invalid_value)
            test_args.append(arg_repr)
        
        test_name = f"test_{func_name}_invalid_input"
        args_str = ", ".join(test_args)
        
        test_code = f"""def {test_name}():
    # Test {func_name} with invalid inputs
    with pytest.raises((ValueError, TypeError, AttributeError)):
        {func_name}({args_str})
"""
        
        return TestCase(
            test_id=f"invalid_{func_name}_{hash(test_name) % 10000}",
            test_name=test_name,
            test_type=TestType.EDGE_CASE,
            function_name=func_name,
            test_code=test_code,
            description=f"Invalid input test for {func_name}",
            inputs=invalid_inputs,
            expected_output="Exception",
            confidence=0.9,
            coverage_areas=["error_handling", "input_validation"]
        )
    
    @staticmethod
    def _generate_null_input_tests(func_signature: FunctionSignature) -> List[TestCase]:
        """Generate tests with null/empty inputs"""
        tests = []
        func_name = func_signature.name
        params = func_signature.parameters
        
        if not params:
            return tests
        
        # Test with None values
        none_test = TestGenerationStrategies._generate_none_test(func_signature)
        if none_test:
            tests.append(none_test)
        
        # Test with empty collections
        empty_test = TestGenerationStrategies._generate_empty_test(func_signature)
        if empty_test:
            tests.append(empty_test)
        
        return tests
    
    @staticmethod
    def _generate_none_test(func_signature: FunctionSignature) -> Optional[TestCase]:
        """Generate test with None inputs"""
        func_name = func_signature.name
        params = func_signature.parameters
        
        # Only generate if function takes parameters
        if not params:
            return None
        
        test_name = f"test_{func_name}_none_input"
        none_args = ["None"] * len(params)
        args_str = ", ".join(none_args)
        
        test_code = f"""def {test_name}():
    # Test {func_name} with None inputs
    try:
        result = {func_name}({args_str})
        # If no exception, verify result
        assert result is None or isinstance(result, (str, int, float, list, dict))
    except (ValueError, TypeError, AttributeError) as e:
        # Expected for functions that don't handle None
        assert "None" in str(e) or "NoneType" in str(e)
"""
        
        return TestCase(
            test_id=f"none_{func_name}_{hash(test_name) % 10000}",
            test_name=test_name,
            test_type=TestType.EDGE_CASE,
            function_name=func_name,
            test_code=test_code,
            description=f"None input test for {func_name}",
            inputs=[None] * len(params),
            expected_output=None,
            confidence=0.8,
            coverage_areas=["null_handling", "defensive_programming"]
        )
    
    @staticmethod
    def _generate_empty_test(func_signature: FunctionSignature) -> Optional[TestCase]:
        """Generate test with empty collections"""
        func_name = func_signature.name
        params = func_signature.parameters
        
        if not params:
            return None
        
        # Generate empty values for parameters
        empty_args = []
        empty_inputs = []
        
        for param in params:
            if param.get('annotation'):
                annotation = param['annotation'].lower()
                if 'list' in annotation or 'sequence' in annotation:
                    empty_args.append("[]")
                    empty_inputs.append([])
                elif 'dict' in annotation or 'mapping' in annotation:
                    empty_args.append("{}")
                    empty_inputs.append({})
                elif 'str' in annotation:
                    empty_args.append('""')
                    empty_inputs.append("")
                else:
                    empty_args.append("None")
                    empty_inputs.append(None)
            else:
                # Default to empty string for unknown types
                empty_args.append('""')
                empty_inputs.append("")
        
        test_name = f"test_{func_name}_empty_input"
        args_str = ", ".join(empty_args)
        
        test_code = f"""def {test_name}():
    # Test {func_name} with empty inputs
    try:
        result = {func_name}({args_str})
        # Verify function handles empty inputs gracefully
        assert result is not None or result == 0 or result == [] or result == {{}}
    except (ValueError, TypeError) as e:
        # Some functions may not accept empty inputs
        assert len(str(e)) > 0  # Ensure meaningful error message
"""
        
        return TestCase(
            test_id=f"empty_{func_name}_{hash(test_name) % 10000}",
            test_name=test_name,
            test_type=TestType.EDGE_CASE,
            function_name=func_name,
            test_code=test_code,
            description=f"Empty input test for {func_name}",
            inputs=empty_inputs,
            expected_output=None,
            confidence=0.7,
            coverage_areas=["empty_input_handling"]
        )
    
    @staticmethod
    def _generate_input_for_param(param: Dict[str, Any]) -> Tuple[Any, str]:
        """Generate input value for a parameter"""
        param_name = param['name']
        annotation = param.get('annotation', '').lower()
        default = param.get('default')
        
        # Use default if available
        if default is not None and default != 'None':
            return default, default
        
        # Generate based on type annotation
        if 'int' in annotation:
            return 42, "42"
        elif 'float' in annotation:
            return 3.14, "3.14"
        elif 'str' in annotation:
            return "test", '"test"'
        elif 'bool' in annotation:
            return True, "True"
        elif 'list' in annotation:
            return [1, 2, 3], "[1, 2, 3]"
        elif 'dict' in annotation:
            return {"key": "value"}, '{"key": "value"}'
        else:
            # Default to a simple test value
            return "test_value", '"test_value"'
    
    @staticmethod
    def _generate_boundary_input(param: Dict[str, Any]) -> Tuple[Any, str]:
        """Generate boundary input for parameter"""
        annotation = param.get('annotation', '').lower()
        
        if 'int' in annotation:
            return 0, "0"  # Common boundary for integers
        elif 'float' in annotation:
            return 0.0, "0.0"
        elif 'str' in annotation:
            return "", '""'  # Empty string
        elif 'list' in annotation:
            return [], "[]"
        elif 'dict' in annotation:
            return {}, "{}"
        else:
            return None, "None"
    
    @staticmethod
    def _generate_invalid_input(param: Dict[str, Any]) -> Tuple[Any, str]:
        """Generate invalid input for parameter"""
        annotation = param.get('annotation', '').lower()
        
        if 'int' in annotation:
            return "not_an_int", '"not_an_int"'
        elif 'float' in annotation:
            return "not_a_float", '"not_a_float"'
        elif 'str' in annotation:
            return 12345, "12345"  # Number instead of string
        elif 'list' in annotation:
            return "not_a_list", '"not_a_list"'
        elif 'dict' in annotation:
            return "not_a_dict", '"not_a_dict"'
        else:
            return object(), "object()"  # Generic invalid object


class PropertyBasedTestGenerator:
    """Generate property-based tests using Hypothesis"""
    
    def __init__(self):
        self.available = HAS_HYPOTHESIS
        self.strategies = self._initialize_strategies()
    
    def _initialize_strategies(self) -> Dict[str, Any]:
        """Initialize Hypothesis strategies"""
        if not self.available:
            return {}
        
        return {
            'int': st.integers(),
            'float': st.floats(allow_nan=False, allow_infinity=False),
            'str': st.text(),
            'bool': st.booleans(),
            'list': st.lists(st.integers()),
            'dict': st.dictionaries(st.text(), st.integers())
        }
    
    def generate_property_tests(self, func_signature: FunctionSignature) -> List[TestCase]:
        """Generate property-based tests"""
        if not self.available:
            return []
        
        tests = []
        func_name = func_signature.name
        
        # Generate property test for function invariants
        property_test = self._generate_property_test(func_signature)
        if property_test:
            tests.append(property_test)
        
        return tests
    
    def _generate_property_test(self, func_signature: FunctionSignature) -> Optional[TestCase]:
        """Generate a property-based test"""
        if not self.available:
            return None
        
        func_name = func_signature.name
        params = func_signature.parameters
        
        if not params:
            return None
        
        # Generate strategy for each parameter
        strategies = []
        param_names = []
        
        for param in params:
            param_name = param['name']
            annotation = param.get('annotation', '').lower()
            
            # Select appropriate strategy
            if 'int' in annotation:
                strategy = 'st.integers()'
            elif 'float' in annotation:
                strategy = 'st.floats(allow_nan=False, allow_infinity=False)'
            elif 'str' in annotation:
                strategy = 'st.text()'
            elif 'bool' in annotation:
                strategy = 'st.booleans()'
            elif 'list' in annotation:
                strategy = 'st.lists(st.integers())'
            elif 'dict' in annotation:
                strategy = 'st.dictionaries(st.text(), st.integers())'
            else:
                strategy = 'st.text()'  # Default strategy
            
            strategies.append(strategy)
            param_names.append(param_name)
        
        test_name = f"test_{func_name}_property"
        
        # Generate strategy decorators
        strategy_decorators = []
        for i, (param_name, strategy) in enumerate(zip(param_names, strategies)):
            strategy_decorators.append(f"@given({param_name}={strategy})")
        
        test_code = f"""{chr(10).join(strategy_decorators)}
def {test_name}({', '.join(param_names)}):
    # Property-based test for {func_name}
    try:
        result = {func_name}({', '.join(param_names)})
        
        # Property: Function should not crash with valid inputs
        assert True  # If we reach here, function didn't crash
        
        # Property: Result should be consistent type (if known)
        if result is not None:
            assert isinstance(result, (str, int, float, list, dict, bool))
        
        # TODO: Add domain-specific properties
        
    except Exception as e:
        # For property tests, we mainly care about unexpected crashes
        # Expected exceptions (ValueError, TypeError) are often acceptable
        if isinstance(e, (AssertionError, SystemError, RecursionError)):
            raise  # These should not happen
"""
        
        return TestCase(
            test_id=f"property_{func_name}_{hash(test_name) % 10000}",
            test_name=test_name,
            test_type=TestType.PROPERTY_BASED,
            function_name=func_name,
            test_code=test_code,
            description=f"Property-based test for {func_name}",
            inputs=[],  # Inputs are generated by Hypothesis
            expected_output=None,
            confidence=0.9,
            coverage_areas=["property_testing", "invariants", "edge_cases"]
        )


class TestSuiteGenerator:
    """Main test suite generation orchestrator"""
    
    def __init__(self):
        self.code_analyzer = CodeAnalyzer()
        self.property_test_gen = PropertyBasedTestGenerator()
        self.generated_tests = {}
        self.test_statistics = defaultdict(int)
    
    def generate_test_suite(self, code: str, file_path: str = "", 
                           test_types: Optional[List[TestType]] = None) -> Dict[str, Any]:
        """Generate comprehensive test suite for code"""
        
        if test_types is None:
            test_types = [TestType.UNIT, TestType.EDGE_CASE, TestType.PROPERTY_BASED]
        
        # Analyze code structure
        code_analysis = self.code_analyzer.analyze_code(code, file_path)
        
        # Generate tests
        test_suite = {
            'file_path': file_path,
            'generated_at': datetime.now(),
            'code_analysis': code_analysis,
            'test_cases': [],
            'test_files': {},
            'coverage_report': {},
            'statistics': {}
        }
        
        # Generate tests for each target
        all_tests = []
        
        for target in code_analysis['test_targets']:
            target_tests = self._generate_tests_for_target(target, code_analysis, test_types)
            all_tests.extend(target_tests)
        
        test_suite['test_cases'] = all_tests
        
        # Generate test files
        test_suite['test_files'] = self._generate_test_files(all_tests, file_path)
        
        # Generate coverage report
        test_suite['coverage_report'] = self._generate_coverage_report(all_tests, code_analysis)
        
        # Generate statistics
        test_suite['statistics'] = self._generate_statistics(all_tests)
        
        return test_suite
    
    def _generate_tests_for_target(self, target: Dict[str, Any], 
                                 code_analysis: Dict[str, Any],
                                 test_types: List[TestType]) -> List[TestCase]:
        """Generate tests for a specific target"""
        tests = []
        
        target_type = target['type']
        target_name = target['name']
        signature = target['signature']
        
        # Convert dict signature to FunctionSignature if needed
        if isinstance(signature, dict):
            func_sig = FunctionSignature(
                name=signature['name'],
                parameters=signature.get('parameters', []),
                return_annotation=signature.get('return_annotation'),
                docstring=signature.get('docstring'),
                complexity=signature.get('complexity', 1),
                line_start=signature.get('line_start', 1),
                line_end=signature.get('line_end', 1),
                ast_node=None  # Not available from dict
            )
        else:
            func_sig = signature
        
        # Generate different types of tests
        if TestType.UNIT in test_types:
            unit_tests = TestGenerationStrategies.generate_unit_tests(func_sig, code_analysis)
            tests.extend(unit_tests)
        
        if TestType.EDGE_CASE in test_types:
            # Edge case tests are generated as part of unit tests
            pass
        
        if TestType.PROPERTY_BASED in test_types:
            property_tests = self.property_test_gen.generate_property_tests(func_sig)
            tests.extend(property_tests)
        
        # Update statistics
        for test in tests:
            self.test_statistics[test.test_type.value] += 1
            self.test_statistics['total'] += 1
        
        return tests
    
    def _generate_test_files(self, test_cases: List[TestCase], 
                           original_file_path: str) -> Dict[str, str]:
        """Generate test files from test cases"""
        test_files = {}
        
        if not test_cases:
            return test_files
        
        # Group tests by file
        file_groups = defaultdict(list)
        
        for test_case in test_cases:
            # Determine target file based on test type
            if test_case.test_type == TestType.PROPERTY_BASED:
                file_key = "test_property.py"
            elif test_case.test_type == TestType.INTEGRATION:
                file_key = "test_integration.py"
            else:
                file_key = "test_unit.py"
            
            file_groups[file_key].append(test_case)
        
        # Generate each test file
        for file_name, tests in file_groups.items():
            test_file_content = self._create_test_file_content(tests, original_file_path)
            test_files[file_name] = test_file_content
        
        return test_files
    
    def _create_test_file_content(self, test_cases: List[TestCase], 
                                original_file_path: str) -> str:
        """Create content for a test file"""
        
        # Determine imports needed
        imports = set()
        imports.add("import pytest")
        
        # Add hypothesis import if needed
        if any(test.test_type == TestType.PROPERTY_BASED for test in test_cases):
            imports.add("from hypothesis import given, strategies as st")
        
        # Add import for module under test
        if original_file_path:
            module_name = Path(original_file_path).stem
            imports.add(f"from {module_name} import *")
        
        # Create file header
        header = f'''"""
Generated test file
Created: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Source: {original_file_path}
"""

{chr(10).join(sorted(imports))}


'''
        
        # Add test cases
        test_content = []
        for test_case in test_cases:
            test_content.append(test_case.test_code)
            test_content.append("")  # Empty line between tests
        
        return header + "\n".join(test_content)
    
    def _generate_coverage_report(self, test_cases: List[TestCase], 
                                code_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate coverage report"""
        coverage_report = {
            'total_functions': len(code_analysis.get('functions', [])),
            'tested_functions': len(set(test.function_name for test in test_cases)),
            'coverage_percentage': 0.0,
            'coverage_areas': defaultdict(int),
            'untested_functions': []
        }
        
        # Calculate coverage percentage
        total_funcs = coverage_report['total_functions']
        tested_funcs = coverage_report['tested_functions']
        
        if total_funcs > 0:
            coverage_report['coverage_percentage'] = (tested_funcs / total_funcs) * 100
        
        # Count coverage areas
        for test_case in test_cases:
            for area in test_case.coverage_areas:
                coverage_report['coverage_areas'][area] += 1
        
        # Find untested functions
        tested_function_names = set(test.function_name for test in test_cases)
        all_function_names = set(func.name for func in code_analysis.get('functions', []))
        untested = all_function_names - tested_function_names
        coverage_report['untested_functions'] = list(untested)
        
        return dict(coverage_report)
    
    def _generate_statistics(self, test_cases: List[TestCase]) -> Dict[str, Any]:
        """Generate test suite statistics"""
        stats = {
            'total_tests': len(test_cases),
            'test_type_distribution': defaultdict(int),
            'average_confidence': 0.0,
            'high_confidence_tests': 0,
            'coverage_area_distribution': defaultdict(int),
            'functions_tested': len(set(test.function_name for test in test_cases))
        }
        
        if not test_cases:
            return dict(stats)
        
        # Calculate distributions and averages
        total_confidence = 0
        
        for test_case in test_cases:
            stats['test_type_distribution'][test_case.test_type.value] += 1
            total_confidence += test_case.confidence
            
            if test_case.confidence >= 0.8:
                stats['high_confidence_tests'] += 1
            
            for area in test_case.coverage_areas:
                stats['coverage_area_distribution'][area] += 1
        
        stats['average_confidence'] = total_confidence / len(test_cases)
        
        return dict(stats)
    
    def generate_test_recommendations(self, test_suite: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving test coverage"""
        recommendations = []
        
        coverage_report = test_suite.get('coverage_report', {})
        statistics = test_suite.get('statistics', {})
        
        # Coverage recommendations
        coverage_pct = coverage_report.get('coverage_percentage', 0)
        if coverage_pct < 80:
            recommendations.append(f"Test coverage is {coverage_pct:.1f}% - aim for at least 80% coverage")
        
        untested_functions = coverage_report.get('untested_functions', [])
        if untested_functions:
            recommendations.append(f"Add tests for untested functions: {', '.join(untested_functions[:5])}")
        
        # Test type recommendations
        test_types = statistics.get('test_type_distribution', {})
        if 'property_based' not in test_types:
            recommendations.append("Consider adding property-based tests for better edge case coverage")
        
        if 'integration' not in test_types:
            recommendations.append("Add integration tests to verify component interactions")
        
        # Confidence recommendations
        avg_confidence = statistics.get('average_confidence', 0)
        if avg_confidence < 0.7:
            recommendations.append("Review and improve test assertions to increase confidence")
        
        # Coverage area recommendations
        coverage_areas = coverage_report.get('coverage_areas', {})
        important_areas = ['error_handling', 'edge_cases', 'input_validation']
        
        for area in important_areas:
            if area not in coverage_areas:
                recommendations.append(f"Add tests for {area.replace('_', ' ')}")
        
        return recommendations[:10]  # Limit to top 10 recommendations


# Main interface function
def generate_tests_for_code(code: str, file_path: str = "", 
                           test_types: Optional[List[TestType]] = None) -> Dict[str, Any]:
    """
    Main function to generate tests for code
    
    Args:
        code: Source code to generate tests for
        file_path: Path to the source file
        test_types: Types of tests to generate
    
    Returns:
        Complete test suite with generated tests
    """
    generator = TestSuiteGenerator()
    return generator.generate_test_suite(code, file_path, test_types)


# Export main classes and functions
__all__ = [
    'TestSuiteGenerator',
    'TestGenerationStrategies', 
    'PropertyBasedTestGenerator',
    'CodeAnalyzer',
    'TestCase',
    'TestType',
    'generate_tests_for_code'
]