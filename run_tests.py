#!/usr/bin/env python3
"""
ABOV3 Genesis - Comprehensive Test Runner
Runs all tests and validates the system is working correctly
"""

import sys
import os
import unittest
import asyncio
import json
from pathlib import Path
from datetime import datetime
import traceback

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import test modules
from tests.test_assistant import suite as assistant_suite


class TestReport:
    """Generate comprehensive test report"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'skipped': 0,
            'test_details': [],
            'coverage': {},
            'performance': {},
            'issues_found': [],
            'recommendations': []
        }
    
    def add_test_result(self, test_name: str, status: str, duration: float, 
                       details: str = None):
        """Add individual test result"""
        self.results['total_tests'] += 1
        
        if status == 'passed':
            self.results['passed'] += 1
        elif status == 'failed':
            self.results['failed'] += 1
        elif status == 'error':
            self.results['errors'] += 1
        elif status == 'skipped':
            self.results['skipped'] += 1
        
        self.results['test_details'].append({
            'test': test_name,
            'status': status,
            'duration': duration,
            'details': details
        })
    
    def generate_report(self) -> str:
        """Generate formatted test report"""
        report = []
        report.append("=" * 80)
        report.append("ABOV3 GENESIS - COMPREHENSIVE TEST REPORT")
        report.append("=" * 80)
        report.append(f"Timestamp: {self.results['timestamp']}")
        report.append("")
        
        # Summary
        report.append("TEST SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Tests: {self.results['total_tests']}")
        report.append(f"✅ Passed: {self.results['passed']}")
        report.append(f"❌ Failed: {self.results['failed']}")
        report.append(f"⚠️  Errors: {self.results['errors']}")
        report.append(f"⏭️  Skipped: {self.results['skipped']}")
        
        # Success rate
        if self.results['total_tests'] > 0:
            success_rate = (self.results['passed'] / self.results['total_tests']) * 100
            report.append(f"\nSuccess Rate: {success_rate:.1f}%")
            
            if success_rate == 100:
                report.append("🎉 PERFECT SCORE - ALL TESTS PASSED!")
            elif success_rate >= 90:
                report.append("✅ EXCELLENT - System is working well")
            elif success_rate >= 70:
                report.append("⚠️  GOOD - Some issues need attention")
            else:
                report.append("❌ CRITICAL - Major issues detected")
        
        report.append("")
        
        # Failed tests details
        if self.results['failed'] > 0 or self.results['errors'] > 0:
            report.append("FAILED/ERROR TESTS")
            report.append("-" * 40)
            for test in self.results['test_details']:
                if test['status'] in ['failed', 'error']:
                    report.append(f"❌ {test['test']}")
                    if test['details']:
                        report.append(f"   Details: {test['details'][:100]}")
            report.append("")
        
        # Issues found
        if self.results['issues_found']:
            report.append("ISSUES DETECTED")
            report.append("-" * 40)
            for issue in self.results['issues_found']:
                report.append(f"• {issue}")
            report.append("")
        
        # Recommendations
        if self.results['recommendations']:
            report.append("RECOMMENDATIONS")
            report.append("-" * 40)
            for rec in self.results['recommendations']:
                report.append(f"• {rec}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_report(self, filepath: str = None):
        """Save report to file"""
        if not filepath:
            filepath = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Report saved to: {filepath}")


class IntegrationTests(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        """Setup test environment"""
        self.test_dir = Path("test_workspace")
        self.test_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Cleanup test environment"""
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_code_generation_workflow(self):
        """Test complete code generation workflow"""
        from abov3.core.assistant_v2 import EnhancedAssistant
        from abov3.core.code_generator import CodeGenerator
        
        # Create assistant with test project
        assistant = EnhancedAssistant()
        assistant.code_generator = CodeGenerator(self.test_dir)
        
        # Mock AI response
        assistant._get_ai_response = unittest.mock.AsyncMock(
            return_value="""Here's a Python hello world:
```python
def hello():
    print("Hello, World!")

if __name__ == "__main__":
    hello()
```"""
        )
        
        # Test code generation
        loop = asyncio.get_event_loop()
        response = loop.run_until_complete(
            assistant._handle_code_generation("write hello world", {})
        )
        
        self.assertIn("Files Created", response)
        
        # Check file was created
        py_files = list(self.test_dir.glob("*.py"))
        self.assertGreater(len(py_files), 0)
    
    def test_error_handling(self):
        """Test error handling and recovery"""
        from abov3.core.error_handler import ErrorHandler, ErrorContext
        
        handler = ErrorHandler()
        context = ErrorContext()
        context.operation = "test_operation"
        
        # Test error handling
        try:
            raise ValueError("Test error")
        except Exception as e:
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(
                handler.handle_error(e, context, auto_recover=True)
            )
            
            self.assertTrue(result['error'])
            self.assertEqual(result['error_type'], 'ValueError')
    
    def test_validation(self):
        """Test code validation"""
        from abov3.core.validator import CodeValidator
        
        validator = CodeValidator()
        
        # Test valid Python code
        valid_code = """
def add(a, b):
    '''Add two numbers'''
    return a + b
"""
        result = validator.validate_code(valid_code, 'python')
        self.assertTrue(result['valid'])
        
        # Test invalid Python code
        invalid_code = "def broken("
        result = validator.validate_code(invalid_code, 'python')
        self.assertFalse(result['valid'])
    
    def test_debugging_tools(self):
        """Test debugging utilities"""
        from abov3.core.debugger import CodeDebugger, PerformanceProfiler
        
        debugger = CodeDebugger()
        profiler = PerformanceProfiler()
        
        # Test function tracing
        @debugger.trace_execution
        def test_function(x):
            return x * 2
        
        result = test_function(5)
        self.assertEqual(result, 10)
        self.assertGreater(len(debugger.execution_trace), 0)
        
        # Test profiling
        with profiler.profile("test_operation"):
            _ = sum(range(1000))
        
        self.assertIn("test_operation", profiler.metrics['execution_times'])


def run_all_tests():
    """Run all test suites"""
    print("\n" + "="*80)
    print("ABOV3 GENESIS - COMPREHENSIVE TESTING SUITE")
    print("="*80 + "\n")
    
    report = TestReport()
    
    # Create test loader
    loader = unittest.TestLoader()
    
    # Load all test suites
    suites = []
    
    # Assistant tests
    print("Loading Assistant tests...")
    suites.append(assistant_suite())
    
    # Integration tests
    print("Loading Integration tests...")
    suites.append(loader.loadTestsFromTestCase(IntegrationTests))
    
    # Combine all suites
    combined_suite = unittest.TestSuite(suites)
    
    # Run tests with custom result handler
    class CustomTestResult(unittest.TextTestResult):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.test_times = {}
            self.current_test_start = None
        
        def startTest(self, test):
            super().startTest(test)
            self.current_test_start = datetime.now()
        
        def addSuccess(self, test):
            super().addSuccess(test)
            duration = (datetime.now() - self.current_test_start).total_seconds()
            report.add_test_result(str(test), 'passed', duration)
        
        def addError(self, test, err):
            super().addError(test, err)
            duration = (datetime.now() - self.current_test_start).total_seconds()
            error_msg = ''.join(traceback.format_exception(*err))
            report.add_test_result(str(test), 'error', duration, error_msg)
        
        def addFailure(self, test, err):
            super().addFailure(test, err)
            duration = (datetime.now() - self.current_test_start).total_seconds()
            error_msg = ''.join(traceback.format_exception(*err))
            report.add_test_result(str(test), 'failed', duration, error_msg)
        
        def addSkip(self, test, reason):
            super().addSkip(test, reason)
            report.add_test_result(str(test), 'skipped', 0, reason)
    
    # Run tests
    runner = unittest.TextTestRunner(
        verbosity=2,
        resultclass=CustomTestResult
    )
    
    print("\n" + "="*80)
    print("RUNNING TESTS")
    print("="*80 + "\n")
    
    result = runner.run(combined_suite)
    
    # Analyze results
    if result.failures:
        report.results['issues_found'].append(f"Found {len(result.failures)} test failures")
    
    if result.errors:
        report.results['issues_found'].append(f"Found {len(result.errors)} test errors")
    
    # Generate recommendations
    if report.results['failed'] > 0:
        report.results['recommendations'].append("Fix failing tests before deployment")
    
    if report.results['errors'] > 0:
        report.results['recommendations'].append("Resolve test errors - may indicate system issues")
    
    # Check specific components
    detection_tests_passed = sum(1 for t in report.results['test_details'] 
                                 if 'detection' in t['test'].lower() and t['status'] == 'passed')
    
    if detection_tests_passed < 5:
        report.results['recommendations'].append("Request detection needs improvement")
    
    # Print report
    print("\n" + report.generate_report())
    
    # Save report
    report.save_report()
    
    # Return success/failure
    return result.wasSuccessful()


def validate_installation():
    """Validate ABOV3 Genesis installation"""
    print("\n" + "="*80)
    print("VALIDATING ABOV3 GENESIS INSTALLATION")
    print("="*80 + "\n")
    
    issues = []
    
    # Check required modules
    required_modules = [
        'abov3.core.assistant',
        'abov3.core.assistant_v2',
        'abov3.core.code_generator',
        'abov3.core.error_handler',
        'abov3.core.validator',
        'abov3.core.debugger'
    ]
    
    for module_name in required_modules:
        try:
            __import__(module_name)
            print(f"✅ {module_name} - OK")
        except ImportError as e:
            print(f"❌ {module_name} - FAILED: {e}")
            issues.append(f"Missing module: {module_name}")
    
    # Check Ollama
    try:
        from abov3.core.ollama_client import OllamaClient
        client = OllamaClient()
        loop = asyncio.get_event_loop()
        available = loop.run_until_complete(client.is_available())
        if available:
            print("✅ Ollama - AVAILABLE")
        else:
            print("⚠️  Ollama - NOT RUNNING (tests will use mocks)")
    except Exception as e:
        print(f"⚠️  Ollama - ERROR: {e}")
    
    # Check project structure
    required_dirs = ['abov3', 'tests', 'abov3/core', 'abov3/agents']
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✅ Directory {dir_name} - EXISTS")
        else:
            print(f"❌ Directory {dir_name} - MISSING")
            issues.append(f"Missing directory: {dir_name}")
    
    if issues:
        print(f"\n⚠️  Found {len(issues)} issues:")
        for issue in issues:
            print(f"  • {issue}")
    else:
        print("\n✅ Installation validated successfully!")
    
    return len(issues) == 0


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║           ABOV3 GENESIS - QUALITY ASSURANCE SUITE           ║
║                                                              ║
║  Comprehensive Testing & Validation for Zero-Defect Delivery ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # First validate installation
    if not validate_installation():
        print("\n❌ Installation validation failed. Please fix issues before running tests.")
        sys.exit(1)
    
    # Run all tests
    success = run_all_tests()
    
    if success:
        print("""
╔══════════════════════════════════════════════════════════════╗
║                    🎉 ALL TESTS PASSED! 🎉                   ║
║                                                              ║
║         ABOV3 Genesis is ready for production use!          ║
╚══════════════════════════════════════════════════════════════╝
""")
        sys.exit(0)
    else:
        print("""
╔══════════════════════════════════════════════════════════════╗
║                  ⚠️  TESTS FAILED ⚠️                          ║
║                                                              ║
║      Please review the test report and fix issues           ║
╚══════════════════════════════════════════════════════════════╝
""")
        sys.exit(1)