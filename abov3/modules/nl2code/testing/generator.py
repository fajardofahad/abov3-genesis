"""
ABOV3 Genesis - Test Generation System
Automatic test generation for comprehensive test coverage
"""

import asyncio
import re
import ast
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import logging

from ..core.processor import FeatureRequirement, TechnicalContext
from ..planning.engine import ImplementationPlan, TaskStep
from ..generation.engine import CodeFile

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Represents a single test case"""
    name: str
    description: str
    test_type: str  # 'unit', 'integration', 'e2e', 'performance', 'security'
    category: str  # 'positive', 'negative', 'edge_case', 'boundary'
    setup: str
    test_code: str
    assertions: List[str]
    dependencies: List[str]
    mock_requirements: List[str]
    expected_coverage: float
    priority: int  # 1-5, 1 being highest priority


@dataclass
class TestSuite:
    """Represents a complete test suite"""
    name: str
    description: str
    test_cases: List[TestCase]
    setup_code: str
    teardown_code: str
    fixtures: List[str]
    mock_data: Dict[str, Any]
    coverage_target: float
    execution_time_estimate: int  # in seconds


@dataclass
class TestGenerationResult:
    """Result of test generation process"""
    success: bool
    test_suites: List[TestSuite]
    test_files: List[CodeFile]
    configuration_files: List[CodeFile]
    coverage_estimate: float
    total_test_cases: int
    errors: List[str]
    warnings: List[str]


class TestGenerator:
    """
    Advanced test generator that creates comprehensive test suites
    including unit tests, integration tests, and end-to-end tests
    """
    
    def __init__(self, ollama_client=None):
        self.ollama_client = ollama_client
        self.test_templates = self._initialize_test_templates()
        self.test_patterns = self._initialize_test_patterns()
        self.assertion_generators = self._initialize_assertion_generators()
        self.mock_generators = self._initialize_mock_generators()
        
    def _initialize_test_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize test templates for different frameworks"""
        return {
            'python': {
                'pytest_unit': '''"""
{description}
Unit tests for {module_name}
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import asyncio

{imports}


class Test{class_name}:
    """Unit test class for {class_name}"""
    
    @pytest.fixture
    def {fixture_name}(self):
        """Fixture for {class_name} instance"""
        {fixture_setup}
        return {fixture_return}
    
    {test_methods}
    
    {parametrized_tests}
    
    {async_tests}
    
    {mock_tests}
    
    {edge_case_tests}
''',
                'pytest_integration': '''"""
{description}
Integration tests for {module_name}
"""

import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

{imports}


@pytest.fixture(scope="session")
def test_db():
    """Create test database"""
    engine = create_engine("sqlite:///./test.db")
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client(test_db):
    """Create test client"""
    def override_get_db():
        try:
            yield test_db
        finally:
            test_db.close()
    
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class Test{class_name}Integration:
    """Integration test class for {class_name}"""
    
    {integration_test_methods}
    
    {api_endpoint_tests}
    
    {database_integration_tests}
    
    {external_service_tests}
''',
                'pytest_e2e': '''"""
{description}
End-to-end tests for {module_name}
"""

import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time

{imports}


@pytest.fixture(scope="session")
def browser():
    """Create browser instance for E2E tests"""
    options = Options()
    options.add_argument("--headless")  # Run in headless mode for CI
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(options=options)
    driver.implicitly_wait(10)
    yield driver
    driver.quit()


@pytest.fixture
def wait(browser):
    """WebDriverWait instance"""
    return WebDriverWait(browser, 10)


class Test{class_name}E2E:
    """End-to-end test class for {class_name}"""
    
    BASE_URL = "http://localhost:3000"  # Configure for your app
    
    {e2e_test_methods}
    
    {user_flow_tests}
    
    {cross_browser_tests}
''',
                'pytest_performance': '''"""
{description}
Performance tests for {module_name}
"""

import pytest
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import psutil
import memory_profiler

{imports}


class Test{class_name}Performance:
    """Performance test class for {class_name}"""
    
    @pytest.mark.performance
    def test_{method_name}_response_time(self, {fixture_name}):
        """Test response time for {method_name}"""
        # Performance threshold (in seconds)
        max_response_time = 0.1
        
        start_time = time.time()
        {performance_test_code}
        end_time = time.time()
        
        response_time = end_time - start_time
        assert response_time < max_response_time, f"Response time {{response_time:.3f}}s exceeds threshold {{max_response_time}}s"
    
    @pytest.mark.performance
    def test_{method_name}_memory_usage(self, {fixture_name}):
        """Test memory usage for {method_name}"""
        import tracemalloc
        
        tracemalloc.start()
        {memory_test_code}
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Memory threshold (in MB)
        max_memory_mb = 50
        current_mb = current / 1024 / 1024
        
        assert current_mb < max_memory_mb, f"Memory usage {{current_mb:.2f}}MB exceeds threshold {{max_memory_mb}}MB"
    
    @pytest.mark.performance
    def test_{method_name}_concurrent_requests(self, {fixture_name}):
        """Test concurrent request handling"""
        import threading
        
        results = []
        errors = []
        
        def make_request():
            try:
                {concurrent_test_code}
                results.append("success")
            except Exception as e:
                errors.append(str(e))
        
        threads = []
        num_threads = 10
        
        for _ in range(num_threads):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(errors) == 0, f"Errors in concurrent requests: {{errors}}"
        assert len(results) == num_threads, f"Expected {{num_threads}} successful requests, got {{len(results)}}"
''',
                'conftest': '''"""
Pytest configuration and shared fixtures
"""

import pytest
import asyncio
from unittest.mock import Mock
import os
from pathlib import Path

# Set test environment
os.environ["TESTING"] = "true"
os.environ["DATABASE_URL"] = "sqlite:///./test.db"

{test_imports}


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_db():
    """Mock database session"""
    mock_db = Mock()
    mock_db.query.return_value.filter.return_value.first.return_value = None
    mock_db.query.return_value.offset.return_value.limit.return_value.all.return_value = []
    mock_db.add = Mock()
    mock_db.commit = Mock()
    mock_db.refresh = Mock()
    mock_db.delete = Mock()
    return mock_db


@pytest.fixture
def sample_data():
    """Sample test data"""
    return {{
        {sample_data_content}
    }}


{custom_fixtures}


@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup after each test"""
    yield
    # Cleanup logic here
    pass
'''
            },
            'javascript': {
                'jest_unit': '''/**
 * {description}
 * Unit tests for {module_name}
 */

const {{ {imports} }} = require('../{module_path}');

describe('{module_name}', () => {{
    let {mock_variables};
    
    beforeEach(() => {{
        {setup_code}
    }});
    
    afterEach(() => {{
        {cleanup_code}
        jest.clearAllMocks();
    }});
    
    {test_methods}
    
    {edge_case_tests}
    
    {async_tests}
    
    {error_handling_tests}
}});
''',
                'jest_integration': '''/**
 * {description}
 * Integration tests for {module_name}
 */

const request = require('supertest');
const app = require('../app');

describe('{module_name} Integration', () => {{
    let server;
    
    beforeAll(async () => {{
        // Setup test database
        {integration_setup}
    }});
    
    afterAll(async () => {{
        // Cleanup test database
        {integration_cleanup}
        if (server) {{
            server.close();
        }}
    }});
    
    beforeEach(async () => {{
        {before_each_setup}
    }});
    
    {api_tests}
    
    {database_tests}
    
    {middleware_tests}
}});
''',
                'jest_e2e': '''/**
 * {description}
 * End-to-end tests for {module_name}
 */

const {{ Builder, By, Key, until }} = require('selenium-webdriver');
const chrome = require('selenium-webdriver/chrome');

describe('{module_name} E2E', () => {{
    let driver;
    const BASE_URL = 'http://localhost:3000';
    
    beforeAll(async () => {{
        const options = new chrome.Options();
        options.addArguments('--headless');
        options.addArguments('--no-sandbox');
        options.addArguments('--disable-dev-shm-usage');
        
        driver = await new Builder()
            .forBrowser('chrome')
            .setChromeOptions(options)
            .build();
    }}, 30000);
    
    afterAll(async () => {{
        if (driver) {{
            await driver.quit();
        }}
    }});
    
    {e2e_test_methods}
    
    {user_journey_tests}
    
    {cross_browser_tests}
}});
''',
                'jest_config': '''module.exports = {{
    testEnvironment: 'node',
    roots: ['<rootDir>/src', '<rootDir>/tests'],
    testMatch: [
        '**/__tests__/**/*.js',
        '**/?(*.)+(spec|test).js'
    ],
    collectCoverageFrom: [
        'src/**/*.js',
        '!src/**/*.test.js',
        '!src/index.js'
    ],
    coverageDirectory: 'coverage',
    coverageReporters: ['text', 'lcov', 'html'],
    coverageThreshold: {{
        global: {{
            branches: 80,
            functions: 80,
            lines: 80,
            statements: 80
        }}
    }},
    setupFilesAfterEnv: ['<rootDir>/tests/setup.js'],
    testTimeout: 10000,
    verbose: true,
    detectOpenHandles: true,
    forceExit: true
}};
'''
            }
        }
    
    def _initialize_test_patterns(self) -> Dict[str, List[str]]:
        """Initialize test generation patterns"""
        return {
            'unit_test_scenarios': [
                'valid_input_returns_expected_output',
                'invalid_input_raises_exception',
                'empty_input_handled_gracefully',
                'boundary_values_processed_correctly',
                'null_values_handled_properly',
                'duplicate_values_handled_correctly'
            ],
            'integration_scenarios': [
                'api_endpoint_returns_correct_response',
                'database_operations_persist_correctly',
                'external_service_integration_works',
                'authentication_flows_correctly',
                'error_responses_formatted_properly',
                'pagination_works_correctly'
            ],
            'e2e_scenarios': [
                'user_can_complete_primary_workflow',
                'error_messages_displayed_to_user',
                'navigation_works_across_pages',
                'forms_submit_and_validate_correctly',
                'responsive_design_works_on_mobile',
                'accessibility_requirements_met'
            ],
            'performance_scenarios': [
                'response_time_under_threshold',
                'memory_usage_within_limits',
                'concurrent_requests_handled',
                'large_dataset_processing',
                'cache_performance_optimized'
            ],
            'security_scenarios': [
                'sql_injection_prevented',
                'xss_attacks_blocked',
                'authentication_required_for_protected_routes',
                'rate_limiting_enforced',
                'input_validation_prevents_malicious_data'
            ]
        }
    
    def _initialize_assertion_generators(self) -> Dict[str, Any]:
        """Initialize assertion generation patterns"""
        return {
            'python': {
                'equality': 'assert {actual} == {expected}',
                'inequality': 'assert {actual} != {expected}',
                'greater_than': 'assert {actual} > {expected}',
                'less_than': 'assert {actual} < {expected}',
                'contains': 'assert {expected} in {actual}',
                'not_contains': 'assert {expected} not in {actual}',
                'is_none': 'assert {actual} is None',
                'is_not_none': 'assert {actual} is not None',
                'isinstance': 'assert isinstance({actual}, {expected})',
                'raises': 'with pytest.raises({exception}):\n        {code}',
                'length': 'assert len({actual}) == {expected}',
                'startswith': 'assert {actual}.startswith({expected})',
                'endswith': 'assert {actual}.endswith({expected})',
                'regex_match': 'assert re.match({pattern}, {actual})',
                'dict_contains': 'assert {key} in {actual}',
                'list_empty': 'assert len({actual}) == 0',
                'status_code': 'assert response.status_code == {expected}'
            },
            'javascript': {
                'equality': 'expect({actual}).toBe({expected});',
                'inequality': 'expect({actual}).not.toBe({expected});',
                'greater_than': 'expect({actual}).toBeGreaterThan({expected});',
                'less_than': 'expect({actual}).toBeLessThan({expected});',
                'contains': 'expect({actual}).toContain({expected});',
                'not_contains': 'expect({actual}).not.toContain({expected});',
                'is_null': 'expect({actual}).toBeNull();',
                'is_not_null': 'expect({actual}).not.toBeNull();',
                'is_undefined': 'expect({actual}).toBeUndefined();',
                'is_defined': 'expect({actual}).toBeDefined();',
                'instanceof': 'expect({actual}).toBeInstanceOf({expected});',
                'throws': 'expect(() => {{ {code} }}).toThrow({expected});',
                'async_throws': 'await expect({code}).rejects.toThrow({expected});',
                'length': 'expect({actual}).toHaveLength({expected});',
                'property': 'expect({actual}).toHaveProperty({property});',
                'array_containing': 'expect({actual}).toEqual(expect.arrayContaining({expected}));',
                'object_containing': 'expect({actual}).toEqual(expect.objectContaining({expected}));',
                'status_code': 'expect(response.status).toBe({expected});'
            }
        }
    
    def _initialize_mock_generators(self) -> Dict[str, Any]:
        """Initialize mock generation patterns"""
        return {
            'python': {
                'simple_mock': 'mock_{name} = Mock()',
                'return_value_mock': 'mock_{name} = Mock(return_value={return_value})',
                'side_effect_mock': 'mock_{name} = Mock(side_effect={side_effect})',
                'async_mock': 'mock_{name} = AsyncMock(return_value={return_value})',
                'patch_decorator': '@patch("{target}")',
                'patch_context': 'with patch("{target}") as mock_{name}:',
                'mock_method': 'mock_{object}.{method} = Mock(return_value={return_value})',
                'mock_property': 'mock_{object}.{property} = {value}',
                'database_mock': '''mock_db = Mock()
mock_db.query.return_value.filter.return_value.first.return_value = {mock_data}
mock_db.query.return_value.offset.return_value.limit.return_value.all.return_value = [{mock_list}]'''
            },
            'javascript': {
                'simple_mock': 'const mock{name} = jest.fn();',
                'return_value_mock': 'const mock{name} = jest.fn().mockReturnValue({return_value});',
                'resolved_value_mock': 'const mock{name} = jest.fn().mockResolvedValue({return_value});',
                'rejected_value_mock': 'const mock{name} = jest.fn().mockRejectedValue({error});',
                'implementation_mock': 'const mock{name} = jest.fn().mockImplementation({implementation});',
                'mock_module': 'jest.mock("{module_path}", () => ({{\n    {mocked_exports}\n}});',
                'spy_on': 'const spy{name} = jest.spyOn({object}, "{method}");',
                'mock_axios': '''jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;'''
            }
        }
    
    async def generate_comprehensive_tests(
        self,
        plan: ImplementationPlan,
        features: List[FeatureRequirement],
        generated_code: List[CodeFile],
        output_path: Optional[Path] = None
    ) -> TestGenerationResult:
        """
        Generate comprehensive test suite for the implementation
        """
        logger.info(f"Generating comprehensive tests for {plan.project_name}")
        
        try:
            test_suites = []
            test_files = []
            config_files = []
            errors = []
            warnings = []
            
            # Generate unit tests
            unit_test_suites = await self._generate_unit_tests(
                features, generated_code, plan.tech_context
            )
            test_suites.extend(unit_test_suites)
            
            # Generate integration tests
            integration_test_suites = await self._generate_integration_tests(
                features, generated_code, plan.tech_context
            )
            test_suites.extend(integration_test_suites)
            
            # Generate E2E tests if it's a web application
            if plan.tech_context.project_type == 'web':
                e2e_test_suites = await self._generate_e2e_tests(
                    features, plan.tech_context
                )
                test_suites.extend(e2e_test_suites)
            
            # Generate performance tests
            performance_test_suites = await self._generate_performance_tests(
                features, generated_code, plan.tech_context
            )
            test_suites.extend(performance_test_suites)
            
            # Generate security tests
            security_test_suites = await self._generate_security_tests(
                features, plan.tech_context
            )
            test_suites.extend(security_test_suites)
            
            # Convert test suites to code files
            for suite in test_suites:
                suite_files = await self._convert_test_suite_to_files(
                    suite, plan.tech_context
                )
                test_files.extend(suite_files)
            
            # Generate test configuration files
            config_files = await self._generate_test_configuration(
                plan.tech_context, test_suites
            )
            
            # Calculate coverage estimate
            coverage_estimate = self._calculate_coverage_estimate(
                test_suites, generated_code
            )
            
            # Enhance with AI if available
            if self.ollama_client:
                enhanced_tests = await self._enhance_tests_with_ai(
                    test_suites, features, plan.tech_context
                )
                if enhanced_tests:
                    test_files.extend(enhanced_tests)
            
            total_test_cases = sum(len(suite.test_cases) for suite in test_suites)
            
            logger.info(f"Generated {total_test_cases} test cases across {len(test_suites)} test suites")
            
            return TestGenerationResult(
                success=True,
                test_suites=test_suites,
                test_files=test_files,
                configuration_files=config_files,
                coverage_estimate=coverage_estimate,
                total_test_cases=total_test_cases,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Test generation failed: {e}")
            return TestGenerationResult(
                success=False,
                test_suites=[],
                test_files=[],
                configuration_files=[],
                coverage_estimate=0.0,
                total_test_cases=0,
                errors=[str(e)],
                warnings=[]
            )
    
    async def _generate_unit_tests(
        self,
        features: List[FeatureRequirement],
        generated_code: List[CodeFile],
        tech_context: TechnicalContext
    ) -> List[TestSuite]:
        """Generate unit tests for individual components"""
        
        test_suites = []
        
        # Group code files by module/component
        code_modules = self._group_code_by_module(generated_code)
        
        for module_name, code_files in code_modules.items():
            if not code_files:
                continue
            
            # Extract functions/classes to test
            testable_units = self._extract_testable_units(code_files)
            
            if not testable_units:
                continue
            
            test_cases = []
            
            for unit in testable_units:
                unit_tests = await self._generate_unit_test_cases(
                    unit, tech_context, features
                )
                test_cases.extend(unit_tests)
            
            if test_cases:
                test_suite = TestSuite(
                    name=f"{module_name}_unit_tests",
                    description=f"Unit tests for {module_name} module",
                    test_cases=test_cases,
                    setup_code=self._generate_unit_test_setup(module_name, tech_context),
                    teardown_code=self._generate_unit_test_teardown(tech_context),
                    fixtures=self._generate_unit_test_fixtures(testable_units, tech_context),
                    mock_data=self._generate_mock_data(testable_units),
                    coverage_target=0.85,
                    execution_time_estimate=len(test_cases) * 2  # 2 seconds per test
                )
                test_suites.append(test_suite)
        
        return test_suites
    
    async def _generate_integration_tests(
        self,
        features: List[FeatureRequirement],
        generated_code: List[CodeFile],
        tech_context: TechnicalContext
    ) -> List[TestSuite]:
        """Generate integration tests for component interactions"""
        
        test_suites = []
        
        # Find API endpoints
        api_endpoints = self._extract_api_endpoints(generated_code, features)
        
        if api_endpoints:
            api_test_cases = []
            
            for endpoint in api_endpoints:
                endpoint_tests = await self._generate_api_test_cases(
                    endpoint, tech_context
                )
                api_test_cases.extend(endpoint_tests)
            
            if api_test_cases:
                test_suite = TestSuite(
                    name="api_integration_tests",
                    description="Integration tests for API endpoints",
                    test_cases=api_test_cases,
                    setup_code=self._generate_integration_test_setup(tech_context),
                    teardown_code=self._generate_integration_test_teardown(tech_context),
                    fixtures=self._generate_integration_fixtures(tech_context),
                    mock_data=self._generate_api_mock_data(api_endpoints),
                    coverage_target=0.90,
                    execution_time_estimate=len(api_test_cases) * 5
                )
                test_suites.append(test_suite)
        
        # Database integration tests
        database_features = [f for f in features if f.category == 'database']
        if database_features:
            db_test_cases = await self._generate_database_test_cases(
                database_features, tech_context
            )
            
            if db_test_cases:
                test_suite = TestSuite(
                    name="database_integration_tests",
                    description="Integration tests for database operations",
                    test_cases=db_test_cases,
                    setup_code=self._generate_db_test_setup(tech_context),
                    teardown_code=self._generate_db_test_teardown(tech_context),
                    fixtures=self._generate_db_fixtures(database_features, tech_context),
                    mock_data={},
                    coverage_target=0.95,
                    execution_time_estimate=len(db_test_cases) * 3
                )
                test_suites.append(test_suite)
        
        return test_suites
    
    async def _generate_e2e_tests(
        self,
        features: List[FeatureRequirement],
        tech_context: TechnicalContext
    ) -> List[TestSuite]:
        """Generate end-to-end tests for user workflows"""
        
        test_suites = []
        
        # Identify user workflows from UI features
        ui_features = [f for f in features if f.category == 'ui']
        workflows = self._identify_user_workflows(ui_features, features)
        
        if workflows:
            e2e_test_cases = []
            
            for workflow in workflows:
                workflow_tests = await self._generate_workflow_test_cases(
                    workflow, tech_context
                )
                e2e_test_cases.extend(workflow_tests)
            
            if e2e_test_cases:
                test_suite = TestSuite(
                    name="e2e_user_workflow_tests",
                    description="End-to-end tests for user workflows",
                    test_cases=e2e_test_cases,
                    setup_code=self._generate_e2e_setup(tech_context),
                    teardown_code=self._generate_e2e_teardown(tech_context),
                    fixtures=self._generate_e2e_fixtures(tech_context),
                    mock_data=self._generate_e2e_mock_data(workflows),
                    coverage_target=0.80,
                    execution_time_estimate=len(e2e_test_cases) * 30  # E2E tests are slower
                )
                test_suites.append(test_suite)
        
        return test_suites
    
    async def _generate_performance_tests(
        self,
        features: List[FeatureRequirement],
        generated_code: List[CodeFile],
        tech_context: TechnicalContext
    ) -> List[TestSuite]:
        """Generate performance tests"""
        
        test_cases = []
        
        # Identify performance-critical components
        critical_components = self._identify_performance_critical_components(
            generated_code, features
        )
        
        for component in critical_components:
            perf_tests = await self._generate_performance_test_cases(
                component, tech_context
            )
            test_cases.extend(perf_tests)
        
        if test_cases:
            test_suite = TestSuite(
                name="performance_tests",
                description="Performance and load tests",
                test_cases=test_cases,
                setup_code=self._generate_performance_test_setup(tech_context),
                teardown_code=self._generate_performance_test_teardown(tech_context),
                fixtures=self._generate_performance_fixtures(tech_context),
                mock_data={},
                coverage_target=0.70,
                execution_time_estimate=len(test_cases) * 10
            )
            return [test_suite]
        
        return []
    
    async def _generate_security_tests(
        self,
        features: List[FeatureRequirement],
        tech_context: TechnicalContext
    ) -> List[TestSuite]:
        """Generate security tests"""
        
        test_cases = []
        
        # Authentication tests
        auth_features = [f for f in features if f.category == 'authentication']
        if auth_features:
            auth_tests = await self._generate_auth_security_tests(
                auth_features, tech_context
            )
            test_cases.extend(auth_tests)
        
        # Input validation tests
        api_features = [f for f in features if f.category == 'api']
        if api_features:
            input_tests = await self._generate_input_validation_tests(
                api_features, tech_context
            )
            test_cases.extend(input_tests)
        
        # General security tests
        general_tests = await self._generate_general_security_tests(tech_context)
        test_cases.extend(general_tests)
        
        if test_cases:
            test_suite = TestSuite(
                name="security_tests",
                description="Security and vulnerability tests",
                test_cases=test_cases,
                setup_code=self._generate_security_test_setup(tech_context),
                teardown_code=self._generate_security_test_teardown(tech_context),
                fixtures=self._generate_security_fixtures(tech_context),
                mock_data={},
                coverage_target=0.90,
                execution_time_estimate=len(test_cases) * 8
            )
            return [test_suite]
        
        return []
    
    def _group_code_by_module(self, code_files: List[CodeFile]) -> Dict[str, List[CodeFile]]:
        """Group code files by module/component"""
        
        modules = {}
        
        for code_file in code_files:
            if code_file.file_type not in ['implementation', 'test']:
                continue
            
            # Extract module name from path
            path_parts = Path(code_file.path).parts
            if len(path_parts) > 1:
                module_name = path_parts[-2]  # Parent directory name
            else:
                module_name = Path(code_file.path).stem
            
            if module_name not in modules:
                modules[module_name] = []
            
            modules[module_name].append(code_file)
        
        return modules
    
    def _extract_testable_units(self, code_files: List[CodeFile]) -> List[Dict[str, Any]]:
        """Extract testable units (functions, classes, methods) from code"""
        
        testable_units = []
        
        for code_file in code_files:
            if code_file.language == 'python':
                units = self._extract_python_testable_units(code_file)
            elif code_file.language == 'javascript':
                units = self._extract_javascript_testable_units(code_file)
            else:
                continue
            
            testable_units.extend(units)
        
        return testable_units
    
    def _extract_python_testable_units(self, code_file: CodeFile) -> List[Dict[str, Any]]:
        """Extract testable units from Python code"""
        
        units = []
        
        try:
            tree = ast.parse(code_file.content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Skip private functions and test functions
                    if not node.name.startswith('_') and not node.name.startswith('test_'):
                        units.append({
                            'type': 'function',
                            'name': node.name,
                            'args': [arg.arg for arg in node.args.args],
                            'file_path': code_file.path,
                            'line_number': node.lineno,
                            'is_async': isinstance(node, ast.AsyncFunctionDef),
                            'docstring': ast.get_docstring(node)
                        })
                
                elif isinstance(node, ast.ClassDef):
                    methods = []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and not item.name.startswith('_'):
                            methods.append({
                                'name': item.name,
                                'args': [arg.arg for arg in item.args.args],
                                'is_async': isinstance(item, ast.AsyncFunctionDef)
                            })
                    
                    units.append({
                        'type': 'class',
                        'name': node.name,
                        'methods': methods,
                        'file_path': code_file.path,
                        'line_number': node.lineno,
                        'docstring': ast.get_docstring(node)
                    })
        
        except SyntaxError as e:
            logger.warning(f"Could not parse Python file {code_file.path}: {e}")
        
        return units
    
    def _extract_javascript_testable_units(self, code_file: CodeFile) -> List[Dict[str, Any]]:
        """Extract testable units from JavaScript code"""
        
        units = []
        
        # Simple regex-based extraction for JavaScript
        # In a production system, you'd use a proper JS parser like Acorn
        
        # Function declarations
        function_pattern = r'(?:async\s+)?function\s+(\w+)\s*\([^)]*\)'
        for match in re.finditer(function_pattern, code_file.content):
            units.append({
                'type': 'function',
                'name': match.group(1),
                'file_path': code_file.path,
                'is_async': 'async' in match.group(0)
            })
        
        # Arrow functions assigned to variables
        arrow_pattern = r'const\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>'
        for match in re.finditer(arrow_pattern, code_file.content):
            units.append({
                'type': 'function',
                'name': match.group(1),
                'file_path': code_file.path,
                'is_async': 'async' in match.group(0)
            })
        
        # Class definitions
        class_pattern = r'class\s+(\w+)(?:\s+extends\s+\w+)?'
        for match in re.finditer(class_pattern, code_file.content):
            units.append({
                'type': 'class',
                'name': match.group(1),
                'file_path': code_file.path,
                'methods': []  # Would need more complex parsing for methods
            })
        
        return units
    
    async def _generate_unit_test_cases(
        self,
        unit: Dict[str, Any],
        tech_context: TechnicalContext,
        features: List[FeatureRequirement]
    ) -> List[TestCase]:
        """Generate test cases for a specific unit"""
        
        test_cases = []
        
        unit_name = unit['name']
        unit_type = unit['type']
        
        # Generate different types of test cases
        scenarios = self.test_patterns['unit_test_scenarios']
        
        for i, scenario in enumerate(scenarios):
            test_case = TestCase(
                name=f"test_{unit_name.lower()}_{scenario}",
                description=f"Test {unit_name} with {scenario.replace('_', ' ')}",
                test_type='unit',
                category=self._categorize_test_scenario(scenario),
                setup=self._generate_test_setup_code(unit, tech_context),
                test_code=self._generate_unit_test_code(unit, scenario, tech_context),
                assertions=self._generate_test_assertions(unit, scenario, tech_context),
                dependencies=unit.get('dependencies', []),
                mock_requirements=self._determine_mock_requirements(unit, scenario),
                expected_coverage=0.15 + (i * 0.05),  # Distribute coverage
                priority=self._calculate_test_priority(scenario, unit)
            )
            test_cases.append(test_case)
        
        return test_cases
    
    def _generate_unit_test_code(
        self,
        unit: Dict[str, Any],
        scenario: str,
        tech_context: TechnicalContext
    ) -> str:
        """Generate test code for a unit test scenario"""
        
        unit_name = unit['name']
        unit_type = unit['type']
        
        if 'python' in tech_context.tech_stack:
            if scenario == 'valid_input_returns_expected_output':
                if unit_type == 'function':
                    return f'''# Arrange
        input_data = {self._generate_sample_input(unit)}
        expected = {self._generate_expected_output(unit)}
        
        # Act
        result = {unit_name}(input_data)
        
        # Assert
        assert result == expected'''
                else:  # class
                    return f'''# Arrange
        instance = {unit_name}()
        input_data = {self._generate_sample_input(unit)}
        expected = {self._generate_expected_output(unit)}
        
        # Act
        result = instance.process(input_data)
        
        # Assert
        assert result == expected'''
            
            elif scenario == 'invalid_input_raises_exception':
                return f'''# Arrange
        invalid_input = {self._generate_invalid_input(unit)}
        
        # Act & Assert
        with pytest.raises(ValueError):
            {unit_name}(invalid_input)'''
            
            elif scenario == 'empty_input_handled_gracefully':
                return f'''# Arrange
        empty_input = {self._generate_empty_input(unit)}
        
        # Act
        result = {unit_name}(empty_input)
        
        # Assert
        assert result is not None
        assert len(result) == 0 or result == {self._generate_empty_output(unit)}'''
        
        elif 'javascript' in tech_context.tech_stack:
            if scenario == 'valid_input_returns_expected_output':
                return f'''// Arrange
        const inputData = {self._generate_sample_input_js(unit)};
        const expected = {self._generate_expected_output_js(unit)};
        
        // Act
        const result = {unit_name}(inputData);
        
        // Assert
        expect(result).toEqual(expected);'''
            
            elif scenario == 'invalid_input_raises_exception':
                return f'''// Arrange
        const invalidInput = {self._generate_invalid_input_js(unit)};
        
        // Act & Assert
        expect(() => {{
            {unit_name}(invalidInput);
        }}).toThrow();'''
        
        return f"// TODO: Implement test for {scenario}"
    
    def _generate_test_assertions(
        self,
        unit: Dict[str, Any],
        scenario: str,
        tech_context: TechnicalContext
    ) -> List[str]:
        """Generate appropriate assertions for test scenario"""
        
        assertions = []
        
        if 'python' in tech_context.tech_stack:
            assertion_templates = self.assertion_generators['python']
        else:
            assertion_templates = self.assertion_generators['javascript']
        
        # Generate scenario-specific assertions
        if scenario == 'valid_input_returns_expected_output':
            assertions.append('result equals expected value')
            assertions.append('result type is correct')
        
        elif scenario == 'invalid_input_raises_exception':
            assertions.append('exception is raised')
            assertions.append('exception type is correct')
        
        elif scenario == 'empty_input_handled_gracefully':
            assertions.append('no exception is raised')
            assertions.append('result handles empty input appropriately')
        
        elif scenario == 'boundary_values_processed_correctly':
            assertions.append('minimum boundary value processed')
            assertions.append('maximum boundary value processed')
        
        return assertions
    
    def _generate_sample_input(self, unit: Dict[str, Any]) -> str:
        """Generate sample input for Python test"""
        
        unit_name = unit['name'].lower()
        
        # Generate based on function name patterns
        if 'user' in unit_name:
            return '{"name": "John Doe", "email": "john@example.com"}'
        elif 'product' in unit_name:
            return '{"name": "Test Product", "price": 999}'
        elif 'calculate' in unit_name:
            return '{"a": 10, "b": 5}'
        elif 'validate' in unit_name:
            return '"valid_input"'
        else:
            return '{"test": "data"}'
    
    def _generate_expected_output(self, unit: Dict[str, Any]) -> str:
        """Generate expected output for Python test"""
        
        unit_name = unit['name'].lower()
        
        if 'create' in unit_name or 'add' in unit_name:
            return 'True'
        elif 'get' in unit_name or 'find' in unit_name:
            return '{"id": 1, "name": "Test"}'
        elif 'calculate' in unit_name:
            return '15'  # Assuming addition
        elif 'validate' in unit_name:
            return 'True'
        else:
            return '{"success": True}'
    
    def _generate_invalid_input(self, unit: Dict[str, Any]) -> str:
        """Generate invalid input for Python test"""
        
        return 'None'  # Simple invalid input
    
    def _generate_empty_input(self, unit: Dict[str, Any]) -> str:
        """Generate empty input for Python test"""
        
        unit_name = unit['name'].lower()
        
        if 'list' in unit_name or 'array' in unit_name:
            return '[]'
        elif 'dict' in unit_name or 'object' in unit_name:
            return '{}'
        else:
            return '""'
    
    def _generate_empty_output(self, unit: Dict[str, Any]) -> str:
        """Generate expected empty output for Python test"""
        
        return '[]'  # Default empty output
    
    # JavaScript equivalents
    def _generate_sample_input_js(self, unit: Dict[str, Any]) -> str:
        """Generate sample input for JavaScript test"""
        
        unit_name = unit['name'].lower()
        
        if 'user' in unit_name:
            return '{ name: "John Doe", email: "john@example.com" }'
        elif 'product' in unit_name:
            return '{ name: "Test Product", price: 999 }'
        elif 'calculate' in unit_name:
            return '{ a: 10, b: 5 }'
        else:
            return '{ test: "data" }'
    
    def _generate_expected_output_js(self, unit: Dict[str, Any]) -> str:
        """Generate expected output for JavaScript test"""
        
        unit_name = unit['name'].lower()
        
        if 'create' in unit_name:
            return 'true'
        elif 'get' in unit_name:
            return '{ id: 1, name: "Test" }'
        elif 'calculate' in unit_name:
            return '15'
        else:
            return '{ success: true }'
    
    def _generate_invalid_input_js(self, unit: Dict[str, Any]) -> str:
        """Generate invalid input for JavaScript test"""
        
        return 'null'
    
    def _categorize_test_scenario(self, scenario: str) -> str:
        """Categorize test scenario"""
        
        if 'valid' in scenario or 'expected' in scenario:
            return 'positive'
        elif 'invalid' in scenario or 'raises' in scenario:
            return 'negative'
        elif 'empty' in scenario or 'boundary' in scenario:
            return 'edge_case'
        else:
            return 'boundary'
    
    def _determine_mock_requirements(self, unit: Dict[str, Any], scenario: str) -> List[str]:
        """Determine what needs to be mocked for this test"""
        
        mocks = []
        
        # Check for external dependencies
        if 'database' in unit.get('dependencies', []):
            mocks.append('database_session')
        
        if 'api' in unit.get('dependencies', []):
            mocks.append('external_api_client')
        
        if 'file' in unit.get('dependencies', []):
            mocks.append('file_system')
        
        return mocks
    
    def _calculate_test_priority(self, scenario: str, unit: Dict[str, Any]) -> int:
        """Calculate test priority (1-5, lower is higher priority)"""
        
        # Critical scenarios get highest priority
        critical_scenarios = ['valid_input_returns_expected_output', 'invalid_input_raises_exception']
        if scenario in critical_scenarios:
            return 1
        
        # Edge cases are medium priority
        edge_scenarios = ['empty_input_handled_gracefully', 'boundary_values_processed_correctly']
        if scenario in edge_scenarios:
            return 2
        
        # Everything else is lower priority
        return 3
    
    def _extract_api_endpoints(
        self, 
        generated_code: List[CodeFile], 
        features: List[FeatureRequirement]
    ) -> List[Dict[str, Any]]:
        """Extract API endpoints from generated code"""
        
        endpoints = []
        
        for code_file in generated_code:
            if 'route' in code_file.path.lower() or 'api' in code_file.path.lower():
                file_endpoints = self._parse_endpoints_from_code(code_file)
                endpoints.extend(file_endpoints)
        
        return endpoints
    
    def _parse_endpoints_from_code(self, code_file: CodeFile) -> List[Dict[str, Any]]:
        """Parse API endpoints from code file"""
        
        endpoints = []
        
        if code_file.language == 'python':
            # Look for FastAPI route decorators
            route_pattern = r'@router\.(get|post|put|delete|patch)\("([^"]+)"'
            for match in re.finditer(route_pattern, code_file.content):
                method = match.group(1).upper()
                path = match.group(2)
                
                endpoints.append({
                    'method': method,
                    'path': path,
                    'file': code_file.path,
                    'framework': 'fastapi'
                })
        
        elif code_file.language == 'javascript':
            # Look for Express route definitions
            route_pattern = r'router\.(get|post|put|delete|patch)\([\'"]([^\'"]+)[\'"]'
            for match in re.finditer(route_pattern, code_file.content):
                method = match.group(1).upper()
                path = match.group(2)
                
                endpoints.append({
                    'method': method,
                    'path': path,
                    'file': code_file.path,
                    'framework': 'express'
                })
        
        return endpoints
    
    async def _generate_api_test_cases(
        self,
        endpoint: Dict[str, Any],
        tech_context: TechnicalContext
    ) -> List[TestCase]:
        """Generate test cases for an API endpoint"""
        
        test_cases = []
        method = endpoint['method']
        path = endpoint['path']
        
        # Generate different test scenarios for the endpoint
        scenarios = [
            'successful_request_returns_200',
            'invalid_input_returns_400',
            'unauthorized_request_returns_401',
            'not_found_returns_404',
            'server_error_returns_500'
        ]
        
        for scenario in scenarios:
            test_case = TestCase(
                name=f"test_{method.lower()}_{path.replace('/', '_').replace('{', '').replace('}', '')}_{scenario}",
                description=f"Test {method} {path} - {scenario.replace('_', ' ')}",
                test_type='integration',
                category='api',
                setup=self._generate_api_test_setup(endpoint, tech_context),
                test_code=self._generate_api_test_code(endpoint, scenario, tech_context),
                assertions=self._generate_api_assertions(scenario),
                dependencies=[],
                mock_requirements=self._determine_api_mock_requirements(endpoint, scenario),
                expected_coverage=0.20,
                priority=1 if 'successful' in scenario else 2
            )
            test_cases.append(test_case)
        
        return test_cases
    
    def _generate_api_test_code(
        self,
        endpoint: Dict[str, Any],
        scenario: str,
        tech_context: TechnicalContext
    ) -> str:
        """Generate API test code"""
        
        method = endpoint['method']
        path = endpoint['path']
        
        if 'python' in tech_context.tech_stack:
            if scenario == 'successful_request_returns_200':
                if method == 'GET':
                    return f'''response = client.get("{path}")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data or isinstance(data, list)'''
                
                elif method == 'POST':
                    return f'''test_data = {self._generate_test_payload(path)}
        response = client.post("{path}", json=test_data)
        assert response.status_code == 201
        data = response.json()
        assert "id" in data'''
            
            elif scenario == 'invalid_input_returns_400':
                return f'''invalid_data = {self._generate_invalid_payload()}
        response = client.post("{path}", json=invalid_data)
        assert response.status_code == 400
        error = response.json()
        assert "error" in error or "detail" in error'''
        
        elif 'javascript' in tech_context.tech_stack:
            if scenario == 'successful_request_returns_200':
                if method == 'GET':
                    return f'''const response = await request(app).get("{path}");
        expect(response.status).toBe(200);
        expect(response.body).toBeDefined();'''
                
                elif method == 'POST':
                    return f'''const testData = {self._generate_test_payload_js(path)};
        const response = await request(app)
            .post("{path}")
            .send(testData);
        expect(response.status).toBe(201);
        expect(response.body.id).toBeDefined();'''
        
        return f"// TODO: Implement {scenario} test for {method} {path}"
    
    def _generate_test_payload(self, path: str) -> str:
        """Generate test payload for Python API test"""
        
        if 'user' in path:
            return '{"name": "Test User", "email": "test@example.com"}'
        elif 'product' in path:
            return '{"name": "Test Product", "price": 999, "description": "Test description"}'
        else:
            return '{"title": "Test Item", "description": "Test description"}'
    
    def _generate_invalid_payload(self) -> str:
        """Generate invalid payload for Python API test"""
        
        return '{"invalid": "missing required fields"}'
    
    def _generate_test_payload_js(self, path: str) -> str:
        """Generate test payload for JavaScript API test"""
        
        if 'user' in path:
            return '{ name: "Test User", email: "test@example.com" }'
        elif 'product' in path:
            return '{ name: "Test Product", price: 999, description: "Test description" }'
        else:
            return '{ title: "Test Item", description: "Test description" }'
    
    def _generate_api_assertions(self, scenario: str) -> List[str]:
        """Generate API-specific assertions"""
        
        assertions = []
        
        if scenario == 'successful_request_returns_200':
            assertions.extend([
                'status code is 200/201',
                'response body is valid JSON',
                'response contains expected fields'
            ])
        
        elif scenario == 'invalid_input_returns_400':
            assertions.extend([
                'status code is 400',
                'error message is present',
                'validation errors are detailed'
            ])
        
        elif scenario == 'unauthorized_request_returns_401':
            assertions.extend([
                'status code is 401',
                'authentication required message'
            ])
        
        return assertions
    
    def _determine_api_mock_requirements(
        self, 
        endpoint: Dict[str, Any], 
        scenario: str
    ) -> List[str]:
        """Determine what needs to be mocked for API tests"""
        
        mocks = []
        
        if 'database' in scenario or any(word in endpoint['path'] for word in ['user', 'product', 'item']):
            mocks.append('database_session')
        
        if 'external' in scenario:
            mocks.append('external_api_client')
        
        if 'auth' in endpoint['path'] or 'login' in endpoint['path']:
            mocks.append('auth_service')
        
        return mocks
    
    async def _convert_test_suite_to_files(
        self,
        test_suite: TestSuite,
        tech_context: TechnicalContext
    ) -> List[CodeFile]:
        """Convert test suite to code files"""
        
        files = []
        
        if 'python' in tech_context.tech_stack:
            files.extend(await self._generate_python_test_files(test_suite, tech_context))
        
        elif 'javascript' in tech_context.tech_stack:
            files.extend(await self._generate_javascript_test_files(test_suite, tech_context))
        
        return files
    
    async def _generate_python_test_files(
        self,
        test_suite: TestSuite,
        tech_context: TechnicalContext
    ) -> List[CodeFile]:
        """Generate Python test files"""
        
        files = []
        
        # Determine template based on test type
        primary_test_type = max(set(tc.test_type for tc in test_suite.test_cases), 
                              key=[tc.test_type for tc in test_suite.test_cases].count)
        
        if primary_test_type == 'unit':
            template_key = 'pytest_unit'
        elif primary_test_type == 'integration':
            template_key = 'pytest_integration'
        elif primary_test_type == 'e2e':
            template_key = 'pytest_e2e'
        elif primary_test_type == 'performance':
            template_key = 'pytest_performance'
        else:
            template_key = 'pytest_unit'
        
        template = self.test_templates['python'][template_key]
        
        # Generate test methods
        test_methods = []
        for test_case in test_suite.test_cases:
            method_code = f'''def {test_case.name}(self, {self._get_fixture_params(test_case, tech_context)}):
        """{test_case.description}"""
        {test_case.test_code}'''
            test_methods.append(method_code)
        
        # Fill template
        content = template.format(
            description=test_suite.description,
            module_name=test_suite.name,
            class_name=test_suite.name.replace('_', '').title(),
            fixture_name='test_fixture',
            fixture_setup=test_suite.setup_code,
            fixture_return='mock_instance',
            imports=self._generate_python_test_imports(test_suite, tech_context),
            test_methods='\n    \n    '.join(test_methods),
            parametrized_tests=self._generate_parametrized_tests(test_suite),
            async_tests=self._generate_async_tests(test_suite),
            mock_tests=self._generate_mock_tests(test_suite),
            edge_case_tests=self._generate_edge_case_tests(test_suite),
            integration_test_methods='\n    \n    '.join(test_methods) if primary_test_type == 'integration' else '',
            api_endpoint_tests=self._generate_api_endpoint_tests(test_suite),
            database_integration_tests=self._generate_db_integration_tests(test_suite),
            external_service_tests=self._generate_external_service_tests(test_suite),
            e2e_test_methods='\n    \n    '.join(test_methods) if primary_test_type == 'e2e' else '',
            user_flow_tests=self._generate_user_flow_tests(test_suite),
            cross_browser_tests=self._generate_cross_browser_tests(test_suite),
            method_name=test_suite.name.split('_')[0],
            performance_test_code='# Performance test code here',
            memory_test_code='# Memory test code here',
            concurrent_test_code='# Concurrent test code here'
        )
        
        files.append(CodeFile(
            path=f"tests/{test_suite.name}.py",
            content=content,
            language="python",
            file_type="test",
            dependencies=["pytest"],
            imports=["pytest"],
            exports=[],
            metadata={"test_suite": test_suite.name, "test_count": len(test_suite.test_cases)}
        ))
        
        return files
    
    async def _generate_test_configuration(
        self,
        tech_context: TechnicalContext,
        test_suites: List[TestSuite]
    ) -> List[CodeFile]:
        """Generate test configuration files"""
        
        files = []
        
        if 'python' in tech_context.tech_stack:
            # Generate pytest.ini
            pytest_config = f'''[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --cov=src
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80
    --tb=short
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    performance: Performance tests
    security: Security tests
    slow: Slow running tests
'''
            
            files.append(CodeFile(
                path="pytest.ini",
                content=pytest_config,
                language="ini",
                file_type="config",
                dependencies=[],
                imports=[],
                exports=[],
                metadata={"type": "test_config"}
            ))
            
            # Generate conftest.py
            conftest_content = self.test_templates['python']['conftest'].format(
                test_imports=self._generate_conftest_imports(tech_context),
                sample_data_content=self._generate_sample_data_content(),
                custom_fixtures=self._generate_custom_fixtures(test_suites, tech_context)
            )
            
            files.append(CodeFile(
                path="tests/conftest.py",
                content=conftest_content,
                language="python",
                file_type="config",
                dependencies=["pytest"],
                imports=["pytest"],
                exports=[],
                metadata={"type": "test_fixtures"}
            ))
        
        elif 'javascript' in tech_context.tech_stack:
            # Generate Jest configuration
            jest_config = self.test_templates['javascript']['jest_config']
            
            files.append(CodeFile(
                path="jest.config.js",
                content=jest_config,
                language="javascript",
                file_type="config",
                dependencies=["jest"],
                imports=[],
                exports=[],
                metadata={"type": "test_config"}
            ))
        
        return files
    
    def _calculate_coverage_estimate(
        self, 
        test_suites: List[TestSuite], 
        generated_code: List[CodeFile]
    ) -> float:
        """Calculate estimated test coverage"""
        
        if not test_suites or not generated_code:
            return 0.0
        
        # Simple heuristic: average of test suite coverage targets
        # weighted by number of test cases
        
        total_weighted_coverage = 0.0
        total_test_cases = 0
        
        for suite in test_suites:
            weight = len(suite.test_cases)
            total_weighted_coverage += suite.coverage_target * weight
            total_test_cases += weight
        
        if total_test_cases == 0:
            return 0.0
        
        return total_weighted_coverage / total_test_cases
    
    # Helper methods for generating specific test content
    def _get_fixture_params(self, test_case: TestCase, tech_context: TechnicalContext) -> str:
        """Get fixture parameters for test method"""
        
        params = []
        
        if test_case.mock_requirements:
            if 'database' in ' '.join(test_case.mock_requirements):
                params.append('mock_db')
            if 'api' in ' '.join(test_case.mock_requirements):
                params.append('mock_api_client')
        
        return ', '.join(params)
    
    def _generate_python_test_imports(
        self, 
        test_suite: TestSuite, 
        tech_context: TechnicalContext
    ) -> str:
        """Generate Python test imports"""
        
        imports = ['import pytest', 'from unittest.mock import Mock, patch, AsyncMock']
        
        # Add framework-specific imports
        if tech_context.framework == 'fastapi':
            imports.extend([
                'from fastapi.testclient import TestClient',
                'from sqlalchemy.orm import Session'
            ])
        
        # Add test-specific imports based on test types
        test_types = set(tc.test_type for tc in test_suite.test_cases)
        
        if 'performance' in test_types:
            imports.extend([
                'import time',
                'import psutil',
                'import tracemalloc'
            ])
        
        if 'e2e' in test_types:
            imports.extend([
                'from selenium import webdriver',
                'from selenium.webdriver.common.by import By'
            ])
        
        return '\n'.join(imports)
    
    def _generate_parametrized_tests(self, test_suite: TestSuite) -> str:
        """Generate parametrized tests"""
        
        # Simple parametrized test example
        return '''@pytest.mark.parametrize("input_data,expected", [
        ({"test": "data1"}, {"result": "expected1"}),
        ({"test": "data2"}, {"result": "expected2"}),
    ])
    def test_parametrized_example(self, input_data, expected):
        """Parametrized test example"""
        # Test implementation here
        pass'''
    
    def _generate_async_tests(self, test_suite: TestSuite) -> str:
        """Generate async tests"""
        
        return '''@pytest.mark.asyncio
    async def test_async_operation(self):
        """Test async operation"""
        # Async test implementation here
        result = await some_async_function()
        assert result is not None'''
    
    def _generate_mock_tests(self, test_suite: TestSuite) -> str:
        """Generate mock-based tests"""
        
        return '''@patch('module.external_dependency')
    def test_with_mock(self, mock_dependency):
        """Test with mocked dependency"""
        mock_dependency.return_value = "mocked_result"
        # Test implementation here
        pass'''
    
    def _generate_edge_case_tests(self, test_suite: TestSuite) -> str:
        """Generate edge case tests"""
        
        return '''def test_edge_case_empty_input(self):
        """Test edge case with empty input"""
        # Edge case test implementation here
        pass
    
    def test_edge_case_boundary_values(self):
        """Test edge case with boundary values"""
        # Boundary test implementation here
        pass'''
    
    # Additional helper methods would continue here...
    # These would include methods for generating specific test patterns,
    # mock data, setup/teardown code, etc.
    
    def _generate_conftest_imports(self, tech_context: TechnicalContext) -> str:
        """Generate conftest.py imports"""
        
        imports = []
        
        if tech_context.framework == 'fastapi':
            imports.extend([
                'from fastapi.testclient import TestClient',
                'from sqlalchemy import create_engine',
                'from sqlalchemy.orm import sessionmaker'
            ])
        
        return '\n'.join(imports)
    
    def _generate_sample_data_content(self) -> str:
        """Generate sample data for tests"""
        
        return '''"users": [
            {"id": 1, "name": "Test User", "email": "test@example.com"},
            {"id": 2, "name": "Another User", "email": "user@example.com"}
        ],
        "products": [
            {"id": 1, "name": "Test Product", "price": 999},
            {"id": 2, "name": "Another Product", "price": 1999}
        ]'''
    
    def _generate_custom_fixtures(
        self, 
        test_suites: List[TestSuite], 
        tech_context: TechnicalContext
    ) -> str:
        """Generate custom fixtures"""
        
        fixtures = []
        
        # Generate fixtures based on test requirements
        all_mocks = set()
        for suite in test_suites:
            for test_case in suite.test_cases:
                all_mocks.update(test_case.mock_requirements)
        
        for mock in all_mocks:
            if mock == 'database_session':
                fixtures.append('''@pytest.fixture
def mock_database_session():
    """Mock database session"""
    session = Mock()
    return session''')
            
            elif mock == 'external_api_client':
                fixtures.append('''@pytest.fixture
def mock_api_client():
    """Mock API client"""
    client = Mock()
    client.get.return_value.status_code = 200
    client.get.return_value.json.return_value = {"data": "test"}
    return client''')
        
        return '\n\n'.join(fixtures) if fixtures else '# No custom fixtures needed'
    
    # Placeholder methods for completeness
    def _generate_test_setup_code(self, unit: Dict[str, Any], tech_context: TechnicalContext) -> str:
        return "# Test setup code here"
    
    def _generate_unit_test_setup(self, module_name: str, tech_context: TechnicalContext) -> str:
        return "# Unit test setup code here"
    
    def _generate_unit_test_teardown(self, tech_context: TechnicalContext) -> str:
        return "# Unit test teardown code here"
    
    def _generate_unit_test_fixtures(self, testable_units: List[Dict], tech_context: TechnicalContext) -> List[str]:
        return ["mock_fixture", "test_data_fixture"]
    
    def _generate_mock_data(self, testable_units: List[Dict]) -> Dict[str, Any]:
        return {"test_data": {"id": 1, "name": "test"}}
    
    def _generate_integration_test_setup(self, tech_context: TechnicalContext) -> str:
        return "# Integration test setup code here"
    
    def _generate_integration_test_teardown(self, tech_context: TechnicalContext) -> str:
        return "# Integration test teardown code here"
    
    def _generate_integration_fixtures(self, tech_context: TechnicalContext) -> List[str]:
        return ["test_client", "test_db"]
    
    def _generate_api_mock_data(self, endpoints: List[Dict]) -> Dict[str, Any]:
        return {"api_responses": {"test": "data"}}
    
    async def _generate_database_test_cases(self, features: List[FeatureRequirement], tech_context: TechnicalContext) -> List[TestCase]:
        return []  # Placeholder
    
    def _generate_db_test_setup(self, tech_context: TechnicalContext) -> str:
        return "# Database test setup code here"
    
    def _generate_db_test_teardown(self, tech_context: TechnicalContext) -> str:
        return "# Database test teardown code here"
    
    def _generate_db_fixtures(self, features: List[FeatureRequirement], tech_context: TechnicalContext) -> List[str]:
        return ["test_database", "test_session"]
    
    def _identify_user_workflows(self, ui_features: List[FeatureRequirement], all_features: List[FeatureRequirement]) -> List[Dict]:
        return []  # Placeholder
    
    async def _generate_workflow_test_cases(self, workflow: Dict, tech_context: TechnicalContext) -> List[TestCase]:
        return []  # Placeholder
    
    def _generate_e2e_setup(self, tech_context: TechnicalContext) -> str:
        return "# E2E test setup code here"
    
    def _generate_e2e_teardown(self, tech_context: TechnicalContext) -> str:
        return "# E2E test teardown code here"
    
    def _generate_e2e_fixtures(self, tech_context: TechnicalContext) -> List[str]:
        return ["browser", "wait"]
    
    def _generate_e2e_mock_data(self, workflows: List[Dict]) -> Dict[str, Any]:
        return {"user_data": {"username": "test", "password": "test123"}}
    
    def _identify_performance_critical_components(self, code_files: List[CodeFile], features: List[FeatureRequirement]) -> List[Dict]:
        return []  # Placeholder
    
    async def _generate_performance_test_cases(self, component: Dict, tech_context: TechnicalContext) -> List[TestCase]:
        return []  # Placeholder
    
    def _generate_performance_test_setup(self, tech_context: TechnicalContext) -> str:
        return "# Performance test setup code here"
    
    def _generate_performance_test_teardown(self, tech_context: TechnicalContext) -> str:
        return "# Performance test teardown code here"
    
    def _generate_performance_fixtures(self, tech_context: TechnicalContext) -> List[str]:
        return ["performance_monitor"]
    
    async def _generate_auth_security_tests(self, features: List[FeatureRequirement], tech_context: TechnicalContext) -> List[TestCase]:
        return []  # Placeholder
    
    async def _generate_input_validation_tests(self, features: List[FeatureRequirement], tech_context: TechnicalContext) -> List[TestCase]:
        return []  # Placeholder
    
    async def _generate_general_security_tests(self, tech_context: TechnicalContext) -> List[TestCase]:
        return []  # Placeholder
    
    def _generate_security_test_setup(self, tech_context: TechnicalContext) -> str:
        return "# Security test setup code here"
    
    def _generate_security_test_teardown(self, tech_context: TechnicalContext) -> str:
        return "# Security test teardown code here"
    
    def _generate_security_fixtures(self, tech_context: TechnicalContext) -> List[str]:
        return ["security_scanner"]
    
    def _generate_api_test_setup(self, endpoint: Dict, tech_context: TechnicalContext) -> str:
        return "# API test setup code here"
    
    async def _generate_javascript_test_files(self, test_suite: TestSuite, tech_context: TechnicalContext) -> List[CodeFile]:
        return []  # Placeholder - would implement JavaScript test file generation
    
    def _generate_api_endpoint_tests(self, test_suite: TestSuite) -> str:
        return "# API endpoint tests here"
    
    def _generate_db_integration_tests(self, test_suite: TestSuite) -> str:
        return "# Database integration tests here"
    
    def _generate_external_service_tests(self, test_suite: TestSuite) -> str:
        return "# External service tests here"
    
    def _generate_user_flow_tests(self, test_suite: TestSuite) -> str:
        return "# User flow tests here"
    
    def _generate_cross_browser_tests(self, test_suite: TestSuite) -> str:
        return "# Cross-browser tests here"
    
    async def _enhance_tests_with_ai(
        self,
        test_suites: List[TestSuite],
        features: List[FeatureRequirement],
        tech_context: TechnicalContext
    ) -> Optional[List[CodeFile]]:
        """Enhance tests with AI suggestions"""
        
        # Placeholder for AI enhancement
        return None