"""
ABOV3 Genesis - NL2Code Module Demo
Demonstration of the Natural Language to Code generation capabilities
"""

import asyncio
import logging
from pathlib import Path
import json
from typing import Dict, Any

from .orchestrator import NL2CodeOrchestrator, generate_app_from_description, add_feature_to_project
from ...core.ollama_client import OllamaClient

# Configure logging for demo
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NL2CodeDemo:
    """
    Interactive demonstration of NL2Code capabilities
    """
    
    def __init__(self, output_base_path: Path = None):
        self.output_base_path = output_base_path or Path("./demo_outputs")
        self.output_base_path.mkdir(parents=True, exist_ok=True)
        
    async def run_complete_demo(self):
        """Run complete demonstration of all NL2Code features"""
        
        print("🚀 ABOV3 Genesis NL2Code Module - Complete Demo")
        print("=" * 60)
        
        # Demo scenarios
        scenarios = [
            {
                "name": "Simple Blog Application",
                "description": "Create a personal blog website with user registration, post creation, comments, and an admin dashboard. Users should be able to write posts in markdown, and visitors can leave comments. Include user authentication and a responsive design.",
                "preferences": {
                    "tech_stack": ["python", "fastapi", "react"],
                    "database": "postgresql",
                    "architecture": "monolith"
                }
            },
            {
                "name": "E-commerce Store",
                "description": "Build an online store for selling products. Include product catalog, shopping cart, user accounts, payment processing, order management, and inventory tracking. Make it mobile-friendly with search functionality.",
                "preferences": {
                    "tech_stack": ["javascript", "node.js", "react"],
                    "database": "mongodb",
                    "deployment": "docker"
                }
            },
            {
                "name": "Task Management API",
                "description": "Create a RESTful API for task management. Users should be able to create, update, delete, and list tasks. Include user authentication, task categories, due dates, priorities, and assignment to team members.",
                "preferences": {
                    "tech_stack": ["python", "fastapi"],
                    "database": "postgresql",
                    "architecture": "api"
                }
            }
        ]
        
        results = []
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n📝 Demo Scenario {i}: {scenario['name']}")
            print("-" * 40)
            
            try:
                result = await self.demo_full_application_generation(
                    scenario["description"],
                    scenario["name"].lower().replace(" ", "_"),
                    scenario["preferences"]
                )
                results.append(result)
                print("✅ Scenario completed successfully!")
                
                # Demo adding a feature to the generated project
                if result["success"]:
                    await self.demo_feature_addition(
                        result["output_path"],
                        "Add email notifications when new comments are posted"
                    )
                
            except Exception as e:
                print(f"❌ Scenario failed: {e}")
                results.append({"success": False, "error": str(e)})
        
        # Generate demo report
        await self.generate_demo_report(results)
        print(f"\n📊 Demo completed! Check {self.output_base_path} for generated projects and reports.")
    
    async def demo_full_application_generation(
        self, 
        description: str, 
        project_name: str,
        preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Demonstrate full application generation"""
        
        print(f"🔧 Generating: {description}")
        
        output_path = self.output_base_path / project_name
        
        try:
            # Initialize orchestrator
            orchestrator = NL2CodeOrchestrator(
                project_path=output_path
            )
            
            # Generate application
            result = await orchestrator.generate_application_from_description(
                description=description,
                project_name=project_name,
                preferences=preferences,
                output_path=output_path
            )
            
            if result["success"]:
                print(f"  ✅ Generated {result['files_generated']} files")
                print(f"  📁 Output: {result['output_path']}")
                print(f"  🧪 Tests: {result['test_cases_generated']} test cases")
                print(f"  ⏱️  Time: {result['execution_time']:.2f} seconds")
            
            await orchestrator.cleanup()
            return result
            
        except Exception as e:
            logger.error(f"Full generation demo failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def demo_feature_addition(
        self, 
        project_path: str, 
        feature_description: str
    ) -> Dict[str, Any]:
        """Demonstrate adding a feature to existing project"""
        
        print(f"🔧 Adding feature: {feature_description}")
        
        try:
            result = await add_feature_to_project(
                description=feature_description,
                project_path=Path(project_path)
            )
            
            if result["success"]:
                print(f"  ✅ Feature added: {result['feature']}")
                print(f"  📁 Files: {', '.join(result['files_generated'])}")
            
            return result
            
        except Exception as e:
            logger.error(f"Feature addition demo failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def demo_individual_components(self):
        """Demonstrate individual NL2Code components"""
        
        print("\n🧩 Individual Component Demos")
        print("=" * 40)
        
        # Initialize components
        orchestrator = NL2CodeOrchestrator()
        
        try:
            # Demo 1: Requirements Analysis
            print("\n1. Natural Language Requirements Analysis")
            print("-" * 30)
            
            test_description = "Create a simple calculator web app that can perform basic arithmetic operations"
            features, tech_context = await orchestrator.nl_processor.analyze_requirements(
                test_description
            )
            
            print(f"  📝 Analyzed: {test_description}")
            print(f"  🎯 Features found: {len(features)}")
            for feature in features[:3]:  # Show first 3
                print(f"    - {feature.name}: {feature.complexity} ({feature.estimated_effort}h)")
            print(f"  🛠️  Tech Stack: {', '.join(tech_context.tech_stack)}")
            
            # Demo 2: Implementation Planning
            print("\n2. Implementation Planning")
            print("-" * 25)
            
            plan = await orchestrator.planning_engine.create_implementation_plan(
                features, tech_context, "demo_calculator"
            )
            
            print(f"  📋 Plan created: {len(plan.milestones)} milestones")
            print(f"  ⏱️  Estimated time: {plan.total_estimated_hours} hours")
            for milestone in plan.milestones[:2]:  # Show first 2
                print(f"    - {milestone.name}: {len(milestone.tasks)} tasks")
            
            # Demo 3: Code Generation (small example)
            print("\n3. Code Generation Sample")
            print("-" * 25)
            
            if orchestrator.ollama_integration:
                simple_task = {
                    "name": "Calculator Functions",
                    "description": "Basic arithmetic operations",
                    "category": "business_logic"
                }
                
                code_result = await orchestrator.ollama_integration.generate_code_with_ai(
                    "Create Python functions for basic calculator operations (add, subtract, multiply, divide)",
                    {"primary_language": "python", "framework": "none"},
                    [simple_task]
                )
                
                if code_result.get("success"):
                    print(f"  💻 Generated {len(code_result['code_blocks'])} code files")
                    if code_result['code_blocks']:
                        sample_code = code_result['code_blocks'][0]['code'][:200] + "..."
                        print(f"  📄 Sample: {sample_code}")
            
            # Demo 4: Test Generation
            print("\n4. Test Generation Sample")
            print("-" * 25)
            
            test_result = await orchestrator.test_generator.generate_comprehensive_tests(
                plan, features, [], self.output_base_path / "test_demo"
            )
            
            print(f"  🧪 Generated {test_result.total_test_cases} test cases")
            print(f"  📊 Coverage estimate: {test_result.coverage_estimate:.1%}")
            print(f"  📁 Test suites: {len(test_result.test_suites)}")
            
        except Exception as e:
            logger.error(f"Component demo failed: {e}")
        
        finally:
            await orchestrator.cleanup()
    
    async def demo_error_handling(self):
        """Demonstrate error handling capabilities"""
        
        print("\n🚨 Error Handling Demo")
        print("=" * 25)
        
        orchestrator = NL2CodeOrchestrator()
        
        # Demo various error scenarios
        error_scenarios = [
            {
                "name": "Invalid Input",
                "description": "",  # Empty description should cause validation error
                "expected_error": "ValidationError"
            },
            {
                "name": "Complex Request",
                "description": "Create a quantum computing blockchain AI system with neural networks and advanced machine learning",
                "expected_error": "May fail due to complexity"
            },
            {
                "name": "Invalid Path",
                "description": "Simple web app",
                "output_path": "/invalid/path/that/cannot/be/created",
                "expected_error": "Path error"
            }
        ]
        
        for scenario in error_scenarios:
            print(f"\n  Testing: {scenario['name']}")
            try:
                if "output_path" in scenario:
                    result = await orchestrator.generate_application_from_description(
                        description=scenario["description"],
                        output_path=Path(scenario["output_path"])
                    )
                else:
                    result = await orchestrator.generate_application_from_description(
                        description=scenario["description"]
                    )
                
                if "error" in result:
                    print(f"    ✅ Error handled gracefully: {result['error_details']['user_message']}")
                else:
                    print(f"    ⚠️  Unexpectedly succeeded")
                    
            except Exception as e:
                print(f"    ✅ Exception caught and handled: {type(e).__name__}")
        
        # Show error statistics
        error_stats = orchestrator.error_handler.get_error_statistics()
        print(f"\n  📊 Error Statistics:")
        print(f"    Total errors: {error_stats['total_errors']}")
        if error_stats['categories']:
            print(f"    By category: {error_stats['categories']}")
        
        await orchestrator.cleanup()
    
    async def generate_demo_report(self, results: list):
        """Generate comprehensive demo report"""
        
        report_data = {
            "demo_metadata": {
                "generated_at": str(asyncio.get_event_loop().time()),
                "total_scenarios": len(results),
                "successful_scenarios": len([r for r in results if r.get("success", False)])
            },
            "scenario_results": results,
            "summary": {
                "success_rate": len([r for r in results if r.get("success", False)]) / len(results) if results else 0,
                "total_files_generated": sum(r.get("files_generated", 0) for r in results if r.get("success")),
                "total_test_cases": sum(r.get("test_cases_generated", 0) for r in results if r.get("success")),
                "average_execution_time": sum(r.get("execution_time", 0) for r in results if r.get("success")) / max(len([r for r in results if r.get("success")]), 1)
            }
        }
        
        # Save report
        report_path = self.output_base_path / "demo_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        # Generate markdown report
        markdown_report = self._generate_markdown_report(report_data)
        report_md_path = self.output_base_path / "demo_report.md"
        with open(report_md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_report)
        
        print(f"\n📋 Demo report saved to {report_path}")
    
    def _generate_markdown_report(self, report_data: Dict[str, Any]) -> str:
        """Generate markdown demo report"""
        
        report = f"""# ABOV3 Genesis NL2Code Module - Demo Report

## Summary

- **Total Scenarios**: {report_data['demo_metadata']['total_scenarios']}
- **Successful**: {report_data['demo_metadata']['successful_scenarios']}
- **Success Rate**: {report_data['summary']['success_rate']:.1%}
- **Total Files Generated**: {report_data['summary']['total_files_generated']}
- **Total Test Cases**: {report_data['summary']['total_test_cases']}
- **Average Execution Time**: {report_data['summary']['average_execution_time']:.2f} seconds

## Scenario Results

"""
        
        for i, result in enumerate(report_data['scenario_results'], 1):
            if result.get("success"):
                report += f"""### Scenario {i}: ✅ Success

- **Project**: {result.get('project_name', 'Unknown')}
- **Files Generated**: {result.get('files_generated', 0)}
- **Test Cases**: {result.get('test_cases_generated', 0)}
- **Execution Time**: {result.get('execution_time', 0):.2f}s
- **Output Path**: `{result.get('output_path', 'N/A')}`

"""
            else:
                report += f"""### Scenario {i}: ❌ Failed

- **Error**: {result.get('error', 'Unknown error')}

"""
        
        report += """## Capabilities Demonstrated

1. **Natural Language Analysis**: Converting plain English descriptions into structured requirements
2. **Implementation Planning**: Creating detailed development plans with milestones and tasks
3. **Multi-file Code Generation**: Generating complete applications with proper structure
4. **Comprehensive Testing**: Automatic generation of unit, integration, and E2E tests
5. **Error Handling**: Robust error handling with user-friendly messages and recovery suggestions
6. **AI Integration**: Enhanced capabilities through Ollama model integration

## Technology Support

The NL2Code module supports:

- **Languages**: Python, JavaScript, TypeScript, HTML, CSS
- **Frameworks**: FastAPI, Express.js, React, Vue.js
- **Databases**: PostgreSQL, MongoDB, SQLite
- **Testing**: Pytest, Jest, Selenium
- **Deployment**: Docker, Cloud platforms

---

*Generated by ABOV3 Genesis NL2Code Module*
"""
        
        return report


async def run_interactive_demo():
    """Run an interactive demo where users can input their own descriptions"""
    
    print("🎯 ABOV3 Genesis NL2Code - Interactive Demo")
    print("=" * 50)
    print("Enter natural language descriptions to generate applications!")
    print("Type 'quit' to exit, 'demo' for automated demo, or 'help' for examples.\n")
    
    demo = NL2CodeDemo()
    
    while True:
        try:
            user_input = input("💬 Describe your application: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            elif user_input.lower() == 'demo':
                await demo.run_complete_demo()
                continue
            
            elif user_input.lower() == 'help':
                print_help_examples()
                continue
            
            # Generate application from user description
            print(f"\n🚀 Generating application from: '{user_input}'")
            
            project_name = input("📝 Project name (or press Enter for auto): ").strip()
            if not project_name:
                project_name = f"user_app_{asyncio.get_event_loop().time():.0f}"
            
            result = await generate_app_from_description(
                description=user_input,
                project_name=project_name,
                output_path=demo.output_base_path / project_name
            )
            
            if result["success"]:
                print(f"\n✅ Success!")
                print(f"  📁 Project: {result['output_path']}")
                print(f"  📄 Files: {result['files_generated']}")
                print(f"  🧪 Tests: {result['test_cases_generated']}")
                print(f"  ⏱️  Time: {result['execution_time']:.2f}s")
                
                # Ask if user wants to add a feature
                feature_input = input("\n🔧 Add a feature? (or press Enter to skip): ").strip()
                if feature_input:
                    feature_result = await add_feature_to_project(
                        description=feature_input,
                        project_path=Path(result["output_path"])
                    )
                    
                    if feature_result["success"]:
                        print(f"  ✅ Feature added: {feature_result['feature']}")
                    else:
                        print(f"  ❌ Feature failed: {feature_result.get('error', 'Unknown error')}")
            else:
                print(f"\n❌ Generation failed: {result.get('error', 'Unknown error')}")
            
            print("\n" + "="*50)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def print_help_examples():
    """Print helpful examples for users"""
    
    examples = [
        "Create a personal blog with user accounts and comments",
        "Build a todo app with drag and drop functionality",
        "Make an e-commerce store with shopping cart and payments",
        "Design a weather app that shows forecasts and maps",
        "Create a chat application with real-time messaging",
        "Build a recipe sharing platform with search and ratings",
        "Make a task management system for teams",
        "Create a photo gallery with upload and sharing features"
    ]
    
    print("\n💡 Example descriptions:")
    for i, example in enumerate(examples, 1):
        print(f"  {i}. {example}")
    print()


async def main():
    """Main demo function"""
    
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "interactive":
            await run_interactive_demo()
        elif sys.argv[1] == "components":
            demo = NL2CodeDemo()
            await demo.demo_individual_components()
        elif sys.argv[1] == "errors":
            demo = NL2CodeDemo()
            await demo.demo_error_handling()
        else:
            demo = NL2CodeDemo()
            await demo.run_complete_demo()
    else:
        # Default to complete demo
        demo = NL2CodeDemo()
        await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())