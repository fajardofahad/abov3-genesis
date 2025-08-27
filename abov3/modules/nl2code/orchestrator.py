"""
ABOV3 Genesis - NL2Code Orchestrator
Main orchestrator that coordinates all NL2Code module components for seamless natural language to code conversion
"""

import asyncio
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import logging
import json

# NL2Code module imports
from .core.processor import NaturalLanguageProcessor, FeatureRequirement, TechnicalContext
from .planning.engine import PlanningEngine, ImplementationPlan
from .generation.engine import CodeGenerationEngine, GenerationResult
from .testing.generator import TestGenerator, TestGenerationResult
from .utils.ollama_integration import NL2CodeOllamaIntegration
from .utils.error_handler import (
    ErrorHandler, handle_errors, log_performance, 
    ValidationError, CodeGenerationError, IntegrationError,
    validate_input, require_configuration
)

# Core ABOV3 imports
from ...core.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class NL2CodeOrchestrator:
    """
    Main orchestrator for the Natural Language to Code conversion system.
    Coordinates all components to provide seamless end-to-end functionality.
    """
    
    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        project_path: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the NL2Code orchestrator
        
        Args:
            ollama_client: Optional OllamaClient instance
            project_path: Path for generated project files
            config: Configuration dictionary
        """
        self.config = config or {}
        self.project_path = Path(project_path) if project_path else Path.cwd() / "generated_project"
        self.error_handler = ErrorHandler()
        
        # Initialize core components
        self.ollama_client = ollama_client or OllamaClient()
        self.ollama_integration = NL2CodeOllamaIntegration(self.ollama_client)
        
        # Initialize processors
        self.nl_processor = NaturalLanguageProcessor(self.ollama_client)
        self.planning_engine = PlanningEngine(self.ollama_client)
        self.code_generator = CodeGenerationEngine(self.ollama_client, self.project_path)
        self.test_generator = TestGenerator(self.ollama_client)
        
        # Workflow state
        self.current_session = None
        self.generation_history = []
        
        logger.info("NL2Code Orchestrator initialized successfully")
    
    @handle_errors({"module": "orchestrator"}, attempt_recovery=True)
    @log_performance
    async def generate_application_from_description(
        self,
        description: str,
        project_name: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None,
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Complete end-to-end application generation from natural language description
        
        Args:
            description: Natural language description of the desired application
            project_name: Optional project name
            preferences: Optional preferences (tech stack, architecture, etc.)
            output_path: Optional output path for generated files
            
        Returns:
            Dictionary containing generation results and metadata
        """
        logger.info(f"Starting complete application generation: {description[:100]}...")
        
        # Validate inputs
        validate_input(description, str, "description", required=True, min_length=10)
        
        project_name = project_name or f"generated_app_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_path = output_path or self.project_path / project_name
        
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_session = {
            "session_id": session_id,
            "started_at": datetime.now(),
            "description": description,
            "project_name": project_name,
            "status": "started"
        }
        
        try:
            # Phase 1: Requirements Analysis
            logger.info("Phase 1: Analyzing natural language requirements...")
            self.current_session["status"] = "analyzing_requirements"
            
            requirements_result = await self._analyze_requirements(description, preferences)
            features = requirements_result["features"]
            tech_context = requirements_result["tech_context"]
            
            # Phase 2: Implementation Planning
            logger.info("Phase 2: Creating implementation plan...")
            self.current_session["status"] = "planning"
            
            implementation_plan = await self._create_implementation_plan(
                features, tech_context, project_name
            )
            
            # Phase 3: Code Generation
            logger.info("Phase 3: Generating application code...")
            self.current_session["status"] = "generating_code"
            
            generation_result = await self._generate_implementation_code(
                implementation_plan, features, output_path
            )
            
            # Phase 4: Test Generation
            logger.info("Phase 4: Generating comprehensive tests...")
            self.current_session["status"] = "generating_tests"
            
            test_result = await self._generate_comprehensive_tests(
                implementation_plan, features, generation_result.files_created, output_path
            )
            
            # Phase 5: Final Assembly and Validation
            logger.info("Phase 5: Final assembly and validation...")
            self.current_session["status"] = "finalizing"
            
            final_result = await self._finalize_generation(
                implementation_plan, generation_result, test_result, output_path
            )
            
            # Update session status
            self.current_session["status"] = "completed"
            self.current_session["completed_at"] = datetime.now()
            
            # Add to history
            self.generation_history.append(self.current_session.copy())
            
            logger.info(f"Application generation completed successfully: {project_name}")
            
            return {
                "success": True,
                "session_id": session_id,
                "project_name": project_name,
                "output_path": str(output_path),
                "features_count": len(features),
                "files_generated": len(generation_result.files_created),
                "test_cases_generated": test_result.total_test_cases,
                "implementation_plan": implementation_plan,
                "generation_summary": final_result,
                "tech_context": tech_context,
                "execution_time": (datetime.now() - self.current_session["started_at"]).total_seconds()
            }
            
        except Exception as e:
            self.current_session["status"] = "failed"
            self.current_session["error"] = str(e)
            logger.error(f"Application generation failed: {e}")
            raise
    
    @handle_errors({"module": "orchestrator"})
    async def _analyze_requirements(
        self, 
        description: str, 
        preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze natural language requirements"""
        
        # Enhanced analysis with AI if available
        if self.ollama_integration:
            ai_analysis = await self.ollama_integration.analyze_requirements_with_ai(
                description, preferences
            )
            
            # Convert AI analysis to our format if needed
            if ai_analysis and "features" in ai_analysis:
                return self._convert_ai_analysis_format(ai_analysis)
        
        # Fallback to standard analysis
        features, tech_context = await self.nl_processor.analyze_requirements(
            description, preferences
        )
        
        return {
            "features": features,
            "tech_context": tech_context,
            "analysis_method": "standard"
        }
    
    @handle_errors({"module": "orchestrator"})
    async def _create_implementation_plan(
        self,
        features: List[FeatureRequirement],
        tech_context: TechnicalContext,
        project_name: str
    ) -> ImplementationPlan:
        """Create detailed implementation plan"""
        
        # Enhanced planning with AI if available
        if self.ollama_integration:
            try:
                features_summary = ", ".join([f.name for f in features])
                ai_plan = await self.ollama_integration.create_implementation_plan_with_ai(
                    project_name,
                    features_summary,
                    tech_context.__dict__
                )
                
                if ai_plan:
                    # Merge AI insights with our plan
                    plan = await self.planning_engine.create_implementation_plan(
                        features, tech_context, project_name
                    )
                    
                    # Enhance plan with AI insights
                    plan = self._enhance_plan_with_ai_insights(plan, ai_plan)
                    return plan
                    
            except Exception as e:
                logger.warning(f"AI planning failed, falling back to standard planning: {e}")
        
        # Standard planning
        return await self.planning_engine.create_implementation_plan(
            features, tech_context, project_name
        )
    
    @handle_errors({"module": "orchestrator"})
    async def _generate_implementation_code(
        self,
        implementation_plan: ImplementationPlan,
        features: List[FeatureRequirement],
        output_path: Path
    ) -> GenerationResult:
        """Generate complete implementation code"""
        
        return await self.code_generator.generate_implementation(
            implementation_plan, features, output_path
        )
    
    @handle_errors({"module": "orchestrator"})
    async def _generate_comprehensive_tests(
        self,
        implementation_plan: ImplementationPlan,
        features: List[FeatureRequirement],
        generated_files: List,
        output_path: Path
    ) -> TestGenerationResult:
        """Generate comprehensive test suite"""
        
        return await self.test_generator.generate_comprehensive_tests(
            implementation_plan, features, generated_files, output_path
        )
    
    @handle_errors({"module": "orchestrator"})
    async def _finalize_generation(
        self,
        implementation_plan: ImplementationPlan,
        generation_result: GenerationResult,
        test_result: TestGenerationResult,
        output_path: Path
    ) -> Dict[str, Any]:
        """Finalize generation with documentation and final checks"""
        
        # Generate project documentation
        readme_content = await self._generate_project_readme(
            implementation_plan, generation_result, test_result
        )
        
        # Write README
        readme_path = output_path / "README.md"
        readme_path.parent.mkdir(parents=True, exist_ok=True)
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        # Generate project summary
        summary = {
            "project_name": implementation_plan.project_name,
            "description": implementation_plan.description,
            "files_generated": {
                "implementation_files": len(generation_result.files_created),
                "test_files": len(test_result.test_files),
                "config_files": len(test_result.configuration_files),
                "documentation_files": 1  # README
            },
            "estimated_development_time": implementation_plan.total_estimated_hours,
            "test_coverage_estimate": test_result.coverage_estimate,
            "tech_stack": implementation_plan.tech_context.tech_stack,
            "milestones": len(implementation_plan.milestones),
            "generation_metrics": generation_result.metrics,
            "success_indicators": {
                "code_generation_successful": generation_result.success,
                "tests_generated": test_result.success,
                "no_critical_errors": len(generation_result.errors) == 0
            }
        }
        
        # Save project metadata
        metadata_path = output_path / ".abov3_project.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                "generated_by": "ABOV3 Genesis NL2Code v1.0.0",
                "generated_at": datetime.now().isoformat(),
                "session_id": self.current_session["session_id"],
                "implementation_plan": self._serialize_implementation_plan(implementation_plan),
                "generation_summary": summary
            }, f, indent=2, ensure_ascii=False)
        
        return summary
    
    @handle_errors({"module": "orchestrator"})
    async def generate_feature_from_description(
        self,
        description: str,
        existing_project_path: Path,
        integration_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a single feature and integrate it into an existing project
        
        Args:
            description: Natural language description of the feature
            existing_project_path: Path to existing project
            integration_context: Context about the existing project
            
        Returns:
            Dictionary containing generation results
        """
        logger.info(f"Generating feature: {description[:100]}...")
        
        validate_input(description, str, "description", required=True, min_length=5)
        
        if not existing_project_path.exists():
            raise ValidationError(
                f"Project path does not exist: {existing_project_path}",
                field="existing_project_path"
            )
        
        try:
            # Analyze the feature request
            features, tech_context = await self.nl_processor.analyze_requirements(
                description, integration_context
            )
            
            if not features:
                raise CodeGenerationError("No features could be extracted from the description")
            
            feature = features[0]  # Focus on the primary feature
            
            # Generate code for the feature
            task_description = f"Implement {feature.name}: {feature.description}"
            
            if self.ollama_integration:
                code_result = await self.ollama_integration.generate_code_with_ai(
                    task_description,
                    tech_context.__dict__,
                    [{"name": feature.name, "description": feature.description}]
                )
                
                if code_result.get("success"):
                    # Write generated files
                    for code_block in code_result["code_blocks"]:
                        file_path = existing_project_path / code_block["file_path"]
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(code_block["code"])
                    
                    return {
                        "success": True,
                        "feature": feature.name,
                        "files_generated": [cb["file_path"] for cb in code_result["code_blocks"]],
                        "integration_suggestions": self._generate_integration_suggestions(feature, integration_context)
                    }
            
            # Fallback to standard generation
            raise CodeGenerationError("Feature generation failed")
            
        except Exception as e:
            logger.error(f"Feature generation failed: {e}")
            raise
    
    @handle_errors({"module": "orchestrator"})
    async def enhance_existing_code(
        self,
        code_path: Path,
        enhancement_description: str,
        enhancement_type: str = "optimize"
    ) -> Dict[str, Any]:
        """
        Enhance existing code based on natural language description
        
        Args:
            code_path: Path to the code file to enhance
            enhancement_description: Description of desired enhancements
            enhancement_type: Type of enhancement (optimize, refactor, add_features, fix_bugs)
            
        Returns:
            Dictionary containing enhancement results
        """
        logger.info(f"Enhancing code: {code_path} - {enhancement_description[:100]}...")
        
        if not code_path.exists():
            raise ValidationError(f"Code file does not exist: {code_path}")
        
        # Read existing code
        with open(code_path, 'r', encoding='utf-8') as f:
            original_code = f.read()
        
        # Determine language
        language = code_path.suffix[1:] if code_path.suffix else 'text'
        
        # Define optimization goals based on enhancement type
        optimization_goals = {
            "optimize": ["performance", "memory_usage", "readability"],
            "refactor": ["code_quality", "maintainability", "design_patterns"],
            "add_features": ["functionality", "extensibility"],
            "fix_bugs": ["error_handling", "edge_cases", "validation"]
        }.get(enhancement_type, ["code_quality"])
        
        if self.ollama_integration:
            # Use AI to optimize the code
            optimization_result = await self.ollama_integration.optimize_code_quality(
                original_code, language, optimization_goals
            )
            
            if optimization_result.get("success"):
                # Backup original code
                backup_path = code_path.with_suffix(code_path.suffix + '.backup')
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_code)
                
                # Write optimized code
                with open(code_path, 'w', encoding='utf-8') as f:
                    f.write(optimization_result["optimized_code"])
                
                return {
                    "success": True,
                    "enhancement_type": enhancement_type,
                    "original_file": str(code_path),
                    "backup_file": str(backup_path),
                    "improvements": optimization_result["improvements"],
                    "size_change": optimization_result["optimized_size"] - optimization_result["original_size"]
                }
        
        raise CodeGenerationError(f"Code enhancement failed for {code_path}")
    
    @handle_errors({"module": "orchestrator"})
    def get_generation_status(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get current generation status"""
        
        if session_id:
            # Find specific session
            for session in self.generation_history:
                if session["session_id"] == session_id:
                    return session
            return {"error": "Session not found"}
        
        # Return current session or last session
        if self.current_session:
            return self.current_session
        elif self.generation_history:
            return self.generation_history[-1]
        else:
            return {"status": "no_sessions"}
    
    @handle_errors({"module": "orchestrator"})
    def get_generation_history(self) -> List[Dict[str, Any]]:
        """Get history of all generation sessions"""
        
        return [
            {
                "session_id": session["session_id"],
                "started_at": session["started_at"].isoformat(),
                "project_name": session.get("project_name"),
                "status": session["status"],
                "description": session["description"][:100] + "..." if len(session["description"]) > 100 else session["description"]
            }
            for session in self.generation_history
        ]
    
    @handle_errors({"module": "orchestrator"})
    async def validate_generated_project(self, project_path: Path) -> Dict[str, Any]:
        """Validate a generated project for completeness and correctness"""
        
        validation_results = {
            "project_path": str(project_path),
            "is_valid": True,
            "issues": [],
            "suggestions": [],
            "files_checked": 0,
            "metadata": {}
        }
        
        try:
            # Check if project exists
            if not project_path.exists():
                validation_results["is_valid"] = False
                validation_results["issues"].append("Project directory does not exist")
                return validation_results
            
            # Load project metadata if available
            metadata_path = project_path / ".abov3_project.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    validation_results["metadata"] = json.load(f)
            
            # Check for required files based on project type
            required_files = self._get_required_files_for_project(project_path)
            
            for file_path in required_files:
                full_path = project_path / file_path
                validation_results["files_checked"] += 1
                
                if not full_path.exists():
                    validation_results["is_valid"] = False
                    validation_results["issues"].append(f"Missing required file: {file_path}")
                else:
                    # Basic file validation
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if len(content.strip()) == 0:
                                validation_results["issues"].append(f"Empty file: {file_path}")
                    except Exception as e:
                        validation_results["issues"].append(f"Cannot read file {file_path}: {e}")
            
            # Generate suggestions based on issues
            if validation_results["issues"]:
                validation_results["suggestions"] = [
                    "Re-run generation to create missing files",
                    "Check file permissions and disk space",
                    "Verify project configuration"
                ]
            
            return validation_results
            
        except Exception as e:
            validation_results["is_valid"] = False
            validation_results["issues"].append(f"Validation error: {str(e)}")
            return validation_results
    
    # Helper methods
    def _convert_ai_analysis_format(self, ai_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Convert AI analysis format to our internal format"""
        
        # This would convert the AI analysis format to our FeatureRequirement and TechnicalContext format
        # For now, return as-is with some basic conversion
        
        features = []
        if "features" in ai_analysis:
            for feature_data in ai_analysis["features"]:
                if isinstance(feature_data, dict):
                    feature = FeatureRequirement(
                        name=feature_data.get("name", "Unknown Feature"),
                        description=feature_data.get("description", ""),
                        priority=feature_data.get("priority", "medium"),
                        category=feature_data.get("category", "business_logic"),
                        complexity=feature_data.get("complexity", "moderate"),
                        dependencies=feature_data.get("dependencies", []),
                        estimated_effort=feature_data.get("estimated_hours", 8),
                        technical_specs=feature_data.get("technical_specs", {}),
                        acceptance_criteria=feature_data.get("acceptance_criteria", [])
                    )
                    features.append(feature)
        
        tech_context_data = ai_analysis.get("technical_context", {})
        tech_context = TechnicalContext(
            project_type=tech_context_data.get("project_type", "web"),
            tech_stack=tech_context_data.get("recommended_tech_stack", ["python"]),
            architecture=tech_context_data.get("architecture_pattern", "monolith"),
            framework=None,
            database=None,
            deployment_target=None,
            performance_requirements={},
            security_requirements=[]
        )
        
        return {
            "features": features,
            "tech_context": tech_context,
            "analysis_method": "ai_enhanced"
        }
    
    def _enhance_plan_with_ai_insights(
        self, 
        plan: ImplementationPlan, 
        ai_insights: Dict[str, Any]
    ) -> ImplementationPlan:
        """Enhance implementation plan with AI insights"""
        
        # This would merge AI insights into the existing plan
        # For now, just return the original plan
        return plan
    
    async def _generate_project_readme(
        self,
        implementation_plan: ImplementationPlan,
        generation_result: GenerationResult,
        test_result: TestGenerationResult
    ) -> str:
        """Generate comprehensive README for the project"""
        
        if self.ollama_integration:
            try:
                doc_result = await self.ollama_integration.generate_documentation_with_ai(
                    f"Project: {implementation_plan.project_name}\n"
                    f"Description: {implementation_plan.description}\n"
                    f"Tech Stack: {', '.join(implementation_plan.tech_context.tech_stack)}\n"
                    f"Files Generated: {len(generation_result.files_created)}\n"
                    f"Test Cases: {test_result.total_test_cases}",
                    "project_readme",
                    "developers"
                )
                
                if doc_result.get("success"):
                    return doc_result["documentation"]
            except Exception as e:
                logger.warning(f"AI documentation generation failed: {e}")
        
        # Fallback to basic README
        return f"""# {implementation_plan.project_name}

{implementation_plan.description}

## Generated by ABOV3 Genesis

This project was automatically generated using ABOV3 Genesis NL2Code module.

### Project Statistics

- **Files Generated**: {len(generation_result.files_created)}
- **Test Cases**: {test_result.total_test_cases}
- **Estimated Coverage**: {test_result.coverage_estimate:.1%}
- **Technology Stack**: {', '.join(implementation_plan.tech_context.tech_stack)}
- **Estimated Development Time**: {implementation_plan.total_estimated_hours} hours

### Getting Started

1. Install dependencies
2. Configure environment variables
3. Run the application
4. Run tests to verify functionality

### Project Structure

Generated files are organized according to best practices for {implementation_plan.tech_context.project_type} applications.

---

*Generated by ABOV3 Genesis NL2Code Module*
"""
    
    def _generate_integration_suggestions(
        self, 
        feature: FeatureRequirement, 
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate suggestions for integrating the feature"""
        
        suggestions = [
            f"Add imports for {feature.name} components",
            "Update routing configuration if needed",
            "Add database migrations if applicable",
            "Update tests to include new feature",
            "Consider API documentation updates"
        ]
        
        return suggestions
    
    def _get_required_files_for_project(self, project_path: Path) -> List[str]:
        """Get list of required files based on project type"""
        
        # Basic required files
        required_files = ["README.md"]
        
        # Check for Python project
        if (project_path / "requirements.txt").exists() or any(project_path.glob("*.py")):
            required_files.extend(["requirements.txt"])
        
        # Check for Node.js project
        if (project_path / "package.json").exists():
            required_files.extend(["package.json"])
        
        # Check for web project
        if any(project_path.glob("*.html")):
            required_files.extend(["index.html"])
        
        return required_files
    
    def _serialize_implementation_plan(self, plan: ImplementationPlan) -> Dict[str, Any]:
        """Serialize implementation plan to JSON-compatible format"""
        
        # Convert dataclasses and other objects to dictionaries
        # This is a simplified version - full implementation would handle all nested objects
        return {
            "project_name": plan.project_name,
            "description": plan.description,
            "total_estimated_hours": plan.total_estimated_hours,
            "milestones_count": len(plan.milestones),
            "tech_context": {
                "project_type": plan.tech_context.project_type,
                "tech_stack": plan.tech_context.tech_stack,
                "architecture": plan.tech_context.architecture
            }
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        
        if self.ollama_integration:
            await self.ollama_integration.cleanup()
        
        if self.ollama_client:
            await self.ollama_client.close()
        
        logger.info("NL2Code Orchestrator cleaned up successfully")


# Convenience functions for external use
async def generate_app_from_description(
    description: str,
    project_name: Optional[str] = None,
    output_path: Optional[Path] = None,
    preferences: Optional[Dict[str, Any]] = None,
    ollama_client: Optional[OllamaClient] = None
) -> Dict[str, Any]:
    """
    Convenience function to generate a complete application from natural language description
    """
    orchestrator = NL2CodeOrchestrator(
        ollama_client=ollama_client,
        project_path=output_path
    )
    
    try:
        return await orchestrator.generate_application_from_description(
            description, project_name, preferences, output_path
        )
    finally:
        await orchestrator.cleanup()


async def add_feature_to_project(
    description: str,
    project_path: Path,
    context: Optional[Dict[str, Any]] = None,
    ollama_client: Optional[OllamaClient] = None
) -> Dict[str, Any]:
    """
    Convenience function to add a feature to an existing project
    """
    orchestrator = NL2CodeOrchestrator(ollama_client=ollama_client)
    
    try:
        return await orchestrator.generate_feature_from_description(
            description, project_path, context
        )
    finally:
        await orchestrator.cleanup()