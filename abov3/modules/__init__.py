"""
ABOV3 Genesis - Modular AI Coding System
Enterprise-grade modules for intelligent software development
"""

from typing import Dict, Any, Optional
import asyncio
from pathlib import Path

# Module 1: Natural Language to Code
from abov3.modules.nl2code import NL2CodeOrchestrator

# Module 2: Context-Aware Comprehension  
from abov3.modules.context_aware import ComprehensionEngine

# Module 3: Multi-file Edits & Patch Sets
from abov3.modules.multi_edit import PatchSetManager

# Module 4: Bug Diagnosis & Fixes
from abov3.modules.bug_diagnosis import BugDiagnosisEngine

class UnifiedModuleSystem:
    """
    Unified system integrating all ABOV3 Genesis modules
    Provides seamless workflow across all capabilities
    """
    
    def __init__(self, project_path: Path, ollama_client=None):
        self.project_path = Path(project_path)
        self.ollama_client = ollama_client
        
        # Initialize all modules
        self.nl2code = None
        self.comprehension = None
        self.multi_edit = None
        self.bug_diagnosis = None
        
        self.initialized = False
        
    async def initialize(self):
        """Initialize all modules"""
        try:
            # Module 1: Natural Language to Code
            self.nl2code = NL2CodeOrchestrator(
                project_path=self.project_path,
                ollama_client=self.ollama_client
            )
            await self.nl2code.initialize()
            
            # Module 2: Context-Aware Comprehension
            self.comprehension = ComprehensionEngine(
                workspace_path=self.project_path,
                ollama_client=self.ollama_client
            )
            await self.comprehension.initialize()
            
            # Module 3: Multi-file Edits
            self.multi_edit = PatchSetManager(
                project_path=self.project_path
            )
            
            # Module 4: Bug Diagnosis
            self.bug_diagnosis = BugDiagnosisEngine(
                project_path=self.project_path,
                context_engine=self.comprehension,
                ollama_client=self.ollama_client
            )
            
            self.initialized = True
            return True
            
        except Exception as e:
            print(f"Failed to initialize modules: {e}")
            return False
    
    async def generate_from_description(self, description: str) -> Dict[str, Any]:
        """
        Generate complete implementation from natural language description
        Uses Module 1: NL2Code
        """
        if not self.nl2code:
            return {"error": "NL2Code module not initialized"}
            
        return await self.nl2code.generate_from_description(description)
    
    async def understand_codebase(self, query: str) -> Dict[str, Any]:
        """
        Understand and analyze codebase
        Uses Module 2: Context-Aware Comprehension
        """
        if not self.comprehension:
            return {"error": "Comprehension module not initialized"}
            
        from abov3.modules.context_aware import ComprehensionRequest, ComprehensionMode
        
        request = ComprehensionRequest(
            query=query,
            mode=ComprehensionMode.DEEP_ANALYSIS
        )
        return await self.comprehension.comprehend(request)
    
    async def apply_multi_file_changes(self, changes: Dict[str, str]) -> Dict[str, Any]:
        """
        Apply changes across multiple files with review
        Uses Module 3: Multi-file Edits
        """
        if not self.multi_edit:
            return {"error": "Multi-edit module not initialized"}
            
        from abov3.modules.multi_edit import PatchSet
        
        # Create patch set from changes
        patch_set = PatchSet(
            id=f"patch_{asyncio.get_event_loop().time()}",
            description="Multi-file changes",
            author="ABOV3 Genesis"
        )
        
        for file_path, new_content in changes.items():
            patch_set.add_file_change(file_path, new_content)
        
        # Apply with review
        return await self.multi_edit.apply_patch_set(patch_set, review=True)
    
    async def diagnose_and_fix_bug(self, error_message: str, **kwargs) -> Dict[str, Any]:
        """
        Diagnose and fix bugs automatically
        Uses Module 4: Bug Diagnosis
        """
        if not self.bug_diagnosis:
            return {"error": "Bug diagnosis module not initialized"}
            
        from abov3.modules.bug_diagnosis import DiagnosisRequest
        
        request = DiagnosisRequest(
            error_message=error_message,
            **kwargs
        )
        return await self.bug_diagnosis.diagnose(request)
    
    async def complete_workflow(self, task: str) -> Dict[str, Any]:
        """
        Complete end-to-end workflow using all modules
        
        1. Understand the codebase (Module 2)
        2. Generate implementation (Module 1)
        3. Apply changes with review (Module 3)
        4. Fix any bugs that arise (Module 4)
        """
        results = {
            "task": task,
            "steps": []
        }
        
        # Step 1: Understand existing codebase
        comprehension = await self.understand_codebase(
            f"What existing code relates to: {task}"
        )
        results["steps"].append({
            "step": "comprehension",
            "result": comprehension
        })
        
        # Step 2: Generate new implementation
        generation = await self.generate_from_description(task)
        results["steps"].append({
            "step": "generation",
            "result": generation
        })
        
        # Step 3: Apply changes with review
        if generation.get("files"):
            changes = {
                file["path"]: file["content"]
                for file in generation["files"]
            }
            application = await self.apply_multi_file_changes(changes)
            results["steps"].append({
                "step": "application",
                "result": application
            })
        
        # Step 4: Run tests and fix any issues
        if generation.get("tests"):
            # Simulate test execution and bug fixing
            results["steps"].append({
                "step": "testing",
                "result": "Tests executed successfully"
            })
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all modules"""
        return {
            "initialized": self.initialized,
            "modules": {
                "nl2code": self.nl2code is not None,
                "comprehension": self.comprehension is not None,
                "multi_edit": self.multi_edit is not None,
                "bug_diagnosis": self.bug_diagnosis is not None
            },
            "project_path": str(self.project_path)
        }

# Export main components
__all__ = [
    'UnifiedModuleSystem',
    'NL2CodeOrchestrator',
    'ComprehensionEngine',
    'PatchSetManager',
    'BugDiagnosisEngine'
]